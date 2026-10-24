//! Holdings and cash replay for the historical valuation engine.
//!
//! A line-for-line port of the two Numba kernels that lived in
//! `portfolio_valuation_kernels.py`:
//!
//! * [`calculate_holdings`] — `_calculate_holdings_numba`: quantities, cost
//!   basis, currencies, cash and last trade prices as of one date.
//! * [`calculate_daily_holdings_chronological`] —
//!   `_calculate_daily_holdings_chronological_numba`: quantities, cash and last
//!   trade prices for every day of a range, in one pass.
//!
//! The two are deliberately *not* identical — the single-date version tracks
//! cost basis and leaves the fallback price alone on a ratio split, the
//! chronological one rescales it; short sales synthesize cash differently. Each
//! difference is preserved exactly, because the golden tests and every cached
//! history were produced by them. Comments explaining the business rules stay
//! with the Python callers; the ones here only mark where the two diverge.
//!
//! Arrays are flat and row-major: `[symbol][account]` is
//! `symbol * num_accounts + account`, `[day][symbol][account]` likewise.

use crate::util::{KernelError, check_len, py_min, wrap};

/// Transaction-type ids as the Python side assigned them. A type the ledger
/// never uses is `-1`.
#[derive(Debug, Clone, Copy)]
pub struct TypeIds {
    pub split: i64,
    pub stock_split: i64,
    pub buy: i64,
    pub deposit: i64,
    pub sell: i64,
    pub withdrawal: i64,
    pub short_sell: i64,
    pub buy_to_cover: i64,
    pub transfer: i64,
    pub spin_off: i64,
    pub fees: i64,
    pub dividend: i64,
    pub interest: i64,
    pub tax: i64,
}

/// One column per field, one row per transaction, sorted by date.
#[derive(Debug, Clone, Copy)]
pub struct Transactions<'a> {
    pub dates: &'a [i64],
    pub symbols: &'a [i64],
    pub accounts: &'a [i64],
    pub to_accounts: &'a [i64],
    pub types: &'a [i64],
    pub quantities: &'a [f64],
    pub prices: &'a [f64],
    pub totals: &'a [f64],
    pub commissions: &'a [f64],
    pub split_ratios: &'a [f64],
}

impl Transactions<'_> {
    fn len(&self) -> usize {
        self.dates.len()
    }

    fn check(&self) -> Result<(), KernelError> {
        let n = self.len();
        check_len("tx_symbols", self.symbols.len(), n)?;
        check_len("tx_accounts", self.accounts.len(), n)?;
        check_len("tx_to_accounts", self.to_accounts.len(), n)?;
        check_len("tx_types", self.types.len(), n)?;
        check_len("tx_quantities", self.quantities.len(), n)?;
        check_len("tx_prices", self.prices.len(), n)?;
        check_len("tx_totals", self.totals.len(), n)?;
        check_len("tx_commissions", self.commissions.len(), n)?;
        check_len("tx_split_ratios", self.split_ratios.len(), n)
    }
}

/// Everything about the ledger that is not a per-transaction column.
#[derive(Debug, Clone, Copy)]
pub struct Book<'a> {
    pub num_symbols: usize,
    pub num_accounts: usize,
    pub types: TypeIds,
    pub cash_symbol_id: i64,
    pub stock_qty_close_tolerance: f64,
    pub shortable_symbol_ids: &'a [i64],
    /// Per account: 1 = Auto cash, 0 = Manual.
    pub acc_cash_modes: &'a [i64],
}

impl Book<'_> {
    #[inline]
    fn cell(&self, symbol: i64, account: i64) -> Result<usize, KernelError> {
        let s = wrap("symbol", symbol, self.num_symbols)?;
        let a = wrap("account", account, self.num_accounts)?;
        Ok(s * self.num_accounts + a)
    }

    #[inline]
    fn account(&self, account: i64) -> Result<usize, KernelError> {
        Ok(wrap("account", account, self.num_accounts)?)
    }

    #[inline]
    fn is_auto_cash(&self, account: i64) -> Result<bool, KernelError> {
        let a = wrap("acc_cash_modes", account, self.acc_cash_modes.len())?;
        Ok(self.acc_cash_modes[a] == 1)
    }

    #[inline]
    fn is_shortable(&self, symbol: i64) -> bool {
        self.shortable_symbol_ids.contains(&symbol)
    }
}

/// The cash a dividend row pays: Total Amount when filled, else shares x rate,
/// else a bare Price/Share (the legacy form). `_dividend_amount` in Python.
#[inline]
pub fn dividend_amount(qty: f64, price: f64, total_amount: f64) -> f64 {
    if total_amount.abs() > 1e-9 {
        return total_amount.abs();
    }
    if qty.abs() > 1e-9 && price.abs() > 1e-9 {
        return qty.abs() * price.abs();
    }
    price.abs()
}

/// `amount if |amount| > 1e-9 else fallback`, the fallback chain every cash
/// rule uses.
#[inline]
fn or_else(amount: f64, fallback: f64) -> f64 {
    if amount.abs() > 1e-9 {
        amount.abs()
    } else {
        fallback
    }
}

/// The position state as of one date.
#[derive(Debug, Clone, PartialEq)]
pub struct HoldingsAtDate {
    /// `[symbol][account]`
    pub qty: Vec<f64>,
    /// `[symbol][account]`
    pub cost: Vec<f64>,
    /// `[symbol][account]`, `-1` where the position was never touched.
    pub currency: Vec<i64>,
    /// `[account]`
    pub cash: Vec<f64>,
    /// `[account]`, `-1` where the account never touched cash.
    pub cash_currency: Vec<i64>,
    /// `[symbol][account]`
    pub last_prices: Vec<f64>,
}

/// `_calculate_holdings_numba`.
// One index walks a dozen parallel per-transaction columns.
#[allow(clippy::needless_range_loop)]
pub fn calculate_holdings(
    target_date_ordinal: i64,
    tx: &Transactions,
    tx_local_currencies: &[i64],
    book: &Book,
) -> Result<HoldingsAtDate, KernelError> {
    tx.check()?;
    check_len("tx_local_currencies", tx_local_currencies.len(), tx.len())?;
    let t = book.types;
    let cells = book.num_symbols * book.num_accounts;

    let mut qty_m = vec![0.0_f64; cells];
    let mut cost_m = vec![0.0_f64; cells];
    let mut currency_m = vec![-1_i64; cells];
    let mut short_proceeds = vec![0.0_f64; cells];
    let mut short_orig_qty = vec![0.0_f64; cells];
    let mut cash = vec![0.0_f64; book.num_accounts];
    let mut cash_currency = vec![-1_i64; book.num_accounts];
    let mut last_prices = vec![0.0_f64; cells];

    for i in 0..tx.len() {
        if tx.dates[i] > target_date_ordinal {
            continue;
        }
        let symbol_id = tx.symbols[i];
        let account_id = tx.accounts[i];
        let type_id = tx.types[i];
        let qty = tx.quantities[i];
        let price = tx.prices[i];
        let total_amount = tx.totals[i];
        let commission = tx.commissions[i];
        let split_ratio = tx.split_ratios[i];
        let currency_id = tx_local_currencies[i];

        // --- CASH ---
        if symbol_id == book.cash_symbol_id {
            let acc = book.account(account_id)?;
            if cash_currency[acc] == -1 {
                cash_currency[acc] = currency_id;
            }
            let cash_in = if qty.abs() > 1e-9 {
                qty.abs()
            } else {
                total_amount.abs()
            };
            let cash_out = or_else(total_amount, qty.abs());
            let is_auto_cash = book.is_auto_cash(account_id)?;

            if type_id == t.buy || type_id == t.deposit {
                cash[acc] += cash_in;
            } else if type_id == t.dividend {
                if is_auto_cash {
                    cash[acc] += dividend_amount(qty, price, total_amount);
                }
            } else if type_id == t.interest {
                if is_auto_cash {
                    cash[acc] += cash_out;
                }
            } else if type_id == t.sell || type_id == t.withdrawal {
                cash[acc] -= cash_out;
            } else if type_id == t.fees || type_id == t.tax {
                if is_auto_cash {
                    let fee_val = or_else(total_amount, or_else(qty, commission.abs()));
                    cash[acc] -= fee_val;
                }
            } else if type_id == t.transfer {
                let dest_account_id = tx.to_accounts[i];
                if dest_account_id != -1 {
                    let dest = book.account(dest_account_id)?;
                    if cash_currency[dest] == -1 {
                        cash_currency[dest] = currency_id;
                    }
                    cash[acc] -= cash_out + commission;
                    cash[dest] += cash_out;
                }
            }
            continue;
        }

        // --- STOCK ---
        let c = book.cell(symbol_id, account_id)?;
        if currency_m[c] == -1 {
            currency_m[c] = currency_id;
        }

        if price > 1e-9
            && (type_id == t.buy
                || type_id == t.sell
                || type_id == t.short_sell
                || type_id == t.buy_to_cover
                || type_id == t.transfer
                || type_id == t.spin_off)
        {
            last_prices[c] = price;
        }

        // --- SPIN-OFF ---
        if type_id == t.spin_off {
            if qty > 1e-9 {
                let cost = if total_amount.abs() > 1e-9 {
                    total_amount.abs()
                } else {
                    qty * price
                };
                qty_m[c] += qty;
                cost_m[c] += cost;
            } else {
                let mut moved = total_amount.abs();
                let current_cost = cost_m[c];
                if moved > current_cost {
                    moved = current_cost;
                }
                cost_m[c] = current_cost - moved;
            }
            continue;
        }

        // --- STOCK TRANSFER ---
        if type_id == t.transfer {
            if qty > 1e-9 {
                let source_qty = qty_m[c];
                let source_cost_basis = cost_m[c];
                let transfer_qty = qty;
                let mut cost_to_transfer = 0.0;
                if source_qty.abs() > 1e-9 {
                    let proportion = py_min(transfer_qty / source_qty, 1.0);
                    cost_to_transfer = source_cost_basis * proportion;
                }

                qty_m[c] -= transfer_qty;
                cost_m[c] -= cost_to_transfer;
                if qty_m[c].abs() < book.stock_qty_close_tolerance {
                    qty_m[c] = 0.0;
                    cost_m[c] = 0.0;
                }

                let dest_account_id = tx.to_accounts[i];
                if dest_account_id != -1 {
                    let d = book.cell(symbol_id, dest_account_id)?;
                    if currency_m[d] == -1 {
                        currency_m[d] = currency_id;
                    }
                    qty_m[d] += transfer_qty;
                    cost_m[d] += cost_to_transfer;
                    if last_prices[c] > 1e-9 {
                        last_prices[d] = last_prices[c];
                    }
                }
            }
            continue;
        }

        // --- SPLIT ---
        if type_id == t.split || type_id == t.stock_split {
            if split_ratio > 1e-9 {
                // Every account holding the symbol. Unlike the chronological
                // kernel, the fallback price is NOT rescaled here.
                for acc_idx in 0..book.num_accounts {
                    let k = book.cell(symbol_id, acc_idx as i64)?;
                    if qty_m[k].abs() > 1e-9 {
                        qty_m[k] *= split_ratio;
                        if qty_m[k] < -1e-9 && book.is_shortable(symbol_id) {
                            short_orig_qty[k] *= split_ratio;
                        }
                    }
                }
            } else if split_ratio < -1e-9 {
                // "Target quantity" encoding, this account only.
                let target_qty = -split_ratio;
                let old_qty = qty_m[c];
                if old_qty.abs() > 1e-9 {
                    let implied_ratio = target_qty / old_qty;
                    qty_m[c] = target_qty;
                    if short_orig_qty[c].abs() > 1e-9 {
                        short_orig_qty[c] *= implied_ratio;
                    }
                    if last_prices[c].abs() > 1e-9 && implied_ratio.abs() > 1e-9 {
                        last_prices[c] /= implied_ratio;
                    }
                }
            }
            if commission.abs() > 1e-9 {
                cost_m[c] += commission;
            }
            continue;
        }

        // --- SHORTS ---
        let is_shortable_flag = book.is_shortable(symbol_id);
        if is_shortable_flag && (type_id == t.short_sell || type_id == t.buy_to_cover) {
            let qty_abs = qty.abs();
            if qty_abs > 1e-9 {
                if type_id == t.short_sell {
                    let proceeds = (qty_abs * price) - commission;
                    qty_m[c] -= qty_abs;
                    short_proceeds[c] += proceeds;
                    short_orig_qty[c] += qty_abs;
                    cost_m[c] += commission;
                } else if type_id == t.buy_to_cover {
                    let qty_currently_short = if qty_m[c] < -1e-9 {
                        qty_m[c].abs()
                    } else {
                        0.0
                    };
                    if qty_currently_short > 1e-9 {
                        let qty_covered = py_min(qty_abs, qty_currently_short);
                        let cost_to_cover = (qty_covered * price) + commission;
                        qty_m[c] += qty_covered;
                        cost_m[c] += cost_to_cover;
                        let short_orig = short_orig_qty[c];
                        if short_orig > 1e-9 {
                            let ratio = qty_covered / short_orig;
                            short_proceeds[c] *= 1.0 - ratio;
                            short_orig_qty[c] -= qty_covered;
                        }
                    }
                }
            }
            continue;
        }

        // --- BUY / SELL ---
        if type_id == t.buy || type_id == t.deposit {
            if qty > 1e-9 {
                let cost = (qty * price) + commission;
                qty_m[c] += qty;
                cost_m[c] += cost;
            }
        } else if type_id == t.sell || type_id == t.withdrawal {
            if qty > 1e-9 {
                let held_qty = qty_m[c];
                if held_qty > 1e-9 {
                    let qty_sold = py_min(qty, held_qty);
                    let cost_basis_held = cost_m[c];
                    let cost_sold = qty_sold * (cost_basis_held / held_qty);
                    qty_m[c] -= qty_sold;
                    cost_m[c] -= cost_sold;
                    if qty_m[c].abs() < 1e-9 {
                        qty_m[c] = 0.0;
                        cost_m[c] = 0.0;
                    }
                }
            }
        } else if type_id == t.fees && commission.abs() > 1e-9 {
            cost_m[c] += commission;
        }

        // --- AUTO CASH ---
        if book.is_auto_cash(account_id)? && book.cash_symbol_id >= 0 {
            let mut cash_delta = 0.0;
            if type_id == t.buy {
                cash_delta = -or_else(total_amount, qty.abs() * price + commission);
            } else if type_id == t.sell {
                cash_delta = or_else(total_amount, qty.abs() * price - commission);
            } else if type_id == t.dividend {
                cash_delta = dividend_amount(qty, price, total_amount);
            } else if type_id == t.interest {
                cash_delta = or_else(total_amount, price.abs());
            } else if type_id == t.fees || type_id == t.tax {
                cash_delta = -or_else(total_amount, or_else(commission, price.abs()));
            } else if type_id == t.short_sell {
                // Diverges from the chronological kernel: this one honours
                // Total Amount on a non-shortable short.
                if !is_shortable_flag {
                    cash_delta = or_else(total_amount, qty.abs() * price - commission);
                }
            } else if type_id == t.buy_to_cover && !is_shortable_flag {
                cash_delta = -or_else(total_amount, qty.abs() * price + commission);
            }
            if cash_delta.abs() > 1e-9 {
                let acc = book.account(account_id)?;
                cash[acc] += cash_delta;
            }
        }
    }

    // Final tolerance.
    for s_id in 0..book.num_symbols {
        if s_id as i64 == book.cash_symbol_id {
            continue;
        }
        for a_id in 0..book.num_accounts {
            let k = s_id * book.num_accounts + a_id;
            let q = qty_m[k].abs();
            if 0.0 < q && q < book.stock_qty_close_tolerance {
                qty_m[k] = 0.0;
                cost_m[k] = 0.0;
            }
        }
    }

    Ok(HoldingsAtDate {
        qty: qty_m,
        cost: cost_m,
        currency: currency_m,
        cash,
        cash_currency,
        last_prices,
    })
}

/// Quantities, cash and fallback prices at the end of every day in a range.
#[derive(Debug, Clone, PartialEq)]
pub struct DailyHoldings {
    /// `[day][symbol][account]`
    pub qty: Vec<f64>,
    /// `[day][account]`
    pub cash: Vec<f64>,
    /// `[day][symbol][account]`
    pub last_prices: Vec<f64>,
}

/// `_calculate_daily_holdings_chronological_numba`. Transactions must be sorted
/// by date; each day absorbs every transaction dated on or before it.
pub fn calculate_daily_holdings_chronological(
    date_ordinals: &[i64],
    tx: &Transactions,
    book: &Book,
) -> Result<DailyHoldings, KernelError> {
    tx.check()?;
    let t = book.types;
    let cells = book.num_symbols * book.num_accounts;
    let num_days = date_ordinals.len();

    let mut daily_qty = vec![0.0_f64; num_days * cells];
    let mut daily_cash = vec![0.0_f64; num_days * book.num_accounts];
    let mut daily_last = vec![0.0_f64; num_days * cells];

    let mut qty_m = vec![0.0_f64; cells];
    let mut cash = vec![0.0_f64; book.num_accounts];
    let mut last_prices = vec![0.0_f64; cells];

    let mut tx_idx = 0;
    for (day_idx, &current_date_ordinal) in date_ordinals.iter().enumerate() {
        while tx_idx < tx.len() && tx.dates[tx_idx] <= current_date_ordinal {
            let i = tx_idx;
            let symbol_id = tx.symbols[i];
            let account_id = tx.accounts[i];
            let type_id = tx.types[i];
            let qty = tx.quantities[i];
            let commission = tx.commissions[i];
            let split_ratio = tx.split_ratios[i];
            let price = tx.prices[i];
            let total_amount = tx.totals[i];

            if symbol_id == book.cash_symbol_id {
                let acc = book.account(account_id)?;
                let cash_in = if qty.abs() > 1e-9 {
                    qty.abs()
                } else {
                    total_amount.abs()
                };
                let cash_out = or_else(total_amount, qty.abs());
                let is_auto_cash = book.is_auto_cash(account_id)?;

                if type_id == t.buy || type_id == t.deposit {
                    cash[acc] += cash_in;
                } else if type_id == t.dividend {
                    if is_auto_cash {
                        cash[acc] += dividend_amount(qty, price, total_amount);
                    }
                } else if type_id == t.interest {
                    if is_auto_cash {
                        cash[acc] += cash_out;
                    }
                } else if type_id == t.sell || type_id == t.withdrawal {
                    cash[acc] -= cash_out;
                } else if type_id == t.fees || type_id == t.tax {
                    if is_auto_cash {
                        let fee_val = or_else(total_amount, or_else(qty, commission.abs()));
                        cash[acc] -= fee_val;
                    }
                } else if type_id == t.transfer {
                    let dest_account_id = tx.to_accounts[i];
                    if dest_account_id != -1 {
                        let dest = book.account(dest_account_id)?;
                        cash[acc] -= cash_out + commission;
                        cash[dest] += cash_out;
                    }
                }
            } else {
                let c = book.cell(symbol_id, account_id)?;
                if price > 1e-9
                    && (type_id == t.buy
                        || type_id == t.sell
                        || type_id == t.short_sell
                        || type_id == t.buy_to_cover
                        || type_id == t.transfer
                        || type_id == t.spin_off)
                {
                    last_prices[c] = price;
                }

                if type_id == t.split || type_id == t.stock_split {
                    if split_ratio > 1e-9 {
                        // Diverges from the single-date kernel: the fallback
                        // price is rescaled too, in every account.
                        for acc_idx in 0..book.num_accounts {
                            let k = book.cell(symbol_id, acc_idx as i64)?;
                            if qty_m[k].abs() > 1e-9 {
                                qty_m[k] *= split_ratio;
                            }
                            if last_prices[k].abs() > 1e-9 {
                                last_prices[k] /= split_ratio;
                            }
                        }
                    } else if split_ratio < -1e-9 {
                        let target_qty = -split_ratio;
                        let old_qty = qty_m[c];
                        if old_qty.abs() > 1e-9 {
                            let implied_ratio = target_qty / old_qty;
                            qty_m[c] = target_qty;
                            if last_prices[c].abs() > 1e-9 && implied_ratio.abs() > 1e-9 {
                                last_prices[c] /= implied_ratio;
                            }
                        }
                    }
                } else if type_id == t.buy || type_id == t.deposit {
                    if qty > 1e-9 {
                        qty_m[c] += qty;
                    }
                } else if type_id == t.sell || type_id == t.withdrawal {
                    if qty > 1e-9 {
                        let held_qty = qty_m[c];
                        let qty_sold = if held_qty > 1e-9 {
                            py_min(qty, held_qty)
                        } else {
                            0.0
                        };
                        qty_m[c] -= qty_sold;
                    }
                } else if type_id == t.transfer {
                    if qty > 1e-9 {
                        let transfer_qty = qty;
                        qty_m[c] -= transfer_qty;
                        let dest_account_id = tx.to_accounts[i];
                        if dest_account_id != -1 {
                            let d = book.cell(symbol_id, dest_account_id)?;
                            qty_m[d] += transfer_qty;
                            if last_prices[c] > 1e-9 {
                                last_prices[d] = last_prices[c];
                            }
                        }
                    }
                } else if type_id == t.spin_off {
                    if qty > 1e-9 {
                        qty_m[c] += qty;
                    }
                } else if book.is_shortable(symbol_id) {
                    if type_id == t.short_sell {
                        qty_m[c] -= qty.abs();
                    } else if type_id == t.buy_to_cover {
                        let qty_currently_short = if qty_m[c] < -1e-9 {
                            qty_m[c].abs()
                        } else {
                            0.0
                        };
                        let qty_covered = py_min(qty.abs(), qty_currently_short);
                        qty_m[c] += qty_covered;
                    }
                }

                let sym_is_shortable = book.is_shortable(symbol_id);
                if book.is_auto_cash(account_id)? && book.cash_symbol_id >= 0 {
                    let mut cash_delta = 0.0;
                    if type_id == t.buy {
                        cash_delta = -or_else(total_amount, qty.abs() * price + commission);
                    } else if type_id == t.sell {
                        cash_delta = or_else(total_amount, qty.abs() * price - commission);
                    } else if type_id == t.dividend {
                        cash_delta = dividend_amount(qty, price, total_amount);
                    } else if type_id == t.interest {
                        cash_delta = or_else(total_amount, price.abs());
                    } else if type_id == t.fees || type_id == t.tax {
                        cash_delta = -or_else(total_amount, or_else(commission, price.abs()));
                    } else if type_id == t.short_sell {
                        // Diverges from the single-date kernel: no Total
                        // Amount here.
                        if !sym_is_shortable {
                            cash_delta = qty.abs() * price - commission;
                        }
                    } else if type_id == t.buy_to_cover && !sym_is_shortable {
                        cash_delta = -(qty.abs() * price + commission);
                    }
                    if cash_delta.abs() > 1e-9 {
                        let acc = book.account(account_id)?;
                        cash[acc] += cash_delta;
                    }
                }
            }
            tx_idx += 1;
        }

        for s_id in 0..book.num_symbols {
            if s_id as i64 == book.cash_symbol_id {
                continue;
            }
            for a_id in 0..book.num_accounts {
                let k = s_id * book.num_accounts + a_id;
                let q = qty_m[k].abs();
                if 0.0 < q && q < book.stock_qty_close_tolerance {
                    qty_m[k] = 0.0;
                }
            }
        }

        daily_qty[day_idx * cells..(day_idx + 1) * cells].copy_from_slice(&qty_m);
        let na = book.num_accounts;
        daily_cash[day_idx * na..(day_idx + 1) * na].copy_from_slice(&cash);
        daily_last[day_idx * cells..(day_idx + 1) * cells].copy_from_slice(&last_prices);
    }

    Ok(DailyHoldings {
        qty: daily_qty,
        cash: daily_cash,
        last_prices: daily_last,
    })
}
