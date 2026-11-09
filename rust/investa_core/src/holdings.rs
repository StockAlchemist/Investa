//! Holdings and cash replay for the historical valuation engine.
//!
//! A line-for-line port of the Numba kernel that lived in
//! `portfolio_valuation_kernels.py`:
//! [`calculate_daily_holdings_chronological`] —
//! `_calculate_daily_holdings_chronological_numba`: quantities, cash and last
//! trade prices for every day of a range, in one pass.
//!
//! Its single-date sibling, `_calculate_holdings_numba`, was removed on
//! 9 Nov 2026 with the per-date history path it served. Comments explaining
//! the business rules stay with the Python callers.
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
                        // The fallback price is rescaled too, in every account
                        // (the removed single-date kernel left it alone).
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
                        // No Total Amount here (the removed single-date
                        // kernel used it).
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
