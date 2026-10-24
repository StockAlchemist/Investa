//! The summary engine's position replay: `_process_numba_core` from
//! `portfolio_analyzer.py`, ported line for line.
//!
//! State is one flat array, `[symbol][account][field]` with [`FIELDS`] fields
//! per position. The Python views `current_state = state[sym, acc]` become the
//! offset of that position's first field.

use crate::util::{KernelError, check_len, py_min, wrap};

/// Transaction types as `portfolio_analyzer` numbers them (`TYPE_*`). They are
/// fixed constants there, not per-ledger ids; Python checks they still agree.
pub mod types {
    pub const BUY: i64 = 0;
    pub const SELL: i64 = 1;
    pub const DEPOSIT: i64 = 2;
    pub const WITHDRAWAL: i64 = 3;
    pub const DIVIDEND: i64 = 4;
    pub const FEES: i64 = 5;
    pub const SPLIT: i64 = 6;
    pub const TRANSFER: i64 = 7;
    pub const SHORT_SELL: i64 = 8;
    pub const BUY_TO_COVER: i64 = 9;
    pub const TAX: i64 = 10;
    pub const INTEREST: i64 = 11;
    pub const SPIN_OFF: i64 = 12;
    pub const UNKNOWN: i64 = -1;
}
use types::*;

/// Fields per position.
pub const FIELDS: usize = 14;
const QTY: usize = 0;
const TOTAL_COST: usize = 1;
const REALIZED_GAIN: usize = 2;
const DIVIDENDS: usize = 3;
const COMMISSIONS: usize = 4;
const SHORT_PROCEEDS: usize = 5;
const SHORT_ORIG_QTY: usize = 6;
const COST_INVESTED: usize = 7;
const CUMULATIVE_INVESTMENT: usize = 8;
const BUY_COST: usize = 9;
const COST_DISPLAY_HIST_FX: usize = 10;
const CURRENCY: usize = 11;
const REALIZED_GAIN_DISPLAY: usize = 12;
const TAXES: usize = 13;

/// One column per field, one row per transaction, in ledger order.
#[derive(Debug, Clone, Copy)]
pub struct Ledger<'a> {
    pub sym_ids: &'a [i64],
    pub acc_ids: &'a [i64],
    pub type_ids: &'a [i64],
    pub qtys: &'a [f64],
    pub prices: &'a [f64],
    pub comms: &'a [f64],
    pub totals: &'a [f64],
    pub split_ratios: &'a [f64],
    pub to_acc_ids: &'a [i64],
    pub local_curr_ids: &'a [i64],
    pub fx_rates_hist: &'a [f64],
}

#[derive(Debug, Clone, PartialEq)]
pub struct Replay {
    /// `[symbol][account][field]`
    pub state: Vec<f64>,
    /// Per transaction: the unit cost a transfer carried, NaN elsewhere.
    pub transfer_costs: Vec<f64>,
}

// One index walks a dozen parallel per-transaction columns.
#[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
pub fn process_core(
    ledger: &Ledger,
    shortable_sym_ids: &[i64],
    stock_qty_tol: f64,
    cash_sym_id: i64,
    num_tx: usize,
    num_syms: usize,
    num_accs: usize,
    acc_cash_modes: &[i64],
) -> Result<Replay, KernelError> {
    let l = ledger;
    for (what, len) in [
        ("sym_ids", l.sym_ids.len()),
        ("acc_ids", l.acc_ids.len()),
        ("type_ids", l.type_ids.len()),
        ("qtys", l.qtys.len()),
        ("prices", l.prices.len()),
        ("comms", l.comms.len()),
        ("totals", l.totals.len()),
        ("split_ratios", l.split_ratios.len()),
        ("to_acc_ids", l.to_acc_ids.len()),
        ("local_curr_ids", l.local_curr_ids.len()),
        ("fx_rates_hist", l.fx_rates_hist.len()),
    ] {
        check_len(what, len, num_tx)?;
    }

    let mut st = vec![0.0_f64; num_syms * num_accs * FIELDS];
    for p in 0..num_syms * num_accs {
        st[p * FIELDS + CURRENCY] = -1.0;
    }
    let mut transfer_costs = vec![f64::NAN; num_tx];

    let pos = |sym: i64, acc: i64| -> Result<usize, KernelError> {
        let s = wrap("symbol", sym, num_syms)?;
        let a = wrap("account", acc, num_accs)?;
        Ok((s * num_accs + a) * FIELDS)
    };
    let is_auto = |acc: i64| -> Result<bool, KernelError> {
        Ok(acc_cash_modes[wrap("acc_cash_modes", acc, acc_cash_modes.len())?] == 1)
    };
    let is_shortable = |sym: i64| shortable_sym_ids.contains(&sym);

    for i in 0..num_tx {
        let sym = l.sym_ids[i];
        let acc = l.acc_ids[i];
        let typ = l.type_ids[i];
        let qty = l.qtys[i];
        let price = l.prices[i];
        let comm = l.comms[i];
        let total = l.totals[i];
        let split = l.split_ratios[i];
        let to_acc = l.to_acc_ids[i];
        let curr = l.local_curr_ids[i];
        let mut fx_rate = l.fx_rates_hist[i];
        if fx_rate.is_nan() {
            fx_rate = 1.0;
        }

        let cs = pos(sym, acc)?;
        // First use fixes the position's currency. A later mismatch was meant
        // to skip the row but never did; neither does this.
        if st[cs + CURRENCY] == -1.0 {
            st[cs + CURRENCY] = curr as f64;
        }

        // --- SPLIT ---
        if typ == SPLIT {
            if split > 1e-9 {
                // A ratio applies to every account holding the symbol.
                for a_idx in 0..num_accs {
                    let h = pos(sym, a_idx as i64)?;
                    if st[h + CURRENCY] != -1.0 {
                        let old_qty = st[h + QTY];
                        if old_qty.abs() >= 1e-9 {
                            st[h + QTY] *= split;
                            if is_shortable(sym) && old_qty < -1e-9 {
                                st[h + SHORT_ORIG_QTY] *= split;
                                if st[h + SHORT_ORIG_QTY].abs() < 1e-9 {
                                    st[h + SHORT_ORIG_QTY] = 0.0;
                                }
                            }
                            if st[h + QTY].abs() < 1e-9 {
                                st[h + QTY] = 0.0;
                            }
                        }
                    }
                }
                if comm != 0.0 {
                    let fee_cost = comm.abs();
                    st[cs + COMMISSIONS] += fee_cost;
                    st[cs + COST_INVESTED] += fee_cost;
                    st[cs + CUMULATIVE_INVESTMENT] += fee_cost;
                }
            } else if split < -1e-9 {
                // A target quantity applies to this account only.
                let target_qty = split.abs();
                if st[cs + CURRENCY] != -1.0 {
                    let old_qty = st[cs + QTY];
                    if old_qty.abs() > 1e-9 {
                        let ratio = target_qty / old_qty;
                        st[cs + QTY] = target_qty;
                        if st[cs + SHORT_ORIG_QTY].abs() > 1e-9 {
                            st[cs + SHORT_ORIG_QTY] *= ratio;
                        }
                    }
                }
                if comm != 0.0 {
                    let fee_cost = comm.abs();
                    st[cs + COMMISSIONS] += fee_cost;
                    st[cs + COST_INVESTED] += fee_cost;
                    st[cs + CUMULATIVE_INVESTMENT] += fee_cost;
                }
            }
            continue;
        }

        // --- TRANSFER ---
        if typ == TRANSFER {
            if qty > 1e-9 && to_acc != -1 && to_acc != acc {
                let from_qty = st[cs + QTY];
                let ts = pos(sym, to_acc)?;
                if sym == cash_sym_id || from_qty >= qty - 1e-9 {
                    if st[ts + CURRENCY] == -1.0 {
                        st[ts + CURRENCY] = st[cs + CURRENCY];
                    }
                    let (cost_t, cost_t_hist, invested_t, cumulative_t, buy_cost_t);
                    if sym == cash_sym_id {
                        cost_t = qty;
                        cost_t_hist = qty * fx_rate;
                        invested_t = qty;
                        cumulative_t = qty;
                        buy_cost_t = qty;
                    } else {
                        let mut proportion = 0.0;
                        if from_qty > 1e-9 {
                            proportion = qty / from_qty;
                        }
                        cost_t = st[cs + TOTAL_COST] * proportion;
                        cost_t_hist = st[cs + COST_DISPLAY_HIST_FX] * proportion;
                        invested_t = st[cs + COST_INVESTED] * proportion;
                        cumulative_t = st[cs + CUMULATIVE_INVESTMENT] * proportion;
                        buy_cost_t = st[cs + BUY_COST] * proportion;
                    }

                    st[cs + QTY] -= qty;
                    st[cs + TOTAL_COST] -= cost_t;
                    st[cs + COST_DISPLAY_HIST_FX] -= cost_t_hist;
                    st[cs + COST_INVESTED] -= invested_t;
                    st[cs + CUMULATIVE_INVESTMENT] -= cumulative_t;
                    st[cs + BUY_COST] -= buy_cost_t;

                    if st[cs + QTY].abs() < stock_qty_tol {
                        for f in [
                            QTY,
                            TOTAL_COST,
                            COST_DISPLAY_HIST_FX,
                            COST_INVESTED,
                            CUMULATIVE_INVESTMENT,
                            BUY_COST,
                        ] {
                            st[cs + f] = 0.0;
                        }
                    }

                    st[ts + QTY] += qty;
                    st[ts + TOTAL_COST] += cost_t;
                    st[ts + COST_DISPLAY_HIST_FX] += cost_t_hist;
                    st[ts + COST_INVESTED] += invested_t;
                    st[ts + CUMULATIVE_INVESTMENT] += cumulative_t;
                    st[ts + BUY_COST] += buy_cost_t;

                    transfer_costs[i] = cost_t / qty;
                }
            }
            continue;
        }

        // --- SPIN-OFF ---
        if typ == SPIN_OFF {
            let amt = total.abs();
            if qty.abs() > 1e-9 {
                st[cs + QTY] += qty.abs();
                st[cs + TOTAL_COST] += amt;
                st[cs + COST_INVESTED] += amt;
                st[cs + CUMULATIVE_INVESTMENT] += amt;
                st[cs + BUY_COST] += amt;
                st[cs + COST_DISPLAY_HIST_FX] += amt * fx_rate;
            } else {
                let base = st[cs + TOTAL_COST];
                let red = if amt < base { amt } else { base };
                let proportion = if base > 1e-9 { red / base } else { 0.0 };
                st[cs + TOTAL_COST] -= red;
                st[cs + COST_INVESTED] -= red;
                st[cs + CUMULATIVE_INVESTMENT] -= red;
                st[cs + BUY_COST] -= red;
                st[cs + COST_DISPLAY_HIST_FX] -= st[cs + COST_DISPLAY_HIST_FX] * proportion;
            }
            continue;
        }

        // --- STANDARD TYPES ---
        let qty_abs = qty.abs();
        let shortable = is_shortable(sym);

        if shortable && (typ == SHORT_SELL || typ == BUY_TO_COVER) {
            if typ == SHORT_SELL {
                let proceeds = (qty_abs * price) - comm;
                st[cs + QTY] -= qty_abs;
                st[cs + SHORT_PROCEEDS] += proceeds;
                st[cs + SHORT_ORIG_QTY] += qty_abs;
                st[cs + COMMISSIONS] += comm;
                st[cs + CUMULATIVE_INVESTMENT] -= proceeds;
                st[cs + COST_DISPLAY_HIST_FX] -= proceeds * fx_rate;
            } else {
                let qty_curr_short = if st[cs + QTY] < -1e-9 {
                    st[cs + QTY].abs()
                } else {
                    0.0
                };
                let qty_covered = py_min(qty_abs, qty_curr_short);
                let cost = (qty_covered * price) + comm;

                if st[cs + SHORT_ORIG_QTY] > 1e-9 {
                    let (avg_proceeds_display_neg, avg_proceeds) =
                        if st[cs + SHORT_ORIG_QTY].abs() > 1e-9 {
                            (
                                st[cs + COST_DISPLAY_HIST_FX] / st[cs + SHORT_ORIG_QTY],
                                st[cs + SHORT_PROCEEDS] / st[cs + SHORT_ORIG_QTY],
                            )
                        } else {
                            (0.0, 0.0)
                        };
                    let proceeds_display_attr = -(avg_proceeds_display_neg * qty_covered);
                    let proceeds_attr = qty_covered * avg_proceeds;
                    let gain = proceeds_attr - cost;

                    st[cs + QTY] += qty_covered;
                    st[cs + SHORT_PROCEEDS] -= proceeds_attr;
                    st[cs + SHORT_ORIG_QTY] -= qty_covered;
                    st[cs + COMMISSIONS] += comm;
                    st[cs + REALIZED_GAIN] += gain;
                    st[cs + CUMULATIVE_INVESTMENT] += cost;
                    st[cs + COST_DISPLAY_HIST_FX] -= avg_proceeds_display_neg * qty_covered;

                    let cost_display_cover = cost * fx_rate;
                    let gain_display = proceeds_display_attr - cost_display_cover;
                    st[cs + REALIZED_GAIN_DISPLAY] += gain_display;

                    if st[cs + SHORT_ORIG_QTY].abs() < 1e-9 {
                        st[cs + SHORT_PROCEEDS] = 0.0;
                        st[cs + SHORT_ORIG_QTY] = 0.0;
                    }
                    if st[cs + QTY].abs() < 1e-9 {
                        st[cs + QTY] = 0.0;
                    }
                }
            }
            continue;
        }

        if typ == BUY || typ == DEPOSIT {
            let cost = if total.abs() > 1e-9 {
                total.abs()
            } else {
                (qty_abs * price) + comm
            };
            st[cs + QTY] += qty_abs;
            st[cs + TOTAL_COST] += cost;
            st[cs + COMMISSIONS] += comm;
            st[cs + COST_INVESTED] += cost;
            st[cs + CUMULATIVE_INVESTMENT] += cost;
            st[cs + BUY_COST] += cost;
            st[cs + COST_DISPLAY_HIST_FX] += cost * fx_rate;
        } else if typ == SELL || typ == WITHDRAWAL {
            let held_qty = st[cs + QTY];
            if sym == cash_sym_id {
                // Cash may go negative; no held-quantity limit.
                let eff_price = if price.abs() > 1e-9 { price } else { 1.0 };
                let qty_to_deduct = if total.abs() > 1e-9 {
                    total.abs()
                } else {
                    qty_abs
                };
                let cost_sold = qty_to_deduct * eff_price;
                let cost_sold_hist = cost_sold * fx_rate;
                let proceeds = (qty_to_deduct * eff_price) - comm;
                let gain = 0.0;
                let gain_display = 0.0;

                st[cs + QTY] -= qty_to_deduct;
                st[cs + TOTAL_COST] -= cost_sold;
                st[cs + COST_DISPLAY_HIST_FX] -= cost_sold_hist;
                st[cs + COMMISSIONS] += comm;
                st[cs + REALIZED_GAIN] += gain;
                st[cs + REALIZED_GAIN_DISPLAY] += gain_display;
                st[cs + COST_INVESTED] -= cost_sold;
                st[cs + CUMULATIVE_INVESTMENT] -= proceeds;
            } else if held_qty > 1e-9 {
                let qty_sold = py_min(qty_abs, held_qty);
                let mut cost_sold = 0.0;
                let mut cost_sold_hist = 0.0;
                if st[cs + TOTAL_COST].abs() > 1e-9 {
                    cost_sold = qty_sold * (st[cs + TOTAL_COST] / held_qty);
                    cost_sold_hist = qty_sold * (st[cs + COST_DISPLAY_HIST_FX] / held_qty);
                }
                let proceeds = if total.abs() > 1e-9 {
                    total.abs()
                } else {
                    (qty_sold * price) - comm
                };
                let gain = proceeds - cost_sold;
                let proceeds_display = proceeds * fx_rate;
                let gain_display = proceeds_display - cost_sold_hist;

                st[cs + QTY] -= qty_sold;
                st[cs + TOTAL_COST] -= cost_sold;
                st[cs + COST_DISPLAY_HIST_FX] -= cost_sold_hist;
                st[cs + COMMISSIONS] += comm;
                st[cs + REALIZED_GAIN] += gain;
                st[cs + REALIZED_GAIN_DISPLAY] += gain_display;
                st[cs + COST_INVESTED] -= cost_sold;
                st[cs + CUMULATIVE_INVESTMENT] -= proceeds;

                if st[cs + QTY].abs() < 1e-9 {
                    st[cs + QTY] = 0.0;
                    st[cs + TOTAL_COST] = 0.0;
                    st[cs + COST_DISPLAY_HIST_FX] = 0.0;
                }
            }
        } else if typ == DIVIDEND {
            let div_amt = price;
            let mut div_effect = div_amt.abs();
            if st[cs + QTY] < -1e-9 {
                div_effect = -div_effect;
            }
            st[cs + DIVIDENDS] += div_effect;
            st[cs + COMMISSIONS] += comm;
            st[cs + COST_DISPLAY_HIST_FX] -= div_effect * fx_rate;

            if sym == cash_sym_id && is_auto(acc)? {
                st[cs + QTY] += div_effect;
                if div_effect > 0.0 {
                    st[cs + TOTAL_COST] += div_effect;
                    st[cs + COST_INVESTED] += div_effect;
                    st[cs + CUMULATIVE_INVESTMENT] += div_effect;
                    st[cs + BUY_COST] += div_effect;
                } else {
                    let abs_div = div_effect.abs();
                    st[cs + TOTAL_COST] -= abs_div;
                    st[cs + COST_INVESTED] -= abs_div;
                    st[cs + CUMULATIVE_INVESTMENT] -= abs_div;
                }
            }
        } else if typ == FEES || typ == TAX {
            let cost_val = if total.abs() > 1e-9 {
                total.abs()
            } else if comm.abs() > 1e-9 {
                comm.abs()
            } else {
                price.abs()
            };
            if typ == TAX {
                st[cs + TAXES] += cost_val;
            } else {
                st[cs + COMMISSIONS] += cost_val;
            }
            st[cs + COST_INVESTED] += cost_val;
            st[cs + CUMULATIVE_INVESTMENT] += cost_val;
            st[cs + COST_DISPLAY_HIST_FX] += cost_val * fx_rate;

            if sym == cash_sym_id && is_auto(acc)? {
                st[cs + QTY] -= cost_val;
                st[cs + TOTAL_COST] -= cost_val;
                st[cs + COST_INVESTED] -= cost_val;
                st[cs + CUMULATIVE_INVESTMENT] -= cost_val;
            }
        } else if typ == INTEREST {
            let int_amt = if total.abs() > 1e-9 {
                total.abs()
            } else {
                price.abs()
            };
            st[cs + DIVIDENDS] += int_amt;
            st[cs + COMMISSIONS] += comm;
            st[cs + COST_DISPLAY_HIST_FX] -= int_amt * fx_rate;

            if sym == cash_sym_id && is_auto(acc)? {
                st[cs + QTY] += int_amt;
                st[cs + TOTAL_COST] += int_amt;
                st[cs + COST_INVESTED] += int_amt;
                st[cs + CUMULATIVE_INVESTMENT] += int_amt;
                st[cs + BUY_COST] += int_amt;
            }
        }

        // --- AUTO CASH ---
        if is_auto(acc)? && sym != cash_sym_id && cash_sym_id >= 0 {
            let cash = pos(cash_sym_id, acc)?;
            if st[cash + CURRENCY] == -1.0 {
                st[cash + CURRENCY] = curr as f64;
            }
            let mut cash_delta = 0.0;
            if typ == BUY {
                cash_delta = -(if total.abs() > 1e-9 {
                    total.abs()
                } else {
                    (qty_abs * price) + comm.abs()
                });
            } else if typ == SELL {
                cash_delta = if total.abs() > 1e-9 {
                    total.abs()
                } else {
                    (qty_abs * price) - comm.abs()
                };
            } else if typ == DIVIDEND {
                cash_delta = price.abs();
            } else if typ == INTEREST {
                cash_delta = if total.abs() > 1e-9 {
                    total.abs()
                } else {
                    price.abs()
                };
            } else if typ == FEES || typ == TAX {
                cash_delta = -(if total.abs() > 1e-9 {
                    total.abs()
                } else if comm.abs() > 1e-9 {
                    comm.abs()
                } else {
                    price.abs()
                });
            } else if typ == SHORT_SELL {
                cash_delta = (qty_abs * price) - comm.abs();
            } else if typ == BUY_TO_COVER {
                cash_delta = -((qty_abs * price) + comm.abs());
            }

            if cash_delta.abs() > 1e-9 {
                st[cash + QTY] += cash_delta;
                if cash_delta > 0.0 {
                    st[cash + TOTAL_COST] += cash_delta;
                    st[cash + COST_INVESTED] += cash_delta;
                    st[cash + CUMULATIVE_INVESTMENT] += cash_delta;
                    st[cash + BUY_COST] += cash_delta;
                    st[cash + COST_DISPLAY_HIST_FX] += cash_delta * fx_rate;
                } else {
                    let abs_delta = cash_delta.abs();
                    st[cash + TOTAL_COST] -= abs_delta;
                    st[cash + COST_INVESTED] -= abs_delta;
                    st[cash + CUMULATIVE_INVESTMENT] -= abs_delta;
                    st[cash + COST_DISPLAY_HIST_FX] -= abs_delta * fx_rate;
                }
            }
        } else if typ == TAX {
            // Reached whenever the Auto-cash branch above is not, which
            // includes every Manual-mode tax row: those are booked to the tax
            // bucket a second time here. Preserved as the Python engine has it.
            let tax_amt = if price.abs() > 1e-9 {
                price.abs()
            } else {
                comm.abs()
            };
            st[cs + TAXES] += tax_amt;
            st[cs + DIVIDENDS] -= tax_amt;
            st[cs + COST_DISPLAY_HIST_FX] += tax_amt * fx_rate;
        }
    }

    Ok(Replay {
        state: st,
        transfer_costs,
    })
}
