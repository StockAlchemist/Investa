//! `portfolio_analyzer._calculate_aggregate_metrics`: per-account and overall
//! totals over the summary rows, their day-change and return percentages,
//! and the overall and per-account MWR.
//!
//! Ported line for line from `src/portfolio_analyzer.py` (frozen in
//! `tests/reference/portfolio_analyzer.py`): the column sums are the module's
//! `safe_sum`, the row selections the same pandas operations, and every
//! ratio Python's own arithmetic on what those return, `* 100` and `* 100.0`
//! as the original wrote them. `safe_sum`, `get_cash_flows_for_mwr` and
//! `calculate_irr` are looked up on `portfolio_analyzer` at call time.

use pyo3::basic::CompareOp;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::mdb::unpack2;
use crate::pa_rows::{Ops, once};
use crate::pyhelp::{es, log, log_exc};

const ACCOUNT_DEFAULTS: [(&str, f64); 14] = [
    ("mwr", f64::NAN),
    ("total_return_pct", f64::NAN),
    ("total_market_value_display", 0.0),
    ("total_realized_gain_display", 0.0),
    ("total_unrealized_gain_display", 0.0),
    ("total_dividends_display", 0.0),
    ("total_commissions_display", 0.0),
    ("total_taxes_display", 0.0),
    ("total_gain_display", 0.0),
    ("total_cash_display", 0.0),
    ("total_cost_invested_display", 0.0),
    ("total_buy_cost_display", 0.0),
    ("total_day_change_display", 0.0),
    ("total_day_change_percent", f64::NAN),
];

/// `account_level_metrics[account]` on the original's defaultdict.
fn entry<'py>(
    metrics: &Bound<'py, PyDict>,
    account: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if let Some(e) = metrics.get_item(account)? {
        return Ok(e);
    }
    let e = PyDict::new(metrics.py());
    for (k, v) in ACCOUNT_DEFAULTS {
        e.set_item(k, v)?;
    }
    metrics.set_item(account, &e)?;
    Ok(e.into_any())
}

/// The day-change percentage of a total against the previous close.
fn day_change_percent<'py>(
    o: &Ops<'py>,
    change: &Bound<'py, PyAny>,
    prev_close: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if o.notna(change)? && o.notna(prev_close)? {
        if o.abs_gt(prev_close, 1e-9)? {
            return Ok(Some(change.div(prev_close)?.mul(100.0)?));
        } else if o.abs_gt(change, 1e-9)? {
            let inf = if change.gt(0)? {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
            return Ok(Some(o.fl(inf)));
        } else if o.abs(change)?.lt(1e-9)? {
            return Ok(Some(o.fl(0.0)));
        }
    }
    Ok(None)
}

/// `gain / denominator * 100.0`, 0.0 for no gain on no money, else NaN.
fn return_pct<'py>(
    o: &Ops<'py>,
    gain: &Bound<'py, PyAny>,
    denominator: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if o.notna(gain)? && o.notna(denominator)? {
        if o.abs_gt(denominator, 1e-9)? {
            return gain.div(denominator)?.mul(100.0);
        } else if o.abs(gain)?.le(1e-9)? {
            return Ok(o.fl(0.0));
        }
    }
    Ok(o.nan())
}

/// `transactions_df[name].astype(str).str.strip().str.upper()`
fn upper<'py>(o: &Ops<'py>, df: &Bound<'py, PyAny>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    df.get_item(name)?
        .call_method1("astype", (&o.str_t,))?
        .getattr("str")?
        .call_method0("strip")?
        .getattr("str")?
        .call_method0("upper")
}

/// `get_cash_flows_for_mwr(...)` on the portfolio basis.
#[allow(clippy::too_many_arguments)]
fn mwr_flows<'py>(
    m: &Bound<'py, PyModule>,
    txs: &Bound<'py, PyAny>,
    market_value: &Bound<'py, PyAny>,
    report_date: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    fx_rates: &Bound<'py, PyAny>,
    historical_fx_rates: &Bound<'py, PyAny>,
    scope: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let kw = PyDict::new(m.py());
    kw.set_item("account_transactions", txs)?;
    kw.set_item("final_account_market_value", market_value)?;
    kw.set_item("end_date", report_date)?;
    kw.set_item("target_currency", display_currency)?;
    kw.set_item("fx_rates", fx_rates)?;
    kw.set_item("historical_fx_rates", historical_fx_rates)?;
    kw.set_item("display_currency", display_currency)?;
    kw.set_item("include_accounts", scope)?;
    kw.set_item("flow_basis", "portfolio")?;
    unpack2(&m.getattr("get_cash_flows_for_mwr")?.call((), Some(&kw))?)
}

type AggregateOut<'py> = (Bound<'py, PyDict>, Bound<'py, PyDict>, bool, bool);

/// `_calculate_aggregate_metrics`: (overall metrics, per-account metrics,
/// has errors, has warnings).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn pa_aggregate_metrics<'py>(
    py: Python<'py>,
    full_summary_df: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    report_date: &Bound<'py, PyAny>,
    fx_rates: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    all_available_accounts: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    historical_fx_rates: &Bound<'py, PyAny>,
) -> PyResult<AggregateOut<'py>> {
    let m = py.import("portfolio_analyzer")?;
    let o = Ops::new(py)?;
    let metrics = PyDict::new(py);
    let has_errors = false;
    let mut has_warnings = false;
    let dc = o.fmt(display_currency)?;
    let col = |label: &str| format!("{label} ({dc})");
    let safe_sum = |df: &Bound<'py, PyAny>, c: &str| -> PyResult<Bound<'py, PyAny>> {
        m.getattr("safe_sum")?.call1((df, c))
    };
    let has_col =
        |df: &Bound<'py, PyAny>, c: &str| -> PyResult<bool> { df.getattr("columns")?.contains(c) };
    let report_str = || report_date.call_method1("strftime", ("%Y-%m-%d",));

    log(py, "info", "Calculating Account-Level & Overall Metrics...")?;
    if full_summary_df.is_none() || full_summary_df.getattr("empty")?.is_truthy()? {
        log(py, "warning", "Input 'full_summary_df' is empty or None.")?;
        let overall = PyDict::new(py);
        for (k, v) in [
            ("market_value", 0.0),
            ("cost_basis_held", 0.0),
            ("unrealized_gain", 0.0),
            ("realized_gain", 0.0),
            ("dividends", 0.0),
            ("commissions", 0.0),
            ("taxes", 0.0),
            ("total_gain", 0.0),
            ("total_cost_invested", 0.0),
            ("total_buy_cost", 0.0),
            ("portfolio_mwr", f64::NAN),
            ("day_change_display", 0.0),
            ("day_change_percent", f64::NAN),
        ] {
            overall.set_item(k, v)?;
        }
        overall.set_item("report_date", report_str()?)?;
        overall.set_item("display_currency", display_currency)?;
        overall.set_item("cumulative_investment", 0.0)?;
        overall.set_item("total_return_pct", f64::NAN)?;
        overall.set_item("total_taxes", 0.0)?;
        overall.set_item("est_annual_income_display", 0.0)?;
        return Ok((overall, metrics.copy()?, has_errors, true));
    }

    let required = [
        "Account".to_string(),
        "Symbol".to_string(),
        "Quantity".to_string(),
        col("Market Value"),
        col("Total Gain"),
        col("Total Buy Cost"),
    ];
    let mut missing = Vec::new();
    for c in &required {
        if !has_col(full_summary_df, c)? {
            missing.push(c.as_str());
        }
    }
    if !missing.is_empty() {
        log(
            py,
            "warning",
            format!(
                "Warning: Aggregate calculation input missing required columns: {}. Totals may be inaccurate.",
                o.fmt(PyList::new(py, &missing)?.as_any())?
            ),
        )?;
        has_warnings = true;
    }

    let mv_col = col("Market Value");
    let dc_col = col("Day Change");
    for account in full_summary_df
        .get_item("Account")?
        .call_method0("unique")?
        .try_iter()?
    {
        let account = account?;
        let fa = o.fmt(&account)?;
        let r = (|| -> PyResult<()> {
            let account_full_df = full_summary_df.get_item(
                full_summary_df
                    .get_item("Account")?
                    .rich_compare(&account, CompareOp::Eq)?,
            )?;
            let me = entry(&metrics, &account)?;
            me.set_item("mwr", f64::NAN)?;
            for label in [
                "Market Value",
                "Realized Gain",
                "Unreal. Gain",
                "Dividends",
                "Commissions",
                "Total Gain",
                "Total Cost Invested",
                "Total Buy Cost",
                "Day Change",
                "FX Gain/Loss",
                "Tax",
            ] {
                let c = col(label);
                if !has_col(&account_full_df, &c)? {
                    log(
                        py,
                        "warning",
                        format!("Warn: Col '{c}' missing for acc '{fa}' agg."),
                    )?;
                    has_warnings = true;
                }
            }

            let fx_gl = safe_sum(&account_full_df, &col("FX Gain/Loss"))?;
            me.set_item("fx_gain_loss_display", &fx_gl)?;
            let cost_basis = safe_sum(&account_full_df, &col("Cost Basis"))?;
            let fx_gl_pct = if o.abs_gt(&cost_basis, 1e-9)? {
                fx_gl.div(&cost_basis)?.mul(100)?
            } else {
                o.nan()
            };
            me.set_item("fx_gain_loss_pct", fx_gl_pct)?;

            me.set_item(
                "total_market_value_display",
                safe_sum(&account_full_df, &mv_col)?,
            )?;
            me.set_item(
                "total_realized_gain_display",
                safe_sum(&account_full_df, &col("Realized Gain"))?,
            )?;
            me.set_item(
                "total_unrealized_gain_display",
                safe_sum(&account_full_df, &col("Unreal. Gain"))?,
            )?;
            me.set_item(
                "total_dividends_display",
                safe_sum(&account_full_df, &col("Dividends"))?,
            )?;
            let taxes_col = col("Taxes");
            let taxes = if has_col(&account_full_df, &taxes_col)? {
                safe_sum(&account_full_df, &taxes_col)?
            } else {
                o.fl(0.0)
            };
            me.set_item("total_taxes_display", taxes)?;
            me.set_item(
                "total_commissions_display",
                safe_sum(&account_full_df, &col("Commissions"))?,
            )?;
            me.set_item(
                "total_gain_display",
                safe_sum(&account_full_df, &col("Total Gain"))?,
            )?;
            me.set_item(
                "total_taxes_display",
                safe_sum(&account_full_df, &col("Tax"))?,
            )?;

            if has_col(&account_full_df, "Name")? {
                let cash_rows = account_full_df.get_item(
                    account_full_df
                        .get_item("Name")?
                        .rich_compare("Cash", CompareOp::Eq)?,
                )?;
                me.set_item("total_cash_display", safe_sum(&cash_rows, &mv_col)?)?;
            } else {
                me.set_item("total_cash_display", 0.0)?;
            }
            me.set_item(
                "total_cost_invested_display",
                safe_sum(&account_full_df, &col("Total Cost Invested"))?,
            )?;
            let buy_cost = safe_sum(&account_full_df, &col("Total Buy Cost"))?;
            me.set_item("total_buy_cost_display", &buy_cost)?;
            let total_gain = me.get_item("total_gain_display")?;
            me.set_item("total_return_pct", return_pct(&o, &total_gain, &buy_cost)?)?;

            let day_change = safe_sum(&account_full_df, &dc_col)?;
            // Log the top contributors to the day change.
            if o.abs(&day_change)?.gt(0)? {
                let cols = PyList::new(py, ["Symbol", dc_col.as_str(), mv_col.as_str()])?;
                let breakdown = account_full_df.get_item(cols)?.call_method0("copy")?;
                let kw = PyDict::new(py);
                kw.set_item("by", &dc_col)?;
                kw.set_item("key", &o.abs)?;
                kw.set_item("ascending", false)?;
                let breakdown = breakdown
                    .call_method("sort_values", (), Some(&kw))?
                    .call_method1("head", (5,))?;
                log(
                    py,
                    "info",
                    format!(
                        "Day Change Breakdown for {fa}: Total={}",
                        o.fmt(&day_change)?
                    ),
                )?;
                for item in breakdown.call_method0("iterrows")?.try_iter()? {
                    let (_, row) = unpack2(&item?)?;
                    log(
                        py,
                        "info",
                        format!(
                            "  > {}: DayChg={}, MV={}",
                            o.fmt(&row.get_item("Symbol")?)?,
                            o.fmt(&row.get_item(&dc_col)?)?,
                            o.fmt(&row.get_item(&mv_col)?)?
                        ),
                    )?;
                }
            }
            me.set_item("total_day_change_display", &day_change)?;
            let current_mv = me.get_item("total_market_value_display")?;
            let mut prev_close = o.nan();
            if o.notna(&current_mv)? && o.notna(&day_change)? {
                prev_close = current_mv.sub(&day_change)?;
            }
            me.set_item("total_day_change_percent", f64::NAN)?;
            if let Some(pct) = day_change_percent(&o, &day_change, &prev_close)? {
                me.set_item("total_day_change_percent", pct)?;
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log_exc(
                    py,
                    "error",
                    format!("Error aggregating metrics for account '{fa}'"),
                    &e,
                )?;
                has_warnings = true;
            }
            Err(e) => return Err(e),
        }
    }

    // --- The overall summary covers the selected accounts only ---
    let mut df_overall = full_summary_df.clone();
    let is_all_accounts_selected = if !include_accounts.is_truthy()? {
        true
    } else {
        let set = py.import("builtins")?.getattr("set")?;
        all_available_accounts.is_truthy()?
            && set
                .call1((include_accounts,))?
                .eq(set.call1((all_available_accounts,))?)?
    };
    if !is_all_accounts_selected {
        log(
            py,
            "debug",
            format!(
                "Aggregating overall metrics for selected accounts: {}",
                o.fmt(include_accounts)?
            ),
        )?;
        if has_col(&df_overall, "Account")? {
            let accounts = full_summary_df.get_item("Account")?;
            let account_mask = accounts.call_method1("isin", (include_accounts,))?;
            let cash_mask = accounts
                .rich_compare(m.getattr("_AGGREGATE_CASH_ACCOUNT_NAME_")?, CompareOp::Eq)?;
            df_overall = full_summary_df.get_item(account_mask.bitor(cash_mask)?)?;
        } else {
            log(
                py,
                "warning",
                "Cannot filter overall metrics: 'Account' column missing.",
            )?;
        }
    }

    let total_gain_col = col("Total Gain");
    let total_buy_cost_col = col("Total Buy Cost");
    let cost_basis_col = col("Cost Basis");
    let unreal_gain_col = col("Unreal. Gain");
    let real_gain_col = col("Realized Gain");
    let divs_col = col("Dividends");
    let comm_col = col("Commissions");
    let taxes_col = col("Taxes");
    let cost_invest_col = col("Total Cost Invested");
    let cum_invest_col = col("Cumulative Investment");
    let fx_gl_col = col("FX Gain/Loss");
    let tax_col = col("Tax");
    for c in [
        &mv_col,
        &total_gain_col,
        &total_buy_cost_col,
        &cost_basis_col,
        &unreal_gain_col,
        &real_gain_col,
        &divs_col,
        &comm_col,
        &cost_invest_col,
        &cum_invest_col,
        &dc_col,
        &fx_gl_col,
        &tax_col,
    ] {
        if !has_col(&df_overall, c)? {
            log(
                py,
                "warning",
                format!("Warning: Column '{c}' missing for overall aggregation."),
            )?;
            has_warnings = true;
        }
    }

    let overall_fx_gl = safe_sum(&df_overall, &fx_gl_col)?;
    let overall_mv = safe_sum(&df_overall, &mv_col)?;

    let mut held_mask = o.pd.getattr("Series")?.call(
        (false,),
        Some(&{
            let kw = PyDict::new(py);
            kw.set_item("index", df_overall.getattr("index")?)?;
            kw
        }),
    )?;
    if has_col(&df_overall, "Quantity")? {
        held_mask = df_overall
            .get_item("Quantity")?
            .call_method0("abs")?
            .rich_compare(1e-9, CompareOp::Gt)?;
    }
    let overall_cost_basis = if held_mask.call_method0("any")?.is_truthy()? {
        safe_sum(
            &df_overall.getattr("loc")?.get_item(&held_mask)?,
            &cost_basis_col,
        )?
    } else {
        o.fl(0.0)
    };
    let overall_unrealized = safe_sum(&df_overall, &unreal_gain_col)?;
    let overall_realized = safe_sum(&df_overall, &real_gain_col)?;
    let overall_dividends = safe_sum(&df_overall, &divs_col)?;
    let overall_commissions = safe_sum(&df_overall, &comm_col)?;
    let overall_taxes = if has_col(&df_overall, &taxes_col)? {
        safe_sum(&df_overall, &taxes_col)?
    } else {
        o.fl(0.0)
    };
    let overall_total_gain = safe_sum(&df_overall, &total_gain_col)?;
    let overall_total_taxes = safe_sum(&df_overall, &tax_col)?;
    let overall_cost_invested = safe_sum(&df_overall, &cost_invest_col)?;
    let overall_cumulative = safe_sum(&df_overall, &cum_invest_col)?;
    let overall_buy_cost = safe_sum(&df_overall, &total_buy_cost_col)?;
    let overall_day_change = safe_sum(&df_overall, &dc_col)?;
    let mut overall_prev_close = o.nan();
    let overall_income = safe_sum(&df_overall, &col("Est. Ann. Income"))?;

    let mut overall_cash = o.fl(0.0);
    if has_col(&df_overall, "Name")? {
        let cash_rows = df_overall.get_item(
            df_overall
                .get_item("Name")?
                .rich_compare("Cash", CompareOp::Eq)?,
        )?;
        overall_cash = safe_sum(&cash_rows, &mv_col)?;
    }

    if o.notna(&overall_mv)? && o.notna(&overall_day_change)? {
        overall_prev_close = overall_mv.sub(&overall_day_change)?;
    }
    let overall_day_change_pct = day_change_percent(&o, &overall_day_change, &overall_prev_close)?
        .unwrap_or_else(|| o.nan());
    let overall_return_pct = return_pct(&o, &overall_total_gain, &overall_buy_cost)?;

    if o.isna(&overall_mv)? || o.isna(&overall_total_gain)? {
        log(
            py,
            "warning",
            "Warning: Critical overall metrics calculated as NaN.",
        )?;
        has_warnings = true;
    }

    let overall_fx_gl_pct = if o.abs_gt(&overall_cost_basis, 1e-9)? {
        overall_fx_gl.div(&overall_cost_basis)?.mul(100)?
    } else {
        o.nan()
    };
    let overall_yield_pct = if o.abs_gt(&overall_mv, 1e-9)? {
        overall_income.div(&overall_mv)?.mul(100)?
    } else {
        o.fl(0.0)
    };

    // --- Overall MWR ---
    let mut overall_mwr = o.nan();
    let has_txs = !transactions_df.is_none() && !transactions_df.getattr("empty")?.is_truthy()?;
    if has_txs {
        let mut tx_for_mwr = transactions_df.clone();
        if !is_all_accounts_selected && has_col(transactions_df, "Account")? {
            let wanted = PyList::empty(py);
            for a in include_accounts.try_iter()? {
                wanted.append(o.str_strip(&a?)?.call_method0("upper")?)?;
            }
            let mut mask =
                upper(&o, transactions_df, "Account")?.call_method1("isin", (&wanted,))?;
            if has_col(transactions_df, "To Account")? {
                mask = mask.bitor(
                    upper(&o, transactions_df, "To Account")?.call_method1("isin", (&wanted,))?,
                )?;
            }
            tx_for_mwr = transactions_df.get_item(mask)?;
        }
        let scope = if include_accounts.is_truthy()? {
            include_accounts
        } else {
            all_available_accounts
        };
        let (dates, flows) = mwr_flows(
            &m,
            &tx_for_mwr,
            &overall_mv,
            report_date,
            display_currency,
            fx_rates,
            historical_fx_rates,
            scope,
        )?;
        if dates.is_truthy()? && flows.is_truthy()? {
            overall_mwr = m.getattr("calculate_irr")?.call1((&dates, &flows))?;
            if o.notna(&overall_mwr)? {
                overall_mwr = overall_mwr.mul(100.0)?;
            } else {
                log(
                    py,
                    "debug",
                    format!(
                        "MWR Calc Failed (Result NaN). Dates: {}, Flows: {}",
                        dates.len()?,
                        flows.len()?
                    ),
                )?;
            }
        } else {
            let count = if !tx_for_mwr.getattr("empty")?.is_truthy()? {
                tx_for_mwr.len()?
            } else {
                0
            };
            log(
                py,
                "debug",
                format!(
                    "MWR Prep Failed (No flows). Market Val: {}, TX Count: {count}",
                    o.fmt(&overall_mv)?
                ),
            )?;
        }
    }

    // --- Per-account MWR ---
    if has_txs && has_col(transactions_df, "Account")? {
        // The normalised account columns are the same for every account.
        let mut account_upper_col = None;
        let mut to_account_upper_col = None;
        let accounts: Vec<Bound<'py, PyAny>> = metrics.keys().iter().collect();
        for account in accounts {
            let r = (|| -> PyResult<()> {
                let account_upper = o.str_strip(&account)?.call_method0("upper")?;
                let acc_series = once(py, &mut account_upper_col, || {
                    upper(&o, transactions_df, "Account")
                })?;
                let mut to_series = o.pd.getattr("Series")?.call(
                    (false,),
                    Some(&{
                        let kw = PyDict::new(py);
                        kw.set_item("index", transactions_df.getattr("index")?)?;
                        kw
                    }),
                )?;
                if has_col(transactions_df, "To Account")? {
                    to_series = once(py, &mut to_account_upper_col, || {
                        upper(&o, transactions_df, "To Account")
                    })?
                    .rich_compare(&account_upper, CompareOp::Eq)?;
                }
                let account_tx = transactions_df.get_item(
                    acc_series
                        .rich_compare(&account_upper, CompareOp::Eq)?
                        .bitor(to_series)?,
                )?;
                if account_tx.getattr("empty")?.is_truthy()? {
                    return Ok(());
                }
                let me = entry(&metrics, &account)?;
                let account_mv = o.get_f(&me, "total_market_value_display", 0.0)?;
                let (dates, flows) = mwr_flows(
                    &m,
                    &account_tx,
                    &account_mv,
                    report_date,
                    display_currency,
                    fx_rates,
                    historical_fx_rates,
                    PyList::new(py, [&account])?.as_any(),
                )?;
                if dates.is_truthy()? && flows.is_truthy()? {
                    let account_mwr = m.getattr("calculate_irr")?.call1((&dates, &flows))?;
                    if o.notna(&account_mwr)? {
                        entry(&metrics, &account)?.set_item("mwr", account_mwr.mul(100.0)?)?;
                        entry(&metrics, &account)?
                            .set_item("portfolio_mwr", account_mwr.mul(100.0)?)?;
                    }
                }
                Ok(())
            })();
            match r {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "warning",
                    format!(
                        "Failed to calculate MWR for account '{}': {}",
                        o.fmt(&account)?,
                        es(py, &e)
                    ),
                )?,
                Err(e) => return Err(e),
            }
        }
    }

    let overall = PyDict::new(py);
    overall.set_item("market_value", &overall_mv)?;
    overall.set_item("cost_basis_held", overall_cost_basis)?;
    overall.set_item("unrealized_gain", overall_unrealized)?;
    overall.set_item("realized_gain", overall_realized)?;
    overall.set_item("dividends", overall_dividends)?;
    overall.set_item("commissions", overall_commissions)?;
    overall.set_item("taxes", overall_taxes)?;
    overall.set_item("total_gain", overall_total_gain)?;
    overall.set_item("total_cost_invested", overall_cost_invested)?;
    overall.set_item("total_buy_cost", overall_buy_cost)?;
    overall.set_item("portfolio_mwr", overall_mwr)?;
    overall.set_item("day_change_display", overall_day_change)?;
    overall.set_item("day_change_percent", overall_day_change_pct)?;
    overall.set_item("report_date", report_str()?)?;
    overall.set_item("display_currency", display_currency)?;
    overall.set_item("cumulative_investment", overall_cumulative)?;
    overall.set_item("total_return_pct", overall_return_pct)?;
    overall.set_item("fx_gain_loss_display", overall_fx_gl)?;
    overall.set_item("fx_gain_loss_pct", overall_fx_gl_pct)?;
    overall.set_item("est_annual_income_display", overall_income)?;
    overall.set_item("dividend_yield_pct", overall_yield_pct)?;
    overall.set_item("cash_balance", overall_cash)?;
    overall.set_item("total_taxes", overall_total_taxes)?;
    log(
        py,
        "debug",
        format!(
            "--- Finished Aggregating Metrics. Overall Market Value: {} ---",
            o.fmt(&overall_mv)?
        ),
    )?;
    Ok((overall, metrics.copy()?, has_errors, has_warnings))
}
