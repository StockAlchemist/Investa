//! `portfolio_history`'s helpers: price unadjustment, the per-date worker,
//! the manual-price and published-NAV lookups, the price grid, the L1-cache
//! pruning, the intraday symbol window and the id mappings.
//!
//! Ported line for line from `src/portfolio_history.py` (frozen in
//! `tests/reference/portfolio_history.py`). They are pandas, NumPy and dict
//! glue, so every operation is Python's own, in the original's order: an
//! in-place `x.loc[m] *= r` is `operator.imul` between the read and the
//! write, as the interpreter does it. Siblings and module state are looked up
//! on `portfolio_history` at call time.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{
    PyAttributeError, PyException, PyKeyError, PyOSError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PySet, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, log_exc, pystr, setitem, unpack};

pub(crate) fn ph(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("portfolio_history")
}

/// `_unadjust_prices`: each symbol's prices with its splits undone, so a
/// date before a split reads the price traded then.
#[pyfunction]
pub fn ph_unadjust_prices<'py>(
    py: Python<'py>,
    adjusted_prices_yf: &Bound<'py, PyAny>,
    yf_to_internal_map: &Bound<'py, PyAny>,
    splits_by_internal_symbol: &Bound<'py, PyAny>,
    processed_warnings: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let operator = py.import("operator")?;
    let date_t = m.getattr("date")?;
    let datetime_t = m.getattr("datetime")?;
    let out = PyDict::new(py);
    let mut unadjusted_count = 0usize;

    for item in adjusted_prices_yf.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (yf_symbol, adj) = (&kv[0], &kv[1]);
        let debug = yf_symbol.eq("AAPL")?;

        // The unadjusted Close; "price" first for cached, already-renamed frames.
        let mut col: Option<&str> = None;
        if !adj.getattr("empty")?.is_truthy()? {
            let columns = adj.getattr("columns")?;
            for c in ["price", "Close", "Adj Close"] {
                if columns.contains(c)? {
                    col = Some(c);
                    break;
                }
            }
        }
        let Some(col) = col else {
            if debug {
                log(
                    py,
                    "debug",
                    "    Skipping: Adjusted price DataFrame is empty or missing 'price'/'Adj Close'.",
                )?;
            }
            out.set_item(yf_symbol, adj.call_method0("copy")?)?;
            continue;
        };
        let working = adj.call_method0("copy")?;
        if col != "price" {
            setitem(&working, "price", working.get_item(col)?)?;
        }

        let internal = yf_to_internal_map.call_method1("get", (yf_symbol,))?;
        let splits = if internal.is_truthy()? {
            splits_by_internal_symbol.call_method1("get", (&internal,))?
        } else {
            py.None().into_bound(py)
        };
        if !splits.is_truthy()? {
            if debug {
                log(
                    py,
                    "debug",
                    format!(
                        "    No splits found for internal symbol '{}'. Copying adjusted prices.",
                        o.fmt(&internal)?
                    ),
                )?;
            }
            out.set_item(yf_symbol, working.call_method0("copy")?)?;
            continue;
        } else if debug {
            log(
                py,
                "debug",
                format!(
                    "    Found splits for '{}'. Proceeding with unadjustment.",
                    o.fmt(&internal)?
                ),
            )?;
        }

        let unadj = working.call_method0("copy")?;
        let utc = kwargs(
            py,
            &[(
                "utc",
                &pyo3::types::PyBool::new(py, true).to_owned().into_any(),
            )],
        )?;
        unadj.setattr(
            "index",
            crate::pyhelp::to_datetime(&pd, &unadj.getattr("index")?, Some(&utc))?,
        )?;
        if unadj.getattr("empty")?.is_truthy()? {
            out.set_item(yf_symbol, &unadj)?;
            continue;
        }
        unadj.call_method(
            "sort_index",
            (),
            Some(&kwargs(
                py,
                &[(
                    "inplace",
                    &pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                )],
            )?),
        )?;

        let factor = pd.getattr("Series")?.call(
            (1.0,),
            Some(&kwargs(
                py,
                &[("index", &unadj.getattr("index")?), ("dtype", &o.float)],
            )?),
        )?;
        // key=lambda x: x.get("Date", date.min)
        let date_min = date_t.getattr("min")?.unbind();
        let key =
            PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<Py<PyAny>> {
                let py = args.py();
                let x = args.get_item(0)?;
                Ok(x.call_method1("get", ("Date", date_min.bind(py)))?.unbind())
            })?;
        let sorted_desc = py.import("builtins")?.getattr("sorted")?.call(
            (&splits,),
            Some(&kwargs(
                py,
                &[
                    ("key", key.as_any()),
                    (
                        "reverse",
                        &pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                    ),
                ],
            )?),
        )?;
        if debug {
            log(
                py,
                "debug",
                format!("    Splits (newest first): {}", o.fmt(&sorted_desc)?),
            )?;
        }

        for split_info in sorted_desc.try_iter()? {
            let split_info = split_info?;
            let r = (|| -> PyResult<()> {
                let raw = split_info.call_method1("get", ("Date",))?;
                let ratio = o.float(&split_info.get_item("Split Ratio")?)?;
                if raw.is_none() {
                    return Err(PyValueError::new_err("Split info missing 'Date'"));
                }
                let split_date = if raw.is_instance(&datetime_t)? {
                    raw.call_method0("date")?
                } else if raw.is_instance(&date_t)? {
                    raw.clone()
                } else {
                    return Err(PyTypeError::new_err(format!(
                        "Invalid split date type: {}",
                        o.fmt(&raw.get_type().into_any())?
                    )));
                };
                if ratio <= 0.0 {
                    if debug {
                        log(
                            py,
                            "warning",
                            format!(
                                "    Invalid split ratio {} on {}. Skipping.",
                                o.fmt(&o.fl(ratio))?,
                                o.fmt(&split_date)?
                            ),
                        )?;
                    }
                    return Ok(());
                }
                // Every date strictly before the split takes the factor.
                let mask = pd
                    .getattr("Series")?
                    .call1((factor.getattr("index")?,))?
                    .getattr("dt")?
                    .getattr("date")?
                    .rich_compare(&split_date, CompareOp::Lt)?
                    .getattr("values")?;
                if debug && split_date.eq(date_t.call1((2020, 8, 31))?)? {
                    log(
                        py,
                        "debug",
                        format!(
                            "    Applying ratio {} for split on {}. Mask sum (dates before): {}",
                            o.fmt(&o.fl(ratio))?,
                            o.fmt(&split_date)?,
                            o.fmt(&mask.call_method0("sum")?)?
                        ),
                    )?;
                }
                let loc = factor.getattr("loc")?;
                let v = operator.call_method1("imul", (loc.get_item(&mask)?, ratio))?;
                loc.set_item(&mask, v)?;
                Ok(())
            })();
            match r {
                Ok(()) => {}
                Err(e)
                    if e.is_instance_of::<PyKeyError>(py)
                        || e.is_instance_of::<PyValueError>(py)
                        || e.is_instance_of::<PyTypeError>(py)
                        || e.is_instance_of::<PyAttributeError>(py) =>
                {
                    let around = || -> PyResult<String> {
                        o.fmt(&split_info.call_method1("get", ("Date", "N/A"))?)
                    };
                    let warn_key =
                        format!("unadjust_split_proc_{}_{}", o.fmt(yf_symbol)?, around()?);
                    if !processed_warnings.contains(&warn_key)? {
                        log(
                            py,
                            "warning",
                            format!(
                                "Hist WARN: Error processing split for {} around {}: {}",
                                o.fmt(yf_symbol)?,
                                around()?,
                                es(py, &e)
                            ),
                        )?;
                        processed_warnings.call_method1("add", (&warn_key,))?;
                    }
                    if debug {
                        log(
                            py,
                            "warning",
                            format!(
                                "    Error processing split around {}: {}",
                                around()?,
                                es(py, &e)
                            ),
                        )?;
                    }
                }
                Err(e) => return Err(e),
            }
        }

        let original = unadj.get_item("price")?.call_method0("copy")?;
        let aligned = factor.call_method(
            "align",
            (unadj.get_item("price")?,),
            Some(&kwargs(
                py,
                &[("join", &pystr(py, "right")), ("fill_value", &o.fl(1.0))],
            )?),
        )?;
        let aligned = unpack(&aligned, 2)?;
        setitem(&unadj, "unadjusted_price", aligned[1].mul(&aligned[0])?)?;

        let new_col = unadj.get_item("unadjusted_price")?;
        if !new_col
            .call_method1(
                "equals",
                (
                    original
                        .call_method1("reindex_like", (unadj.get_item("unadjusted_price")?,))?,
                ),
            )?
            .is_truthy()?
        {
            unadjusted_count += 1;
        }
        let rename = PyDict::new(py);
        rename.set_item("unadjusted_price", "price")?;
        let result = unadj
            .get_item(PyList::new(py, ["unadjusted_price"])?)?
            .call_method(
                "rename",
                (),
                Some(&kwargs(py, &[("columns", rename.as_any())])?),
            )?;
        out.set_item(yf_symbol, result)?;
    }

    log(
        py,
        "info",
        format!(
            "--- Finished Price Unadjustment ({unadjusted_count} symbols processed with splits) ---"
        ),
    )?;
    Ok(out)
}

/// `_calculate_daily_metrics_worker`: one date's value and net flow (the
/// multiprocessing path), or a row of failure flags.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_daily_metrics_worker<'py>(
    py: Python<'py>,
    eval_date: &Bound<'py, PyAny>,
    transactions_df: &Bound<'py, PyAny>,
    historical_prices_yf_unadjusted: &Bound<'py, PyAny>,
    historical_prices_yf_adjusted: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    target_currency: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    manual_overrides_dict: &Bound<'py, PyAny>,
    symbol_to_id: &Bound<'py, PyAny>,
    id_to_symbol: &Bound<'py, PyAny>,
    account_to_id: &Bound<'py, PyAny>,
    id_to_account: &Bound<'py, PyAny>,
    type_to_id: &Bound<'py, PyAny>,
    currency_to_id: &Bound<'py, PyAny>,
    id_to_currency: &Bound<'py, PyAny>,
    calc_method: &Bound<'py, PyAny>,
    included_accounts: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = historical_prices_yf_adjusted;
    let m = ph(py)?;
    let o = Ops::new(py)?;
    py.import("logging")?.call_method(
        "basicConfig",
        (),
        Some(&kwargs(
            py,
            &[
                ("level", &m.getattr("LOGGING_LEVEL")?),
                (
                    "format",
                    &pystr(
                        py,
                        "%(asctime)s [%(levelname)-8s] PID:%(process)d {%(module)s:%(lineno)d} %(message)s",
                    ),
                ),
                ("datefmt", &pystr(py, "%Y-%m-%d %H:%M:%S")),
                ("force", &pyo3::types::PyBool::new(py, true).to_owned().into_any()),
            ],
        )?),
    )?;
    let fd = o.fmt(eval_date)?;
    let body = (|| -> PyResult<Bound<'py, PyDict>> {
        let time = m.getattr("time")?;
        let start_time = time.call_method0("time")?;
        let dummy_warnings = PySet::empty(py)?;
        let args = PyTuple::new(
            py,
            [
                eval_date,
                transactions_df,
                historical_prices_yf_unadjusted,
                historical_fx_yf,
                target_currency,
                internal_to_yf_map,
                account_currency_map,
                default_currency,
                manual_overrides_dict,
                dummy_warnings.as_any(),
                symbol_to_id,
                id_to_symbol,
                account_to_id,
                id_to_account,
                type_to_id,
                currency_to_id,
                id_to_currency,
            ],
        )?;
        let value = m
            .getattr("_calculate_portfolio_value_at_date_unadjusted")?
            .call(
                args,
                Some(&kwargs(
                    py,
                    &[
                        ("method", calc_method),
                        ("included_accounts", included_accounts),
                        ("account_cash_mode_map", account_cash_mode_map),
                    ],
                )?),
            )?;
        let value = unpack(&value, 2)?;
        let (value_main, failed_main) = (value[0].clone(), value[1].clone());
        let end_time_main = time.call_method0("time")?;

        let compare = m.getattr("HISTORICAL_COMPARE_METHODS")?.is_truthy()?;
        if compare && calc_method.eq("numba")? {
            let r = (|| -> PyResult<()> {
                let start_py = time.call_method0("time")?;
                let py_value = m
                    .getattr("_calculate_portfolio_value_at_date_unadjusted_python")?
                    .call1((
                        eval_date,
                        transactions_df,
                        historical_prices_yf_unadjusted,
                        historical_fx_yf,
                        target_currency,
                        internal_to_yf_map,
                        account_currency_map,
                        default_currency,
                        manual_overrides_dict,
                        &dummy_warnings,
                    ))?;
                let py_value = unpack(&py_value, 2)?.remove(0);
                let end_py = time.call_method0("time")?;
                let diff = if o.notna(&value_main)? && o.notna(&py_value)? {
                    o.abs(&value_main.sub(&py_value)?)?
                } else {
                    o.nan()
                };
                let line = || -> PyResult<String> {
                    Ok(format!(
                        "COMPARE {fd}: Numba={} ({}s), Python={} ({}s), Diff={}",
                        o.fmt_spec(&value_main, ".4f")?,
                        o.fmt_spec(&end_time_main.sub(&start_time)?, ".4f")?,
                        o.fmt_spec(&py_value, ".4f")?,
                        o.fmt_spec(&end_py.sub(&start_py)?, ".4f")?,
                        o.fmt_spec(&diff, ".4f")?
                    ))
                };
                if diff.gt(1e-3)? {
                    log(py, "warning", line()?)?;
                } else if eval_date.getattr("day")?.eq(1)? {
                    log(py, "info", line()?)?;
                }
                Ok(())
            })();
            match r {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "error",
                    format!(
                        "Error during comparison calculation for {fd}: {}",
                        es(py, &e)
                    ),
                )?,
                Err(e) => return Err(e),
            }
        }

        if failed_main.is_truthy()? {
            log(py, "debug", format!("Valuation failed for date: {fd}"))?;
        }
        let (net_flow, flow_failed) = if o.isna(&value_main)? {
            (
                o.nan(),
                pyo3::types::PyBool::new(py, true).to_owned().into_any(),
            )
        } else {
            let r = m.getattr("_calculate_daily_net_cash_flow")?.call(
                (
                    eval_date,
                    transactions_df,
                    target_currency,
                    historical_fx_yf,
                    account_currency_map,
                    default_currency,
                    &dummy_warnings,
                ),
                Some(&kwargs(
                    py,
                    &[
                        ("included_accounts", included_accounts),
                        (
                            "historical_prices_yf_unadjusted",
                            historical_prices_yf_unadjusted,
                        ),
                        ("internal_to_yf_map", internal_to_yf_map),
                    ],
                )?),
            )?;
            let r = unpack(&r, 2)?;
            (r[0].clone(), r[1].clone())
        };
        let row = PyDict::new(py);
        row.set_item("Date", eval_date)?;
        row.set_item("value", &value_main)?;
        row.set_item("net_flow", net_flow)?;
        row.set_item("value_lookup_failed", &failed_main)?;
        row.set_item("flow_lookup_failed", flow_failed)?;
        row.set_item("bench_lookup_failed", false)?;
        Ok(row)
    })();
    match body {
        Ok(row) => Ok(row),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "critical",
                format!(
                    "!!! CRITICAL ERROR in worker process for date {fd}: {}",
                    es(py, &e)
                ),
            )?;
            log_exc(py, "error", "Worker Traceback:", &e)?;
            let row = PyDict::new(py);
            row.set_item("Date", eval_date)?;
            row.set_item("value", f64::NAN)?;
            row.set_item("net_flow", f64::NAN)?;
            row.set_item("value_lookup_failed", true)?;
            row.set_item("flow_lookup_failed", true)?;
            row.set_item("bench_lookup_failed", true)?;
            row.set_item("worker_error", true)?;
            row.set_item("worker_error_msg", es(py, &e))?;
            Ok(row)
        }
        Err(e) => Err(e),
    }
}

/// `manual_prices_by_symbol_id`: the user's manual prices as {symbol id: price},
/// matched case-insensitively.
#[pyfunction]
pub fn ph_manual_prices_by_symbol_id<'py>(
    py: Python<'py>,
    manual_overrides_dict: &Bound<'py, PyAny>,
    id_to_symbol: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = Ops::new(py)?;
    let out = PyDict::new(py);
    if !manual_overrides_dict.is_truthy()? {
        return Ok(out);
    }
    let isfinite = py.import("numpy")?.getattr("isfinite")?;
    let norm = |v: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        o.str_t
            .call1((v,))?
            .call_method0("upper")?
            .call_method0("strip")
    };
    let by_name = PyDict::new(py);
    for item in manual_overrides_dict.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let raw = if kv[1].is_instance_of::<PyDict>() {
            kv[1].call_method1("get", ("price",))?
        } else {
            kv[1].clone()
        };
        let price = match o.float(&raw) {
            Ok(v) => v,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        if isfinite.call1((price,))?.is_truthy()? && price > 1e-9 {
            by_name.set_item(norm(&kv[0])?, price)?;
        }
    }
    if !by_name.is_truthy()? {
        return Ok(out);
    }
    for item in id_to_symbol.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        if by_name.contains(norm(&kv[1])?)? {
            out.set_item(&kv[0], by_name.as_any().get_item(norm(&kv[1])?)?)?;
        }
    }
    Ok(out)
}

/// `_fund_nav_series`: the published NAV series for any of `symbol_ids` the
/// fund store covers; a missing store answers nothing.
#[pyfunction]
pub fn ph_fund_nav_series<'py>(
    py: Python<'py>,
    symbol_ids: &Bound<'py, PyAny>,
    date_range: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let out = PyDict::new(py);
    if !symbol_ids.is_truthy()? {
        return Ok(out);
    }
    let opened = (|| -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let module = py.import("market_db")?;
        let class = module.getattr("MarketDatabase").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(format!(
                "cannot import name 'MarketDatabase' from 'market_db' ({})",
                module
                    .getattr("__file__")
                    .and_then(|f| o.fmt(&f))
                    .unwrap_or_else(|_| "unknown location".into())
            ))
        })?;
        let db = class.call0()?;
        let coverage = db.call_method0("get_fund_nav_coverage")?;
        Ok((db, coverage))
    })();
    let (db, coverage) = match opened {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("Fund NAV store unavailable: {}", es(py, &e)),
            )?;
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    if !coverage.is_truthy()? {
        return Ok(out);
    }
    let norm = |v: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        o.str_t
            .call1((v,))?
            .call_method0("upper")?
            .call_method0("strip")
    };
    let have = PyDict::new(py);
    for code in coverage.try_iter()? {
        let code = code?;
        have.set_item(norm(&code)?, &code)?;
    }
    let start = date_range.call_method0("min")?.call_method0("date")?;
    let end = date_range.call_method0("max")?.call_method0("date")?;

    for item in symbol_ids.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (sym_id, symbol) = (&kv[0], &kv[1]);
        let stored = have.call_method1("get", (norm(symbol)?,))?;
        if !stored.is_truthy()? {
            continue;
        }
        let frame = match db.call_method1("get_fund_nav", (&stored, &start, &end)) {
            Ok(f) => f,
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "debug",
                    format!(
                        "Fund NAV read failed for {}: {}",
                        o.fmt(symbol)?,
                        es(py, &e)
                    ),
                )?;
                continue;
            }
            Err(e) => return Err(e),
        };
        if frame.is_none()
            || frame.getattr("empty")?.is_truthy()?
            || !frame.getattr("columns")?.contains("price")?
        {
            continue;
        }
        let mut series = frame.get_item("price")?.call_method0("dropna")?;
        if series.getattr("empty")?.is_truthy()? {
            continue;
        }
        if !series
            .getattr("index")?
            .is_instance(&pd.getattr("DatetimeIndex")?)?
            || series.getattr("index")?.getattr("tz")?.is_none()
        {
            series = series.call_method0("copy")?;
            let utc = kwargs(
                py,
                &[(
                    "utc",
                    &pyo3::types::PyBool::new(py, true).to_owned().into_any(),
                )],
            )?;
            series.setattr(
                "index",
                crate::pyhelp::to_datetime(&pd, &series.getattr("index")?, Some(&utc))?,
            )?;
        }
        out.set_item(sym_id, &series)?;
        let index = series.getattr("index")?;
        log(
            py,
            "info",
            format!(
                "Fund NAV: {} valued from {} published NAVs ({} to {})",
                o.fmt(symbol)?,
                series.len()?,
                o.fmt(&index.call_method0("min")?.call_method0("date")?)?,
                o.fmt(&index.call_method0("max")?.call_method0("date")?)?
            ),
        )?;
    }
    Ok(out)
}

/// `align_prices_to_grid` without building the union frame:
///
/// ```python
/// raw_prices = pd.DataFrame(price_series_by_id).sort_index()
/// fill_index = raw_prices.index.union(date_range)
/// return raw_prices.reindex(fill_index).ffill().bfill().reindex(date_range)
/// ```
///
/// With float64 series and a grid all on unique, increasing, NaT-free UTC
/// stamps (in any unit; compared as nanoseconds), the union is ordered by time and a symbol's column is
/// empty wherever it has no stamp, so each grid point takes the symbol's last
/// valid (non-NaN) price at or before it; points before its first valid price
/// take that price (the backfill), and a symbol with none stays empty. The
/// values are copied, never computed. The frame is the grid by the dict's keys,
/// one column-major block as the reindex leaves it. Any other input returns
/// `None` for the pandas path.
fn aligned_natively<'py>(
    pd: &Bound<'py, PyModule>,
    price_series_by_id: &Bound<'py, PyAny>,
    date_range: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = date_range.py();
    let np = py.import("numpy")?;
    // Any unit (pandas 3 picks it from the input): the sweep compares the
    // stamps as nanosecond integers, so the units need not agree.
    let utc_ns = |index: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(index.get_type().is(&pd.getattr("DatetimeIndex")?)
            && index.getattr("tz")?.str()?.to_str()? == "UTC"
            && index.getattr("is_unique")?.is_truthy()?
            && index.getattr("is_monotonic_increasing")?.is_truthy()?
            && !index.getattr("hasnans")?.is_truthy()?)
    };
    // A stamp past nanosecond range cannot be compared that way: leave it to pandas.
    let ns = |index: &Bound<'py, PyAny>| -> Option<Bound<'py, PyAny>> {
        crate::pyhelp::ns_ints(index).ok()
    };
    let Ok(series_by_id) = price_series_by_id.cast::<PyDict>() else {
        return Ok(None);
    };
    if series_by_id.is_empty() || !utc_ns(date_range)? {
        return Ok(None);
    }
    let series_type = pd.getattr("Series")?;
    let float64 = np.call_method1("dtype", ("float64",))?;
    let mut columns = Vec::with_capacity(series_by_id.len());
    for (key, series) in series_by_id.iter() {
        if !series.get_type().is(&series_type)
            || !series.getattr("dtype")?.eq(&float64)?
            || !utc_ns(&series.getattr("index")?)?
        {
            return Ok(None);
        }
        let Some(stamps) = ns(&series.getattr("index")?) else {
            return Ok(None);
        };
        let stamps: numpy::PyReadonlyArray1<i64> = stamps.extract()?;
        let values: numpy::PyReadonlyArray1<f64> = series.call_method0("to_numpy")?.extract()?;
        columns.push((key, stamps, values));
    }
    let Some(grid) = ns(date_range) else {
        return Ok(None);
    };
    let grid: numpy::PyReadonlyArray1<i64> = grid.extract()?;
    let grid = grid.as_array();
    let mut block = Vec::with_capacity(columns.len() * grid.len());
    for (_, stamps, values) in &columns {
        let (stamps, values) = (stamps.as_array(), values.as_array());
        let first_valid = values
            .iter()
            .copied()
            .find(|v| !v.is_nan())
            .unwrap_or(f64::NAN);
        let (mut j, mut last) = (0usize, f64::NAN);
        for &point in grid.iter() {
            while j < stamps.len() && stamps[j] <= point {
                if !values[j].is_nan() {
                    last = values[j];
                }
                j += 1;
            }
            block.push(if last.is_nan() { first_valid } else { last });
        }
    }
    let block = numpy::ndarray::Array2::from_shape_vec((columns.len(), grid.len()), block)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    // The columns as `DataFrame(dict)` labels them: pandas 3 first turns keys
    // that step evenly into a range (a RangeIndex); pandas 2 has no such step.
    let mut keys = PyList::new(py, columns.iter().map(|(k, _, _)| k))?.into_any();
    if let Ok(to_range) = py
        .import("pandas.core.indexes.base")
        .and_then(|m| m.getattr("maybe_sequence_to_range"))
    {
        keys = to_range.call1((keys,))?;
    }
    let kw = PyDict::new(py);
    kw.set_item("index", date_range)?;
    kw.set_item("columns", pd.getattr("Index")?.call1((keys,))?)?;
    let frame = pd.getattr("DataFrame")?.call(
        (numpy::IntoPyArray::into_pyarray(block, py).getattr("T")?,),
        Some(&kw),
    )?;
    Ok(Some(frame))
}

/// `align_prices_to_grid`: each symbol's last observed price carried onto
/// the grid, never interpolated across a gap.
#[pyfunction]
pub fn ph_align_prices_to_grid<'py>(
    py: Python<'py>,
    price_series_by_id: &Bound<'py, PyAny>,
    date_range: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    if let Some(aligned) = aligned_natively(&pd, price_series_by_id, date_range)? {
        return Ok(aligned);
    }
    let raw = pd
        .getattr("DataFrame")?
        .call1((price_series_by_id,))?
        .call_method0("sort_index")?;
    let fill_index = raw.getattr("index")?.call_method1("union", (date_range,))?;
    raw.call_method1("reindex", (fill_index,))?
        .call_method0("ffill")?
        .call_method0("bfill")?
        .call_method1("reindex", (date_range,))
}

/// `_prune_holdings_cache`: deletes L1 holdings-cache files older than
/// `max_age_days`.
#[pyfunction]
pub fn ph_prune_holdings_cache<'py>(
    py: Python<'py>,
    holdings_cache_dir: &Bound<'py, PyAny>,
    max_age_days: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let m = ph(py)?;
    let r = (|| -> PyResult<()> {
        let os = m.getattr("os")?;
        let cutoff = m
            .getattr("time")?
            .call_method0("time")?
            .sub(max_age_days.mul(86400)?)?;
        let mut removed = 0usize;
        let entries = os.call_method1("scandir", (holdings_cache_dir,))?;
        with1(py, &entries, |entries| {
            for entry in entries.try_iter()? {
                let entry = entry?;
                let r = (|| -> PyResult<bool> {
                    if entry.call_method0("is_file")?.is_truthy()?
                        && entry
                            .call_method0("stat")?
                            .getattr("st_mtime")?
                            .lt(&cutoff)?
                    {
                        os.call_method1("remove", (entry.getattr("path")?,))?;
                        return Ok(true);
                    }
                    Ok(false)
                })();
                match r {
                    Ok(true) => removed += 1,
                    Ok(false) => {}
                    Err(e) if e.is_instance_of::<PyOSError>(py) => continue,
                    Err(e) => return Err(e),
                }
            }
            Ok(())
        })?;
        if removed > 0 {
            log(
                py,
                "info",
                format!("L1 Cache PRUNE: removed {removed} stale cache files."),
            )?;
        }
        Ok(())
    })();
    match r {
        Ok(()) => Ok(()),
        Err(e) if e.is_instance_of::<PyException>(py) => log(
            py,
            "warning",
            format!("L1 Cache PRUNE Error: {}", es(py, &e)),
        ),
        Err(e) => Err(e),
    }
}

/// `_intraday_symbols_for_window`: the symbols held in the window, the
/// benchmarks, and anything traded in it (a same-day round trip is zero in
/// every daily snapshot but still moves that day's intraday path).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_intraday_symbols_for_window<'py>(
    py: Python<'py>,
    symbols_yf: &Bound<'py, PyAny>,
    benchmark_symbols_yf: &Bound<'py, PyAny>,
    holdings_qty: &Bound<'py, PyAny>,
    full_start_date: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    id_to_symbol: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
    transactions: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let np = py.import("numpy")?;
    let pd = py.import("pandas")?;
    let b = py.import("builtins")?;
    let all = || {
        b.getattr("list")?
            .call1((symbols_yf,))?
            .cast_into::<PyList>()
            .map_err(Into::into)
    };
    if holdings_qty.is_none() || holdings_qty.getattr("shape")?.get_item(0)?.eq(0)? {
        return all();
    }
    let offset = b
        .getattr("max")?
        .call1((0, start_date.sub(full_start_date)?.getattr("days")?))?;
    // holdings_qty[start_offset:]
    let active = holdings_qty.get_item(b.getattr("slice")?.call1((&offset, py.None()))?)?;
    if active.getattr("shape")?.get_item(0)?.eq(0)? {
        return all();
    }
    let any_kw = kwargs(py, &[("axis", PyTuple::new(py, [0, 2])?.as_any())])?;
    let held_sids = np
        .call_method1(
            "where",
            (np.call_method(
                "any",
                (np.call_method1("abs", (&active,))?
                    .rich_compare(1e-6, CompareOp::Gt)?,),
                Some(&any_kw),
            )?,),
        )?
        .get_item(0)?;
    let needed = b.getattr("set")?.call1((benchmark_symbols_yf,))?;
    let internal = PySet::empty(py)?;
    for sid in held_sids.try_iter()? {
        internal.add(id_to_symbol.get_item(b.getattr("int")?.call1((sid?,))?)?)?;
    }
    if !transactions.getattr("empty")?.is_truthy()? {
        let days = pd
            .call_method1("to_datetime", (transactions.get_item("Date")?,))?
            .getattr("dt")?
            .getattr("date")?;
        let in_window = days
            .rich_compare(start_date, CompareOp::Ge)?
            .bitand(days.rich_compare(end_date, CompareOp::Le)?)?;
        let traded = transactions
            .getattr("loc")?
            .get_item(PyTuple::new(py, [in_window, pystr(py, "Symbol")])?)?;
        internal.call_method1("update", (traded.call_method0("dropna")?,))?;
    }
    for s in internal.iter() {
        if internal_to_yf_map.contains(&s)? {
            needed.call_method1("add", (internal_to_yf_map.get_item(&s)?,))?;
        }
    }
    let out = PyList::empty(py);
    for s in symbols_yf.try_iter()? {
        let s = s?;
        if needed.contains(&s)? {
            out.append(s)?;
        }
    }
    Ok(out)
}

/// `generate_mappings`: the string-to-id maps the valuation kernels index by.
#[pyfunction]
pub fn ph_generate_mappings<'py>(
    py: Python<'py>,
    transactions_df_effective: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let m = ph(py)?;
    let b = py.import("builtins")?;
    let normalize = m.getattr("_normalize_series")?;
    let df = transactions_df_effective;

    let symbol_to_id = PyDict::new(py);
    for (i, s) in normalize
        .call1((df.get_item("Symbol")?,))?
        .call_method0("unique")?
        .try_iter()?
        .enumerate()
    {
        symbol_to_id.set_item(s?, i)?;
    }
    let id_to_symbol = PyDict::new(py);
    for (s, i) in symbol_to_id.iter() {
        id_to_symbol.set_item(i, s)?;
    }

    let set = b.getattr("set")?;
    let source = set.call1((normalize
        .call1((df.get_item("Account")?,))?
        .call_method0("unique")?,))?;
    let mut dest = set.call0()?;
    if df.getattr("columns")?.contains("To Account")? {
        dest = set.call1((normalize
            .call1((df.get_item("To Account")?,))?
            .call_method0("dropna")?
            .call_method0("unique")?,))?;
    }
    let all_accounts = b
        .getattr("sorted")?
        .call1((b.getattr("list")?.call1((source.bitor(&dest)?,))?,))?;
    let account_to_id = PyDict::new(py);
    for (i, a) in all_accounts.try_iter()?.enumerate() {
        account_to_id.set_item(a?, i)?;
    }
    let id_to_account = PyDict::new(py);
    for (a, i) in account_to_id.iter() {
        id_to_account.set_item(i, a)?;
    }

    let type_to_id = PyDict::new(py);
    for (i, t) in df
        .get_item("Type")?
        .call_method0("unique")?
        .try_iter()?
        .enumerate()
    {
        type_to_id.set_item(t?.call_method0("lower")?.call_method0("strip")?, i)?;
    }
    let currency_to_id = PyDict::new(py);
    for (i, c) in df
        .get_item("Local Currency")?
        .call_method0("unique")?
        .try_iter()?
        .enumerate()
    {
        currency_to_id.set_item(c?, i)?;
    }
    let id_to_currency = PyDict::new(py);
    for (c, i) in currency_to_id.iter() {
        id_to_currency.set_item(i, c)?;
    }

    let cash = m.getattr("CASH_SYMBOL_CSV")?;
    if !symbol_to_id.contains(&cash)? {
        let cash_id = symbol_to_id.len();
        symbol_to_id.set_item(&cash, cash_id)?;
        id_to_symbol.set_item(cash_id, &cash)?;
    }
    PyTuple::new(
        py,
        [
            symbol_to_id.into_any(),
            id_to_symbol.into_any(),
            account_to_id.into_any(),
            id_to_account.into_any(),
            type_to_id.into_any(),
            currency_to_id.into_any(),
            id_to_currency.into_any(),
        ],
    )
}
