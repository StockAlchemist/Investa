//! `portfolio_history`'s daily results and their accumulation: the daily
//! value/flow frame from its feather cache, the chronological pass or the
//! per-date worker pool, the TWR return with its spike and transfer healing
//! (`_load_or_calculate_daily_results`), and the accumulated gains,
//! resampling and range normalisation (`_calculate_accumulated_gains_and_resample`).
//!
//! Ported line for line from `src/portfolio_history.py` (frozen in
//! `tests/reference/portfolio_history.py`). The kernels (`portfolio_twr`,
//! `benchmark_accumulated`, `money_weighted`, the replay and the valuation)
//! were already Rust; this is the pandas, file and multiprocessing glue, so
//! every operation is Python's own, in the original's order. Siblings and
//! module state are looked up on `portfolio_history` at call time.
//!
//! The two `traceback.print_exc()` calls print the caught exception without
//! a frame for this function, which is no longer Python.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyNotImplementedError, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PySet, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::ph_small::ph;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, log_exc, pystr, setitem, unpack};

fn tru(py: Python<'_>) -> Bound<'_, PyAny> {
    PyBool::new(py, true).to_owned().into_any()
}

fn fals(py: Python<'_>) -> Bound<'_, PyAny> {
    PyBool::new(py, false).to_owned().into_any()
}

/// `traceback.print_exc()` inside the original's `except`.
fn print_exc(py: Python<'_>, e: &PyErr) -> PyResult<()> {
    py.import("traceback")?
        .call_method1("print_exception", (e.value(py),))?;
    Ok(())
}

/// `slice(a, b)`, for `.loc[a:b]` on labels.
fn pyslice<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr("slice")?.call1((a, b))
}

fn debug_enabled(py: Python<'_>) -> PyResult<bool> {
    let logging = py.import("logging")?;
    logging
        .call_method0("getLogger")?
        .call_method1("isEnabledFor", (logging.getattr("DEBUG")?,))?
        .is_truthy()
}

/// `df.rename(columns={old: new}, inplace=True)`
fn rename_inplace<'py>(df: &Bound<'py, PyAny>, old: &str, new: &str) -> PyResult<()> {
    let py = df.py();
    let map = PyDict::new(py);
    map.set_item(old, new)?;
    df.call_method(
        "rename",
        (),
        Some(&kwargs(
            py,
            &[("columns", map.as_any()), ("inplace", &tru(py))],
        )?),
    )?;
    Ok(())
}

/// The daily accumulated-gain columns renamed to their final names.
fn rename_accumulated<'py>(df: &Bound<'py, PyAny>, benchmarks: &Bound<'py, PyAny>) -> PyResult<()> {
    rename_inplace(
        df,
        "Portfolio Accumulated Gain Daily",
        "Portfolio Accumulated Gain",
    )?;
    for bm in benchmarks.try_iter()? {
        let bm = Ops::new(df.py())?.fmt(&bm?)?;
        let daily = format!("{bm} Accumulated Gain Daily");
        if df.getattr("columns")?.contains(&daily)? {
            rename_inplace(df, &daily, &format!("{bm} Accumulated Gain"))?;
        }
    }
    Ok(())
}

/// `_calculate_accumulated_gains_and_resample`: (the output frame, the TWR
/// factor over the requested range, a status note).
#[pyfunction]
pub fn ph_accumulated_gains_and_resample<'py>(
    py: Python<'py>,
    daily_df: &Bound<'py, PyAny>,
    benchmark_symbols_yf: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    start_date_filter: &Bound<'py, PyAny>,
    end_date_filter: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, String)> {
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let core = py.import("investa_core")?;
    let empty = || pd.getattr("DataFrame")?.call0();
    if daily_df.getattr("empty")?.is_truthy()? {
        return Ok((empty()?, o.nan(), " (No daily data for final calcs)".into()));
    }
    if !daily_df.getattr("columns")?.contains("daily_return")? {
        return Ok((empty()?, o.nan(), " (Daily return column missing)".into()));
    }
    let mut status = String::new();
    let mut twr = o.nan();
    let results = daily_df.call_method0("copy")?;
    log(
        py,
        "debug",
        format!(
            "[_calculate_accumulated_gains_and_resample] Input daily_df shape: {}",
            o.fmt(&results.getattr("shape")?)?
        ),
    )?;
    if debug_enabled(py)? {
        log(
            py,
            "debug",
            format!(
                "Input daily_df tail:\n{}",
                o.fmt(&results.call_method0("tail")?.call_method0("to_string")?)?
            ),
        )?;
    }
    let f64_kw = kwargs(py, &[("dtype", &np.getattr("float64")?)])?;
    let as_f64 = |s: Bound<'py, PyAny>| np.call_method("asarray", (s,), Some(&f64_kw));
    let has =
        |df: &Bound<'py, PyAny>, c: &str| -> PyResult<bool> { df.getattr("columns")?.contains(c) };

    let body = (|| -> PyResult<Bound<'py, PyAny>> {
        // Gain factors, their running product and the drawdown, in one scan.
        let twr_parts = core.call_method1(
            "portfolio_twr",
            (as_f64(results.get_item("daily_return")?)?,),
        )?;
        let twr_parts = unpack(&twr_parts, 3)?;
        let gain_factors = pd.getattr("Series")?.call(
            (&twr_parts[0],),
            Some(&kwargs(
                py,
                &[
                    ("index", &results.getattr("index")?),
                    ("name", &pystr(py, "daily_return")),
                ],
            )?),
        )?;
        setitem(&results, "Portfolio Accumulated Gain Daily", &twr_parts[1])?;

        // The TWR factor over the requested range only.
        if !results.getattr("empty")?.is_truthy()? {
            let tz = results.getattr("index")?.getattr("tz")?;
            let timestamp = pd.getattr("Timestamp")?;
            let ts_start = timestamp
                .call1((start_date_filter,))?
                .call_method1("tz_localize", (&tz,))?;
            let ts_end = timestamp
                .call1((end_date_filter,))?
                .call_method1("tz_localize", (&tz,))?
                .add(pd.getattr("Timedelta")?.call(
                    (),
                    Some(&kwargs(
                        py,
                        &[
                            ("hours", &23i64.into_pyobject(py)?.into_any()),
                            ("minutes", &59i64.into_pyobject(py)?.into_any()),
                        ],
                    )?),
                )?)?;
            let period = gain_factors
                .getattr("loc")?
                .get_item(pyslice(py, &ts_start, &ts_end)?)?;
            twr = if !period.getattr("empty")?.is_truthy()? {
                period.call_method0("prod")?
            } else if !results.getattr("empty")?.is_truthy()? {
                results
                    .get_item("Portfolio Accumulated Gain Daily")?
                    .call_method0("dropna")?
                    .getattr("iloc")?
                    .get_item(-1)?
            } else {
                o.nan()
            };
        }

        for bm in benchmark_symbols_yf.try_iter()? {
            let bm = o.fmt(&bm?)?;
            let price_col = format!("{bm} Price");
            let accum_col = format!("{bm} Accumulated Gain Daily");
            if has(&results, &price_col)? {
                let bench = results.get_item(&price_col)?;
                if !bench
                    .call_method0("dropna")?
                    .getattr("index")?
                    .getattr("is_unique")?
                    .is_truthy()?
                {
                    // The original reindexes by date, which refuses duplicate dates.
                    bench
                        .call_method0("dropna")?
                        .call_method(
                            "pct_change",
                            (),
                            Some(&kwargs(py, &[("fill_method", &py.None().into_bound(py))])?),
                        )?
                        .call_method1("reindex", (results.getattr("index")?,))?;
                }
                let accum = core.call_method1("benchmark_accumulated", (as_f64(bench)?,))?;
                if accum.is_none() {
                    setitem(&results, &accum_col, f64::NAN)?;
                } else {
                    setitem(&results, &accum_col, accum)?;
                }
            } else {
                setitem(&results, &accum_col, f64::NAN)?;
            }
        }
        status += " Daily accum gain calc complete.";

        if has(&results, "value")? && has(&results, "net_flow")? {
            let mw = core.call_method1(
                "money_weighted",
                (
                    as_f64(results.get_item("value")?)?,
                    as_f64(results.get_item("net_flow")?)?,
                ),
            )?;
            let mw = unpack(&mw, 3)?;
            setitem(&results, "Cumulative Net Flow", &mw[0])?;
            setitem(&results, "Absolute Gain ($)", &mw[1])?;
            setitem(&results, "Absolute ROI (%)", &mw[2])?;
        }
        setitem(&results, "drawdown", &twr_parts[2])?;

        let in_list = |v: &[&str]| -> PyResult<bool> {
            for x in v {
                if interval.eq(*x)? {
                    return Ok(true);
                }
            }
            Ok(false)
        };
        let resampled;
        if in_list(&["D", "1d"])? {
            resampled = results.clone();
            rename_accumulated(&resampled, benchmark_symbols_yf)?;
            // Daily points at 16:00 New York, so they land on their trading day.
            if !resampled.getattr("empty")?.is_truthy()? {
                let r = (|| -> PyResult<()> {
                    let dates = resampled.getattr("index")?.getattr("date")?;
                    let naive = pd.getattr("DatetimeIndex")?.call1((dates,))?;
                    let close = naive
                        .call_method1("tz_localize", ("America/New_York",))?
                        .add(pd.getattr("Timedelta")?.call(
                            (),
                            Some(&kwargs(
                                py,
                                &[("hours", &16i64.into_pyobject(py)?.into_any())],
                            )?),
                        )?)?;
                    resampled.setattr("index", close.call_method1("tz_convert", ("UTC",))?)?;
                    log(
                        py,
                        "info",
                        "Shifted Daily index to 16:00 EST to correct visual date alignment.",
                    )?;
                    Ok(())
                })();
                match r {
                    Ok(()) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => log(
                        py,
                        "error",
                        format!("Failed to shift daily index: {}", es(py, &e)),
                    )?,
                    Err(e) => return Err(e),
                }
            }
        } else if in_list(&["W", "M", "ME"])? && !results.getattr("empty")?.is_truthy()? {
            let fi = o.fmt(interval)?;
            let r = (|| -> PyResult<Bound<'py, PyAny>> {
                log(
                    py,
                    "debug",
                    format!(
                        "Resampling '{fi}': Input results_df shape: {}",
                        o.fmt(&results.getattr("shape")?)?
                    ),
                )?;
                let freq = if interval.eq("M")? {
                    pystr(py, "ME")
                } else {
                    interval.clone()
                };
                let agg = PyDict::new(py);
                for (k, v) in [
                    ("value", "last"),
                    ("daily_gain", "sum"),
                    ("net_flow", "sum"),
                    ("Portfolio Accumulated Gain Daily", "last"),
                    ("Cumulative Net Flow", "last"),
                    ("Absolute Gain ($)", "last"),
                    ("drawdown", "last"),
                ] {
                    agg.set_item(k, v)?;
                }
                for bm in benchmark_symbols_yf.try_iter()? {
                    let bm = o.fmt(&bm?)?;
                    let price_col = format!("{bm} Price");
                    let accum_col = format!("{bm} Accumulated Gain Daily");
                    if has(&results, &price_col)? {
                        agg.set_item(&price_col, "last")?;
                    }
                    if has(&results, &accum_col)? {
                        agg.set_item(&accum_col, "last")?;
                    }
                }
                let out = results
                    .call_method1("resample", (freq,))?
                    .call_method1("agg", (agg,))?;
                log(
                    py,
                    "debug",
                    format!(
                        "Resampling '{fi}': Output final_df_resampled shape: {}",
                        o.fmt(&out.getattr("shape")?)?
                    ),
                )?;
                rename_accumulated(&out, benchmark_symbols_yf)?;
                let denom = out
                    .get_item("Cumulative Net Flow")?
                    .call_method1("replace", (0, f64::NAN))?;
                setitem(
                    &out,
                    "Absolute ROI (%)",
                    out.get_item("Absolute Gain ($)")?
                        .div(denom.call_method0("abs")?)?
                        .mul(100.0)?,
                )?;
                Ok(out)
            })();
            resampled = match r {
                Ok(out) => {
                    status += &format!(" Resampled to '{fi}' with Absolute metrics.");
                    out
                }
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    log(
                        py,
                        "warning",
                        format!(
                            "Hist WARN: Failed resampling to interval '{fi}': {}. Returning daily results.",
                            es(py, &e)
                        ),
                    )?;
                    status += &format!(" Resampling failed ('{fi}').");
                    results.clone()
                }
                Err(e) => return Err(e),
            };
        } else {
            resampled = results.clone();
            rename_accumulated(&resampled, benchmark_symbols_yf)?;
        }

        let keep = PyList::empty(py);
        for c in [
            "value",
            "daily_gain",
            "daily_return",
            "Portfolio Accumulated Gain",
            "Absolute Gain ($)",
            "Absolute ROI (%)",
            "Cumulative Net Flow",
            "drawdown",
        ] {
            keep.append(c)?;
        }
        for bm in benchmark_symbols_yf.try_iter()? {
            let bm = o.fmt(&bm?)?;
            for c in [format!("{bm} Price"), format!("{bm} Accumulated Gain")] {
                if has(&resampled, &c)? {
                    keep.append(c)?;
                }
            }
        }
        let existing = PyList::empty(py);
        for c in keep.iter() {
            if resampled.getattr("columns")?.contains(&c)? {
                existing.append(c)?;
            }
        }
        let mut out = resampled.get_item(&existing)?.call_method0("copy")?;
        let names = PyDict::new(py);
        names.set_item("value", "Portfolio Value")?;
        names.set_item("daily_gain", "Portfolio Daily Gain")?;
        out.call_method(
            "rename",
            (),
            Some(&kwargs(
                py,
                &[("columns", names.as_any()), ("inplace", &tru(py))],
            )?),
        )?;
        log(
            py,
            "debug",
            format!(
                "[_calculate_accumulated_gains_and_resample] Final final_df_output shape: {}",
                o.fmt(&out.getattr("shape")?)?
            ),
        )?;
        if debug_enabled(py)? {
            log(
                py,
                "debug",
                format!(
                    "Final final_df_output tail:\n{}",
                    o.fmt(&out.call_method0("tail")?.call_method0("to_string")?)?
                ),
            )?;
        }
        if interval.ne("D")? && has(&out, "daily_return")? {
            out.call_method(
                "drop",
                (),
                Some(&kwargs(
                    py,
                    &[
                        ("columns", PyList::new(py, ["daily_return"])?.as_any()),
                        ("inplace", &tru(py)),
                    ],
                )?),
            )?;
        }

        // The requested range, normalised to the point before it.
        if start_date_filter.is_truthy()? && end_date_filter.is_truthy()? {
            let r = (|| -> PyResult<()> {
                let timestamp = pd.getattr("Timestamp")?;
                let pd_start = timestamp.call1((start_date_filter,))?;
                let pd_end =
                    timestamp
                        .call1((end_date_filter,))?
                        .add(pd.getattr("Timedelta")?.call(
                            (),
                            Some(&kwargs(
                                py,
                                &[
                                    ("hours", &23i64.into_pyobject(py)?.into_any()),
                                    ("minutes", &59i64.into_pyobject(py)?.into_any()),
                                    ("seconds", &59i64.into_pyobject(py)?.into_any()),
                                ],
                            )?),
                        )?)?;
                if !out.getattr("index")?.getattr("tz")?.is_none() {
                    out.setattr(
                        "index",
                        out.getattr("index")?
                            .call_method1("tz_localize", (py.None(),))?,
                    )?;
                }
                let naive = resampled.call_method0("copy")?;
                if !naive.getattr("index")?.getattr("tz")?.is_none() {
                    naive.setattr(
                        "index",
                        naive
                            .getattr("index")?
                            .call_method1("tz_localize", (py.None(),))?,
                    )?;
                }
                let available = out.getattr("index")?.call_method0("sort_values")?;
                let visible_mask = available
                    .rich_compare(&pd_start, CompareOp::Ge)?
                    .bitand(available.rich_compare(&pd_end, CompareOp::Le)?)?;
                let visible = available.get_item(visible_mask)?;
                if !visible.getattr("empty")?.is_truthy()? {
                    let first = visible.get_item(0)?;
                    for col in out.getattr("columns")?.try_iter()? {
                        let col = col?;
                        if !col.contains("Accumulated Gain")? {
                            continue;
                        }
                        let fc = o.fmt(&col)?;
                        let mut divisor = o.fl(0.0);
                        let mut found = false;
                        let data = out.get_item(&col)?;
                        if data
                            .call_method0("isna")?
                            .call_method0("all")?
                            .is_truthy()?
                            || data
                                .rich_compare(0, CompareOp::Eq)?
                                .call_method0("all")?
                                .is_truthy()?
                        {
                            log(
                                py,
                                "debug",
                                format!(
                                    "Normalization skipped for {fc}: Column is empty or all zeros."
                                ),
                            )?;
                            continue;
                        }
                        // t-1: the last point before the visible range.
                        if naive.getattr("columns")?.contains(&col)? {
                            let before = naive
                                .getattr("index")?
                                .rich_compare(&first, CompareOp::Lt)?;
                            let pre = naive
                                .getattr("loc")?
                                .get_item(PyTuple::new(py, [before, col.clone()])?)?
                                .call_method0("dropna")?;
                            if !pre.getattr("empty")?.is_truthy()? {
                                let t1 = pre.getattr("iloc")?.get_item(-1)?;
                                if o.notna(&t1)? && t1.ne(0)? {
                                    divisor = t1;
                                    found = true;
                                    log(
                                        py,
                                        "debug",
                                        format!(
                                            "BUG-10 FIX: Using t-1 baseline for {fc}: {}",
                                            o.fmt(&divisor)?
                                        ),
                                    )?;
                                }
                            }
                        }
                        if !found {
                            let t0 = out
                                .getattr("loc")?
                                .get_item(PyTuple::new(py, [first.clone(), col.clone()])?)?;
                            if o.notna(&t0)? && t0.ne(0)? {
                                divisor = t0;
                                found = true;
                                log(
                                    py,
                                    "debug",
                                    format!(
                                        "BUG-10 FIX: Fallback to t0 baseline for {fc}: {}",
                                        o.fmt(&divisor)?
                                    ),
                                )?;
                            } else {
                                log(
                                    py,
                                    "debug",
                                    format!(
                                        "t0 baseline invalid for {fc} (val={}). Searching forward...",
                                        o.fmt(&t0)?
                                    ),
                                )?;
                                for d in visible.try_iter()? {
                                    let d = d?;
                                    let val = out
                                        .getattr("loc")?
                                        .get_item(PyTuple::new(py, [d.clone(), col.clone()])?)?;
                                    if o.notna(&val)? && val.ne(0)? {
                                        divisor = val;
                                        found = true;
                                        log(
                                            py,
                                            "debug",
                                            format!(
                                                "Found valid baseline for {fc} at {}: {}",
                                                o.fmt(&d)?,
                                                o.fmt(&divisor)?
                                            ),
                                        )?;
                                        break;
                                    }
                                }
                            }
                        }
                        if found && divisor.ne(0)? {
                            if o.abs(&divisor)?.lt(1e-4)? {
                                log(
                                    py,
                                    "warning",
                                    format!(
                                        "Normalization divisor for {fc} is too small ({}). Skipping normalization to prevent spikes.",
                                        o.fmt(&divisor)?
                                    ),
                                )?;
                            } else {
                                setitem(&out, &col, out.get_item(&col)?.div(&divisor)?)?;
                            }
                        } else {
                            log(
                                py,
                                "warning",
                                format!(
                                    "Could not find ANY valid normalization baseline for {fc} in range."
                                ),
                            )?;
                        }
                    }
                }
                out = out
                    .getattr("loc")?
                    .get_item(pyslice(py, &pd_start, &pd_end)?)?;
                log(
                    py,
                    "debug",
                    format!(
                        "Filtered and normalized final output to range: {} - {}",
                        o.fmt(start_date_filter)?,
                        o.fmt(end_date_filter)?
                    ),
                )?;
                Ok(())
            })();
            match r {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "warning",
                    format!("Could not apply final date filter: {}", es(py, &e)),
                )?,
                Err(e) => return Err(e),
            }
        }
        Ok(out)
    })();
    match body {
        Ok(out) => Ok((out, twr, status)),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log_exc(
                py,
                "error",
                "Hist CRITICAL: Accum gain/resample calc error",
                &e,
            )?;
            status += " Accum gain/resample calc failed.";
            Ok((empty()?, o.nan(), status))
        }
        Err(e) => Err(e),
    }
}

/// The arguments the daily-results pass threads through its stages.
struct Daily<'a, 'py> {
    py: Python<'py>,
    m: &'a Bound<'py, PyModule>,
    o: &'a Ops<'py>,
    pd: &'a Bound<'py, PyModule>,
    np: &'a Bound<'py, PyModule>,
    fd: String,
    interval: &'a Bound<'py, PyAny>,
    worker_signals: &'a Bound<'py, PyAny>,
    csv_file: &'a Bound<'py, PyAny>,
    start_date: &'a Bound<'py, PyAny>,
    end_date: &'a Bound<'py, PyAny>,
    txs: &'a Bound<'py, PyAny>,
    prices_unadjusted: &'a Bound<'py, PyAny>,
    prices_adjusted: &'a Bound<'py, PyAny>,
    fx: &'a Bound<'py, PyAny>,
    display_currency: &'a Bound<'py, PyAny>,
    internal_to_yf: &'a Bound<'py, PyAny>,
    account_currency_map: &'a Bound<'py, PyAny>,
    default_currency: &'a Bound<'py, PyAny>,
    symbol_to_id: &'a Bound<'py, PyAny>,
    id_to_symbol: &'a Bound<'py, PyAny>,
    account_to_id: &'a Bound<'py, PyAny>,
    id_to_account: &'a Bound<'py, PyAny>,
    manual_overrides: &'a Bound<'py, PyAny>,
    type_to_id: &'a Bound<'py, PyAny>,
    currency_to_id: &'a Bound<'py, PyAny>,
    id_to_currency: &'a Bound<'py, PyAny>,
    all_holdings_qty: &'a Bound<'py, PyAny>,
    all_cash_balances: &'a Bound<'py, PyAny>,
    all_last_prices: &'a Bound<'py, PyAny>,
    all_holdings_start_date: &'a Bound<'py, PyAny>,
    included_accounts: &'a Bound<'py, PyAny>,
    num_processes: &'a Bound<'py, PyAny>,
    account_cash_mode_map: &'a Bound<'py, PyAny>,
}

const INTRADAY: [&str; 8] = ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"];

impl<'py> Daily<'_, 'py> {
    fn kw(&self, pairs: &[(&str, &Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
        kwargs(self.py, pairs)
    }
    fn empty_frame(&self) -> PyResult<Bound<'py, PyAny>> {
        self.pd.getattr("DataFrame")?.call0()
    }
    fn is_intraday(&self) -> PyResult<bool> {
        for i in INTRADAY {
            if self.interval.eq(i)? {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// The cache validation and load; `Ok(None)` when the frame must be
    /// calculated.
    fn load_cache(
        &self,
        cache_file: &Bound<'py, PyAny>,
        cache_key: &Bound<'py, PyAny>,
        meta_file: &Bound<'py, PyAny>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let (py, o, pd, fd) = (self.py, self.o, self.pd, &self.fd);
        let os = self.m.getattr("os")?;
        let os_path = os.getattr("path")?;
        log(
            py,
            "info",
            format!(
                "Hist Daily (Scope: {fd}): Found daily cache files (.feather & .meta.json). Validating..."
            ),
        )?;
        let r = (|| -> PyResult<Option<Bound<'py, PyAny>>> {
            let f = py
                .import("builtins")?
                .getattr("open")?
                .call1((meta_file, "r"))?;
            let json = self.m.getattr("json")?;
            let metadata =
                with1(py, &f, |f| json.call_method1("load", (f,)))?.expect("files do not suppress");
            if metadata
                .call_method1("get", ("cache_key",))?
                .ne(cache_key)?
            {
                log(
                    py,
                    "warning",
                    format!(
                        "Hist WARN (Scope: {fd}): Daily cache key MISMATCH in metadata. Recalculating."
                    ),
                )?;
                return Err(PyValueError::new_err("Cache key mismatch"));
            }
            // A cache made from a CSV is stale once the CSV is newer.
            if !metadata.call_method1("get", ("csv_mtime",))?.is_none() {
                let csv = self.csv_file;
                let fc = o.fmt(csv)?;
                if csv.is_truthy()?
                    && csv.is_instance(&o.str_t)?
                    && csv.call_method0("strip")?.is_truthy()?
                {
                    let r = (|| -> PyResult<()> {
                        if os_path.call_method1("exists", (csv,))?.is_truthy()? {
                            let current = os_path.call_method1("getmtime", (csv,))?;
                            let cached = metadata.call_method1("get", ("csv_mtime",))?;
                            if current.gt(cached)? {
                                log(
                                    py,
                                    "warning",
                                    format!(
                                        "Hist WARN (Scope: {fd}): Transactions CSV file '{fc}' modified since cache was created. Recalculating."
                                    ),
                                )?;
                                return Err(PyValueError::new_err("CSV file modified"));
                            }
                        } else {
                            log(
                                py,
                                "warning",
                                format!(
                                    "Hist WARN (Scope: {fd}): Transactions CSV file '{fc}' (for mtime check) not found, but cache expected one. Recalculating."
                                ),
                            )?;
                            return Err(PyValueError::new_err(
                                "CSV file for mtime check not found",
                            ));
                        }
                        Ok(())
                    })();
                    match r {
                        Ok(()) => {}
                        Err(e) if e.is_instance_of::<PyOSError>(py) => {
                            log(
                                py,
                                "warning",
                                format!(
                                    "Hist WARN (Scope: {fd}): Could not access transactions CSV file '{fc}' for mtime check: {}. Recalculating.",
                                    es(py, &e)
                                ),
                            )?;
                            return Err(PyValueError::new_err("CSV mtime check failed (OS error)"));
                        }
                        Err(e) => return Err(e),
                    }
                } else {
                    log(
                        py,
                        "warning",
                        format!(
                            "Hist WARN (Scope: {fd}): Cache metadata expects 'csv_mtime' but no valid CSV path provided for current validation (path: '{fc}'). Recalculating."
                        ),
                    )?;
                    return Err(PyValueError::new_err(
                        "Cache expected CSV mtime, but no CSV path provided now.",
                    ));
                }
            }

            log(
                py,
                "info",
                format!("Hist Daily (Scope: {fd}): Cache metadata valid. Loading Feather data..."),
            )?;
            let mut df = pd.call_method1("read_feather", (cache_file,))?;
            if !df.is_instance(&pd.getattr("DataFrame")?)? {
                return Err(PyValueError::new_err(
                    "Loaded Feather cache is not a DataFrame.",
                ));
            }
            let utc = self.kw(&[("utc", &tru(py))])?;
            let inplace = self.kw(&[("inplace", &tru(py))])?;
            let columns = df.getattr("columns")?;
            if columns.contains("Date")? {
                setitem(
                    &df,
                    "Date",
                    pd.call_method("to_datetime", (df.get_item("Date")?,), Some(&utc))?,
                )?;
                df.call_method("set_index", ("Date",), Some(&inplace))?;
            } else if columns.contains("index")? {
                rename_inplace(&df, "index", "Date")?;
                setitem(
                    &df,
                    "Date",
                    pd.call_method("to_datetime", (df.get_item("Date")?,), Some(&utc))?,
                )?;
                df.call_method("set_index", ("Date",), Some(&inplace))?;
            } else if !df
                .getattr("index")?
                .is_instance(&pd.getattr("DatetimeIndex")?)?
            {
                if df
                    .getattr("index")?
                    .is_instance(&pd.getattr("RangeIndex")?)?
                {
                    log(
                        py,
                        "warning",
                        "Loaded Feather cache has RangeIndex and no Date column. Cache might be invalid.",
                    )?;
                    return Err(PyValueError::new_err(
                        "Invalid cache structure (missing Date index/column)",
                    ));
                }
                df.setattr(
                    "index",
                    pd.call_method(
                        "to_datetime",
                        (df.getattr("index")?,),
                        Some(&self.kw(&[("errors", &pystr(py, "coerce")), ("utc", &tru(py))])?),
                    )?,
                )?;
                df = df.get_item(pd.call_method1("notnull", (df.getattr("index")?,))?)?;
            }
            df.call_method("sort_index", (), Some(&inplace))?;
            let missing = PyList::empty(py);
            for c in ["value", "net_flow", "daily_return", "daily_gain"] {
                if !df.getattr("columns")?.contains(c)? {
                    missing.append(c)?;
                }
            }
            if !missing.is_truthy()? && !df.getattr("empty")?.is_truthy()? {
                log(
                    py,
                    "info",
                    format!(
                        "Hist Daily (Scope: {fd}): Loaded {} rows from Feather cache.",
                        df.len()?
                    ),
                )?;
                Ok(Some(df))
            } else {
                log(
                    py,
                    "warning",
                    format!(
                        "Hist WARN (Scope: {fd}): Feather cache missing columns ({}), empty, or failed validation. Recalculating.",
                        o.fmt(&missing)?
                    ),
                )?;
                Ok(None)
            }
        })();
        match r {
            Ok(v) => Ok(v),
            // ValueError covers json.JSONDecodeError, its subclass.
            Err(e) if e.is_instance_of::<PyValueError>(py) => {
                log(
                    py,
                    "warning",
                    format!(
                        "Hist WARN (Scope: {fd}): Daily cache validation failed: {}. Recalculating.",
                        es(py, &e)
                    ),
                )?;
                Ok(None)
            }
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "warning",
                    format!(
                        "Hist WARN (Scope: {fd}): Error reading/validating daily cache: {}. Recalculating.",
                        es(py, &e)
                    ),
                )?;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }

    /// The calculation grid: daily history, then intraday market-hours bars
    /// on NYSE days for an intraday request.
    fn grid(&self, calc_start: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let (py, pd, np) = (self.py, self.pd, self.np);
        let timedelta = self.m.getattr("timedelta")?;
        let day = timedelta.call(
            (),
            Some(&self.kw(&[("days", &1i64.into_pyobject(py)?.into_any())])?),
        )?;
        let date_range = |start: &Bound<'py, PyAny>, end: &Bound<'py, PyAny>, freq: &str| {
            pd.call_method(
                "date_range",
                (),
                Some(&kwargs(
                    py,
                    &[("start", start), ("end", end), ("freq", &pystr(py, freq))],
                )?),
            )
        };
        let mut grid = if self.is_intraday()? {
            let freq_map = PyDict::new(py);
            for (k, v) in [
                ("1h", "h"),
                ("60m", "h"),
                ("1m", "1min"),
                ("2m", "2min"),
                ("5m", "5min"),
                ("15m", "15min"),
                ("30m", "30min"),
                ("90m", "90min"),
            ] {
                freq_map.set_item(k, v)?;
            }
            let active_freq = freq_map.call_method1("get", (self.interval, "D"))?;
            let mut historical = date_range(calc_start, &self.start_date.sub(&day)?, "D")?;
            if !historical.getattr("empty")?.is_truthy()? {
                historical = historical.call_method1("tz_localize", ("UTC",))?;
            }
            let timestamp = pd.getattr("Timestamp")?;
            let utc_kw = self.kw(&[("tz", &pystr(py, "UTC"))])?;
            let end_of_period = timestamp.call((self.end_date,), Some(&utc_kw))?.add(&day)?;
            // Clip to now, with the bar in progress, so the 1D graph tracks live bars.
            let now = timestamp.call_method("now", (), Some(&utc_kw))?;
            let (bound, inclusive) = if end_of_period.gt(&now)? {
                log(
                    py,
                    "info",
                    format!(
                        "[_load_or_calculate] 1D Graph Clip: now={}, bound={}",
                        self.o.fmt(&now)?,
                        self.o.fmt(&now)?
                    ),
                )?;
                (now, "both")
            } else {
                (end_of_period, "left")
            };
            let mut active = pd.call_method(
                "date_range",
                (),
                Some(&self.kw(&[
                    ("start", &timestamp.call((self.start_date,), Some(&utc_kw))?),
                    ("end", &bound),
                    ("freq", &active_freq),
                    ("inclusive", &pystr(py, inclusive)),
                ])?),
            )?;
            // Market hours only, so ffill does not draw the overnight gap.
            if !active.getattr("empty")?.is_truthy()? {
                let ny = active.call_method1("tz_convert", ("America/New_York",))?;
                let keep = ny.call_method1("indexer_between_time", ("09:30", "16:00"))?;
                active = active.get_item(keep)?;
            }
            // And NYSE days only, so the anchor point is never a Sunday.
            if !active.getattr("empty")?.is_truthy()? {
                let r = (|| -> PyResult<Bound<'py, PyAny>> {
                    let ny = active.call_method1("tz_convert", ("America/New_York",))?;
                    let schedule = self.m.getattr("nyse_schedule")?.call1((
                        ny.get_item(0)?.call_method0("date")?,
                        ny.get_item(-1)?.call_method0("date")?,
                    ))?;
                    let tradable = py
                        .import("builtins")?
                        .getattr("set")?
                        .call1((schedule.getattr("index")?.getattr("date")?,))?;
                    let flags = PyList::empty(py);
                    for d in ny.getattr("date")?.try_iter()? {
                        flags.append(tradable.contains(d?)?)?;
                    }
                    active.get_item(np.call_method1("array", (flags,))?)
                })();
                match r {
                    Ok(a) => active = a,
                    Err(e) if e.is_instance_of::<PyException>(py) => log(
                        py,
                        "warning",
                        format!(
                            "Hist WARN: Could not drop non-trading days from the intraday range: {}",
                            es(py, &e)
                        ),
                    )?,
                    Err(e) => return Err(e),
                }
            }
            let (h_empty, a_empty) = (
                historical.getattr("empty")?.is_truthy()?,
                active.getattr("empty")?.is_truthy()?,
            );
            if h_empty && a_empty {
                pd.getattr("DatetimeIndex")?.call1((PyList::empty(py),))?
            } else if h_empty {
                active
            } else if a_empty {
                historical
            } else {
                historical
                    .call_method1("append", (active,))?
                    .call_method0("unique")?
                    .call_method0("sort_values")?
            }
        } else {
            date_range(calc_start, self.end_date, "D")?.call_method1("tz_localize", ("UTC",))?
        };
        if grid.getattr("tz")?.is_none() {
            grid = grid.call_method1("tz_localize", ("UTC",))?;
        }
        Ok(grid)
    }

    /// The replay kernel's inputs from a ledger, and its call.
    fn replay(&self, grid: &Bound<'py, PyAny>) -> PyResult<Vec<Bound<'py, PyAny>>> {
        let (py, m, o, pd, np) = (self.py, self.m, self.o, self.pd, self.np);
        let mut sorted = self
            .txs
            .call_method(
                "sort_values",
                (),
                Some(&self.kw(&[("by", PyList::new(py, ["Date", "original_index"])?.as_any())])?),
            )?
            .call_method0("copy")?;
        let int64 = np.getattr("int64")?;
        let float64 = np.getattr("float64")?;
        let i64_kw = self.kw(&[("dtype", &int64)])?;
        let ordinals = |v: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            np.call_method(
                "array",
                (v.getattr("values")?.call_method1("astype", ("int64",))?,),
                Some(&i64_kw),
            )
        };
        let date_ordinals = ordinals(grid.clone())?;
        let tx_dates = ordinals(pd.call_method(
            "to_datetime",
            (sorted.get_item("Date")?,),
            Some(&self.kw(&[("utc", &tru(py))])?),
        )?)?;
        let tx_symbols = sorted
            .get_item("Symbol")?
            .call_method1("map", (self.symbol_to_id,))?
            .getattr("values")?
            .call_method1("astype", (&int64,))?;
        let normalize = m.getattr("_normalize_series")?;
        let mut account_ids = normalize
            .call1((sorted.get_item("Account")?,))?
            .call_method1("map", (self.account_to_id,))?;
        if account_ids
            .call_method0("isna")?
            .call_method0("any")?
            .is_truthy()?
        {
            let valid = account_ids.call_method0("notna")?;
            sorted = sorted.get_item(&valid)?;
            account_ids = account_ids.get_item(&valid)?;
        }
        let tx_accounts = account_ids
            .getattr("values")?
            .call_method1("astype", (&int64,))?;
        let to_accounts = if sorted.getattr("columns")?.contains("To Account")? {
            normalize
                .call1((sorted.get_item("To Account")?,))?
                .call_method1("map", (self.account_to_id,))?
        } else {
            pd.getattr("Series")?.call(
                (-1,),
                Some(&self.kw(&[("index", &sorted.getattr("index")?)])?),
            )?
        };
        let tx_to_accounts = to_accounts
            .call_method1("fillna", (-1,))?
            .getattr("values")?
            .call_method1("astype", (&int64,))?;
        let tx_types = sorted
            .get_item("Type")?
            .getattr("str")?
            .call_method0("lower")?
            .getattr("str")?
            .call_method0("strip")?
            .call_method1("map", (self.type_to_id,))?
            .call_method1("fillna", (-1,))?
            .getattr("values")?
            .call_method1("astype", (&int64,))?;
        let floats = |c: &str| -> PyResult<Bound<'py, PyAny>> {
            sorted
                .get_item(c)?
                .call_method1("fillna", (0.0,))?
                .getattr("values")?
                .call_method1("astype", (&float64,))
        };
        let zeros = || {
            np.call_method(
                "zeros",
                (sorted.len()?,),
                Some(&kwargs(py, &[("dtype", &float64)])?),
            )
        };
        let columns = sorted.getattr("columns")?;
        let tx_qty = floats("Quantity")?;
        let tx_comm = floats("Commission")?;
        let tx_split = floats("Split Ratio")?;
        let price_col = if columns.contains("Price/Share")? {
            "Price/Share"
        } else {
            "Price"
        };
        let tx_prices = if columns.contains(price_col)? {
            floats(price_col)?
        } else {
            zeros()?
        };
        let tx_totals = if columns.contains("Total Amount")? {
            floats("Total Amount")?
        } else {
            zeros()?
        };
        let type_id = |n: &str| self.type_to_id.call_method1("get", (n, -1));
        let mut first_ids = Vec::new();
        for n in [
            "split",
            "stock split",
            "buy",
            "deposit",
            "sell",
            "withdrawal",
            "short sell",
            "buy to cover",
            "transfer",
            "spin off",
        ] {
            first_ids.push(type_id(n)?);
        }
        let cash_symbol_id = self
            .symbol_to_id
            .call_method1("get", (m.getattr("CASH_SYMBOL_CSV")?, -1))?;
        let shortable = PyList::empty(py);
        for s in m.getattr("SHORTABLE_SYMBOLS")?.try_iter()? {
            let s = s?;
            if self.symbol_to_id.contains(&s)? {
                shortable.append(self.symbol_to_id.get_item(&s)?)?;
            }
        }
        let shortable_ids = np.call_method("array", (shortable,), Some(&i64_kw))?;
        let num_symbols = self.symbol_to_id.len()?;
        let num_accounts = self.account_to_id.len()?;
        let mut cash_ids = Vec::new();
        for n in ["dividend", "interest", "fees", "tax"] {
            cash_ids.push(type_id(n)?);
        }
        let modes = if self.account_cash_mode_map.is_truthy()? {
            self.account_cash_mode_map.clone()
        } else {
            PyDict::new(py).into_any()
        };
        let cash_modes = np.call_method("zeros", (num_accounts,), Some(&i64_kw))?;
        for item in modes.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let upper = kv[0].call_method0("upper")?.call_method0("strip")?;
            if self.account_to_id.contains(&upper)?
                && o.str_t
                    .call1((&kv[1],))?
                    .call_method0("lower")?
                    .eq("auto")?
            {
                cash_modes.set_item(self.account_to_id.get_item(&upper)?, 1)?;
            }
        }
        let mut args: Vec<Bound<'py, PyAny>> = vec![
            date_ordinals,
            tx_dates,
            tx_symbols,
            tx_to_accounts,
            tx_accounts,
            tx_types,
            tx_qty,
            tx_comm,
            tx_split,
            tx_prices,
            tx_totals,
            num_symbols.into_pyobject(py)?.into_any(),
            num_accounts.into_pyobject(py)?.into_any(),
        ];
        args.extend(first_ids);
        args.extend(cash_ids);
        args.push(cash_symbol_id);
        args.push(m.getattr("STOCK_QUANTITY_CLOSE_TOLERANCE")?);
        args.push(shortable_ids);
        args.push(cash_modes);
        let out = m
            .getattr("_calculate_daily_holdings_chronological_numba")?
            .call1(PyTuple::new(py, args)?)?;
        unpack(&out, 3)
    }

    /// Prices multiplied back by the splits after each date, so they match
    /// the ledger's raw quantities.
    fn split_raw_prices(&self) -> PyResult<Bound<'py, PyDict>> {
        let (py, pd, np) = (self.py, self.pd, self.np);
        let operator = py.import("operator")?;
        let raw = PyDict::new(py);
        let txs = self.txs;
        let mut splits = txs.get_item(
            txs.get_item("Type")?
                .getattr("str")?
                .call_method0("lower")?
                .getattr("str")?
                .call_method0("strip")?
                .call_method1("isin", (PyList::new(py, ["split", "stock split"])?,))?,
        )?;
        // Deduplicated as the replay does: "All Accounts" first, per symbol and month.
        if !splits.getattr("empty")?.is_truthy()? {
            splits = splits.call_method0("copy")?;
            let all_accounts = splits
                .get_item("Account")?
                .call_method1("astype", (&self.o.str_t,))?
                .getattr("str")?
                .call_method0("lower")?
                .rich_compare("all accounts", CompareOp::Eq)?;
            setitem(
                &splits,
                "__split_priority",
                np.call_method1("where", (all_accounts, 0, 1))?,
            )?;
            setitem(
                &splits,
                "__ym",
                pd.call_method1("to_datetime", (splits.get_item("Date")?,))?
                    .getattr("dt")?
                    .call_method1("to_period", ("M",))?,
            )?;
            let sort_cols = PyList::new(py, ["Symbol", "__ym", "__split_priority"])?;
            if splits.getattr("columns")?.contains("original_index")? {
                sort_cols.append("original_index")?;
            }
            splits = splits.call_method(
                "sort_values",
                (),
                Some(&self.kw(&[("by", sort_cols.as_any())])?),
            )?;
            splits = splits.call_method(
                "drop_duplicates",
                (),
                Some(&self.kw(&[(
                    "subset",
                    PyList::new(py, ["Symbol", "__ym", "Split Ratio"])?.as_any(),
                )])?),
            )?;
        }
        let coerce = self.kw(&[("errors", &pystr(py, "coerce"))])?;
        let utc = self.kw(&[("utc", &tru(py))])?;
        for item in self.prices_unadjusted.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let (yf_sym, price_df) = (&kv[0], &kv[1]);
            if price_df.getattr("empty")?.is_truthy()? {
                raw.set_item(yf_sym, price_df)?;
                continue;
            }
            let ledger_syms = PyList::empty(py);
            for item in self.internal_to_yf.call_method0("items")?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                if kv[1].eq(yf_sym)? {
                    ledger_syms.append(&kv[0])?;
                }
            }
            let sym_splits = splits.get_item(
                splits
                    .get_item("Symbol")?
                    .call_method1("isin", (ledger_syms,))?,
            )?;
            if sym_splits.getattr("empty")?.is_truthy()? {
                raw.set_item(yf_sym, price_df)?;
                continue;
            }
            let new_df = price_df.call_method0("copy")?;
            let factors = pd.getattr("Series")?.call(
                (1.0,),
                Some(&self.kw(&[("index", &new_df.getattr("index")?)])?),
            )?;
            let newest_first = sym_splits.call_method(
                "sort_values",
                (),
                Some(&self.kw(&[("by", &pystr(py, "Date")), ("ascending", &fals(py))])?),
            )?;
            for row in newest_first.call_method0("iterrows")?.try_iter()? {
                let row = unpack(&row?, 2)?.remove(1);
                let s_date = pd.call_method("to_datetime", (row.get_item("Date")?,), Some(&utc))?;
                let mut ratio = pd.call_method(
                    "to_numeric",
                    (row.call_method1("get", ("Split Ratio",))?,),
                    Some(&coerce),
                )?;
                let qty = pd.call_method(
                    "to_numeric",
                    (row.call_method1("get", ("Quantity",))?,),
                    Some(&coerce),
                )?;
                // Some importers put the ratio in Quantity.
                if (ratio.is_none() || ratio.le(1e-9)?)
                    && (!qty.is_none() && qty.gt(0)? && qty.le(20.0)?)
                {
                    ratio = qty.clone();
                }
                if ratio.is_truthy()? && ratio.gt(1e-9)? {
                    factors.setattr(
                        "index",
                        pd.call_method("to_datetime", (factors.getattr("index")?,), Some(&utc))?,
                    )?;
                    let mask = factors
                        .getattr("index")?
                        .rich_compare(&s_date, CompareOp::Lt)?;
                    let loc = factors.getattr("loc")?;
                    let v = operator.call_method1("imul", (loc.get_item(&mask)?, &ratio))?;
                    loc.set_item(&mask, v)?;
                }
            }
            for c in ["price", "Close", "Adj Close", "Open", "High", "Low"] {
                if new_df.getattr("columns")?.contains(c)? {
                    setitem(&new_df, c, new_df.get_item(c)?.mul(&factors)?)?;
                }
            }
            raw.set_item(yf_sym, new_df)?;
        }
        Ok(raw)
    }

    /// The chronological pass: holdings (L1 slice or replay), valuation and
    /// flows on the grid. Returns (frame, status).
    fn chrono(&self, status: &mut String) -> PyResult<Bound<'py, PyAny>> {
        let (py, m, pd, fd) = (self.py, self.m, self.pd, &self.fd);
        let txs = self.txs;
        let first_tx_date = if !txs.getattr("empty")?.is_truthy()? {
            txs.get_item("Date")?
                .call_method0("min")?
                .call_method0("date")?
        } else {
            self.start_date.clone()
        };
        // Always from the first transaction, so the cost and TWR context hold.
        let calc_start = first_tx_date;
        let grid = self.grid(&calc_start)?;

        let use_l1 = !self.all_holdings_qty.is_none()
            && !self.all_cash_balances.is_none()
            && !self.is_intraday()?
            && !self.included_accounts.is_truthy()?;
        let (qty, cash, last) = if use_l1 {
            log(
                py,
                "info",
                format!("Hist Daily (Scope: {fd}): Using L1 cached holdings..."),
            )?;
            *status = " Using pre-calculated daily values...".into();
            let l1_start = if self.all_holdings_start_date.is_truthy()? {
                self.all_holdings_start_date.clone()
            } else {
                txs.get_item("Date")?
                    .call_method0("min")?
                    .call_method0("date")?
            };
            let mut offset = calc_start.sub(&l1_start)?.getattr("days")?;
            if offset.lt(0)? {
                offset = 0i64.into_pyobject(py)?.into_any();
            }
            let needed = grid.len()?;
            let window = py
                .import("builtins")?
                .getattr("slice")?
                .call1((&offset, offset.add(needed)?))?;
            let qty = self.all_holdings_qty.get_item(&window)?;
            let cash = self.all_cash_balances.get_item(&window)?;
            let last = if self.all_last_prices.is_none() {
                py.None().into_bound(py)
            } else {
                self.all_last_prices.get_item(&window)?
            };
            log(
                py,
                "info",
                format!(
                    "Hist Daily: Using L1 cache with offset {} days.",
                    self.o.fmt(&offset)?
                ),
            )?;
            (qty, cash, last)
        } else {
            log(
                py,
                "info",
                format!(
                    "Hist Daily (Scope: {fd}): L1 cache miss. Calculating daily holdings chronologically (numba_chrono)..."
                ),
            )?;
            *status = " Calculating daily values (chrono)...".into();
            let mut parts = self.replay(&grid)?;
            let last = parts.pop().expect("three parts");
            let cash = parts.pop().expect("three parts");
            let qty = parts.pop().expect("three parts");
            (qty, cash, last)
        };

        let include_ids = if self.included_accounts.is_truthy()? {
            let ids = PySet::empty(py)?;
            for acc in self.included_accounts.try_iter()? {
                let key = self
                    .o
                    .str_t
                    .call1((acc?,))?
                    .call_method0("upper")?
                    .call_method0("strip")?;
                ids.add(self.account_to_id.call_method1("get", (key,))?)?;
            }
            ids.discard(py.None())?;
            ids.into_any()
        } else {
            py.import("builtins")?
                .getattr("set")?
                .call1((self.account_to_id.call_method0("values")?,))?
        };

        let raw_prices = self.split_raw_prices()?;
        let valued = m.getattr("_value_daily_holdings_vectorized")?.call(
            (),
            Some(&self.kw(&[
                ("date_range", &grid),
                ("daily_holdings_qty_np", &qty),
                ("daily_last_prices_np", &last),
                ("daily_cash_balances_np", &cash),
                ("historical_prices_yf_unadjusted", raw_prices.as_any()),
                ("historical_fx_yf", self.fx),
                ("target_currency", self.display_currency),
                ("account_currency_map", self.account_currency_map),
                ("id_to_symbol", self.id_to_symbol),
                ("id_to_account", self.id_to_account),
                ("internal_to_yf_map", self.internal_to_yf),
                ("account_ids_to_include_set", &include_ids),
                ("default_currency", self.default_currency),
                ("interval", self.interval),
                ("manual_overrides_dict", self.manual_overrides),
            ])?),
        )?;
        let valued = unpack(&valued, 3)?;
        if valued[1].is_truthy()? {
            *status += &self.o.fmt(&valued[2])?;
        }
        let flows = m
            .getattr("_calculate_daily_net_cash_flow_vectorized")?
            .call(
                (),
                Some(&self.kw(&[
                    ("date_range", &grid),
                    ("transactions_df", txs),
                    ("target_currency", self.display_currency),
                    ("historical_fx_yf", self.fx),
                    ("default_currency", self.default_currency),
                    ("included_accounts", self.included_accounts),
                    ("historical_prices_yf_unadjusted", raw_prices.as_any()),
                    ("internal_to_yf_map", self.internal_to_yf),
                ])?),
            )?;
        let flows = unpack(&flows, 2)?;
        if flows[1].is_truthy()? {
            *status += " (flow errors)";
        }
        let data = PyDict::new(py);
        data.set_item("value", &valued[0])?;
        data.set_item("net_flow", &flows[0])?;
        pd.getattr("DataFrame")?
            .call((data,), Some(&self.kw(&[("index", &grid)])?))
    }

    /// The per-date worker pass (the fallback path): the dates, the pool and
    /// the results frame; `Err(status)` for the original's early returns.
    fn workers(&self, status: &mut String) -> PyResult<Result<Bound<'py, PyAny>, String>> {
        let (py, m, o, pd, fd) = (self.py, self.m, self.o, self.pd, &self.fd);
        *status = " Calculating daily values...".into();
        let txs = self.txs;
        let calc_start = if !txs.getattr("empty")?.is_truthy()? {
            txs.get_item("Date")?
                .call_method0("min")?
                .call_method0("date")?
        } else {
            self.start_date.clone()
        };
        let calc_end = self.end_date.clone();
        let adjusted = self.prices_adjusted;
        let mut source = Some(pystr(py, "SPY"));
        if !adjusted.contains("SPY")? {
            source = if adjusted.is_truthy()? {
                Some(
                    adjusted
                        .call_method0("keys")?
                        .try_iter()?
                        .next()
                        .transpose()?
                        .expect("non-empty"),
                )
            } else {
                None
            };
        }
        let source_text = match &source {
            Some(s) => o.fmt(s)?,
            None => "None".into(),
        };
        let mut market_days = pd.getattr("Index")?.call(
            (PyList::empty(py),),
            Some(&self.kw(&[("dtype", &pystr(py, "object"))])?),
        )?;
        log(
            py,
            "debug",
            format!(
                "[_load_or_calculate_daily_results] Attempting to use '{source_text}' for market days."
            ),
        )?;
        if let Some(src) = source
            .as_ref()
            .filter(|s| adjusted.contains(*s).unwrap_or(false))
        {
            let bench = adjusted.get_item(src)?;
            let index = bench.getattr("index")?;
            if !bench.getattr("empty")?.is_truthy()?
                && (index.is_instance(&pd.getattr("DatetimeIndex")?)?
                    || index.is_instance(&pd.getattr("Index")?)?)
            {
                let r = (|| -> PyResult<()> {
                    let dt = pd.call_method(
                        "to_datetime",
                        (&index,),
                        Some(&self.kw(&[("errors", &pystr(py, "coerce")), ("utc", &tru(py))])?),
                    )?;
                    let valid = dt.call_method0("dropna")?;
                    if !valid.getattr("empty")?.is_truthy()? {
                        let mut intraday = false;
                        for i in ["1m", "5m", "15m", "30m", "60m", "1h", "90m"] {
                            if self.interval.eq(i)? {
                                intraday = true;
                                break;
                            }
                        }
                        market_days = if intraday {
                            valid.call_method0("unique")?
                        } else {
                            pd.getattr("Index")?
                                .call1((valid.getattr("date")?,))?
                                .call_method0("unique")?
                        };
                        log(
                            py,
                            "info",
                            format!(
                                "  Successfully created market_days_index from '{source_text}' ({} points).",
                                market_days.len()?
                            ),
                        )?;
                    } else {
                        log(
                            py,
                            "warning",
                            format!(
                                "  Benchmark '{source_text}' index could not be converted to valid datetimes."
                            ),
                        )?;
                    }
                    Ok(())
                })();
                match r {
                    Ok(()) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => log(
                        py,
                        "warning",
                        format!(
                            "WARN: Failed converting benchmark index for market days: {}",
                            es(py, &e)
                        ),
                    )?,
                    Err(e) => return Err(e),
                }
            } else {
                log(
                    py,
                    "warning",
                    format!("  Benchmark df '{source_text}' is empty or has invalid index type."),
                )?;
            }
        }
        let dates = if market_days.getattr("empty")?.is_truthy()? {
            log(
                py,
                "warning",
                format!("Hist WARN (Scope: {fd}): No market days found. Using business day range."),
            )?;
            pd.call_method(
                "date_range",
                (),
                Some(&self.kw(&[
                    ("start", &calc_start),
                    ("end", &calc_end),
                    ("freq", &pystr(py, "B")),
                ])?),
            )?
            .getattr("date")?
            .call_method0("tolist")?
        } else {
            let r = market_days
                .get_item(
                    market_days
                        .rich_compare(&calc_start, CompareOp::Ge)?
                        .bitand(market_days.rich_compare(&calc_end, CompareOp::Le)?)?,
                )?
                .call_method0("tolist")?;
            log(
                py,
                "debug",
                format!(
                    "  Filtered market days index to range {} - {}. Found {} dates.",
                    o.fmt(&calc_start)?,
                    o.fmt(&calc_end)?,
                    r.len()?
                ),
            )?;
            r
        };
        if !dates.is_truthy()? {
            log(
                py,
                "error",
                format!(
                    "Hist ERROR (Scope: {fd}): No calculation dates found in range {} to {}.",
                    o.fmt(&calc_start)?,
                    o.fmt(&calc_end)?
                ),
            )?;
            return Ok(Err(format!("{status} No calculation dates found.")));
        }
        let b = py.import("builtins")?;
        let n_dates = dates.len()?;
        log(
            py,
            "info",
            format!(
                "Hist Daily (Scope: {fd}): Determined {n_dates} calculation dates from {} to {}",
                o.fmt(&b.getattr("min")?.call1((&dates,))?)?,
                o.fmt(&b.getattr("max")?.call1((&dates,))?)?
            ),
        )?;
        let head = dates.get_item(b.getattr("slice")?.call1((py.None(), 5))?)?;
        let tail = dates.get_item(b.getattr("slice")?.call1((-5, py.None()))?)?;
        log(
            py,
            "debug",
            format!(
                "  First 5 calc dates: {}, Last 5 calc dates: {}",
                o.fmt(&head)?,
                o.fmt(&tail)?
            ),
        )?;
        log(
            py,
            "info",
            format!("Hist Daily (Scope: {fd}): Calculating {n_dates} daily metrics parallel..."),
        )?;

        let worker = m.getattr("partial")?.call(
            (m.getattr("_calculate_daily_metrics_worker")?,),
            Some(&self.kw(&[
                ("transactions_df", txs),
                ("historical_prices_yf_unadjusted", self.prices_unadjusted),
                ("historical_prices_yf_adjusted", self.prices_adjusted),
                ("historical_fx_yf", self.fx),
                ("target_currency", self.display_currency),
                ("internal_to_yf_map", self.internal_to_yf),
                ("account_currency_map", self.account_currency_map),
                ("default_currency", self.default_currency),
                ("manual_overrides_dict", self.manual_overrides),
                ("symbol_to_id", self.symbol_to_id),
                ("id_to_symbol", self.id_to_symbol),
                ("account_to_id", self.account_to_id),
                ("id_to_account", self.id_to_account),
                ("type_to_id", self.type_to_id),
                ("currency_to_id", self.currency_to_id),
                ("id_to_currency", self.id_to_currency),
                ("calc_method", &m.getattr("HISTORICAL_CALC_METHOD")?),
                ("included_accounts", self.included_accounts),
                ("account_cash_mode_map", self.account_cash_mode_map),
            ])?),
        )?;
        let results = PyList::empty(py);
        let time = m.getattr("time")?;
        let pool_start = time.call_method0("time")?;
        let mut processes = self.num_processes.clone();
        if processes.is_none() {
            match m
                .getattr("os")?
                .call_method0("cpu_count")
                .and_then(|c| c.sub(1))
            {
                Ok(n) => processes = b.getattr("max")?.call1((1, n))?,
                Err(e) if e.is_instance_of::<PyNotImplementedError>(py) => {
                    processes = 1i64.into_pyobject(py)?.into_any()
                }
                Err(e) => return Err(e),
            }
        }
        processes = b.getattr("max")?.call1((1, &processes))?;

        let pooled = (|| -> PyResult<()> {
            let chunksize = b.getattr("max")?.call1((
                1,
                (n_dates as i64)
                    .into_pyobject(py)?
                    .floor_div(processes.mul(4)?)?,
            ))?;
            let mut last_percent: i64 = -1;
            let consume = |iterator: Bound<'py, PyAny>, last_percent: &mut i64| -> PyResult<()> {
                for (i, result) in iterator.try_iter()?.enumerate() {
                    let result = result?;
                    if i % 100 == 0 && i > 0 {
                        log(py, "info", format!("  Processed {i}/{n_dates} days..."))?;
                    }
                    if result.is_truthy()? {
                        results.append(&result)?;
                    }
                    if self.worker_signals.is_truthy()?
                        && self.worker_signals.hasattr("progress")?
                    {
                        let r = (|| -> PyResult<()> {
                            let percent = (((i + 1) as f64 / n_dates as f64) * 100.0) as i64;
                            if percent > *last_percent {
                                self.worker_signals
                                    .getattr("progress")?
                                    .call_method1("emit", (percent,))?;
                                *last_percent = percent;
                            }
                            Ok(())
                        })();
                        match r {
                            Ok(()) => {}
                            Err(e) if e.is_instance_of::<PyException>(py) => log(
                                py,
                                "warning",
                                format!("Warning: Failed to emit progress: {}", es(py, &e)),
                            )?,
                            Err(e) => return Err(e),
                        }
                    }
                }
                log(
                    py,
                    "info",
                    format!("  Finished processing all {n_dates} days."),
                )?;
                Ok(())
            };
            if processes.eq(1)? {
                consume(
                    b.getattr("map")?.call1((&worker, &dates))?,
                    &mut last_percent,
                )?;
            } else {
                let pool = m.getattr("multiprocessing")?.call_method(
                    "Pool",
                    (),
                    Some(&self.kw(&[
                        ("processes", &processes),
                        ("maxtasksperchild", &50i64.into_pyobject(py)?.into_any()),
                    ])?),
                );
                let pool = pool?;
                let r = (|| {
                    let it = pool.call_method(
                        "imap_unordered",
                        (&worker, &dates),
                        Some(&self.kw(&[("chunksize", &chunksize)])?),
                    )?;
                    consume(it, &mut last_percent)
                })();
                // Graceful shutdown, so terminate() never cuts a result pipe.
                let shutdown = pool
                    .call_method0("close")
                    .and_then(|_| pool.call_method0("join"));
                match shutdown {
                    Ok(_) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => {
                        log(
                            py,
                            "warning",
                            format!("Pool shutdown warning: {}", es(py, &e)),
                        )?;
                        match pool.call_method0("terminate") {
                            Ok(_) => {}
                            Err(e) if e.is_instance_of::<PyException>(py) => {}
                            Err(e) => return Err(e),
                        }
                    }
                    Err(e) => return Err(e),
                }
                r?;
            }
            Ok(())
        })();
        let pool_failed = match pooled {
            Ok(()) => None,
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "error",
                    format!("Hist CRITICAL (Scope: {fd}): Pool failed: {}", es(py, &e)),
                )?;
                print_exc(py, &e)?;
                Some(None)
            }
            Err(e) => Some(Some(e)),
        };
        // finally:
        let pool_end = time.call_method0("time")?;
        log(
            py,
            "info",
            format!(
                "Hist Daily (Scope: {fd}): Pool finished. Received {} results total.",
                results.len()
            ),
        )?;
        if results.is_truthy()? {
            log(
                py,
                "debug",
                format!("  Sample result [0]: {}", o.fmt(&results.get_item(0)?)?),
            )?;
            if results.len() > 1 {
                log(
                    py,
                    "debug",
                    format!(
                        "  Sample result [-1]: {}",
                        o.fmt(&results.as_any().get_item(-1)?)?
                    ),
                )?;
            }
        }
        log(
            py,
            "info",
            format!(
                "Hist Daily (Scope: {fd}): Pool finished in {}s.",
                o.fmt_spec(&pool_end.sub(&pool_start)?, ".2f")?
            ),
        )?;
        match pool_failed {
            None => {}
            Some(None) => return Ok(Err(format!("{status} Multiprocessing failed."))),
            Some(Some(e)) => return Err(e),
        }

        let successful = PyList::empty(py);
        for r in results.iter() {
            if !r
                .call_method1("get", ("worker_error", false))?
                .is_truthy()?
            {
                successful.append(r)?;
            }
        }
        let failed_count = n_dates as i64 - successful.len() as i64;
        if failed_count > 0 {
            *status += &format!(" ({failed_count} dates failed in worker).");
            for r in results.iter() {
                if r.call_method1("get", ("worker_error", false))?
                    .is_truthy()?
                {
                    log(
                        py,
                        "error",
                        format!(
                            "Worker Error Sample: {}",
                            o.fmt(&r.call_method1("get", ("error_message", "Unknown Error"))?)?
                        ),
                    )?;
                    break;
                }
            }
        }
        if !successful.is_truthy()? {
            log(
                py,
                "error",
                format!(
                    "Hist ERROR (Scope: {fd}): No successful results from workers. daily_results_list size: {}",
                    results.len()
                ),
            )?;
            return Ok(Err(format!(
                "{status} All daily calculations failed in worker."
            )));
        }

        let built = (|| -> PyResult<Result<Bound<'py, PyAny>, String>> {
            let df = pd.getattr("DataFrame")?.call1((&successful,))?;
            setitem(
                &df,
                "Date",
                pd.call_method(
                    "to_datetime",
                    (df.get_item("Date")?,),
                    Some(&self.kw(&[("utc", &tru(py))])?),
                )?,
            )?;
            let inplace = self.kw(&[("inplace", &tru(py))])?;
            df.call_method("set_index", ("Date",), Some(&inplace))?;
            df.call_method("sort_index", (), Some(&inplace))?;
            let coerce = self.kw(&[("errors", &pystr(py, "coerce"))])?;
            for c in ["value", "net_flow"] {
                setitem(
                    &df,
                    c,
                    pd.call_method("to_numeric", (df.get_item(c)?,), Some(&coerce))?,
                )?;
            }
            if df.getattr("empty")?.is_truthy()? {
                return Ok(Err(format!(
                    "{status} All rows dropped due to NaN portfolio value."
                )));
            }
            Ok(Ok(df))
        })();
        match built {
            Ok(v) => Ok(v),
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "error",
                    format!(
                        "Hist ERROR (Scope: {fd}): Post-processing failed: {}",
                        es(py, &e)
                    ),
                )?;
                print_exc(py, &e)?;
                Ok(Err(format!("{status} Error processing results.")))
            }
            Err(e) => Err(e),
        }
    }

    /// The daily gain and TWR return, with flow-driven spikes and transfer
    /// artifacts healed.
    fn derive(&self, df: &Bound<'py, PyAny>) -> PyResult<()> {
        let (py, o, pd, np) = (self.py, self.o, self.pd, self.np);
        let loc = df.getattr("loc")?;
        let at = df.getattr("at")?;
        let cell =
            |rows: &Bound<'py, PyAny>, col: &str| PyTuple::new(py, [rows.clone(), pystr(py, col)]);
        let previous = df.get_item("value")?.call_method1("shift", (1,))?;
        let flow = df.get_item("net_flow")?.call_method1("fillna", (0.0,))?;
        setitem(
            df,
            "daily_gain",
            df.get_item("value")?.sub(&previous)?.sub(&flow)?,
        )?;
        setitem(df, "daily_return", f64::NAN)?;
        // Deposits count from the start of the day, withdrawals at its end.
        let adjusted_prev = previous.add(flow.call_method(
            "clip",
            (),
            Some(&self.kw(&[("lower", &o.fl(0.0))])?),
        )?)?;
        let valid = adjusted_prev
            .call_method0("notna")?
            .bitand(o.abs(&adjusted_prev)?.rich_compare(0.1, CompareOp::Gt)?)?;
        let invalid = valid.call_method0("__invert__")?;
        if invalid.call_method0("any")?.is_truthy()? {
            let ignored = valid.call_method0("__invert__")?.call_method0("sum")?;
            if ignored.gt(0)? {
                log(
                    py,
                    "debug",
                    format!(
                        "TWR: Ignoring {} days where capital at risk was too small (<$0.10) — likely pre-funding period.",
                        o.fmt(&ignored)?
                    ),
                )?;
            }
        }
        let ret = loc
            .get_item(cell(&valid, "daily_gain")?)?
            .div(adjusted_prev.getattr("loc")?.get_item(&valid)?)?;
        loc.set_item(cell(&valid, "daily_return")?, ret)?;

        // Flow-driven spikes (a small book, a large flow) are capped; market moves kept.
        let threshold = 0.5;
        let returns = || df.get_item("daily_return");
        let spike = returns()?
            .rich_compare(threshold, CompareOp::Gt)?
            .bitor(returns()?.rich_compare(-threshold, CompareOp::Lt)?)?;
        let spike_info = |mask: &Bound<'py, PyAny>| -> PyResult<Vec<String>> {
            let mut out = Vec::new();
            for d in df.getattr("index")?.get_item(mask)?.try_iter()? {
                let d = d?;
                let v = at
                    .get_item(PyTuple::new(py, [d.clone(), pystr(py, "daily_return")])?)?
                    .mul(100)?;
                out.push(format!(
                    "{} ({}%)",
                    o.fmt(&d.call_method1("strftime", ("%Y-%m-%d",))?)?,
                    o.fmt_spec(&v, ".1f")?
                ));
            }
            Ok(out)
        };
        let listed = |info: &[String]| -> String {
            let shown: Vec<&str> = info.iter().take(12).map(String::as_str).collect();
            format!(
                "{}{}",
                shown.join(", "),
                if info.len() > 12 { "..." } else { "" }
            )
        };
        let pct = o.fmt_spec(&o.fl(threshold * 100.0), ".0f")?;
        if spike.call_method0("any")?.is_truthy()? {
            let flow_abs = df
                .get_item("net_flow")?
                .call_method1("fillna", (0.0,))?
                .call_method0("abs")?;
            let value_abs = df.get_item("value")?.call_method0("abs")?;
            let flow_driven = spike.bitand(
                value_abs
                    .rich_compare(1000.0, CompareOp::Lt)?
                    .bitand(flow_abs.rich_compare(value_abs.mul(0.5)?, CompareOp::Gt)?)?,
            )?;
            if flow_driven.call_method0("any")?.is_truthy()? {
                match spike_info(&flow_driven) {
                    Ok(info) => log(
                        py,
                        "warning",
                        format!(
                            "BUG-04 FIX: Capped {} FLOW-DRIVEN TWR spikes (> {pct}%): {}",
                            info.len(),
                            listed(&info)
                        ),
                    )?,
                    Err(e) if e.is_instance_of::<PyException>(py) => log(
                        py,
                        "error",
                        format!("Failed to log spike info: {}", es(py, &e)),
                    )?,
                    Err(e) => return Err(e),
                }
                loc.set_item(cell(&flow_driven, "daily_return")?, 0.0)?;
            }
            let market = spike.bitand(flow_driven.call_method0("__invert__")?)?;
            if market.call_method0("any")?.is_truthy()? {
                match spike_info(&market) {
                    Ok(info) => log(
                        py,
                        "info",
                        format!(
                            "BUG-04 FIX: Preserved {} legitimate market-driven returns (> {pct}%): {}",
                            info.len(),
                            listed(&info)
                        ),
                    )?,
                    Err(e) if e.is_instance_of::<PyException>(py) => {}
                    Err(e) => return Err(e),
                }
            }
        }

        // A transfer day's spike is healed when the flow, not the market, made it.
        let high_vol = returns()?
            .rich_compare(0.1, CompareOp::Gt)?
            .bitor(returns()?.rich_compare(-0.1, CompareOp::Lt)?)?;
        if high_vol.call_method0("any")?.is_truthy()? {
            let txs = self.txs;
            let transfer_days = txs
                .get_item(
                    txs.get_item("Type")?
                        .getattr("str")?
                        .call_method0("lower")?
                        .getattr("str")?
                        .call_method0("strip")?
                        .rich_compare("transfer", CompareOp::Eq)?,
                )?
                .get_item("Date")?
                .call_method0("unique")?;
            let transfer_dates = pd
                .call_method1("to_datetime", (transfer_days,))?
                .getattr("date")?;
            for item in df
                .get_item(&high_vol)?
                .call_method0("iterrows")?
                .try_iter()?
            {
                let kv = unpack(&item?, 2)?;
                let (idx, row) = (&kv[0], &kv[1]);
                if !transfer_dates.contains(idx.call_method0("date")?)? {
                    continue;
                }
                let abs_or_zero = |col: &str| -> PyResult<Bound<'py, PyAny>> {
                    if o.notna(&row.call_method1("get", (col,))?)? {
                        o.abs(&row.call_method1("get", (col, 0.0))?)
                    } else {
                        Ok(o.fl(0.0))
                    }
                };
                let net_flow_val = abs_or_zero("net_flow")?;
                let gain_val = abs_or_zero("daily_gain")?;
                if net_flow_val.gt(gain_val.mul(0.5)?)? {
                    log(
                        py,
                        "info",
                        format!(
                            "TWR HEALING: Zeroed flow-driven artifact on {} (Return: {}%, NetFlow: {}, Gain: {}).",
                            o.fmt(&idx.call_method0("date")?)?,
                            o.fmt_spec(&row.get_item("daily_return")?.mul(100)?, ".1f")?,
                            o.fmt_spec(&net_flow_val, ".0f")?,
                            o.fmt_spec(&gain_val, ".0f")?
                        ),
                    )?;
                    at.set_item(
                        PyTuple::new(py, [idx.clone(), pystr(py, "daily_return")])?,
                        0.0,
                    )?;
                }
            }
        }

        let anomalies = df.get_item(
            df.get_item("value")?
                .rich_compare(1.0, CompareOp::Lt)?
                .bitand(
                    df.get_item("net_flow")?
                        .rich_compare(1000.0, CompareOp::Gt)?,
                )?
                .bitand(returns()?.rich_compare(-0.99, CompareOp::Lt)?)?,
        )?;
        if !anomalies.getattr("empty")?.is_truthy()? {
            log(
                py,
                "warning",
                "CRITICAL TWR ANOMALY DETECTED: Deposits made but Portfolio Value is Zero.",
            )?;
            log(
                py,
                "warning",
                o.fmt(
                    &anomalies
                        .get_item(PyList::new(py, ["value", "net_flow", "daily_return"])?)?
                        .call_method0("head")?
                        .call_method0("to_string")?,
                )?,
            )?;
        }
        let gain = df.get_item("daily_gain")?;
        let zero_gain = gain
            .call_method0("notna")?
            .bitand(o.abs(&gain)?.rich_compare(1e-9, CompareOp::Lt)?)?;
        let zero_prev = previous
            .call_method0("notna")?
            .bitand(o.abs(&previous)?.rich_compare(1e-9, CompareOp::Le)?)?;
        loc.set_item(cell(&zero_gain.bitand(zero_prev)?, "daily_return")?, 0.0)?;
        if !df.getattr("empty")?.is_truthy()? {
            let first = df.getattr("index")?.get_item(0)?;
            loc.set_item(cell(&first, "daily_gain")?, f64::NAN)?;
            loc.set_item(cell(&first, "daily_return")?, f64::NAN)?;
        }
        let columns = df.getattr("columns")?;
        if !columns.contains("daily_gain")? || !columns.contains("daily_return")? {
            log(py, "error", "Failed to calculate derived daily metrics.")?;
        }
        let _ = np;
        Ok(())
    }

    /// The feather cache and its metadata.
    fn save(
        &self,
        df: &Bound<'py, PyAny>,
        cache_file: &Bound<'py, PyAny>,
        cache_key: &Bound<'py, PyAny>,
        meta_file: &Bound<'py, PyAny>,
    ) -> PyResult<()> {
        let (py, m, o, fd) = (self.py, self.m, self.o, &self.fd);
        log(
            py,
            "info",
            format!(
                "Hist Daily (Scope: {fd}): Saving daily results to cache: {}",
                o.fmt(cache_file)?
            ),
        )?;
        let r = (|| -> PyResult<()> {
            let os = m.getattr("os")?;
            let os_path = os.getattr("path")?;
            let dir = os_path.call_method1("dirname", (cache_file,))?;
            if dir.is_truthy()? {
                os.call_method(
                    "makedirs",
                    (&dir,),
                    Some(&self.kw(&[("exist_ok", &tru(py))])?),
                )?;
            }
            let csv = self.csv_file;
            let mtime = (|| -> PyResult<Bound<'py, PyAny>> {
                if csv.is_truthy()?
                    && csv.is_instance(&o.str_t)?
                    && csv.call_method0("strip")?.is_truthy()?
                    && os_path.call_method1("exists", (csv,))?.is_truthy()?
                {
                    os_path.call_method1("getmtime", (csv,))
                } else {
                    Ok(py.None().into_bound(py))
                }
            })();
            let mtime = match mtime {
                Ok(v) => v,
                Err(e) if e.is_instance_of::<PyOSError>(py) => py.None().into_bound(py),
                Err(e) => return Err(e),
            };
            let meta = PyDict::new(py);
            meta.set_item("cache_key", cache_key)?;
            meta.set_item(
                "cache_timestamp",
                m.getattr("datetime")?
                    .call_method0("now")?
                    .call_method0("isoformat")?,
            )?;
            meta.set_item("csv_mtime", mtime)?;
            let f = py
                .import("builtins")?
                .getattr("open")?
                .call1((meta_file, "w"))?;
            let json = m.getattr("json")?;
            with1(py, &f, |f| {
                json.call_method(
                    "dump",
                    (&meta, f),
                    Some(&self.kw(&[("indent", &2i64.into_pyobject(py)?.into_any())])?),
                )?;
                Ok(())
            })?;
            df.getattr("index")?.setattr("name", "Date")?;
            df.call_method0("reset_index")?
                .call_method1("to_feather", (cache_file,))?;
            Ok(())
        })();
        match r {
            Ok(()) => Ok(()),
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "warning",
                format!(
                    "Hist WARN (Scope: {fd}): Error writing daily cache: {}",
                    es(py, &e)
                ),
            ),
            Err(e) => Err(e),
        }
    }
}

/// `_load_or_calculate_daily_results`: (daily frame, whether it came from
/// the cache, a status note).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_load_or_calculate_daily_results<'py>(
    py: Python<'py>,
    use_daily_results_cache: &Bound<'py, PyAny>,
    daily_results_cache_file: &Bound<'py, PyAny>,
    daily_results_cache_key: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    worker_signals: &Bound<'py, PyAny>,
    transactions_csv_file: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    transactions_df_effective: &Bound<'py, PyAny>,
    historical_prices_yf_unadjusted: &Bound<'py, PyAny>,
    historical_prices_yf_adjusted: &Bound<'py, PyAny>,
    historical_fx_yf: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    internal_to_yf_map: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    symbol_to_id: &Bound<'py, PyAny>,
    id_to_symbol: &Bound<'py, PyAny>,
    account_to_id: &Bound<'py, PyAny>,
    id_to_account: &Bound<'py, PyAny>,
    manual_overrides_dict: &Bound<'py, PyAny>,
    type_to_id: &Bound<'py, PyAny>,
    currency_to_id: &Bound<'py, PyAny>,
    id_to_currency: &Bound<'py, PyAny>,
    all_holdings_qty: &Bound<'py, PyAny>,
    all_cash_balances: &Bound<'py, PyAny>,
    all_last_prices: &Bound<'py, PyAny>,
    all_holdings_start_date: &Bound<'py, PyAny>,
    included_accounts_list: &Bound<'py, PyAny>,
    num_processes: &Bound<'py, PyAny>,
    current_hist_version: &Bound<'py, PyAny>,
    filter_desc: &Bound<'py, PyAny>,
    calc_method: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, bool, String)> {
    let _ = current_hist_version;
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let d = Daily {
        py,
        m: &m,
        o: &o,
        pd: &pd,
        np: &np,
        fd: o.fmt(filter_desc)?,
        interval,
        worker_signals,
        csv_file: transactions_csv_file,
        start_date,
        end_date,
        txs: transactions_df_effective,
        prices_unadjusted: historical_prices_yf_unadjusted,
        prices_adjusted: historical_prices_yf_adjusted,
        fx: historical_fx_yf,
        display_currency,
        internal_to_yf: internal_to_yf_map,
        account_currency_map,
        default_currency,
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        manual_overrides: manual_overrides_dict,
        type_to_id,
        currency_to_id,
        id_to_currency,
        all_holdings_qty,
        all_cash_balances,
        all_last_prices,
        all_holdings_start_date,
        included_accounts: included_accounts_list,
        num_processes,
        account_cash_mode_map,
    };
    let fd = d.fd.clone();
    let mut daily_df = d.empty_frame()?;
    let mut from_cache = false;
    let mut status = String::new();
    log(
        py,
        "debug",
        format!(
            "[_load_or_calculate_daily_results] Received date range: {} to {}",
            o.fmt(start_date)?,
            o.fmt(end_date)?
        ),
    )?;
    let meta_file = if daily_results_cache_file.is_truthy()? {
        daily_results_cache_file.call_method1("replace", (".feather", ".meta.json"))?
    } else {
        py.None().into_bound(py)
    };

    let use_cache = use_daily_results_cache.is_truthy()?;
    if use_cache && daily_results_cache_file.is_truthy()? && daily_results_cache_key.is_truthy()? {
        let os_path = m.getattr("os")?.getattr("path")?;
        if os_path
            .call_method1("exists", (daily_results_cache_file,))?
            .is_truthy()?
            && meta_file.is_truthy()?
            && os_path.call_method1("exists", (&meta_file,))?.is_truthy()?
        {
            if let Some(df) = d.load_cache(
                daily_results_cache_file,
                daily_results_cache_key,
                &meta_file,
            )? {
                daily_df = df;
                from_cache = true;
                status = " Daily results loaded from cache.".into();
            } else {
                daily_df = d.empty_frame()?;
            }
        } else {
            log(
                py,
                "info",
                format!("Hist Daily (Scope: {fd}): Daily cache file not found. Calculating."),
            )?;
        }
    } else if !use_cache {
        log(
            py,
            "info",
            format!("Hist Daily (Scope: {fd}): Daily results caching disabled. Calculating."),
        )?;
    } else {
        log(
            py,
            "info",
            format!("Hist Daily (Scope: {fd}): Daily cache file/key invalid. Calculating."),
        )?;
    }

    if !from_cache {
        if calc_method.eq("numba_chrono")? {
            daily_df = d.chrono(&mut status)?;
        }
        if daily_df.getattr("empty")?.is_truthy()? {
            match d.workers(&mut status)? {
                Ok(df) => daily_df = df,
                Err(early) => return Ok((d.empty_frame()?, false, early)),
            }
        }
    }

    // The derived gain and return, for the calculated and the cached frame alike.
    if !daily_df.getattr("empty")?.is_truthy()? {
        let columns = daily_df.getattr("columns")?;
        if !columns.contains("daily_gain")? || !columns.contains("daily_return")? {
            match d.derive(&daily_df) {
                Ok(()) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => log(
                    py,
                    "error",
                    format!("Error calculating derived metrics: {}", es(py, &e)),
                )?,
                Err(e) => return Err(e),
            }
        }
    }

    if !from_cache {
        status += &format!(" {} days calculated.", daily_df.len()?);
        if use_cache
            && daily_results_cache_file.is_truthy()?
            && daily_results_cache_key.is_truthy()?
            && !daily_df.getattr("empty")?.is_truthy()?
        {
            d.save(
                &daily_df,
                daily_results_cache_file,
                daily_results_cache_key,
                &meta_file,
            )?;
        }
    }
    Ok((daily_df, from_cache, status))
}
