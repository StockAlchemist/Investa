//! `portfolio_history.calculate_historical_performance`: the whole history
//! request, from split deduplication and the id mappings through input
//! preparation, the L1 holdings, the price and FX fetches (with the intraday
//! merge and the leading-edge fill), the daily results and the benchmark
//! alignment, to the accumulation and the status line.
//!
//! Ported line for line from `src/portfolio_history.py` (frozen in
//! `tests/reference/portfolio_history.py`). It orchestrates the market
//! provider and pandas, so every call is Python's own, in the original's
//! order. Siblings and module state are looked up on `portfolio_history` at
//! call time. Two of the original's unbound-variable paths are kept (see
//! the port doc): an empty effective ledger reads `full_end_date_tx` before
//! it is set, and a benchmark frame with no usable column reuses the
//! previous benchmark's series.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyUnboundLocalError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList, PyTuple};

use crate::pa_rows::Ops;
use crate::ph_small::ph;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, setitem, unpack};

const INTRADAY: [&str; 8] = ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"];

fn tru(py: Python<'_>) -> Bound<'_, PyAny> {
    PyBool::new(py, true).to_owned().into_any()
}

fn unbound(name: &str) -> PyErr {
    PyUnboundLocalError::new_err(format!(
        "cannot access local variable '{name}' where it is not associated with a value"
    ))
}

fn one_of(v: &Bound<'_, PyAny>, items: &[&str]) -> PyResult<bool> {
    for i in items {
        if v.eq(*i)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// The stabilization fill without pandas' per-table machinery:
///
/// ```python
/// new_index = pd.Index([full_start_ts]).union(v.index)
/// v = v.reindex(new_index).sort_index()
/// v = v.interpolate(method="time").bfill().ffill()
/// ```
///
/// On a frame of float64 and int64 columns (the price tables carry an int64
/// Volume) over a unique, increasing, NaT-free UTC index (s, ms, us or ns)
/// that starts after `full_start_ts`, that is: every column cast to float64,
/// as the reindex's missing start row casts it, the start stamp prepended in
/// the union's unit, the finer of the two (the union takes pandas' generic
/// path, the stamp's own one-row index having no freq, so the result's freq is
/// inferred), and each column's missing cells replaced by `np.interp` over the
/// int64 stamps of its valid ones, counted in that unit. That is `interpolate(method="time")` exactly; the leading cells
/// it puts back to NaN are then backfilled with the first valid value, which
/// is what `np.interp` holds them at, and the trailing ones it already holds at
/// the last. A column with no valid cell stays empty. The values go in a
/// column-major block, as `reindex` builds them. Any other frame, or a start
/// that does not sort first, returns `None` for the pandas path.
fn stabilized<'py>(
    pd: &Bound<'py, PyAny>,
    v: &Bound<'py, PyAny>,
    full_start_ts: &Bound<'py, PyAny>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = v.py();
    let np = py.import("numpy")?;
    let index = v.getattr("index")?;
    let plain = v.get_type().is(&pd.getattr("DataFrame")?)
        && !v.getattr("attrs")?.is_truthy()?
        && v.getattr("flags")?
            .getattr("allows_duplicate_labels")?
            .is_truthy()?
        && index.get_type().is(&pd.getattr("DatetimeIndex")?)
        && index.getattr("tz")?.str()?.to_str()? == "UTC"
        && index.getattr("is_unique")?.is_truthy()?
        && index.getattr("is_monotonic_increasing")?.is_truthy()?
        && !index.getattr("hasnans")?.is_truthy()?
        && full_start_ts.getattr("tz")?.str()?.to_str()? == "UTC";
    if !plain {
        return Ok(None);
    }
    let rows = index.len()?;
    let float64 = np.call_method1("dtype", ("float64",))?;
    let int64 = np.call_method1("dtype", ("int64",))?;
    let mut ncols = 0usize;
    for dtype in v.getattr("dtypes")?.try_iter()? {
        let dtype = dtype?;
        if !dtype.eq(&float64)? && !dtype.eq(&int64)? {
            return Ok(None);
        }
        ncols += 1;
    }
    if rows == 0 || ncols == 0 {
        return Ok(None);
    }
    // The union's unit: the finer of the index's and the start stamp's (pandas
    // casts both to it), in which both convert exactly. The interpolation
    // runs on integer stamps in this unit, as pandas' does.
    // The start's side of the union is `pd.Index([full_start_ts])`, whose unit
    // is pandas' to choose: nanoseconds in pandas 2 whatever the stamp's own,
    // the stamp's unit in pandas 3.
    let start_index = pd
        .getattr("Index")?
        .call1((PyList::new(py, [full_start_ts])?,))?;
    let index_unit: String = index.getattr("unit")?.extract()?;
    let start_unit: String = start_index.getattr("unit")?.extract()?;
    let (Some(a), Some(b)) = (ticks_per_day(&index_unit), ticks_per_day(&start_unit)) else {
        return Ok(None);
    };
    let (unit, day) = if a >= b {
        (index_unit, a)
    } else {
        (start_unit, b)
    };
    let start: i64 = start_index
        .call_method1("as_unit", (&unit,))?
        .getattr("asi8")?
        .get_item(0)?
        .extract()?;
    let stamps = index.call_method1("as_unit", (&unit,))?.getattr("asi8")?;
    let first: i64 = stamps.get_item(0)?.extract()?;
    if start >= first {
        return Ok(None);
    }
    let x = np.call_method1(
        "concatenate",
        ((
            np.call_method1("array", (PyList::new(py, [start])?, &int64))?,
            &stamps,
        ),),
    )?;
    let mut new_index = pd.getattr("DatetimeIndex")?.call(
        (x.call_method1("view", (format!("M8[{unit}]"),))?,),
        Some(&kwargs(py, &[("tz", &index.getattr("tz")?)])?),
    )?;
    let ints: numpy::PyReadonlyArray1<i64> = x.extract()?;
    if !surely_no_freq(ints.as_slice()?, day) {
        new_index = new_index.call_method1("_with_freq", ("infer",))?;
    }

    let block = np.call_method1("empty", (PyTuple::new(py, [ncols, rows + 1])?, &float64))?;
    let slice = py.import("builtins")?.getattr("slice")?;
    let all = slice.call1((py.None(),))?;
    block.set_item(
        PyTuple::new(py, [all.clone(), 0usize.into_pyobject(py)?.into_any()])?,
        f64::NAN,
    )?;
    block.set_item(
        PyTuple::new(py, [all, slice.call1((1usize, py.None()))?])?,
        v.getattr("values")?.getattr("T")?,
    )?;
    for j in 0..ncols {
        let col = block.get_item(j)?;
        let invalid = np.call_method1("isnan", (&col,))?;
        let valid = invalid.call_method0("__invert__")?;
        if !valid.call_method0("any")?.is_truthy()? {
            continue;
        }
        let filled = np.call_method1(
            "interp",
            (
                x.get_item(&invalid)?,
                x.get_item(&valid)?,
                col.get_item(&valid)?,
            ),
        )?;
        col.set_item(&invalid, filled)?;
    }
    let frame = pd.getattr("DataFrame")?.call(
        (block.getattr("T")?,),
        Some(&kwargs(
            py,
            &[("index", &new_index), ("columns", &v.getattr("columns")?)],
        )?),
    )?;
    Ok(Some(frame))
}

/// Ticks in a day for a datetime64 unit; None for a unit the fill leaves to
/// pandas.
fn ticks_per_day(unit: &str) -> Option<i64> {
    match unit {
        "s" => Some(86_400),
        "ms" => Some(86_400_000),
        "us" => Some(86_400_000_000),
        "ns" => Some(86_400_000_000_000),
        _ => None,
    }
}

/// Whether pandas' frequency inference (`_FrequencyInferer.get_freq`) must
/// answer `None` for these UTC stamps (`day` ticks to a day), unique and increasing: true
/// only where every rule it tries is ruled out, false where it might find one
/// (the caller then asks pandas). The rules, in its order:
/// - with whole-day steps, the yearly, quarterly and monthly rules need one
///   month-to-month step; the daily rule, one step; business days, steps of
///   exactly {1, 3} days; week-of-month, one weekday;
/// - with shorter steps, business hours are steps of {1, 17}, {1, 65} or
///   {1, 17, 65} hours, and any other rule needs one step.
#[allow(non_snake_case)]
fn surely_no_freq(stamps: &[i64], DAY: i64) -> bool {
    let HOUR = DAY / 24;
    if stamps.len() < 3 {
        return false;
    }
    let steps: std::collections::BTreeSet<i64> = stamps.windows(2).map(|w| w[1] - w[0]).collect();
    if steps.len() < 2 {
        return false;
    }
    let smallest = *steps.iter().next().expect("two steps");
    let is = |want: &[i64]| steps.iter().copied().eq(want.iter().copied());
    if smallest % DAY != 0 {
        return !(is(&[HOUR, 17 * HOUR])
            || is(&[HOUR, 65 * HOUR])
            || is(&[HOUR, 17 * HOUR, 65 * HOUR]));
    }
    if is(&[DAY, 3 * DAY]) {
        return false;
    }
    let days: Vec<i64> = stamps.iter().map(|s| s.div_euclid(DAY)).collect();
    let weekdays: std::collections::BTreeSet<i64> =
        days.iter().map(|d| (d + 3).rem_euclid(7)).collect();
    if weekdays.len() < 2 {
        return false;
    }
    let months: Vec<i64> = days.iter().map(|&d| month_number(d)).collect();
    let month_steps: std::collections::BTreeSet<i64> =
        months.windows(2).map(|w| w[1] - w[0]).collect();
    month_steps.len() > 1
}

/// `year * 12 + month` of the civil (proleptic Gregorian) date `days` after
/// 1970-01-01, as pandas' `fields["Y"] * 12 + fields["M"]` (H. Hinnant's
/// `civil_from_days`).
fn month_number(days: i64) -> i64 {
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    year * 12 + month
}

/// `investa_core.ph_stabilize(v, full_start_ts)`: [`stabilized`], or the
/// pandas sequence where it declines, so the tests can hold the fill against
/// pandas directly.
#[pyfunction]
pub fn ph_stabilize<'py>(
    v: &Bound<'py, PyAny>,
    full_start_ts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = v.py();
    let pd = py.import("pandas")?;
    if let Some(filled) = stabilized(pd.as_any(), v, full_start_ts)? {
        return Ok(filled);
    }
    let new_index = pd
        .getattr("Index")?
        .call1((PyList::new(py, [full_start_ts])?,))?
        .call_method1("union", (v.getattr("index")?,))?;
    v.call_method1("reindex", (new_index,))?
        .call_method0("sort_index")?
        .call_method(
            "interpolate",
            (),
            Some(&kwargs(py, &[("method", &pystr(py, "time"))])?),
        )?
        .call_method0("bfill")?
        .call_method0("ffill")
}

/// `calculate_historical_performance`: (daily frame, adjusted prices, FX
/// tables, status line with the TWR factor).
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn ph_historical_performance<'py>(
    py: Python<'py>,
    all_transactions_df_cleaned: &Bound<'py, PyAny>,
    original_transactions_df_for_ignored: &Bound<'py, PyAny>,
    ignored_indices_from_load: &Bound<'py, PyAny>,
    ignored_reasons_from_load: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    benchmark_symbols_yf: &Bound<'py, PyAny>,
    display_currency: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
    use_raw_data_cache: &Bound<'py, PyAny>,
    use_daily_results_cache: &Bound<'py, PyAny>,
    num_processes: &Bound<'py, PyAny>,
    include_accounts: &Bound<'py, PyAny>,
    worker_signals: &Bound<'py, PyAny>,
    exclude_accounts: &Bound<'py, PyAny>,
    user_symbol_map: &Bound<'py, PyAny>,
    manual_overrides_dict: &Bound<'py, PyAny>,
    user_excluded_symbols: &Bound<'py, PyAny>,
    original_csv_file_path: &Bound<'py, PyAny>,
    calc_method: &Bound<'py, PyAny>,
    account_cash_mode_map: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let _ = num_processes;
    let m = ph(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let b = py.import("builtins")?;
    let timedelta = m.getattr("timedelta")?;
    let days = |n: i64| -> PyResult<Bound<'py, PyAny>> {
        timedelta.call(
            (),
            Some(&kwargs(py, &[("days", &n.into_pyobject(py)?.into_any())])?),
        )
    };
    let empty_frame = || pd.getattr("DataFrame")?.call0();
    let early = |status: String| -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(
            py,
            [
                empty_frame()?,
                PyDict::new(py).into_any(),
                PyDict::new(py).into_any(),
                pystr(py, &status),
            ],
        )
    };
    let utc = kwargs(py, &[("utc", &tru(py))])?;

    // Splits deduplicated first, so holdings and prices see the same set.
    let all_tx = m
        .getattr("_deduplicate_split_transactions")?
        .call1((all_transactions_df_cleaned,))?;
    let start_time = m.getattr("time")?.call_method0("time")?;
    let mut has_errors = false;
    let mut has_warnings = false;
    let mut status_parts: Vec<String> = Vec::new();
    let version = m.getattr("CURRENT_HIST_VERSION")?;

    let mut maps: Vec<Bound<'py, PyAny>> = (0..7).map(|_| PyDict::new(py).into_any()).collect();
    let mapped = (|| -> PyResult<()> {
        if !all_tx.is_none() && !all_tx.getattr("empty")?.is_truthy()? {
            maps = unpack(&m.getattr("generate_mappings")?.call1((&all_tx,))?, 7)?;
            log(
                py,
                "info",
                "Generated full string-to-ID mappings for historical calculation.",
            )?;
        }
        Ok(())
    })();
    match mapped {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "CRITICAL ERROR creating full string-to-ID mappings: {}",
                    es(py, &e)
                ),
            )?;
            return early(format!("Error: Mapping creation failed: {}", es(py, &e)));
        }
        Err(e) => return Err(e),
    }
    let (
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        type_to_id,
        currency_to_id,
        id_to_currency,
    ) = (
        &maps[0], &maps[1], &maps[2], &maps[3], &maps[4], &maps[5], &maps[6],
    );

    if !m.getattr("MARKET_PROVIDER_AVAILABLE")?.is_truthy()? {
        return early("Error: MarketDataProvider not available.".into());
    }
    if start_date.ge(end_date)? {
        return early("Error: Start date must be before end date.".into());
    }
    if !one_of(
        interval,
        &[
            "D", "W", "M", "ME", "1d", "1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m",
        ],
    )? {
        return early(format!("Error: Invalid interval '{}'.", o.fmt(interval)?));
    }
    let intraday = one_of(interval, &INTRADAY)?;

    let benchmarks = PyList::empty(py);
    if benchmark_symbols_yf.is_truthy()? && benchmark_symbols_yf.is_instance_of::<PyList>() {
        for bm in benchmark_symbols_yf.try_iter()? {
            let bm = bm?;
            if bm.is_instance(&o.str_t)? && bm.call_method0("strip")?.is_truthy()? {
                benchmarks.append(bm.call_method0("upper")?.call_method0("strip")?)?;
            }
        }
    }
    let fv = o.fmt(&version)?;
    if !benchmark_symbols_yf.is_truthy()? {
        log(
            py,
            "info",
            format!("Hist INFO ({fv}): No benchmark symbols provided."),
        )?;
    } else if !benchmarks.is_truthy()? {
        log(
            py,
            "warning",
            format!("Hist WARN ({fv}): Invalid benchmark_symbols_yf type. Ignoring benchmarks."),
        )?;
        has_warnings = true;
    }

    // The daily cache needs the source path to key on.
    let mut use_daily_cache = use_daily_results_cache.clone();
    if use_daily_cache.is_truthy()? && !original_csv_file_path.is_truthy()? {
        log(
            py,
            "critical",
            "CRITICAL WARNING: `calculate_historical_performance` called with `use_daily_results_cache=True` but NO `original_csv_file_path`. Disabling cache to prevent stale data.",
        )?;
        use_daily_cache = PyBool::new(py, false).to_owned().into_any();
    }

    // --- 1. Inputs ---
    let prep = m.getattr("_prepare_historical_inputs")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("preloaded_transactions_df", &all_tx),
                (
                    "original_transactions_df_for_ignored",
                    original_transactions_df_for_ignored,
                ),
                ("ignored_indices_from_load", ignored_indices_from_load),
                ("ignored_reasons_from_load", ignored_reasons_from_load),
                ("account_currency_map", account_currency_map),
                ("default_currency", default_currency),
                ("include_accounts", include_accounts),
                ("exclude_accounts", exclude_accounts),
                ("start_date", start_date),
                ("end_date", end_date),
                ("benchmark_symbols_yf", benchmarks.as_any()),
                ("display_currency", display_currency),
                ("current_hist_version", &version),
                (
                    "raw_cache_prefix",
                    &m.getattr("HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX")?,
                ),
                (
                    "daily_cache_prefix",
                    &m.getattr("DAILY_RESULTS_CACHE_PATH_PREFIX")?,
                ),
                ("user_symbol_map", user_symbol_map),
                ("user_excluded_symbols", user_excluded_symbols),
                ("original_csv_file_path", original_csv_file_path),
                ("interval", interval),
            ],
        )?),
    )?;
    let p = unpack(&prep, 17)?;
    let txs = p[0].clone();
    let ignored_reasons = &p[3];
    let symbols_all = &p[7];
    let fx_pairs = &p[8];
    let internal_to_yf = &p[9];
    let raw_cache_key = &p[13];
    let daily_cache_file = &p[14];
    let daily_cache_key = &p[15];
    let filter_desc = o.fmt(&p[16])?;
    if txs.is_none() {
        return early("Error: Failed to prepare inputs from preloaded DataFrame.".into());
    }
    status_parts.push(format!("Inputs prepared ({filter_desc})"));
    if ignored_reasons.is_truthy()? {
        status_parts.push(format!("{} tx ignored (load/prep)", ignored_reasons.len()?));
    }

    // --- The full range from the effective ledger ---
    let mut full_end_date_tx: Option<Bound<'py, PyAny>> = None;
    let ranged = (|| -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        if txs.getattr("empty")?.is_truthy()? {
            return Err(PyValueError::new_err(
                "Effective transaction DataFrame is empty.",
            ));
        }
        if !pd
            .getattr("api")?
            .getattr("types")?
            .call_method1("is_datetime64_any_dtype", (txs.get_item("Date")?,))?
            .is_truthy()?
        {
            setitem(
                &txs,
                "Date",
                crate::pyhelp::to_datetime(
                    &pd,
                    &txs.get_item("Date")?,
                    Some(&kwargs(
                        py,
                        &[("errors", &pystr(py, "coerce")), ("utc", &tru(py))],
                    )?),
                )?,
            )?;
            txs.call_method(
                "dropna",
                (),
                Some(&kwargs(
                    py,
                    &[
                        ("subset", PyList::new(py, ["Date"])?.as_any()),
                        ("inplace", &tru(py)),
                    ],
                )?),
            )?;
        }
        if txs.getattr("empty")?.is_truthy()? {
            return Err(PyValueError::new_err(
                "Effective transaction DataFrame became empty after date conversion.",
            ));
        }
        let mut full_start = b.getattr("min")?.call1((
            start_date,
            txs.get_item("Date")?
                .call_method0("min")?
                .call_method0("date")?,
        ))?;
        if intraday {
            full_start = full_start.sub(days(3)?)?;
        }
        let end_tx = txs
            .get_item("Date")?
            .call_method0("max")?
            .call_method0("date")?;
        full_end_date_tx = Some(end_tx.clone());
        let fetch_end = b.getattr("max")?.call1((end_date, &end_tx))?;
        log(
            py,
            "info",
            format!(
                "Determined full transaction range: {} to {}. Fetching data up to {} (requested start_date: {}, interval: {}).",
                o.fmt(&full_start)?,
                o.fmt(&end_tx)?,
                o.fmt(&fetch_end)?,
                o.fmt(start_date)?,
                o.fmt(interval)?
            ),
        )?;
        Ok((full_start, fetch_end))
    })();
    let (full_start_date, _) = match ranged {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Could not determine full date range from transactions: {}. Using original UI start/end.",
                    es(py, &e)
                ),
            )?;
            (start_date.clone(), end_date.clone())
        }
        Err(e) => return Err(e),
    };
    let end_tx = || {
        full_end_date_tx
            .clone()
            .ok_or_else(|| unbound("full_end_date_tx"))
    };

    // Daily views end at the last close unless a transaction is later.
    let mut clamped_end = end_date.clone();
    if one_of(interval, &["D", "1d", "W", "M", "ME"])? {
        let latest = m.getattr("get_latest_trading_date")?.call0()?;
        if end_date.gt(&latest)? && end_tx()?.le(&latest)? {
            log(
                py,
                "info",
                format!(
                    "Clamping end_date from {} to {} (Latest Trading Date).",
                    o.fmt(end_date)?,
                    o.fmt(&latest)?
                ),
            )?;
            clamped_end = latest;
        }
    }
    let _min_fetch_end = b
        .getattr("max")?
        .call1((clamped_end.add(days(1)?)?, end_tx()?.add(days(1)?)?))?;
    let _today = b
        .getattr("max")?
        .call1((end_date, m.getattr("date")?.call_method0("today")?))?;
    let mut fetch_end = b.getattr("max")?.call1((end_date, end_tx()?))?;
    if fetch_end.le(&clamped_end)? {
        fetch_end = clamped_end.add(days(1)?)?;
    }
    log(
        py,
        "info",
        format!(
            "Date Range Config: Request={}..{}, ClampedEnd={}, FetchEnd={}",
            o.fmt(start_date)?,
            o.fmt(end_date)?,
            o.fmt(&clamped_end)?,
            o.fmt(&fetch_end)?
        ),
    )?;

    let provider = m.getattr("get_shared_mdp")?.call(
        (),
        Some(&kwargs(
            py,
            &[(
                "hist_data_cache_dir_name",
                &pystr(py, "historical_data_cache"),
            )],
        )?),
    )?;
    let l1 = m.getattr("_get_or_calculate_all_daily_holdings")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("all_transactions_df", &all_tx),
                ("start_date", &full_start_date),
                ("end_date", &fetch_end),
                ("symbol_to_id", symbol_to_id),
                ("account_to_id", account_to_id),
                ("type_to_id", type_to_id),
                ("account_cash_mode_map", account_cash_mode_map),
            ],
        )?),
    )?;
    let l1 = unpack(&l1, 3)?;
    let (all_qty, all_cash, all_last) = (&l1[0], &l1[1], &l1[2]);
    if all_qty.is_none() {
        status_parts.push("L1 Cache Error".into());
        has_errors = true;
    }

    // The integrity check only needs the symbols held today and the benchmarks.
    let mut active = b.getattr("set")?.call1((&benchmarks,))?;
    let r = (|| -> PyResult<()> {
        let last_idx = all_qty.getattr("shape")?.get_item(0)?.sub(1)?;
        if last_idx.ge(0)? {
            let held = np
                .call_method1(
                    "where",
                    (np.call_method1("abs", (all_qty.get_item(&last_idx)?,))?
                        .rich_compare(1e-6, CompareOp::Gt)?,),
                )?
                .get_item(0)?;
            for sid in held.try_iter()? {
                let sid = sid?;
                if id_to_symbol.contains(&sid)? {
                    let internal = id_to_symbol.get_item(&sid)?;
                    let yf = internal_to_yf.call_method1("get", (internal,))?;
                    if yf.is_truthy()? {
                        active.call_method1("add", (yf,))?;
                    }
                }
            }
        }
        log(
            py,
            "info",
            format!(
                "Integrity check restricted to {} active symbols.",
                active.len()?
            ),
        )?;
        Ok(())
    })();
    match r {
        Ok(()) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!(
                    "Failed to determine active symbols for integrity check: {}",
                    es(py, &e)
                ),
            )?;
            active = py.None().into_bound(py);
        }
        Err(e) => return Err(e),
    }

    // --- 3. Adjusted daily prices for the whole range ---
    let raw_cache = if interval.eq("1d")? {
        use_raw_data_cache.clone()
    } else {
        PyBool::new(py, false).to_owned().into_any()
    };
    let fetched = provider.call_method(
        "get_historical_data",
        (),
        Some(&kwargs(
            py,
            &[
                ("symbols_yf", symbols_all),
                ("start_date", &full_start_date),
                ("end_date", &fetch_end),
                ("interval", &pystr(py, "1d")),
                ("use_cache", &raw_cache),
                (
                    "cache_key",
                    &pystr(
                        py,
                        &format!(
                            "{}_{}",
                            o.fmt(raw_cache_key)?,
                            o.fmt(&fetch_end.call_method0("isoformat")?)?
                        ),
                    ),
                ),
                ("integrity_check_symbols", &active),
            ],
        )?),
    )?;
    let fetched = unpack(&fetched, 2)?;
    let prices = fetched[0].clone();
    let fetch_failed_prices = fetched[1].clone();

    let dti = pd.getattr("DatetimeIndex")?;
    let not_utc = |idx: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(!idx.is_instance(&dti)? || idx.getattr("tz")?.is_none())
    };
    let keep_last = kwargs(py, &[("keep", &pystr(py, "last"))])?;

    // Intraday bars for the window, merged after the daily history.
    if intraday {
        let mut symbols = b.getattr("list")?.call1((symbols_all,))?;
        match m.getattr("_intraday_symbols_for_window")?.call1((
            symbols_all,
            &benchmarks,
            all_qty,
            &full_start_date,
            start_date,
            &fetch_end,
            id_to_symbol,
            internal_to_yf,
            &all_tx,
        )) {
            Ok(s) => symbols = s,
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "warning",
                format!("Failed to narrow intraday symbol list: {}", es(py, &e)),
            )?,
            Err(e) => return Err(e),
        }
        log(
            py,
            "info",
            format!(
                "Intraday interval '{}' detected. Fetching coverage for {}/{} active symbols...",
                o.fmt(interval)?,
                symbols.len()?,
                symbols_all.len()?
            ),
        )?;
        let r = (|| -> PyResult<()> {
            let est_today = || m.getattr("get_est_today")?.call0();
            let mut intraday_end = fetch_end.clone();
            if intraday_end.ge(est_today()?)? {
                intraday_end = est_today()?.add(days(1)?)?;
            }
            let intraday_start = if interval.contains("m")? {
                b.getattr("max")?
                    .call1((start_date, est_today()?.sub(days(59)?)?))?
            } else {
                start_date.clone()
            };
            let got = provider.call_method(
                "get_historical_data",
                (),
                Some(&kwargs(
                    py,
                    &[
                        ("symbols_yf", &symbols),
                        ("start_date", &intraday_start),
                        ("end_date", &intraday_end),
                        ("interval", interval),
                        ("use_cache", &tru(py)),
                    ],
                )?),
            )?;
            let intra = unpack(&got, 2)?.remove(0);
            if intra.is_truthy()? {
                log(
                    py,
                    "info",
                    format!(
                        "Merging {} intraday price series into main price history...",
                        intra.len()?
                    ),
                )?;
                for item in intra.call_method0("items")?.try_iter()? {
                    let kv = unpack(&item?, 2)?;
                    let (sym, intra_df) = (&kv[0], &kv[1]);
                    if intra_df.is_none() || intra_df.getattr("empty")?.is_truthy()? {
                        continue;
                    }
                    if not_utc(&intra_df.getattr("index")?)? {
                        intra_df.setattr(
                            "index",
                            crate::pyhelp::to_datetime(
                                &pd,
                                &intra_df.getattr("index")?,
                                Some(&utc),
                            )?,
                        )?;
                    }
                    if prices.contains(sym)? {
                        let daily = prices.get_item(sym)?;
                        if not_utc(&daily.getattr("index")?)? {
                            daily.setattr(
                                "index",
                                crate::pyhelp::to_datetime(
                                    &pd,
                                    &daily.getattr("index")?,
                                    Some(&utc),
                                )?,
                            )?;
                        }
                        // The daily history before the first bar, then every bar.
                        let intra_start = intra_df.getattr("index")?.call_method0("min")?;
                        let older = daily
                            .getattr("index")?
                            .rich_compare(&intra_start, CompareOp::Lt)?;
                        let mut merged = pd.call_method1(
                            "concat",
                            (PyList::new(
                                py,
                                [daily.getattr("loc")?.get_item(older)?, intra_df.clone()],
                            )?,),
                        )?;
                        let dup = merged
                            .getattr("index")?
                            .call_method("duplicated", (), Some(&keep_last))?
                            .call_method0("__invert__")?;
                        merged = merged.get_item(dup)?.call_method0("sort_index")?;
                        if !merged.getattr("empty")?.is_truthy()? {
                            merged = merged.call_method0("ffill")?.call_method0("bfill")?;
                        }
                        prices.set_item(sym, merged)?;
                    } else {
                        prices.set_item(
                            sym,
                            intra_df.call_method0("ffill")?.call_method0("bfill")?,
                        )?;
                    }
                }
            }
            Ok(())
        })();
        match r {
            Ok(()) => {}
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "error",
                format!("Failed to fetch/merge intraday data: {}", es(py, &e)),
            )?,
            Err(e) => return Err(e),
        }
    }

    log(
        py,
        "info",
        format!(
            "Fetching/Loading historical FX rates ({} pairs)...",
            fx_pairs.len()?
        ),
    )?;
    let fx_got = provider.call_method(
        "get_historical_fx_rates",
        (),
        Some(&kwargs(
            py,
            &[
                ("fx_pairs_yf", fx_pairs),
                ("start_date", &full_start_date),
                ("end_date", &fetch_end),
                ("interval", &pystr(py, "1d")),
                ("use_cache", use_raw_data_cache),
                ("cache_key", raw_cache_key),
            ],
        )?),
    )?;
    let fx_got = unpack(&fx_got, 2)?;
    let fx = fx_got[0].clone();
    let fetch_failed_fx = fx_got[1].clone();
    if intraday {
        let got = provider.call_method(
            "get_historical_fx_rates",
            (),
            Some(&kwargs(
                py,
                &[
                    ("fx_pairs_yf", fx_pairs),
                    ("start_date", start_date),
                    ("end_date", &fetch_end),
                    ("interval", interval),
                    ("use_cache", use_raw_data_cache),
                ],
            )?),
        )?;
        let intra_fx = unpack(&got, 2)?.remove(0);
        for item in intra_fx.call_method0("items")?.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let (pair, h_df) = (&kv[0], &kv[1]);
            if fx.contains(pair)? {
                let d_df = fx.get_item(pair)?;
                d_df.setattr(
                    "index",
                    crate::pyhelp::to_datetime(&pd, &d_df.getattr("index")?, Some(&utc))?,
                )?;
                let h = h_df.call_method0("copy")?;
                h.setattr(
                    "index",
                    crate::pyhelp::to_datetime(&pd, &h.getattr("index")?, Some(&utc))?,
                )?;
                if !h.getattr("empty")?.is_truthy()? {
                    let h_start = h.getattr("index")?.call_method0("min")?;
                    let older = d_df
                        .getattr("index")?
                        .rich_compare(&h_start, CompareOp::Lt)?;
                    let combined = pd
                        .call_method1("concat", (PyList::new(py, [d_df.get_item(older)?, h])?,))?
                        .call_method0("sort_index")?;
                    let dup = combined
                        .getattr("index")?
                        .call_method("duplicated", (), Some(&keep_last))?
                        .call_method0("__invert__")?;
                    fx.set_item(pair, combined.get_item(dup)?)?;
                }
            } else {
                fx.set_item(pair, h_df)?;
            }
        }
    }

    let fetch_failed = if fetch_failed_prices.is_truthy()? {
        fetch_failed_prices
    } else {
        fetch_failed_fx
    };
    if fetch_failed.is_truthy()? {
        // Partial data still draws: a missing symbol does not fail the graph.
        log(
            py,
            "warning",
            "History: Some symbols failed to fetch, but proceeding with available data.",
        )?;
        has_errors = false;
    }
    status_parts.push("Raw adjusted data loaded/fetched".into());
    if has_errors {
        status_parts.push("Fetch Error".into());
        let status = format!("Finished with Errors ({filter_desc})")
            + &if status_parts.is_empty() {
                String::new()
            } else {
                format!(" [{}]", status_parts.join("; "))
            };
        return PyTuple::new(py, [empty_frame()?, prices, fx, pystr(py, &status)]);
    }

    // Every table on a UTC index, filled back to the start of the history.
    let full_start_ts = pd.getattr("Timestamp")?.call(
        (&full_start_date,),
        Some(&kwargs(py, &[("tz", &pystr(py, "UTC"))])?),
    )?;
    for s_map in [&prices, &fx] {
        let items = b.getattr("list")?.call1((s_map.call_method0("items")?,))?;
        for item in items.try_iter()? {
            let kv = unpack(&item?, 2)?;
            let (k, mut v) = (kv[0].clone(), kv[1].clone());
            if not_utc(&v.getattr("index")?)? {
                v.setattr(
                    "index",
                    crate::pyhelp::to_datetime(&pd, &v.getattr("index")?, Some(&utc))?,
                )?;
            }
            if !v.is_none() && !v.getattr("empty")?.is_truthy()? {
                let min_idx = v.getattr("index")?.call_method0("min")?;
                if min_idx.gt(&full_start_ts)? {
                    log(
                        py,
                        "info",
                        format!(
                            "STABILIZATION: Smart-Filling {} from {} to {}",
                            o.fmt(&k)?,
                            o.fmt(&min_idx)?,
                            o.fmt(&full_start_ts)?
                        ),
                    )?;
                    v = match stabilized(&pd, &v, &full_start_ts)? {
                        Some(filled) => filled,
                        None => {
                            let new_index = pd
                                .getattr("Index")?
                                .call1((PyList::new(py, [&full_start_ts])?,))?
                                .call_method1("union", (v.getattr("index")?,))?;
                            v = v
                                .call_method1("reindex", (new_index,))?
                                .call_method0("sort_index")?;
                            v.call_method(
                                "interpolate",
                                (),
                                Some(&kwargs(py, &[("method", &pystr(py, "time"))])?),
                            )?
                            .call_method0("bfill")?
                            .call_method0("ffill")?
                        }
                    };
                    s_map.set_item(&k, v)?;
                }
            }
        }
    }

    // --- 5 & 6. The daily results ---
    let daily_file = if daily_cache_file.is_truthy()? {
        daily_cache_file.clone()
    } else {
        py.None().into_bound(py)
    };
    let use_cache_now = if use_daily_cache.is_truthy()? {
        PyBool::new(py, !intraday).to_owned().into_any()
    } else {
        use_daily_cache.clone()
    };
    let method = if calc_method.is_truthy()? {
        calc_method.clone()
    } else {
        m.getattr("HISTORICAL_CALC_METHOD")?
    };
    let daily = m.getattr("_load_or_calculate_daily_results")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("daily_results_cache_file", &daily_file),
                ("daily_results_cache_key", daily_cache_key),
                ("interval", interval),
                ("worker_signals", worker_signals),
                ("transactions_csv_file", original_csv_file_path),
                ("transactions_df_effective", &txs),
                ("start_date", start_date),
                ("end_date", &clamped_end),
                ("historical_prices_yf_unadjusted", &prices),
                ("historical_prices_yf_adjusted", &prices),
                ("historical_fx_yf", &fx),
                ("display_currency", display_currency),
                ("internal_to_yf_map", internal_to_yf),
                ("account_currency_map", account_currency_map),
                ("default_currency", default_currency),
                ("symbol_to_id", symbol_to_id),
                ("id_to_symbol", id_to_symbol),
                ("account_to_id", account_to_id),
                ("id_to_account", id_to_account),
                ("manual_overrides_dict", manual_overrides_dict),
                ("type_to_id", type_to_id),
                ("currency_to_id", currency_to_id),
                ("id_to_currency", id_to_currency),
                ("all_holdings_qty", all_qty),
                ("all_cash_balances", all_cash),
                ("all_last_prices", all_last),
                ("all_holdings_start_date", &full_start_date),
                ("use_daily_results_cache", &use_cache_now),
                ("included_accounts_list", include_accounts),
                ("current_hist_version", &version),
                ("filter_desc", &pystr(py, &filter_desc)),
                ("calc_method", &method),
                ("account_cash_mode_map", account_cash_mode_map),
            ],
        )?),
    )?;
    let daily = unpack(&daily, 3)?;
    let mut daily_df = daily[0].clone();
    let status_daily = daily[2].clone();
    let failed_text = |s: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(s.contains("Error")? || s.call_method0("lower")?.contains("failed")?)
    };
    if status_daily.is_truthy()? {
        status_parts.push(o.fmt(&status_daily.call_method0("strip")?)?);
    }
    if daily_df.is_none() || daily_df.getattr("empty")?.is_truthy()? {
        if failed_text(&status_daily)? {
            has_errors = true;
        } else {
            has_warnings = true;
        }
        status_parts.push("Daily calc failed/empty".into());
    }
    if failed_text(&status_daily)? {
        has_errors = true;
    } else if status_daily.call_method0("upper")?.contains("WARN")? {
        has_warnings = true;
    }

    // --- Benchmarks, after the daily frame so they never invalidate its cache ---
    if !daily_df.getattr("empty")?.is_truthy()? && benchmarks.is_truthy()? {
        if !daily_df.getattr("index")?.is_instance(&dti)? {
            daily_df.setattr(
                "index",
                crate::pyhelp::to_datetime(&pd, &daily_df.getattr("index")?, Some(&utc))?,
            )?;
        }
        if daily_df.getattr("index")?.getattr("tz")?.is_none() {
            daily_df.setattr(
                "index",
                daily_df
                    .getattr("index")?
                    .call_method1("tz_localize", ("UTC",))?,
            )?;
        }
        if daily_df
            .getattr("index")?
            .call_method0("duplicated")?
            .call_method0("any")?
            .is_truthy()?
        {
            let unique_rows = daily_df
                .get_item(
                    daily_df
                        .getattr("index")?
                        .call_method0("duplicated")?
                        .call_method0("__invert__")?,
                )?
                .len()?;
            log(
                py,
                "info",
                format!(
                    "Deduplicating portfolio daily_df.index: {} -> {unique_rows}",
                    daily_df.len()?
                ),
            )?;
            daily_df = daily_df.get_item(
                daily_df
                    .getattr("index")?
                    .call_method("duplicated", (), Some(&keep_last))?
                    .call_method0("__invert__")?,
            )?;
        }
        let mut bench_start = full_start_date.clone();
        if intraday {
            bench_start = b
                .getattr("max")?
                .call1((&full_start_date, start_date.sub(days(5)?)?))?;
        }
        let got = provider.call_method(
            "get_historical_data",
            (),
            Some(&kwargs(
                py,
                &[
                    ("symbols_yf", benchmarks.as_any()),
                    ("start_date", &bench_start),
                    ("end_date", &fetch_end),
                    ("use_cache", &tru(py)),
                    ("interval", interval),
                ],
            )?),
        )?;
        let bench = unpack(&got, 2)?.remove(0);
        if bench.is_truthy()? {
            log(
                py,
                "info",
                format!(
                    "Fetched {} benchmarks. Keys: {}",
                    bench.len()?,
                    o.fmt(&b.getattr("list")?.call1((bench.call_method0("keys")?,))?)?
                ),
            )?;
            // The original's `bm_series` outlives its loop iteration.
            let mut bm_series: Option<Bound<'py, PyAny>> = None;
            for bm in benchmarks.iter() {
                let fb = o.fmt(&bm)?;
                let price_col = format!("{fb} Price");
                if !bench.contains(&bm)? {
                    setitem(&daily_df, &price_col, f64::NAN)?;
                    continue;
                }
                let bm_df = bench.get_item(&bm)?;
                if bm_df.is_none() || bm_df.getattr("empty")?.is_truthy()? {
                    log(
                        py,
                        "warning",
                        format!("No price data found for benchmark {fb}"),
                    )?;
                    setitem(&daily_df, &price_col, f64::NAN)?;
                    continue;
                }
                let columns = bm_df.getattr("columns")?;
                let mut col = None;
                for c in ["price", "Adj Close", "Close"] {
                    if columns.contains(c)? {
                        col = Some(c);
                        break;
                    }
                }
                if let Some(c) = col {
                    log(
                        py,
                        "info",
                        format!(
                            "Benchmark {fb}: Using col '{c}', count={}, first={}",
                            bm_df.len()?,
                            o.fmt(&bm_df.getattr("index")?.get_item(0)?)?
                        ),
                    )?;
                    bm_series = Some(bm_df.get_item(c)?.call_method0("copy")?);
                }
                let mut series = bm_series.clone().ok_or_else(|| unbound("bm_series"))?;
                series.setattr("name", &price_col)?;
                if not_utc(&series.getattr("index")?)? {
                    series.setattr(
                        "index",
                        crate::pyhelp::to_datetime(&pd, &series.getattr("index")?, Some(&utc))?,
                    )?;
                }
                if series
                    .getattr("index")?
                    .call_method0("duplicated")?
                    .call_method0("any")?
                    .is_truthy()?
                {
                    let unique = series
                        .get_item(
                            series
                                .getattr("index")?
                                .call_method0("duplicated")?
                                .call_method0("__invert__")?,
                        )?
                        .len()?;
                    log(
                        py,
                        "info",
                        format!(
                            "Deduplicating benchmark {fb} index: {} -> {unique}",
                            series.len()?
                        ),
                    )?;
                    series = series.get_item(
                        series
                            .getattr("index")?
                            .call_method("duplicated", (), Some(&keep_last))?
                            .call_method0("__invert__")?,
                    )?;
                }
                bm_series = Some(series.clone());
                let aligned = (|| -> PyResult<()> {
                    // 120h bridges an MLK-day weekend.
                    let tol = pd.getattr("Timedelta")?.call1(("120h",))?;
                    let s_tz = series.getattr("index")?.getattr("tz")?;
                    let d_tz = daily_df.getattr("index")?.getattr("tz")?;
                    if s_tz.is_none() && !d_tz.is_none() {
                        series.setattr(
                            "index",
                            series
                                .getattr("index")?
                                .call_method1("tz_localize", (&d_tz,))?,
                        )?;
                    } else if !s_tz.is_none() && d_tz.is_none() {
                        series.setattr(
                            "index",
                            series
                                .getattr("index")?
                                .call_method1("tz_localize", (py.None(),))?,
                        )?;
                    }
                    let mut is_intra = false;
                    for x in ["m", "h", "min"] {
                        if interval.contains(x)? {
                            is_intra = true;
                            break;
                        }
                    }
                    let aligned = if is_intra {
                        series
                            .call_method1("reindex", (daily_df.getattr("index")?,))?
                            .call_method(
                                "interpolate",
                                (),
                                Some(&kwargs(py, &[("method", &pystr(py, "time"))])?),
                            )?
                            .call_method0("ffill")?
                            .call_method0("bfill")?
                    } else {
                        series.call_method(
                            "reindex",
                            (daily_df.getattr("index")?,),
                            Some(&kwargs(
                                py,
                                &[("method", &pystr(py, "ffill")), ("tolerance", &tol)],
                            )?),
                        )?
                    };
                    setitem(&daily_df, &price_col, aligned)?;
                    let on_day = daily_df
                        .getattr("index")?
                        .getattr("date")?
                        .rich_compare(m.getattr("date")?.call1((2026, 1, 20))?, CompareOp::Eq)?;
                    let tuesday = daily_df
                        .getattr("loc")?
                        .get_item(PyTuple::new(py, [on_day, pystr(py, &price_col)])?)?
                        .call_method0("notna")?
                        .call_method0("sum")?;
                    log(
                        py,
                        "info",
                        format!(
                            "Benchmark {fb}: Aligned. Tuesday points: {}",
                            o.fmt(&tuesday)?
                        ),
                    )?;
                    Ok(())
                })();
                match aligned {
                    Ok(()) => {}
                    Err(e) if e.is_instance_of::<PyException>(py) => {
                        log(
                            py,
                            "warning",
                            format!(
                                "Benchmark alignment failed for {fb}: {}. Falling back to strict join.",
                                es(py, &e)
                            ),
                        )?;
                        setitem(&daily_df, &price_col, &series)?;
                    }
                    Err(e) => return Err(e),
                }
            }
        }
    }

    // --- 7. Accumulation, resampling and the requested range ---
    let acc = m
        .getattr("_calculate_accumulated_gains_and_resample")?
        .call(
            (),
            Some(&kwargs(
                py,
                &[
                    ("daily_df", &daily_df),
                    ("benchmark_symbols_yf", benchmarks.as_any()),
                    ("interval", interval),
                    ("start_date_filter", start_date),
                    ("end_date_filter", &clamped_end),
                ],
            )?),
        )?;
    let acc = unpack(&acc, 3)?;
    let daily_df = acc[0].clone();
    let twr = acc[1].clone();
    if acc[2].is_truthy()? {
        status_parts.push(o.fmt(&acc[2].call_method0("strip")?)?);
    }

    let end_time = m.getattr("time")?.call_method0("time")?;
    log(
        py,
        "info",
        format!(
            "Total Historical Calc Time: {} seconds",
            o.fmt_spec(&end_time.sub(&start_time)?, ".2f")?
        ),
    )?;
    let prefix = if has_errors {
        "Finished with Errors"
    } else if has_warnings {
        "Finished with Warnings"
    } else {
        "Success"
    };
    let mut status = format!("{prefix} ({filter_desc})");
    if !status_parts.is_empty() {
        status += &format!(" [{}]", status_parts.join("; "));
    }
    status += &if o.notna(&twr)? {
        format!("|||TWR_FACTOR:{}", o.fmt_spec(&twr, ".6f")?)
    } else {
        "|||TWR_FACTOR:NaN".into()
    };
    PyTuple::new(py, [daily_df, prices, fx, pystr(py, &status)])
}
