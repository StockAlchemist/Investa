//! The first block of `finutils`: the periods-per-year inference, the file
//! hash (SQLite sidecars included), the currency symbol, the dividend details
//! and robust yield, the cash and external-flow classifiers, the NPV and IRR
//! glue around the Rust solver, and the transaction-column readers the cash
//! flow builders share.
//!
//! Ported line for line from `src/finutils.py` (frozen in
//! `tests/reference/finutils_p16.py`). Strings are compared with Python's own
//! methods (`.lower()`, `.upper()`, `.strip()`), files are read through
//! Python's `open` and hashed by `hashlib`, and the dividend arithmetic, which
//! the original does on Python floats after `float()`, is native. Constants and
//! siblings are looked up on `finutils` at call time.

use pyo3::exceptions::{
    PyException, PyFileNotFoundError, PyOSError, PyPermissionError, PyTypeError, PyValueError,
};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};

use crate::fr::{builtin, fl, int, none, or_else, pybool};
use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, log_exc, pystr, unpack};

pub(crate) fn fu(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("finutils")
}

fn is_value_or_type(py: Python<'_>, e: &PyErr) -> bool {
    e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py)
}

#[pyfunction]
pub fn fu_infer_periods_per_year<'py>(
    py: Python<'py>,
    index: &Bound<'py, PyAny>,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = fu(py)?.getattr("pd")?;
    if index.is_instance(&pd.getattr("DatetimeIndex")?)? && index.len()? > 1 {
        let span_years = index
            .get_item(-1)?
            .sub(index.get_item(0)?)?
            .getattr("days")?
            .div(365.25)?;
        if span_years.gt(0)? {
            return int(py, index.len()? as i64)?.div(span_years);
        }
    }
    builtin(py, "float")?.call1((default,))
}

#[pyfunction]
pub fn fu_get_file_hash<'py>(
    py: Python<'py>,
    filepath: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let o = Ops::new(py)?;
    let hasher = m.getattr("hashlib")?.call_method0("sha256")?;
    let chunk_size = m.getattr("HASH_CHUNK_SIZE")?;
    let open = builtin(py, "open")?;

    // Hash one file into the main hasher; False when it has disappeared.
    let update_with_file = |path: &Bound<'py, PyAny>| -> PyResult<bool> {
        let file = match open.call1((path, "rb")) {
            Ok(f) => f,
            Err(e) if e.is_instance_of::<PyFileNotFoundError>(py) => return Ok(false),
            Err(e) => return Err(e),
        };
        let done = with1(py, &file, |f| {
            loop {
                let chunk = f.call_method1("read", (&chunk_size,))?;
                if !chunk.is_truthy()? {
                    return Ok(());
                }
                hasher.call_method1("update", (chunk,))?;
            }
        });
        match done {
            Ok(_) => Ok(true),
            Err(e) if e.is_instance_of::<PyFileNotFoundError>(py) => Ok(false),
            Err(e) => Err(e),
        }
    };

    let body = || -> PyResult<Bound<'py, PyAny>> {
        // 1. Hash the main file
        if !update_with_file(filepath)? {
            log(
                py,
                "warning",
                format!(
                    "Warning: Main file not found for hashing: {}",
                    o.fmt(filepath)?
                ),
            )?;
            return m.getattr("HASH_ERROR_NOT_FOUND");
        }
        // 2. SQLite companion files
        let lower = filepath.call_method0("lower")?;
        let suffixes = PyTuple::new(py, [".db", ".sqlite", ".sqlite3"])?;
        if lower.call_method1("endswith", (suffixes,))?.is_truthy()? {
            let exists = py.import("os")?.getattr("path")?.getattr("exists")?;
            for suffix in ["-wal", "-shm"] {
                let companion = filepath.add(suffix)?;
                if exists.call1((&companion,))?.is_truthy()? {
                    update_with_file(&companion)?;
                }
            }
        }
        hasher.call_method0("hexdigest")
    };
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyPermissionError>(py) => {
            log(
                py,
                "error",
                format!(
                    "Permission denied accessing file {} for hashing.",
                    o.fmt(filepath)?
                ),
            )?;
            m.getattr("HASH_ERROR_PERMISSION")
        }
        Err(e) if e.is_instance_of::<PyOSError>(py) => {
            log(
                py,
                "error",
                format!(
                    "I/O error hashing file {}: {}",
                    o.fmt(filepath)?,
                    es(py, &e)
                ),
            )?;
            m.getattr("HASH_ERROR_IO")
        }
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log_exc(
                py,
                "exception",
                format!("Unexpected error hashing file {}", o.fmt(filepath)?),
                &e,
            )?;
            m.getattr("HASH_ERROR_UNEXPECTED")
        }
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fu_currency_symbol_from_code<'py>(
    py: Python<'py>,
    currency_code: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !currency_code.is_truthy()? || !currency_code.is_instance_of::<PyString>() {
        return Ok(pystr(py, ""));
    }
    let upper = currency_code.call_method0("upper")?;
    fu(py)?
        .getattr("CURRENCY_SYMBOLS")?
        .call_method1("get", (&upper, &upper))
}

/// `float(info.get(k, 0.0) or 0.0)`
fn float_or_zero(info: &Bound<'_, PyAny>, k: &str) -> PyResult<f64> {
    let py = info.py();
    let v = or_else(info.call_method1("get", (k, 0.0))?, || Ok(fl(py, 0.0)))?;
    builtin(py, "float")?.call1((v,))?.extract()
}

/// The same, with `(ValueError, TypeError)` read as 0.0.
fn float_or_zero_guarded(info: &Bound<'_, PyAny>, k: &str) -> PyResult<f64> {
    match float_or_zero(info, k) {
        Ok(v) => Ok(v),
        Err(e) if is_value_or_type(info.py(), &e) => Ok(0.0),
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fu_dividend_details<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    if !info.is_truthy()? {
        out.set_item("frequency_months", 3)?;
        out.set_item("indicated_annual_rate", 0.0)?;
        return Ok(out);
    }
    let last_div_val = float_or_zero(info, "lastDividendValue")?;
    let mut div_rate = float_or_zero_guarded(info, "dividendRate")?;
    let trailing_rate = float_or_zero_guarded(info, "trailingAnnualDividendRate")?;
    if div_rate <= 0.0 {
        div_rate = if trailing_rate != 0.0 {
            trailing_rate
        } else {
            0.0
        };
    }
    if last_div_val <= 0.0 && div_rate <= 0.0 {
        out.set_item("frequency_months", 3)?;
        out.set_item("indicated_annual_rate", 0.0)?;
        return Ok(out);
    }

    let mut freq = 4i64;
    let mut consistent = false;
    if last_div_val > 0.0 && div_rate > 0.0 {
        let ratio = div_rate / last_div_val;
        // The annual-rate / last-paid ratio maps to a payment cadence.
        for (lo, hi, f) in [
            (10.0, 14.5, 12),
            (3.4, 5.8, 4),
            (1.4, 2.6, 2),
            (0.7, 1.3, 1),
        ] {
            if lo <= ratio && ratio <= hi {
                freq = f;
                consistent = true;
                break;
            }
        }
    }

    // Indicated (forward-looking) annual rate.
    let indicated_rate = if div_rate > 0.0 && (!consistent || last_div_val <= 0.0) {
        div_rate
    } else if last_div_val > 0.0 {
        last_div_val * freq as f64
    } else {
        div_rate
    };
    out.set_item("frequency_months", 12i64.div_euclid(freq))?;
    out.set_item("indicated_annual_rate", indicated_rate)?;
    out.set_item("last_dividend_value", last_div_val)?;
    Ok(out)
}

/// `float(x)` with `(TypeError, ValueError)` as None.
fn try_float(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    match builtin(py, "float")?.call1((x,)) {
        Ok(v) => Ok(Some(v.extract()?)),
        Err(e) if is_value_or_type(py, &e) => Ok(None),
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fu_robust_dividend_yield<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !info.is_truthy()? {
        return Ok(none(py));
    }
    let m = fu(py)?;
    let get = |k: &str| info.call_method1("get", (k,));

    // Funds/ETFs sometimes carry an explicit 'yield' fraction.
    let explicit = get("yield")?;
    if !explicit.is_none()
        && let Some(v) = try_float(py, &explicit)?
        && v > 0.0
    {
        return Ok(fl(py, if v < 1.0 { v } else { v / 100.0 }));
    }

    // 1. Indicated annual rate / current price.
    let indicated = or_else(
        m.getattr("get_dividend_details")?
            .call1((info,))?
            .call_method1("get", ("indicated_annual_rate",))?,
        || Ok(fl(py, 0.0)),
    )?;
    let price = or_else(get("currentPrice")?, || get("regularMarketPrice"))?;
    let price = try_float(py, &price)?.unwrap_or(0.0);
    if indicated.gt(0)? && price > 0.0 {
        return indicated.div(price);
    }

    // 2. Trailing annual dividend yield (already a fraction).
    if let Some(ty) = try_float(py, &get("trailingAnnualDividendYield")?)?
        && ty > 0.0
    {
        return Ok(fl(py, ty));
    }

    // 3. Last resort: raw Yahoo dividendYield, encoding inferred from magnitude.
    if let Some(rv) = try_float(py, &get("dividendYield")?)?
        && rv > 0.0
    {
        let cutoff: f64 = m.getattr("_RAW_YIELD_FRACTION_CUTOFF")?.extract()?;
        return Ok(fl(py, if rv > cutoff { rv / 100.0 } else { rv }));
    }
    Ok(none(py))
}

#[pyfunction]
pub fn fu_is_cash_symbol(py: Python<'_>, symbol: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !symbol.is_instance_of::<PyString>() {
        return Ok(false);
    }
    let lower = symbol.call_method0("lower")?;
    let cash = fu(py)?.getattr("CASH_SYMBOL_CSV")?.call_method0("lower")?;
    Ok(lower.call_method1("startswith", (cash,))?.is_truthy()?
        || lower.call_method1("startswith", ("cash (",))?.is_truthy()?)
}

#[pyfunction]
#[pyo3(signature = (symbol, tx_type, note=None))]
pub fn fu_is_external_flow_row(
    py: Python<'_>,
    symbol: &Bound<'_, PyAny>,
    tx_type: &Bound<'_, PyAny>,
    note: Option<&Bound<'_, PyAny>>,
) -> PyResult<bool> {
    let _ = note;
    if !symbol.is_instance_of::<PyString>()
        || symbol
            .call_method0("upper")?
            .ne(fu(py)?.getattr("CASH_SYMBOL_CSV")?.call_method0("upper")?)?
    {
        return Ok(false);
    }
    if !tx_type.is_instance_of::<PyString>() {
        return Ok(false);
    }
    let t = tx_type.call_method0("strip")?.call_method0("lower")?;
    Ok(t.eq("deposit")? || t.eq("withdrawal")?)
}

#[pyfunction]
pub fn fu_external_flow_mask<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    if df.getattr("empty")?.is_truthy()? {
        return pd.getattr("Series")?.call(
            (PyList::empty(py),),
            Some(&kwargs(
                py,
                &[
                    ("dtype", &builtin(py, "bool")?),
                    ("index", &df.getattr("index")?),
                ],
            )?),
        );
    }
    let str_t = builtin(py, "str")?;
    let symbol_upper = df
        .get_item("Symbol")?
        .call_method1("astype", (&str_t,))?
        .getattr("str")?
        .call_method0("upper")?;
    let type_lower = df
        .get_item("Type")?
        .call_method1("astype", (&str_t,))?
        .getattr("str")?
        .call_method0("lower")?
        .getattr("str")?
        .call_method0("strip")?;
    let cash = fu(py)?.getattr("CASH_SYMBOL_CSV")?.call_method0("upper")?;
    symbol_upper
        .rich_compare(cash, pyo3::basic::CompareOp::Eq)?
        .bitand(type_lower.call_method1("isin", (PyList::new(py, ["deposit", "withdrawal"])?,))?)
}

#[pyfunction]
pub fn fu_irr_date_ticks<'py>(
    py: Python<'py>,
    dates: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let np = m.getattr("np")?;
    let int64 = np.getattr("int64")?;
    let date_t = py.import("datetime")?.getattr("date")?;
    let mut all_dates = true;
    for d in dates.try_iter()? {
        if !d?.get_type().is(&date_t) {
            all_dates = false;
            break;
        }
    }
    if all_dates {
        let day_us: i64 = m.getattr("_DAY_US")?.extract()?;
        let ticks = PyList::empty(py);
        for d in dates.try_iter()? {
            let ordinal: i64 = d?.call_method0("toordinal")?.extract()?;
            ticks.append(ordinal * day_us)?;
        }
        return np.call_method("array", (ticks,), Some(&kwargs(py, &[("dtype", &int64)])?));
    }
    let attempt = || -> PyResult<Bound<'py, PyAny>> {
        let d0 = dates.get_item(0)?;
        let one_us = m.getattr("_ONE_US")?;
        let ticks = PyList::empty(py);
        for d in dates.try_iter()? {
            ticks.append(d?.sub(&d0)?.floor_div(&one_us)?)?;
        }
        np.call_method("array", (ticks,), Some(&kwargs(py, &[("dtype", &int64)])?))
    };
    match attempt() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => Ok(none(py)),
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fu_npv<'py>(
    py: Python<'py>,
    rate: &Bound<'py, PyAny>,
    dates: &Bound<'py, PyAny>,
    cash_flows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let np = m.getattr("np")?;
    let nan = np.getattr("nan")?;
    let numeric = rate.is_instance_of::<PyInt>() || rate.is_instance_of::<PyFloat>();
    if !numeric || !np.call_method1("isfinite", (rate,))?.is_truthy()? {
        return Ok(nan);
    }
    if dates.len()? != cash_flows.len()? {
        return Err(PyValueError::new_err(
            "The lengths of dates and cash_flows do not match.",
        ));
    }
    if dates.len()? == 0 {
        return Ok(fl(py, 0.0));
    }
    if fl(py, 1.0).add(rate)?.le(1e-18)? {
        return Ok(nan);
    }
    let prepared = (|| -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let start = dates.get_item(0)?;
        let days = PyList::empty(py);
        for d in dates.try_iter()? {
            days.append(d?.sub(&start)?.getattr("days")?)?;
        }
        let days = np.call_method(
            "array",
            (days,),
            Some(&kwargs(py, &[("dtype", &np.getattr("int64")?)])?),
        )?;
        let flows = np.call_method(
            "asarray",
            (cash_flows,),
            Some(&kwargs(py, &[("dtype", &np.getattr("float64")?)])?),
        )?;
        Ok((days, flows))
    })();
    let (days, flows) = match prepared {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => return Ok(nan),
        Err(e) => return Err(e),
    };
    let r = m.getattr("investa_core")?.getattr("npv")?.call1((
        builtin(py, "float")?.call1((rate,))?,
        &days,
        flows,
    ))?;
    let r = unpack(&r, 2)?;
    if r[1].ge(0)? {
        let o = Ops::new(py)?;
        log(
            py,
            "warning",
            format!(
                "NPV Calc Error: Invalid time delta ({}) at index {}.",
                o.fmt(&days.get_item(&r[1])?.div(365.0)?)?,
                o.fmt(&r[1])?
            ),
        )?;
    }
    Ok(r[0].clone())
}

/// `f"{d.strftime('%Y-%m-%d')}: {f:,.2f}"`
fn readable(o: &Ops<'_>, d: &Bound<'_, PyAny>, f: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(format!(
        "{}: {}",
        d.call_method1("strftime", ("%Y-%m-%d",))?.str()?,
        o.fmt_spec(f, ",.2f")?
    ))
}

#[pyfunction]
pub fn fu_calculate_irr<'py>(
    py: Python<'py>,
    dates: &Bound<'py, PyAny>,
    cash_flows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let np = m.getattr("np")?;
    let nan = np.getattr("nan")?;
    let (nd, nc) = (dates.len()?, cash_flows.len()?);
    if nd < 2 || nc < 2 || nd != nc {
        log(py, "debug", "DEBUG IRR: Fail - Length mismatch or < 2")?;
        return Ok(nan);
    }
    for cf in cash_flows.try_iter()? {
        let cf = cf?;
        if !(cf.is_instance_of::<PyInt>() || cf.is_instance_of::<PyFloat>()) {
            log(py, "debug", "DEBUG IRR: Fail - Non-numeric cash flows")?;
            return Ok(nan);
        }
    }
    let ticks = m.getattr("_irr_date_ticks")?.call1((dates,))?;
    if ticks.is_none() {
        log(
            py,
            "debug",
            "DEBUG IRR: Fail - dates are not mutually comparable",
        )?;
        return Ok(nan);
    }
    let flows = np.call_method(
        "asarray",
        (cash_flows,),
        Some(&kwargs(py, &[("dtype", &np.getattr("float64")?)])?),
    )?;
    let float_t = py.get_type::<PyFloat>();
    let kinds = PyList::empty(py);
    for cf in cash_flows.try_iter()? {
        let cf = cf?;
        kinds.append(if cf.get_type().is(&float_t) {
            0u8
        } else if cf.is_instance_of::<PyInt>() {
            1
        } else {
            2
        })?;
    }
    let kinds = np.call_method(
        "array",
        (kinds,),
        Some(&kwargs(py, &[("dtype", &np.getattr("uint8")?)])?),
    )?;
    let r = m
        .getattr("investa_core")?
        .getattr("calculate_irr")?
        .call1((ticks, flows, kinds))?;
    let r = unpack(&r, 6)?;
    let (rate, status, first, last, scaled, unsorted) = (&r[0], &r[1], &r[2], &r[3], &r[4], &r[5]);
    if unsorted.is_truthy()? {
        log(
            py,
            "warning",
            "NPV Calc Error: cash-flow dates are not sorted; every NPV dated before the first flow returned NaN.",
        )?;
    }
    let ok = m.getattr("_IRR_OK")?;
    let failed = m.getattr("_IRR_SOLVER_FAILED")?;
    let o = Ops::new(py)?;
    let solver_failed = status.eq(&failed)?;
    if solver_failed || (status.eq(&ok)? && rate.gt(10.0)?) {
        let window = builtin(py, "slice")?.call1((first, last.add(1)?))?;
        let used_dates = builtin(py, "list")?.call1((dates.get_item(&window)?,))?;
        let float = builtin(py, "float")?;
        let mut used: Vec<f64> = Vec::new();
        for cf in cash_flows.get_item(&window)?.try_iter()? {
            used.push(float.call1((cf?,))?.extract()?);
        }
        if scaled.is_truthy()? {
            // max(): the first element, replaced only by a strictly larger one.
            let mut scale = used[0].abs();
            for v in &used[1..] {
                if v.abs() > scale {
                    scale = v.abs();
                }
            }
            used = used.iter().map(|cf| cf / scale).collect();
        }
        let used_py = PyList::new(py, &used)?;
        if solver_failed {
            let mut flows_text = Vec::new();
            for (d, f) in used_dates.try_iter()?.zip(used_py.iter()) {
                flows_text.push(readable(&o, &d?, &f)?);
            }
            let flow_str = if flows_text.len() > 100 {
                format!(
                    "[{} ... {}] ({} flows)",
                    flows_text[..10].join(", "),
                    flows_text[flows_text.len() - 10..].join(", "),
                    flows_text.len()
                )
            } else {
                format!("[{}]", flows_text.join(", "))
            };
            log(
                py,
                "warning",
                format!("IRR Calculation Failed. Flows used: {flow_str}"),
            )?;
            return Ok(nan);
        }
        log(
            py,
            "warning",
            format!(
                "Extreme IRR detected ({}%). Investigating flows...",
                o.fmt_spec(&rate.mul(100)?, ".1f")?
            ),
        )?;
        let pairs =
            builtin(py, "list")?.call1((builtin(py, "zip")?.call1((&used_dates, &used_py))?,))?;
        let key = PyCFunction::new_closure(py, None, None, |args, _| -> PyResult<Py<PyAny>> {
            let py = args.py();
            Ok(builtin(py, "abs")?
                .call1((args.get_item(0)?.get_item(1)?,))?
                .unbind())
        })?;
        let top = builtin(py, "sorted")?
            .call(
                (pairs,),
                Some(&kwargs(
                    py,
                    &[("key", key.as_any()), ("reverse", &pybool(py, true))],
                )?),
            )?
            .get_item(builtin(py, "slice")?.call1((none(py), 10))?)?;
        let mut top_text = Vec::new();
        for p in top.try_iter()? {
            let df = unpack(&p?, 2)?;
            top_text.push(readable(&o, &df[0], &df[1])?);
        }
        log(
            py,
            "warning",
            format!("Top Flows contributing to IRR: [{}]", top_text.join(", ")),
        )?;
    } else if !status.eq(&ok)? {
        log(
            py,
            "debug",
            format!(
                "DEBUG IRR: no IRR (status {}) for {} flows",
                o.fmt(status)?,
                nc
            ),
        )?;
    }
    Ok(rate.clone())
}

#[pyfunction]
#[pyo3(signature = (df, name, with_types=None))]
pub fn fu_numeric_column<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    with_types: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let np = fu(py)?.getattr("np")?;
    let pd = py.import("pandas")?;
    let n = df.len()?;
    let float64 = np.getattr("float64")?;
    let (values, numpy_typed) = if !df.getattr("columns")?.contains(name)? {
        (
            np.call_method1("full", (n, np.getattr("nan")?))?,
            np.call_method(
                "zeros",
                (n,),
                Some(&kwargs(py, &[("dtype", &builtin(py, "bool")?)])?),
            )?,
        )
    } else {
        let col = df.get_item(name)?;
        let dtype = col.getattr("dtype")?;
        let zeros = || {
            np.call_method(
                "zeros",
                (n,),
                Some(&kwargs(py, &[("dtype", &builtin(py, "bool")?)])?),
            )
        };
        if dtype.is_instance(&np.getattr("dtype")?)?
            && pystr(py, "fiu").contains(dtype.getattr("kind")?)?
        {
            (
                col.call_method("to_numpy", (), Some(&kwargs(py, &[("dtype", &float64)])?))?,
                zeros()?,
            )
        } else {
            let generic = np.getattr("generic")?;
            let to_numeric = pd.getattr("to_numeric")?;
            let isna = pd.getattr("isna")?;
            let float = builtin(py, "float")?;
            let mut vals = Vec::with_capacity(n);
            let mut typed = Vec::with_capacity(n);
            for v in col.call_method0("tolist")?.try_iter()? {
                let v = v?;
                let v = if crate::pyhelp::numeric_passthrough(&v) {
                    v
                } else {
                    to_numeric.call(
                        (v,),
                        Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
                    )?
                };
                vals.push(if v.is_none() || isna.call1((&v,))?.is_truthy()? {
                    f64::NAN
                } else {
                    float.call1((&v,))?.extract::<f64>()?
                });
                typed.push(v.is_instance(&generic)?);
            }
            (
                np.call_method(
                    "array",
                    (PyList::new(py, vals)?,),
                    Some(&kwargs(py, &[("dtype", &float64)])?),
                )?,
                np.call_method(
                    "array",
                    (PyList::new(py, typed)?,),
                    Some(&kwargs(py, &[("dtype", &builtin(py, "bool")?)])?),
                )?,
            )
        }
    };
    let with_types = match with_types {
        Some(w) => w.is_truthy()?,
        None => false,
    };
    if with_types {
        return Ok(PyTuple::new(py, [values, numpy_typed])?.into_any());
    }
    Ok(values)
}

#[pyfunction]
pub fn fu_commission_column<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let np = m.getattr("np")?;
    let values = m.getattr("_numeric_column")?.call1((df, "Commission"))?;
    np.call_method1(
        "where",
        (np.call_method1("isnan", (&values,))?, 0.0, &values),
    )
}

#[pyfunction]
#[pyo3(signature = (values, i, default=None))]
pub fn fu_cell_text<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    i: &Bound<'py, PyAny>,
    default: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if values.is_none() {
        return Ok(default.cloned().unwrap_or_else(|| pystr(py, "")));
    }
    Ok(values.get_item(i)?.str()?.into_any())
}
