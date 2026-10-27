//! The FX and historical-price lookups of `finutils`: the current conversion
//! rate (cross rates via USD, the static fallback), the per-frame price-table
//! cache and its builder, the historical price (the Rust table's binary
//! search where it applies, pandas' `asof` otherwise, with the backfill rule),
//! and the historical USD bridge.
//!
//! Ported line for line from `src/finutils.py` (frozen in
//! `tests/reference/finutils_p16.py`). The price table was already Rust
//! (`investa_core.PriceTable`); the cache stays the module's `_PRICE_TABLES`
//! dict, keyed by `id(frame)` with a weakref that evicts the entry when the
//! frame dies. Rates are combined with Python's own operations, and day
//! ticks are Python integers (a far date's nanoseconds overflow 64 bits).

use pyo3::exceptions::{
    PyException, PyOverflowError, PyTypeError, PyValueError, PyZeroDivisionError,
};
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyString, PyTuple};

use crate::fr::{builtin, none};
use crate::fu::fu;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log};

/// `"USDTHB=X" if code == "THB" else f"{code}=X"`
fn pair_for<'py>(o: &Ops<'py>, upper: &Bound<'py, PyAny>) -> PyResult<String> {
    if upper.eq("THB")? {
        Ok("USDTHB=X".to_owned())
    } else {
        Ok(format!("{}=X", o.fmt(upper)?))
    }
}

/// A valid currency code: truthy and a string.
fn code_ok(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(x.is_truthy()? && x.is_instance_of::<PyString>())
}

#[pyfunction]
pub fn fu_conversion_rate<'py>(
    py: Python<'py>,
    from_curr: &Bound<'py, PyAny>,
    to_curr: &Bound<'py, PyAny>,
    fx_rates: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let nan = m.getattr("np")?.getattr("nan")?;
    if !code_ok(from_curr)? || !code_ok(to_curr)? {
        log(
            py,
            "debug",
            "get_conversion_rate: Invalid from_curr or to_curr input.",
        )?;
        return Ok(nan);
    }
    if from_curr.eq(to_curr)? {
        return Ok(1.0f64.into_pyobject(py)?.into_any());
    }
    if !fx_rates.is_instance_of::<pyo3::types::PyDict>() {
        log(
            py,
            "debug",
            "Warning: get_conversion_rate received invalid fx_rates type. Returning NaN",
        )?;
        return Ok(nan);
    }
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let config = m.getattr("config")?;
    let has_fallback = config.hasattr("STATIC_FX_FALLBACK")?;

    // The rate per USD of one side: the code, else its Yahoo pair, else the
    // static fallback; USD itself is 1.0.
    let per_usd = |upper: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let mut rate = fx_rates.call_method1("get", (upper,))?;
        if rate.is_none() {
            rate = fx_rates.call_method1("get", (pair_for(&o, upper)?,))?;
        }
        if pd.call_method1("isna", (&rate,))?.is_truthy()? && has_fallback {
            rate = config
                .getattr("STATIC_FX_FALLBACK")?
                .call_method1("get", (upper,))?;
            if pd.call_method1("notna", (&rate,))?.is_truthy()? {
                log(
                    py,
                    "info",
                    format!("FX Fallback: Using STATIC rate for {}", o.fmt(upper)?),
                )?;
            }
        }
        let name = o.fmt(upper)?;
        log(
            py,
            "debug",
            format!(
                "get_conversion_rate: Looking up {name}/USD (key '{name}'): {}",
                o.fmt(&rate)?
            ),
        )?;
        if upper.eq("USD")? {
            rate = 1.0f64.into_pyobject(py)?.into_any();
        }
        Ok(rate)
    };
    let from_upper = from_curr.call_method0("upper")?;
    let to_upper = to_curr.call_method0("upper")?;
    let rate_a = per_usd(&from_upper)?;
    let rate_b = per_usd(&to_upper)?;

    let mut rate_b_per_a = nan.clone();
    if pd.call_method1("notna", (&rate_a,))?.is_truthy()?
        && pd.call_method1("notna", (&rate_b,))?.is_truthy()?
        && builtin(py, "abs")?.call1((&rate_a,))?.gt(1e-9)?
    {
        // The division and its debug log share the original's `try`.
        let attempt = || -> PyResult<Bound<'py, PyAny>> {
            let r = rate_b.div(&rate_a)?;
            log(
                py,
                "debug",
                format!(
                    "DEBUG get_conv_rate: {}/{} = {} / {} = {}",
                    o.fmt(to_curr)?,
                    o.fmt(from_curr)?,
                    o.fmt(&rate_b)?,
                    o.fmt(&rate_a)?,
                    o.fmt(&r)?
                ),
            )?;
            Ok(r)
        };
        match attempt() {
            Ok(r) => rate_b_per_a = r,
            Err(e)
                if e.is_instance_of::<PyZeroDivisionError>(py)
                    || e.is_instance_of::<PyTypeError>(py) => {}
            Err(e) => return Err(e),
        }
    }

    if pd.call_method1("isna", (&rate_b_per_a,))?.is_truthy()? {
        log(
            py,
            "warning",
            format!(
                "Warning: Current FX rate lookup failed for {}->{}. Returning NaN",
                o.fmt(from_curr)?,
                o.fmt(to_curr)?
            ),
        )?;
        return Ok(nan);
    }
    builtin(py, "float")?.call1((rate_b_per_a,))
}

#[pyfunction]
pub fn fu_price_table<'py>(py: Python<'py>, df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let key = builtin(py, "id")?.call1((df,))?;
    let index = df.getattr("index")?;
    let signature = PyTuple::new(
        py,
        [
            builtin(py, "len")?.call1((df,))?,
            builtin(py, "tuple")?.call1((df.getattr("columns")?,))?,
            index.get_item(0)?,
            index.get_item(-1)?,
        ],
    )?;
    let tables = m.getattr("_PRICE_TABLES")?;
    let hit = tables.call_method1("get", (&key,))?;
    if !hit.is_none() && hit.get_item(0)?.call0()?.is(df) && hit.get_item(1)?.eq(&signature)? {
        return hit.get_item(2);
    }
    let table = m.getattr("_build_price_table")?.call1((df,))?;
    // The callback holds the dict itself, as the original's lambda holds the
    // module's globals: it can fire at interpreter shutdown, when no import works.
    let (k, held) = (key.clone().unbind(), tables.clone().unbind());
    let evict = PyCFunction::new_closure(py, None, None, move |args, _| -> PyResult<Py<PyAny>> {
        let py = args.py();
        held.bind(py).call_method1("pop", (k.bind(py), py.None()))?;
        Ok(py.None())
    })?;
    let weak = match py.import("weakref")?.getattr("ref")?.call1((df, evict)) {
        Ok(w) => w,
        Err(e) if e.is_instance_of::<PyTypeError>(py) => return Ok(table),
        Err(e) => return Err(e),
    };
    tables.set_item(
        key,
        PyTuple::new(py, [weak, signature.into_any(), table.clone()])?,
    )?;
    Ok(table)
}

#[pyfunction]
pub fn fu_build_price_table<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let pd = py.import("pandas")?;
    let np = m.getattr("np")?;
    let idx = df.getattr("index")?;
    if !idx.is_instance(&pd.getattr("DatetimeIndex")?)? {
        return Ok(none(py));
    }
    let tz = idx.getattr("tz")?;
    if !tz.is_none() && tz.str()?.to_str()? != "UTC" {
        return Ok(none(py)); // midnight in another zone: leave it to pandas
    }
    if !df.getattr("columns")?.getattr("is_unique")?.is_truthy()? {
        return Ok(none(py));
    }
    let mut chosen = None;
    for c in ["price", "Adj Close", "Close"] {
        if df.getattr("columns")?.contains(c)? {
            chosen = Some(c);
            break;
        }
    }
    let Some(name) = chosen else {
        return Ok(none(py));
    };
    let mut df = df.clone();
    let mut column = df.get_item(name)?;
    let types = pd.getattr("api")?.getattr("types")?;
    if !types
        .call_method1("is_numeric_dtype", (&column,))?
        .is_truthy()?
        || types
            .call_method1("is_bool_dtype", (&column,))?
            .is_truthy()?
    {
        return Ok(none(py));
    }
    if !idx.getattr("is_monotonic_increasing")?.is_truthy()? {
        df = df.call_method0("sort_index")?; // as the original sorts before its asof
        column = df.get_item(name)?;
    }
    let valid = py.import("operator")?.call_method1(
        "invert",
        (df.call_method0("isna")?
            .call_method(
                "any",
                (),
                Some(&kwargs(
                    py,
                    &[("axis", &1i64.into_pyobject(py)?.into_any())],
                )?),
            )?
            .call_method0("to_numpy")?,),
    )?;
    let ticks = match df
        .getattr("index")?
        .call_method1("as_unit", ("ns",))
        .and_then(|i| i.getattr("asi8"))
    {
        Ok(t) => t,
        Err(e)
            if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyOverflowError>(py) =>
        {
            return Ok(none(py));
        }
        Err(e) => return Err(e),
    };
    let values = column.call_method(
        "to_numpy",
        (),
        Some(&kwargs(py, &[("dtype", &np.getattr("float64")?)])?),
    )?;
    let contiguous = |a: Bound<'py, PyAny>| np.call_method1("ascontiguousarray", (a,));
    m.getattr("investa_core")?.getattr("PriceTable")?.call1((
        contiguous(ticks.get_item(&valid)?)?,
        contiguous(values.get_item(&valid)?)?,
        builtin(py, "int")?.call1((ticks.get_item(0)?,))?,
        builtin(py, "float")?.call1((values.get_item(0)?,))?,
    ))
}

/// `res.get(col)` until one is not missing: price, Adj Close, Close.
fn series_price<'py>(
    pd: &Bound<'py, PyModule>,
    res: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let missing = |p: &Bound<'py, PyAny>| -> PyResult<bool> {
        Ok(p.is_none() || pd.call_method1("isna", (p,))?.is_truthy()?)
    };
    let mut price = res.call_method1("get", ("price",))?;
    if missing(&price)? {
        price = res.call_method1("get", ("Adj Close",))?;
    }
    if missing(&price)? {
        price = res.call_method1("get", ("Close",))?;
    }
    Ok(price)
}

#[pyfunction]
pub fn fu_historical_price<'py>(
    py: Python<'py>,
    symbol_key: &Bound<'py, PyAny>,
    target_date: &Bound<'py, PyAny>,
    prices_dict: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let pd = py.import("pandas")?;
    let nan = m.getattr("np")?.getattr("nan")?;
    let dt = py.import("datetime")?;
    let date_t = dt.getattr("date")?;
    if target_date.is_none()
        || !(target_date.is_instance(&date_t)?
            || target_date.is_instance(&dt.getattr("datetime")?)?
            || target_date.is_instance(&pd.getattr("Timestamp")?)?)
    {
        return Ok(nan);
    }
    if !prices_dict.contains(symbol_key)?
        || prices_dict
            .get_item(symbol_key)?
            .getattr("empty")?
            .is_truthy()?
    {
        return Ok(nan);
    }
    let mut df = prices_dict.get_item(symbol_key)?;

    // A calendar day against a frame the Rust table covers: binary search.
    if target_date.get_type().is(&date_t) {
        let ns = target_date
            .call_method0("toordinal")?
            .sub(m.getattr("_EPOCH_ORDINAL")?)?
            .mul(m.getattr("_DAY_NS")?)?;
        if m.getattr("_TS_MIN_NS")?.le(&ns)? && ns.le(m.getattr("_TS_MAX_NS")?)? {
            let table = m.getattr("_price_table")?.call1((&df,))?;
            if !table.is_none() {
                return table.call_method1("asof", (ns,));
            }
        }
    }

    let o = Ops::new(py)?;
    let timestamp = pd.getattr("Timestamp")?;
    let datetime_index = pd.getattr("DatetimeIndex")?;
    let body = |df: &mut Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if !df
            .getattr("index")?
            .getattr("is_monotonic_increasing")?
            .is_truthy()?
        {
            *df = df.call_method0("sort_index")?;
        }
        let first_try = || -> PyResult<Bound<'py, PyAny>> {
            let mut target_ts = timestamp.call1((target_date,))?;
            let index = df.getattr("index")?;
            if index.is_instance(&datetime_index)? {
                let idx_tz = index.getattr("tz")?;
                let target_tz = target_ts.getattr("tz")?;
                if !idx_tz.is_none() && target_tz.is_none() {
                    target_ts = target_ts.call_method1("tz_localize", (&idx_tz,))?;
                } else if idx_tz.is_none() && !target_tz.is_none() {
                    target_ts = target_ts.call_method1("tz_localize", (none(py),))?;
                } else if !idx_tz.is_none() && !target_tz.is_none() {
                    target_ts = target_ts.call_method1("tz_convert", (&idx_tz,))?;
                }
            } else if index.len()? > 0
                && index.get_item(0)?.is_instance(&date_t)?
                && !index.get_item(0)?.is_instance(&dt.getattr("datetime")?)?
            {
                // A date-object index: look up by the date.
                target_ts = target_ts.call_method0("date")?;
            }
            df.call_method1("asof", (target_ts,))
        };
        let res = match first_try() {
            Ok(r) => r,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                log(
                    py,
                    "debug",
                    format!(
                        "AsOf initial fail for {} on {}: {}",
                        o.fmt(symbol_key)?,
                        o.fmt(target_date)?,
                        es(py, &e)
                    ),
                )?;
                let mut ts_target = timestamp.call1((target_date,))?;
                let index = df.getattr("index")?;
                if index.is_instance(&datetime_index)?
                    && !index.getattr("tz")?.is_none()
                    && ts_target.getattr("tz")?.is_none()
                {
                    ts_target = ts_target.call_method1("tz_localize", (index.getattr("tz")?,))?;
                }
                df.call_method1("asof", (ts_target,))?
            }
            Err(e) => return Err(e),
        };

        // Extract price from result
        let mut price = nan.clone();
        if res.is_instance(&pd.getattr("Series")?)? {
            price = series_price(&pd, &res)?;
        } else if res.is_instance(&pd.getattr("DataFrame")?)? {
            // Duplicate index case
            if !res.getattr("empty")?.is_truthy()? {
                for c in ["price", "Adj Close", "Close"] {
                    if res.getattr("columns")?.contains(c)? {
                        price = res.get_item(c)?.getattr("iloc")?.get_item(-1)?;
                        break;
                    }
                }
            }
        } else {
            pd.call_method1("notna", (&res,))?.is_truthy()?; // Scalar?
        }

        if price.is_none() || pd.call_method1("isna", (&price,))?.is_truthy()? {
            // Before the start of the data: backfill with the first row.
            let backfill = || -> PyResult<Option<Bound<'py, PyAny>>> {
                let first_ts = timestamp.call1((df.getattr("index")?.get_item(0)?,))?;
                let mut target_eval = timestamp.call1((target_date,))?;
                let first_tz = first_ts.getattr("tz")?;
                if !first_tz.is_none() && target_eval.getattr("tz")?.is_none() {
                    target_eval = target_eval.call_method1("tz_localize", (&first_tz,))?;
                } else if first_tz.is_none() && !target_eval.getattr("tz")?.is_none() {
                    target_eval = target_eval.call_method1("tz_localize", (none(py),))?;
                }
                if target_eval.lt(&first_ts)? {
                    let columns = df.getattr("columns")?;
                    let mut col = "price";
                    if !columns.contains("price")? {
                        if columns.contains("Adj Close")? {
                            col = "Adj Close";
                        } else if columns.contains("Close")? {
                            col = "Close";
                        }
                    }
                    let first = df.get_item(col)?.getattr("iloc")?.get_item(0)?;
                    if pd.call_method1("notna", (&first,))?.is_truthy()? {
                        return Ok(Some(builtin(py, "float")?.call1((first,))?));
                    }
                }
                Ok(None)
            };
            match backfill() {
                Ok(Some(v)) => return Ok(v),
                Ok(None) => {}
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    log(
                        py,
                        "debug",
                        format!("Comparison fail in get_historical_price: {}", es(py, &e)),
                    )?;
                }
                Err(e) => return Err(e),
            }
        }
        if pd.call_method1("notna", (&price,))?.is_truthy()? {
            builtin(py, "float")?.call1((price,))
        } else {
            Ok(nan.clone())
        }
    };
    match body(&mut df) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "ERROR getting historical price for {} on {}: {}",
                    o.fmt(symbol_key)?,
                    o.fmt(target_date)?,
                    es(py, &e)
                ),
            )?;
            Ok(nan)
        }
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn fu_rate_via_usd_bridge<'py>(
    py: Python<'py>,
    from_curr: &Bound<'py, PyAny>,
    to_curr: &Bound<'py, PyAny>,
    target_date: &Bound<'py, PyAny>,
    historical_fx_data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = fu(py)?;
    let pd = py.import("pandas")?;
    let nan = m.getattr("np")?.getattr("nan")?;
    let o = Ops::new(py)?;
    let date_t = py.import("datetime")?.getattr("date")?;
    if !code_ok(from_curr)? || !code_ok(to_curr)? || !target_date.is_instance(&date_t)? {
        log(
            py,
            "debug",
            format!(
                "Hist FX Bridge: Invalid input - from={}, to={}, date={}",
                o.fmt(from_curr)?,
                o.fmt(to_curr)?,
                o.fmt(target_date)?
            ),
        )?;
        return Ok(nan);
    }
    if from_curr.eq(to_curr)? {
        return Ok(1.0f64.into_pyobject(py)?.into_any());
    }
    if !historical_fx_data.is_instance_of::<pyo3::types::PyDict>() {
        log(
            py,
            "warning",
            "Hist FX Bridge: Invalid historical_fx_data type received.",
        )?;
        return Ok(nan);
    }

    let historical_price = m.getattr("get_historical_price")?;
    let per_usd = |upper: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if upper.eq("USD")? {
            return Ok(1.0f64.into_pyobject(py)?.into_any());
        }
        let pair = pair_for(&o, upper)?;
        log(
            py,
            "debug",
            format!(
                "Hist FX Bridge: Getting historical price for {pair} on {}",
                o.fmt(target_date)?
            ),
        )?;
        let price = historical_price.call1((&pair, target_date, historical_fx_data))?;
        if !price.is_none() && pd.call_method1("notna", (&price,))?.is_truthy()? {
            return Ok(price);
        }
        Ok(nan.clone())
    };
    let rate_a = per_usd(&from_curr.call_method0("upper")?)?;
    let rate_b = per_usd(&to_curr.call_method0("upper")?)?;

    let both = pd.call_method1("notna", (&rate_a,))?.is_truthy()?
        && pd.call_method1("notna", (&rate_b,))?.is_truthy()?;
    let mut rate_b_per_a = nan.clone();
    if both && builtin(py, "abs")?.call1((&rate_a,))?.gt(1e-12)? {
        match rate_b.div(&rate_a) {
            Ok(r) => rate_b_per_a = r,
            Err(e)
                if e.is_instance_of::<PyZeroDivisionError>(py)
                    || e.is_instance_of::<PyTypeError>(py)
                    || e.is_instance_of::<PyOverflowError>(py) => {}
            Err(e) => return Err(e),
        }
    }

    if pd.call_method1("isna", (&rate_b_per_a,))?.is_truthy()? {
        if both {
            log(
                py,
                "warning",
                format!(
                    "Hist FX Bridge: Calculation failed for {}->{} on {} despite finding intermediate rates ({}, {}).",
                    o.fmt(from_curr)?,
                    o.fmt(to_curr)?,
                    o.fmt(target_date)?,
                    o.fmt(&rate_a)?,
                    o.fmt(&rate_b)?
                ),
            )?;
        }
        return Ok(nan);
    }
    builtin(py, "float")?.call1((rate_b_per_a,))
}
