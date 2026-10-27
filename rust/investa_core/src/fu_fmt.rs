//! The last block of `finutils`: the Yahoo symbol mapper, the safe column
//! sum, and the display formatters (currency, percentage, large numbers,
//! integers and floats with separators).
//!
//! Ported line for line from `src/finutils.py` (frozen in
//! `tests/reference/finutils_p16.py`). Symbols are normalized and matched with
//! Python's own string methods, `len` and `re` (a ticker can carry non-ASCII
//! text, and `len` counts code points); numbers are rendered by Python's
//! `format` with the original's nested specs, so a `decimals` that cannot
//! build a spec fails the same way. `is_cash_symbol` is looked up on the
//! module at call time.

use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};

use crate::fr::{builtin, none};
use crate::fu::fu;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr};

/// Default system mappings for common problematic tickers.
const SYSTEM_MAP: [(&str, &str); 6] = [
    ("BRK.B", "BRK-B"),
    ("BRK.A", "BRK-A"),
    ("BF.B", "BF-B"),
    ("BF.A", "BF-A"),
    ("RDS.A", "RDS-A"),
    ("RDS.B", "RDS-B"),
];

#[pyfunction]
pub fn fu_map_to_yf_symbol<'py>(
    py: Python<'py>,
    internal_symbol: &Bound<'py, PyAny>,
    user_symbol_map: &Bound<'py, PyAny>,
    user_excluded_symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !internal_symbol.is_truthy()? || !internal_symbol.is_instance_of::<PyString>() {
        return Ok(none(py));
    }
    let o = Ops::new(py)?;
    let len = builtin(py, "len")?;
    let text = |x: &Bound<'py, PyAny>| o.fmt(x);
    log(
        py,
        "debug",
        format!(
            "map_to_yf_symbol: Received internal_symbol='{}'",
            text(internal_symbol)?
        ),
    )?;

    let normalized = internal_symbol
        .call_method0("upper")?
        .call_method0("strip")?;
    let n = text(&normalized)?;
    log(py, "debug", format!("  Normalized to: '{n}'"))?;

    // 1. Excluded and cash symbols first
    let m = fu(py)?;
    if m.getattr("is_cash_symbol")?
        .call1((internal_symbol,))?
        .is_truthy()?
    {
        log(
            py,
            "debug",
            format!("  Symbol '{n}' is CASH. Returning None."),
        )?;
        return Ok(none(py));
    }
    if user_excluded_symbols.contains(&normalized)? {
        log(
            py,
            "debug",
            format!("  map_to_yf_symbol: Symbol '{n}' is in EXCLUSION list. Returning None."),
        )?;
        return Ok(none(py));
    }

    // 2. The explicit map, its keys normalized too
    let normalized_map = PyDict::new(py);
    for item in user_symbol_map.call_method0("items")?.try_iter()? {
        let kv = crate::pyhelp::unpack(&item?, 2)?;
        let k = kv[0].call_method0("upper")?.call_method0("strip")?;
        let v = kv[1].call_method0("upper")?.call_method0("strip")?;
        normalized_map.set_item(k, v)?;
    }
    if let Some(mapped) = normalized_map.get_item(&normalized)? {
        log(
            py,
            "debug",
            format!("  Found in explicit map: '{n}' -> '{}'", text(&mapped)?),
        )?;
        return Ok(mapped);
    }
    let system = PyDict::new(py);
    for (k, v) in SYSTEM_MAP {
        system.set_item(k, v)?;
    }
    if let Some(mapped) = system.get_item(&normalized)? {
        log(
            py,
            "debug",
            format!("  Applied SYSTEM_MAP: '{n}' -> '{}'", text(&mapped)?),
        )?;
        return Ok(mapped);
    }

    // 3. Thai stocks (:BKK -> .BK)
    if normalized.contains(":BKK")? {
        let base = normalized.call_method1("replace", (":BKK", ""))?;
        if base.contains(".")? || len.call1((&base,))?.eq(0)? {
            log(
                py,
                "warning",
                format!(
                    "map_to_yf_symbol: Skipping potentially invalid BKK conversion: {}",
                    text(internal_symbol)?
                ),
            )?;
            return Ok(none(py));
        }
        return Ok(pystr(
            py,
            &format!("{}.BK", text(&base.call_method0("upper")?)?),
        ));
    }

    // 4a. Dots to dashes for short US-style class tickers
    if normalized.contains(".")? && len.call1((&normalized,))?.le(6)? {
        let parts = normalized.call_method1("split", (".",))?;
        if len.call1((&parts,))?.eq(2)? && len.call1((parts.get_item(1)?,))?.le(2)? {
            let suffix = parts.get_item(1)?.call_method0("upper")?;
            let known = ["BK", "L", "TO", "V"];
            let mut is_known = false;
            for k in known {
                if suffix.eq(k)? {
                    is_known = true;
                    break;
                }
            }
            if !is_known {
                let converted = normalized.call_method1("replace", (".", "-"))?;
                log(
                    py,
                    "debug",
                    format!(
                        "  Applied DOT-to-DASH heuristic: '{n}' -> '{}'",
                        text(&converted)?
                    ),
                )?;
                return Ok(converted);
            }
        }
    }

    // 4b. Obviously invalid tickers (custom names, fund tags)
    let matched = py
        .import("re")?
        .call_method1("match", (r"^[A-Z0-9\.\-\^:]+$", &normalized))?;
    if !matched.is_truthy()? {
        log(
            py,
            "warning",
            format!(
                "map_to_yf_symbol: Skipping symbol with invalid characters: {}",
                text(internal_symbol)?
            ),
        )?;
        return Ok(none(py));
    }

    // 4c. Extremely long symbols
    if len.call1((&normalized,))?.gt(15)? {
        log(
            py,
            "warning",
            format!(
                "map_to_yf_symbol: Skipping excessively long symbol: {}",
                text(internal_symbol)?
            ),
        )?;
        return Ok(none(py));
    }

    // 5. The normalized symbol
    let upper = normalized.call_method0("upper")?;
    log(
        py,
        "debug",
        format!(
            "  No rules applied. Returning normalized symbol: '{}'",
            text(&upper)?
        ),
    )?;
    Ok(upper)
}

#[pyfunction]
pub fn fu_safe_sum<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    col: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let zero = || -> PyResult<Bound<'py, PyAny>> { Ok(crate::fr::fl(py, 0.0)) };
    if !df.getattr("columns")?.contains(col)? {
        return zero();
    }
    let pd = py.import("pandas")?;
    let body = || -> PyResult<Bound<'py, PyAny>> {
        let series = pd
            .getattr("to_numeric")?
            .call(
                (df.get_item(col)?,),
                Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
            )?
            .call_method1("fillna", (0.0,))?;
        let total = series.call_method0("sum")?;
        if pd.call_method1("notna", (&total,))?.is_truthy()? {
            builtin(py, "float")?.call1((total,))
        } else {
            zero()
        }
    };
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Error in safe_sum for column {}: {}",
                    Ops::new(py)?.fmt(col)?,
                    es(py, &e)
                ),
            )?;
            zero()
        }
        Err(e) => Err(e),
    }
}

/// `value is None or pd.isna(value)`, raising where the original raises.
fn missing(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(value.is_none()
        || py
            .import("pandas")?
            .call_method1("isna", (value,))?
            .is_truthy()?)
}

fn na(py: Python<'_>) -> Bound<'_, PyAny> {
    pystr(py, "N/A")
}

/// `try: ... except (ValueError, TypeError): return "N/A"`
fn or_na<'py>(
    py: Python<'py>,
    body: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyValueError>(py) || e.is_instance_of::<PyTypeError>(py) => {
            Ok(na(py))
        }
        Err(e) => Err(e),
    }
}

/// `f"{x:,.{decimals}f}"`
fn grouped<'py>(
    o: &Ops<'py>,
    x: &Bound<'py, PyAny>,
    decimals: &Bound<'py, PyAny>,
) -> PyResult<String> {
    let spec = format!(",.{}f", o.fmt(decimals)?);
    o.fmt_spec(x, &spec)
}

#[pyfunction]
pub fn fu_format_currency_value<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    currency_symbol: &Bound<'py, PyAny>,
    decimals: &Bound<'py, PyAny>,
    show_plus_sign: &Bound<'py, PyAny>,
    is_abs: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if missing(py, value)? {
        return Ok(na(py));
    }
    or_na(py, || {
        let o = Ops::new(py)?;
        let float = builtin(py, "float")?;
        let abs = builtin(py, "abs")?;
        let is_abs = is_abs.is_truthy()?;
        let mut val = if is_abs {
            abs.call1((float.call1((value,))?,))?
        } else {
            float.call1((value,))?
        };
        if abs.call1((&val,))?.lt(1e-9)? && !is_abs {
            val = 0.0f64.into_pyobject(py)?.into_any();
        }
        let sign = if show_plus_sign.is_truthy()? && val.gt(1e-9)? {
            "+"
        } else {
            ""
        };
        let number = grouped(&o, &val, decimals)?;
        let symbol = o.fmt(currency_symbol)?;
        Ok(pystr(
            py,
            &if is_abs {
                format!("{symbol}{number}")
            } else {
                format!("{sign}{symbol}{number}")
            },
        ))
    })
}

#[pyfunction]
pub fn fu_format_percentage_value<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    decimals: &Bound<'py, PyAny>,
    show_plus_sign: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if missing(py, value)? {
        return Ok(na(py));
    }
    or_na(py, || {
        let o = Ops::new(py)?;
        let val = builtin(py, "float")?.call1((value,))?;
        if fu(py)?
            .getattr("np")?
            .call_method1("isinf", (&val,))?
            .is_truthy()?
        {
            return Ok(pystr(py, "Inf %"));
        }
        let sign = if show_plus_sign.is_truthy()? && val.gt(1e-9)? {
            "+"
        } else {
            ""
        };
        Ok(pystr(
            py,
            &format!("{sign}{}%", grouped(&o, &val, decimals)?),
        ))
    })
}

#[pyfunction]
pub fn fu_format_large_number_display<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    currency_symbol: &Bound<'py, PyAny>,
    decimals: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if missing(py, value)? {
        return Ok(na(py));
    }
    or_na(py, || {
        let o = Ops::new(py)?;
        let val = builtin(py, "float")?.call1((value,))?;
        let symbol = o.fmt(currency_symbol)?;
        for (scale, suffix) in [(1e12, "T"), (1e9, "B"), (1e6, "M")] {
            if val.ge(scale)? {
                let number = grouped(&o, &val.div(scale)?, decimals)?;
                return Ok(pystr(py, &format!("{symbol}{number} {suffix}")));
            }
        }
        Ok(pystr(
            py,
            &format!("{symbol}{}", grouped(&o, &val, decimals)?),
        ))
    })
}

#[pyfunction]
pub fn fu_format_integer_with_commas<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if missing(py, value)? {
        return Ok(na(py));
    }
    or_na(py, || {
        let o = Ops::new(py)?;
        let rounded = builtin(py, "round")?.call1((builtin(py, "float")?.call1((value,))?,))?;
        let as_int = builtin(py, "int")?.call1((rounded,))?;
        Ok(pystr(py, &o.fmt_spec(&as_int, ",")?))
    })
}

#[pyfunction]
pub fn fu_format_float_with_commas<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    decimals: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if missing(py, value)? {
        return Ok(na(py));
    }
    or_na(py, || {
        let o = Ops::new(py)?;
        let val = builtin(py, "float")?.call1((value,))?;
        Ok(pystr(py, &grouped(&o, &val, decimals)?))
    })
}
