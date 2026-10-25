//! What the ported storage and glue code needs from Python, in one place.
//!
//! The modules ported in phases 11 onward work on the same Python objects the
//! originals did — dicts from the routes, frames from pandas — so wherever the
//! original relied on a Python operation's exact behaviour (`str()`, `.title()`,
//! `pd.isna`, `json.dumps`, `datetime.now()`, logging with the active
//! exception), the Rust calls that operation rather than imitating it. The
//! data work in between is Rust.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString, PyTuple};

/// `logging.<level>(message)` through the root logger, as the originals log.
pub fn log(py: Python<'_>, level: &str, message: impl AsRef<str>) -> PyResult<()> {
    py.import("logging")?
        .call_method1(level, (message.as_ref(),))?;
    Ok(())
}

/// `logging.<level>(message, exc_info=True)` inside the `except` that caught
/// `err`: the record carries that exception, as it does in Python.
pub fn log_exc(py: Python<'_>, level: &str, message: impl AsRef<str>, err: &PyErr) -> PyResult<()> {
    let info = PyTuple::new(
        py,
        [
            err.get_type(py).into_any(),
            err.value(py).clone().into_any(),
            err.traceback(py)
                .map(|t| t.into_any())
                .unwrap_or_else(|| py.None().into_bound(py)),
        ],
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("exc_info", info)?;
    py.import("logging")?
        .call_method(level, (message.as_ref(),), Some(&kwargs))?;
    Ok(())
}

/// `str(obj)`, as an f-string's `{obj}` renders it.
pub fn s(obj: &Bound<'_, PyAny>) -> PyResult<String> {
    Ok(obj.str()?.to_str()?.to_owned())
}

/// `str(err)` for a caught exception.
pub fn es(py: Python<'_>, err: &PyErr) -> String {
    err.value(py)
        .str()
        .map(|s| s.to_string())
        .unwrap_or_default()
}

/// `bool(pd.isna(obj))`: raises where Python's `if pd.isna(x):` would.
pub fn isna(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<bool> {
    py.import("pandas")?
        .call_method1("isna", (obj,))?
        .is_truthy()
}

/// `datetime.now().isoformat()`.
pub fn now_iso(py: Python<'_>) -> PyResult<String> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method0("now")?
        .call_method0("isoformat")?
        .extract()
}

/// Whether `err` is an instance of `sqlite3.<class>`.
pub fn is_sqlite(py: Python<'_>, err: &PyErr, class: &str) -> PyResult<bool> {
    let cls = py.import("sqlite3")?.getattr(class)?;
    err.value(py).is_instance(&cls)
}

/// `dict.get(key)` on any mapping-like object (Python's `obj.get(key)`).
pub fn get<'py>(obj: &Bound<'py, PyAny>, key: &str) -> PyResult<Bound<'py, PyAny>> {
    obj.call_method1("get", (key,))
}

/// `obj.get(key, default)`.
pub fn get_or<'py>(
    obj: &Bound<'py, PyAny>,
    key: &str,
    default: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    obj.call_method1("get", (key, default))
}

pub fn pystr<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyAny> {
    PyString::new(py, text).into_any()
}

/// `obj[key] = value` as a Python statement performs it.
///
/// The interpreter holds `obj` on its evaluation stack during the store, one
/// reference more than a call from Rust. pandas reads that count
/// (`DataFrame.__setitem__` warns of chained assignment when
/// `sys.getrefcount(self) <= 3`), so the extra reference is held here too:
/// without it every column assignment raises a FutureWarning the original
/// never did.
pub fn setitem<'py>(
    obj: &Bound<'py, PyAny>,
    key: impl IntoPyObject<'py>,
    value: impl IntoPyObject<'py>,
) -> PyResult<()> {
    let stack = obj.clone();
    let r = obj.set_item(key, value);
    drop(stack);
    r
}
