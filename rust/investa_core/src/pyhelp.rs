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

/// `a, b, ... = item` for `n` names, with Python's unpacking errors.
pub fn unpack<'py>(item: &Bound<'py, PyAny>, n: usize) -> PyResult<Vec<Bound<'py, PyAny>>> {
    use pyo3::exceptions::{PyTypeError, PyValueError};
    let it = match item.try_iter() {
        Ok(it) => it,
        Err(e) if e.is_instance_of::<PyTypeError>(item.py()) => {
            return Err(PyTypeError::new_err(format!(
                "cannot unpack non-iterable {} object",
                item.get_type().name()?
            )));
        }
        Err(e) => return Err(e),
    };
    let mut it = it;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        match it.next().transpose()? {
            Some(v) => out.push(v),
            None => {
                return Err(PyValueError::new_err(format!(
                    "not enough values to unpack (expected {n}, got {})",
                    out.len()
                )));
            }
        }
    }
    if it.next().transpose()?.is_some() {
        return Err(PyValueError::new_err(format!(
            "too many values to unpack (expected {n})"
        )));
    }
    Ok(out)
}

/// `pd.to_datetime(arg, **kwargs)`, without pandas' cache probe when `arg`
/// already holds datetimes and the call asks only for UTC.
///
/// For a `DatetimeIndex` or a datetime `Series`, pandas parses nothing: unless
/// `should_cache` finds repeats among the first elements, the answer is
/// `values.tz_convert(None).tz_localize("utc")` (or `tz_localize("utc")` when
/// naive), and `errors` never comes into it. The probe is the whole cost: it
/// boxes up to 500 Timestamps into a Python set, about 2.5 ms on a daily price
/// index against 0.01 ms for the conversion. Here it runs on the int64 values,
/// which compare exactly as the Timestamps do (NaT too: one sentinel value, one
/// singleton). Any other argument or keyword, or an input pandas would cache,
/// goes to pandas itself.
pub fn to_datetime<'py>(
    pd: &Bound<'py, PyAny>,
    arg: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    match utc_fast_path(pd, arg, kwargs)? {
        Some(result) => Ok(result),
        None => pd.call_method("to_datetime", (arg,), kwargs),
    }
}

fn utc_fast_path<'py>(
    pd: &Bound<'py, PyAny>,
    arg: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let py = arg.py();
    let Some(kwargs) = kwargs else {
        return Ok(None);
    };
    for (key, value) in kwargs.iter() {
        let ok = match key.extract::<&str>()? {
            "utc" => value.is(pyo3::types::PyBool::new(py, true)),
            "errors" => matches!(value.extract::<&str>().ok(), Some("raise" | "coerce")),
            _ => false,
        };
        if !ok {
            return Ok(None);
        }
    }
    if !kwargs.contains("utc")? {
        return Ok(None);
    }
    let is_series = arg.is_instance(&pd.getattr("Series")?)?;
    let values = if arg.is_instance(&pd.getattr("DatetimeIndex")?)? {
        arg.clone()
    } else if is_series {
        let values = arg.getattr("_values")?;
        if !values.is_instance(&pd.getattr("arrays")?.getattr("DatetimeArray")?)? {
            return Ok(None);
        }
        values
    } else {
        return Ok(None);
    };
    if would_cache(&values)? {
        return Ok(None);
    }
    let utc = if values.getattr("tz")?.is_none() {
        values.call_method1("tz_localize", ("utc",))?
    } else {
        values
            .call_method1("tz_convert", (py.None(),))?
            .call_method1("tz_localize", ("utc",))?
    };
    if !is_series {
        return Ok(Some(utc));
    }
    let kw = PyDict::new(py);
    kw.set_item("index", arg.getattr("index")?)?;
    kw.set_item("name", arg.getattr("name")?)?;
    Ok(Some(arg.getattr("_constructor")?.call((utc,), Some(&kw))?))
}

/// `pandas.core.tools.datetimes.should_cache(values)` with its defaults:
/// no cache at 50 elements or fewer, otherwise cache unless more than 70% of
/// the first tenth (at most 500) are distinct.
fn would_cache(values: &Bound<'_, PyAny>) -> PyResult<bool> {
    let n = values.len()?;
    if n <= 50 {
        return Ok(false);
    }
    let check = if n <= 5000 { n / 10 } else { 500 };
    let ints: numpy::PyReadonlyArray1<i64> = values.getattr("asi8")?.extract()?;
    let ints = ints.as_array();
    let distinct: std::collections::HashSet<i64> = ints.iter().take(check).copied().collect();
    Ok(distinct.len() as f64 <= check as f64 * 0.7)
}

/// `investa_core.to_datetime(arg, **kwargs)`: [`to_datetime`] for Python, so
/// the tests can hold it against `pd.to_datetime` directly.
#[pyfunction(name = "to_datetime")]
#[pyo3(signature = (arg, **kwargs))]
pub fn to_datetime_py<'py>(
    py: Python<'py>,
    arg: &Bound<'py, PyAny>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    to_datetime(py.import("pandas")?.as_any(), arg, kwargs)
}

/// Whether `pd.to_numeric(v, errors="coerce")` (or `"raise"`) hands `v` back
/// unchanged: a Python int, float or bool, numpy's float64 included, passes
/// `is_number` before pandas converts anything. Per-value callers skip the
/// call for these, which are nearly all the cells a transaction row holds.
pub fn numeric_passthrough(v: &Bound<'_, PyAny>) -> bool {
    v.is_instance_of::<pyo3::types::PyFloat>() || v.is_instance_of::<pyo3::types::PyInt>()
}

/// `v.as_unit("ns")` for a DatetimeIndex, `v.dt.as_unit("ns")` for a datetime
/// Series: the same instants counted in nanoseconds. Wherever stamps become
/// integers to compare, every side must count in one unit, and pandas 3 picks
/// the unit from the input (seconds from a `date`, microseconds from a string),
/// where pandas 2 always used nanoseconds; under pandas 2 this changes nothing.
pub fn as_ns<'py>(v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let pd = v.py().import("pandas")?;
    if v.is_instance(&pd.getattr("Series")?)? {
        v.getattr("dt")?.call_method1("as_unit", ("ns",))
    } else {
        v.call_method1("as_unit", ("ns",))
    }
}

/// The stamps of a DatetimeIndex or datetime Series as nanosecond `int64`s
/// (UTC for zone-aware stamps, NaT as the int64 minimum): `as_ns(v).asi8`,
/// through `.array` for a Series. The same integers `.values.astype("int64")`
/// gave, without pandas 3.1's deprecation of `.values` on zone-aware stamps.
pub fn ns_ints<'py>(v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let pd = v.py().import("pandas")?;
    let ns = as_ns(v)?;
    if v.is_instance(&pd.getattr("Series")?)? {
        ns.getattr("array")?.getattr("asi8")
    } else {
        ns.getattr("asi8")
    }
}

/// The installed pandas' major version, read from `pandas.__version__`.
pub fn pandas_major(pd: &Bound<'_, PyModule>) -> PyResult<u32> {
    let version: String = pd.getattr("__version__")?.extract()?;
    Ok(version
        .split('.')
        .next()
        .and_then(|m| m.parse().ok())
        .unwrap_or(0))
}

/// A text column as pandas 2's `astype(str)` received it. pandas 3 reads
/// text into its `str` dtype, holding SQL NULLs as NaN, and `astype(str)`
/// keeps them missing; pandas 2 read them into an object column as None,
/// which `astype(str)` stringified to "None", the value the loader's cleaning
/// then blanks or lowercases. So under pandas 3 a `str` column's missing
/// values become "None" first; anything else is `col` itself.
pub fn text_input<'py>(col: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let pd = col.py().import("pandas")?;
    if pandas_major(&pd)? >= 3
        && col
            .getattr("dtype")?
            .is_instance(&pd.getattr("StringDtype")?)?
    {
        return col.call_method1("fillna", ("None",));
    }
    Ok(col.clone())
}

/// The datetime unit `pd.to_datetime` gives a Series of `YYYY-MM-DD` strings:
/// "ns" in pandas 2, "us" in pandas 3. Where the Rust builds stamps natively
/// in place of that call, it builds them in this unit. Asked once per process.
pub fn parsed_date_unit(py: Python<'_>) -> PyResult<&'static str> {
    static UNIT: std::sync::OnceLock<&'static str> = std::sync::OnceLock::new();
    if let Some(unit) = UNIT.get() {
        return Ok(unit);
    }
    let pd = py.import("pandas")?;
    let probe = pd.getattr("Series")?.call1((vec!["2000-01-03"],))?;
    let unit: String = pd
        .call_method1("to_datetime", (probe,))?
        .getattr("dt")?
        .getattr("unit")?
        .extract()?;
    let unit: &'static str = match unit.as_str() {
        "s" => "s",
        "ms" => "ms",
        "us" => "us",
        _ => "ns",
    };
    Ok(UNIT.get_or_init(|| unit))
}

/// `df.iterrows()`, without building a Series per row.
///
/// `iterrows` yields `(label, Series(df.values[i], index=df.columns))`. The
/// ported loops only ever read a row with `row[col]` and `row.get(col,
/// default)`, which on that Series return the very `df.values` cell at the
/// column's position. So where those lookups cannot differ (an exact
/// `DataFrame` whose columns are a plain, unique `Index` and whose `values`
/// are an object array), the rows come back as [`RowView`]s over the same
/// cells. Any other frame gets the real `iterrows`.
pub fn iterrows<'py>(df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = df.py();
    let pd = py.import("pandas")?;
    let columns = df.getattr("columns")?;
    let plain = df.get_type().is(&pd.getattr("DataFrame")?)
        && columns.get_type().is(&pd.getattr("Index")?)
        && columns.getattr("is_unique")?.is_truthy()?;
    if !plain {
        return df.call_method0("iterrows");
    }
    let values = df.getattr("values")?;
    let object = py.import("numpy")?.call_method1("dtype", ("O",))?;
    if !values.getattr("dtype")?.eq(&object)? {
        return df.call_method0("iterrows");
    }
    let at = PyDict::new(py);
    for (j, label) in columns.try_iter()?.enumerate() {
        at.set_item(label?, j)?;
    }
    let at = at.unbind();
    let frame = df.clone().unbind();
    let cells = values.call_method0("tolist")?;
    let out = pyo3::types::PyList::empty(py);
    for (i, (label, row)) in df
        .getattr("index")?
        .try_iter()?
        .zip(cells.try_iter()?)
        .enumerate()
    {
        let label = label?;
        let view = RowView {
            cells: row?.cast_into::<pyo3::types::PyList>()?.unbind(),
            at: at.clone_ref(py),
            frame: frame.clone_ref(py),
            position: i,
            label: label.clone().unbind(),
        };
        out.append((label, Bound::new(py, view)?))?;
    }
    Ok(out.into_any())
}

/// One row of [`iterrows`]: `row[col]` and `row.get(col, default)` read the
/// cell directly; a key that is not text goes to the Series `iterrows` would
/// have yielded.
#[pyclass(frozen)]
pub struct RowView {
    cells: Py<pyo3::types::PyList>,
    at: Py<PyDict>,
    frame: Py<PyAny>,
    position: usize,
    label: Py<PyAny>,
}

impl RowView {
    /// `df._constructor_sliced(df.values[i], index=df.columns, name=label).__finalize__(df)`
    fn series<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let df = self.frame.bind(py);
        let kw = PyDict::new(py);
        kw.set_item("index", df.getattr("columns")?)?;
        kw.set_item("name", self.label.bind(py))?;
        df.getattr("_constructor_sliced")?
            .call((df.getattr("values")?.get_item(self.position)?,), Some(&kw))?
            .call_method1("__finalize__", (df,))
    }
}

#[pymethods]
impl RowView {
    fn __getitem__<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        if !key.is_exact_instance_of::<PyString>() {
            return self.series(py)?.get_item(key);
        }
        match self.at.bind(py).get_item(key)? {
            Some(j) => self.cells.bind(py).get_item(j.extract()?),
            None => Err(pyo3::exceptions::PyKeyError::new_err(key.clone().unbind())),
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get<'py>(
        &self,
        py: Python<'py>,
        key: &Bound<'py, PyAny>,
        default: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let default = default.unwrap_or_else(|| py.None().into_bound(py));
        if !key.is_exact_instance_of::<PyString>() {
            return self.series(py)?.call_method1("get", (key, default));
        }
        match self.at.bind(py).get_item(key)? {
            Some(j) => self.cells.bind(py).get_item(j.extract()?),
            None => Ok(default),
        }
    }
}

/// `investa_core.iterrows(df)`: [`iterrows`] for Python, so the tests can
/// hold it against `DataFrame.iterrows` directly.
#[pyfunction(name = "iterrows")]
pub fn iterrows_py<'py>(df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    iterrows(df)
}
