//! `market_db.MarketDatabase`: the market-data archive (daily and intraday
//! bars, FX, corporate actions, price bases, fund NAVs, share counts).
//!
//! Ported line for line from `src/market_db.py` (frozen in
//! `tests/reference/market_db.py`). Each method takes the `MarketDatabase`
//! object as `slf`; its connection, write lock, column cache and the sibling
//! methods are reached through it, and module functions through `market_db`,
//! so subclasses and patches see the same calls the original made.
//!
//! The arithmetic is native: the ISO day strings, the split and dividend
//! factors (NumPy's cumprod, stable argsort, searchsorted and allclose,
//! reproduced), the read-time multiplies and the value cleaning of the
//! upserts. Frame plumbing (`to_numeric`, `groupby`, `merge`,
//! `read_sql_query`) is the same pandas calls. Where NumPy would format or
//! fail in a way the native path does not reproduce (a year outside
//! 1..=9999, a NaT among dates it compares), the original NumPy expression
//! runs instead.

use numpy::ndarray::Array1;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};

use crate::pyconn::PyConn;
use crate::pyhelp::{es, log, now_iso, pystr, s, setitem};

const BASIS_SPLIT_ADJ: &str = "split_adj";
const ADJUST_SPLIT: &str = "split";
const ADJUST_TOTAL_RETURN: &str = "total_return";
const MIN_SPLIT_RATIO: f64 = 1e-6;
const MAX_SPLIT_RATIO: f64 = 1e6;

fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("market_db")
}

fn kw<'py>(py: Python<'py>, pairs: &[(&str, Bound<'py, PyAny>)]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for (k, v) in pairs {
        d.set_item(*k, v)?;
    }
    Ok(d)
}

/// `with a, b as conn:` — enter `a`, then build and enter `b`, run `body`,
/// exit in reverse with the
/// exception (if any), as the statement does. A context manager that
/// swallows the exception is honoured.
pub fn with2<'py, T>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    let a_exit = a.get_type().getattr("__exit__")?;
    a.get_type().getattr("__enter__")?.call1((a,))?;
    // `with a, b:` evaluates `b` only once `a` is entered.
    let inner = b().and_then(|b| with1(py, &b, body));
    let none = py.None();
    match inner {
        Ok(v) => {
            a_exit.call1((a, &none, &none, &none))?;
            Ok(v)
        }
        Err(e) => {
            let swallowed = a_exit
                .call1((a, e.get_type(py), e.value(py), e.traceback(py)))?
                .is_truthy()?;
            if swallowed { Ok(None) } else { Err(e) }
        }
    }
}

/// `with cm as value:`.
pub fn with1<'py, T>(
    py: Python<'py>,
    cm: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    let t = cm.get_type();
    let exit = t.getattr("__exit__")?;
    let value = t.getattr("__enter__")?.call1((cm,))?;
    let none = py.None();
    match body(&value) {
        Ok(v) => {
            exit.call1((cm, &none, &none, &none))?;
            Ok(Some(v))
        }
        Err(e) => {
            let swallowed = exit
                .call1((cm, e.get_type(py), e.value(py), e.traceback(py)))?
                .is_truthy()?;
            if swallowed { Ok(None) } else { Err(e) }
        }
    }
}

// --- day strings ----------------------------------------------------------

/// Days since 1970-01-01 to (year, month, day), proleptic Gregorian
/// (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// `_iso_days(idx)`: `idx.strftime("%Y-%m-%d").tolist()` via NumPy — the wall
/// date of a tz-aware index, NaT as NaN.
#[pyfunction]
pub fn mdb_iso_days<'py>(py: Python<'py>, idx: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let idx = if idx.getattr("tz")?.is_none() {
        idx.clone()
    } else {
        idx.call_method1("tz_localize", (py.None(),))?
    };
    let np = py.import("numpy")?;
    let values = idx.call_method0("to_numpy")?;
    // ticks per day for the index's unit
    let unit: String = np
        .call_method1("datetime_data", (values.getattr("dtype")?,))?
        .get_item(0)?
        .extract()?;
    let per_day: Option<i64> = match unit.as_str() {
        "ns" => Some(86_400_000_000_000),
        "us" => Some(86_400_000_000),
        "ms" => Some(86_400_000),
        "s" => Some(86_400),
        "D" => Some(1),
        _ => None,
    };
    let ticks: Option<Vec<i64>> = per_day
        .map(|_| -> PyResult<Vec<i64>> {
            let ints = values.call_method1("view", ("int64",))?;
            let arr: PyReadonlyArray1<'_, i64> = ints.extract()?;
            Ok(arr.as_array().to_vec())
        })
        .transpose()?;
    let nan = np.getattr("nan")?;
    let mut out: Vec<Bound<'py, PyAny>> = Vec::new();
    let mut native = per_day.is_some();
    if let (Some(per_day), Some(ticks)) = (per_day, ticks.as_ref()) {
        for &t in ticks {
            if t == i64::MIN {
                out.push(nan.clone());
                continue;
            }
            let (y, m, d) = civil_from_days(t.div_euclid(per_day));
            if !(1..=9999).contains(&y) {
                native = false;
                break;
            }
            out.push(PyString::new(py, &format!("{y:04}-{m:02}-{d:02}")).into_any());
        }
    }
    if native {
        return PyList::new(py, out);
    }
    // NumPy's own formatting for years it writes differently.
    let days = np.call_method(
        "datetime_as_string",
        (values.call_method1("astype", ("datetime64[D]",))?,),
        Some(&kw(py, &[("unit", pystr(py, "D"))])?),
    )?;
    let out = days.call_method1("astype", (py.import("builtins")?.getattr("object")?,))?;
    let nat = idx.call_method0("isna")?;
    if nat.call_method0("any")?.is_truthy()? {
        out.set_item(&nat, &nan)?;
    }
    Ok(out.call_method0("tolist")?.cast_into::<PyList>()?)
}

// --- read-time adjustment ---------------------------------------------------

/// The date strings as Rust strings, or None when one is not a `str` (NumPy's
/// object comparisons would raise, and the NumPy path should be the one to).
fn plain_strings(list: &Bound<'_, PyAny>) -> PyResult<Option<Vec<String>>> {
    let mut out = Vec::new();
    for v in list.try_iter()? {
        let v = v?;
        if !v.is_exact_instance_of::<PyString>() {
            return Ok(None);
        }
        out.push(v.extract()?);
    }
    Ok(Some(out))
}

/// `np.searchsorted(sorted, x, side)` over strings (code-point order).
fn searchsorted(sorted: &[String], x: &str, right: bool) -> usize {
    if right {
        sorted.partition_point(|d| d.as_str() <= x)
    } else {
        sorted.partition_point(|d| d.as_str() < x)
    }
}

fn f64_column<'py>(series: &Bound<'py, PyAny>) -> PyResult<Vec<f64>> {
    let py = series.py();
    let numeric = py.import("pandas")?.call_method(
        "to_numeric",
        (series,),
        Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
    )?;
    let arr = numeric.call_method(
        "to_numpy",
        (),
        Some(&kw(py, &[("dtype", pystr(py, "float64"))])?),
    )?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    Ok(arr.as_array().to_vec())
}

/// `MarketDatabase._future_split_factors(date_strs, splits)`: for each date
/// the product of the split ratios dated after it, or None when there is
/// nothing to apply.
#[pyfunction]
pub fn mdb_future_split_factors<'py>(
    py: Python<'py>,
    date_strs: &Bound<'py, PyAny>,
    splits: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if splits.is_none() || splits.getattr("empty")?.is_truthy()? || date_strs.len()? == 0 {
        return Ok(py.None());
    }
    let rows = if splits.getattr("columns")?.contains("kind")? {
        splits.get_item(
            splits
                .get_item("kind")?
                .call_method1("__eq__", ("split",))?,
        )?
    } else {
        splits.clone()
    };
    if rows.getattr("empty")?.is_truthy()? {
        return Ok(py.None());
    }
    let ratios = f64_column(&rows.get_item("value")?)?;
    let dates: Vec<Bound<'py, PyAny>> = rows
        .get_item("date")?
        .call_method1("astype", (py.import("builtins")?.getattr("str")?,))?
        .call_method0("to_numpy")?
        .try_iter()?
        .collect::<PyResult<_>>()?;
    let mut kept: Vec<(String, f64)> = Vec::new();
    for (r, d) in ratios.iter().zip(&dates) {
        if r.is_finite() && *r > 0.0 && *r != 1.0 {
            kept.push((d.extract()?, *r));
        }
    }
    if kept.is_empty() {
        return Ok(py.None());
    }
    let Some(days) = plain_strings(date_strs)? else {
        // A NaN day among the strings: NumPy's comparison raises; let it.
        return original(py, "_future_split_factors_numpy", (date_strs, splits));
    };
    // np.argsort(kind="stable") on the date strings
    kept.sort_by(|a, b| a.0.cmp(&b.0));
    let (sorted_dates, ratios): (Vec<String>, Vec<f64>) = kept.into_iter().unzip();
    // suffix[i] = product of ratios[i:], by np.cumprod over the reversed array
    let mut suffix = vec![1.0; ratios.len() + 1];
    let mut acc = 1.0;
    for (i, r) in ratios.iter().enumerate().rev() {
        acc = if i == ratios.len() - 1 { *r } else { acc * r };
        suffix[i] = acc;
    }
    let factors: Vec<f64> = days
        .iter()
        .map(|d| suffix[searchsorted(&sorted_dates, d, true)])
        .collect();
    // np.allclose(factors, 1.0): |x - 1| <= 1e-08 + 1e-05 * |1|, NaN and inf never close
    let tol = 1e-08 + 1e-05 * 1.0_f64;
    if factors
        .iter()
        .all(|f| f.is_finite() && (f - 1.0).abs() <= tol)
    {
        return Ok(py.None());
    }
    Ok(Array1::from_vec(factors)
        .into_pyarray(py)
        .into_any()
        .unbind())
}

/// The frozen NumPy implementation of a static helper, for inputs the
/// native path leaves to it.
fn original<'py>(
    py: Python<'py>,
    name: &str,
    args: impl pyo3::call::PyCallArgs<'py>,
) -> PyResult<Py<PyAny>> {
    Ok(module(py)?.getattr(name)?.call1(args)?.unbind())
}

/// `MarketDatabase._dividend_factors(date_strs, split_adjusted_close,
/// dividends)`: CRSP-style back-adjustment that reinvests dividends.
#[pyfunction]
pub fn mdb_dividend_factors<'py>(
    py: Python<'py>,
    date_strs: &Bound<'py, PyAny>,
    split_adjusted_close: &Bound<'py, PyAny>,
    dividends: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if dividends.is_none() || dividends.getattr("empty")?.is_truthy()? || date_strs.len()? == 0 {
        return Ok(py.None());
    }
    let rows = if dividends.getattr("columns")?.contains("kind")? {
        dividends.get_item(
            dividends
                .get_item("kind")?
                .call_method1("__eq__", ("dividend",))?,
        )?
    } else {
        dividends.clone()
    };
    if rows.getattr("empty")?.is_truthy()? {
        return Ok(py.None());
    }
    // searchsorted assumes a sorted haystack; on anything else (or a NaN day,
    // or closes that are not a float64 array) NumPy's own result is the answer.
    let days = plain_strings(date_strs)?.filter(|d| d.windows(2).all(|w| w[0] <= w[1]));
    let closes: Option<PyReadonlyArray1<'_, f64>> = split_adjusted_close.extract().ok();
    let (Some(days), Some(closes)) = (days, closes) else {
        return original(
            py,
            "_dividend_factors_numpy",
            (date_strs, split_adjusted_close, dividends),
        );
    };
    let closes = closes.as_array();
    let mut factors = vec![1.0_f64; days.len()];
    let mut applied = false;
    let float = py.get_type::<PyFloat>();
    // rows.iterrows(): the date and value cells as Python objects
    for pair in rows.call_method0("iterrows")?.try_iter()? {
        let row = pair?.get_item(1)?;
        let ex_date: String = row.get_item("date")?.str()?.extract()?;
        let value = row.get_item("value")?;
        let amount: f64 = if value.is_none() {
            0.0
        } else {
            float.call1((&value,))?.extract()?
        };
        if amount <= 0.0 {
            continue;
        }
        let position = searchsorted(&days, &ex_date, false);
        if position == 0 || position >= days.len() {
            continue;
        }
        let prev_close = closes[position - 1];
        if !prev_close.is_finite() || prev_close <= 0.0 {
            continue;
        }
        let ratio = 1.0 - (amount / prev_close);
        if !(0.0 < ratio && ratio <= 1.0) {
            let prev = split_adjusted_close.get_item(position - 1)?;
            log(
                py,
                "debug",
                format!(
                    "Dividend factor skipped: {} vs prev close {} on {}",
                    s(&PyFloat::new(py, amount).into_any())?,
                    s(&prev)?,
                    ex_date
                ),
            )?;
            continue;
        }
        for f in &mut factors[..position] {
            *f *= ratio;
        }
        applied = true;
    }
    if !applied {
        return Ok(py.None());
    }
    Ok(Array1::from_vec(factors)
        .into_pyarray(py)
        .into_any()
        .unbind())
}

fn float64_values(frame: &Bound<'_, PyAny>, col: &str) -> PyResult<Vec<f64>> {
    let py = frame.py();
    let arr = frame.get_item(col)?.call_method(
        "to_numpy",
        (),
        Some(&kw(py, &[("dtype", pystr(py, "float64"))])?),
    )?;
    let arr: PyReadonlyArray1<'_, f64> = arr.extract()?;
    Ok(arr.as_array().to_vec())
}

fn set_column(frame: &Bound<'_, PyAny>, col: &str, values: Vec<f64>) -> PyResult<()> {
    let py = frame.py();
    setitem(frame, col, Array1::from_vec(values).into_pyarray(py))
}

/// `MarketDatabase._adjust_frame`: move `frame` from its stored basis onto
/// the one `adjust` asks for.
#[pyfunction]
pub fn mdb_adjust_frame<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    frame: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    stored_basis: &Bound<'py, PyAny>,
    adjust: &Bound<'py, PyAny>,
    actions: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let _ = symbol;
    if frame.getattr("empty")?.is_truthy()? {
        return Ok(frame.clone());
    }
    let columns = frame.getattr("columns")?;
    let mut price_cols = Vec::new();
    for c in ["Open", "High", "Low", "Close", "Adj Close"] {
        if columns.contains(c)? {
            price_cols.push(c);
        }
    }
    if price_cols.is_empty() {
        return Ok(frame.clone());
    }
    let stored_is_split_adj = stored_basis.eq(BASIS_SPLIT_ADJ)?;
    let wants_split_adj = adjust.eq(ADJUST_SPLIT)? || adjust.eq(ADJUST_TOTAL_RETURN)?;
    let needs_split_work = stored_is_split_adj != wants_split_adj;
    let needs_dividends = adjust.eq(ADJUST_TOTAL_RETURN)?;
    if !needs_split_work && !needs_dividends {
        return Ok(frame.clone());
    }
    let date_strs = module(py)?
        .getattr("_iso_days")?
        .call1((frame.getattr("index")?,))?;
    let factors = if needs_split_work {
        slf.getattr("_future_split_factors")?
            .call1((&date_strs, actions))?
    } else {
        py.None().into_bound(py)
    };
    if !factors.is_none() {
        let f: PyReadonlyArray1<'_, f64> = factors.extract()?;
        let f = f.as_array().to_vec();
        let net: Vec<f64> = if stored_is_split_adj {
            f
        } else {
            f.iter().map(|x| 1.0 / x).collect()
        };
        for col in &price_cols {
            let v = float64_values(frame, col)?;
            set_column(frame, col, v.iter().zip(&net).map(|(a, b)| a * b).collect())?;
        }
        if columns.contains("Volume")? {
            let v = float64_values(frame, "Volume")?;
            set_column(
                frame,
                "Volume",
                v.iter().zip(&net).map(|(a, b)| a / b).collect(),
            )?;
        }
    }
    if needs_dividends && frame.getattr("columns")?.contains("Close")? {
        let close = frame.get_item("Close")?.call_method(
            "to_numpy",
            (),
            Some(&kw(py, &[("dtype", pystr(py, "float64"))])?),
        )?;
        let div = slf
            .getattr("_dividend_factors")?
            .call1((&date_strs, close, actions))?;
        if !div.is_none() {
            let d: PyReadonlyArray1<'_, f64> = div.extract()?;
            let d = d.as_array().to_vec();
            for col in &price_cols {
                let v = float64_values(frame, col)?;
                set_column(frame, col, v.iter().zip(&d).map(|(a, b)| a * b).collect())?;
            }
        }
    }
    Ok(frame.clone())
}

// --- writes ------------------------------------------------------------------

/// `a, b = item` with Python's own errors: "cannot unpack non-iterable X
/// object", "not enough values to unpack (expected 2, got N)", "too many
/// values to unpack (expected 2)".
pub fn unpack2<'py>(item: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let mut it = match item.try_iter() {
        Ok(it) => it,
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyTypeError>(item.py()) => {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "cannot unpack non-iterable {} object",
                item.get_type().name()?
            )));
        }
        Err(e) => return Err(e),
    };
    let a = it.next().transpose()?;
    let b = it.next().transpose()?;
    match (a, b) {
        (Some(a), Some(b)) => {
            if it.next().transpose()?.is_some() {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "too many values to unpack (expected 2)",
                ));
            }
            Ok((a, b))
        }
        (a, _) => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "not enough values to unpack (expected 2, got {})",
            usize::from(a.is_some())
        ))),
    }
}

/// The day strings of an index: `_iso_days` for a DatetimeIndex, else each
/// element's `strftime` (or its first ten characters).
fn index_days<'py>(py: Python<'py>, idx: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let pd = py.import("pandas")?;
    if idx.is_instance(&pd.getattr("DatetimeIndex")?)? {
        return Ok(module(py)?
            .getattr("_iso_days")?
            .call1((idx,))?
            .cast_into::<PyList>()?);
    }
    let out = PyList::empty(py);
    let slice10 = py
        .import("builtins")?
        .getattr("slice")?
        .call1((py.None(), 10))?;
    for ts in idx.try_iter()? {
        let ts = ts?;
        if ts.hasattr("strftime")? {
            out.append(ts.call_method1("strftime", ("%Y-%m-%d",))?)?;
        } else {
            out.append(ts.str()?.get_item(&slice10)?)?;
        }
    }
    Ok(out)
}

/// `clean_floats`: the column as native floats, NaN as None.
fn clean_floats<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    col: Option<&str>,
    n: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let none = py.None().into_bound(py);
    let Some(col) = col else {
        return Ok(vec![none; n]);
    };
    if !df.getattr("columns")?.contains(col)? {
        return Ok(vec![none; n]);
    }
    Ok(f64_column(&df.get_item(col)?)?
        .into_iter()
        .map(|v| {
            if v.is_nan() {
                none.clone()
            } else {
                PyFloat::new(py, v).into_any()
            }
        })
        .collect())
}

/// `[0 if v != v else int(v) for v in volumes]`: an infinite or out-of-range
/// volume goes through Python's `int` so it raises (or grows) as there.
fn clean_volumes<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    n: usize,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    if !df.getattr("columns")?.contains("Volume")? {
        return (0..n)
            .map(|_| Ok(0i64.into_pyobject(py)?.into_any()))
            .collect();
    }
    let int = py.import("builtins")?.getattr("int")?;
    f64_column(&df.get_item("Volume")?)?
        .into_iter()
        .map(|v| {
            if v.is_nan() {
                Ok(0i64.into_pyobject(py)?.into_any())
            } else if v.is_finite() && v.abs() < 9.0e15 {
                Ok((v.trunc() as i64).into_pyobject(py)?.into_any())
            } else {
                int.call1((v,))
            }
        })
        .collect()
}

const OHLCV_FORCE_SQL: &str = "
                            INSERT OR REPLACE INTO daily_ohlcv
                            (symbol, date, open, high, low, close, adj_close,
                             volume, interval, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ";
const OHLCV_UPSERT_SQL: &str = "
                            INSERT INTO daily_ohlcv
                            (symbol, date, open, high, low, close, adj_close,
                             volume, interval, source)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            ON CONFLICT(symbol, date, interval) DO UPDATE SET
                                open = excluded.open, high = excluded.high,
                                low = excluded.low, close = excluded.close,
                                adj_close = excluded.adj_close,
                                volume = excluded.volume,
                                source = excluded.source
                            WHERE daily_ohlcv.source IS NULL
                               OR daily_ohlcv.source = excluded.source
                               OR daily_ohlcv.source = 'yahoo'
                        ";
const OHLCV_LEGACY_SQL: &str = "
                        INSERT OR REPLACE INTO daily_ohlcv
                        (symbol, date, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ";
const SYNC_STAMP_SQL: &str = "
                INSERT INTO sync_metadata (symbol, last_synced) VALUES (?, ?)
                ON CONFLICT(symbol) DO UPDATE SET last_synced = excluded.last_synced
            ";

/// `MarketDatabase.upsert_ohlcv`
#[pyfunction]
pub fn mdb_upsert_ohlcv<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    df: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    force: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if df.getattr("empty")?.is_truthy()? {
        return Ok(());
    }
    let n = df.len()?;
    let date_strs = index_days(py, &df.getattr("index")?)?;
    let opens = clean_floats(py, df, Some("Open"), n)?;
    let highs = clean_floats(py, df, Some("High"), n)?;
    let lows = clean_floats(py, df, Some("Low"), n)?;
    let close_col = if df.getattr("columns")?.contains("Close")? {
        "Close"
    } else {
        "price"
    };
    let closes = clean_floats(py, df, Some(close_col), n)?;
    let adj = if df.getattr("columns")?.contains("Adj Close")? {
        clean_floats(py, df, Some("Adj Close"), n)?
    } else {
        clean_floats(py, df, None, n)?
    };
    let vols = clean_volumes(py, df, n)?;
    let mut params = Vec::with_capacity(n);
    for i in 0..n {
        params.push(PyTuple::new(
            py,
            [
                symbol.clone(),
                date_strs.get_item(i)?,
                opens[i].clone(),
                highs[i].clone(),
                lows[i].clone(),
                closes[i].clone(),
                adj[i].clone(),
                vols[i].clone(),
                interval.clone(),
                source.clone(),
            ],
        )?);
    }
    let force = force.is_truthy()?;
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            let inserted = (|| -> PyResult<()> {
                if slf
                    .call_method1("_has_column", (conn, "daily_ohlcv", "source"))?
                    .is_truthy()?
                {
                    let sql = if force {
                        OHLCV_FORCE_SQL
                    } else {
                        OHLCV_UPSERT_SQL
                    };
                    pc.executemany(&pystr(py, sql), PyList::new(py, &params)?.as_any())?;
                } else {
                    let short = PyList::empty(py);
                    for p in &params {
                        short.append(p.get_slice(0, 9))?;
                    }
                    pc.executemany(&pystr(py, OHLCV_LEGACY_SQL), short.as_any())?;
                }
                Ok(())
            })();
            if let Err(e) = inserted {
                if !e.is_instance_of::<pyo3::exceptions::PyException>(py) {
                    return Err(e);
                }
                log(
                    py,
                    "error",
                    format!(
                        "DB Upsert Error for {} ({n} rows): {}",
                        s(symbol)?,
                        es(py, &e)
                    ),
                )?;
            }
            pc.commit()?;
            let now = pystr(py, &now_iso(py)?);
            pc.run(SYNC_STAMP_SQL, vec![symbol.clone(), now])?;
            pc.commit()
        },
    )?;
    if let Err(exc) = slf.call_method1("upsert_actions", (symbol, df)) {
        if !exc.is_instance_of::<pyo3::exceptions::PyException>(py) {
            return Err(exc);
        }
        log(
            py,
            "warning",
            format!("Actions upsert failed for {}: {}", s(symbol)?, es(py, &exc)),
        )?;
    }
    Ok(())
}

/// `MarketDatabase.upsert_actions`: the Dividends / Stock Splits columns of a
/// fetched frame, as corporate-action rows.
#[pyfunction]
pub fn mdb_upsert_actions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    df: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    if df.is_none() || df.getattr("empty")?.is_truthy()? {
        return Ok(0);
    }
    let date_strs = index_days(py, &df.getattr("index")?)?;
    let now = pystr(py, &now_iso(py)?);
    let none = py.None().into_bound(py);
    let mut params: Vec<Bound<'py, PyTuple>> = Vec::new();
    let columns = df.getattr("columns")?;
    if columns.contains("Stock Splits")? {
        for (i, ratio) in f64_column(&df.get_item("Stock Splits")?)?
            .into_iter()
            .enumerate()
        {
            // 0 and NaN both mean "no split on this day".
            if ratio == 0.0 || ratio.is_nan() || ratio == 1.0 {
                continue;
            }
            if !(MIN_SPLIT_RATIO..=MAX_SPLIT_RATIO).contains(&ratio) {
                log(
                    py,
                    "warning",
                    format!(
                        "Actions: ignoring out-of-range split ratio {} for {} on {}",
                        s(&numpy_float(py, ratio)?)?,
                        s(symbol)?,
                        s(&date_strs.get_item(i)?)?
                    ),
                )?;
                continue;
            }
            params.push(PyTuple::new(
                py,
                [
                    symbol.clone(),
                    date_strs.get_item(i)?,
                    pystr(py, "split"),
                    PyFloat::new(py, ratio).into_any(),
                    none.clone(),
                    source.clone(),
                    now.clone(),
                ],
            )?);
        }
    }
    if columns.contains("Dividends")? {
        for (i, amount) in f64_column(&df.get_item("Dividends")?)?
            .into_iter()
            .enumerate()
        {
            if amount == 0.0 || amount.is_nan() || amount < 0.0 {
                continue;
            }
            params.push(PyTuple::new(
                py,
                [
                    symbol.clone(),
                    date_strs.get_item(i)?,
                    pystr(py, "dividend"),
                    PyFloat::new(py, amount).into_any(),
                    none.clone(),
                    source.clone(),
                    now.clone(),
                ],
            )?);
        }
    }
    if params.is_empty() {
        return Ok(0);
    }
    let lock = slf.getattr("_write_lock")?;
    let count = params.len();
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            pc.executemany(
                &pystr(
                    py,
                    "
                INSERT OR REPLACE INTO corporate_action
                (symbol, date, kind, value, currency, source, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ",
                ),
                PyList::new(py, &params)?.as_any(),
            )?;
            pc.commit()
        },
    )?;
    Ok(count)
}

/// A NumPy float64 scalar, for log lines the original formatted from one.
fn numpy_float<'py>(py: Python<'py>, x: f64) -> PyResult<Bound<'py, PyAny>> {
    py.import("numpy")?.getattr("float64")?.call1((x,))
}

/// `MarketDatabase.set_price_basis`
#[pyfunction]
pub fn mdb_set_price_basis<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    basis: &Bound<'py, PyAny>,
    conn: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let m = module(py)?;
    let valid = PyTuple::new(py, [m.getattr("BASIS_RAW")?, m.getattr("BASIS_SPLIT_ADJ")?])?;
    if !valid.contains(basis)? {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown price basis: {}",
            s(basis)?
        )));
    }
    let sql = "
            INSERT INTO sync_metadata (symbol, price_basis) VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET price_basis = excluded.price_basis
        ";
    if !conn.is_none() {
        PyConn::new(conn)?.run(sql, vec![symbol.clone(), basis.clone()])?;
        return Ok(());
    }
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |own| {
            let pc = PyConn::new(own)?;
            pc.run(sql, vec![symbol.clone(), basis.clone()])?;
            pc.commit()
        },
    )?;
    Ok(())
}

/// `MarketDatabase.upsert_fx`
#[pyfunction]
pub fn mdb_upsert_fx<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    pair: &Bound<'py, PyAny>,
    df: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if df.getattr("empty")?.is_truthy()? {
        return Ok(());
    }
    let columns = df.getattr("columns")?;
    let col = if columns.contains("Close")? {
        pystr(py, "Close")
    } else if !df.getattr("empty")?.is_truthy()? {
        columns.get_item(0)?
    } else {
        py.None().into_bound(py)
    };
    if !col.is_truthy()? {
        return Ok(());
    }
    let date_strs = index_days(py, &df.getattr("index")?)?;
    let rates = f64_column(&df.get_item(&col)?)?;
    let rows = PyList::empty(py);
    for (d, r) in date_strs.iter().zip(rates) {
        rows.append(PyTuple::new(py, [d, PyFloat::new(py, r).into_any()])?)?;
    }
    let kwargs = kw(
        py,
        &[("interval", interval.clone()), ("source", source.clone())],
    )?;
    slf.call_method("upsert_fx_rows", (pair, rows), Some(&kwargs))?;
    Ok(())
}

/// `MarketDatabase.upsert_fx_rows`: (yyyy-MM-dd, rate) pairs, how many written.
#[pyfunction]
pub fn mdb_upsert_fx_rows<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    pair: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
    fill_only: &Bound<'py, PyAny>,
) -> PyResult<i64> {
    let float = py.get_type::<PyFloat>();
    let slice10 = py
        .import("builtins")?
        .getattr("slice")?
        .call1((py.None(), 10))?;
    let mut clean: Vec<Bound<'py, PyTuple>> = Vec::new();
    for item in rows.try_iter()? {
        let (day, rate) = unpack2(&item?)?;
        if rate.is_none() {
            continue;
        }
        let value: f64 = match float.call1((&rate,)) {
            Ok(v) => v.extract()?,
            Err(e)
                if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                    || e.is_instance_of::<pyo3::exceptions::PyValueError>(py) =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        if value.is_nan() || value <= 0.0 {
            continue;
        }
        clean.push(PyTuple::new(
            py,
            [
                pair.clone(),
                day.str()?.get_item(&slice10)?,
                PyFloat::new(py, value).into_any(),
                interval.clone(),
                source.clone(),
            ],
        )?);
    }
    if clean.is_empty() {
        return Ok(0);
    }
    let verb = if fill_only.is_truthy()? {
        "INSERT OR IGNORE"
    } else {
        "INSERT OR REPLACE"
    };
    let lock = slf.getattr("_write_lock")?;
    let written = with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            let rows = if slf
                .call_method1("_has_column", (conn, "daily_fx", "source"))?
                .is_truthy()?
            {
                let sql = format!(
                    "{verb} INTO daily_fx (pair, date, rate, interval, source)\n                          VALUES (?, ?, ?, ?, ?)"
                );
                pc.executemany(&pystr(py, &sql), PyList::new(py, &clean)?.as_any())?
            } else {
                let sql = format!(
                    "{verb} INTO daily_fx (pair, date, rate, interval)\n                          VALUES (?, ?, ?, ?)"
                );
                let short = PyList::empty(py);
                for c in &clean {
                    short.append(c.get_slice(0, 4))?;
                }
                pc.executemany(&pystr(py, &sql), short.as_any())?
            };
            let written = rows.rowcount();
            pc.commit()?;
            Ok(written)
        },
    )?;
    Ok(written.filter(|w| *w >= 0).unwrap_or(0))
}

/// `MarketDatabase._has_column`: a PRAGMA lookup cached on the instance.
#[pyfunction]
pub fn mdb_has_column<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    conn: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
    column: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let key = PyTuple::new(py, [table, column])?;
    let cache = slf
        .getattr("__dict__")?
        .call_method1("setdefault", ("_column_cache", PyDict::new(py)))?;
    if !cache.contains(&key)? {
        let rows = PyConn::new(conn)?.run(&format!("PRAGMA table_info({})", s(table)?), vec![])?;
        let mut found = false;
        for row in rows.all(py) {
            if row.bind(py).get_item(1)?.eq(column)? {
                found = true;
                break;
            }
        }
        cache.set_item(&key, found)?;
    }
    cache.get_item(&key)?.is_truthy()
}

/// `MarketDatabase.upsert_fund_nav`
#[pyfunction]
pub fn mdb_upsert_fund_nav<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    fund_code: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    currency: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    let float = py.get_type::<PyFloat>();
    let params = PyList::empty(py);
    for item in rows.try_iter()? {
        let (day, nav) = unpack2(&item?)?;
        if day.is_truthy()? && !nav.is_none() {
            params.append(PyTuple::new(
                py,
                [
                    fund_code.clone(),
                    day,
                    float.call1((&nav,))?,
                    currency.clone(),
                    source.clone(),
                ],
            )?)?;
        }
    }
    if params.is_empty() {
        return Ok(0);
    }
    let count = params.len();
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            pc.executemany(
                &pystr(
                    py,
                    "
                INSERT OR REPLACE INTO fund_nav
                (fund_code, date, nav, currency, source)
                VALUES (?, ?, ?, ?, ?)
                ",
                ),
                params.as_any(),
            )?;
            pc.commit()
        },
    )?;
    Ok(count)
}

/// `MarketDatabase.upsert_share_counts`
#[pyfunction]
pub fn mdb_upsert_share_counts<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
    source: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    let float = py.get_type::<PyFloat>();
    let params = PyList::empty(py);
    for item in rows.call_method0("items")?.try_iter()? {
        let item = item?;
        let (symbol, shares) = (item.get_item(0)?, item.get_item(1)?);
        if shares.is_truthy()? && float.call1((&shares,))?.extract::<f64>()? > 0.0 {
            params.append(PyTuple::new(
                py,
                [
                    symbol,
                    float.call1((&shares,))?,
                    as_of.call_method0("isoformat")?,
                    source.clone(),
                ],
            )?)?;
        }
    }
    if params.is_empty() {
        return Ok(0);
    }
    let count = params.len();
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            pc.executemany(
                &pystr(
                    py,
                    "
                INSERT INTO share_count (symbol, shares, as_of, source)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(symbol) DO UPDATE SET
                    shares = excluded.shares,
                    as_of = excluded.as_of,
                    source = excluded.source
                ",
                ),
                params.as_any(),
            )?;
            pc.commit()
        },
    )?;
    Ok(count)
}

/// `MarketDatabase.upsert_intraday`: one row at a time, as the original
/// (each row's failure is logged and the rest still written).
#[pyfunction]
pub fn mdb_upsert_intraday<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    df: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if df.getattr("empty")?.is_truthy()? {
        return Ok(());
    }
    let pd = py.import("pandas")?;
    let float = py.get_type::<PyFloat>();
    let int = py.import("builtins")?.getattr("int")?;
    let clean = |val: Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        if pd.call_method1("isna", (&val,))?.is_truthy()? {
            Ok(py.None().into_bound(py))
        } else {
            float.call1((val,))
        }
    };
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || slf.call_method0("_get_connection"),
        |conn| {
            let pc = PyConn::new(conn)?;
            for pair in df.call_method0("iterrows")?.try_iter()? {
                let pair = pair?;
                let (timestamp, row) = (pair.get_item(0)?, pair.get_item(1)?);
                let ts_str = timestamp.call_method0("isoformat")?;
                let open_val = clean(row.call_method1("get", ("Open",))?)?;
                let high_val = clean(row.call_method1("get", ("High",))?)?;
                let low_val = clean(row.call_method1("get", ("Low",))?)?;
                let close_val = clean(
                    row.call_method1("get", ("Close", row.call_method1("get", ("price",))?))?,
                )?;
                let adj_close_val = clean(row.call_method1("get", ("Adj Close", &close_val))?)?;
                let vol = row.call_method1("get", ("Volume", 0))?;
                let volume_val = if pd.call_method1("notna", (&vol,))?.is_truthy()? {
                    int.call1((row.call_method1("get", ("Volume", 0))?,))?
                } else {
                    0i64.into_pyobject(py)?.into_any()
                };
                let written = pc.run(
                    "
                        INSERT OR REPLACE INTO intraday_ohlcv 
                        (symbol, timestamp, open, high, low, close, adj_close, volume, interval)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ",
                    vec![
                        symbol.clone(),
                        ts_str.clone(),
                        open_val,
                        high_val,
                        low_val,
                        close_val,
                        adj_close_val,
                        volume_val,
                        interval.clone(),
                    ],
                );
                if let Err(e) = written {
                    if !e.is_instance_of::<pyo3::exceptions::PyException>(py) {
                        return Err(e);
                    }
                    log(
                        py,
                        "error",
                        format!(
                            "DB Intraday Upsert Error for {} on {}: {}",
                            s(symbol)?,
                            s(&ts_str)?,
                            es(py, &e)
                        ),
                    )?;
                }
            }
            pc.commit()
        },
    )?;
    Ok(())
}

// --- reads -------------------------------------------------------------------

fn conn_of<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    slf.call_method0("_get_connection")
}

/// `with self._get_connection() as conn: body(conn)`
fn with_conn<'py, T>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<Option<T>> {
    let conn = conn_of(slf)?;
    with1(py, &conn, body)
}

fn read_frame<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &str,
    params: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = conn.py();
    crate::pyconn::read_frame(conn, &pystr(py, sql), Some(&params), None)
}

fn numeric_columns(df: &Bound<'_, PyAny>, cols: &[&str]) -> PyResult<()> {
    let py = df.py();
    let pd = py.import("pandas")?;
    for c in cols {
        let v = pd.call_method(
            "to_numeric",
            (df.get_item(*c)?,),
            Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
        )?;
        setitem(df, *c, v)?;
    }
    Ok(())
}

const OHLCV_NAMES: [&str; 6] = ["Open", "High", "Low", "Close", "Adj Close", "Volume"];

fn check_adjustment(py: Python<'_>, adjust: &Bound<'_, PyAny>) -> PyResult<()> {
    let valid = module(py)?.getattr("VALID_ADJUSTMENTS")?;
    if !valid.contains(adjust)? {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown adjustment {}; expected one of {}",
            adjust.repr()?,
            s(&valid)?
        )));
    }
    Ok(())
}

/// `_rust_reader_bound`
#[pyfunction]
pub fn mdb_rust_reader_bound(py: Python<'_>) -> PyResult<bool> {
    let m = module(py)?;
    if m.getattr("_RUST_READER")?.is_none() {
        let bound = (|| -> PyResult<bool> {
            let file = py.import("_sqlite3")?.getattr("__file__")?;
            Ok(crate::sqlite_read::bind(&file.extract::<String>()?))
        })();
        match bound {
            Ok(b) => m.setattr("_RUST_READER", b)?,
            Err(exc) => {
                if !exc.is_instance_of::<pyo3::exceptions::PyException>(py) {
                    return Err(exc);
                }
                log(
                    py,
                    "info",
                    format!("MarketDB: Rust price reader unavailable ({})", es(py, &exc)),
                )?;
                m.setattr("_RUST_READER", false)?;
            }
        }
    }
    m.getattr("_RUST_READER")?.is_truthy()
}

/// `_read_daily_frames`: {symbol: frame} from the Rust reader, or None for
/// the pandas path.
#[pyfunction]
pub fn mdb_read_daily_frames<'py>(
    py: Python<'py>,
    db_path: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    if !symbols.is_exact_instance_of::<PyList>() || !interval.is_exact_instance_of::<PyString>() {
        return Ok(py.None());
    }
    let mut names: Vec<String> = Vec::new();
    for sym in symbols.try_iter()? {
        let sym = sym?;
        if !sym.is_exact_instance_of::<PyString>() {
            return Ok(py.None());
        }
        names.push(sym.extract()?);
    }
    if !module(py)?
        .getattr("_rust_reader_bound")?
        .call0()?
        .is_truthy()?
    {
        return Ok(py.None());
    }
    let (path, interval, start, end): (String, String, String, String) = (
        db_path.extract()?,
        interval.extract()?,
        start.extract()?,
        end.extract()?,
    );
    let outcome =
        py.detach(|| crate::sqlite_read::read_daily_ohlcv(&path, &names, &interval, &start, &end));
    let batch = match outcome {
        Err(e) => {
            log(
                py,
                "info",
                format!(
                    "MarketDB: Rust price read declined ({}); reading via pandas",
                    e.0
                ),
            )?;
            return Ok(py.None());
        }
        Ok(crate::sqlite_read::Outcome::Unsupported(_)) => return Ok(py.None()),
        Ok(crate::sqlite_read::Outcome::Batch(b)) => b,
    };
    let pd = py.import("pandas")?;
    let stamps = Array1::from_vec(batch.dates_ns)
        .into_pyarray(py)
        .call_method1("view", ("M8[ns]",))?;
    let columns: Vec<Bound<'py, PyAny>> = batch
        .columns
        .into_iter()
        .map(|c| match c {
            crate::sqlite_read::Column::Int(v) => Array1::from_vec(v).into_pyarray(py).into_any(),
            crate::sqlite_read::Column::Real(v) => Array1::from_vec(v).into_pyarray(py).into_any(),
        })
        .collect();
    let frames = PyDict::new(py);
    for (k, symbol) in batch.symbols.iter().enumerate() {
        let (a, b) = (batch.offsets[k] as isize, batch.offsets[k + 1] as isize);
        let slice = py.import("builtins")?.getattr("slice")?.call1((a, b))?;
        let data = PyDict::new(py);
        for (name, col) in OHLCV_NAMES.iter().zip(&columns) {
            data.set_item(*name, col.get_item(&slice)?)?;
        }
        let index = pd.call_method(
            "DatetimeIndex",
            (stamps.get_item(&slice)?,),
            Some(&kw(py, &[("name", pystr(py, "date"))])?),
        )?;
        let frame = pd.call_method("DataFrame", (data,), Some(&kw(py, &[("index", index)])?))?;
        frames.set_item(symbol, frame)?;
    }
    Ok(frames.into_any().unbind())
}

/// `MarketDatabase.get_actions`
#[pyfunction]
pub fn mdb_get_actions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    kind: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let symbols = PyList::new(py, symbols.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    if symbols.is_empty() {
        return Ok(PyDict::new(py));
    }
    let marks = vec!["?"; symbols.len()].join(", ");
    let mut query =
        format!("SELECT symbol, date, kind, value FROM corporate_action WHERE symbol IN ({marks})");
    let params = PyList::new(py, symbols.iter())?;
    if kind.is_truthy()? {
        query.push_str(" AND kind = ?");
        params.append(kind)?;
    }
    query.push_str(" ORDER BY symbol, date ASC");
    let frame = with_conn(py, slf, |conn| read_frame(conn, &query, params.into_any()))?
        .expect("no CM swallows");
    let out = PyDict::new(py);
    if frame.getattr("empty")?.is_truthy()? {
        return Ok(out);
    }
    for pair in frame.call_method1("groupby", ("symbol",))?.try_iter()? {
        let pair = pair?;
        out.set_item(pair.get_item(0)?, pair.get_item(1)?)?;
    }
    Ok(out)
}

/// `MarketDatabase.get_price_basis`: {symbol: basis}, legacy by default.
#[pyfunction]
pub fn mdb_get_price_basis<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let symbols = PyList::new(py, symbols.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let out = PyDict::new(py);
    if symbols.is_empty() {
        return Ok(out);
    }
    let marks = vec!["?"; symbols.len()].join(", ");
    let legacy = module(py)?.getattr("BASIS_SPLIT_ADJ")?;
    for sym in symbols.iter() {
        out.set_item(sym, &legacy)?;
    }
    let read = with_conn(py, slf, |conn| {
        let rows = PyConn::new(conn)?.execute(
            &pystr(
                py,
                &format!("SELECT symbol, price_basis FROM sync_metadata WHERE symbol IN ({marks})"),
            ),
            symbols.as_any(),
        )?;
        for row in rows.all(py) {
            let (symbol, basis) = unpack2(row.bind(py))?;
            if basis.is_truthy()? {
                out.set_item(symbol, basis)?;
            }
        }
        Ok(())
    });
    if let Err(exc) = read {
        if !exc.is_instance_of::<pyo3::exceptions::PyException>(py) {
            return Err(exc);
        }
        log(
            py,
            "debug",
            format!(
                "price_basis unavailable ({}); assuming legacy basis",
                es(py, &exc)
            ),
        )?;
    }
    Ok(out)
}

/// The basis and actions read-time work of `get_ohlcv` / `get_ohlcv_batch`.
fn adjusted<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    df: Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    adjust: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let legacy = module(py)?.getattr("BASIS_SPLIT_ADJ")?;
    let basis = slf
        .call_method1("get_price_basis", (PyList::new(py, [symbol])?,))?
        .call_method1("get", (symbol, &legacy))?;
    if !basis.eq(&legacy)? || !adjust.eq(ADJUST_SPLIT)? {
        let actions = slf
            .call_method1("get_actions", (PyList::new(py, [symbol])?,))?
            .call_method1("get", (symbol,))?;
        return slf.call_method1("_adjust_frame", (df, symbol, basis, adjust, actions));
    }
    Ok(df)
}

/// `MarketDatabase.get_ohlcv`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mdb_get_ohlcv<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    adjust: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    check_adjustment(py, adjust)?;
    let (start, end) = (start_date.str()?.into_any(), end_date.str()?.into_any());
    let frames = module(py)?.getattr("_read_daily_frames")?.call1((
        slf.getattr("db_path")?,
        PyList::new(py, [symbol])?,
        interval,
        &start,
        &end,
    ))?;
    if frames.is_truthy()?
        && symbol.is_exact_instance_of::<PyString>()
        && frames.contains(symbol)?
    {
        return adjusted(py, slf, frames.get_item(symbol)?, symbol, adjust);
    }
    // No rows (whose empty frame keeps the SQL column names), or rows only
    // pandas reads correctly: the original read.
    let query = "
            SELECT date, open, high, low, close, adj_close, volume
            FROM daily_ohlcv
            WHERE symbol = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ";
    let params = PyTuple::new(py, [symbol.clone(), interval.clone(), start, end])?.into_any();
    let df = with_conn(py, slf, |conn| read_frame(conn, query, params))?.expect("no CM swallows");
    if !df.getattr("empty")?.is_truthy()? {
        let pd = py.import("pandas")?;
        setitem(
            &df,
            "date",
            pd.call_method1("to_datetime", (df.get_item("date")?,))?,
        )?;
        df.call_method(
            "set_index",
            ("date",),
            Some(&kw(
                py,
                &[("inplace", true.into_pyobject(py)?.to_owned().into_any())],
            )?),
        )?;
        df.setattr("columns", PyList::new(py, OHLCV_NAMES)?)?;
        numeric_columns(&df, &OHLCV_NAMES)?;
        return adjusted(py, slf, df, symbol, adjust);
    }
    Ok(df)
}

/// `MarketDatabase.get_ohlcv_batch`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn mdb_get_ohlcv_batch<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
    adjust: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    check_adjustment(py, adjust)?;
    if !symbols.is_truthy()? {
        return Ok(PyDict::new(py).into_any());
    }
    let legacy = module(py)?.getattr("BASIS_SPLIT_ADJ")?;
    let (start, end) = (start_date.str()?.into_any(), end_date.str()?.into_any());
    let frames = module(py)?.getattr("_read_daily_frames")?.call1((
        slf.getattr("db_path")?,
        symbols,
        interval,
        &start,
        &end,
    ))?;
    let not_default = !adjust.eq(ADJUST_SPLIT)?;
    if !frames.is_none() {
        if !frames.is_truthy()? {
            return Ok(PyDict::new(py).into_any());
        }
        let bases = slf.call_method1("get_price_basis", (symbols,))?;
        let needs_work = PyList::empty(py);
        for sym in frames.try_iter()? {
            let sym = sym?;
            if !bases.call_method1("get", (&sym, &legacy))?.eq(&legacy)? || not_default {
                needs_work.append(sym)?;
            }
        }
        let actions = if needs_work.is_truthy()? {
            slf.call_method1("get_actions", (&needs_work,))?
        } else {
            PyDict::new(py).into_any()
        };
        for sym in needs_work.iter() {
            let adjusted_frame = slf.call_method1(
                "_adjust_frame",
                (
                    frames.get_item(&sym)?,
                    &sym,
                    bases.call_method1("get", (&sym, &legacy))?,
                    adjust,
                    actions.call_method1("get", (&sym,))?,
                ),
            )?;
            frames.set_item(&sym, adjusted_frame)?;
        }
        return Ok(frames);
    }
    let marks = vec!["?"; symbols.len()?].join(", ");
    let query = format!(
        "
            SELECT symbol, date, open, high, low, close, adj_close, volume 
            FROM daily_ohlcv 
            WHERE symbol IN ({marks}) AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY symbol, date ASC
        "
    );
    let results = PyDict::new(py);
    let df_all = with_conn(py, slf, |conn| {
        // `symbols + [interval, ...]`, inside the `with` as in the original
        let params = symbols.add(PyList::new(py, [interval.clone(), start, end])?)?;
        read_frame(conn, &query, params)
    })?
    .expect("no CM swallows");
    if !df_all.getattr("empty")?.is_truthy()? {
        let pd = py.import("pandas")?;
        setitem(
            &df_all,
            "date",
            pd.call_method1("to_datetime", (df_all.get_item("date")?,))?,
        )?;
        let bases = slf.call_method1("get_price_basis", (symbols,))?;
        let needs_work = PyList::empty(py);
        for sym in df_all
            .get_item("symbol")?
            .call_method0("unique")?
            .try_iter()?
        {
            let sym = sym?;
            if !bases.call_method1("get", (&sym, &legacy))?.eq(&legacy)? || not_default {
                needs_work.append(sym)?;
            }
        }
        let actions = if needs_work.is_truthy()? {
            slf.call_method1("get_actions", (&needs_work,))?
        } else {
            PyDict::new(py).into_any()
        };
        for pair in df_all.call_method1("groupby", ("symbol",))?.try_iter()? {
            let pair = pair?;
            let (sym, group) = (pair.get_item(0)?, pair.get_item(1)?);
            let df = group.call_method(
                "drop",
                (),
                Some(&kw(
                    py,
                    &[("columns", PyList::new(py, ["symbol"])?.into_any())],
                )?),
            )?;
            df.call_method(
                "set_index",
                ("date",),
                Some(&kw(
                    py,
                    &[("inplace", true.into_pyobject(py)?.to_owned().into_any())],
                )?),
            )?;
            df.setattr("columns", PyList::new(py, OHLCV_NAMES)?)?;
            numeric_columns(&df, &OHLCV_NAMES)?;
            let df = if needs_work.contains(&sym)? {
                slf.call_method1(
                    "_adjust_frame",
                    (
                        df,
                        &sym,
                        bases.call_method1("get", (&sym, &legacy))?,
                        adjust,
                        actions.call_method1("get", (&sym,))?,
                    ),
                )?
            } else {
                df
            };
            results.set_item(sym, df)?;
        }
    }
    Ok(results.into_any())
}

/// `MarketDatabase.get_intraday`
#[pyfunction]
pub fn mdb_get_intraday<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    start_ts: &Bound<'py, PyAny>,
    end_ts: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let query = "
            SELECT timestamp, open, high, low, close, adj_close, volume 
            FROM intraday_ohlcv 
            WHERE symbol = ? AND interval = ? AND timestamp BETWEEN ? AND ?
            ORDER BY timestamp ASC
        ";
    let start = start_ts.call_method0("isoformat")?;
    let end = end_ts.call_method0("isoformat")?;
    let params = PyTuple::new(py, [symbol.clone(), interval.clone(), start, end])?.into_any();
    let df = with_conn(py, slf, |conn| read_frame(conn, query, params))?.expect("no CM swallows");
    if !df.getattr("empty")?.is_truthy()? {
        let pd = py.import("pandas")?;
        let stamps = pd.call_method(
            "to_datetime",
            (df.get_item("timestamp")?,),
            Some(&kw(
                py,
                &[
                    ("format", pystr(py, "ISO8601")),
                    ("errors", pystr(py, "coerce")),
                    ("utc", true.into_pyobject(py)?.to_owned().into_any()),
                ],
            )?),
        )?;
        setitem(&df, "timestamp", stamps)?;
        df.call_method(
            "set_index",
            ("timestamp",),
            Some(&kw(
                py,
                &[("inplace", true.into_pyobject(py)?.to_owned().into_any())],
            )?),
        )?;
        df.getattr("index")?.setattr("name", "Date")?;
        df.setattr("columns", PyList::new(py, OHLCV_NAMES)?)?;
        numeric_columns(&df, &OHLCV_NAMES)?;
    }
    Ok(df)
}

/// `MarketDatabase.get_data_quality`: per-symbol summary of flagged defects.
#[pyfunction]
pub fn mdb_get_data_quality<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let mut query =
        String::from("SELECT symbol, kind, severity, occurred_on, detail FROM data_quality");
    let mut params = PyList::empty(py);
    if !symbols.is_none() {
        let wanted = PyList::empty(py);
        for sym in symbols.try_iter()? {
            let sym = sym?;
            if sym.is_truthy()? {
                wanted.append(sym)?;
            }
        }
        if wanted.is_empty() {
            return Ok(PyDict::new(py));
        }
        query.push_str(&format!(
            " WHERE symbol IN ({})",
            vec!["?"; wanted.len()].join(",")
        ));
        params = wanted;
    }
    let read = with_conn(py, slf, |conn| {
        Ok(PyConn::new(conn)?
            .execute(&pystr(py, &query), params.as_any())?
            .all(py))
    });
    let rows = match read {
        Ok(rows) => rows.expect("no CM swallows"),
        Err(exc) => {
            if !exc.is_instance_of::<pyo3::exceptions::PyException>(py) {
                return Err(exc);
            }
            log(
                py,
                "debug",
                format!("data_quality unavailable ({})", es(py, &exc)),
            )?;
            return Ok(PyDict::new(py));
        }
    };
    let out = PyDict::new(py);
    for row in rows {
        let row = row.bind(py);
        let fields: Vec<Bound<'py, PyAny>> = row.try_iter()?.collect::<PyResult<_>>()?;
        let [symbol, kind, severity, occurred_on, detail] =
            <[Bound<'py, PyAny>; 5]>::try_from(fields)
                .map_err(|_| pyo3::exceptions::PyValueError::new_err("expected 5 columns"))?;
        let template = PyDict::new(py);
        template.set_item("symbol", &symbol)?;
        template.set_item("severity", "medium")?;
        template.set_item("findings", 0)?;
        template.set_item("kinds", PyList::empty(py))?;
        template.set_item("occurred_on", py.None())?;
        template.set_item("detail", py.None())?;
        let entry = out.call_method1("setdefault", (&symbol, template))?;
        entry.set_item("findings", entry.get_item("findings")?.add(1)?)?;
        let kinds = entry.get_item("kinds")?;
        if !kinds.contains(&kind)? {
            kinds.call_method1("append", (&kind,))?;
        }
        if severity.eq("high")? && !entry.get_item("severity")?.eq("high")? {
            entry.set_item("severity", "high")?;
            entry.set_item("detail", &detail)?;
            entry.set_item("occurred_on", &occurred_on)?;
        } else if entry.get_item("detail")?.is_none() {
            entry.set_item("detail", &detail)?;
            entry.set_item("occurred_on", &occurred_on)?;
        }
    }
    Ok(out)
}

/// `get_fx` / `get_fund_nav`: a dated series with a `price` column.
fn priced_series<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    query: &str,
    params: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let df = with_conn(py, slf, |conn| read_frame(conn, query, params))?.expect("no CM swallows");
    if !df.getattr("empty")?.is_truthy()? {
        let pd = py.import("pandas")?;
        setitem(
            &df,
            "date",
            pd.call_method1("to_datetime", (df.get_item("date")?,))?,
        )?;
        df.call_method(
            "set_index",
            ("date",),
            Some(&kw(
                py,
                &[("inplace", true.into_pyobject(py)?.to_owned().into_any())],
            )?),
        )?;
        df.setattr("columns", PyList::new(py, ["price"])?)?;
    }
    Ok(df)
}

/// `MarketDatabase.get_fx`
#[pyfunction]
pub fn mdb_get_fx<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    pair: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
    interval: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let query = "
            SELECT date, rate 
            FROM daily_fx 
            WHERE pair = ? AND interval = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ";
    let params = PyTuple::new(
        py,
        [
            pair.clone(),
            interval.clone(),
            start_date.str()?.into_any(),
            end_date.str()?.into_any(),
        ],
    )?;
    priced_series(py, slf, query, params.into_any())
}

/// `MarketDatabase.get_fund_nav`
#[pyfunction]
pub fn mdb_get_fund_nav<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    fund_code: &Bound<'py, PyAny>,
    start_date: &Bound<'py, PyAny>,
    end_date: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let query = "
            SELECT date, nav
            FROM fund_nav
            WHERE fund_code = ? AND date BETWEEN ? AND ?
            ORDER BY date ASC
        ";
    let params = PyTuple::new(
        py,
        [
            fund_code.clone(),
            start_date.str()?.into_any(),
            end_date.str()?.into_any(),
        ],
    )?;
    priced_series(py, slf, query, params.into_any())
}

/// Iterate a statement's rows inside `with conn`, swallowing (and logging at
/// debug level) any failure, as the fund NAV and share count readers do.
fn rows_or_debug<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    sql: &str,
    params: Bound<'py, PyAny>,
    what: &str,
    mut each: impl FnMut(&Bound<'py, PyAny>) -> PyResult<()>,
) -> PyResult<()> {
    let read = with_conn(py, slf, |conn| {
        for row in PyConn::new(conn)?
            .execute(&pystr(py, sql), &params)?
            .all(py)
        {
            each(row.bind(py))?;
        }
        Ok(())
    });
    if let Err(exc) = read {
        if !exc.is_instance_of::<pyo3::exceptions::PyException>(py) {
            return Err(exc);
        }
        log(
            py,
            "debug",
            format!("{what} unavailable ({})", es(py, &exc)),
        )?;
    }
    Ok(())
}

/// `MarketDatabase.get_latest_fund_navs`: {FUND_CODE: (date, nav)}.
#[pyfunction]
pub fn mdb_get_latest_fund_navs<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let query = "
            SELECT f.fund_code, f.date, f.nav
            FROM fund_nav AS f
            JOIN (
                SELECT fund_code, MAX(date) AS date
                FROM fund_nav GROUP BY fund_code
            ) AS newest
              ON f.fund_code = newest.fund_code AND f.date = newest.date
        ";
    let out = PyDict::new(py);
    let float = py.get_type::<PyFloat>();
    rows_or_debug(
        py,
        slf,
        query,
        PyTuple::empty(py).into_any(),
        "fund_nav",
        |row| {
            let (code, day, nav) = (row.get_item(0)?, row.get_item(1)?, row.get_item(2)?);
            if nav.is_none() {
                return Ok(());
            }
            out.set_item(
                code.str()?.call_method0("upper")?.call_method0("strip")?,
                PyTuple::new(py, [day.str()?.into_any(), float.call1((nav,))?])?,
            )
        },
    )?;
    Ok(out)
}

/// `MarketDatabase.get_fund_nav_coverage`
#[pyfunction]
pub fn mdb_get_fund_nav_coverage<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let query = "
            SELECT fund_code, MIN(date), MAX(date), COUNT(*)
            FROM fund_nav GROUP BY fund_code
        ";
    let out = PyDict::new(py);
    rows_or_debug(
        py,
        slf,
        query,
        PyTuple::empty(py).into_any(),
        "fund_nav",
        |row| {
            out.set_item(
                row.get_item(0)?,
                PyTuple::new(py, [row.get_item(1)?, row.get_item(2)?, row.get_item(3)?])?,
            )
        },
    )?;
    Ok(out)
}

/// `MarketDatabase.get_share_counts`: entries no older than `max_age_days`.
#[pyfunction]
pub fn mdb_get_share_counts<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    max_age_days: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let symbols = PyList::new(py, symbols.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let out = PyDict::new(py);
    if symbols.is_empty() {
        return Ok(out);
    }
    let query = format!(
        "SELECT symbol, shares, as_of FROM share_count WHERE symbol IN ({})",
        vec!["?"; symbols.len()].join(", ")
    );
    let cutoff = if max_age_days.is_none() {
        py.None().into_bound(py)
    } else {
        // `date.today()` as the original has it (not the market's day).
        let dt = py.import("datetime")?;
        let delta = dt
            .getattr("timedelta")?
            .call((), Some(&kw(py, &[("days", max_age_days.clone())])?))?;
        dt.getattr("date")?
            .call_method0("today")?
            .sub(delta)?
            .call_method0("isoformat")?
    };
    let float = py.get_type::<PyFloat>();
    let empty = pystr(py, "");
    rows_or_debug(py, slf, &query, symbols.into_any(), "share_count", |row| {
        let (symbol, shares, as_of) = (row.get_item(0)?, row.get_item(1)?, row.get_item(2)?);
        let as_of = if as_of.is_truthy()? {
            as_of
        } else {
            empty.clone()
        };
        if cutoff.is_truthy()? && as_of.lt(&cutoff)? {
            return Ok(());
        }
        if shares.is_truthy()? {
            out.set_item(symbol, float.call1((shares,))?)?;
        }
        Ok(())
    })?;
    Ok(out)
}

/// `MarketDatabase.get_latest_closes`: {symbol: most recent close}.
#[pyfunction]
pub fn mdb_get_latest_closes<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let symbols = PyList::new(py, symbols.try_iter()?.collect::<PyResult<Vec<_>>>()?)?;
    let out = PyDict::new(py);
    if symbols.is_empty() {
        return Ok(out);
    }
    let marks = vec!["?"; symbols.len()].join(", ");
    let params = PyList::new(py, symbols.iter())?;
    let mut bound = "";
    if as_of.is_truthy()? {
        bound = "AND date <= ?";
        params.append(as_of.call_method0("isoformat")?)?;
    }
    let query = format!(
        "
            SELECT symbol, close FROM daily_ohlcv
            WHERE (symbol, date) IN (
                SELECT symbol, MAX(date) FROM daily_ohlcv
                WHERE symbol IN ({marks}) AND interval = '1d'
                  AND close IS NOT NULL {bound}
                GROUP BY symbol
            ) AND interval = '1d'
        "
    );
    let float = py.get_type::<PyFloat>();
    with_conn(py, slf, |conn| {
        for row in PyConn::new(conn)?
            .execute(&pystr(py, &query), params.as_any())?
            .all(py)
        {
            let (symbol, close) = unpack2(row.bind(py))?;
            if close.is_truthy()? {
                out.set_item(symbol, float.call1((close,))?)?;
            }
        }
        Ok(())
    })?;
    Ok(out)
}

fn id_column(table: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    Ok(if table.eq("daily_ohlcv")? {
        "symbol"
    } else {
        "pair"
    })
}

fn strptime_day<'py>(py: Python<'py>, text: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    py.import("datetime")?
        .getattr("datetime")?
        .call_method1("strptime", (text, "%Y-%m-%d"))?
        .call_method0("date")
}

/// `MarketDatabase.get_last_date`
#[pyfunction]
pub fn mdb_get_last_date<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let query = format!(
        "SELECT MAX(date) FROM {} WHERE {} = ?",
        s(table)?,
        id_column(table)?
    );
    let found = with_conn(py, slf, |conn| {
        let res = PyConn::new(conn)?
            .run(&query, vec![symbol.clone()])?
            .first(py);
        if let Some(res) = res {
            let first = res.bind(py).get_item(0)?;
            if first.is_truthy()? {
                return Ok(Some(strptime_day(py, &first)?.unbind()));
            }
        }
        Ok(None)
    })?;
    Ok(found.flatten().unwrap_or_else(|| py.None()))
}

/// `get_last_dates` / `get_first_dates`
fn extreme_dates<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
    agg: &str,
) -> PyResult<Bound<'py, PyDict>> {
    let col = id_column(table)?;
    let marks = vec!["?"; symbols.len()?].join(", ");
    let query = format!(
        "SELECT {col}, {agg}(date) FROM {} WHERE {col} IN ({marks}) GROUP BY {col}",
        s(table)?
    );
    let results = PyDict::new(py);
    with_conn(py, slf, |conn| {
        for row in PyConn::new(conn)?
            .execute(&pystr(py, &query), symbols)?
            .all(py)
        {
            let row = row.bind(py);
            if row.get_item(1)?.is_truthy()? {
                results.set_item(row.get_item(0)?, strptime_day(py, &row.get_item(1)?)?)?;
            }
        }
        Ok(())
    })?;
    Ok(results)
}

/// `MarketDatabase.get_last_dates`
#[pyfunction]
pub fn mdb_get_last_dates<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    extreme_dates(py, slf, symbols, table, "MAX")
}

/// `MarketDatabase.get_first_dates`
#[pyfunction]
pub fn mdb_get_first_dates<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    table: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    extreme_dates(py, slf, symbols, table, "MIN")
}

/// `MarketDatabase.get_sync_metadata_batch`
#[pyfunction]
pub fn mdb_get_sync_metadata_batch<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let marks = vec!["?"; symbols.len()?].join(", ");
    let query = format!("SELECT symbol, last_synced FROM sync_metadata WHERE symbol IN ({marks})");
    let results = PyDict::new(py);
    let fromiso = py
        .import("datetime")?
        .getattr("datetime")?
        .getattr("fromisoformat")?;
    with_conn(py, slf, |conn| {
        for row in PyConn::new(conn)?
            .execute(&pystr(py, &query), symbols)?
            .all(py)
        {
            let row = row.bind(py);
            if row.get_item(1)?.is_truthy()? {
                match fromiso.call1((row.get_item(1)?,)) {
                    Ok(v) => results.set_item(row.get_item(0)?, v)?,
                    Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {}
                    Err(e) => return Err(e),
                }
            }
        }
        Ok(())
    })?;
    Ok(results)
}

/// `MarketDatabase.check_integrity`: a fetched frame against the stored bars
/// for the settled days they share.
#[pyfunction]
pub fn mdb_check_integrity<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    new_df: &Bound<'py, PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let ok = || Ok((true, py.None()));
    if new_df.getattr("empty")?.is_truthy()? {
        return ok();
    }
    let live_date = match py
        .import("utils_time")
        .and_then(|m| m.getattr("get_est_today"))
        .and_then(|f| f.call0())
        .and_then(|d| d.call_method0("isoformat"))
    {
        Ok(d) => d,
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyException>(py) => py
            .import("datetime")?
            .getattr("date")?
            .call_method0("today")?
            .call_method0("isoformat")?,
        Err(e) => return Err(e),
    };
    let new_days = module(py)?
        .getattr("_iso_days")?
        .call1((new_df.getattr("index")?,))?;
    let overlap = PyList::empty(py);
    for d in new_days.try_iter()? {
        let d = d?;
        if d.lt(&live_date)? {
            overlap.append(d)?;
        }
    }
    if overlap.is_empty() {
        return ok();
    }
    let builtins = py.import("builtins")?;
    let db_data = with_conn(py, slf, |conn| {
        let guard = if slf
            .call_method1("_has_column", (conn, "daily_ohlcv", "source"))?
            .is_truthy()?
        {
            " AND (source IS NULL OR source = 'yahoo')"
        } else {
            ""
        };
        let query = format!(
            "
                SELECT date, adj_close, close
                FROM daily_ohlcv
                WHERE symbol = ? AND date BETWEEN ? AND ?{guard}
            "
        );
        let params = PyList::new(
            py,
            [
                symbol.clone(),
                builtins.getattr("min")?.call1((&overlap,))?,
                builtins.getattr("max")?.call1((&overlap,))?,
            ],
        )?;
        read_frame(conn, &query, params.into_any())
    })?
    .expect("no CM swallows");
    if db_data.getattr("empty")?.is_truthy()? {
        return ok();
    }
    let pd = py.import("pandas")?;
    let incoming = new_df.call_method0("copy")?;
    setitem(&incoming, "_day", &new_days)?;
    let new_col = if incoming.getattr("columns")?.contains("Adj Close")? {
        "Adj Close"
    } else {
        "Close"
    };
    if !incoming.getattr("columns")?.contains(new_col)? {
        return ok();
    }
    let renames = PyDict::new(py);
    renames.set_item(new_col, "_new")?;
    let right = incoming
        .get_item(PyList::new(py, ["_day", new_col])?)?
        .call_method(
            "rename",
            (),
            Some(&kw(py, &[("columns", renames.into_any())])?),
        )?;
    let merged = db_data.call_method(
        "merge",
        (right,),
        Some(&kw(
            py,
            &[
                ("left_on", pystr(py, "date")),
                ("right_on", pystr(py, "_day")),
                ("how", pystr(py, "inner")),
            ],
        )?),
    )?;
    if merged.getattr("empty")?.is_truthy()? {
        return ok();
    }
    let coerce = |col: &str| -> PyResult<Bound<'py, PyAny>> {
        pd.call_method(
            "to_numeric",
            (merged.get_item(col)?,),
            Some(&kw(py, &[("errors", pystr(py, "coerce"))])?),
        )
    };
    let db_vals = coerce("adj_close")?;
    let db_vals =
        db_vals.call_method1("where", (db_vals.call_method0("notna")?, coerce("close")?))?;
    let new_vals = coerce("_new")?;
    let valid = db_vals
        .call_method0("notna")?
        .call_method1("__and__", (new_vals.call_method0("notna")?,))?
        .call_method1("__and__", (db_vals.call_method1("__ne__", (0,))?,))?;
    if !valid.call_method0("any")?.is_truthy()? {
        return ok();
    }
    let diffs = db_vals
        .sub(&new_vals)?
        .call_method0("abs")?
        .div(db_vals.call_method0("abs")?)?
        .call_method1("where", (&valid,))?;
    let worst = diffs.call_method0("max")?;
    if pd.call_method1("notna", (&worst,))?.is_truthy()? && worst.gt(0.001)? {
        let at = diffs.call_method0("idxmax")?;
        let row = merged.getattr("loc")?.get_item(&at)?;
        let message = format!(
            "Inconsistency detected on {}: DB={}, New={} (diff={})",
            s(&row.get_item("date")?)?,
            s(&db_vals.get_item(diffs.call_method0("idxmax")?)?)?,
            s(&new_vals.get_item(diffs.call_method0("idxmax")?)?)?,
            s(&builtins.getattr("format")?.call1((&worst, ".4%"))?)?
        );
        return Ok((false, pystr(py, &message).unbind()));
    }
    ok()
}

/// `MarketDatabase._init_db`: the four core tables.
#[pyfunction]
pub fn mdb_init_db<'py>(py: Python<'py>, slf: &Bound<'py, PyAny>) -> PyResult<()> {
    let lock = slf.getattr("_write_lock")?;
    with2(
        py,
        &lock,
        || conn_of(slf),
        |conn| {
            let pc = PyConn::new(conn)?;
            for sql in [
                "
                CREATE TABLE IF NOT EXISTS daily_ohlcv (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    interval TEXT DEFAULT '1d',
                    source TEXT DEFAULT 'yahoo',
                    PRIMARY KEY (symbol, date, interval)
                )
            ",
                "
                CREATE TABLE IF NOT EXISTS daily_fx (
                    pair TEXT,
                    date TEXT,
                    rate REAL,
                    interval TEXT DEFAULT '1d',
                    source TEXT DEFAULT 'yahoo',
                    PRIMARY KEY (pair, date, interval)
                )
            ",
                "
                CREATE TABLE IF NOT EXISTS sync_metadata (
                    symbol TEXT PRIMARY KEY,
                    last_synced TEXT,
                    inception_date TEXT,
                    info_json TEXT
                )
            ",
                "
                CREATE TABLE IF NOT EXISTS intraday_ohlcv (
                    symbol TEXT,
                    timestamp TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    adj_close REAL,
                    volume INTEGER,
                    interval TEXT,
                    PRIMARY KEY (symbol, timestamp, interval)
                )
            ",
            ] {
                pc.run(sql, vec![])?;
            }
            pc.commit()
        },
    )?;
    Ok(())
}
