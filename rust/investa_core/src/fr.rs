//! The statement and ratio glue of `financial_ratios`: statement lookups with
//! their aliases, free cash flow from a statement, trailing-twelve-month
//! columns, the point-in-time price, the ratio history (its resolved-lookup
//! path and the original's cell-by-cell fallback), historical multiples, the
//! current valuation ratios, and the WACC's input reading.
//!
//! Ported line for line from `src/financial_ratios.py` (frozen in
//! `tests/reference/financial_ratios_p15.py`). Every figure is combined with
//! Python's own operations in the original's operand order, so `x or y` falls
//! through a filed zero exactly as it did and a NumPy scalar stays one. The
//! arithmetic kernels (`key_ratios`, `wacc`, `scalar_projection`, the NumPy
//! statistics) were already Rust and are called as the original called them;
//! siblings and constants are looked up on `financial_ratios` at call time.
//! `_StatementLookup` is a Rust class here, filling its rows natively.

use std::collections::HashMap;

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyAttributeError, PyException, PyTypeError};
use pyo3::intern;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PySlice, PyTuple};

use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

pub(crate) fn fr(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("financial_ratios")
}

pub(crate) fn none(py: Python<'_>) -> Bound<'_, PyAny> {
    py.None().into_bound(py)
}

pub(crate) fn fl(py: Python<'_>, v: f64) -> Bound<'_, PyAny> {
    PyFloat::new(py, v).into_any()
}

pub(crate) fn pybool(py: Python<'_>, v: bool) -> Bound<'_, PyAny> {
    PyBool::new(py, v).to_owned().into_any()
}

pub(crate) fn int(py: Python<'_>, v: i64) -> PyResult<Bound<'_, PyAny>> {
    Ok(v.into_pyobject(py)?.into_any())
}

pub(crate) fn builtin<'py>(py: Python<'py>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    py.import("builtins")?.getattr(name)
}

/// `a or b()`
pub(crate) fn or_else<'py>(
    a: Bound<'py, PyAny>,
    b: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if a.is_truthy()? { Ok(a) } else { b() }
}

/// `x or default` for a float default.
pub(crate) fn or_f<'py>(x: &Bound<'py, PyAny>, default: f64) -> PyResult<Bound<'py, PyAny>> {
    if x.is_truthy()? {
        Ok(x.clone())
    } else {
        Ok(fl(x.py(), default))
    }
}

/// `getattr(obj, name, None)`
pub(crate) fn getattr_or_none<'py>(
    obj: &Bound<'py, PyAny>,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    match obj.getattr(name) {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyAttributeError>(obj.py()) => Ok(none(obj.py())),
        Err(e) => Err(e),
    }
}

/// `df is None or df.empty`
pub(crate) fn none_or_empty(df: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(df.is_none() || df.getattr(intern!(df.py(), "empty"))?.is_truthy()?)
}

pub(crate) fn coerce<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
    kwargs(py, &[("errors", &pystr(py, "coerce"))])
}

// --- statement lookups ----------------------------------------------------

/// `_get_statement_value`: the cell, else the first alias that has one.
pub(crate) fn get_statement_value<'py>(
    m: &Bound<'py, PyModule>,
    df: &Bound<'py, PyAny>,
    item_name: &Bound<'py, PyAny>,
    period_column: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = m.py();
    if none_or_empty(df)? || !df.getattr("columns")?.contains(period_column)? {
        return Ok(none(py));
    }
    let pd = py.import("pandas")?;
    let float = builtin(py, "float")?;
    let cell = |label: &Bound<'py, PyAny>| -> PyResult<Option<Bound<'py, PyAny>>> {
        let val = df
            .getattr("loc")?
            .get_item(PyTuple::new(py, [label.clone(), period_column.clone()])?)?;
        if pd.call_method1("notna", (&val,))?.is_truthy()? {
            return Ok(Some(float.call1((val,))?));
        }
        Ok(None)
    };
    // 1. Exact match
    if df.getattr("index")?.contains(item_name)?
        && let Some(v) = cell(item_name)?
    {
        return Ok(v);
    }
    // 2. Alias fallback
    let aliases = m
        .getattr("STATEMENT_ALIASES")?
        .call_method1("get", (item_name, PyList::empty(py)))?;
    for alias in aliases.try_iter()? {
        let alias = alias?;
        if alias.ne(item_name)?
            && df.getattr("index")?.contains(&alias)?
            && let Some(v) = cell(&alias)?
        {
            return Ok(v);
        }
    }
    Ok(none(py))
}

#[pyfunction]
pub fn fr_get_statement_value<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
    item_name: &Bound<'py, PyAny>,
    period_column: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    get_statement_value(&fr(py)?, df, item_name, period_column)
}

/// `_get_statement_value(df, item, col)` through the module, as the
/// original's global lookup reaches it.
pub(crate) struct Gsv<'py> {
    f: Bound<'py, PyAny>,
}

impl<'py> Gsv<'py> {
    pub(crate) fn new(m: &Bound<'py, PyModule>) -> PyResult<Self> {
        Ok(Gsv {
            f: m.getattr("_get_statement_value")?,
        })
    }
    pub(crate) fn get(
        &self,
        df: &Bound<'py, PyAny>,
        item: &str,
        col: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        self.f.call1((df, item, col))
    }
}

#[pyfunction]
pub fn fr_extract_fcf_from_statement<'py>(
    py: Python<'py>,
    cf_df: &Bound<'py, PyAny>,
    col: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if none_or_empty(cf_df)? || !cf_df.getattr("columns")?.contains(col)? {
        return Ok(none(py));
    }
    let g = Gsv::new(&fr(py)?)?;
    let fcf = g.get(cf_df, "Free Cash Flow", col)?;
    if !fcf.is_none() {
        return Ok(fcf);
    }
    let ocf = g.get(cf_df, "Operating Cash Flow", col)?;
    let capex = g.get(cf_df, "Capital Expenditure", col)?;
    if !ocf.is_none() && !capex.is_none() {
        return if capex.lt(0)? {
            ocf.add(capex)
        } else {
            ocf.sub(capex)
        };
    }
    Ok(none(py))
}

#[pyfunction]
pub fn fr_is_level_row(py: Python<'_>, label: &Bound<'_, PyAny>) -> PyResult<bool> {
    let text = label.str()?;
    for marker in fr(py)?.getattr("_LEVEL_ROW_MARKERS")?.try_iter()? {
        if text.contains(marker?)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[pyfunction]
pub fn fr_is_consecutive_window<'py>(
    py: Python<'py>,
    newest: &Bound<'py, PyAny>,
    oldest: &Bound<'py, PyAny>,
    quarters: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    if pd.call_method1("isna", (newest,))?.is_truthy()?
        || pd.call_method1("isna", (oldest,))?.is_truthy()?
        || quarters.lt(2)?
    {
        return Ok(pybool(py, false));
    }
    let m = fr(py)?;
    let steps = quarters.sub(1)?;
    let days = newest.sub(oldest)?.getattr("days")?;
    // The chained comparison's own result: NumPy's bool for a NumPy count.
    let low = m
        .getattr("_MIN_QUARTER_SPAN_DAYS")?
        .mul(&steps)?
        .rich_compare(&days, CompareOp::Le)?;
    if !low.is_truthy()? {
        return Ok(low);
    }
    days.rich_compare(
        m.getattr("_MAX_QUARTER_SPAN_DAYS")?.mul(&steps)?,
        CompareOp::Le,
    )
}

/// `builtins.slice(a, b)`
fn slice<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    builtin(py, "slice")?.call1((a, b))
}

/// `df.iloc[:, :0]`
fn no_columns<'py>(py: Python<'py>, df: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    df.getattr("iloc")?.get_item(PyTuple::new(
        py,
        [
            PySlice::full(py).into_any(),
            slice(py, &none(py), &int(py, 0)?)?,
        ],
    )?)
}

#[pyfunction]
#[pyo3(signature = (statement_df, min_quarters=None))]
pub fn fr_to_trailing_twelve_months<'py>(
    py: Python<'py>,
    statement_df: &Bound<'py, PyAny>,
    min_quarters: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if none_or_empty(statement_df)? {
        return Ok(statement_df.clone());
    }
    let four = int(py, 4)?;
    let min_quarters = min_quarters.unwrap_or(&four);
    let m = fr(py)?;
    let pd = py.import("pandas")?;

    let periods = builtin(py, "list")?.call1((statement_df.getattr("columns")?,))?;
    let n = int(py, periods.len()? as i64)?;
    if n.lt(min_quarters)? {
        return no_columns(py, statement_df);
    }

    let stamps = pd.getattr("to_datetime")?.call(
        (pd.getattr("Index")?.call1((&periods,))?,),
        Some(&coerce(py)?),
    )?;

    let consecutive = m.getattr("_is_consecutive_window")?;
    let level_row = m.getattr("_is_level_row")?;
    let invert = py.import("operator")?.getattr("invert")?;
    let one = int(py, 1)?;
    let columns = PyDict::new(py);
    let count = n.sub(min_quarters)?.add(&one)?;
    for i in builtin(py, "range")?.call1((count,))?.try_iter()? {
        let i = i?;
        let oldest = i.add(min_quarters)?.sub(&one)?;
        if !consecutive
            .call1((
                stamps.get_item(&i)?,
                stamps.get_item(&oldest)?,
                min_quarters,
            ))?
            .is_truthy()?
        {
            continue;
        }
        let window = statement_df.getattr("iloc")?.get_item(PyTuple::new(
            py,
            [
                PySlice::full(py).into_any(),
                slice(py, &i, &oldest.add(&one)?)?,
            ],
        )?)?;
        // `min_count` keeps a row that no quarter reported as NaN.
        let summed = window.call_method(
            "sum",
            (),
            Some(&kwargs(py, &[("axis", &one), ("min_count", min_quarters)])?),
        )?;
        let averaged = window.call_method("mean", (), Some(&kwargs(py, &[("axis", &one)])?))?;
        let levels = summed
            .getattr("index")?
            .call_method1("map", (&level_row,))?;
        let picked = summed.call_method1("where", (invert.call1((levels,))?, averaged))?;
        columns.set_item(periods.get_item(&i)?, picked)?;
    }

    if !columns.is_truthy()? {
        return no_columns(py, statement_df);
    }
    pd.getattr("DataFrame")?.call1((columns,))
}

#[pyfunction]
pub fn fr_price_on_or_before<'py>(
    py: Python<'py>,
    prices_data: &Bound<'py, PyAny>,
    dt: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if prices_data.is_none() || dt.is_none() {
        return Ok(none(py));
    }
    let pd = py.import("pandas")?;
    let body = || -> PyResult<Bound<'py, PyAny>> {
        let mut series = none(py);
        if prices_data.is_instance(&pd.getattr("DataFrame")?)? {
            for col in ["price", "Price", "Adj Close", "Close", "adj_close", "close"] {
                if prices_data.getattr("columns")?.contains(col)? {
                    series = prices_data.get_item(col)?;
                    break;
                }
            }
            if series.is_none() && !prices_data.getattr("empty")?.is_truthy()? {
                series = prices_data.getattr("iloc")?.get_item(PyTuple::new(
                    py,
                    [PySlice::full(py).into_any(), int(py, 0)?],
                )?)?;
            }
        } else if prices_data.is_instance(&pd.getattr("Series")?)? {
            series = prices_data.clone();
        } else if prices_data.is_instance_of::<PyDict>() {
            series = pd.getattr("Series")?.call1((prices_data,))?;
        }

        if none_or_empty(&series)? {
            return Ok(none(py));
        }

        // Convert index to datetime if needed
        if !series
            .getattr("index")?
            .is_instance(&pd.getattr("DatetimeIndex")?)?
        {
            let idx = pd
                .getattr("to_datetime")?
                .call((series.getattr("index")?,), Some(&coerce(py)?))?;
            series.setattr("index", idx)?;
            let mask = series.getattr("index")?.call_method0("notnull")?;
            series = series.get_item(mask)?;
        }

        // Normalize timezone on index to naive UTC for safe comparison
        if !getattr_or_none(&series.getattr("index")?, "tz")?.is_none() {
            let idx = series.getattr("index")?;
            let naive = match idx
                .call_method1("tz_convert", ("UTC",))
                .and_then(|i| i.call_method1("tz_localize", (none(py),)))
            {
                Ok(v) => v,
                Err(e) if e.is_instance_of::<PyException>(py) => series
                    .getattr("index")?
                    .call_method1("tz_localize", (none(py),))?,
                Err(e) => return Err(e),
            };
            series.setattr("index", naive)?;
        }

        let mut dt_ts = pd.getattr("to_datetime")?.call1((dt,))?;
        if dt_ts.is_none() || pd.call_method1("isna", (&dt_ts,))?.is_truthy()? {
            return Ok(none(py));
        }
        if !getattr_or_none(&dt_ts, "tzinfo")?.is_none() {
            dt_ts = match dt_ts
                .call_method1("tz_convert", ("UTC",))
                .and_then(|t| t.call_method1("tz_localize", (none(py),)))
            {
                Ok(v) => v,
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    dt_ts.call_method1("tz_localize", (none(py),))?
                }
                Err(e) => return Err(e),
            };
        }

        series = series.call_method0("sort_index")?;
        let sub = series
            .getattr("loc")?
            .get_item(slice(py, &none(py), &dt_ts)?)?;
        if sub.getattr("empty")?.is_truthy()? {
            return Ok(none(py));
        }
        let last_dt = sub.getattr("index")?.get_item(-1)?;
        if dt_ts.sub(last_dt)?.getattr("days")?.gt(60)? {
            return Ok(none(py));
        }
        let val = builtin(py, "float")?.call1((sub.getattr("iloc")?.get_item(-1)?,))?;
        if val.gt(0)?
            && py
                .import("numpy")?
                .call_method1("isfinite", (&val,))?
                .is_truthy()?
        {
            Ok(val)
        } else {
            Ok(none(py))
        }
    };
    match body() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("_price_on_or_before error: {}", es(py, &e)),
            )?;
            Ok(none(py))
        }
        Err(e) => Err(e),
    }
}

/// `_opt_number`: None stays None; ints, floats, bools and NumPy numbers
/// convert; anything else raises.
pub(crate) fn opt_number<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let py = value.py();
    if value.is_none() {
        return Ok(none(py));
    }
    if value.is_instance_of::<PyBool>()
        || value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance(&py.import("numpy")?.getattr("number")?)?
    {
        return builtin(py, "float")?.call1((value,));
    }
    Err(PyTypeError::new_err(format!(
        "not a number: {}",
        value.repr()?
    )))
}

#[pyfunction]
pub fn fr_opt_number<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    opt_number(value)
}

/// `type(v) not in (float, int, bool)`
pub(crate) fn not_plain_number(v: &Bound<'_, PyAny>) -> bool {
    let py = v.py();
    let t = v.get_type();
    !(t.is(py.get_type::<PyFloat>())
        || t.is(py.get_type::<PyInt>())
        || t.is(py.get_type::<PyBool>()))
}

#[pyfunction]
pub fn fr_scalar_projection<'py>(
    py: Python<'py>,
    base: &Bound<'py, PyAny>,
    applied_growth: &Bound<'py, PyAny>,
    discount_rate: &Bound<'py, PyAny>,
    projection_years: &Bound<'py, PyAny>,
    terminal_growth_rate: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let years = py
        .import("operator")?
        .call_method1("index", (projection_years,))?;
    let numpy_terms = [base, applied_growth, discount_rate, terminal_growth_rate]
        .iter()
        .any(|v| not_plain_number(v));
    let float = builtin(py, "float")?;
    let args = PyTuple::new(
        py,
        [
            float.call1((base,))?,
            float.call1((applied_growth,))?,
            float.call1((discount_rate,))?,
            years,
            float.call1((terminal_growth_rate,))?,
            pybool(py, numpy_terms),
        ],
    )?;
    let out = fr(py)?
        .getattr("investa_core")?
        .getattr("scalar_projection")?
        .call1(args)?;
    let parts = unpack(&out, 4)?;
    if numpy_terms {
        let f64_t = py.import("numpy")?.getattr("float64")?;
        return PyTuple::new(
            py,
            [
                f64_t.call1((&parts[1],))?,
                f64_t.call1((&parts[2],))?,
                f64_t.call1((&parts[3],))?,
            ],
        );
    }
    PyTuple::new(py, [parts[1].clone(), parts[2].clone(), parts[3].clone()])
}

// --- the ratio history ----------------------------------------------------

/// `next((col for col in df.columns if pd.to_datetime(col, errors="coerce")
/// == period), None)`
fn first_matching_column<'py>(
    pd: &Bound<'py, PyModule>,
    df: &Bound<'py, PyAny>,
    period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = pd.py();
    let to_datetime = pd.getattr("to_datetime")?;
    for col in df.getattr("columns")?.try_iter()? {
        let col = col?;
        if to_datetime.call((&col,), Some(&coerce(py)?))?.eq(period)? {
            return Ok(col);
        }
    }
    Ok(none(py))
}

/// The shared head of both ratio paths: the parsed common periods, or the
/// empty frame the original returns (with its log).
fn common_periods<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
) -> PyResult<Result<Bound<'py, PyAny>, Bound<'py, PyAny>>> {
    let pd = py.import("pandas")?;
    let empty = || pd.getattr("DataFrame")?.call0();
    if none_or_empty(financials_df)? || none_or_empty(balance_sheet_df)? {
        log(
            py,
            "warning",
            "Financials or Balance Sheet data is missing/empty. Cannot calculate historical ratios.",
        )?;
        return Ok(Err(empty()?));
    }
    let parsed = (|| -> PyResult<Option<Bound<'py, PyAny>>> {
        let to_datetime = pd.getattr("to_datetime")?;
        let fin = to_datetime
            .call((financials_df.getattr("columns")?,), Some(&coerce(py)?))?
            .call_method0("dropna")?;
        let bs = to_datetime
            .call((balance_sheet_df.getattr("columns")?,), Some(&coerce(py)?))?
            .call_method0("dropna")?;
        if fin.getattr("empty")?.is_truthy()? || bs.getattr("empty")?.is_truthy()? {
            log(
                py,
                "warning",
                "No valid period columns found in financial statements for ratio calculation.",
            )?;
            return Ok(None);
        }
        let set = builtin(py, "set")?;
        let common = set
            .call1((fin,))?
            .call_method1("intersection", (set.call1((bs,))?,))?;
        Ok(Some(
            builtin(py, "sorted")?.call1((builtin(py, "list")?.call1((common,))?,))?,
        ))
    })();
    let common = match parsed {
        Ok(Some(c)) => c,
        Ok(None) => return Ok(Err(empty()?)),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Could not parse period columns for ratio calculation: {}",
                    es(py, &e)
                ),
            )?;
            return Ok(Err(empty()?));
        }
        Err(e) => return Err(e),
    };
    if !common.is_truthy()? {
        log(
            py,
            "warning",
            "No common periods found between income statement and balance sheet for ratio calculation.",
        )?;
        return Ok(Err(empty()?));
    }
    Ok(Ok(common))
}

fn skip_period_log(py: Python<'_>, period: &Bound<'_, PyAny>) -> PyResult<()> {
    let day = period.call_method1("strftime", ("%Y-%m-%d",))?;
    log(
        py,
        "debug",
        format!(
            "Skipping period {} due to missing original column name match.",
            day.str()?
        ),
    )
}

/// `(a / b) * k if cond else np.nan`
fn ratio_or_nan<'py>(
    nan: &Bound<'py, PyAny>,
    cond: bool,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    k: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    if !cond {
        return Ok(nan.clone());
    }
    let r = a.div(b)?;
    match k {
        Some(k) => r.mul(k),
        None => Ok(r),
    }
}

/// `x and x != 0 and y is not None`
fn nonzero_and_some(x: &Bound<'_, PyAny>, y: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(x.is_truthy()? && x.ne(0)? && !y.is_none())
}

/// `x and x > 0`
fn positive(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(x.is_truthy()? && x.gt(0)?)
}

#[pyfunction]
#[pyo3(signature = (financials_df, balance_sheet_df, cashflow_df=None, periods_per_year=None, prices_df=None))]
pub fn fr_key_ratios_python<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: Option<&Bound<'py, PyAny>>,
    periods_per_year: Option<&Bound<'py, PyAny>>,
    prices_df: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let nothing = none(py);
    let one = int(py, 1)?;
    let cashflow_df = cashflow_df.unwrap_or(&nothing);
    let periods_per_year = periods_per_year.unwrap_or(&one);
    let prices_df = prices_df.unwrap_or(&nothing);
    let common = match common_periods(py, financials_df, balance_sheet_df)? {
        Ok(c) => c,
        Err(empty) => return Ok(empty),
    };

    let m = fr(py)?;
    let pd = py.import("pandas")?;
    let nan = py.import("numpy")?.getattr("nan")?;
    let g = Gsv::new(&m)?;
    let fin = financials_df;
    let bs = balance_sheet_df;
    let cf = cashflow_df;
    let price_on = m.getattr("_price_on_or_before")?;
    let hundred = int(py, 100)?;
    let hundred_f = fl(py, 100.0);
    let abs = builtin(py, "abs")?;
    let rows = PyList::empty(py);

    for (i, period_dt) in common.try_iter()?.enumerate() {
        let period_dt = period_dt?;
        let i_obj = int(py, i as i64)?;
        let period_str_fin = first_matching_column(&pd, fin, &period_dt)?;
        let period_str_bs = first_matching_column(&pd, bs, &period_dt)?;
        if !period_str_fin.is_truthy()? || !period_str_bs.is_truthy()? {
            skip_period_log(py, &period_dt)?;
            continue;
        }
        let (pf, pb) = (&period_str_fin, &period_str_bs);
        let r = PyDict::new(py);
        r.set_item("Period", &period_dt)?;

        // 1. Profitability & Operations
        let revenue = g.get(fin, "Total Revenue", pf)?;
        let cost_of_revenue = g.get(fin, "Cost Of Revenue", pf)?;
        let mut gross_profit = g.get(fin, "Gross Profit", pf)?;
        if gross_profit.is_none() && !revenue.is_none() && !cost_of_revenue.is_none() {
            gross_profit = revenue.sub(&cost_of_revenue)?;
        }
        let mut operating_income = or_else(g.get(fin, "Operating Income", pf)?, || {
            g.get(fin, "Operating Revenue Or Loss", pf)
        })?;
        let ebit = g.get(fin, "Ebit", pf)?;
        if operating_income.is_none() && !ebit.is_none() {
            operating_income = ebit.clone();
        }
        let mut net_income = g.get(fin, "Net Income", pf)?;
        if net_income.is_none() {
            net_income = g.get(fin, "Net Income From Continuing Ops", pf)?;
        }
        let total_equity = or_else(g.get(bs, "Stockholders Equity", pb)?, || {
            or_else(g.get(bs, "Total Stockholder Equity", pb)?, || {
                g.get(bs, "Total Equity Gross Minority Interest", pb)
            })
        })?;
        let total_assets = g.get(bs, "Total Assets", pb)?;

        let (mut avg_equity, mut avg_assets) = (total_equity.clone(), total_assets.clone());
        if i_obj.ge(periods_per_year)? {
            let prev_period_dt = common.get_item(i_obj.sub(periods_per_year)?)?;
            let prev_bs = first_matching_column(&pd, bs, &prev_period_dt)?;
            if prev_bs.is_truthy()? {
                let prev_equity = g.get(bs, "Total Stockholder Equity", &prev_bs)?;
                let prev_assets = g.get(bs, "Total Assets", &prev_bs)?;
                if !total_equity.is_none() && !prev_equity.is_none() {
                    avg_equity = total_equity.add(&prev_equity)?.div(2)?;
                }
                if !total_assets.is_none() && !prev_assets.is_none() {
                    avg_assets = total_assets.add(&prev_assets)?.div(2)?;
                }
            }
        }

        let rev_ok = |y: &Bound<'py, PyAny>| nonzero_and_some(&revenue, y);
        r.set_item(
            "Gross Profit Margin (%)",
            ratio_or_nan(
                &nan,
                rev_ok(&gross_profit)?,
                &gross_profit,
                &revenue,
                Some(&hundred),
            )?,
        )?;
        r.set_item(
            "Operating Margin (%)",
            ratio_or_nan(
                &nan,
                rev_ok(&operating_income)?,
                &operating_income,
                &revenue,
                Some(&hundred),
            )?,
        )?;
        r.set_item(
            "Net Profit Margin (%)",
            ratio_or_nan(
                &nan,
                rev_ok(&net_income)?,
                &net_income,
                &revenue,
                Some(&hundred),
            )?,
        )?;
        r.set_item(
            "Return on Equity (ROE) (%)",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&avg_equity, &net_income)?,
                &net_income,
                &avg_equity,
                Some(&hundred),
            )?,
        )?;
        r.set_item(
            "Return on Assets (ROA) (%)",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&avg_assets, &net_income)?,
                &net_income,
                &avg_assets,
                Some(&hundred),
            )?,
        )?;

        // 2. Liquidity
        let current_assets = or_else(g.get(bs, "Current Assets", pb)?, || {
            g.get(bs, "Total Current Assets", pb)
        })?;
        let current_liabilities = or_else(g.get(bs, "Current Liabilities", pb)?, || {
            g.get(bs, "Total Current Liabilities", pb)
        })?;
        let inventory = g.get(bs, "Inventory", pb)?;
        let liq_ok = nonzero_and_some(&current_liabilities, &current_assets)?;
        r.set_item(
            "Current Ratio",
            ratio_or_nan(&nan, liq_ok, &current_assets, &current_liabilities, None)?,
        )?;
        let quick = if liq_ok {
            let inv = if inventory.is_truthy()? {
                inventory.clone()
            } else {
                int(py, 0)?
            };
            current_assets.sub(inv)?.div(&current_liabilities)?
        } else {
            nan.clone()
        };
        r.set_item("Quick Ratio", quick)?;

        // 3. Solvency
        let total_liab = or_else(
            g.get(bs, "Total Liabilities Net Minority Interest", pb)?,
            || {
                or_else(g.get(bs, "Total Liab", pb)?, || {
                    g.get(bs, "Total Liabilities", pb)
                })
            },
        )?;
        r.set_item(
            "Debt-to-Equity Ratio",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&total_equity, &total_liab)?,
                &total_liab,
                &total_equity,
                None,
            )?,
        )?;
        let lt_debt = or_else(g.get(bs, "Long Term Debt", pb)?, || {
            or_else(
                g.get(bs, "Long Term Debt And Capital Lease Obligation", pb)?,
                || g.get(bs, "Long Term Debt Non Current", pb),
            )
        })?;
        r.set_item(
            "Long-Term Debt to Equity",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&total_equity, &lt_debt)?,
                &lt_debt,
                &total_equity,
                None,
            )?,
        )?;

        let mut interest_exp = g.get(fin, "Interest Expense", pf)?;
        if !interest_exp.is_none() && interest_exp.lt(0)? {
            interest_exp = abs.call1((&interest_exp,))?;
        }
        r.set_item(
            "Interest Coverage Ratio",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&interest_exp, &ebit)?,
                &ebit,
                &interest_exp,
                None,
            )?,
        )?;

        // 4. Efficiency
        r.set_item(
            "Asset Turnover",
            ratio_or_nan(
                &nan,
                nonzero_and_some(&avg_assets, &revenue)?,
                &revenue,
                &avg_assets,
                None,
            )?,
        )?;

        // 5. ROIC
        let pretax = g.get(fin, "Pretax Income", pf)?;
        let tax = g.get(fin, "Tax Provision", pf)?;
        let total_debt = g.get(bs, "Total Debt", pb)?;
        let cash = g.get(bs, "Cash And Cash Equivalents", pb)?;
        let mut roic = nan.clone();
        if !pretax.is_none() && total_equity.is_truthy()? {
            let operating_earnings = pretax.add(or_f(&interest_exp, 0.0)?)?;
            let mut tax_rate = if positive(&pretax)? && !tax.is_none() {
                tax.div(&pretax)?
            } else {
                none(py)
            };
            if tax_rate.is_none() || !(fl(py, 0.0).le(&tax_rate)? && tax_rate.le(0.6)?) {
                tax_rate = fl(py, 0.21);
            }
            let nopat = operating_earnings.mul(fl(py, 1.0).sub(&tax_rate)?)?;
            let invested = total_equity
                .add(or_f(&total_debt, 0.0)?)?
                .sub(or_f(&cash, 0.0)?)?;
            if invested.is_truthy()? && invested.ne(0)? {
                let candidate = nopat.div(&invested)?;
                if fl(py, -2.0).lt(&candidate)? && candidate.lt(3.0)? {
                    roic = candidate.mul(&hundred_f)?;
                }
            }
        }
        r.set_item("Return on Invested Capital (ROIC) (%)", roic)?;

        // 6. Cash Flow & Shares
        r.set_item("Free Cash Flow Margin (%)", &nan)?;
        r.set_item("Diluted Shares Outstanding", &nan)?;
        let (mut ocf, mut capex, mut div_paid) = (none(py), none(py), none(py));
        let mut period_str_cf = none(py);
        if !cf.is_none() && !cf.getattr("empty")?.is_truthy()? {
            period_str_cf = first_matching_column(&pd, cf, &period_dt)?;
            if !period_str_cf.is_none() {
                let pc = &period_str_cf;
                ocf = g.get(cf, "Operating Cash Flow", pc)?;
                capex = g.get(cf, "Capital Expenditure", pc)?;
                div_paid = or_else(g.get(cf, "Common Stock Dividend Paid", pc)?, || {
                    g.get(cf, "Cash Dividends Paid", pc)
                })?;
                if !ocf.is_none() && !capex.is_none() && revenue.is_truthy()? {
                    r.set_item(
                        "Free Cash Flow Margin (%)",
                        ocf.add(&capex)?.div(&revenue)?.mul(&hundred_f)?,
                    )?;
                }
            }
        }

        let shares = or_else(g.get(fin, "Diluted Average Shares", pf)?, || {
            g.get(bs, "Ordinary Shares Number", pb)
        })?;
        if shares.is_truthy()? {
            r.set_item("Diluted Shares Outstanding", &shares)?;
        }

        // 7. Level figures & YoY Growth
        r.set_item("Total Revenue", &revenue)?;
        r.set_item("Operating Income", &operating_income)?;
        r.set_item("Net Income", &net_income)?;

        let mut eps = g.get(fin, "Diluted EPS", pf)?;
        if eps.is_none() && !net_income.is_none() && positive(&shares)? {
            eps = net_income.div(&shares)?;
        }
        r.set_item("Diluted EPS", &eps)?;

        r.set_item("Revenue Growth YoY (%)", &nan)?;
        r.set_item("EPS Growth YoY (%)", &nan)?;
        if i_obj.ge(periods_per_year)? {
            let prev_dt = common.get_item(i_obj.sub(periods_per_year)?)?;
            let prev_fin_col = first_matching_column(&pd, fin, &prev_dt)?;
            if prev_fin_col.is_truthy()? {
                let prev_rev = g.get(fin, "Total Revenue", &prev_fin_col)?;
                if positive(&prev_rev)? && !revenue.is_none() {
                    r.set_item(
                        "Revenue Growth YoY (%)",
                        revenue.div(&prev_rev)?.sub(1)?.mul(&hundred_f)?,
                    )?;
                }
                let prev_eps = g.get(fin, "Diluted EPS", &prev_fin_col)?;
                if positive(&prev_eps)? && !eps.is_none() && eps.gt(0)? {
                    r.set_item(
                        "EPS Growth YoY (%)",
                        eps.div(&prev_eps)?.sub(1)?.mul(&hundred_f)?,
                    )?;
                }
            }
        }

        // 8. Historical Valuation Multiples over Time
        let price = price_on.call1((prices_df, &period_dt))?;
        r.set_item("Price", &price)?;
        for k in [
            "P/E Ratio",
            "P/S Ratio",
            "P/B Ratio",
            "P/FCF Ratio",
            "EV/EBITDA",
            "EV/Sales",
            "Dividend Yield (%)",
        ] {
            r.set_item(k, &nan)?;
        }

        if !price.is_none() && positive(&shares)? {
            let mkt_cap = price.mul(&shares)?;
            if positive(&eps)? {
                r.set_item("P/E Ratio", price.div(&eps)?)?;
            } else if positive(&net_income)? {
                r.set_item("P/E Ratio", mkt_cap.div(&net_income)?)?;
            }
            if positive(&revenue)? {
                r.set_item("P/S Ratio", mkt_cap.div(&revenue)?)?;
            }
            if positive(&total_equity)? {
                r.set_item("P/B Ratio", mkt_cap.div(&total_equity)?)?;
            }
            let fcf = if !ocf.is_none() && !capex.is_none() {
                ocf.add(&capex)?
            } else {
                none(py)
            };
            if positive(&fcf)? {
                r.set_item("P/FCF Ratio", mkt_cap.div(&fcf)?)?;
            }
            let ev = mkt_cap
                .add(or_f(&total_debt, 0.0)?)?
                .sub(or_f(&cash, 0.0)?)?;
            if positive(&revenue)? && ev.gt(0)? {
                r.set_item("EV/Sales", ev.div(&revenue)?)?;
            }
            let mut ebitda = or_else(g.get(fin, "EBITDA", pf)?, || {
                or_else(g.get(fin, "Ebitda", pf)?, || {
                    g.get(fin, "Normalized EBITDA", pf)
                })
            })?;
            if ebitda.is_none() {
                let base_ebit = if !ebit.is_none() {
                    ebit.clone()
                } else {
                    operating_income.clone()
                };
                if !base_ebit.is_none() {
                    let cf_ok = !cf.is_none() && !period_str_cf.is_none();
                    let from_cf = |label: &str| -> PyResult<Bound<'py, PyAny>> {
                        if cf_ok {
                            g.get(cf, label, &period_str_cf)
                        } else {
                            Ok(none(py))
                        }
                    };
                    let depr = or_else(g.get(fin, "Reconciled Depreciation", pf)?, || {
                        or_else(g.get(fin, "Depreciation And Amortization", pf)?, || {
                            or_else(from_cf("Depreciation And Amortization")?, || {
                                or_else(from_cf("Depreciation Amortization Depletion")?, || {
                                    from_cf("Depreciation")
                                })
                            })
                        })
                    })?;
                    ebitda = if !depr.is_none() {
                        base_ebit.add(abs.call1((&depr,))?)?
                    } else {
                        base_ebit
                    };
                }
            }
            if positive(&ebitda)? && ev.gt(0)? {
                r.set_item("EV/EBITDA", ev.div(&ebitda)?)?;
            }
            if !div_paid.is_none() && mkt_cap.gt(0)? {
                r.set_item(
                    "Dividend Yield (%)",
                    abs.call1((&div_paid,))?.div(&mkt_cap)?.mul(&hundred_f)?,
                )?;
            }
        }
        rows.append(r)?;
    }

    if !rows.is_truthy()? {
        return pd.getattr("DataFrame")?.call0();
    }
    let frame = pd
        .getattr("DataFrame")?
        .call1((rows,))?
        .call_method1("set_index", ("Period",))?;
    frame.call_method(
        "sort_index",
        (),
        Some(&kwargs(py, &[("ascending", &pybool(py, false))])?),
    )
}

#[pyfunction]
pub fn fr_numeric_statement(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<bool> {
    if none_or_empty(df)? {
        return Ok(true);
    }
    if !(df.getattr("index")?.getattr("is_unique")?.is_truthy()?
        && df.getattr("columns")?.getattr("is_unique")?.is_truthy()?)
    {
        return Ok(false);
    }
    let np = py.import("numpy")?;
    let dtype_t = np.getattr("dtype")?;
    let number_t = np.getattr("number")?;
    let na = py.import("pandas")?.getattr("NA")?;
    let numeric_kinds = pystr(py, "fiub");
    let columns = df.getattr("columns")?;
    for (dtype, column) in df.getattr("dtypes")?.try_iter()?.zip(columns.try_iter()?) {
        let (dtype, column) = (dtype?, column?);
        let is_dtype = dtype.is_instance(&dtype_t)?;
        if is_dtype && numeric_kinds.contains(dtype.getattr("kind")?)? {
            continue;
        }
        if !(is_dtype && dtype.getattr("kind")?.eq("O")?) {
            return Ok(false);
        }
        for v in df.get_item(&column)?.call_method0("tolist")?.try_iter()? {
            let v = v?;
            if v.is_none()
                || v.is_instance_of::<PyBool>()
                || v.is_instance_of::<PyInt>()
                || v.is_instance_of::<PyFloat>()
                || v.is_instance(&number_t)?
                || v.is(&na)
            {
                continue;
            }
            return Ok(false);
        }
    }
    Ok(true)
}

/// `_StatementLookup`: `_get_statement_value` for one numeric frame, resolved
/// a row at a time — the label's value, else the first of its aliases that
/// has one.
#[pyclass(module = "investa_core")]
pub struct StatementLookup {
    rows: Py<PyDict>,
    columns: Py<PyDict>,
    matrix: Vec<Vec<f64>>,
    width: usize,
    resolved: HashMap<String, Vec<f64>>,
}

impl StatementLookup {
    fn resolve(&mut self, py: Python<'_>, item: &str) -> PyResult<&Vec<f64>> {
        if !self.resolved.contains_key(item) {
            let mut out = vec![f64::NAN; self.width];
            let mut chain = vec![item.to_owned()];
            let aliases = fr(py)?
                .getattr("STATEMENT_ALIASES")?
                .call_method1("get", (item, PyList::empty(py)))?;
            for a in aliases.try_iter()? {
                let a: String = a?.extract()?;
                if a != item {
                    chain.push(a);
                }
            }
            let rows = self.rows.bind(py);
            for label in chain {
                if let Some(i) = rows.get_item(&label)? {
                    let values = &self.matrix[i.extract::<usize>()?];
                    for (o, v) in out.iter_mut().zip(values) {
                        if o.is_nan() && !v.is_nan() {
                            *o = *v;
                        }
                    }
                }
            }
            self.resolved.insert(item.to_owned(), out);
        }
        Ok(&self.resolved[item])
    }
}

#[pymethods]
impl StatementLookup {
    #[new]
    fn new(py: Python<'_>, df: &Bound<'_, PyAny>) -> PyResult<Self> {
        let rows = PyDict::new(py);
        for (i, label) in df.getattr("index")?.try_iter()?.enumerate() {
            rows.set_item(label?, i)?;
        }
        let columns = PyDict::new(py);
        for (j, label) in df.getattr("columns")?.try_iter()?.enumerate() {
            columns.set_item(label?, j)?;
        }
        let object = builtin(py, "object")?;
        let values = df.call_method("to_numpy", (), Some(&kwargs(py, &[("dtype", &object)])?))?;
        let isna = py.import("pandas")?.getattr("isna")?;
        let float = builtin(py, "float")?;
        let mut matrix = Vec::new();
        for row in values.call_method0("tolist")?.try_iter()? {
            let mut out = Vec::new();
            for v in row?.try_iter()? {
                let v = v?;
                out.push(if v.is_none() {
                    f64::NAN
                } else if let Ok(f) = v.cast::<PyFloat>() {
                    f.value()
                } else if isna.call1((&v,))?.is_truthy()? {
                    f64::NAN
                } else {
                    float.call1((&v,))?.extract::<f64>()?
                });
            }
            matrix.push(out);
        }
        let width = columns.len();
        Ok(StatementLookup {
            rows: rows.unbind(),
            columns: columns.unbind(),
            matrix,
            width,
            resolved: HashMap::new(),
        })
    }

    fn row<'py>(&mut self, py: Python<'py>, item: &str) -> PyResult<Bound<'py, PyAny>> {
        let values = self.resolve(py, item)?.clone();
        py.import("numpy")?
            .getattr("array")?
            .call1((PyList::new(py, values)?,))
    }

    /// The value, NaN for None.
    fn get<'py>(
        &mut self,
        py: Python<'py>,
        item: &str,
        column: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let j: usize = self.columns.bind(py).as_any().get_item(column)?.extract()?;
        let v = self.resolve(py, item)?[j];
        py.import("numpy")?.getattr("float64")?.call1((v,))
    }
}

#[pyfunction]
pub fn fr_period_columns<'py>(
    py: Python<'py>,
    df: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let to_datetime = py.import("pandas")?.getattr("to_datetime")?;
    let out = PyList::empty(py);
    for col in df.getattr("columns")?.try_iter()? {
        let col = col?;
        out.append(PyTuple::new(
            py,
            [to_datetime.call((&col,), Some(&coerce(py)?))?, col],
        )?)?;
    }
    Ok(out)
}

#[pyfunction]
pub fn fr_first_column<'py>(
    py: Python<'py>,
    columns: &Bound<'py, PyAny>,
    period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    for item in columns.try_iter()? {
        let pc = unpack(&item?, 2)?;
        if pc[0].eq(period)? {
            return Ok(pc[1].clone());
        }
    }
    Ok(none(py))
}

#[pyfunction]
#[pyo3(signature = (financials_df, balance_sheet_df, cashflow_df=None, periods_per_year=None, prices_df=None))]
pub fn fr_key_ratios_timeseries<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: Option<&Bound<'py, PyAny>>,
    periods_per_year: Option<&Bound<'py, PyAny>>,
    prices_df: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let nothing = none(py);
    let one = int(py, 1)?;
    let cashflow_df = cashflow_df.unwrap_or(&nothing);
    let periods_per_year = periods_per_year.unwrap_or(&one);
    let prices_df = prices_df.unwrap_or(&nothing);
    let m = fr(py)?;
    let numeric = m.getattr("_numeric_statement")?;
    for df in [financials_df, balance_sheet_df, cashflow_df] {
        if !numeric.call1((df,))?.is_truthy()? {
            return m.getattr("_key_ratios_python")?.call1((
                financials_df,
                balance_sheet_df,
                cashflow_df,
                periods_per_year,
                prices_df,
            ));
        }
    }
    let common = match common_periods(py, financials_df, balance_sheet_df)? {
        Ok(c) => c,
        Err(empty) => return Ok(empty),
    };

    let has_cashflow = !cashflow_df.is_none() && !cashflow_df.getattr("empty")?.is_truthy()?;
    let lookup = m.getattr("_StatementLookup")?;
    let frames = PyDict::new(py);
    frames.set_item("fin", lookup.call1((financials_df,))?)?;
    frames.set_item("bs", lookup.call1((balance_sheet_df,))?)?;
    frames.set_item(
        "cf",
        if has_cashflow {
            lookup.call1((cashflow_df,))?
        } else {
            none(py)
        },
    )?;
    let period_columns = m.getattr("_period_columns")?;
    let first_column = m.getattr("_first_column")?;
    let fin_columns = period_columns.call1((financials_df,))?;
    let bs_columns = period_columns.call1((balance_sheet_df,))?;
    let cf_columns = if has_cashflow {
        period_columns.call1((cashflow_df,))?
    } else {
        PyList::empty(py).into_any()
    };

    let nan = py.import("numpy")?.getattr("nan")?;
    let lookups = m.getattr("_RATIO_LOOKUPS")?;
    let price_on = m.getattr("_price_on_or_before")?;
    let float = builtin(py, "float")?;
    let periods = PyList::empty(py);
    let inputs = PyList::empty(py);
    let flags = PyList::empty(py);
    for (i, period_dt) in common.try_iter()?.enumerate() {
        let period_dt = period_dt?;
        let i_obj = int(py, i as i64)?;
        let period_str_fin = first_column.call1((&fin_columns, &period_dt))?;
        let period_str_bs = first_column.call1((&bs_columns, &period_dt))?;
        if !period_str_fin.is_truthy()? || !period_str_bs.is_truthy()? {
            skip_period_log(py, &period_dt)?;
            continue;
        }
        let period_str_cf = if has_cashflow {
            first_column.call1((&cf_columns, &period_dt))?
        } else {
            none(py)
        };
        let (mut prev_bs, mut prev_fin) = (none(py), none(py));
        if i_obj.ge(periods_per_year)? {
            let prev_dt = common.get_item(i_obj.sub(periods_per_year)?)?;
            prev_bs = first_column.call1((&bs_columns, &prev_dt))?;
            prev_fin = first_column.call1((&fin_columns, &prev_dt))?;
        }
        let columns = PyDict::new(py);
        columns.set_item("fin", &period_str_fin)?;
        columns.set_item("bs", &period_str_bs)?;
        columns.set_item("cf", &period_str_cf)?;
        columns.set_item("prev_bs", or_else(prev_bs, || Ok(none(py)))?)?;
        columns.set_item("prev_fin", or_else(prev_fin, || Ok(none(py)))?)?;
        let row = PyList::empty(py);
        for entry in lookups.try_iter()? {
            let si = unpack(&entry?, 2)?;
            let column = columns.as_any().get_item(&si[0])?;
            let frame = frames
                .as_any()
                .get_item(si[0].call_method1("replace", ("prev_", ""))?)?;
            row.append(if frame.is_none() || column.is_none() {
                nan.clone()
            } else {
                frame.call_method1("get", (&si[1], column))?
            })?;
        }
        let price = price_on.call1((prices_df, &period_dt))?;
        row.append(if price.is_none() {
            nan.clone()
        } else {
            float.call1((price,))?
        })?;
        periods.append(&period_dt)?;
        inputs.append(row)?;
        flags.append(PyTuple::new(
            py,
            [
                pybool(py, !columns.as_any().get_item("prev_bs")?.is_none()),
                pybool(py, !columns.as_any().get_item("prev_fin")?.is_none()),
                pybool(py, !period_str_cf.is_none()),
            ],
        )?)?;
    }

    let pd = py.import("pandas")?;
    if !periods.is_truthy()? {
        return pd.getattr("DataFrame")?.call0();
    }

    let fields = m.getattr("_RATIO_FIELDS")?;
    let optional = m.getattr("_RATIO_OPTIONAL")?;
    let computed = m
        .getattr("investa_core")?
        .call_method1("key_ratios", (inputs, flags))?;
    let rows = PyList::empty(py);
    let zip = builtin(py, "zip")?;
    for pair in zip.call1((periods, computed))?.try_iter()? {
        let pv = unpack(&pair?, 2)?;
        let r = PyDict::new(py);
        r.set_item("Period", &pv[0])?;
        for fv in zip.call1((&fields, &pv[1]))?.try_iter()? {
            let fv = unpack(&fv?, 2)?;
            let keep = !fv[1].is_none() || optional.contains(&fv[0])?;
            r.set_item(&fv[0], if keep { fv[1].clone() } else { nan.clone() })?;
        }
        rows.append(r)?;
    }
    let frame = pd
        .getattr("DataFrame")?
        .call1((rows,))?
        .call_method1("set_index", ("Period",))?;
    frame.call_method(
        "sort_index",
        (),
        Some(&kwargs(py, &[("ascending", &pybool(py, false))])?),
    )
}

#[pyfunction]
#[pyo3(signature = (financials_df, balance_sheet_df, cashflow_df=None, prices_df=None))]
pub fn fr_historical_mean_multiples<'py>(
    py: Python<'py>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    cashflow_df: Option<&Bound<'py, PyAny>>,
    prices_df: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new(py);
    let nothing = none(py);
    let prices_df = prices_df.unwrap_or(&nothing);
    if prices_df.is_none() {
        return Ok(out);
    }
    let m = fr(py)?;
    let ratios = match m.getattr("calculate_key_ratios_timeseries")?.call(
        (
            financials_df,
            balance_sheet_df,
            cashflow_df.unwrap_or(&nothing),
        ),
        Some(&kwargs(py, &[("prices_df", prices_df)])?),
    ) {
        Ok(r) => r,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!(
                    "historical_mean_multiples: ratio timeseries failed: {}",
                    es(py, &e)
                ),
            )?;
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    if none_or_empty(&ratios)? {
        return Ok(out);
    }

    let pd = py.import("pandas")?;
    let core = m.getattr("investa_core")?;
    let bands = m.getattr("HISTORICAL_MULTIPLE_BANDS")?;
    let min_obs = m.getattr("HISTORICAL_MULTIPLE_MIN_OBSERVATIONS")?;
    let len = builtin(py, "len")?;
    let int_t = builtin(py, "int")?;
    let o = Ops::new(py)?;
    for item in m
        .getattr("_HISTORICAL_MULTIPLE_COLUMNS")?
        .call_method0("items")?
        .try_iter()?
    {
        let kc = unpack(&item?, 2)?;
        let (key, column) = (&kc[0], &kc[1]);
        if !ratios.getattr("columns")?.contains(column)? {
            continue;
        }
        let series = pd
            .getattr("to_numeric")?
            .call((ratios.get_item(column)?,), Some(&coerce(py)?))?
            .call_method0("dropna")?;
        let lh = unpack(&bands.get_item(key)?, 2)?;
        let mask = series
            .rich_compare(&lh[0], CompareOp::Ge)?
            .bitand(series.rich_compare(&lh[1], CompareOp::Le)?)?;
        let usable = series.get_item(mask)?;
        if len.call1((&usable,))?.lt(&min_obs)? {
            continue;
        }

        let periods = PyList::empty(py);
        for p in usable.getattr("index")?.try_iter()? {
            periods.append(p?.str()?.get_item(slice(py, &none(py), &int(py, 4)?)?)?)?;
        }
        let span = if periods.is_truthy()? {
            pystr(
                py,
                &format!(
                    "{}-{}",
                    o.fmt(&builtin(py, "min")?.call1((&periods,))?)?,
                    o.fmt(&builtin(py, "max")?.call1((&periods,))?)?
                ),
            )
        } else {
            pystr(py, "")
        };
        let values = || usable.getattr("values")?.call_method0("tolist");
        let entry = PyDict::new(py);
        entry.set_item("value", core.call_method1("np_median", (values()?,))?)?;
        entry.set_item("n", int_t.call1((len.call1((&usable,))?,))?)?;
        entry.set_item("span", span)?;
        entry.set_item(
            "low",
            core.call_method1("np_percentile", (values()?, 25.0))?,
        )?;
        entry.set_item(
            "high",
            core.call_method1("np_percentile", (values()?, 75.0))?,
        )?;
        entry.set_item(
            "dropped",
            int_t.call1((len.call1((&series,))?.sub(len.call1((&usable,))?)?,))?,
        )?;
        out.set_item(key, entry)?;
    }
    Ok(out)
}

#[pyfunction]
pub fn fr_load_price_history_for_multiples<'py>(
    py: Python<'py>,
    mdp: &Bound<'py, PyAny>,
    yf_symbol: &Bound<'py, PyAny>,
    years: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let body = || -> PyResult<Option<Bound<'py, PyAny>>> {
        let timedelta = py.import("datetime")?.getattr("timedelta")?;
        let get_est_today = py.import("utils_time")?.getattr("get_est_today")?;
        let end = get_est_today.call0()?;
        let days = int(py, 365)?.mul(years)?.add(30)?;
        let start = end.sub(timedelta.call((), Some(&kwargs(py, &[("days", &days)])?))?)?;
        let got = mdp.call_method1(
            "get_historical_data",
            (PyList::new(py, [yf_symbol])?, start, &end),
        )?;
        let hist = unpack(&got, 2)?.swap_remove(0);
        if hist.is_truthy()? && hist.contains(yf_symbol)? {
            let frame = hist.get_item(yf_symbol)?;
            if !frame.is_none() && builtin(py, "len")?.call1((&frame,))?.is_truthy()? {
                return Ok(Some(frame));
            }
        }
        Ok(None)
    };
    match body() {
        Ok(Some(frame)) => return Ok(frame),
        Ok(None) => {}
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let o = Ops::new(py)?;
            log(
                py,
                "debug",
                format!(
                    "Price history unavailable for {}: {}",
                    o.fmt(yf_symbol)?,
                    es(py, &e)
                ),
            )?;
        }
        Err(e) => return Err(e),
    }
    Ok(none(py))
}

#[pyfunction]
#[pyo3(signature = (ticker_info, financials_df_latest_annual=None, balance_sheet_df_latest_annual=None))]
pub fn fr_current_valuation_ratios<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df_latest_annual: Option<&Bound<'py, PyAny>>,
    balance_sheet_df_latest_annual: Option<&Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let _ = financials_df_latest_annual;
    let nan = py.import("numpy")?.getattr("nan")?;
    let ratios = PyDict::new(py);
    for k in [
        "P/E Ratio (TTM)",
        "Forward P/E Ratio",
        "Price-to-Sales (P/S) Ratio (TTM)",
        "Price-to-Book (P/B) Ratio (MRQ)",
        "Dividend Yield (%)",
        "Enterprise Value to EBITDA",
    ] {
        ratios.set_item(k, &nan)?;
    }
    if !ticker_info.is_truthy()? {
        return Ok(ratios);
    }
    let ti = ticker_info;
    let get = |k: &str| ti.call_method1("get", (k,));

    let current_price = or_else(get("currentPrice")?, || get("regularMarketPrice"))?;
    let market_cap = get("marketCap")?;

    ratios.set_item("P/E Ratio (TTM)", get("trailingPE")?)?;
    ratios.set_item("Forward P/E Ratio", get("forwardPE")?)?;
    if !get("dividendYield")?.is_none() {
        ratios.set_item("Dividend Yield (%)", ti.get_item("dividendYield")?)?;
    }

    let trailing_revenue = get("totalRevenue")?;
    if market_cap.is_truthy()? && trailing_revenue.is_truthy()? && trailing_revenue.ne(0)? {
        ratios.set_item(
            "Price-to-Sales (P/S) Ratio (TTM)",
            market_cap.div(&trailing_revenue)?,
        )?;
    }

    let book_value_per_share = get("bookValue")?;
    let nothing = none(py);
    let bs = balance_sheet_df_latest_annual.unwrap_or(&nothing);
    if current_price.is_truthy()?
        && book_value_per_share.is_truthy()?
        && book_value_per_share.ne(0)?
    {
        ratios.set_item(
            "Price-to-Book (P/B) Ratio (MRQ)",
            current_price.div(&book_value_per_share)?,
        )?;
    } else if market_cap.is_truthy()? && !bs.is_none() && !bs.getattr("empty")?.is_truthy()? {
        // Fallback to marketCap / latest total equity
        let latest = bs.getattr("columns")?.get_item(0)?;
        let total_equity_latest =
            Gsv::new(&fr(py)?)?.get(bs, "Total Stockholder Equity", &latest)?;
        if total_equity_latest.is_truthy()? && total_equity_latest.ne(0)? {
            ratios.set_item(
                "Price-to-Book (P/B) Ratio (MRQ)",
                market_cap.div(&total_equity_latest)?,
            )?;
        }
    }

    ratios.set_item("Enterprise Value to EBITDA", get("enterpriseToEbitda")?)?;
    Ok(ratios)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn fr_calculate_wacc<'py>(
    py: Python<'py>,
    ticker_info: &Bound<'py, PyAny>,
    financials_df: &Bound<'py, PyAny>,
    balance_sheet_df: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
    market_return: &Bound<'py, PyAny>,
    default_tax_rate: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = fr(py)?;
    let body = || -> PyResult<Bound<'py, PyDict>> {
        let o = Ops::new(py)?;
        let ti = ticker_info;
        let get = |k: &str| ti.call_method1("get", (k,));
        let mut raw_beta = get("beta")?;
        if raw_beta.is_none()
            || py
                .import("pandas")?
                .call_method1("isna", (&raw_beta,))?
                .is_truthy()?
        {
            log(
                py,
                "debug",
                format!("Beta missing for {}, using 1.0", o.fmt(&get("symbol")?)?),
            )?;
            raw_beta = none(py);
        }

        let g = Gsv::new(&m)?;
        let mut total_debt = get("totalDebt")?;
        if total_debt.is_none()
            && !balance_sheet_df.is_none()
            && !balance_sheet_df.getattr("empty")?.is_truthy()?
        {
            total_debt = g.get(
                balance_sheet_df,
                "Total Debt",
                &balance_sheet_df.getattr("columns")?.get_item(0)?,
            )?;
        }

        let (mut interest_expense, mut income_tax_expense, mut pretax_income) =
            (none(py), none(py), none(py));
        let mut operating_income = none(py);
        let have_statements =
            !financials_df.is_none() && !financials_df.getattr("empty")?.is_truthy()?;
        if have_statements {
            let latest = financials_df.getattr("columns")?.get_item(0)?;
            interest_expense = g.get(financials_df, "Interest Expense", &latest)?;
            income_tax_expense = g.get(financials_df, "Tax Provision", &latest)?;
            pretax_income = g.get(financials_df, "Pretax Income", &latest)?;
            operating_income = g.get(financials_df, "Operating Income", &latest)?;
        }

        let float = builtin(py, "float")?;
        let bands = PyList::empty(py);
        for band in m.getattr("SIZE_PREMIUM_BANDS")?.try_iter()? {
            let tp = unpack(&band?, 2)?;
            bands.append(PyTuple::new(
                py,
                [float.call1((&tp[0],))?, float.call1((&tp[1],))?],
            )?)?;
        }
        let args = PyTuple::new(
            py,
            [
                opt_number(&raw_beta)?,
                float.call1((risk_free_rate,))?,
                float.call1((market_return,))?,
                float.call1((default_tax_rate,))?,
                opt_number(&get("marketCap")?)?,
                opt_number(&total_debt)?,
                pybool(py, have_statements),
                opt_number(&interest_expense)?,
                opt_number(&income_tax_expense)?,
                opt_number(&pretax_income)?,
                opt_number(&operating_income)?,
                m.getattr("MAX_COST_OF_DEBT")?,
                m.getattr("MIN_DISCOUNT_RATE")?,
                m.getattr("MAX_DISCOUNT_RATE")?,
                bands.into_any(),
            ],
        )?;
        let r = unpack(&m.getattr("investa_core")?.getattr("wacc")?.call1(args)?, 7)?;
        let out = PyDict::new(py);
        if r[6].is_none() {
            out.set_item("wacc", &r[0])?;
            out.set_item("cost_of_equity", &r[1])?;
            out.set_item("beta", &r[4])?;
            out.set_item("size_premium", &r[5])?;
            out.set_item("method", "Cost of Equity (No Market Cap)")?;
            return Ok(out);
        }
        let w = unpack(&r[6], 2)?;
        out.set_item("wacc", &r[0])?;
        out.set_item("cost_of_equity", &r[1])?;
        out.set_item("cost_of_debt", &r[2])?;
        out.set_item("tax_rate", &r[3])?;
        out.set_item("weight_equity", &w[0])?;
        out.set_item("weight_debt", &w[1])?;
        out.set_item("beta", &r[4])?;
        out.set_item("size_premium", &r[5])?;
        out.set_item("method", "WACC")?;
        Ok(out)
    };
    match body() {
        Ok(d) => Ok(d),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!("Error calculating WACC: {}", es(py, &e)),
            )?;
            let out = PyDict::new(py);
            out.set_item("wacc", 0.10)?;
            out.set_item("method", "Default (10%) due to error")?;
            Ok(out)
        }
        Err(e) => Err(e),
    }
}
