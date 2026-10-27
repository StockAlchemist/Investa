//! The glue of `buffett_metrics` and `buffett_value`: the conversions into
//! and out of the metric kernels, the ratio bridge, the per-company entry
//! point under its EDGAR session, the price-dependent value metrics, the
//! value score's per-model grouping, and the value frame.
//!
//! Ported line for line from `src/buffett_metrics.py` and
//! `src/buffett_value.py` (frozen in `tests/reference/*_p15.py`). The metric
//! and percentile kernels were already Rust. The per-value conversions
//! (`_raw_number`, `_finite`) run natively here: `_for_rust` meets about a
//! thousand values per company across the whole universe, and they are pure
//! functions whose semantics are reproduced exactly (Python's `float()`, its
//! `isfinite`, its `__float__` probe). Everything else is Python's own
//! operations, with siblings looked up on the modules at call time.

use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

fn bm(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("buffett_metrics")
}

fn bv(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("buffett_value")
}

/// `_finite`: a finite float, or None.
pub(crate) fn finite(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    if value.is_none() {
        return Ok(None);
    }
    // float(value), with its TypeError/ValueError meaning "not a number".
    let number = if let Ok(f) = value.cast::<PyFloat>() {
        f.value()
    } else {
        match py.import("builtins")?.getattr("float")?.call1((value,)) {
            Ok(f) => f.extract::<f64>()?,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                return Ok(None);
            }
            Err(e) => return Err(e),
        }
    };
    Ok(if number.is_finite() {
        Some(number)
    } else {
        None
    })
}

/// `_safe_divide`: num / den for two finite numbers and a non-zero den.
pub(crate) fn safe_divide(
    py: Python<'_>,
    num: &Bound<'_, PyAny>,
    den: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    let n = finite(py, num)?;
    let d = finite(py, den)?;
    Ok(match (n, d) {
        (Some(n), Some(d)) if d != 0.0 => Some(n / d),
        _ => None,
    })
}

#[pyfunction]
pub fn bm_finite(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    finite(py, value)
}

#[pyfunction]
pub fn bm_safe_divide(
    py: Python<'_>,
    numerator: &Bound<'_, PyAny>,
    denominator: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    safe_divide(py, numerator, denominator)
}

/// `_raw_number`: None stays None, numbers become floats (NaN and infinity
/// kept), anything else raises as the arithmetic on it would have.
fn raw_number<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if value.is_none() {
        return Ok(value.clone());
    }
    if let Ok(f) = value.cast::<PyFloat>() {
        return Ok(PyFloat::new(py, f.value()).into_any());
    }
    let b = py.import("builtins")?;
    let numeric = value.is_instance(&b.getattr("bool")?)?
        || value.is_instance(&b.getattr("int")?)?
        || value.is_instance(&b.getattr("float")?)?
        || value.hasattr("__float__")?;
    if numeric {
        return b.getattr("float")?.call1((value,));
    }
    Err(PyTypeError::new_err(format!(
        "concept value is not a number: {}",
        value.repr()?
    )))
}

#[pyfunction]
pub fn bm_raw_number<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    raw_number(py, value)
}

/// `(x or {}).items()`
fn items_or_empty<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if x.is_truthy()? {
        x.call_method0("items")
    } else {
        PyDict::new(py).call_method0("items")
    }
}

/// `_for_rust`: every concept series keyed by `str(period)`, values as floats.
#[pyfunction]
pub fn bm_for_rust<'py>(
    py: Python<'py>,
    concepts: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = Ops::new(py)?;
    let out = PyDict::new(py);
    for item in items_or_empty(py, concepts)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let series = PyDict::new(py);
        for p in items_or_empty(py, &kv[1])?.try_iter()? {
            let pv = unpack(&p?, 2)?;
            series.set_item(o.str_t.call1((&pv[0],))?, raw_number(py, &pv[1])?)?;
        }
        out.set_item(&kv[0], series)?;
    }
    Ok(out)
}

/// `_ratios_for_rust`: the ratio labels the metrics read, through `_finite`.
#[pyfunction]
pub fn bm_ratios_for_rust<'py>(
    py: Python<'py>,
    ratios_by_period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = Ops::new(py)?;
    let labels = bm(py)?.getattr("_RATIO_LABELS")?;
    let out = PyDict::new(py);
    for item in items_or_empty(py, ratios_by_period)?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let row = &kv[1];
        let d = PyDict::new(py);
        for label in labels.try_iter()? {
            let label = label?;
            if row.contains(&label)? {
                d.set_item(&label, finite(py, &row.call_method1("get", (&label,))?)?)?;
            }
        }
        out.set_item(o.str_t.call1((&kv[0],))?, d)?;
    }
    Ok(out)
}

/// `_from_rust`: the kernel's (key, value, is_int) rows as the metrics dict.
#[pyfunction]
pub fn bm_from_rust<'py>(
    py: Python<'py>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let int = py.import("builtins")?.getattr("int")?;
    let out = PyDict::new(py);
    for r in rows.try_iter()? {
        let r = unpack(&r?, 3)?;
        let v = if r[2].is_truthy()? {
            int.call1((&r[1],))?
        } else {
            r[1].clone()
        };
        out.set_item(&r[0], v)?;
    }
    Ok(out)
}

/// One model's metrics through its kernel, as `_compute_*_metrics` do.
fn model_metrics<'py>(
    py: Python<'py>,
    kernel: &str,
    concepts: &Bound<'py, PyAny>,
    sector: Option<&Bound<'py, PyAny>>,
    ratios: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bm(py)?;
    let core = m.getattr("investa_core")?;
    let for_rust = m.getattr("_for_rust")?;
    let rows = match sector {
        None => core.getattr(kernel)?.call1((
            for_rust.call1((concepts,))?,
            m.getattr("_ratios_for_rust")?.call1((ratios,))?,
        ))?,
        Some(sector) => core.getattr(kernel)?.call1((
            for_rust.call1((concepts,))?,
            for_rust.call1((sector,))?,
            m.getattr("_ratios_for_rust")?.call1((ratios,))?,
        ))?,
    };
    m.getattr("_from_rust")?.call1((rows,))
}

#[pyfunction]
pub fn bm_generic_metrics<'py>(
    py: Python<'py>,
    concepts: &Bound<'py, PyAny>,
    ratios_by_period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model_metrics(
        py,
        "buffett_generic_metrics",
        concepts,
        None,
        ratios_by_period,
    )
}

#[pyfunction]
pub fn bm_bank_metrics<'py>(
    py: Python<'py>,
    concepts: &Bound<'py, PyAny>,
    sector: &Bound<'py, PyAny>,
    ratios_by_period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model_metrics(
        py,
        "buffett_bank_metrics",
        concepts,
        Some(sector),
        ratios_by_period,
    )
}

#[pyfunction]
pub fn bm_reit_metrics<'py>(
    py: Python<'py>,
    concepts: &Bound<'py, PyAny>,
    sector: &Bound<'py, PyAny>,
    ratios_by_period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    model_metrics(
        py,
        "buffett_reit_metrics",
        concepts,
        Some(sector),
        ratios_by_period,
    )
}

/// `_ratios_by_period`: the ratio engine's output re-keyed by period string,
/// so the ranking and the rest of the app share one ROE.
#[pyfunction]
pub fn bm_ratios_by_period<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = bm(py)?;
    let o = Ops::new(py)?;
    let ratios = py.import("financial_ratios")?;
    let statements = m
        .getattr("edgar_provider")?
        .getattr("get_statements")?
        .call1((cik,))?;
    let frame = ratios.getattr("calculate_key_ratios_timeseries")?.call1((
        statements.call_method1("get", ("financials",))?,
        statements.call_method1("get", ("balance_sheet",))?,
    ))?;
    let out = PyDict::new(py);
    if frame.is_none() || frame.getattr("empty")?.is_truthy()? {
        return Ok(out);
    }
    for item in frame.call_method0("iterrows")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let key = match kv[0].call_method1("strftime", ("%Y-%m-%d",)) {
            Ok(k) => k,
            Err(e) if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py) => {
                o.str_t.call1((&kv[0],))?
            }
            Err(e) => return Err(e),
        };
        let row = PyDict::new(py);
        for f in kv[1]
            .call_method0("to_dict")?
            .call_method0("items")?
            .try_iter()?
        {
            let f = unpack(&f?, 2)?;
            if f[0].ne("Period")? {
                row.set_item(&f[0], &f[1])?;
            }
        }
        out.set_item(key, row)?;
    }
    Ok(out)
}

/// `compute_metrics`: `_compute_metrics` inside one per-company EDGAR session.
#[pyfunction]
pub fn bm_compute_metrics<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bm(py)?;
    let session = m
        .getattr("edgar_provider")?
        .getattr("company_session")?
        .call1((cik,))?;
    with1(py, &session, |_| {
        m.getattr("_compute_metrics")?
            .call1((cik, symbol, name, model))
    })
    .map(|v| v.unwrap_or_else(|| py.None().into_bound(py)))
}

/// `_compute_metrics`: one company's full metric set; never raises.
#[pyfunction]
pub fn bm_compute_metrics_inner<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bm(py)?;
    let o = Ops::new(py)?;
    let edgar = m.getattr("edgar_provider")?;
    let fs = o.fmt(symbol)?;
    let result = m.getattr("CompanyMetrics")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("cik", cik),
                ("symbol", symbol),
                ("name", name),
                ("model", model),
            ],
        )?),
    )?;
    let gates = result.getattr("gate_failures")?;
    let concepts = match edgar.getattr("get_concept_values")?.call1((cik,)) {
        Ok(c) => c,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!("Metrics: concept load failed for {fs}: {}", es(py, &e)),
            )?;
            gates.call_method1("append", ("fundamentals_unavailable",))?;
            return Ok(result);
        }
        Err(e) => return Err(e),
    };
    if !concepts.is_truthy()? || !concepts.contains("net_income")? {
        gates.call_method1("append", ("no_fundamentals",))?;
        return Ok(result);
    }
    let b = py.import("builtins")?;
    let periods = b.getattr("sorted")?.call1((b
        .getattr("set")?
        .call1((concepts.call_method1("get", ("net_income", PyDict::new(py)))?,))?,))?;
    result.setattr("period_count", periods.len()?)?;
    let none = || py.None().into_bound(py);
    result.setattr(
        "latest_period",
        if periods.is_truthy()? {
            periods.get_item(-1)?
        } else {
            none()
        },
    )?;
    result.setattr(
        "first_period",
        if periods.is_truthy()? {
            periods.get_item(0)?
        } else {
            none()
        },
    )?;

    // Share counts step at splits; rebuilt once from same-filing ratios.
    if concepts
        .call_method1("get", ("shares_diluted",))?
        .is_truthy()?
    {
        match edgar
            .getattr("split_consistent_series")?
            .call1((cik, "shares_diluted"))
        {
            Ok(s) => concepts.set_item("shares_diluted", s)?,
            Err(e) if e.is_instance_of::<PyException>(py) => log(
                py,
                "warning",
                format!(
                    "Metrics: share-count reconstruction failed for {fs}: {}",
                    es(py, &e)
                ),
            )?,
            Err(e) => return Err(e),
        }
    }
    let ratios = match m.getattr("_ratios_by_period")?.call1((cik,)) {
        Ok(r) => r,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!("Metrics: ratio computation failed for {fs}: {}", es(py, &e)),
            )?;
            PyDict::new(py).into_any()
        }
        Err(e) => return Err(e),
    };
    let list = b.getattr("list")?;
    let metrics = if model.eq("bank")? || model.eq("insurer")? {
        let sector = edgar
            .getattr("get_concept_values")?
            .call1((cik, list.call1((m.getattr("BANK_CONCEPTS")?,))?))?;
        m.getattr("_compute_bank_metrics")?
            .call1((&concepts, sector, &ratios))?
    } else if model.eq("reit")? {
        let sector = edgar
            .getattr("get_concept_values")?
            .call1((cik, list.call1((m.getattr("REIT_CONCEPTS")?,))?))?;
        m.getattr("_compute_reit_metrics")?
            .call1((&concepts, sector, &ratios))?
    } else {
        m.getattr("_compute_generic_metrics")?
            .call1((&concepts, &ratios))?
    };
    result.setattr("metrics", &metrics)?;

    let all_keys = m.getattr("_COVERAGE_KEYS")?;
    let keys = all_keys.call_method1("get", (model, all_keys.get_item("generic")?))?;
    let mut resolved = 0usize;
    for k in keys.try_iter()? {
        if !metrics.call_method1("get", (k?,))?.is_none() {
            resolved += 1;
        }
    }
    let n = keys.len()?;
    result.setattr(
        "coverage",
        if n > 0 {
            resolved as f64 / n as f64
        } else {
            0.0
        },
    )?;
    Ok(result)
}

/// `_pct_yield`: a yield in percent, or None.
#[pyfunction]
pub fn bv_pct_yield(
    py: Python<'_>,
    numerator: &Bound<'_, PyAny>,
    market_cap: &Bound<'_, PyAny>,
) -> PyResult<Option<f64>> {
    Ok(safe_divide(py, numerator, market_cap)?.map(|v| v * 100.0))
}

/// `compute_market_metrics`: the price-dependent value metrics for one company.
#[pyfunction]
pub fn bv_compute_market_metrics<'py>(
    py: Python<'py>,
    company: &Bound<'py, PyAny>,
    price: &Bound<'py, PyAny>,
    shares_outstanding: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = bv(py)?;
    let o = Ops::new(py)?;
    let result = PyDict::new(py);
    result.set_item("price", price)?;
    for k in [
        "market_cap",
        "earnings_yield",
        "fcf_yield",
        "ev_to_ebit",
        "price_to_book",
        "price_to_sales",
    ] {
        result.set_item(k, py.None())?;
    }
    let (Some(price), Some(shares)) = (finite(py, price)?, finite(py, shares_outstanding)?) else {
        return Ok(result);
    };
    if price <= 0.0 || shares <= 0.0 {
        return Ok(result);
    }
    let market_cap = price * shares;
    result.set_item("market_cap", market_cap)?;

    let concepts = m
        .getattr("edgar_provider")?
        .getattr("get_concept_values")?
        .call1((
            company.getattr("cik")?,
            PyList::new(
                py,
                [
                    "revenue",
                    "equity",
                    "pretax_income",
                    "interest_expense",
                    "net_income",
                    "cash",
                ],
            )?,
        ))?;
    let latest = company.getattr("latest_period")?;
    if !latest.is_truthy()? {
        return Ok(result);
    }
    let at = |concept: &str| -> PyResult<Option<f64>> {
        finite(
            py,
            &concepts
                .call_method1("get", (concept, PyDict::new(py)))?
                .call_method1("get", (&latest,))?,
        )
    };
    let owner = company.call_method1("get", ("owner_earnings_latest",))?;
    let owner = if owner.is_truthy()? {
        owner
    } else {
        company.call_method1("get", ("ffo_latest",))?
    };
    let net_income = at("net_income")?;
    let equity = at("equity")?;
    let revenue = at("revenue")?;
    let cash = match at("cash")? {
        Some(c) if c != 0.0 => c,
        _ => 0.0,
    };
    let mut total_debt = company.call_method1("get", ("total_debt",))?;
    if total_debt.is_none() && company.call_method1("get", ("debt_free",))?.is_truthy()? {
        total_debt = o.fl(0.0);
    }
    let opt = |v: Option<f64>| -> Bound<'py, PyAny> {
        match v {
            Some(x) => o.fl(x),
            None => py.None().into_bound(py),
        }
    };
    let mc = o.fl(market_cap);
    let pct = |num: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        Ok(opt(safe_divide(py, num, &mc)?.map(|v| v * 100.0)))
    };
    result.set_item("earnings_yield", pct(&opt(net_income))?)?;
    result.set_item("fcf_yield", pct(&owner)?)?;
    result.set_item("price_to_book", opt(safe_divide(py, &mc, &opt(equity))?))?;
    result.set_item("price_to_sales", opt(safe_divide(py, &mc, &opt(revenue))?))?;

    if !total_debt.is_none() {
        let ev = mc.add(&total_debt)?.sub(cash)?;
        let pretax = at("pretax_income")?;
        let interest = match at("interest_expense")? {
            Some(i) if i != 0.0 => i,
            _ => 0.0,
        };
        if let Some(pretax) = pretax {
            let ebit = pretax + interest;
            if ebit > 0.0 {
                result.set_item("ev_to_ebit", opt(safe_divide(py, &ev, &o.fl(ebit))?))?;
            }
        }
    }
    Ok(result)
}

/// `score_value`: the two yields' sector-relative percentiles blended into a
/// 0–100 score within each model.
#[pyfunction]
pub fn bv_score_value<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bv(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let percentile_fn = py
        .import("buffett_rank")?
        .getattr("_winsorised_percentile")?;
    let series = pd.getattr("Series")?;
    if frame.getattr("empty")?.is_truthy()? {
        return series.call((), Some(&kwargs(py, &[("dtype", &o.float)])?));
    }
    let scores = series.call(
        (f64::NAN,),
        Some(&kwargs(
            py,
            &[("index", &frame.getattr("index")?), ("dtype", &o.float)],
        )?),
    )?;
    let weights_map = m.getattr("VALUE_WEIGHTS")?;
    let direction = m.getattr("HIGHER_IS_BETTER")?;
    let coerce = kwargs(py, &[("errors", &pystr(py, "coerce"))])?;
    let f64_kw = kwargs(py, &[("dtype", &np.getattr("float64")?)])?;
    for item in frame.call_method1("groupby", ("model",))?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (model, group) = (&kv[0], &kv[1]);
        if group.len()? < 5 {
            log(
                py,
                "warning",
                format!(
                    "Value: only {} '{}' companies — percentile scores for this group are unreliable",
                    group.len()?,
                    o.fmt(model)?
                ),
            )?;
        }
        let columns = PyList::empty(py);
        let weights = PyList::empty(py);
        for metric in weights_map.try_iter()? {
            let metric = metric?;
            let higher = direction.get_item(&metric)?;
            if !group.getattr("columns")?.contains(&metric)? {
                continue;
            }
            let mut values =
                pd.call_method("to_numeric", (group.get_item(&metric)?,), Some(&coerce))?;
            // A negative multiple is meaningless, not cheap.
            if !higher.is_truthy()? {
                values = values.call_method1(
                    "where",
                    (values.rich_compare(0, pyo3::basic::CompareOp::Gt)?,),
                )?;
            }
            let percentile = percentile_fn.call1((values, &higher))?;
            columns.append(np.call_method1(
                "ascontiguousarray",
                (percentile.call_method("to_numpy", (), Some(&f64_kw))?,),
            )?)?;
            weights.append(o.float(&weights_map.call_method1("get", (&metric, 0.0))?)?)?;
        }
        let loc = scores.getattr("loc")?;
        if columns.is_truthy()? {
            loc.set_item(
                group.getattr("index")?,
                m.getattr("investa_core")?
                    .getattr("weighted_renormalised")?
                    .call1((columns, weights))?,
            )?;
        } else {
            loc.set_item(group.getattr("index")?, f64::NAN)?;
        }
    }
    Ok(scores)
}

/// `build_value_frame`: the price-dependent metrics for a universe.
#[pyfunction]
pub fn bv_build_value_frame<'py>(
    py: Python<'py>,
    companies: &Bound<'py, PyAny>,
    market_data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bv(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let compute = m.getattr("compute_market_metrics")?;
    let rows = PyList::empty(py);
    for company in companies.try_iter()? {
        let company = company?;
        let symbol = company.getattr("symbol")?;
        let quote = market_data.call_method1("get", (&symbol,))?;
        let quote = if quote.is_truthy()? {
            quote
        } else {
            PyDict::new(py).into_any()
        };
        let metrics = match compute.call1((
            &company,
            quote.call_method1("get", ("price",))?,
            quote.call_method1("get", ("shares",))?,
        )) {
            Ok(v) => v,
            Err(e) if e.is_instance_of::<PyException>(py) => {
                log(
                    py,
                    "warning",
                    format!("Value: failed for {}: {}", o.fmt(&symbol)?, es(py, &e)),
                )?;
                PyDict::new(py).into_any()
            }
            Err(e) => return Err(e),
        };
        let row = PyDict::new(py);
        row.set_item("symbol", &symbol)?;
        row.set_item("model", company.getattr("model")?)?;
        row.call_method1("update", (metrics,))?;
        rows.append(row)?;
    }
    let mut frame = pd.getattr("DataFrame")?.call1((rows,))?;
    if !frame.getattr("empty")?.is_truthy()? {
        frame = frame.call_method(
            "set_index",
            ("symbol",),
            Some(&kwargs(
                py,
                &[(
                    "drop",
                    &pyo3::types::PyBool::new(py, false).to_owned().into_any(),
                )],
            )?),
        )?;
    }
    Ok(frame)
}
