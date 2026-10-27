//! The glue of `buffett_rank`: the hard gates, the confidence multiplier,
//! the universe frame, the per-model quality scoring, the quality/value
//! composite, and the re-blend and re-rank of a stored run.
//!
//! Ported line for line from `src/buffett_rank.py` (frozen in
//! `tests/reference/buffett_rank_p15.py`). The percentile and pillar
//! kernels were already Rust; this is pandas and dict glue, every operation
//! Python's own, with the pillar tables, weights and siblings looked up on
//! `buffett_rank` at call time.

use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};

use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{log, pystr, setitem, unpack};

fn br(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("buffett_rank")
}

fn fals(py: Python<'_>) -> Bound<'_, PyAny> {
    PyBool::new(py, false).to_owned().into_any()
}

/// `evaluate_gates`: the hard quality gates' failures; None never triggers one.
#[pyfunction]
pub fn br_evaluate_gates<'py>(
    py: Python<'py>,
    company: &Bound<'py, PyAny>,
    min_periods: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let o = Ops::new(py)?;
    let failures = PyList::empty(py);
    let metrics = company.getattr("metrics")?;
    for f in company.getattr("gate_failures")?.try_iter()? {
        let f = f?;
        if f.eq("no_fundamentals")? || f.eq("fundamentals_unavailable")? {
            return Ok(failures);
        }
    }
    let periods = company.getattr("period_count")?;
    if periods.lt(min_periods)? {
        failures.append(format!("insufficient_history({}y)", o.fmt(&periods)?))?;
    }
    let below = |key: &str, t: f64| -> PyResult<bool> {
        let v = metrics.call_method1("get", (key,))?;
        Ok(!v.is_none() && v.lt(t)?)
    };
    let above = |key: &str, t: f64| -> PyResult<bool> {
        let v = metrics.call_method1("get", (key,))?;
        Ok(!v.is_none() && v.gt(t)?)
    };
    let model = company.getattr("model")?;
    if model.eq("bank")? || model.eq("insurer")? {
        if below("equity_to_assets_latest", 4.0)? {
            failures.append("undercapitalised")?;
        }
        if above("loss_years", 2.0)? {
            failures.append("persistent_losses")?;
        }
        if below("roe_median", 0.0)? {
            failures.append("unprofitable")?;
        }
    } else if model.eq("reit")? {
        if above("debt_to_ffo", 15.0)? {
            failures.append("excessive_leverage")?;
        }
        if above("negative_ffo_years", 2.0)? {
            failures.append("persistent_negative_ffo")?;
        }
    } else {
        if !metrics.call_method1("get", ("debt_free",))?.is_truthy()?
            && below("interest_coverage", 2.0)?
        {
            failures.append("interest_not_covered")?;
        }
        if above("net_debt_to_owner_earnings", 6.0)? {
            failures.append("debt_not_serviceable")?;
        }
        if above("negative_owner_earnings_years", 2.0)? {
            failures.append("persistent_cash_burn")?;
        }
        if below("roe_median", 0.0)? {
            failures.append("unprofitable")?;
        }
    }
    Ok(failures)
}

/// `confidence_factor`: coverage mapped to a multiplier in [floor, 1.0].
#[pyfunction]
pub fn br_confidence_factor<'py>(
    py: Python<'py>,
    coverage: &Bound<'py, PyAny>,
    floor: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if coverage.is_none() {
        return Ok(floor.clone());
    }
    let b = py.import("builtins")?;
    let one = 1.0f64.into_pyobject(py)?.into_any();
    let scaled = floor.add(one.sub(floor)?.mul(coverage)?)?;
    let capped = b.getattr("min")?.call1((&one, scaled))?;
    b.getattr("float")?
        .call1((b.getattr("max")?.call1((floor, capped))?,))
}

/// `metrics_to_frame`: one row per company.
#[pyfunction]
pub fn br_metrics_to_frame<'py>(
    py: Python<'py>,
    companies: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let rows = PyList::empty(py);
    for company in companies.try_iter()? {
        let c = company?;
        let row = PyDict::new(py);
        for k in [
            "symbol",
            "cik",
            "name",
            "model",
            "period_count",
            "latest_period",
            "coverage",
        ] {
            row.set_item(k, c.getattr(k)?)?;
        }
        let gates = c.getattr("gate_failures")?;
        row.set_item(
            "gate_failures",
            pystr(py, ",").call_method1("join", (&gates,))?,
        )?;
        row.set_item("eligible", !gates.is_truthy()?)?;
        row.call_method1("update", (c.getattr("metrics")?,))?;
        rows.append(row)?;
    }
    let mut frame = pd.getattr("DataFrame")?.call1((rows,))?;
    if !frame.getattr("empty")?.is_truthy()? {
        frame = frame.call_method(
            "set_index",
            ("symbol",),
            Some(&kwargs(py, &[("drop", &fals(py))])?),
        )?;
    }
    Ok(frame)
}

/// `score_quality`: the universe ranked on quality alone, within each model.
#[pyfunction]
pub fn br_score_quality<'py>(
    py: Python<'py>,
    companies: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = br(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let frame = m.getattr("metrics_to_frame")?.call1((companies,))?;
    if frame.getattr("empty")?.is_truthy()? {
        return Ok(frame);
    }
    let eligible = frame
        .get_item(frame.get_item("eligible")?)?
        .call_method0("copy")?;
    if eligible.getattr("empty")?.is_truthy()? {
        log(py, "warning", "Rank: no companies passed the quality gates")?;
        setitem(&frame, "quality_score", f64::NAN)?;
        return Ok(frame);
    }
    let by_model = m.getattr("PILLARS_BY_MODEL")?;
    let generic = m.getattr("GENERIC_PILLARS")?;
    let parts = PyList::empty(py);
    for item in eligible.call_method1("groupby", ("model",))?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (model, group) = (&kv[0], &kv[1]);
        let spec = by_model.call_method1("get", (model, &generic))?;
        if group.len()? < 5 {
            log(
                py,
                "warning",
                format!(
                    "Rank: only {} eligible '{}' companies — percentile scores for this group are unreliable",
                    group.len()?,
                    o.fmt(model)?
                ),
            )?;
        }
        let pillars = m.getattr("_pillar_scores")?.call1((group, spec))?;
        setitem(
            &pillars,
            "quality_score",
            m.getattr("_quality_score")?.call1((&pillars,))?,
        )?;
        parts.append(pillars)?;
    }
    let pillar_frame = if parts.is_truthy()? {
        pd.call_method1("concat", (parts,))?
    } else {
        pd.getattr("DataFrame")?.call0()?
    };
    frame.call_method(
        "join",
        (pillar_frame,),
        Some(&kwargs(py, &[("how", &pystr(py, "left"))])?),
    )
}

/// `combine`: (q * quality + v * value) * confidence, ranked best first.
#[pyfunction]
pub fn br_combine<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
    value_scores: &Bound<'py, PyAny>,
    quality_weight: &Bound<'py, PyAny>,
    value_weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = br(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let mut result = frame.call_method0("copy")?;
    if !value_scores.is_none() {
        setitem(
            &result,
            "value_score",
            value_scores.call_method1("reindex", (result.getattr("index")?,))?,
        )?;
    } else if !result.getattr("columns")?.contains("value_score")? {
        setitem(&result, "value_score", f64::NAN)?;
    }
    let coerce = kwargs(py, &[("errors", &pystr(py, "coerce"))])?;
    let quality = pd.call_method(
        "to_numeric",
        (result.call_method1("get", ("quality_score",))?,),
        Some(&coerce),
    )?;
    let value = pd.call_method(
        "to_numeric",
        (result.call_method1("get", ("value_score",))?,),
        Some(&coerce),
    )?;
    let has_value = value.call_method0("notna")?;
    let blended = quality.call_method0("copy")?;
    let mixed = quality
        .get_item(&has_value)?
        .mul(quality_weight)?
        .add(value.get_item(&has_value)?.mul(value_weight)?)?;
    setitem(&blended, &has_value, mixed)?;
    setitem(
        &result,
        "confidence",
        result
            .get_item("coverage")?
            .call_method1("apply", (m.getattr("confidence_factor")?,))?,
    )?;
    setitem(
        &result,
        "composite_score",
        blended.mul(result.get_item("confidence")?)?,
    )?;
    result = result.call_method(
        "sort_values",
        ("composite_score",),
        Some(&kwargs(
            py,
            &[
                ("ascending", &fals(py)),
                ("na_position", &pystr(py, "last")),
            ],
        )?),
    )?;
    let ranks = py
        .import("builtins")?
        .getattr("range")?
        .call1((1, result.len()? + 1))?;
    setitem(
        &result,
        "rank",
        np.call_method1(
            "where",
            (
                result.get_item("composite_score")?.call_method0("notna")?,
                ranks,
                f64::NAN,
            ),
        )?,
    )?;
    Ok(result)
}

/// `blend_scores`: ((1 - a) * base + a * ai) * confidence, each term falling
/// back rather than penalising.
#[pyfunction]
pub fn br_blend_scores<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
    quality_weight: &Bound<'py, PyAny>,
    ai_weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let series = pd.getattr("Series")?;
    let coerce = kwargs(py, &[("errors", &pystr(py, "coerce"))])?;
    let index_kw = kwargs(py, &[("index", &frame.getattr("index")?)])?;
    let quality = pd.call_method(
        "to_numeric",
        (frame.get_item("quality_score")?,),
        Some(&coerce),
    )?;
    let mut value = pd.call_method(
        "to_numeric",
        (frame.call_method1("get", ("value_score",))?,),
        Some(&coerce),
    )?;
    if !value.is_instance(&series)? {
        value = series.call((f64::NAN,), Some(&index_kw))?;
    }
    let mut confidence = pd.call_method(
        "to_numeric",
        (frame.call_method1("get", ("confidence",))?,),
        Some(&coerce),
    )?;
    if !confidence.is_instance(&series)? {
        confidence = series.call((1.0,), Some(&index_kw))?;
    }
    let one = o.fl(1.0);
    let mut base = quality.call_method1(
        "where",
        (
            value.call_method0("isna")?,
            quality
                .mul(quality_weight)?
                .add(value.mul(one.sub(quality_weight)?)?)?,
        ),
    )?;
    let b = py.import("builtins")?;
    let w = b.getattr("float")?.call1((b
        .getattr("min")?
        .call1((b.getattr("max")?.call1((ai_weight, 0.0))?, 1.0))?,))?;
    if w.gt(0.0)? && frame.getattr("columns")?.contains("ai_score")? {
        let ai = pd.call_method("to_numeric", (frame.get_item("ai_score")?,), Some(&coerce))?;
        base = base.call_method1(
            "where",
            (
                ai.call_method0("isna")?,
                base.mul(one.sub(&w)?)?.add(ai.mul(&w)?)?,
            ),
        )?;
    }
    base.mul(confidence.call_method1("fillna", (1.0,))?)
}

/// `rerank`: a stored run re-ordered under a different blend, its stored
/// rank kept as `base_rank`.
#[pyfunction]
pub fn br_rerank<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
    quality_weight: &Bound<'py, PyAny>,
    ai_weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = br(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let np = py.import("numpy")?;
    let mut result = frame.call_method0("copy")?;
    if result.getattr("empty")?.is_truthy()? {
        setitem(
            &result,
            "base_rank",
            pd.getattr("Series")?
                .call((), Some(&kwargs(py, &[("dtype", &o.float)])?))?,
        )?;
        return Ok(result);
    }
    setitem(&result, "base_rank", result.call_method1("get", ("rank",))?)?;
    setitem(
        &result,
        "composite_score",
        m.getattr("blend_scores")?
            .call1((&result, quality_weight, ai_weight))?,
    )?;
    result = result.call_method(
        "sort_values",
        (PyList::new(py, ["composite_score", "base_rank"])?,),
        Some(&kwargs(
            py,
            &[
                ("ascending", PyList::new(py, [false, true])?.as_any()),
                ("na_position", &pystr(py, "last")),
            ],
        )?),
    )?;
    let ranks = np.call_method1("arange", (1, result.len()? + 1))?;
    setitem(
        &result,
        "rank",
        np.call_method1(
            "where",
            (
                result.get_item("composite_score")?.call_method0("notna")?,
                ranks,
                f64::NAN,
            ),
        )?,
    )?;
    Ok(result)
}
