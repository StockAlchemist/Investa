//! The pandas side of `risk_metrics`: the value-array conversion, the
//! drawdown series, the annualized statistics' guards, beta's alignment and
//! covariance, alpha, the returns series, and the two entry points (all risk
//! metrics, the benchmark scoreboard).
//!
//! Ported line for line from `src/risk_metrics.py` (frozen in
//! `tests/reference/risk_metrics_p16.py`). The reductions were already Rust
//! (`stats.rs`) and are called as the original called them; the covariance
//! stays pandas' (its BLAS dot product sums in the BLAS's own order).
//! Siblings and `infer_periods_per_year` are looked up on `risk_metrics`.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::fr::{builtin, fl};
use crate::pl_helpers::kwargs;
use crate::pyhelp::unpack;

fn rm(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("risk_metrics")
}

fn core<'py>(m: &Bound<'py, PyModule>, name: &str) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("investa_core")?.getattr(name)
}

/// `_values(series)` through the module.
fn values<'py>(
    m: &Bound<'py, PyModule>,
    series: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    m.getattr("_values")?.call1((series,))
}

#[pyfunction]
pub fn rm_values<'py>(py: Python<'py>, series: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let np = rm(py)?.getattr("np")?;
    let arr = series.call_method(
        "to_numpy",
        (),
        Some(&kwargs(
            py,
            &[
                ("dtype", &np.getattr("float64")?),
                ("na_value", &np.getattr("nan")?),
            ],
        )?),
    )?;
    np.call_method1("ascontiguousarray", (arr,))
}

fn empty(x: &Bound<'_, PyAny>) -> PyResult<bool> {
    x.getattr("empty")?.is_truthy()
}

/// `series.empty or len(series) < 2`
fn too_short(series: &Bound<'_, PyAny>) -> PyResult<bool> {
    Ok(empty(series)? || series.len()? < 2)
}

#[pyfunction]
pub fn rm_drawdown_series<'py>(
    py: Python<'py>,
    series: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    if empty(series)? {
        return pd
            .getattr("Series")?
            .call((), Some(&kwargs(py, &[("dtype", &builtin(py, "float")?)])?));
    }
    let m = rm(py)?;
    let dd = core(&m, "drawdown_series")?.call1((values(&m, series)?,))?;
    pd.getattr("Series")?.call(
        (dd,),
        Some(&kwargs(
            py,
            &[
                ("index", &series.getattr("index")?),
                ("name", &series.getattr("name")?),
            ],
        )?),
    )
}

#[pyfunction]
pub fn rm_max_drawdown<'py>(
    py: Python<'py>,
    series: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = rm(py)?;
    builtin(py, "float")?.call1((core(&m, "max_drawdown")?.call1((values(&m, series)?,))?,))
}

#[pyfunction]
pub fn rm_volatility<'py>(
    py: Python<'py>,
    returns_series: &Bound<'py, PyAny>,
    periods_per_year: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if too_short(returns_series)? {
        return Ok(fl(py, 0.0));
    }
    let m = rm(py)?;
    let float = builtin(py, "float")?;
    let v = core(&m, "volatility")?.call1((
        values(&m, returns_series)?,
        float.call1((periods_per_year,))?,
    ))?;
    float.call1((v,))
}

#[pyfunction]
pub fn rm_sharpe_ratio<'py>(
    py: Python<'py>,
    returns_series: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
    periods_per_year: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if too_short(returns_series)? {
        return Ok(fl(py, 0.0));
    }
    let m = rm(py)?;
    let float = builtin(py, "float")?;
    let v = core(&m, "sharpe_ratio")?.call1((
        values(&m, returns_series)?,
        float.call1((risk_free_rate,))?,
        float.call1((periods_per_year,))?,
    ))?;
    float.call1((v,))
}

#[pyfunction]
pub fn rm_sortino_ratio<'py>(
    py: Python<'py>,
    returns_series: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
    periods_per_year: &Bound<'py, PyAny>,
    target_return: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if too_short(returns_series)? {
        return Ok(fl(py, 0.0));
    }
    let m = rm(py)?;
    let float = builtin(py, "float")?;
    let v = core(&m, "sortino_ratio")?.call1((
        values(&m, returns_series)?,
        float.call1((risk_free_rate,))?,
        float.call1((periods_per_year,))?,
        float.call1((target_return,))?,
    ))?;
    float.call1((v,))
}

/// `df.iloc[:, j]`
fn column<'py>(df: &Bound<'py, PyAny>, j: i64) -> PyResult<Bound<'py, PyAny>> {
    let py = df.py();
    df.getattr("iloc")?
        .get_item((pyo3::types::PySlice::full(py), j))
}

/// `pd.concat([a, b], axis=1, **extra).dropna()`
fn aligned<'py>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    inner: bool,
) -> PyResult<Bound<'py, PyAny>> {
    let one = 1i64.into_pyobject(py)?.into_any();
    let join = crate::pyhelp::pystr(py, "inner");
    let mut kw: Vec<(&str, &Bound<'py, PyAny>)> = vec![("axis", &one)];
    if inner {
        kw.push(("join", &join));
    }
    py.import("pandas")?
        .getattr("concat")?
        .call((PyList::new(py, [a, b])?,), Some(&kwargs(py, &kw)?))?
        .call_method0("dropna")
}

#[pyfunction]
pub fn rm_beta<'py>(
    py: Python<'py>,
    returns_series: &Bound<'py, PyAny>,
    benchmark_returns: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if empty(returns_series)? || empty(benchmark_returns)? {
        return Ok(fl(py, 1.0));
    }
    let both = aligned(py, returns_series, benchmark_returns, false)?;
    if both.len()? < 2 {
        return Ok(fl(py, 1.0));
    }
    let m = rm(py)?;
    let cov = column(&both, 0)?.call_method1("cov", (column(&both, 1)?,))?;
    let var = core(&m, "pd_var")?.call1((values(&m, &column(&both, 1)?)?, 1))?;
    if var.eq(0)? {
        return Ok(fl(py, 1.0));
    }
    builtin(py, "float")?.call1((cov.div(var)?,))
}

#[pyfunction]
pub fn rm_alpha<'py>(
    py: Python<'py>,
    portfolio_return_ann: &Bound<'py, PyAny>,
    benchmark_return_ann: &Bound<'py, PyAny>,
    beta: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let expected = risk_free_rate.add(beta.mul(benchmark_return_ann.sub(risk_free_rate)?)?)?;
    let alpha = portfolio_return_ann.sub(expected)?;
    builtin(py, "float")?.call1((alpha,))
}

#[pyfunction]
pub fn rm_returns<'py>(py: Python<'py>, series: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let m = rm(py)?;
    let r = core(&m, "pct_change")?.call1((values(&m, series)?,))?;
    py.import("pandas")?
        .getattr("Series")?
        .call(
            (r,),
            Some(&kwargs(py, &[("index", &series.getattr("index")?)])?),
        )?
        .call_method0("dropna")
}

/// `returns.clip(lower=-0.90, upper=1.0)`
fn clipped<'py>(py: Python<'py>, returns: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    returns.call_method(
        "clip",
        (),
        Some(&kwargs(
            py,
            &[("lower", &fl(py, -0.90)), ("upper", &fl(py, 1.0))],
        )?),
    )
}

#[pyfunction]
pub fn rm_all_risk_metrics<'py>(
    py: Python<'py>,
    portfolio_values: &Bound<'py, PyAny>,
    risk_free_rate: &Bound<'py, PyAny>,
    periods_per_year: &Bound<'py, PyAny>,
    benchmark_values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let metrics = PyDict::new(py);
    if empty(portfolio_values)? {
        return Ok(metrics);
    }
    let m = rm(py)?;
    let returns_of = m.getattr("_returns")?;
    let mut returns = returns_of.call1((portfolio_values,))?;

    // Clip data artifacts (missing flows, splits).
    if !empty(&returns)? {
        returns = clipped(py, &returns)?;
    }

    // MDD works on the value series.
    let mdd = m
        .getattr("calculate_max_drawdown")?
        .call1((portfolio_values,))?;

    if returns.len()? < 5 {
        metrics.set_item("Max Drawdown", mdd)?;
        metrics.set_item("Volatility (Ann.)", 0.0)?;
        metrics.set_item("Sharpe Ratio", 0.0)?;
        metrics.set_item("Sortino Ratio", 0.0)?;
        metrics.set_item("insufficient_data", true)?;
        return Ok(metrics);
    }

    let ppy = if periods_per_year.is_none() {
        m.getattr("infer_periods_per_year")?
            .call1((returns.getattr("index")?,))?
    } else {
        periods_per_year.clone()
    };

    metrics.set_item("Max Drawdown", mdd)?;
    metrics.set_item(
        "Volatility (Ann.)",
        m.getattr("calculate_volatility")?.call1((&returns, &ppy))?,
    )?;
    metrics.set_item(
        "Sharpe Ratio",
        m.getattr("calculate_sharpe_ratio")?
            .call1((&returns, risk_free_rate, &ppy))?,
    )?;
    metrics.set_item(
        "Sortino Ratio",
        m.getattr("calculate_sortino_ratio")?
            .call1((&returns, risk_free_rate, &ppy))?,
    )?;

    // Add Alpha & Beta if benchmark provided
    if !benchmark_values.is_none() && !empty(benchmark_values)? {
        let bench_returns = returns_of.call1((benchmark_values,))?;
        if !empty(&bench_returns)? {
            let beta = m
                .getattr("calculate_beta")?
                .call1((&returns, &bench_returns))?;
            let mean = core(&m, "pd_mean")?;
            let port_ann = mean.call1((values(&m, &returns)?,))?.mul(&ppy)?;
            let bench_ann = mean.call1((values(&m, &bench_returns)?,))?.mul(&ppy)?;
            let alpha = m.getattr("calculate_alpha")?.call1((
                port_ann,
                bench_ann,
                &beta,
                risk_free_rate,
            ))?;
            metrics.set_item("Beta", beta)?;
            metrics.set_item("Alpha", alpha)?;
        }
    }
    Ok(metrics)
}

#[pyfunction]
pub fn rm_benchmark_scoreboard<'py>(
    py: Python<'py>,
    portfolio_values: &Bound<'py, PyAny>,
    benchmark_values: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    if portfolio_values.is_none() || empty(portfolio_values)? {
        return Ok(out);
    }
    let m = rm(py)?;
    let returns_of = m.getattr("_returns")?;
    let clipped_returns = |s: &Bound<'py, PyAny>| clipped(py, &returns_of.call1((s,))?);

    let port_ret = clipped_returns(portfolio_values)?;
    if port_ret.len()? < 20 {
        return Ok(out);
    }
    let ppy = m
        .getattr("infer_periods_per_year")?
        .call1((port_ret.getattr("index")?,))?;
    let float = builtin(py, "float")?;
    let total = |s: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
        let iloc = s.getattr("iloc")?;
        float
            .call1((iloc.get_item(-1)?,))?
            .div(float.call1((iloc.get_item(0)?,))?)?
            .sub(1.0)?
            .mul(100.0)
    };

    for item in benchmark_values.call_method0("items")?.try_iter()? {
        let nb = unpack(&item?, 2)?;
        let (name, bvals) = (&nb[0], &nb[1]);
        if bvals.is_none() || empty(bvals)? {
            continue;
        }
        let both = aligned(py, &port_ret, &clipped_returns(bvals)?, true)?;
        if both.len()? < 20 {
            continue;
        }
        let stats = core(&m, "scoreboard_stats")?.call1((
            values(&m, &column(&both, 0)?)?,
            values(&m, &column(&both, 1)?)?,
            float.call1((&ppy,))?,
        ))?;
        let st = unpack(&stats, 5)?;
        let port_total = total(portfolio_values)?;
        let bench_total = total(bvals)?;
        let row = PyDict::new(py);
        row.set_item("name", name)?;
        for (k, v) in ["alpha", "beta", "r2", "tracking_error", "information_ratio"]
            .iter()
            .zip(&st)
        {
            row.set_item(*k, float.call1((v,))?)?;
        }
        row.set_item(
            "excess_return",
            float.call1((port_total.sub(bench_total)?,))?,
        )?;
        out.append(row)?;
    }
    Ok(out)
}
