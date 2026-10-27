//! `valuation_history`: a company's multiples against its own fifteen-year
//! record — the yearly observations priced as they could have been seen, and
//! today's percentile among them.
//!
//! Ported line for line from `src/valuation_history.py` (frozen in
//! `tests/reference/valuation_history.py`). The band statistics were
//! already Rust (`band_stats`); this is the glue: the store and EDGAR reads,
//! the point-in-time `as_of` context, and the per-year arithmetic, all with
//! Python's own operations on the objects those reads return. Constants and
//! siblings are looked up on `valuation_history` at call time.

use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, unpack};

fn vh(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("valuation_history")
}

/// `_percentile_of`: the share of the history at or below `value`, in percent.
#[pyfunction]
pub fn vh_percentile_of<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    population: &Bound<'py, PyAny>,
) -> PyResult<f64> {
    let _ = py;
    if !population.is_truthy()? {
        return Ok(f64::NAN);
    }
    let mut count = 0usize;
    for v in population.try_iter()? {
        if v?.le(value)? {
            count += 1;
        }
    }
    Ok(100.0 * count as f64 / population.len()? as f64)
}

/// `_describe_percentile`: where today sits, in words — a comparison, never advice.
#[pyfunction]
pub fn vh_describe_percentile(pct: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    Ok(if pct.ge(90)? {
        "dearer than almost all of its own history"
    } else if pct.ge(65)? {
        "dearer than usual for this company"
    } else if pct.le(10)? {
        "cheaper than almost all of its own history"
    } else if pct.le(35)? {
        "cheaper than usual for this company"
    } else {
        "around its own long-run average"
    })
}

/// `_price_on_or_before`: the last positive close on or before `when`, or None.
#[pyfunction]
pub fn vh_price_on_or_before<'py>(
    py: Python<'py>,
    prices: &Bound<'py, PyAny>,
    when: &Bound<'py, PyAny>,
) -> PyResult<Option<f64>> {
    let r = (|| -> PyResult<Option<f64>> {
        let o = Ops::new(py)?;
        let upto = py
            .import("builtins")?
            .getattr("slice")?
            .call1((py.None(), o.str_t.call1((when,))?))?;
        let window = prices.getattr("loc")?.get_item(upto)?;
        if window.getattr("empty")?.is_truthy()? {
            return Ok(None);
        }
        let value = o.float(&window.getattr("iloc")?.get_item(-1)?)?;
        Ok(if value > 0.0 { Some(value) } else { None })
    })();
    match r {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => Ok(None),
        Err(e) => Err(e),
    }
}

/// `_load_prices`: split-adjusted closes from the local store, or None.
#[pyfunction]
pub fn vh_load_prices<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let none = || py.None().into_bound(py);
    let r = (|| -> PyResult<Bound<'py, PyAny>> {
        let module = py.import("market_db")?;
        let class = module.getattr("MarketDatabase").map_err(|_| {
            pyo3::exceptions::PyImportError::new_err(
                "cannot import name 'MarketDatabase' from 'market_db'",
            )
        })?;
        let frame = class
            .call0()?
            .call_method1("get_ohlcv", (symbol.call_method0("upper")?, start, end))?;
        if frame.is_none()
            || frame.getattr("empty")?.is_truthy()?
            || !frame.getattr("columns")?.contains("Close")?
        {
            return Ok(none());
        }
        let series = frame.get_item("Close")?.call_method0("dropna")?;
        Ok(if series.getattr("empty")?.is_truthy()? {
            none()
        } else {
            series
        })
    })();
    match r {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let o = Ops::new(py)?;
            log(
                py,
                "debug",
                format!(
                    "Valuation bands: no price history for {}: {}",
                    o.fmt(symbol)?,
                    es(py, &e)
                ),
            )?;
            Ok(none())
        }
        Err(e) => Err(e),
    }
}

/// `(values.get(concept) or {}).get(period_end)`
fn concept_at<'py>(
    values: &Bound<'py, PyAny>,
    concept: &str,
    period: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = values.py();
    let by_period = values.call_method1("get", (concept,))?;
    let by_period = if by_period.is_truthy()? {
        by_period
    } else {
        PyDict::new(py).into_any()
    };
    by_period.call_method1("get", (period,))
}

/// `_observations`: one multiple per fiscal year, each priced a filing lag
/// after the year closed and with the figures filed by then.
#[pyfunction]
pub fn vh_observations<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    prices: &Bound<'py, PyAny>,
    shares: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = vh(py)?;
    let o = Ops::new(py)?;
    let edgar = py.import("edgar_provider")?;
    let date_t = m.getattr("date")?;
    let years = m.getattr("BAND_YEARS")?;
    let earliest = today.call_method(
        "replace",
        (),
        Some(&kwargs(
            py,
            &[("year", &today.getattr("year")?.sub(&years)?)],
        )?),
    )?;
    let history = PyDict::new(py);
    let earnings = PyList::empty(py);
    let fcf_list = PyList::empty(py);
    history.set_item("earnings", &earnings)?;
    history.set_item("free_cash_flow", &fcf_list)?;
    let lag = m.getattr("timedelta")?.call(
        (),
        Some(&kwargs(py, &[("days", &m.getattr("FILING_LAG_DAYS")?)])?),
    )?;
    let concepts = PyList::new(py, ["net_income", "operating_cash_flow", "capex"])?;

    for period_end in py
        .import("builtins")?
        .getattr("sorted")?
        .call1((shares,))?
        .try_iter()?
    {
        let period_end = period_end?;
        let text = o.str_t.call1((&period_end,))?.get_item(
            py.import("builtins")?
                .getattr("slice")?
                .call1((py.None(), 10))?,
        )?;
        let ended = match date_t.call_method1("fromisoformat", (text,)) {
            Ok(d) => d,
            Err(e) if e.is_instance_of::<PyValueError>(py) => continue,
            Err(e) => return Err(e),
        };
        if ended.lt(&earliest)? {
            continue;
        }
        let known_on = ended.add(&lag)?;
        if known_on.ge(today)? {
            continue;
        }
        let price = m
            .getattr("_price_on_or_before")?
            .call1((prices, &known_on))?;
        let share_count = shares.call_method1("get", (&period_end,))?;
        if !price.is_truthy()? || !share_count.is_truthy()? {
            continue;
        }
        let market_cap = price.mul(&share_count)?;
        let ctx = edgar
            .getattr("as_of")?
            .call1((known_on.call_method0("isoformat")?,))?;
        let values = with1(py, &ctx, |_| {
            edgar.getattr("get_concept_values")?.call1((cik, &concepts))
        })?
        .expect("as_of does not suppress");
        let net_income = concept_at(&values, "net_income", &period_end)?;
        let operating = concept_at(&values, "operating_cash_flow", &period_end)?;
        let capex = concept_at(&values, "capex", &period_end)?;
        if net_income.is_truthy()? && net_income.gt(0)? {
            earnings.append(market_cap.div(&net_income)?)?;
        }
        if !operating.is_none() {
            let capex = if capex.is_truthy()? { capex } else { o.fl(0.0) };
            let fcf = operating.sub(&capex)?;
            if fcf.gt(0)? {
                fcf_list.append(market_cap.div(&fcf)?)?;
            }
        }
    }
    Ok(history)
}

/// `bands`: today's multiples against the company's own fifteen-year record,
/// or [] for anything unmeasurable.
#[pyfunction]
pub fn vh_bands<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = vh(py)?;
    let o = Ops::new(py)?;
    let np = py.import("numpy")?;
    let today = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("date")?.call_method0("today")?
    };
    let fs = o.fmt(symbol)?;
    let measured = (|| -> PyResult<Option<(Bound<'py, PyAny>, Bound<'py, PyDict>)>> {
        let edgar = py.import("edgar_provider")?;
        let series = |concept: &str| {
            edgar
                .getattr("split_consistent_series")?
                .call1((cik, concept))
        };
        let mut shares = series("shares_diluted")?;
        if shares.len()? < m.getattr("MIN_OBSERVATIONS")?.extract::<usize>()? {
            for fallback in ["shares_outstanding", "shares_basic"] {
                let candidate = series(fallback)?;
                if candidate.len()? > shares.len()? {
                    shares = candidate;
                }
            }
        }
        if !shares.is_truthy()? {
            return Ok(None);
        }
        let years = m.getattr("BAND_YEARS")?;
        let start = today.call_method(
            "replace",
            (),
            Some(&kwargs(
                py,
                &[("year", &today.getattr("year")?.sub(&years)?.sub(1)?)],
            )?),
        )?;
        let prices = m.getattr("_load_prices")?.call1((symbol, start, &today))?;
        if prices.is_none() {
            return Ok(None);
        }
        let history = m
            .getattr("_observations")?
            .call1((cik, &prices, &shares, &today))?;
        let latest_price = o.fl(o.float(&prices.getattr("iloc")?.get_item(-1)?)?);
        let latest_period = py.import("builtins")?.getattr("max")?.call1((&shares,))?;
        let current_cap = latest_price.mul(shares.get_item(&latest_period)?)?;
        let values = edgar.getattr("get_concept_values")?.call1((
            cik,
            PyList::new(py, ["net_income", "operating_cash_flow", "capex"])?,
        ))?;
        let income = concept_at(&values, "net_income", &latest_period)?;
        let ocf = concept_at(&values, "operating_cash_flow", &latest_period)?;
        let capex = concept_at(&values, "capex", &latest_period)?;
        let fcf = if ocf.is_none() {
            py.None().into_bound(py)
        } else {
            ocf.sub(if capex.is_truthy()? { capex } else { o.fl(0.0) })?
        };
        let current = PyDict::new(py);
        let ratio = |d: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyAny>> {
            if d.is_truthy()? && d.gt(0)? {
                current_cap.div(d)
            } else {
                Ok(py.None().into_bound(py))
            }
        };
        current.set_item("earnings", ratio(&income)?)?;
        current.set_item("free_cash_flow", ratio(&fcf)?)?;
        Ok(Some((history, current)))
    })();
    let (history, current) = match measured {
        Ok(Some(v)) => v,
        Ok(None) => return Ok(PyList::empty(py)),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("Valuation bands unavailable for {fs}: {}", es(py, &e)),
            )?;
            return Ok(PyList::empty(py));
        }
        Err(e) => return Err(e),
    };

    let results = PyList::empty(py);
    let isfinite = np.getattr("isfinite")?;
    let min_obs = m.getattr("MIN_OBSERVATIONS")?;
    for metric in m.getattr("_METRICS")?.try_iter()? {
        let parts = unpack(&metric?, 3)?;
        let (key, label) = (&parts[0], &parts[1]);
        let population = PyList::empty(py);
        for v in history
            .call_method1("get", (key, PyList::empty(py)))?
            .try_iter()?
        {
            let v = v?;
            if isfinite.call1((&v,))?.is_truthy()? && v.gt(0)? {
                population.append(v)?;
            }
        }
        let now = current.call_method1("get", (key,))?;
        if population.len().into_pyobject(py)?.lt(&min_obs)?
            || !now.is_truthy()?
            || !isfinite.call1((&now,))?.is_truthy()?
        {
            continue;
        }
        let floats: Vec<f64> = population
            .iter()
            .map(|v| o.float(&v))
            .collect::<PyResult<_>>()?;
        let stats = py
            .import("investa_core")?
            .getattr("band_stats")?
            .call1((floats, o.float(&now)?))?;
        let s = unpack(&stats, 6)?;
        let row = PyDict::new(py);
        row.set_item("metric", key)?;
        row.set_item("label", label)?;
        row.set_item("current", o.float(&now)?)?;
        for (name, v) in ["median", "p25", "p75", "low", "high", "percentile"]
            .iter()
            .zip(s.iter())
        {
            row.set_item(*name, v)?;
        }
        row.set_item("observations", population.len())?;
        row.set_item("display", format!("{}x", o.fmt_spec(&now, ".1f")?))?;
        row.set_item("median_display", format!("{}x", o.fmt_spec(&s[0], ".1f")?))?;
        row.set_item(
            "summary",
            m.getattr("_describe_percentile")?.call1((&s[5],))?,
        )?;
        results.append(row)?;
    }
    Ok(results)
}
