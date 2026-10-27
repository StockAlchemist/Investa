//! `track_record`: one company's measured record for a reader — the
//! ranking's own metrics labelled and formatted per pillar, its behaviour in
//! the two downturns the filings reach, the numbers it later restated, its
//! latest ranked row and its valuation bands.
//!
//! Ported line for line from `src/track_record.py` (frozen in
//! `tests/reference/track_record_p15.py`). The downturn window search was
//! already Rust (`stress_window`); the rest is formatting and assembly, and
//! every figure is formatted with Python's own `format()` (and rounded with
//! its `round`), so the text is the original's to the character. Tables,
//! constants and siblings are looked up on `track_record` at call time.

use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, unpack};

fn tr(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("track_record")
}

/// `key.replace("_", " ").capitalize()`
fn humanise<'py>(key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    key.call_method1("replace", ("_", " "))?
        .call_method0("capitalize")
}

/// `_format`: the reader-facing form of one metric, or None.
#[pyfunction]
pub fn tr_format<'py>(
    py: Python<'py>,
    value: &Bound<'py, PyAny>,
    unit: &Bound<'py, PyAny>,
    denominator: &Bound<'py, PyAny>,
) -> PyResult<Option<String>> {
    if value.is_none() {
        return Ok(None);
    }
    let o = Ops::new(py)?;
    let r = (|| -> PyResult<String> {
        let f = |v: &Bound<'py, PyAny>, spec: &str| o.fmt_spec(v, spec);
        if unit.eq("percent")? {
            return Ok(format!("{}%", f(value, ".1f")?));
        }
        if unit.eq("cagr")? {
            return Ok(format!("{}%/yr", f(&value.mul(100)?, "+.1f")?));
        }
        if unit.eq("points")? {
            return Ok(format!("{} pts", f(value, ".1f")?));
        }
        if unit.eq("times")? {
            return Ok(format!("{}×", f(value, ".1f")?));
        }
        if unit.eq("share")? {
            if denominator.is_truthy()? {
                let count = py
                    .import("builtins")?
                    .getattr("round")?
                    .call1((value.mul(denominator)?,))?;
                return Ok(format!(
                    "{} of {} years",
                    f(&count, ".0f")?,
                    f(denominator, ".0f")?
                ));
            }
            return Ok(format!("{}% of years", f(&value.mul(100)?, ".0f")?));
        }
        if unit.eq("years")? {
            if denominator.is_truthy()? {
                return Ok(format!(
                    "{} of {} years",
                    f(value, ".0f")?,
                    f(denominator, ".0f")?
                ));
            }
            return Ok(format!("{} years", f(value, ".0f")?));
        }
        f(value, ".2f")
    })();
    match r {
        Ok(s) => Ok(Some(s)),
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// `labelled_metrics`: the pillars of `model`, each metric labelled and
/// formatted; an unmeasurable metric is kept with a null value.
#[pyfunction]
pub fn tr_labelled_metrics<'py>(
    py: Python<'py>,
    metrics: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = tr(py)?;
    let pillars = m.getattr("buffett_rank")?.getattr("PILLARS_BY_MODEL")?;
    let spec = pillars.call_method1("get", (model,))?;
    let spec = if spec.is_truthy()? {
        spec
    } else {
        pillars.get_item("generic")?
    };
    let labels = m.getattr("_METRIC_LABELS")?;
    let denominators = m.getattr("_DENOMINATORS")?;
    let titles = m.getattr("PILLAR_TITLES")?;
    let format = m.getattr("_format")?;
    let groups = PyList::empty(py);
    for item in spec.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let (pillar, entries) = (&kv[0], &kv[1]);
        let items = PyList::empty(py);
        for entry in entries.try_iter()? {
            let e = unpack(&entry?, 2)?;
            let (key, higher) = (&e[0], &e[1]);
            let lu = labels.call_method1(
                "get",
                (key, PyTuple::new(py, [humanise(key)?, pystr(py, "ratio")])?),
            )?;
            let lu = unpack(&lu, 2)?;
            let value = metrics.call_method1("get", (key,))?;
            let denominator =
                metrics.call_method1("get", (denominators.call_method1("get", (key, ""))?,))?;
            let d = PyDict::new(py);
            d.set_item("key", key)?;
            d.set_item("label", &lu[0])?;
            d.set_item("unit", &lu[1])?;
            d.set_item("value", &value)?;
            d.set_item("display", format.call1((&value, &lu[1], &denominator))?)?;
            d.set_item("note", py.None())?;
            d.set_item("higher_is_better", higher)?;
            items.append(d)?;
        }
        let g = PyDict::new(py);
        g.set_item("key", pillar)?;
        g.set_item(
            "title",
            titles.call_method1("get", (pillar, humanise(pillar)?))?,
        )?;
        g.set_item("items", items)?;
        groups.append(g)?;
    }
    Ok(groups)
}

/// `_by_fiscal_year`: {calendar year of the period end: value}, the latest
/// within a year winning.
#[pyfunction]
pub fn tr_by_fiscal_year<'py>(
    py: Python<'py>,
    series: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let o = Ops::new(py)?;
    let b = py.import("builtins")?;
    let out = PyDict::new(py);
    for period_end in b.getattr("sorted")?.call1((series,))?.try_iter()? {
        let period_end = period_end?;
        let r = (|| -> PyResult<Bound<'py, PyAny>> {
            let year = o
                .str_t
                .call1((&period_end,))?
                .get_item(b.getattr("slice")?.call1((py.None(), 4))?)?;
            b.getattr("int")?.call1((year,))
        })();
        let year = match r {
            Ok(y) => y,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        out.set_item(year, series.get_item(&period_end)?)?;
    }
    Ok(out)
}

/// `stress_response`: each metric's peak-to-trough fall in each downturn,
/// and the year it recovered.
#[pyfunction]
pub fn tr_stress_response<'py>(
    py: Python<'py>,
    values: &Bound<'py, PyAny>,
    periods: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = tr(py)?;
    let o = Ops::new(py)?;
    let b = py.import("builtins")?;
    let get = |k: &str| values.call_method1("get", (k, PyDict::new(py)));
    let fcf = PyDict::new(py);
    let ocf = get("operating_cash_flow")?;
    let capex = get("capex")?;
    for item in ocf.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        if kv[1].is_none() {
            continue;
        }
        // EDGAR reports capex as a positive outflow.
        let c = capex.call_method1("get", (&kv[0],))?;
        let c = if c.is_truthy()? { c } else { o.fl(0.0) };
        fcf.set_item(&kv[0], kv[1].sub(c)?)?;
    }
    let by_fy = m.getattr("_by_fiscal_year")?;
    let by_metric = PyDict::new(py);
    by_metric.set_item("revenue", by_fy.call1((get("revenue")?,))?)?;
    by_metric.set_item("net_income", by_fy.call1((get("net_income")?,))?)?;
    by_metric.set_item("free_cash_flow", by_fy.call1((&fcf,))?)?;
    let int = b.getattr("int")?;

    let results = PyList::empty(py);
    for window in periods.try_iter()? {
        let window = window?;
        let items = PyList::empty(py);
        for metric in m.getattr("_STRESS_METRICS")?.try_iter()? {
            let cl = unpack(&metric?, 2)?;
            let (concept, label) = (&cl[0], &cl[1]);
            let annual = by_metric.call_method1("get", (concept,))?;
            let annual = if annual.is_truthy()? {
                annual
            } else {
                PyDict::new(py).into_any()
            };
            let pairs = PyList::empty(py);
            for item in annual.call_method0("items")?.try_iter()? {
                let kv = unpack(&item?, 2)?;
                pairs.append(PyTuple::new(
                    py,
                    [int.call1((&kv[0],))?, o.fl(o.float(&kv[1])?)],
                )?)?;
            }
            let ints = |k: &str| -> PyResult<Bound<'py, PyList>> {
                let l = PyList::empty(py);
                for y in window.get_item(k)?.try_iter()? {
                    l.append(int.call1((y?,))?)?;
                }
                Ok(l)
            };
            let found = py
                .import("investa_core")?
                .getattr("stress_window")?
                .call1((pairs, ints("peak_years")?, ints("trough_years")?))?;
            if found.is_none() {
                continue;
            }
            let f = unpack(&found, 6)?;
            let (peak_year, trough_year, peak, trough, change, recovered) =
                (&f[0], &f[1], &f[2], &f[3], &f[4], &f[5]);
            let recovery = if trough.ge(peak)? {
                py.None().into_bound(py)
            } else if recovered.is_truthy()? {
                pystr(py, &format!("back in {}", o.fmt(recovered)?))
            } else {
                pystr(py, "not back to its peak")
            };
            let d = PyDict::new(py);
            d.set_item("metric", concept)?;
            d.set_item("label", label)?;
            d.set_item("peak_year", peak_year)?;
            d.set_item("trough_year", trough_year)?;
            d.set_item("peak", peak)?;
            d.set_item("trough", trough)?;
            d.set_item("change_pct", change)?;
            d.set_item("display", format!("{}%", o.fmt_spec(change, "+.0f")?))?;
            d.set_item("recovered_year", recovered)?;
            d.set_item("recovery_display", recovery)?;
            items.append(d)?;
        }
        let w = PyDict::new(py);
        w.set_item("key", window.get_item("key")?)?;
        w.set_item("label", window.get_item("label")?)?;
        w.set_item("covered", items.is_truthy()?)?;
        w.set_item("items", items)?;
        results.append(w)?;
    }
    Ok(results)
}

/// `_money`: a filing-scale figure, short enough to sit twice on one line.
#[pyfunction]
pub fn tr_money(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<String> {
    let o = Ops::new(py)?;
    let magnitude = o.abs(value)?;
    let sign = if value.lt(0)? { "-" } else { "" };
    for (scale, suffix) in [(1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")] {
        if magnitude.ge(scale)? {
            return Ok(format!(
                "{sign}${}{suffix}",
                o.fmt_spec(&magnitude.div(scale)?, ",.2f")?
            ));
        }
    }
    Ok(format!("{sign}${}", o.fmt_spec(&magnitude, ",.0f")?))
}

/// `revisions`: the numbers this company changed after first reporting them.
#[pyfunction]
pub fn tr_revisions<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = tr(py)?;
    let o = Ops::new(py)?;
    let out = PyDict::new(py);
    let fetched = (|| -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
        let edgar = py.import("edgar_provider")?;
        let found = edgar.getattr("revisions")?.call1((cik,))?;
        let labels = edgar.getattr("concept_labels")?.call0()?;
        Ok((found, labels))
    })();
    let (found, labels) = match fetched {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!(
                    "Track record: revision history unavailable for {}: {}",
                    o.fmt(cik)?,
                    es(py, &e)
                ),
            )?;
            out.set_item("count", 0)?;
            out.set_item("items", PyList::empty(py))?;
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    let money = m.getattr("_money")?;
    let items = PyList::empty(py);
    let head = found.get_item(
        py.import("builtins")?
            .getattr("slice")?
            .call1((py.None(), limit))?,
    )?;
    for row in head.try_iter()? {
        let row = row?;
        let concept = row.get_item("concept")?;
        let d = PyDict::new(py);
        d.set_item("concept", &concept)?;
        d.set_item(
            "label",
            labels.call_method1("get", (&concept, humanise(&concept)?))?,
        )?;
        d.set_item("period_end", row.get_item("period_end")?)?;
        d.set_item("original", row.get_item("original")?)?;
        d.set_item("current", row.get_item("current")?)?;
        d.set_item("change_pct", row.get_item("change_pct")?)?;
        d.set_item(
            "display",
            format!(
                "{} → {}",
                o.fmt(&money.call1((row.get_item("original")?,))?)?,
                o.fmt(&money.call1((row.get_item("current")?,))?)?
            ),
        )?;
        d.set_item(
            "change_display",
            format!("{}%", o.fmt_spec(&row.get_item("change_pct")?, "+.1f")?),
        )?;
        d.set_item("first_filed", row.get_item("first_filed")?)?;
        d.set_item("restated_filed", row.get_item("restated_filed")?)?;
        items.append(d)?;
    }
    out.set_item("count", found.len()?)?;
    out.set_item("items", items)?;
    Ok(out)
}

/// `_ranked_row`: the company's row in the most recent run, or None.
#[pyfunction]
pub fn tr_ranked_row<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let r = (|| -> PyResult<Bound<'py, PyAny>> {
        let store = py.import("buffett_store")?.getattr("get_store")?.call0()?;
        store.call_method(
            "get_symbol_history",
            (symbol,),
            Some(&kwargs(
                py,
                &[("limit", &1i64.into_pyobject(py)?.into_any())],
            )?),
        )
    })();
    let history = match r {
        Ok(h) => h,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let o = Ops::new(py)?;
            log(
                py,
                "debug",
                format!(
                    "Track record: no ranking history for {}: {}",
                    o.fmt(symbol)?,
                    es(py, &e)
                ),
            )?;
            return Ok(py.None().into_bound(py));
        }
        Err(e) => return Err(e),
    };
    if history.is_truthy()? {
        history.get_item(0)
    } else {
        Ok(py.None().into_bound(py))
    }
}

/// `_bands` / `_stress`: the two sections that may fail on their own.
fn guarded<'py>(
    py: Python<'py>,
    what: &str,
    who: &Bound<'py, PyAny>,
    f: impl FnOnce() -> PyResult<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    match f() {
        Ok(v) => Ok(v),
        Err(e) if e.is_instance_of::<PyException>(py) => {
            let o = Ops::new(py)?;
            log(
                py,
                "debug",
                format!(
                    "Track record: {what} unavailable for {}: {}",
                    o.fmt(who)?,
                    es(py, &e)
                ),
            )?;
            Ok(PyList::empty(py).into_any())
        }
        Err(e) => Err(e),
    }
}

#[pyfunction]
pub fn tr_bands<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "valuation bands", symbol, || {
        py.import("valuation_history")?
            .getattr("bands")?
            .call1((symbol, cik))
    })
}

#[pyfunction]
pub fn tr_stress<'py>(py: Python<'py>, cik: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    guarded(py, "stress history", cik, || {
        let values = py
            .import("edgar_provider")?
            .getattr("get_concept_values")?
            .call1((
                cik,
                PyList::new(
                    py,
                    ["revenue", "net_income", "operating_cash_flow", "capex"],
                )?,
            ))?;
        tr(py)?.getattr("stress_response")?.call1((values,))
    })
}

/// `build`: the whole track record for one filer; never raises for an
/// unmeasurable company.
#[pyfunction]
pub fn tr_build<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = tr(py)?;
    let o = Ops::new(py)?;
    let pipeline = py.import("buffett_pipeline")?;
    let metrics_mod = m.getattr("buffett_metrics")?;
    let rank_mod = m.getattr("buffett_rank")?;
    let symbol = symbol.call_method0("upper")?;
    let fs = o.fmt(&symbol)?;
    let model = match pipeline.getattr("infer_model_from_facts")?.call1((cik,)) {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!(
                    "Track record: model inference failed for {fs}: {}",
                    es(py, &e)
                ),
            )?;
            pystr(py, "generic")
        }
        Err(e) => return Err(e),
    };
    let display_name = if name.is_truthy()? {
        name.clone()
    } else {
        symbol.clone()
    };
    let company =
        metrics_mod
            .getattr("compute_metrics")?
            .call1((cik, &symbol, &display_name, &model))?;
    let list = py.import("builtins")?.getattr("list")?;
    let gates = (|| -> PyResult<Bound<'py, PyAny>> {
        let own = list.call1((company.getattr("gate_failures")?,))?;
        own.add(rank_mod.getattr("evaluate_gates")?.call1((&company,))?)
    })();
    let gate_failures = match gates {
        Ok(all) => all,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!(
                    "Track record: gate evaluation failed for {fs}: {}",
                    es(py, &e)
                ),
            )?;
            list.call1((company.getattr("gate_failures")?,))?
        }
        Err(e) => return Err(e),
    };

    let ranked = m.getattr("_ranked_row")?.call1((&symbol,))?;
    let rank = if ranked.is_truthy()? {
        let r = PyDict::new(py);
        for k in [
            "run_id",
            "rank",
            "composite_score",
            "quality_score",
            "value_score",
            "confidence",
        ] {
            r.set_item(k, ranked.call_method1("get", (k,))?)?;
        }
        let pillars = PyDict::new(py);
        for k in m.getattr("_PILLAR_SCORE_KEYS")?.try_iter()? {
            let k = k?;
            pillars.set_item(&k, ranked.call_method1("get", (&k,))?)?;
        }
        r.set_item("pillars", pillars)?;
        r.into_any()
    } else {
        py.None().into_bound(py)
    };

    let out = PyDict::new(py);
    out.set_item("symbol", &symbol)?;
    out.set_item(
        "name",
        if name.is_truthy()? {
            name.clone()
        } else if ranked.is_truthy()? {
            ranked.call_method1("get", ("name",))?
        } else {
            py.None().into_bound(py)
        },
    )?;
    out.set_item("cik", cik)?;
    out.set_item("model", &model)?;
    out.set_item("period_count", company.getattr("period_count")?)?;
    out.set_item("first_period", company.getattr("first_period")?)?;
    out.set_item("latest_period", company.getattr("latest_period")?)?;
    out.set_item(
        "window_years",
        py.import("builtins")?.getattr("min")?.call1((
            company.getattr("period_count")?,
            metrics_mod.getattr("DURABILITY_WINDOW")?,
        ))?,
    )?;
    out.set_item("coverage", company.getattr("coverage")?)?;
    out.set_item("gate_failures", gate_failures)?;
    out.set_item("rank", rank)?;
    out.set_item(
        "groups",
        m.getattr("labelled_metrics")?
            .call1((company.getattr("metrics")?, &model))?,
    )?;
    out.set_item("revisions", m.getattr("revisions")?.call1((cik,))?)?;
    out.set_item("stress", m.getattr("_stress")?.call1((cik,))?)?;
    out.set_item(
        "valuation_bands",
        m.getattr("_bands")?.call1((&symbol, cik))?,
    )?;
    Ok(out)
}
