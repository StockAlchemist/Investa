//! `buffett_pipeline`: the end-to-end ranking run — the universe collapsed
//! to one listing per filer, sector routing (with inference from the facts
//! when SIC is missing), the metrics and gates, the price-quality gate, the
//! market data, the value score, the composite and the persisted snapshot.
//!
//! Ported line for line from `src/buffett_pipeline.py` (frozen in
//! `tests/reference/buffett_pipeline_p15.py`). The `RunResult` dataclass
//! stays Python. Everything is orchestration of Python objects (the EDGAR
//! store and its sessions, the archive, the provider, the rank store), so
//! every call is Python's own, in the original's order, with siblings and
//! module state looked up on `buffett_pipeline` at call time.

use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PySet, PyTuple};

use crate::mdb::with1;
use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, setitem, unpack};

fn bp(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("buffett_pipeline")
}

/// `_resolve_universe`: (cik, preferred listing) per filer, the shortest
/// symbol standing in for the primary listing.
#[pyfunction]
pub fn bp_resolve_universe<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bp(py)?;
    let uni = m.getattr("universe")?;
    let b = py.import("builtins")?;
    let entries = uni.getattr("get_rankable_universe")?.call0()?;
    let grouped = uni.getattr("group_by_cik")?.call1((entries,))?;
    // key=lambda e: (len(e.symbol), e.symbol)
    let key = PyCFunction::new_closure(py, None, None, |args, _| -> PyResult<Py<PyAny>> {
        let py = args.py();
        let symbol = args.get_item(0)?.getattr("symbol")?;
        Ok(
            PyTuple::new(py, [symbol.len()?.into_pyobject(py)?.into_any(), symbol])?
                .into_any()
                .unbind(),
        )
    })?;
    let mut resolved = PyList::empty(py).into_any();
    for item in b
        .getattr("sorted")?
        .call1((grouped.call_method0("items")?,))?
        .try_iter()?
    {
        let kv = unpack(&item?, 2)?;
        let preferred = b
            .getattr("sorted")?
            .call((&kv[1],), Some(&kwargs(py, &[("key", key.as_any())])?))?
            .get_item(0)?;
        resolved.call_method1("append", (PyTuple::new(py, [kv[0].clone(), preferred])?,))?;
    }
    if limit.is_truthy()? {
        resolved = resolved.get_item(b.getattr("slice")?.call1((py.None(), limit))?)?;
    }
    Ok(resolved)
}

/// `infer_model_from_facts`: bank, REIT or generic from what a filer reports.
#[pyfunction]
pub fn bp_infer_model_from_facts(py: Python<'_>, cik: &Bound<'_, PyAny>) -> PyResult<&'static str> {
    let edgar = py.import("edgar_provider")?;
    let signals = match edgar.getattr("get_concept_values")?.call1((
        cik,
        PyList::new(
            py,
            [
                "deposits",
                "loans",
                "net_interest_income",
                "real_estate_net",
            ],
        )?,
    )) {
        Ok(s) => s,
        Err(e) if e.is_instance_of::<PyException>(py) => return Ok("generic"),
        Err(e) => return Err(e),
    };
    let has = |k: &str| -> PyResult<bool> { signals.call_method1("get", (k,))?.is_truthy() };
    if has("deposits")? && (has("loans")? || has("net_interest_income")?) {
        return Ok("bank");
    }
    if has("real_estate_net")? {
        return Ok("reit");
    }
    Ok("generic")
}

/// `collect_metrics`: metrics and gate results for every filer.
#[pyfunction]
pub fn bp_collect_metrics<'py>(
    py: Python<'py>,
    filers: &Bound<'py, PyAny>,
    progress_every: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let m = bp(py)?;
    let edgar = py.import("edgar_provider")?;
    let sic_mod = m.getattr("edgar_sic")?;
    let sic_map = sic_mod.getattr("get_sic_map")?.call0()?;
    let mut unmapped = 0usize;
    let mut inferred = 0usize;
    let companies = PyList::empty(py);
    let total = filers.len()?;
    for (i, item) in filers.try_iter()?.enumerate() {
        let index = i + 1;
        let kv = unpack(&item?, 2)?;
        let (cik, entry) = (&kv[0], &kv[1]);
        let session = edgar.getattr("company_session")?.call1((cik,))?;
        let company = with1(py, &session, |_| {
            let sic = sic_map.call_method1("get", (cik,))?;
            let model = if sic.is_none() {
                unmapped += 1;
                let model = m.getattr("infer_model_from_facts")?.call1((cik,))?;
                if model.ne("generic")? {
                    inferred += 1;
                }
                model
            } else {
                sic_mod.getattr("model_for_sic")?.call1((sic,))?
            };
            m.getattr("compute_metrics")?.call1((
                cik,
                entry.getattr("symbol")?,
                entry.getattr("name")?,
                model,
            ))
        })?
        .unwrap_or_else(|| py.None().into_bound(py));
        // Gates add to any failure `compute_metrics` recorded.
        company.getattr("gate_failures")?.call_method1(
            "extend",
            (m.getattr("buffett_rank")?
                .getattr("evaluate_gates")?
                .call1((&company,))?,),
        )?;
        companies.append(&company)?;
        // index % progress_every, raising as Python's does for a zero.
        if index
            .into_pyobject(py)?
            .into_any()
            .rem(progress_every)?
            .eq(0)?
        {
            log(
                py,
                "info",
                format!("Rank: metrics for {index}/{total} filers"),
            )?;
        }
    }
    if unmapped > 0 {
        log(
            py,
            "warning",
            format!(
                "Rank: {unmapped} filers have no SIC code; {inferred} were routed to a sector model by inference from their filings, the rest scored as generic"
            ),
        )?;
    }
    Ok(companies)
}

/// `fetch_market_data`: archived closes and cached share counts, the stale
/// counts fetched in batches; per-symbol failures are non-fatal.
#[pyfunction]
pub fn bp_fetch_market_data<'py>(
    py: Python<'py>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = bp(py)?;
    let o = Ops::new(py)?;
    let b = py.import("builtins")?;
    let market_data = py.import("market_data")?;
    let db_class = py.import("market_db")?.getattr("MarketDatabase")?;
    let symbols = b.getattr("list")?.call1((symbols,))?;
    let result = PyDict::new(py);
    let db = db_class.call0()?;
    let prices = db.call_method1("get_latest_closes", (&symbols,))?;
    let cached = db.call_method(
        "get_share_counts",
        (&symbols,),
        Some(&kwargs(
            py,
            &[("max_age_days", &m.getattr("SHARE_COUNT_MAX_AGE_DAYS")?)],
        )?),
    )?;
    let stale = PyList::empty(py);
    for s in symbols.try_iter()? {
        let s = s?;
        if !cached.contains(&s)? {
            stale.append(s)?;
        }
    }
    log(
        py,
        "info",
        format!(
            "Rank: {}/{} prices from the archive; share counts cached for {}, fetching {}",
            prices.len()?,
            symbols.len()?,
            cached.len()?,
            stale.len()
        ),
    )?;
    if stale.is_truthy()? {
        let provider = market_data.getattr("get_shared_mdp")?.call0()?;
        let fetched = PyDict::new(py);
        let batch_size: usize = m.getattr("MARKET_BATCH_SIZE")?.extract()?;
        let n = stale.len();
        let mut start = 0usize;
        while start < n {
            let batch = stale.get_slice(start, (start + batch_size).min(n));
            match provider
                .call_method1("get_ticker_details_batch", (PySet::new(py, batch.iter())?,))
            {
                Ok(details) => {
                    for symbol in batch.iter() {
                        let d = details.call_method1("get", (&symbol,))?;
                        let d = if d.is_truthy()? {
                            d
                        } else {
                            PyDict::new(py).into_any()
                        };
                        let shares = d.call_method1("get", ("sharesOutstanding",))?;
                        if shares.is_truthy()? {
                            fetched.set_item(&symbol, o.float(&shares)?)?;
                        }
                    }
                    log(
                        py,
                        "info",
                        format!("Rank: share counts {}/{n}", (start + batch_size).min(n)),
                    )?;
                }
                Err(e) if e.is_instance_of::<PyException>(py) => {
                    log(
                        py,
                        "warning",
                        format!("Rank: share-count batch failed: {}", es(py, &e)),
                    )?;
                }
                Err(e) => return Err(e),
            }
            start += batch_size;
        }
        if fetched.is_truthy()? {
            db.call_method1(
                "upsert_share_counts",
                (&fetched, m.getattr("date")?.call_method0("today")?),
            )?;
            cached.call_method1("update", (&fetched,))?;
        }
    }
    for s in symbols.try_iter()? {
        let s = s?;
        let price = prices.call_method1("get", (&s,))?;
        let shares = cached.call_method1("get", (&s,))?;
        if price.is_truthy()? && shares.is_truthy()? {
            let d = PyDict::new(py);
            d.set_item("price", price)?;
            d.set_item("shares", shares)?;
            result.set_item(&s, d)?;
        }
    }
    Ok(result)
}

/// `_apply_price_quality_gate`: exclude what a known-bad price history would
/// mis-rank; a missing scan never blocks a run.
#[pyfunction]
pub fn bp_apply_price_quality_gate(
    py: Python<'_>,
    companies: &Bound<'_, PyAny>,
) -> PyResult<usize> {
    let flags = (|| {
        let db = py.import("market_db")?.getattr("MarketDatabase")?.call0()?;
        let symbols = PyList::empty(py);
        for c in companies.try_iter()? {
            let sym = c?.getattr("symbol")?;
            if sym.is_truthy()? {
                symbols.append(sym)?;
            }
        }
        db.call_method1("get_data_quality", (symbols,))
    })();
    let flags = match flags {
        Ok(f) => f,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "debug",
                format!("Rank: price-quality flags unavailable ({})", es(py, &e)),
            )?;
            return Ok(0);
        }
        Err(e) => return Err(e),
    };
    let mut gated = 0usize;
    for c in companies.try_iter()? {
        let c = c?;
        let flag = flags.call_method1("get", (c.getattr("symbol")?,))?;
        let gates = c.getattr("gate_failures")?;
        if flag.is_truthy()?
            && flag.call_method1("get", ("severity",))?.eq("high")?
            && !gates.contains("price_history_unreliable")?
        {
            gates.call_method1("append", ("price_history_unreliable",))?;
            gated += 1;
        }
    }
    if gated > 0 {
        log(
            py,
            "info",
            format!("Rank: {gated} excluded for unreliable price history"),
        )?;
    }
    Ok(gated)
}

/// `_exclusion_records`: one record per gated company.
#[pyfunction]
pub fn bp_exclusion_records<'py>(
    py: Python<'py>,
    companies: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let out = PyList::empty(py);
    for c in companies.try_iter()? {
        let c = c?;
        let gates = c.getattr("gate_failures")?;
        if !gates.is_truthy()? {
            continue;
        }
        let d = PyDict::new(py);
        for k in ["symbol", "cik", "name", "model"] {
            d.set_item(k, c.getattr(k)?)?;
        }
        d.set_item("reasons", pystr(py, ", ").call_method1("join", (&gates,))?)?;
        d.set_item("period_count", c.getattr("period_count")?)?;
        d.set_item("coverage", c.getattr("coverage")?)?;
        out.append(d)?;
    }
    Ok(out)
}

/// `run`: a full ranking run, persisted unless asked not to be.
#[pyfunction]
pub fn bp_run<'py>(
    py: Python<'py>,
    limit: &Bound<'py, PyAny>,
    persist: &Bound<'py, PyAny>,
    skip_market_data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = bp(py)?;
    let o = Ops::new(py)?;
    let time = m.getattr("time")?;
    let rank = m.getattr("buffett_rank")?;
    let value = m.getattr("buffett_value")?;
    let started = time.call_method0("time")?;
    let filers = m
        .getattr("_resolve_universe")?
        .call((), Some(&kwargs(py, &[("limit", limit)])?))?;
    log(
        py,
        "info",
        format!("Rank: starting run over {} filers", filers.len()?),
    )?;

    let store = if persist.is_truthy()? {
        m.getattr("get_store")?.call0()?
    } else {
        py.None().into_bound(py)
    };
    let run_id = if store.is_truthy()? {
        let params = PyDict::new(py);
        params.set_item("limit", limit)?;
        params.set_item("skip_market_data", skip_market_data)?;
        params.set_item("quality_weight", rank.getattr("QUALITY_WEIGHT")?)?;
        params.set_item("value_weight", rank.getattr("VALUE_WEIGHT")?)?;
        store.call_method1("start_run", (filers.len()?, params))?
    } else {
        py.None().into_bound(py)
    };

    let companies = m.getattr("collect_metrics")?.call1((&filers,))?;
    m.getattr("_apply_price_quality_gate")?
        .call1((&companies,))?;
    let eligible = PyList::empty(py);
    for c in companies.try_iter()? {
        let c = c?;
        if !c.getattr("gate_failures")?.is_truthy()? {
            eligible.append(c)?;
        }
    }
    let exclusions = m.getattr("_exclusion_records")?.call1((&companies,))?;
    log(
        py,
        "info",
        format!(
            "Rank: {} eligible, {} excluded by gates",
            eligible.len(),
            exclusions.len()?
        ),
    )?;
    let mut quality = rank.getattr("score_quality")?.call1((&companies,))?;

    let mut value_scores = py.None().into_bound(py);
    if !skip_market_data.is_truthy()? && eligible.is_truthy()? {
        let symbols = PyList::empty(py);
        for c in eligible.iter() {
            symbols.append(c.getattr("symbol")?)?;
        }
        let market = m.getattr("fetch_market_data")?.call1((symbols,))?;
        log(
            py,
            "info",
            format!(
                "Rank: market data for {}/{} symbols",
                market.len()?,
                eligible.len()
            ),
        )?;
        let value_frame = value
            .getattr("build_value_frame")?
            .call1((&eligible, &market))?;
        if !value_frame.getattr("empty")?.is_truthy()? {
            value_scores = value.getattr("score_value")?.call1((&value_frame,))?;
            // The value metrics ride along so a rank stays explainable.
            let carry = PyList::empty(py);
            for c in [
                "price",
                "market_cap",
                "earnings_yield",
                "fcf_yield",
                "ev_to_ebit",
                "price_to_book",
                "price_to_sales",
            ] {
                if value_frame.getattr("columns")?.contains(c)? {
                    carry.append(c)?;
                }
            }
            quality = quality.call_method(
                "join",
                (value_frame.get_item(carry)?,),
                Some(&kwargs(py, &[("how", &pystr(py, "left"))])?),
            )?;
        }
    }

    let ranked = rank.getattr("combine")?.call1((&quality, &value_scores))?;
    let ranked_eligible = ranked
        .get_item(ranked.get_item("eligible")?)?
        .call_method0("copy")?;
    setitem(
        &ranked_eligible,
        "rank",
        py.import("builtins")?
            .getattr("range")?
            .call1((1, ranked_eligible.len()? + 1))?,
    )?;
    let duration = time.call_method0("time")?.sub(&started)?;
    if store.is_truthy()? && !run_id.is_none() {
        store.call_method1("save_scores", (&run_id, &ranked_eligible))?;
        store.call_method1("save_exclusions", (&run_id, &exclusions))?;
        store.call_method1(
            "finish_run",
            (&run_id, ranked_eligible.len()?, exclusions.len()?),
        )?;
    }
    log(
        py,
        "info",
        format!(
            "Rank: run complete in {}s — {} ranked, {} excluded",
            o.fmt_spec(&duration, ".0f")?,
            ranked_eligible.len()?,
            exclusions.len()?
        ),
    )?;
    let stats = PyDict::new(py);
    stats.set_item("filers", filers.len()?)?;
    stats.set_item("eligible", eligible.len())?;
    stats.set_item("excluded", exclusions.len()?)?;
    stats.set_item("ranked", ranked_eligible.len()?)?;
    m.getattr("RunResult")?.call(
        (),
        Some(&kwargs(
            py,
            &[
                ("run_id", &run_id),
                ("ranked", &ranked_eligible),
                ("exclusions", &exclusions),
                ("duration_seconds", &duration),
                ("stats", stats.as_any()),
            ],
        )?),
    )
}
