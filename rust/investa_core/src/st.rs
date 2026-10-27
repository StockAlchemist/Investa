//! `strategies`: the market-trend indicator and the strategy allocation —
//! month-end closes and the moving-average readings, the signal prices, the
//! ranked positions under the industry cap, the allocation with its stale and
//! short-book warnings, and the serialisable payloads.
//!
//! Ported line for line from `src/strategies.py` (frozen in
//! `tests/reference/strategies_p15.py`). The strategy registry and its
//! dataclasses stay Python: they are definitions, not calculation. The SMA
//! arithmetic was already Rust (`trend_sma`); the rest is pandas and dict
//! glue, formatted with Python's own `format()`. Constants and siblings are
//! looked up on `strategies` at call time.

use pyo3::basic::CompareOp;
use pyo3::exceptions::{PyException, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet};

use crate::pa_rows::Ops;
use crate::pl_helpers::kwargs;
use crate::pyhelp::{es, log, pystr, setitem, unpack};

fn st(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("strategies")
}

/// `str(x.date() if hasattr(x, "date") else x)`
fn date_text<'py>(o: &Ops<'py>, stamp: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let v = if stamp.hasattr("date")? {
        stamp.call_method0("date")?
    } else {
        stamp.clone()
    };
    o.str_t.call1((v,))
}

/// `month_end_closes`: the last actual close of each calendar month, on its own date.
#[pyfunction]
pub fn st_month_end_closes<'py>(
    py: Python<'py>,
    prices: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let mut clean = prices.call_method0("dropna")?;
    if clean.getattr("empty")?.is_truthy()? {
        return Ok(clean);
    }
    let index = clean.getattr("index")?;
    if index.is_instance(&pd.getattr("DatetimeIndex")?)? && !index.getattr("tz")?.is_none() {
        clean = clean.call_method0("copy")?;
        clean.setattr("index", index.call_method1("tz_localize", (py.None(),))?)?;
    }
    clean
        .call_method1(
            "groupby",
            (clean.getattr("index")?.call_method1("to_period", ("M",))?,),
        )?
        .call_method1("tail", (1,))
}

/// `_next_month_end`: the calendar month-end of `today`'s month.
#[pyfunction]
pub fn st_next_month_end<'py>(
    py: Python<'py>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = st(py)?;
    let date_t = m.getattr("date")?;
    let year = today.getattr("year")?;
    if today.getattr("month")?.eq(12)? {
        return date_t.call1((year, 12, 31));
    }
    let first = date_t.call1((&year, today.getattr("month")?.add(1)?, 1))?;
    first.sub(m.getattr("timedelta")?.call(
        (),
        Some(&kwargs(
            py,
            &[("days", &1i64.into_pyobject(py)?.into_any())],
        )?),
    )?)
}

/// `evaluate_trend_signal`: the active and provisional readings of the
/// moving-average rule, or None without enough history.
#[pyfunction]
pub fn st_evaluate_trend_signal<'py>(
    py: Python<'py>,
    prices: &Bound<'py, PyAny>,
    sma_months: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let none = || py.None().into_bound(py);
    let monthly = m.getattr("month_end_closes")?.call1((prices,))?;
    if monthly.getattr("empty")?.is_truthy()? {
        return Ok(none());
    }
    let today = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("get_est_today")?.call0()?
    };
    let current = pd
        .getattr("Period")?
        .call((&today,), Some(&kwargs(py, &[("freq", &pystr(py, "M"))])?))?;
    // The running month's close is not a month-end yet.
    let periods = monthly
        .getattr("index")?
        .call_method1("to_period", ("M",))?;
    let completed = monthly.get_item(periods.rich_compare(&current, CompareOp::Lt)?)?;
    if completed.len()?.into_pyobject(py)?.lt(sma_months)? {
        return Ok(none());
    }
    let dropped = prices.call_method0("dropna")?;
    let latest_close = o.float(&dropped.getattr("iloc")?.get_item(-1)?)?;
    let closes = completed
        .call_method("to_numpy", (), Some(&kwargs(py, &[("dtype", &o.float)])?))?
        .call_method0("tolist")?;
    let sma_parts = py.import("investa_core")?.getattr("trend_sma")?.call1((
        closes,
        py.import("builtins")?
            .getattr("int")?
            .call1((sma_months,))?,
        latest_close,
    ))?;
    let s = unpack(&sma_parts, 4)?;
    let (sma, provisional_sma, flip_close, smas) = (&s[0], &s[1], &s[2], &s[3]);
    let decision_close = o.fl(o.float(&completed.getattr("iloc")?.get_item(-1)?)?);
    let decision_stamp = completed.getattr("index")?.get_item(-1)?;
    let state = if decision_close.ge(sma)? { "in" } else { "out" };
    let latest_stamp = prices
        .call_method0("dropna")?
        .getattr("index")?
        .get_item(-1)?;
    let latest = o.fl(latest_close);
    let provisional_state = if latest.ge(provisional_sma)? {
        "in"
    } else {
        "out"
    };
    let distance = if flip_close.is_truthy()? {
        latest.div(flip_close)?.sub(1.0)?.mul(100.0)?
    } else {
        none()
    };

    let history = PyList::empty(py);
    for (i, item) in completed.call_method0("items")?.try_iter()?.enumerate() {
        let kv = unpack(&item?, 2)?;
        let d = PyDict::new(py);
        d.set_item("date", date_text(&o, &kv[0])?)?;
        d.set_item("close", o.float(&kv[1])?)?;
        d.set_item("sma", smas.get_item(i)?)?;
        history.append(d)?;
    }
    let history = history.as_any().get_item(
        py.import("builtins")?
            .getattr("slice")?
            .call1((-36, py.None()))?,
    )?;

    let out = PyDict::new(py);
    out.set_item("state", state)?;
    out.set_item("sma_months", sma_months)?;
    out.set_item("decision_date", date_text(&o, &decision_stamp)?)?;
    out.set_item("decision_close", &decision_close)?;
    out.set_item("sma", sma)?;
    out.set_item("governs_month", o.str_t.call1((&current,))?)?;
    out.set_item("provisional_state", provisional_state)?;
    out.set_item("provisional_sma", provisional_sma)?;
    out.set_item("latest_close", latest_close)?;
    out.set_item("latest_date", date_text(&o, &latest_stamp)?)?;
    out.set_item("flip_close", flip_close)?;
    out.set_item("distance_pct", distance)?;
    out.set_item("would_flip", provisional_state != state)?;
    out.set_item(
        "next_decision_date",
        o.str_t
            .call1((m.getattr("_next_month_end")?.call1((&today,))?,))?,
    )?;
    out.set_item("history", history)?;
    Ok(out.into_any())
}

/// The shared market-data provider's daily history, `(history, failed)`.
fn mdp_history<'py>(
    py: Python<'py>,
    symbols: Bound<'py, PyAny>,
    start: &Bound<'py, PyAny>,
    end: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let mdp = py
        .import("server.route_utils")?
        .getattr("get_mdp")?
        .call0()?;
    let got = mdp.call_method1("get_historical_data", (symbols, start, end))?;
    Ok(unpack(&got, 2)?.remove(0))
}

/// The `price` column of a history frame without NaNs, or None.
fn price_series<'py>(frame: &Bound<'py, PyAny>) -> PyResult<Option<Bound<'py, PyAny>>> {
    if frame.is_none()
        || frame.getattr("empty")?.is_truthy()?
        || !frame.getattr("columns")?.contains("price")?
    {
        return Ok(None);
    }
    let s = frame.get_item("price")?.call_method0("dropna")?;
    Ok(if s.getattr("empty")?.is_truthy()? {
        None
    } else {
        Some(s)
    })
}

/// `load_signal_prices`: daily closes for the signal symbol.
#[pyfunction]
pub fn st_load_signal_prices<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let today = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("get_est_today")?.call0()?
    };
    let start = today.sub(m.getattr("timedelta")?.call(
        (),
        Some(&kwargs(
            py,
            &[("days", &m.getattr("_SIGNAL_LOOKBACK_DAYS")?)],
        )?),
    )?)?;
    let history = match mdp_history(py, PyList::new(py, [symbol])?.into_any(), &start, &today) {
        Ok(h) => h,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "error",
                format!(
                    "Strategies: price fetch failed for {}: {}",
                    o.fmt(symbol)?,
                    es(py, &e)
                ),
            )?;
            return Ok(py.None().into_bound(py));
        }
        Err(e) => return Err(e),
    };
    Ok(price_series(&history.call_method1("get", (symbol,))?)?
        .unwrap_or_else(|| py.None().into_bound(py)))
}

/// `latest_closes`: the most recent close per symbol.
#[pyfunction]
pub fn st_latest_closes<'py>(
    py: Python<'py>,
    symbols: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let out = PyDict::new(py);
    let today = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("get_est_today")?.call0()?
    };
    let wanted_set = PySet::empty(py)?;
    for s in symbols.try_iter()? {
        let s = s?;
        if s.is_truthy()? {
            wanted_set.add(s)?;
        }
    }
    let wanted = py
        .import("builtins")?
        .getattr("sorted")?
        .call1((wanted_set,))?;
    if !wanted.is_truthy()? {
        return Ok(out);
    }
    let start = today.sub(m.getattr("timedelta")?.call(
        (),
        Some(&kwargs(
            py,
            &[("days", &14i64.into_pyobject(py)?.into_any())],
        )?),
    )?)?;
    let history = match mdp_history(
        py,
        py.import("builtins")?.getattr("list")?.call1((&wanted,))?,
        &start,
        &today,
    ) {
        Ok(h) => h,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!(
                    "Strategies: quote fetch failed for {}: {}",
                    o.fmt(&wanted)?,
                    es(py, &e)
                ),
            )?;
            return Ok(out);
        }
        Err(e) => return Err(e),
    };
    for s in wanted.try_iter()? {
        let s = s?;
        if let Some(series) = price_series(&history.call_method1("get", (&s,))?)? {
            out.set_item(&s, o.float(&series.getattr("iloc")?.get_item(-1)?)?)?;
        }
    }
    Ok(out)
}

/// `market_trend_signal`: one index's close against its own moving average,
/// tagged so it describes itself.
#[pyfunction]
pub fn st_market_trend_signal<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    sma_months: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = st(py)?;
    let prices = m.getattr("load_signal_prices")?.call1((symbol, today))?;
    if prices.is_none() {
        return Ok(prices);
    }
    let signal = m
        .getattr("evaluate_trend_signal")?
        .call1((&prices, sma_months, today))?;
    if signal.is_none() {
        return Ok(signal);
    }
    signal.set_item("signal_symbol", symbol)?;
    signal.set_item(
        "signal_name",
        m.getattr("_SIGNAL_DISPLAY_NAMES")?
            .call_method1("get", (symbol.call_method0("upper")?, symbol))?,
    )?;
    signal.set_item("market_timezone", m.getattr("MARKET_SIGNAL_TIMEZONE")?)?;
    signal.set_item("advisory_only", true)?;
    Ok(signal)
}

/// `_optional_float`: `float(value)`, or None for a non-number or NaN.
#[pyfunction]
pub fn st_optional_float(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
    let o = Ops::new(py)?;
    match o.float(value) {
        Ok(v) => Ok(if v.is_nan() { None } else { Some(v) }),
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// `_industry_label`: the display name of a SIC code's industry.
#[pyfunction]
pub fn st_industry_label<'py>(
    py: Python<'py>,
    sic: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !sic.is_truthy()? {
        return Ok(pystr(py, "Unclassified"));
    }
    let m = st(py)?;
    let exact = m.getattr("_SIC_EXACT")?.call_method1("get", (sic,))?;
    if exact.is_truthy()? {
        return Ok(exact);
    }
    let major = sic.get_item(
        py.import("builtins")?
            .getattr("slice")?
            .call1((py.None(), 2))?,
    )?;
    m.getattr("_SIC_GROUPS")?
        .call_method1("get", (major, "Other"))
}

/// `_apply_sector_cap`: walk the ranked list, skipping a name once its
/// industry is full.
#[pyfunction]
pub fn st_apply_sector_cap<'py>(
    py: Python<'py>,
    frame: &Bound<'py, PyAny>,
    scores: &Bound<'py, PyAny>,
    sleeve: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let b = py.import("builtins")?;
    let order = scores
        .call_method(
            "sort_values",
            (),
            Some(&kwargs(
                py,
                &[(
                    "ascending",
                    &pyo3::types::PyBool::new(py, false).to_owned().into_any(),
                )],
            )?),
        )?
        .getattr("index")?;
    let sic_map = match py
        .import("edgar_sic")
        .and_then(|s| s.getattr("get_sic_map")?.call0())
    {
        Ok(v) => v,
        Err(e) if e.is_instance_of::<PyException>(py) => {
            log(
                py,
                "warning",
                format!(
                    "Strategies: SIC map unavailable, industry cap disabled: {}",
                    es(py, &e)
                ),
            )?;
            PyDict::new(py).into_any()
        }
        Err(e) => return Err(e),
    };
    let codes = PyDict::new(py);
    let int = b.getattr("int")?;
    for item in frame.get_item("cik")?.call_method0("items")?.try_iter()? {
        let kv = unpack(&item?, 2)?;
        let key = match o.float(&kv[1]).and_then(|f| int.call1((f,))) {
            Ok(i) => o.str_t.call1((i,))?.call_method1("zfill", (10,))?,
            Err(e)
                if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) =>
            {
                continue;
            }
            Err(e) => return Err(e),
        };
        let sic = sic_map.call_method1("get", (key,))?;
        if !sic.is_none() {
            codes.set_item(
                &kv[0],
                o.str_t
                    .call1((int.call1((sic,))?,))?
                    .call_method1("zfill", (4,))?,
            )?;
        }
    }
    let counts = PyDict::new(py);
    let picked = PyList::empty(py);
    let max_per = sleeve.getattr("max_per_sector")?;
    let digits = sleeve.getattr("sector_digits")?;
    let top_n = sleeve.getattr("top_n")?;
    for index in order.try_iter()? {
        let index = index?;
        if pd
            .call_method1("isna", (scores.call_method1("get", (&index,))?,))?
            .is_truthy()?
        {
            continue;
        }
        let full = codes.call_method1("get", (&index, ""))?;
        // An unclassified company is its own industry.
        let group = if full.is_truthy()? {
            full.get_item(b.getattr("slice")?.call1((py.None(), &digits))?)?
        } else {
            pystr(py, &format!("?{}", o.fmt(&index)?))
        };
        if max_per.is_truthy()? && counts.call_method1("get", (&group, 0))?.ge(&max_per)? {
            continue;
        }
        counts.set_item(&group, counts.call_method1("get", (&group, 0))?.add(1)?)?;
        picked.append(&index)?;
        if picked.len().into_pyobject(py)?.into_any().eq(&top_n)? {
            break;
        }
    }
    let result = frame
        .getattr("loc")?
        .get_item(&picked)?
        .call_method0("copy")?;
    setitem(&result, "_score", scores.getattr("loc")?.get_item(&picked)?)?;
    let labels = PyList::empty(py);
    let label = m.getattr("_industry_label")?;
    for i in picked.iter() {
        labels.append(label.call1((codes.call_method1("get", (i, ""))?,))?)?;
    }
    setitem(&result, "_industry", labels)?;
    Ok(result)
}

/// `_ranking_positions`: the top-N names from the stored ranking, equally
/// weighted, priced live with the snapshot close as the fallback.
#[pyfunction]
pub fn st_ranking_positions<'py>(
    py: Python<'py>,
    sleeve: &Bound<'py, PyAny>,
    capital: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
    ai_weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let pd = py.import("pandas")?;
    let out = PyDict::new(py);
    let store = || py.import("buffett_store")?.getattr("get_store")?.call0();
    let run = store()?.call_method0("get_run")?;
    if !run.is_truthy()? {
        out.set_item("positions", PyList::empty(py))?;
        out.set_item("run", py.None())?;
        out.set_item("error", "No completed ranking run yet")?;
        return Ok(out);
    }
    let mut frame = store()?.call_method1("get_scores_frame", (run.get_item("run_id")?,))?;
    if frame.is_none() || frame.getattr("empty")?.is_truthy()? {
        out.set_item("positions", PyList::empty(py))?;
        out.set_item("run", &run)?;
        out.set_item("error", "Ranking run has no scores")?;
        return Ok(out);
    }
    let weight = if ai_weight.is_none() {
        sleeve.getattr("ai_weight")?
    } else {
        ai_weight.clone()
    };
    frame = m.getattr("attach_ai_scores")?.call1((&frame,))?;

    let min_cap = sleeve.getattr("min_market_cap")?;
    if min_cap.is_truthy()? && frame.getattr("columns")?.contains("market_cap")? {
        let caps = pd.call_method(
            "to_numeric",
            (frame.get_item("market_cap")?,),
            Some(&kwargs(py, &[("errors", &pystr(py, "coerce"))])?),
        )?;
        frame = frame.get_item(caps.rich_compare(&min_cap, CompareOp::Ge)?)?;
        if frame.getattr("empty")?.is_truthy()? {
            out.set_item("positions", PyList::empty(py))?;
            out.set_item("run", &run)?;
            out.set_item(
                "error",
                format!(
                    "No companies meet the minimum market cap of ${}",
                    o.fmt_spec(&min_cap, ",.0f")?
                ),
            )?;
            return Ok(out);
        }
    }
    let blended =
        m.getattr("blend_scores")?
            .call1((&frame, sleeve.getattr("quality_weight")?, &weight))?;
    let picked = m
        .getattr("_apply_sector_cap")?
        .call1((&frame, &blended, sleeve))?;
    let top_n = sleeve.getattr("top_n")?;
    let per_position = if top_n.is_truthy()? {
        capital.div(&top_n)?
    } else {
        o.fl(0.0)
    };

    let symbols = PyList::empty(py);
    for item in picked.call_method0("iterrows")?.try_iter()? {
        let row = unpack(&item?, 2)?.remove(1);
        symbols.append(o.str_t.call1((row.get_item("symbol")?,))?)?;
    }
    let live = m.getattr("latest_closes")?.call1((&symbols, today))?;

    let positions = PyList::empty(py);
    let mut live_count = 0usize;
    let floor = py.import("math")?.getattr("floor")?;
    for item in picked.call_method0("iterrows")?.try_iter()? {
        let row = unpack(&item?, 2)?.remove(1);
        let symbol = o.str_t.call1((row.get_item("symbol")?,))?;
        let snapshot = row.call_method1("get", ("price",))?;
        let snapshot =
            if !snapshot.is_none() && !pd.call_method1("isna", (&snapshot,))?.is_truthy()? {
                o.fl(o.float(&snapshot)?)
            } else {
                py.None().into_bound(py)
            };
        let quoted = live.call_method1("get", (&symbol,))?;
        let quoted_ok = quoted.is_truthy()? && quoted.gt(0)?;
        let price = if quoted_ok { quoted.clone() } else { snapshot };
        if quoted.is_truthy()? && quoted.gt(0)? {
            live_count += 1;
        }
        let shares = if price.is_truthy()? && price.gt(0)? {
            floor.call1((per_position.div(&price)?,))?
        } else {
            py.None().into_bound(py)
        };
        let d = PyDict::new(py);
        d.set_item("symbol", &symbol)?;
        d.set_item("name", row.call_method1("get", ("name",))?)?;
        d.set_item("role", "stock")?;
        d.set_item(
            "weight",
            if top_n.is_truthy()? {
                o.fl(1.0).div(&top_n)?
            } else {
                o.fl(0.0)
            },
        )?;
        d.set_item("amount", &per_position)?;
        d.set_item("price", &price)?;
        d.set_item("shares", &shares)?;
        d.set_item(
            "cost",
            if shares.is_truthy()? && price.is_truthy()? {
                shares.mul(&price)?
            } else {
                py.None().into_bound(py)
            },
        )?;
        d.set_item("score", o.float(&row.get_item("_score")?)?)?;
        d.set_item("industry", row.call_method1("get", ("_industry",))?)?;
        d.set_item(
            "ai_rating",
            m.getattr("_optional_float")?
                .call1((row.call_method1("get", ("ai_rating",))?,))?,
        )?;
        positions.append(d)?;
    }
    let price_source = if positions.is_empty() || live_count == 0 {
        "snapshot"
    } else if live_count == positions.len() {
        "live"
    } else {
        "mixed"
    };
    out.set_item("positions", positions)?;
    out.set_item("run", &run)?;
    out.set_item("error", py.None())?;
    out.set_item("price_source", price_source)?;
    out.set_item("ai_weight", &weight)?;
    Ok(out)
}

/// `ranking_age_days`: whole days between a run's completion and `today`.
#[pyfunction]
pub fn st_ranking_age_days<'py>(
    py: Python<'py>,
    finished_at: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let none = py.None().into_bound(py);
    if !finished_at.is_truthy()? {
        return Ok(none);
    }
    let m = st(py)?;
    let o = Ops::new(py)?;
    let finished = match m
        .getattr("datetime")?
        .call_method1("fromisoformat", (o.str_t.call1((finished_at,))?,))
        .and_then(|d| d.call_method0("date"))
    {
        Ok(d) => d,
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            return Ok(none);
        }
        Err(e) => return Err(e),
    };
    let today = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("date")?.call_method0("today")?
    };
    let zero = m.getattr("timedelta")?.call1((0,))?;
    py.import("builtins")?
        .getattr("max")?
        .call1((today.sub(finished)?, zero))?
        .getattr("days")
}

/// `_sleeve_label`: the sleeve's one-line description.
#[pyfunction]
pub fn st_sleeve_label(
    py: Python<'_>,
    sleeve: &Bound<'_, PyAny>,
    ai_weight: &Bound<'_, PyAny>,
) -> PyResult<String> {
    let o = Ops::new(py)?;
    let mut parts = vec![
        format!("top {}", o.fmt(&sleeve.getattr("top_n")?)?),
        format!(
            "{} quality",
            o.fmt_spec(&sleeve.getattr("quality_weight")?, ".0%")?
        ),
    ];
    if ai_weight.gt(0)? {
        parts.push(format!(
            "AI review {} of the score",
            o.fmt_spec(ai_weight, ".0%")?
        ));
    }
    Ok(format!("Quality sleeve ({})", parts.join(", ")))
}

/// `build_allocation`: what the strategy says to hold today, sized to
/// `capital`, with stale-snapshot and short-book warnings.
#[pyfunction]
pub fn st_build_allocation<'py>(
    py: Python<'py>,
    strategy: &Bound<'py, PyAny>,
    capital: &Bound<'py, PyAny>,
    today: &Bound<'py, PyAny>,
    ai_weight: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = st(py)?;
    let o = Ops::new(py)?;
    let ranking = strategy.getattr("ranking")?;
    let warnings = PyList::empty(py);
    let built = m
        .getattr("_ranking_positions")?
        .call1((&ranking, capital, today, ai_weight))?;
    let fallback = if ai_weight.is_none() {
        ranking.getattr("ai_weight")?
    } else {
        ai_weight.clone()
    };
    let applied = built.call_method1("get", ("ai_weight", fallback))?;
    let error = built.get_item("error")?;
    if error.is_truthy()? {
        warnings.append(&error)?;
    }
    let run = built.get_item("run")?;
    let run = if run.is_truthy()? {
        run
    } else {
        PyDict::new(py).into_any()
    };
    let ranked_at = run.call_method1("get", ("finished_at",))?;
    let age = m.getattr("ranking_age_days")?.call1((&ranked_at, today))?;
    let stale_days = m.getattr("STALE_RANKING_DAYS")?;
    if !age.is_none() && age.ge(&stale_days)? {
        let short_date = o.str_t.call1((&ranked_at,))?.get_item(
            py.import("builtins")?
                .getattr("slice")?
                .call1((py.None(), 10))?,
        )?;
        warnings.append(format!(
            "Ranking snapshot is {} days old (last run {}). The daily worker may not be running — refresh with `python src/buffett_rank_worker.py` before acting on this list.",
            o.fmt(&age)?,
            o.fmt(&short_date)?
        ))?;
    }
    let positions = built.get_item("positions")?;
    let wanted = ranking.getattr("top_n")?;
    let amounts = PyList::empty(py);
    for p in positions.try_iter()? {
        amounts.append(p?.get_item("amount")?)?;
    }
    let allocated = py.import("builtins")?.getattr("sum")?.call1((amounts,))?;
    let short = positions.is_truthy()? && positions.len()?.into_pyobject(py)?.lt(&wanted)?;
    if short {
        warnings.append(format!(
            "The ranking produced only {} of the {} names this rule calls for, so ${} of ${} is unallocated. Weights are deliberately left at the rule's {} rather than spread over a shorter list. Check that the last ranking run covered the whole universe.",
            positions.len()?,
            o.fmt(&wanted)?,
            o.fmt_spec(&capital.sub(&allocated)?, ",.0f")?,
            o.fmt_spec(capital, ",.0f")?,
            o.fmt_spec(&o.fl(1.0).div(&wanted)?, ".1%")?
        ))?;
    }
    let sleeve = PyDict::new(py);
    sleeve.set_item("key", "ranking")?;
    sleeve.set_item(
        "label",
        m.getattr("_sleeve_label")?.call1((&ranking, &applied))?,
    )?;
    sleeve.set_item("weight", 1.0)?;
    sleeve.set_item("amount", capital)?;
    sleeve.set_item("positions_requested", &wanted)?;
    sleeve.set_item("positions_filled", positions.len()?)?;
    sleeve.set_item("amount_allocated", &allocated)?;
    sleeve.set_item("positions", &positions)?;
    sleeve.set_item("run_id", run.call_method1("get", ("run_id",))?)?;
    sleeve.set_item("ranked_at", &ranked_at)?;
    sleeve.set_item(
        "price_source",
        built.call_method1("get", ("price_source", "snapshot"))?,
    )?;

    let out = PyDict::new(py);
    out.set_item("strategy_id", strategy.getattr("id")?)?;
    out.set_item("name", strategy.getattr("name")?)?;
    out.set_item("capital", capital)?;
    let as_of = if today.is_truthy()? {
        today.clone()
    } else {
        m.getattr("get_est_today")?.call0()?
    };
    out.set_item("as_of", o.str_t.call1((as_of,))?)?;
    out.set_item("ranking_age_days", &age)?;
    out.set_item("ranking_is_stale", !age.is_none() && age.ge(&stale_days)?)?;
    out.set_item("is_short", short)?;
    out.set_item("ai_weight", &applied)?;
    out.set_item("sleeves", PyList::new(py, [sleeve])?)?;
    out.set_item("warnings", warnings)?;
    Ok(out)
}

/// `strategy_payload`: a strategy's serialisable description.
#[pyfunction]
pub fn st_strategy_payload<'py>(
    py: Python<'py>,
    strategy: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = st(py)?;
    let ranking = strategy.getattr("ranking")?;
    let out = PyDict::new(py);
    for k in ["id", "name", "summary", "sleeves", "backtest"] {
        out.set_item(k, strategy.getattr(k)?)?;
    }
    out.set_item(
        "risks",
        py.import("builtins")?
            .getattr("list")?
            .call1((strategy.getattr("risks")?,))?,
    )?;
    out.set_item(
        "is_default",
        strategy
            .getattr("id")?
            .eq(m.getattr("DEFAULT_STRATEGY_ID")?)?,
    )?;
    let r = PyDict::new(py);
    for k in [
        "quality_weight",
        "top_n",
        "max_per_sector",
        "sector_digits",
        "min_market_cap",
        "ai_weight",
    ] {
        r.set_item(k, ranking.getattr(k)?)?;
    }
    r.set_item("ai_note", m.getattr("AI_BACKTEST_CAVEAT")?)?;
    r.set_item("rebalance", "Each January")?;
    out.set_item("ranking", r)?;
    Ok(out)
}
