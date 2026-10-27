//! `db_utils`: the global screener cache (`screener_cache`).
//!
//! Ported line for line from `src/db_utils.py` (frozen in
//! `tests/reference/db_utils.py`), as `db_portfolio.rs` is. The order in
//! which the original evaluates things decides which failures its handlers
//! catch, so it is kept: values built before a `try` raise past it, values
//! built inside are caught. Python's own `sum()` (compensated for floats
//! since 3.12), `len()`, `json` and `list.sort(key=...)` are called rather
//! than imitated.

use pyo3::prelude::*;
use pyo3::types::{PyCFunction, PyDict, PyList, PyString, PyTuple};

use crate::pyconn::PyConn;
use crate::pyhelp::{es, get, get_or, is_sqlite, log, now_iso, pystr, s};

const UPSERT_SQL: &str = "
    INSERT INTO screener_cache (
        symbol, name, price, intrinsic_value, margin_of_safety,
        pe_ratio, market_cap, sector,
        ai_moat, ai_financial_strength, ai_predictability, ai_growth, ai_summary,
        ai_sentiment, ai_catalysts,
        last_fiscal_year_end, most_recent_quarter, universe, updated_at, valuation_details
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol, universe) DO UPDATE SET
        name=excluded.name,
        price=excluded.price,
        intrinsic_value=excluded.intrinsic_value,
        margin_of_safety=excluded.margin_of_safety,
        pe_ratio=excluded.pe_ratio,
        market_cap=excluded.market_cap,
        sector=excluded.sector,
        ai_moat=COALESCE(excluded.ai_moat, screener_cache.ai_moat),
        ai_financial_strength=COALESCE(excluded.ai_financial_strength, screener_cache.ai_financial_strength),
        ai_predictability=COALESCE(excluded.ai_predictability, screener_cache.ai_predictability),
        ai_growth=COALESCE(excluded.ai_growth, screener_cache.ai_growth),
        ai_summary=COALESCE(excluded.ai_summary, screener_cache.ai_summary),
        ai_sentiment=COALESCE(excluded.ai_sentiment, screener_cache.ai_sentiment),
        ai_catalysts=COALESCE(excluded.ai_catalysts, screener_cache.ai_catalysts),
        last_fiscal_year_end=excluded.last_fiscal_year_end,
        most_recent_quarter=excluded.most_recent_quarter,
        updated_at=excluded.updated_at,
        valuation_details=COALESCE(excluded.valuation_details, screener_cache.valuation_details)
    ";

const REFRESH_SQL: &str = "
    UPDATE screener_cache SET
        name = COALESCE(?, name),
        price = ?,
        intrinsic_value = ?,
        margin_of_safety = ?,
        pe_ratio = ?,
        market_cap = ?,
        sector = COALESCE(?, sector),
        ai_moat = COALESCE(?, ai_moat),
        ai_financial_strength = COALESCE(?, ai_financial_strength),
        ai_predictability = COALESCE(?, ai_predictability),
        ai_growth = COALESCE(?, ai_growth),
        ai_summary = COALESCE(?, ai_summary),
        ai_sentiment = COALESCE(?, ai_sentiment),
        ai_catalysts = COALESCE(?, ai_catalysts),
        last_fiscal_year_end = COALESCE(?, last_fiscal_year_end),
        most_recent_quarter = COALESCE(?, most_recent_quarter),
        updated_at = ?,
        valuation_details = COALESCE(?, valuation_details)
    WHERE symbol = ?
    ";

const AI_UPDATE_SQL: &str = "
        UPDATE screener_cache SET
            ai_moat = ?,
            ai_financial_strength = ?,
            ai_predictability = ?,
            ai_growth = ?,
            ai_moat_analysis = ?,
            ai_financial_strength_analysis = ?,
            ai_predictability_analysis = ?,
            ai_growth_perspective_analysis = ?,
            ai_summary = ?,
            ai_sentiment = ?,
            ai_catalysts = ?,
            intrinsic_value = ?,
            margin_of_safety = ?,
            valuation_details = ?,
            name = COALESCE(?, name),
            price = COALESCE(?, price),
            pe_ratio = COALESCE(?, pe_ratio),
            market_cap = COALESCE(?, market_cap),
            sector = COALESCE(?, sector),
            last_fiscal_year_end = COALESCE(?, last_fiscal_year_end),
            most_recent_quarter = COALESCE(?, most_recent_quarter),
            updated_at = ?
        WHERE symbol = ?
        ";

const AI_INSERT_SQL: &str = "
                INSERT INTO screener_cache (
                    symbol, universe, ai_moat, ai_financial_strength,
                    ai_predictability, ai_growth,
                    ai_moat_analysis, ai_financial_strength_analysis,
                    ai_predictability_analysis, ai_growth_perspective_analysis,
                    ai_summary,
                    ai_sentiment, ai_catalysts, intrinsic_value, margin_of_safety, valuation_details,
                    name, price, pe_ratio, market_cap, sector,
                    last_fiscal_year_end, most_recent_quarter, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ";

const IV_UPDATE_SQL: &str = "
        UPDATE screener_cache SET
            intrinsic_value = ?,
            margin_of_safety = ?,
            last_fiscal_year_end = COALESCE(?, last_fiscal_year_end),
            most_recent_quarter = COALESCE(?, most_recent_quarter),
            name = COALESCE(?, name),
            price = COALESCE(?, price),
            pe_ratio = COALESCE(?, pe_ratio),
            market_cap = COALESCE(?, market_cap),
            sector = COALESCE(?, sector),
            updated_at = ?,
            valuation_details = COALESCE(?, valuation_details)
        WHERE symbol = ?
        ";

const IV_INSERT_SQL: &str = "
                INSERT INTO screener_cache (
                    symbol, universe, intrinsic_value, margin_of_safety,
                    last_fiscal_year_end, most_recent_quarter,
                    name, price, pe_ratio, market_cap, sector, updated_at, valuation_details
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ";

const DISTINCT_SQL: &str = "
            SELECT * FROM screener_cache t1
            WHERE updated_at = (
                SELECT max(updated_at)
                FROM screener_cache t2
                WHERE t2.symbol = t1.symbol
            )
            ";

fn db_utils(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("db_utils")
}

fn close_quietly(conn: &Bound<'_, PyAny>) {
    let _ = conn.call_method0("close");
}

/// `json.dumps(v)` when `v` is a list, else `v`.
fn dumps_if_list<'py>(py: Python<'py>, v: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if v.is_instance_of::<PyList>() {
        py.import("json")?.call_method1("dumps", (v,))
    } else {
        Ok(v)
    }
}

/// `json.dumps(v, cls=NpEncoder)` when `v` is a dict, else `v`.
fn dumps_if_dict<'py>(py: Python<'py>, v: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if v.is_instance_of::<PyDict>() {
        let kwargs = PyDict::new(py);
        kwargs.set_item("cls", db_utils(py)?.getattr("NpEncoder")?)?;
        py.import("json")?.call_method("dumps", (v,), Some(&kwargs))
    } else {
        Ok(v)
    }
}

/// `sum(valid) / len(valid) if valid else None` over the four AI scores.
fn ai_score<'py>(py: Python<'py>, row: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let valid = PyList::empty(py);
    for key in [
        "ai_moat",
        "ai_financial_strength",
        "ai_predictability",
        "ai_growth",
    ] {
        let v = get(row, key)?;
        if !v.is_none() {
            valid.append(v)?;
        }
    }
    if valid.is_empty() {
        return Ok(py.None().into_bound(py));
    }
    let total = py.import("builtins")?.getattr("sum")?.call1((&valid,))?;
    total.div(valid.len())
}

/// `json.loads` a string `ai_catalysts` in place; a failure leaves it.
fn parse_catalysts(
    py: Python<'_>,
    row: &Bound<'_, PyAny>,
    only_value_errors: bool,
) -> PyResult<()> {
    let value = get(row, "ai_catalysts")?;
    if value.is_truthy()? && value.is_instance_of::<PyString>() {
        match py.import("json")?.call_method1("loads", (&value,)) {
            Ok(parsed) => row.set_item("ai_catalysts", parsed)?,
            Err(e) => {
                let swallowed = if only_value_errors {
                    e.is_instance_of::<pyo3::exceptions::PyValueError>(py)
                        || e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                } else {
                    e.is_instance_of::<pyo3::exceptions::PyException>(py)
                };
                if !swallowed {
                    return Err(e);
                }
            }
        }
    }
    Ok(())
}

/// `has_ai_review`: a summary longer than twenty characters.
fn set_has_ai_review(py: Python<'_>, row: &Bound<'_, PyAny>) -> PyResult<()> {
    if row.contains("ai_summary")? {
        let summary = row.get_item("ai_summary")?;
        let has = !summary.is_none() && summary.len()? > 20;
        let _ = py;
        row.set_item("has_ai_review", has)?;
    }
    Ok(())
}

fn rows_as_dicts<'py>(
    py: Python<'py>,
    rows: &crate::pyconn::SqlRows,
) -> PyResult<(Bound<'py, PyList>, Vec<Bound<'py, PyAny>>)> {
    let mut cols = Vec::new();
    for d in rows.description(py).try_iter()? {
        cols.push(d?.get_item(0)?);
    }
    let out = PyList::empty(py);
    for row in rows.all(py) {
        let row = row.bind(py);
        let d = PyDict::new(py);
        for (c, v) in cols.iter().zip(row.try_iter()?) {
            d.set_item(c, v?)?;
        }
        out.append(d)?;
    }
    Ok((out, cols))
}

/// `upsert_screener_results`
#[pyfunction]
pub fn db_upsert_screener_results<'py>(
    py: Python<'py>,
    results: &Bound<'py, PyAny>,
    universe: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if !results.is_truthy()? {
        return Ok(());
    }
    let db_conn = db_utils(py)?.getattr("open_screener_db_conn")?.call0()?;
    if db_conn.is_none() {
        log(
            py,
            "error",
            "upsert_screener_results: could not open global screener DB",
        )?;
        return Ok(());
    }
    let built = (|| -> PyResult<Bound<'py, PyList>> {
        let now = pystr(py, &now_iso(py)?);
        let data = PyList::empty(py);
        for r in results.try_iter()? {
            let r = r?;
            let mut cells = Vec::with_capacity(20);
            for key in [
                "symbol",
                "name",
                "price",
                "intrinsic_value",
                "margin_of_safety",
                "pe_ratio",
                "market_cap",
                "sector",
                "ai_moat",
                "ai_financial_strength",
                "ai_predictability",
                "ai_growth",
                "ai_summary",
                "ai_sentiment",
            ] {
                cells.push(get(&r, key)?);
            }
            cells.push(dumps_if_list(py, get(&r, "ai_catalysts")?)?);
            cells.push(get(&r, "last_fiscal_year_end")?);
            cells.push(get(&r, "most_recent_quarter")?);
            cells.push(universe.clone());
            cells.push(now.clone());
            cells.push(dumps_if_dict(py, get(&r, "valuation_details")?)?);
            data.append(PyTuple::new(py, cells)?)?;
        }
        Ok(data)
    })();
    // Built before the `try`: a failure here raises without closing, as the original.
    let data = built?;
    let outcome = (|| -> PyResult<()> {
        let conn = PyConn::new(&db_conn)?;
        conn.executemany(&pystr(py, UPSERT_SQL), data.as_any())?;
        conn.commit()?;
        log(
            py,
            "info",
            format!("Upserted {} screener results to DB cache.", results.len()?),
        )?;
        Ok(())
    })();
    let handled = match outcome {
        Ok(()) => Ok(()),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!("Error upserting screener results: {}", es(py, &e)),
            )?;
            db_conn.call_method0("rollback").map(|_| ())
        }
        Err(e) => Err(e),
    };
    close_quietly(&db_conn);
    handled
}

/// `refresh_screener_rows_by_symbol`: UPDATE-only, by symbol, any universe.
#[pyfunction]
pub fn db_refresh_screener_rows<'py>(py: Python<'py>, results: &Bound<'py, PyAny>) -> PyResult<()> {
    if !results.is_truthy()? {
        return Ok(());
    }
    let db_conn = db_utils(py)?.getattr("open_screener_db_conn")?.call0()?;
    if db_conn.is_none() {
        log(
            py,
            "error",
            "refresh_screener_rows_by_symbol: could not open global screener DB",
        )?;
        return Ok(());
    }
    let built = (|| -> PyResult<Bound<'py, PyList>> {
        let now = pystr(py, &now_iso(py)?);
        let rows = PyList::empty(py);
        for r in results.try_iter()? {
            let r = r?;
            let sym = get(&r, "symbol")?;
            if !sym.is_truthy()? {
                continue;
            }
            let mut cells = Vec::with_capacity(19);
            for key in [
                "name",
                "price",
                "intrinsic_value",
                "margin_of_safety",
                "pe_ratio",
                "market_cap",
                "sector",
                "ai_moat",
                "ai_financial_strength",
                "ai_predictability",
                "ai_growth",
                "ai_summary",
                "ai_sentiment",
            ] {
                cells.push(get(&r, key)?);
            }
            cells.push(dumps_if_list(py, get(&r, "ai_catalysts")?)?);
            cells.push(get(&r, "last_fiscal_year_end")?);
            cells.push(get(&r, "most_recent_quarter")?);
            cells.push(now.clone());
            cells.push(dumps_if_dict(py, get(&r, "valuation_details")?)?);
            cells.push(sym.call_method0("upper")?);
            rows.append(PyTuple::new(py, cells)?)?;
        }
        Ok(rows)
    })();
    let rows = built?;
    let outcome = (|| -> PyResult<()> {
        let conn = PyConn::new(&db_conn)?;
        let done = conn.executemany(&pystr(py, REFRESH_SQL), rows.as_any())?;
        conn.commit()?;
        log(
            py,
            "info",
            format!(
                "Refreshed {} screener rows by symbol (UPDATE-only).",
                done.rowcount()
            ),
        )?;
        Ok(())
    })();
    let handled = match outcome {
        Ok(()) => Ok(()),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!("Error in refresh_screener_rows_by_symbol: {}", es(py, &e)),
            )?;
            db_conn.call_method0("rollback").map(|_| ())
        }
        Err(e) => Err(e),
    };
    close_quietly(&db_conn);
    handled
}

/// `get_cached_screener_results`: {SYMBOL: row}, rows with AI data last so
/// they win.
#[pyfunction]
pub fn db_get_cached_screener_results<'py>(
    py: Python<'py>,
    symbols: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    if !symbols.is_truthy()? {
        return Ok(PyDict::new(py));
    }
    let conn = db_utils(py)?.getattr("open_screener_db_conn")?.call0()?;
    if conn.is_none() {
        return Ok(PyDict::new(py));
    }
    let outcome = (|| -> PyResult<Bound<'py, PyDict>> {
        let pc = PyConn::new(&conn)?;
        let marks = vec!["?"; symbols.len()?].join(",");
        let sql = format!(
            "SELECT * FROM screener_cache WHERE symbol IN ({marks}) ORDER BY symbol, (ai_summary IS NOT NULL) ASC, updated_at ASC"
        );
        let rows = pc.execute(&pystr(py, &sql), symbols)?;
        let (dicts, _) = rows_as_dicts(py, &rows)?;
        let mapping = PyDict::new(py);
        for row in dicts.iter() {
            row.set_item("ai_score", ai_score(py, &row)?)?;
            parse_catalysts(py, &row, false)?;
            mapping.set_item(row.get_item("symbol")?.call_method0("upper")?, &row)?;
        }
        Ok(mapping)
    })();
    let handled = match outcome {
        Ok(m) => Ok(m),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!("Error fetching cached screener results: {}", es(py, &e)),
            )?;
            Ok(PyDict::new(py))
        }
        Err(e) => Err(e),
    };
    close_quietly(&conn);
    handled
}

/// `get_screener_results_by_universe`
#[pyfunction]
pub fn db_get_screener_results_by_universe<'py>(
    py: Python<'py>,
    universe: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let conn = db_utils(py)?.getattr("open_screener_db_conn")?.call0()?;
    if conn.is_none() {
        return Ok(PyList::empty(py));
    }
    let outcome = (|| -> PyResult<Bound<'py, PyList>> {
        let pc = PyConn::new(&conn)?;
        let rows = pc.run(
            "SELECT * FROM screener_cache WHERE universe = ? ORDER BY margin_of_safety DESC",
            vec![universe.clone()],
        )?;
        let (dicts, _) = rows_as_dicts(py, &rows)?;
        for row in dicts.iter() {
            set_has_ai_review(py, &row)?;
            row.set_item("ai_score", ai_score(py, &row)?)?;
        }
        Ok(dicts)
    })();
    let handled = match outcome {
        Ok(list) => Ok(list),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!(
                    "Error fetching screener results for universe {}: {}",
                    s(universe)?,
                    es(py, &e)
                ),
            )?;
            Ok(PyList::empty(py))
        }
        Err(e) => Err(e),
    };
    close_quietly(&conn);
    handled
}

/// `get_all_distinct_screener_results`: the newest row per symbol, sorted by
/// margin of safety (missing last).
#[pyfunction]
pub fn db_get_all_distinct_screener_results<'py>(py: Python<'py>) -> PyResult<Bound<'py, PyList>> {
    let conn = db_utils(py)?.getattr("open_screener_db_conn")?.call0()?;
    if conn.is_none() {
        return Ok(PyList::empty(py));
    }
    let outcome = (|| -> PyResult<Bound<'py, PyDict>> {
        let pc = PyConn::new(&conn)?;
        let rows = pc.run(DISTINCT_SQL, vec![])?;
        let (dicts, cols) = rows_as_dicts(py, &rows)?;
        let at = PyList::new(py, &cols)?.call_method1("index", ("symbol",))?;
        let by_symbol = PyDict::new(py);
        for (row, raw) in dicts.iter().zip(rows.all(py)) {
            by_symbol.set_item(raw.bind(py).get_item(&at)?, row)?;
        }
        Ok(by_symbol)
    })();
    let results = match outcome {
        Ok(d) => Ok(d),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!("Error fetching distinct screener results: {}", es(py, &e)),
            )?;
            Ok(PyDict::new(py))
        }
        Err(e) => Err(e),
    };
    close_quietly(&conn);
    let results = results?;

    let final_list = PyList::empty(py);
    for row in results.values().iter() {
        set_has_ai_review(py, &row)?;
        parse_catalysts(py, &row, true)?;
        row.set_item("ai_score", ai_score(py, &row)?)?;
        final_list.append(row)?;
    }
    let key = PyCFunction::new_closure(py, None, None, |args, _kwargs| -> PyResult<Py<PyAny>> {
        let py = args.py();
        let x = args.get_item(0)?;
        let mos = get(&x, "margin_of_safety")?;
        Ok(PyTuple::new(
            py,
            [
                (!mos.is_none()).into_pyobject(py)?.to_owned().into_any(),
                mos,
            ],
        )?
        .into_any()
        .unbind())
    })?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("key", key)?;
    kwargs.set_item("reverse", true)?;
    final_list.call_method("sort", (), Some(&kwargs))?;
    Ok(final_list)
}

/// `_is_synthetic_symbol`
#[pyfunction]
pub fn db_is_synthetic_symbol(symbol: &Bound<'_, PyAny>) -> PyResult<bool> {
    let s: String = symbol
        .call_method0("upper")?
        .call_method0("strip")?
        .extract()?;
    Ok(
        ["PORTFOLIO-WIDE", "CASH-AGGREGATE", "TOTAL", "PORTFOLIO"].contains(&s.as_str())
            || s.starts_with("PORTFOLIO-")
            || s.starts_with("CASH-"),
    )
}

fn info_get<'py>(
    py: Python<'py>,
    info: &Bound<'py, PyAny>,
    key: &str,
) -> PyResult<Bound<'py, PyAny>> {
    if info.is_truthy()? {
        get(info, key)
    } else {
        Ok(py.None().into_bound(py))
    }
}

fn str_or_none<'py>(py: Python<'py>, v: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    if v.is_instance_of::<PyString>() {
        v
    } else {
        py.None().into_bound(py)
    }
}

/// `update_ai_review_in_cache`: every universe's row for the symbol, or a new
/// row in `universe` when there is none.
#[pyfunction]
pub fn db_update_ai_review_in_cache<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    ai_data: &Bound<'py, PyAny>,
    info: &Bound<'py, PyAny>,
    universe: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let m = db_utils(py)?;
    if m.getattr("_is_synthetic_symbol")?
        .call1((symbol,))?
        .is_truthy()?
    {
        log(
            py,
            "debug",
            format!(
                "update_ai_review_in_cache: skipping synthetic symbol '{}'",
                s(symbol)?
            ),
        )?;
        return Ok(());
    }
    let conn = m.getattr("open_screener_db_conn")?.call0()?;
    if conn.is_none() {
        log(
            py,
            "error",
            format!(
                "update_ai_review_in_cache: could not open global screener DB for {}",
                s(symbol)?
            ),
        )?;
        return Ok(());
    }
    let r = do_ai_update(py, &conn, symbol, ai_data, info, universe);
    close_quietly(&conn);
    r.map(|_| ())
}

fn do_ai_update<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    sym: &Bound<'py, PyAny>,
    data: &Bound<'py, PyAny>,
    inf: &Bound<'py, PyAny>,
    univ: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let json = py.import("json")?;
    let now = pystr(py, &now_iso(py)?);
    let scorecard = get_or(data, "scorecard", &PyDict::new(py).into_any())?;
    let analysis = get_or(data, "analysis", &PyDict::new(py).into_any())?;
    let analysis = if analysis.is_truthy()? {
        analysis
    } else {
        PyDict::new(py).into_any()
    };
    let short_name = info_get(py, inf, "shortName")?;
    let price = info_get(py, inf, "currentPrice")?;
    let pe = info_get(py, inf, "trailingPE")?;
    let mcap = info_get(py, inf, "marketCap")?;
    let sector = info_get(py, inf, "sector")?;
    let fy_end = info_get(py, inf, "lastFiscalYearEnd")?;
    let mrq = info_get(py, inf, "mostRecentQuarter")?;
    let sentiment = get(data, "sentiment")?;
    let catalysts = if get(data, "catalysts")?.is_truthy()? {
        json.call_method1(
            "dumps",
            (get_or(data, "catalysts", &PyList::empty(py).into_any())?,),
        )?
    } else {
        py.None().into_bound(py)
    };
    let iv = get(data, "intrinsic_value")?;
    let mos = get(data, "margin_of_safety")?;
    let details = if get(data, "valuation_details")?.is_truthy()? {
        json.call_method1(
            "dumps",
            (get_or(
                data,
                "valuation_details",
                &PyDict::new(py).into_any(),
            )?,),
        )?
    } else {
        py.None().into_bound(py)
    };
    let moat_text = str_or_none(py, get(&analysis, "moat")?);
    let fin_text = str_or_none(py, get(&analysis, "financial_strength")?);
    let pred_text = str_or_none(py, get(&analysis, "predictability")?);
    let growth_text = str_or_none(py, get(&analysis, "growth_perspective")?);

    let outcome = (|| -> PyResult<bool> {
        let pc = PyConn::new(conn)?;
        let summary = get(data, "summary")?;
        let updated = pc.run(
            AI_UPDATE_SQL,
            vec![
                get(&scorecard, "moat")?,
                get(&scorecard, "financial_strength")?,
                get(&scorecard, "predictability")?,
                get(&scorecard, "growth")?,
                moat_text.clone(),
                fin_text.clone(),
                pred_text.clone(),
                growth_text.clone(),
                summary.clone(),
                sentiment.clone(),
                catalysts.clone(),
                iv.clone(),
                mos.clone(),
                details.clone(),
                short_name.clone(),
                price.clone(),
                pe.clone(),
                mcap.clone(),
                sector.clone(),
                fy_end.clone(),
                mrq.clone(),
                now.clone(),
                sym.call_method0("upper")?,
            ],
        )?;
        if updated.rowcount() == 0 {
            pc.run(
                AI_INSERT_SQL,
                vec![
                    sym.call_method0("upper")?,
                    univ.clone(),
                    get(&scorecard, "moat")?,
                    get(&scorecard, "financial_strength")?,
                    get(&scorecard, "predictability")?,
                    get(&scorecard, "growth")?,
                    moat_text.clone(),
                    fin_text.clone(),
                    pred_text.clone(),
                    growth_text.clone(),
                    get(data, "summary")?,
                    sentiment.clone(),
                    catalysts.clone(),
                    iv.clone(),
                    mos.clone(),
                    details.clone(),
                    short_name.clone(),
                    price.clone(),
                    pe.clone(),
                    mcap.clone(),
                    sector.clone(),
                    fy_end.clone(),
                    mrq.clone(),
                    now.clone(),
                ],
            )?;
        }
        pc.commit()?;
        Ok(true)
    })();
    match outcome {
        Ok(v) => Ok(v),
        Err(e) if is_sqlite(py, &e, "Error")? => {
            log(
                py,
                "error",
                format!("Error in do_update for {}: {}", s(sym)?, es(py, &e)),
            )?;
            conn.call_method0("rollback")?;
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

/// `update_intrinsic_value_in_cache`
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn db_update_intrinsic_value_in_cache<'py>(
    py: Python<'py>,
    symbol: &Bound<'py, PyAny>,
    intrinsic_value: &Bound<'py, PyAny>,
    margin_of_safety: &Bound<'py, PyAny>,
    last_fiscal_year_end: &Bound<'py, PyAny>,
    most_recent_quarter: &Bound<'py, PyAny>,
    info: &Bound<'py, PyAny>,
    universe: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let m = db_utils(py)?;
    if m.getattr("_is_synthetic_symbol")?
        .call1((symbol,))?
        .is_truthy()?
    {
        log(
            py,
            "debug",
            format!(
                "update_intrinsic_value_in_cache: skipping synthetic symbol '{}'",
                s(symbol)?
            ),
        )?;
        return Ok(());
    }
    let conn = m.getattr("open_screener_db_conn")?.call0()?;
    if conn.is_none() {
        log(
            py,
            "error",
            format!(
                "update_intrinsic_value_in_cache: could not open global screener DB for {}",
                s(symbol)?
            ),
        )?;
        return Ok(());
    }
    let r = (|| -> PyResult<bool> {
        let now = pystr(py, &now_iso(py)?);
        let name = info_get(py, info, "shortName")?;
        let price = info_get(py, info, "currentPrice")?;
        let pe = info_get(py, info, "trailingPE")?;
        let mcap = info_get(py, info, "marketCap")?;
        let sector = info_get(py, info, "sector")?;
        let details = info_get(py, info, "valuation_details")?;
        let outcome = (|| -> PyResult<bool> {
            let pc = PyConn::new(&conn)?;
            let updated = pc.run(
                IV_UPDATE_SQL,
                vec![
                    intrinsic_value.clone(),
                    margin_of_safety.clone(),
                    last_fiscal_year_end.clone(),
                    most_recent_quarter.clone(),
                    name.clone(),
                    price.clone(),
                    pe.clone(),
                    mcap.clone(),
                    sector.clone(),
                    now.clone(),
                    dumps_if_dict(py, details.clone())?,
                    symbol.call_method0("upper")?,
                ],
            )?;
            if updated.rowcount() == 0 {
                pc.run(
                    IV_INSERT_SQL,
                    vec![
                        symbol.call_method0("upper")?,
                        universe.clone(),
                        intrinsic_value.clone(),
                        margin_of_safety.clone(),
                        last_fiscal_year_end.clone(),
                        most_recent_quarter.clone(),
                        name.clone(),
                        price.clone(),
                        pe.clone(),
                        mcap.clone(),
                        sector.clone(),
                        now.clone(),
                        dumps_if_dict(py, details.clone())?,
                    ],
                )?;
            }
            pc.commit()?;
            Ok(true)
        })();
        match outcome {
            Ok(v) => Ok(v),
            Err(e) if is_sqlite(py, &e, "Error")? => {
                log(
                    py,
                    "error",
                    format!("Error in do_iv_update for {}: {}", s(symbol)?, es(py, &e)),
                )?;
                conn.call_method0("rollback")?;
                Ok(false)
            }
            Err(e) => Err(e),
        }
    })();
    close_quietly(&conn);
    r.map(|_| ())
}
