//! `buffett_store.BuffettRankStore`: the append-only history of ranking runs.
//!
//! Ported line for line from `src/buffett_store.py` (frozen in
//! `tests/reference/buffett_store.py`). Methods take the store object as
//! `slf`; its `_connect`, `_write_lock` and sibling methods are reached
//! through it. The store's connections use `sqlite3.Row`, but its rows never
//! leave the store — they are read as `dict(row)`, `row["run_id"]` or
//! `row[0]` — so statements run in Rust (`PyConn::new_internal`) and
//! `pyconn::row_dict` reproduces the Row lookups.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::mdb::{with1, with2};
use crate::pyconn::{PyConn, row_dict};
use crate::pyhelp::{es, isna, log, now_iso, pystr, s};

const SCORE_COLUMNS: [&str; 20] = [
    "cik",
    "name",
    "model",
    "rank",
    "composite_score",
    "quality_score",
    "value_score",
    "confidence",
    "coverage",
    "returns_on_capital",
    "financial_strength",
    "predictability",
    "growth",
    "capital_allocation",
    "price",
    "market_cap",
    "earnings_yield",
    "fcf_yield",
    "period_count",
    "latest_period",
];

fn connect<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    slf.call_method0("_connect")
}

fn locked<'py, T>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let lock = slf.getattr("_write_lock")?;
    Ok(with2(py, &lock, || connect(slf), body)?.expect("no CM swallows"))
}

fn unlocked<'py, T>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let conn = connect(slf)?;
    Ok(with1(py, &conn, body)?.expect("no CM swallows"))
}

/// `_escape_like`: neutralise LIKE wildcards (backslash first).
#[pyfunction]
pub fn rs_escape_like<'py>(term: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    term.call_method1("replace", ("\\", "\\\\"))?
        .call_method1("replace", ("%", "\\%"))?
        .call_method1("replace", ("_", "\\_"))
}

/// `_coerce`: NumPy/pandas scalars and NaN into SQLite-friendly values.
#[pyfunction]
pub fn rs_coerce<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::types::{PyFloat, PyInt};
    let none = py.None().into_bound(py);
    if value.is_none() {
        return Ok(none);
    }
    if value.is_instance_of::<PyFloat>() && isna(py, value)? {
        return Ok(none);
    }
    match isna(py, value) {
        Ok(true) => return Ok(none),
        Ok(false) => {}
        Err(e)
            if e.is_instance_of::<pyo3::exceptions::PyTypeError>(py)
                || e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {}
        Err(e) => return Err(e),
    }
    if value.hasattr("item")? {
        match value.call_method0("item") {
            Ok(v) => return Ok(v),
            Err(e)
                if e.is_instance_of::<pyo3::exceptions::PyAttributeError>(py)
                    || e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => {}
            Err(e) => return Err(e),
        }
    }
    if value.is_instance_of::<PyInt>()
        || value.is_instance_of::<PyFloat>()
        || value.is_instance_of::<PyString>()
    {
        return Ok(value.clone());
    }
    Ok(value.str()?.into_any())
}

/// `BuffettRankStore._init_db`
#[pyfunction]
pub fn rs_init_db<'py>(py: Python<'py>, slf: &Bound<'py, PyAny>) -> PyResult<()> {
    locked(py, slf, |conn| {
        let pc = PyConn::new_internal(conn)?;
        for sql in [
            "
                CREATE TABLE IF NOT EXISTS rank_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    universe_size INTEGER,
                    ranked_count INTEGER,
                    excluded_count INTEGER,
                    parameters TEXT
                )
            ",
            "
                CREATE TABLE IF NOT EXISTS rank_scores (
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    cik TEXT,
                    name TEXT,
                    model TEXT,
                    rank INTEGER,
                    composite_score REAL,
                    quality_score REAL,
                    value_score REAL,
                    confidence REAL,
                    coverage REAL,
                    returns_on_capital REAL,
                    financial_strength REAL,
                    predictability REAL,
                    growth REAL,
                    capital_allocation REAL,
                    price REAL,
                    market_cap REAL,
                    earnings_yield REAL,
                    fcf_yield REAL,
                    period_count INTEGER,
                    latest_period TEXT,
                    PRIMARY KEY (run_id, symbol)
                )
            ",
            "
                CREATE TABLE IF NOT EXISTS rank_exclusions (
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    cik TEXT,
                    name TEXT,
                    model TEXT,
                    reasons TEXT,
                    period_count INTEGER,
                    coverage REAL,
                    PRIMARY KEY (run_id, symbol)
                )
            ",
            "CREATE INDEX IF NOT EXISTS idx_scores_run_rank ON rank_scores (run_id, rank)",
        ] {
            pc.run(sql, vec![])?;
        }
        pc.commit()
    })
}

/// `BuffettRankStore.start_run`: the new run's id.
#[pyfunction]
pub fn rs_start_run<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    universe_size: &Bound<'py, PyAny>,
    parameters: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    locked(py, slf, |conn| {
        let pc = PyConn::new_internal(conn)?;
        let params = py.import("json")?.call_method1("dumps", (parameters,))?;
        let rows = pc.run(
            "INSERT INTO rank_runs (started_at, universe_size, parameters) VALUES (?, ?, ?)",
            vec![pystr(py, &now_iso(py)?), universe_size.clone(), params],
        )?;
        pc.commit()?;
        py.import("builtins")?
            .getattr("int")?
            .call1((rows.lastrowid(py),))
    })
}

/// `BuffettRankStore.finish_run`, then the retention rule.
#[pyfunction]
pub fn rs_finish_run<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    ranked: &Bound<'py, PyAny>,
    excluded: &Bound<'py, PyAny>,
) -> PyResult<()> {
    locked(py, slf, |conn| {
        let pc = PyConn::new_internal(conn)?;
        pc.run(
            "UPDATE rank_runs SET finished_at = ?, ranked_count = ?, excluded_count = ?\n                   WHERE run_id = ?",
            vec![pystr(py, &now_iso(py)?), ranked.clone(), excluded.clone(), run_id.clone()],
        )?;
        pc.commit()
    })?;
    slf.call_method0("prune_runs")?;
    Ok(())
}

/// `BuffettRankStore.prune_runs`: all but the newest `keep` finished runs,
/// and unfinished runs older than `stale_hours`. How many went.
#[pyfunction]
pub fn rs_prune_runs<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    keep: &Bound<'py, PyAny>,
    stale_hours: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    let dt = py.import("datetime")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("hours", stale_hours)?;
    let cutoff = dt
        .getattr("datetime")?
        .call_method0("now")?
        .sub(dt.getattr("timedelta")?.call((), Some(&kwargs))?)?
        .call_method0("isoformat")?;
    let doomed = locked(
        py,
        slf,
        |conn| -> PyResult<Option<Vec<Bound<'py, PyAny>>>> {
            let pc = PyConn::new_internal(conn)?;
            let rows = pc.run(
                "SELECT run_id FROM rank_runs
                       WHERE (
                           finished_at IS NOT NULL AND run_id NOT IN (
                               SELECT run_id FROM rank_runs WHERE finished_at IS NOT NULL
                               ORDER BY run_id DESC LIMIT ?
                           )
                       ) OR (
                           finished_at IS NULL AND started_at < ?
                       )",
                vec![keep.clone(), cutoff.clone()],
            )?;
            let doomed: Vec<Bound<'py, PyAny>> = rows
                .all(py)
                .iter()
                .map(|r| r.bind(py).get_item(0))
                .collect::<PyResult<_>>()?;
            if doomed.is_empty() {
                return Ok(None);
            }
            let marks = vec!["?"; doomed.len()].join(",");
            let ids = PyList::new(py, &doomed)?;
            for table in ["rank_scores", "rank_exclusions", "rank_runs"] {
                pc.execute(
                    &pystr(
                        py,
                        &format!("DELETE FROM {table} WHERE run_id IN ({marks})"),
                    ),
                    ids.as_any(),
                )?;
            }
            pc.commit()?;
            Ok(Some(doomed))
        },
    )?;
    let Some(doomed) = doomed else {
        return Ok(0);
    };
    let vacuum = (|| -> PyResult<()> {
        let conn = connect(slf)?;
        with1(py, &conn, |conn| {
            conn.setattr("isolation_level", py.None())?;
            PyConn::new_internal(conn)?.run("VACUUM", vec![])?;
            Ok(())
        })?;
        Ok(())
    })();
    if let Err(exc) = vacuum {
        if !crate::pyhelp::is_sqlite(py, &exc, "Error")? {
            return Err(exc);
        }
        log(
            py,
            "warning",
            format!("Rank store: VACUUM after pruning failed: {}", es(py, &exc)),
        )?;
    }
    log(
        py,
        "info",
        format!(
            "Rank store: pruned {} run(s), keeping {}",
            doomed.len(),
            s(keep)?
        ),
    )?;
    Ok(doomed.len())
}

/// `BuffettRankStore.save_scores`: the ranked rows of one run.
#[pyfunction]
pub fn rs_save_scores<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    frame: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    if frame.getattr("empty")?.is_truthy()? {
        return Ok(0);
    }
    let frame_columns = frame.getattr("columns")?;
    let mut columns = Vec::new();
    for c in SCORE_COLUMNS {
        if frame_columns.contains(c)? {
            columns.push(c);
        }
    }
    let coerce = py.import("buffett_store")?.getattr("_coerce")?;
    let rows = PyList::empty(py);
    for pair in frame.call_method0("iterrows")?.try_iter()? {
        let pair = pair?;
        let (symbol, record) = (pair.get_item(0)?, pair.get_item(1)?);
        let mut cells = vec![run_id.clone(), symbol.str()?.into_any()];
        for column in &columns {
            cells.push(coerce.call1((record.call_method1("get", (*column,))?,))?);
        }
        rows.append(PyTuple::new(py, cells)?)?;
    }
    let statement = format!(
        "INSERT OR REPLACE INTO rank_scores (run_id, symbol, {}) VALUES ({})",
        columns.join(", "),
        vec!["?"; columns.len() + 2].join(", ")
    );
    let count = rows.len();
    locked(py, slf, |conn| {
        let pc = PyConn::new_internal(conn)?;
        pc.executemany(&pystr(py, &statement), rows.as_any())?;
        pc.commit()
    })?;
    Ok(count)
}

/// `BuffettRankStore.save_exclusions`
#[pyfunction]
pub fn rs_save_exclusions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    records: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    if !records.is_truthy()? {
        return Ok(0);
    }
    let rows = PyList::empty(py);
    for record in records.try_iter()? {
        let record = record?;
        let mut cells = vec![run_id.clone()];
        for key in [
            "symbol",
            "cik",
            "name",
            "model",
            "reasons",
            "period_count",
            "coverage",
        ] {
            cells.push(record.call_method1("get", (key,))?);
        }
        rows.append(PyTuple::new(py, cells)?)?;
    }
    let count = rows.len();
    locked(py, slf, |conn| {
        let pc = PyConn::new_internal(conn)?;
        pc.executemany(
            &pystr(
                py,
                "INSERT OR REPLACE INTO rank_exclusions
                   (run_id, symbol, cik, name, model, reasons, period_count, coverage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ),
            rows.as_any(),
        )?;
        pc.commit()
    })?;
    Ok(count)
}

/// `BuffettRankStore.latest_run_id`
#[pyfunction]
pub fn rs_latest_run_id<'py>(py: Python<'py>, slf: &Bound<'py, PyAny>) -> PyResult<Py<PyAny>> {
    let row = unlocked(py, slf, |conn| {
        let rows = PyConn::new_internal(conn)?.run(
            "SELECT run_id FROM rank_runs WHERE finished_at IS NOT NULL ORDER BY run_id DESC LIMIT 1",
            vec![],
        )?;
        let cols = rows.column_names(py)?;
        rows.first(py)
            .map(|r| row_dict(py, &cols, r.bind(py)))
            .transpose()
    })?;
    match row {
        Some(row) => Ok(py
            .import("builtins")?
            .getattr("int")?
            .call1((row.get_item("run_id")?,))?
            .unbind()),
        None => Ok(py.None()),
    }
}

/// `run_id or self.latest_run_id()`
fn resolve_run<'py>(
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if run_id.is_truthy()? {
        Ok(run_id.clone())
    } else {
        slf.call_method0("latest_run_id")
    }
}

/// The optional model and name/symbol search filters of the paged readers.
fn filters<'py>(
    py: Python<'py>,
    query: &mut String,
    params: &Bound<'py, PyList>,
    model: Option<&Bound<'py, PyAny>>,
    search: &Bound<'py, PyAny>,
) -> PyResult<()> {
    if let Some(model) = model
        && model.is_truthy()?
    {
        query.push_str(" AND model = ?");
        params.append(model)?;
    }
    if search.is_truthy()? && search.call_method0("strip")?.is_truthy()? {
        query.push_str(" AND (symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')");
        let escaped = py
            .import("buffett_store")?
            .getattr("_escape_like")?
            .call1((search.call_method0("strip")?,))?;
        let pattern = format!("%{}%", s(&escaped)?);
        params.append(&pattern)?;
        params.append(&pattern)?;
    }
    Ok(())
}

fn dict_rows<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    query: &str,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let rows = PyConn::new_internal(conn)?.execute(&pystr(py, query), params)?;
    let cols = rows.column_names(py)?;
    let out = PyList::empty(py);
    for r in rows.all(py) {
        out.append(row_dict(py, &cols, r.bind(py))?)?;
    }
    Ok(out)
}

fn count<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    query: &str,
    params: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = PyConn::new_internal(conn)?.execute(&pystr(py, query), params)?;
    let first = rows.first(py).ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err("'NoneType' object is not subscriptable")
    })?;
    py.import("builtins")?
        .getattr("int")?
        .call1((first.bind(py).get_item(0)?,))
}

/// `BuffettRankStore.get_ranked`: one page of the ranked list.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn rs_get_ranked<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
    offset: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
    search: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return Ok(PyList::empty(py));
    }
    let mut query = String::from("SELECT * FROM rank_scores WHERE run_id = ?");
    let params = PyList::new(py, [&run_id])?;
    filters(py, &mut query, &params, Some(model), search)?;
    query.push_str(" ORDER BY rank ASC LIMIT ? OFFSET ?");
    params.append(limit)?;
    params.append(offset)?;
    unlocked(py, slf, |conn| dict_rows(py, conn, &query, params.as_any()))
}

/// `BuffettRankStore.count_ranked`
#[pyfunction]
pub fn rs_count_ranked<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    model: &Bound<'py, PyAny>,
    search: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return Ok(0i64.into_pyobject(py)?.into_any());
    }
    let mut query = String::from("SELECT COUNT(*) FROM rank_scores WHERE run_id = ?");
    let params = PyList::new(py, [&run_id])?;
    filters(py, &mut query, &params, Some(model), search)?;
    unlocked(py, slf, |conn| count(py, conn, &query, params.as_any()))
}

/// `BuffettRankStore.get_exclusions`
#[pyfunction]
pub fn rs_get_exclusions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
    offset: &Bound<'py, PyAny>,
    search: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return Ok(PyList::empty(py));
    }
    let mut query = String::from("SELECT * FROM rank_exclusions WHERE run_id = ?");
    let params = PyList::new(py, [&run_id])?;
    filters(py, &mut query, &params, None, search)?;
    query.push_str(" ORDER BY symbol LIMIT ? OFFSET ?");
    params.append(limit)?;
    params.append(offset)?;
    unlocked(py, slf, |conn| dict_rows(py, conn, &query, params.as_any()))
}

/// `BuffettRankStore.count_exclusions`
#[pyfunction]
pub fn rs_count_exclusions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
    search: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return Ok(0i64.into_pyobject(py)?.into_any());
    }
    let mut query = String::from("SELECT COUNT(*) FROM rank_exclusions WHERE run_id = ?");
    let params = PyList::new(py, [&run_id])?;
    filters(py, &mut query, &params, None, search)?;
    unlocked(py, slf, |conn| count(py, conn, &query, params.as_any()))
}

/// `BuffettRankStore.get_run`
#[pyfunction]
pub fn rs_get_run<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return Ok(py.None());
    }
    let rows = unlocked(py, slf, |conn| {
        dict_rows(
            py,
            conn,
            "SELECT * FROM rank_runs WHERE run_id = ?",
            PyTuple::new(py, [&run_id])?.as_any(),
        )
    })?;
    Ok(if rows.is_empty() {
        py.None()
    } else {
        rows.get_item(0)?.unbind()
    })
}

/// `BuffettRankStore.get_scores_frame`: every scored row of a run.
#[pyfunction]
pub fn rs_get_scores_frame<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    run_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let pd = py.import("pandas")?;
    let run_id = resolve_run(slf, run_id)?;
    if run_id.is_none() {
        return pd.call_method0("DataFrame");
    }
    let rows = unlocked(py, slf, |conn| {
        dict_rows(
            py,
            conn,
            "SELECT * FROM rank_scores WHERE run_id = ? ORDER BY rank ASC",
            PyTuple::new(py, [&run_id])?.as_any(),
        )
    })?;
    pd.call_method1("DataFrame", (rows,))
}

/// `BuffettRankStore.get_symbol_history`: one company's rank across runs.
#[pyfunction]
pub fn rs_get_symbol_history<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    limit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    unlocked(py, slf, |conn| {
        let params = PyTuple::new(py, [symbol.call_method0("upper")?, limit.clone()])?;
        dict_rows(
            py,
            conn,
            "SELECT s.*, r.started_at FROM rank_scores s
                       JOIN rank_runs r ON r.run_id = s.run_id
                       WHERE s.symbol = ? ORDER BY s.run_id DESC LIMIT ?",
            params.as_any(),
        )
    })
}
