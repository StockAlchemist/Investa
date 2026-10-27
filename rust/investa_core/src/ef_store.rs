//! `edgar_provider.EdgarFactStore`: the XBRL fact store, and the per-company
//! read session the ranking scores each company inside.
//!
//! Ported line for line from `src/edgar_provider.py` (frozen in
//! `tests/reference/edgar_provider.py`). Store methods take the store as
//! `slf` (its `_connect`, `_write_lock`, `db_path` and siblings are reached
//! through it); module state — the `as_of` and session context variables,
//! `_CompanySession`, the concept tables — is read from `edgar_provider` at
//! call time. The readers' row loops, which build the nested dicts the
//! concept resolution reads, are Rust; each dict is filled in the order the
//! rows arrive, so last-write-wins and insertion order are the original's.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PySet, PyTuple};

use crate::mdb::{with1, with2};
use crate::pyconn::{PyConn, SqlRows};
use crate::pyhelp::{now_iso, pystr};

fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("edgar_provider")
}

/// `x or ""`
fn or_empty<'py>(py: Python<'py>, x: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if x.is_truthy()? {
        Ok(x)
    } else {
        Ok(pystr(py, ""))
    }
}

/// `_effective_as_of`: the explicit date, else the block's `as_of`.
#[pyfunction]
pub fn ef_effective_as_of<'py>(
    py: Python<'py>,
    explicit: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    if !explicit.is_none() {
        return Ok(explicit.clone());
    }
    module(py)?.getattr("_as_of_var")?.call_method0("get")
}

/// `_session_for`: the open session for `cik` on this thread, or None.
#[pyfunction]
pub fn ef_session_for<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let session = module(py)?.getattr("_session_var")?.call_method0("get")?;
    if !session.is_none()
        && session.getattr("cik")?.eq(cik)?
        && session
            .getattr("thread")?
            .eq(py.import("threading")?.call_method0("get_ident")?)?
    {
        return Ok(session);
    }
    Ok(py.None().into_bound(py))
}

/// `company_session(cik)`: share one connection and each tag's rows across
/// the block's reads of `cik`. Inside an open session for the same company it
/// does nothing; on exit it resets the context variable and closes the
/// session's connections.
#[pyclass(module = "investa_core")]
pub struct CompanySession {
    cik: Py<PyAny>,
    token: Option<Py<PyAny>>,
    session: Option<Py<PyAny>>,
}

#[pymethods]
impl CompanySession {
    #[new]
    fn new(cik: Py<PyAny>) -> Self {
        CompanySession {
            cik,
            token: None,
            session: None,
        }
    }

    fn __enter__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let m = module(py)?;
        let cik = self.cik.bind(py);
        if !m.getattr("_session_for")?.call1((cik,))?.is_none() {
            return Ok(py.None()); // nested: the outer block's session serves
        }
        let session = m.getattr("_CompanySession")?.call1((cik,))?;
        let token = m
            .getattr("_session_var")?
            .call_method1("set", (&session,))?;
        self.token = Some(token.unbind());
        self.session = Some(session.unbind());
        Ok(py.None())
    }

    fn __exit__(
        &mut self,
        py: Python<'_>,
        _exc_type: &Bound<'_, PyAny>,
        _exc: &Bound<'_, PyAny>,
        _tb: &Bound<'_, PyAny>,
    ) -> PyResult<bool> {
        if let (Some(token), Some(session)) = (self.token.take(), self.session.take()) {
            module(py)?
                .getattr("_session_var")?
                .call_method1("reset", (token,))?;
            for conn in session
                .bind(py)
                .getattr("connections")?
                .call_method0("values")?
                .try_iter()?
            {
                conn?.call_method0("close")?;
            }
        }
        Ok(false)
    }
}

fn connect<'py>(slf: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    slf.call_method0("_connect")
}

fn with_conn<'py, T>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let conn = connect(slf)?;
    Ok(with1(py, &conn, body)?.expect("no CM swallows"))
}

fn locked<'py, T>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    body: impl FnOnce(&Bound<'py, PyAny>) -> PyResult<T>,
) -> PyResult<T> {
    let lock = slf.getattr("_write_lock")?;
    Ok(with2(py, &lock, || connect(slf), body)?.expect("no CM swallows"))
}

/// The `WHERE cik = ? AND tag IN (...)` read the multi-tag readers share.
fn tag_query<'py>(
    py: Python<'py>,
    select: &str,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyList>,
    as_of: &Bound<'py, PyAny>,
    order: &str,
) -> PyResult<(String, Bound<'py, PyList>)> {
    let mut query = format!(
        "{select} IN ({})\n        ",
        vec!["?"; tags.len()].join(",")
    );
    let params = PyList::new(py, [cik])?;
    for t in tags.iter() {
        params.append(t)?;
    }
    if as_of.is_truthy()? {
        query.push_str(" AND filed <= ?");
        params.append(as_of)?;
    }
    query.push_str(order);
    Ok((query, params))
}

fn rows_of<'py>(
    conn: &Bound<'py, PyAny>,
    query: &str,
    params: &Bound<'py, PyList>,
) -> PyResult<SqlRows> {
    let py = conn.py();
    PyConn::new(conn)?.execute(&pystr(py, query), params.as_any())
}

/// `setdefault(key, factory())` on a Python dict.
fn entry<'py>(
    d: &Bound<'py, PyDict>,
    key: &Bound<'py, PyAny>,
    make: impl FnOnce() -> Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    match d.get_item(key)? {
        Some(v) => Ok(v),
        None => {
            let v = make();
            d.set_item(key, &v)?;
            Ok(v)
        }
    }
}

fn listed<'py>(py: Python<'py>, tags: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    PyList::new(py, tags.try_iter()?.collect::<PyResult<Vec<_>>>()?)
}

/// `EdgarFactStore.get_many_tag_spans`: {tag: {(start, end): (value, unit)}}
/// from the quarterly table, newest filing winning per span.
#[pyfunction]
pub fn ef_get_many_tag_spans<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_list = listed(py, tags)?;
    let result = PyDict::new(py);
    if tag_list.is_empty() {
        return Ok(result);
    }
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    let (query, params) = tag_query(
        py,
        "\n            SELECT tag, period_start, period_end, val, unit FROM quarterly_facts\n            WHERE cik = ? AND tag",
        cik,
        &tag_list,
        &as_of,
        " ORDER BY period_end, filed",
    )?;
    with_conn(py, slf, |conn| {
        for row in rows_of(conn, &query, &params)?.all(py) {
            let row = row.bind(py);
            let val = row.get_item(3)?;
            if val.is_none() {
                continue;
            }
            let spans = entry(&result, &row.get_item(0)?, || PyDict::new(py).into_any())?;
            let key = PyTuple::new(py, [or_empty(py, row.get_item(1)?)?, row.get_item(2)?])?;
            spans.set_item(key, PyTuple::new(py, [val, row.get_item(4)?])?)?;
        }
        Ok(())
    })?;
    Ok(result)
}

/// `EdgarFactStore.get_many_tag_spans_by_filing`: {tag: [((filed, accn),
/// {(start, end): (value, unit)}), ...]}, newest filing first.
#[pyfunction]
pub fn ef_get_many_tag_spans_by_filing<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_list = listed(py, tags)?;
    let for_tag = PyDict::new(py);
    if tag_list.is_empty() {
        return Ok(for_tag);
    }
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    let (query, params) = tag_query(
        py,
        "\n            SELECT tag, filed, accn, period_start, period_end, val, unit\n            FROM quarterly_facts\n            WHERE cik = ? AND tag",
        cik,
        &tag_list,
        &as_of,
        " ORDER BY filed DESC, accn DESC",
    )?;
    let grouped = PyDict::new(py);
    with_conn(py, slf, |conn| {
        for row in rows_of(conn, &query, &params)?.all(py) {
            let row = row.bind(py);
            let val = row.get_item(5)?;
            if val.is_none() {
                continue;
            }
            let tag = row.get_item(0)?;
            let filings =
                entry(&grouped, &tag, || PyDict::new(py).into_any())?.cast_into::<PyDict>()?;
            let key = PyTuple::new(
                py,
                [
                    or_empty(py, row.get_item(1)?)?,
                    or_empty(py, row.get_item(2)?)?,
                ],
            )?
            .into_any();
            let spans = match filings.get_item(&key)? {
                Some(s) => s,
                None => {
                    let s = PyDict::new(py).into_any();
                    filings.set_item(&key, &s)?;
                    entry(&for_tag, &tag, || PyList::empty(py).into_any())?
                        .call_method1("append", (PyTuple::new(py, [&key, &s])?,))?;
                    s
                }
            };
            let span = PyTuple::new(py, [or_empty(py, row.get_item(3)?)?, row.get_item(4)?])?;
            spans.set_item(span, PyTuple::new(py, [val, row.get_item(6)?])?)?;
        }
        Ok(())
    })?;
    Ok(for_tag)
}

/// `EdgarFactStore.get_tag_series`: {period_end: (value, unit)} for one tag.
#[pyfunction]
pub fn ef_get_tag_series<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tag: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    let mut query = String::from(
        "\n            SELECT period_end, val, unit, filed FROM facts\n            WHERE cik = ? AND tag = ?\n        ",
    );
    let params = PyList::new(py, [cik, tag])?;
    if as_of.is_truthy()? {
        query.push_str(" AND filed <= ?");
        params.append(&as_of)?;
    }
    query.push_str(" ORDER BY period_end, filed");
    let result = PyDict::new(py);
    with_conn(py, slf, |conn| {
        for row in rows_of(conn, &query, &params)?.all(py) {
            let row = row.bind(py);
            let val = row.get_item(1)?;
            if !val.is_none() {
                result.set_item(row.get_item(0)?, PyTuple::new(py, [val, row.get_item(2)?])?)?;
            }
        }
        Ok(())
    })?;
    Ok(result)
}

/// `EdgarFactStore._many_tag_series_query`
#[pyfunction]
pub fn ef_many_tag_series_query<'py>(
    py: Python<'py>,
    cik: &Bound<'py, PyAny>,
    tag_list: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<(String, Bound<'py, PyList>)> {
    tag_query(
        py,
        "\n            SELECT tag, period_end, val, unit FROM facts\n            WHERE cik = ? AND tag",
        cik,
        &listed(py, tag_list)?,
        as_of,
        " ORDER BY period_end, filed",
    )
}

fn fetch_series<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    conn: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let qp = slf.call_method1("_many_tag_series_query", (cik, tags, as_of))?;
    let (query, params) = (qp.get_item(0)?, qp.get_item(1)?);
    let result = PyDict::new(py);
    for row in PyConn::new(conn)?.execute(&query, &params)?.all(py) {
        let row = row.bind(py);
        let val = row.get_item(2)?;
        if !val.is_none() {
            entry(&result, &row.get_item(0)?, || PyDict::new(py).into_any())?
                .set_item(row.get_item(1)?, PyTuple::new(py, [val, row.get_item(3)?])?)?;
        }
    }
    Ok(result)
}

fn fetch_by_filing<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let (query, params) = tag_query(
        py,
        "\n                SELECT tag, filed, accn, period_end, val FROM facts\n                WHERE cik = ? AND tag",
        cik,
        &listed(py, tags)?,
        as_of,
        " ORDER BY filed DESC, accn DESC",
    )?;
    let seen = PyDict::new(py);
    let for_tag = PyDict::new(py);
    for row in rows_of(conn, &query, &params)?.all(py) {
        let row = row.bind(py);
        let val = row.get_item(4)?;
        if val.is_none() {
            continue;
        }
        let tag = row.get_item(0)?;
        let filings = entry(&seen, &tag, || PyDict::new(py).into_any())?.cast_into::<PyDict>()?;
        let key = PyTuple::new(
            py,
            [
                or_empty(py, row.get_item(1)?)?,
                or_empty(py, row.get_item(2)?)?,
            ],
        )?
        .into_any();
        let values = match filings.get_item(&key)? {
            Some(v) => v,
            None => {
                let v = PyDict::new(py).into_any();
                filings.set_item(&key, &v)?;
                entry(&for_tag, &tag, || PyList::empty(py).into_any())?
                    .call_method1("append", (&v,))?;
                v
            }
        };
        values.set_item(row.get_item(3)?, val)?;
    }
    Ok(for_tag)
}

/// `EdgarFactStore._read_tags` for the two cached readers.
fn read_tags<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    reader: &str,
    cik: &Bound<'py, PyAny>,
    tag_list: &Bound<'py, PyList>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let m = module(py)?;
    let fetch =
        |conn: &Bound<'py, PyAny>, tags: &Bound<'py, PyAny>| -> PyResult<Bound<'py, PyDict>> {
            if reader == "series" {
                fetch_series(py, slf, conn, cik, tags, as_of)
            } else {
                fetch_by_filing(py, conn, cik, tags, as_of)
            }
        };
    let session = m.getattr("_session_for")?.call1((cik,))?;
    if session.is_none() {
        return with_conn(py, slf, |conn| fetch(conn, tag_list.as_any()));
    }
    let key = PyTuple::new(
        py,
        [pystr(py, reader), slf.getattr("db_path")?, as_of.clone()],
    )?;
    let pair = session.getattr("loaded")?.call_method1(
        "setdefault",
        (
            key,
            PyTuple::new(
                py,
                [PySet::empty(py)?.into_any(), PyDict::new(py).into_any()],
            )?,
        ),
    )?;
    let (loaded, rows) = (pair.get_item(0)?, pair.get_item(1)?);
    let wanted = PySet::new(py, tag_list.iter())?;
    if !wanted.call_method1("__le__", (&loaded,))?.is_truthy()? {
        let prefetch = if reader == "series" {
            m.getattr("_all_chain_tags")?.call0()?
        } else {
            m.getattr("_tags_for")?.call1((m
                .getattr("_SHARE_SCALED")?
                .add(m.getattr("_PER_SHARE_SCALED")?)?,))?
        };
        let order = PyList::new(py, tag_list.iter())?;
        for t in prefetch.try_iter()? {
            order.append(t?)?;
        }
        let missing = PyList::empty(py);
        let builtins = py.import("builtins")?;
        for t in builtins
            .getattr("dict")?
            .call_method1("fromkeys", (order,))?
            .try_iter()?
        {
            let t = t?;
            if !loaded.contains(&t)? {
                missing.append(t)?;
            }
        }
        let connections = session.getattr("connections")?;
        let path = slf.getattr("db_path")?;
        let mut conn = connections.call_method1("get", (&path,))?;
        if conn.is_none() {
            conn = connect(slf)?;
            connections.set_item(&path, &conn)?;
        }
        rows.call_method1("update", (fetch(&conn, missing.as_any())?,))?;
        loaded.call_method1("update", (missing,))?;
    }
    let out = PyDict::new(py);
    let dict = py.import("builtins")?.getattr("dict")?;
    for item in rows.call_method0("items")?.try_iter()? {
        let item = item?;
        let (tag, value) = (item.get_item(0)?, item.get_item(1)?);
        if !wanted.contains(&tag)? {
            continue;
        }
        let copied = if reader == "series" {
            dict.call1((value,))?
        } else {
            let l = PyList::empty(py);
            for f in value.try_iter()? {
                l.append(dict.call1((f?,))?)?;
            }
            l.into_any()
        };
        out.set_item(tag, copied)?;
    }
    Ok(out)
}

/// `EdgarFactStore.get_many_tag_series`: {tag: {period_end: (value, unit)}}.
#[pyfunction]
pub fn ef_get_many_tag_series<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_list = listed(py, tags)?;
    if tag_list.is_empty() {
        return Ok(PyDict::new(py));
    }
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    read_tags(py, slf, "series", cik, &tag_list, &as_of)
}

/// `EdgarFactStore.get_tag_series_by_filing`: {tag: [{period_end: value}, ...]},
/// newest filing first.
#[pyfunction]
pub fn ef_get_tag_series_by_filing<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_list = listed(py, tags)?;
    if tag_list.is_empty() {
        return Ok(PyDict::new(py));
    }
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    read_tags(py, slf, "by_filing", cik, &tag_list, &as_of)
}

/// `EdgarFactStore.get_tag_revisions`: {(tag, period_end): [(filed, value, form)]}.
#[pyfunction]
pub fn ef_get_tag_revisions<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    as_of: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyDict>> {
    let tag_list = listed(py, tags)?;
    let result = PyDict::new(py);
    if tag_list.is_empty() {
        return Ok(result);
    }
    let as_of = module(py)?.getattr("_effective_as_of")?.call1((as_of,))?;
    let mut query = format!(
        "\n            SELECT tag, period_end, filed, val, form FROM facts\n            WHERE cik = ? AND tag IN ({}) AND val IS NOT NULL\n        ",
        vec!["?"; tag_list.len()].join(",")
    );
    let params = PyList::new(py, [cik])?;
    for t in tag_list.iter() {
        params.append(t)?;
    }
    if as_of.is_truthy()? {
        query.push_str(" AND filed <= ?");
        params.append(&as_of)?;
    }
    query.push_str(" ORDER BY period_end, filed");
    with_conn(py, slf, |conn| {
        for row in rows_of(conn, &query, &params)?.all(py) {
            let row = row.bind(py);
            let filed = row.get_item(2)?;
            if filed.is_truthy()? {
                let key = PyTuple::new(py, [row.get_item(0)?, row.get_item(1)?])?;
                entry(&result, key.as_any(), || PyList::empty(py).into_any())?.call_method1(
                    "append",
                    (PyTuple::new(
                        py,
                        [filed, row.get_item(3)?, row.get_item(4)?],
                    )?,),
                )?;
            }
        }
        Ok(())
    })?;
    Ok(result)
}

/// `EdgarFactStore.has_data`
#[pyfunction]
pub fn ef_has_data<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    with_conn(py, slf, |conn| {
        Ok(PyConn::new(conn)?
            .run(
                "SELECT 1 FROM facts WHERE cik = ? LIMIT 1",
                vec![cik.clone()],
            )?
            .first(py)
            .is_some())
    })
}

/// `EdgarFactStore._init_db`
#[pyfunction]
pub fn ef_init_db<'py>(py: Python<'py>, slf: &Bound<'py, PyAny>) -> PyResult<()> {
    locked(py, slf, |conn| {
        let pc = PyConn::new(conn)?;
        for sql in [
            "
                CREATE TABLE IF NOT EXISTS facts (
                    cik TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    period_start TEXT,
                    val REAL,
                    unit TEXT,
                    form TEXT,
                    accn TEXT NOT NULL,
                    filed TEXT,
                    PRIMARY KEY (cik, tag, period_end, accn)
                )
            ",
            "CREATE INDEX IF NOT EXISTS idx_facts_cik_tag ON facts (cik, tag)",
            "
                CREATE TABLE IF NOT EXISTS ingest_log (
                    cik TEXT PRIMARY KEY,
                    ingested_at TEXT,
                    fact_count INTEGER
                )
            ",
            "
                CREATE TABLE IF NOT EXISTS quarterly_facts (
                    cik TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    val REAL,
                    unit TEXT,
                    form TEXT,
                    accn TEXT NOT NULL,
                    filed TEXT,
                    PRIMARY KEY (cik, tag, period_start, period_end, accn)
                )
            ",
            "CREATE INDEX IF NOT EXISTS idx_quarterly_cik_tag ON quarterly_facts (cik, tag)",
            "
                CREATE TABLE IF NOT EXISTS quarterly_ingest_log (
                    cik TEXT PRIMARY KEY,
                    ingested_at TEXT,
                    fact_count INTEGER
                )
            ",
        ] {
            pc.run(sql, vec![])?;
        }
        pc.commit()
    })
}

fn upsert<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    sql: &str,
    rows: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    if !rows.is_truthy()? {
        return Ok(0);
    }
    locked(py, slf, |conn| {
        let pc = PyConn::new(conn)?;
        pc.executemany(&pystr(py, sql), rows)?;
        pc.commit()
    })?;
    rows.len()
}

/// `EdgarFactStore.upsert_facts`
#[pyfunction]
pub fn ef_upsert_facts<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    upsert(
        py,
        slf,
        "INSERT OR REPLACE INTO facts
                   (cik, tag, period_end, period_start, val, unit, form, accn, filed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
}

/// `EdgarFactStore.upsert_quarterly_facts`
#[pyfunction]
pub fn ef_upsert_quarterly_facts<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<usize> {
    upsert(
        py,
        slf,
        "INSERT OR REPLACE INTO quarterly_facts
                   (cik, tag, period_start, period_end, val, unit, form, accn, filed)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
}

fn mark<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    table: &str,
    cik: &Bound<'py, PyAny>,
    count: &Bound<'py, PyAny>,
) -> PyResult<()> {
    locked(py, slf, |conn| {
        let pc = PyConn::new(conn)?;
        pc.run(
            &format!(
                "INSERT OR REPLACE INTO {table} (cik, ingested_at, fact_count) VALUES (?, ?, ?)"
            ),
            vec![cik.clone(), pystr(py, &now_iso(py)?), count.clone()],
        )?;
        pc.commit()
    })
}

/// `EdgarFactStore.mark_ingested`
#[pyfunction]
pub fn ef_mark_ingested<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    fact_count: &Bound<'py, PyAny>,
) -> PyResult<()> {
    mark(py, slf, "ingest_log", cik, fact_count)
}

/// `EdgarFactStore.mark_quarterly_ingested`
#[pyfunction]
pub fn ef_mark_quarterly_ingested<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
    fact_count: &Bound<'py, PyAny>,
) -> PyResult<()> {
    mark(py, slf, "quarterly_ingest_log", cik, fact_count)
}

/// `EdgarFactStore.ingested_ciks`
#[pyfunction]
pub fn ef_ingested_ciks<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PySet>> {
    with_conn(py, slf, |conn| {
        let out = PySet::empty(py)?;
        for row in PyConn::new(conn)?
            .run("SELECT cik FROM ingest_log", vec![])?
            .all(py)
        {
            out.add(row.bind(py).get_item(0)?)?;
        }
        Ok(out)
    })
}

/// `EdgarFactStore.quarterly_ingest_state`: (when, how many) or None.
#[pyfunction]
pub fn ef_quarterly_ingest_state<'py>(
    py: Python<'py>,
    slf: &Bound<'py, PyAny>,
    cik: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let row = with_conn(py, slf, |conn| {
        Ok(PyConn::new(conn)?
            .run(
                "SELECT ingested_at, fact_count FROM quarterly_ingest_log WHERE cik = ?",
                vec![cik.clone()],
            )?
            .first(py))
    })?;
    let Some(row) = row else {
        return Ok(py.None());
    };
    let row = row.bind(py);
    if !row.get_item(0)?.is_truthy()? {
        return Ok(py.None());
    }
    let when = match py
        .import("datetime")?
        .getattr("datetime")?
        .call_method1("fromisoformat", (row.get_item(0)?,))
    {
        Ok(w) => w,
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => return Ok(py.None()),
        Err(e) => return Err(e),
    };
    let count = or_zero(py, row.get_item(1)?)?;
    match py.import("builtins")?.getattr("int")?.call1((count,)) {
        Ok(n) => Ok(PyTuple::new(py, [when, n])?.into_any().unbind()),
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyValueError>(py) => Ok(py.None()),
        Err(e) => Err(e),
    }
}

fn or_zero<'py>(py: Python<'py>, x: Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if x.is_truthy()? {
        Ok(x)
    } else {
        Ok(0i64.into_pyobject(py)?.into_any())
    }
}
