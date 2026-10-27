//! `sql_store`: the SQL the routes, the services and the AI workers used to
//! run inline — users, holding tags, the IBKR pending queue, idempotency
//! receipts, snapshots, the settings currency map, the archive's data-quality
//! probe and sync stamps, the heatmap's EDGAR facts, and the screener reads.
//!
//! Ported line for line from `src/sql_store.py` (frozen in
//! `tests/reference/sql_store_p17.py`). Every statement runs through
//! `PyConn`, which behaves as the connection's own `execute` (the same
//! errors, the implicit BEGIN, Python's own path for a connection Rust must
//! not touch); the frames come from `pyconn::read_frame`, the
//! `pd.read_sql_query` twin. Values are handled with Python's own
//! operations, and constants (`PENDING_COLUMNS`, `_SNAPSHOT_INSERT`) and
//! `pd`/`datetime`/`date` are read from the module at call time.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString, PyTuple};

use crate::fr::{builtin, none, pybool};
use crate::pl_helpers::kwargs;
use crate::pyconn::{PyConn, SqlRows, read_frame, row_dict};
use crate::pyhelp::pystr;

fn ss(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("sql_store")
}

fn s<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyAny> {
    PyString::new(py, text).into_any()
}

/// `conn.execute(sql, params)`
fn run<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &str,
    params: Vec<Bound<'py, PyAny>>,
) -> PyResult<SqlRows> {
    PyConn::new(conn)?.run(sql, params)
}

/// `conn.execute(sql, params).fetchone()`
fn fetchone<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &str,
    params: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyAny>> {
    let py = conn.py();
    let rows = run(conn, sql, params)?;
    Ok(rows
        .first(py)
        .map(|r| r.into_bound(py))
        .unwrap_or_else(|| none(py)))
}

/// `conn.execute(sql, params).fetchall()`
fn fetchall<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &str,
    params: Vec<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyList>> {
    let py = conn.py();
    PyList::new(py, run(conn, sql, params)?.all(py))
}

/// `conn.execute(sql, params).rowcount`
fn rowcount<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &str,
    params: Vec<Bound<'py, PyAny>>,
) -> PyResult<i64> {
    Ok(run(conn, sql, params)?.rowcount())
}

// --- users (the global database) -------------------------------------------

#[pyfunction]
pub fn ss_find_user_id<'py>(
    conn: &Bound<'py, PyAny>,
    username: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT id FROM users WHERE username = ?",
        vec![username.clone()],
    )
}

#[pyfunction]
pub fn ss_insert_user<'py>(
    conn: &Bound<'py, PyAny>,
    username: &Bound<'py, PyAny>,
    hashed_password: &Bound<'py, PyAny>,
    created_at: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let rows = run(
        conn,
        "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
        vec![
            username.clone(),
            hashed_password.clone(),
            created_at.clone(),
        ],
    )?;
    Ok(rows.lastrowid(conn.py()))
}

#[pyfunction]
pub fn ss_user_credentials<'py>(
    conn: &Bound<'py, PyAny>,
    username: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT id, username, hashed_password FROM users WHERE username = ?",
        vec![username.clone()],
    )
}

#[pyfunction]
pub fn ss_user_record<'py>(
    conn: &Bound<'py, PyAny>,
    username: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT id, username, is_active, created_at, alias FROM users WHERE username = ?",
        vec![username.clone()],
    )
}

#[pyfunction]
pub fn ss_set_user_alias<'py>(
    conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
    alias: &Bound<'py, PyAny>,
) -> PyResult<()> {
    run(
        conn,
        "UPDATE users SET alias = ? WHERE id = ?",
        vec![alias.clone(), user_id.clone()],
    )?;
    Ok(())
}

#[pyfunction]
pub fn ss_delete_user<'py>(conn: &Bound<'py, PyAny>, user_id: &Bound<'py, PyAny>) -> PyResult<i64> {
    rowcount(
        conn,
        "DELETE FROM users WHERE id = ?",
        vec![user_id.clone()],
    )
}

#[pyfunction]
pub fn ss_user_password_hash<'py>(
    conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT hashed_password FROM users WHERE id = ?",
        vec![user_id.clone()],
    )
}

#[pyfunction]
pub fn ss_set_user_password<'py>(
    conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
    hashed_password: &Bound<'py, PyAny>,
) -> PyResult<()> {
    run(
        conn,
        "UPDATE users SET hashed_password = ? WHERE id = ?",
        vec![hashed_password.clone(), user_id.clone()],
    )?;
    Ok(())
}

// --- a user's portfolio database --------------------------------------------

#[pyfunction]
pub fn ss_set_holding_tags<'py>(
    conn: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    account: &Bound<'py, PyAny>,
) -> PyResult<i64> {
    rowcount(
        conn,
        "UPDATE transactions SET Tags = ? WHERE Symbol = ? AND Account = ?",
        vec![tags.clone(), symbol.clone(), account.clone()],
    )
}

/// `c.replace('/', '_').replace(' ', '_')`, a column's bind name.
fn bind_name(col: &str) -> String {
    col.replace(['/', ' '], "_")
}

/// The pending columns, as Rust strings (they are fixed ASCII names).
fn pending_columns<'py>(m: &Bound<'py, PyModule>) -> PyResult<(Bound<'py, PyAny>, Vec<String>)> {
    let cols = m.getattr("PENDING_COLUMNS")?;
    let names = cols
        .try_iter()?
        .map(|c| c?.extract::<String>())
        .collect::<PyResult<Vec<_>>>()?;
    Ok((cols, names))
}

#[pyfunction]
pub fn ss_stage_pending_transaction<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    tx_data: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyTuple>> {
    let m = ss(py)?;
    let pc = PyConn::new(conn)?;
    let get = |k: &str| tx_data.call_method1("get", (k,));
    let get_or = |k: &str, d: Bound<'py, PyAny>| tx_data.call_method1("get", (k, d));

    let ext_id = get("ExternalID")?;
    if ext_id.is_truthy()? {
        // Check main table
        let main = pc.run(
            "SELECT id FROM transactions WHERE ExternalID = ?",
            vec![ext_id.clone()],
        )?;
        if main
            .first(py)
            .is_some_and(|r| r.bind(py).is_truthy().unwrap_or(true))
        {
            return PyTuple::new(py, [pybool(py, false), s(py, "duplicate_main")]);
        }
        // Check pending table
        let pending = pc.run(
            "SELECT id FROM pending_transactions WHERE ExternalID = ?",
            vec![ext_id.clone()],
        )?;
        if pending
            .first(py)
            .is_some_and(|r| r.bind(py).is_truthy().unwrap_or(true))
        {
            return PyTuple::new(py, [pybool(py, false), s(py, "duplicate_pending")]);
        }
    }

    let (_, db_cols) = pending_columns(&m)?;
    let placeholders = db_cols
        .iter()
        .map(|c| format!(":{}", bind_name(c)))
        .collect::<Vec<_>>()
        .join(", ");
    let cols_str = db_cols
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!("INSERT INTO pending_transactions ({cols_str}) VALUES ({placeholders});");

    let pd = m.getattr("pd")?;
    let date_types = PyTuple::new(py, [m.getattr("datetime")?, m.getattr("date")?])?;
    let zero = 0i64.into_pyobject(py)?.into_any();
    let sql_data = PyDict::new(py);
    for col in &db_cols {
        let mut val = if col != "user_id" {
            get(col)?
        } else {
            user_id.clone()
        };

        // Fallback for Total Amount if missing (or provided as 'Amount')
        if col == "Total Amount" && val.is_none() {
            val = get("Amount")?; // Try alternate key
            if val.is_none() {
                // Calculate
                let q = get_or("Quantity", zero.clone())?;
                let p = get_or("Price/Share", zero.clone())?;
                let c = get_or("Commission", zero.clone())?;
                let t = get_or("Type", s(py, ""))?.call_method0("upper")?;
                if q.is_truthy()? && p.is_truthy()? {
                    // `(q * p) + (c if t == "BUY" else -c)`, left to right.
                    let gross = q.mul(&p)?;
                    let fee = if t.eq("BUY")? { c } else { c.neg()? };
                    val = gross.add(fee)?;
                }
            }
        }

        if col == "Type" && val.is_instance_of::<PyString>() {
            val = val.call_method0("strip")?.call_method0("title")?;
        }
        if col == "Date" && val.is_instance(&date_types)? {
            val = val.call_method1("strftime", ("%Y-%m-%d",))?;
        }
        let value = if pd.call_method1("isna", (&val,))?.is_truthy()? {
            none(py)
        } else {
            val
        };
        sql_data.set_item(bind_name(col), value)?;
    }

    let inserted = pc.execute(&pystr(py, &sql), sql_data.as_any())?;
    PyTuple::new(py, [pybool(py, true), inserted.lastrowid(py)])
}

#[pyfunction]
pub fn ss_pending_transaction_records<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let query = s(py, "SELECT * FROM pending_transactions ORDER BY Date DESC");
    let df = read_frame(conn, &query, None, None)?;
    df.call_method(
        "to_dict",
        (),
        Some(&kwargs(py, &[("orient", &s(py, "records"))])?),
    )
}

#[pyfunction]
pub fn ss_approve_pending_transactions<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    ids: &Bound<'py, PyAny>,
    accounts: &Bound<'py, PyAny>,
) -> PyResult<i64> {
    let m = ss(py)?;
    let pc = PyConn::new(conn)?;
    let (cols, names) = pending_columns(&m)?;
    let cols_str = names
        .iter()
        .map(|c| format!("\"{c}\""))
        .collect::<Vec<_>>()
        .join(", ");
    let ext_id_idx = cols.call_method1("index", ("ExternalID",))?;
    let select = format!("SELECT {cols_str} FROM pending_transactions WHERE id = ?");
    let placeholders = vec!["?"; names.len()].join(", ");
    let insert = format!("INSERT INTO transactions ({cols_str}) VALUES ({placeholders})");
    let list = builtin(py, "list")?;

    let mut approved_count = 0i64;
    for p_id in ids.try_iter()? {
        let p_id = p_id?;
        // Fetch from pending
        let found = pc.run(&select, vec![p_id.clone()])?;
        let Some(row) = found.first(py).map(|r| r.into_bound(py)) else {
            continue;
        };
        if !row.is_truthy()? {
            continue;
        }
        let row = list.call1((row,))?;
        if accounts.contains(&p_id)? {
            row.set_item(
                cols.call_method1("index", ("Account",))?,
                accounts.get_item(&p_id)?,
            )?;
        }
        let ext_id = row.get_item(&ext_id_idx)?;
        if ext_id.is_truthy()? {
            let booked = pc.run(
                "SELECT 1 FROM transactions WHERE ExternalID = ? LIMIT 1",
                vec![ext_id],
            )?;
            if booked
                .first(py)
                .is_some_and(|r| r.bind(py).is_truthy().unwrap_or(true))
            {
                pc.run(
                    "DELETE FROM pending_transactions WHERE id = ?",
                    vec![p_id.clone()],
                )?;
                continue;
            }
        }
        // Insert into main table
        pc.execute(&pystr(py, &insert), &row)?;
        // Delete from pending
        pc.run(
            "DELETE FROM pending_transactions WHERE id = ?",
            vec![p_id.clone()],
        )?;
        approved_count += 1;
    }
    Ok(approved_count)
}

#[pyfunction]
pub fn ss_reject_pending_transactions<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    ids: &Bound<'py, PyAny>,
) -> PyResult<i64> {
    let pc = PyConn::new(conn)?; // the original's `conn.cursor()` comes first
    let n = ids.len()?;
    let sql = format!(
        "DELETE FROM pending_transactions WHERE id IN ({})",
        vec!["?"; n].join(",")
    );
    let params = builtin(py, "tuple")?.call1((ids,))?;
    Ok(pc.execute(&pystr(py, &sql), &params)?.rowcount())
}

#[pyfunction]
pub fn ss_begin_immediate(conn: &Bound<'_, PyAny>) -> PyResult<()> {
    run(conn, "BEGIN IMMEDIATE", vec![])?;
    Ok(())
}

#[pyfunction]
pub fn ss_find_receipt<'py>(
    conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
    endpoint: &Bound<'py, PyAny>,
    request_key: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT fingerprint, response FROM transaction_receipts WHERE user_id=? AND endpoint=? AND request_key=?",
        vec![user_id.clone(), endpoint.clone(), request_key.clone()],
    )
}

#[pyfunction]
pub fn ss_insert_receipt<'py>(
    conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
    endpoint: &Bound<'py, PyAny>,
    request_key: &Bound<'py, PyAny>,
    fingerprint: &Bound<'py, PyAny>,
    response: &Bound<'py, PyAny>,
) -> PyResult<()> {
    run(
        conn,
        "INSERT INTO transaction_receipts (user_id, endpoint, request_key, fingerprint, response) VALUES (?, ?, ?, ?, ?)",
        vec![
            user_id.clone(),
            endpoint.clone(),
            request_key.clone(),
            fingerprint.clone(),
            response.clone(),
        ],
    )?;
    Ok(())
}

#[pyfunction]
pub fn ss_replace_portfolio_snapshots<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    snapshot_date: &Bound<'py, PyAny>,
    rows: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let pc = PyConn::new(conn)?;
    pc.run(
        "DELETE FROM portfolio_snapshots WHERE snapshot_date=?",
        vec![snapshot_date.clone()],
    )?;
    let insert = ss(py)?.getattr("_SNAPSHOT_INSERT")?;
    for row in rows.try_iter()? {
        pc.execute(&insert, &row?)?;
    }
    Ok(())
}

#[pyfunction]
pub fn ss_symbol_currency_pairs<'py>(conn: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    fetchall(
        conn,
        "SELECT DISTINCT Symbol, \"Local Currency\" FROM transactions WHERE Symbol IS NOT NULL AND Symbol != ''",
        vec![],
    )
}

// --- the market archive -------------------------------------------------------

#[pyfunction]
pub fn ss_data_quality_table_present(conn: &Bound<'_, PyAny>) -> PyResult<bool> {
    fetchone(
        conn,
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_quality'",
        vec![],
    )?
    .is_truthy()
}

#[pyfunction]
pub fn ss_touch_sync_metadata<'py>(
    conn: &Bound<'py, PyAny>,
    symbols: &Bound<'py, PyAny>,
    synced_at: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let pc = PyConn::new(conn)?;
    for sym in symbols.try_iter()? {
        pc.run(
            "INSERT INTO sync_metadata (symbol, last_synced) VALUES (?, ?) \
             ON CONFLICT(symbol) DO UPDATE SET last_synced = excluded.last_synced",
            vec![sym?, synced_at.clone()],
        )?;
    }
    Ok(())
}

// --- the EDGAR fact store -------------------------------------------------------

#[pyfunction]
pub fn ss_facts_for<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    tags: &Bound<'py, PyAny>,
    ciks: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let marks = |n: usize| vec!["?"; n].join(",");
    let sql = format!(
        "SELECT cik, tag, period_end, val FROM facts WHERE tag IN ({}) AND cik IN ({})",
        marks(tags.len()?),
        marks(ciks.len()?)
    );
    let params = PyList::empty(py);
    for v in tags.try_iter()?.chain(ciks.try_iter()?) {
        params.append(v?)?;
    }
    let rows = PyConn::new(conn)?.execute(&pystr(py, &sql), params.as_any())?;
    PyList::new(py, rows.all(py))
}

// --- the screener cache -----------------------------------------------------------

#[pyfunction]
pub fn ss_screener_rows<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    sql: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let read = || -> PyResult<Bound<'py, PyList>> {
        let rows = PyConn::new(conn)?.execute(sql, PyTuple::empty(py).as_any())?;
        let columns = rows.column_names(py)?;
        let out = PyList::empty(py);
        for r in rows.all(py) {
            out.append(row_dict(py, &columns, r.bind(py))?)?;
        }
        Ok(out)
    };
    let result = read();
    // The original set `sqlite3.Row` on the connection before reading, so it
    // is left there whether or not the query ran.
    conn.setattr("row_factory", py.import("sqlite3")?.getattr("Row")?)?;
    result
}

#[pyfunction]
pub fn ss_cached_ai_review<'py>(
    conn: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    universe: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    fetchone(
        conn,
        "SELECT ai_summary, last_fiscal_year_end, most_recent_quarter FROM screener_cache WHERE symbol=? AND universe=?",
        vec![symbol.clone(), universe.clone()],
    )
}

#[pyfunction]
pub fn ss_symbols_missing_ai_review<'py>(conn: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    fetchall(
        conn,
        "SELECT symbol, universe FROM screener_cache \
         WHERE ai_summary IS NULL \
            OR length(trim(ai_summary)) <= 20 \
            OR lower(ai_summary) LIKE '%error%'",
        vec![],
    )
}

#[pyfunction]
pub fn ss_ai_review_frame<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
    dimensions: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let dims: Vec<String> = dimensions
        .try_iter()?
        .map(|d| d?.str()?.extract::<String>())
        .collect::<PyResult<_>>()?;
    let wanted = dims
        .iter()
        .map(|c| format!("{c} > 0"))
        .collect::<Vec<_>>()
        .join(" OR ");
    let query = format!(
        "SELECT symbol, {}, updated_at FROM screener_cache WHERE {wanted}",
        dims.join(", ")
    );
    read_frame(conn, &s(py, &query), None, None)
}
