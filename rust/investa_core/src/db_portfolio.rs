//! `db_utils`: the per-user portfolio database (transactions, watchlists).
//!
//! Ported line for line from `src/db_utils.py` (frozen in
//! `tests/reference/db_utils.py`). Statements run through `PyConn` on the
//! caller's own connection; the Python operations whose exact behaviour the
//! original relied on (`str.title`, `strptime`, `pd.isna`, `str()` of a dict
//! in a log line) are the same calls. Sibling functions are looked up on the
//! `db_utils` module at call time, so a patched attribute is the one used.

use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};

use crate::pyconn::PyConn;
use crate::pyhelp::{es, get, get_or, isna, log, log_exc, now_iso, pystr, s};

const OUTFLOW_TYPES: [&str; 9] = [
    "buy",
    "withdrawal",
    "fees",
    "fee",
    "tax",
    "withholding tax",
    "split",
    "stock split",
    "buy to cover",
];

const TRANSACTION_COLUMNS: [&str; 15] = [
    "Date",
    "Type",
    "Symbol",
    "Quantity",
    "Price/Share",
    "Total Amount",
    "Commission",
    "Account",
    "Split Ratio",
    "Note",
    "Local Currency",
    "To Account",
    "Tags",
    "ExternalID",
    "user_id",
];

/// `col.replace("/", "_per_").replace(" ", "_").replace(".", "_")`
fn placeholder(col: &str) -> String {
    col.replace('/', "_per_").replace([' ', '.'], "_")
}

/// `_apply_total_amount_sign`: outflow types negative, inflow positive.
#[pyfunction]
pub fn db_apply_total_amount_sign<'py>(
    py: Python<'py>,
    transaction_data: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let total = get(transaction_data, "Total Amount")?;
    if total.is_none() {
        return Ok(transaction_data.clone());
    }
    let total_f: f64 = match py.get_type::<PyFloat>().call1((&total,)) {
        Ok(f) => f.extract()?,
        Err(e) if e.is_instance_of::<PyTypeError>(py) || e.is_instance_of::<PyValueError>(py) => {
            return Ok(transaction_data.clone());
        }
        Err(e) => return Err(e),
    };
    if total_f.is_nan() || total_f.abs() < 1e-9 {
        return Ok(transaction_data.clone());
    }
    let tx_type: String = get_or(transaction_data, "Type", &pystr(py, ""))?
        .str()?
        .call_method0("lower")?
        .call_method0("strip")?
        .extract()?;
    let signed = if OUTFLOW_TYPES.contains(&tx_type.as_str()) {
        -total_f.abs()
    } else {
        total_f.abs()
    };
    if signed == total_f {
        return Ok(transaction_data.clone());
    }
    let result = py
        .import("builtins")?
        .getattr("dict")?
        .call1((transaction_data,))?;
    result.set_item("Total Amount", signed)?;
    Ok(result)
}

/// The three date shapes the forms send, each tried with `datetime.strptime`.
fn valid_date_string(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let dt = py.import("datetime")?.getattr("datetime")?;
    for fmt in ["%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S"] {
        match dt.call_method1("strptime", (value, fmt)) {
            Ok(_) => return Ok(true),
            Err(e) if e.is_instance_of::<PyValueError>(py) => continue,
            Err(e) => return Err(e),
        }
    }
    Ok(false)
}

fn is_date_like(py: Python<'_>, value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let m = py.import("datetime")?;
    let classes = PyTuple::new(py, [m.getattr("datetime")?, m.getattr("date")?])?;
    value.is_instance(&classes)
}

/// `value.strip().title()`
fn title<'py>(value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    value.call_method0("strip")?.call_method0("title")
}

fn rollback_quietly(conn: &Bound<'_, PyAny>) {
    let _ = conn.call_method0("rollback");
}

/// `add_transaction_to_db`: (True, new id), (True, existing id) for a known
/// ExternalID, or (False, None).
#[pyfunction]
pub fn db_add_transaction<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    transaction_data: &Bound<'py, PyAny>,
    commit: &Bound<'py, PyAny>,
) -> PyResult<(bool, Py<PyAny>)> {
    let commit = commit.is_truthy()?;
    let transaction_data = db_apply_total_amount_sign(py, transaction_data)?;

    // --- Deduplication Check ---
    let external_id = get(&transaction_data, "ExternalID")?;
    if external_id.is_truthy()? {
        let checked = (|| -> PyResult<Option<Py<PyAny>>> {
            let conn = PyConn::new(db_conn)?;
            let rows = conn.run(
                "SELECT id FROM transactions WHERE ExternalID = ?",
                vec![external_id.clone()],
            )?;
            Ok(rows.first(py))
        })();
        match checked {
            Ok(Some(existing)) => {
                let id = existing.bind(py).get_item(0)?;
                log(
                    py,
                    "info",
                    format!(
                        "Transaction with ExternalID {} already exists (ID: {}). Skipping.",
                        s(&external_id)?,
                        s(&id)?
                    ),
                )?;
                return Ok((true, id.unbind()));
            }
            Ok(None) => {}
            Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
                log(
                    py,
                    "warning",
                    format!(
                        "Error checking for existing ExternalID {}: {}",
                        s(&external_id)?,
                        es(py, &e)
                    ),
                )?;
            }
            Err(e) => return Err(e),
        }
    }

    let quoted: Vec<String> = TRANSACTION_COLUMNS
        .iter()
        .map(|c| {
            if c.contains('/') || c.contains(' ') {
                format!("\"{c}\"")
            } else {
                (*c).to_string()
            }
        })
        .collect();
    let mut placeholders = Vec::new();
    let data = PyDict::new(py);
    for col in TRANSACTION_COLUMNS {
        let name = placeholder(col);
        placeholders.push(format!(":{name}"));
        let value = get(&transaction_data, col)?;
        let cell = if col == "Type" && value.is_instance_of::<PyString>() {
            title(&value)?
        } else if col == "Date" {
            if is_date_like(py, &value)? {
                value.call_method1("strftime", ("%Y-%m-%d",))?
            } else if value.is_instance_of::<PyString>() {
                if valid_date_string(py, &value)? {
                    // Truncate to YYYY-MM-DD to maintain DB standard
                    value.get_item(
                        py.import("builtins")?
                            .getattr("slice")?
                            .call1((py.None(), 10))?,
                    )?
                } else {
                    log(
                        py,
                        "error",
                        format!("Invalid date string for DB: {}", s(&value)?),
                    )?;
                    return Ok((false, py.None()));
                }
            } else {
                py.None().into_bound(py)
            }
        } else if isna(py, &value)? {
            py.None().into_bound(py)
        } else {
            value
        };
        data.set_item(name, cell)?;
    }
    let sql = format!(
        "INSERT INTO transactions ({}) VALUES ({});",
        quoted.join(", "),
        placeholders.join(", ")
    );

    let inserted = (|| -> PyResult<Py<PyAny>> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.execute(&pystr(py, &sql), data.as_any())?;
        if commit {
            conn.commit()?;
        }
        Ok(rows.lastrowid(py).unbind())
    })();
    match inserted {
        Ok(new_id) => {
            log(
                py,
                "info",
                format!(
                    "Successfully added transaction with ID: {}.",
                    s(new_id.bind(py))?
                ),
            )?;
            Ok((true, new_id))
        }
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            log_exc(
                py,
                "error",
                format!(
                    "Error adding transaction to database. Data: {} - Error: {}",
                    s(&transaction_data)?,
                    es(py, &e)
                ),
                &e,
            )?;
            if commit {
                rollback_quietly(db_conn);
            }
            Ok((false, py.None()))
        }
        Err(e) => Err(e),
    }
}

/// `update_transaction_in_db`
#[pyfunction]
pub fn db_update_transaction<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    transaction_id: &Bound<'py, PyAny>,
    new_data_dict: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let new_data_dict = db_apply_total_amount_sign(py, new_data_dict)?;
    let mut set_clauses: Vec<String> = Vec::new();
    let values = PyDict::new(py);
    for item in new_data_dict.call_method0("items")?.try_iter()? {
        let item = item?;
        let key = item.get_item(0)?;
        let value = item.get_item(1)?;
        let Ok(key_str) = key.extract::<String>() else {
            continue; // not a column name (`key not in db_columns`)
        };
        if !TRANSACTION_COLUMNS.contains(&key_str.as_str()) {
            continue;
        }
        let ph = placeholder(&key_str);
        set_clauses.push(format!("\"{key_str}\" = :{ph}"));
        let cell = if key_str == "Type" && value.is_instance_of::<PyString>() {
            title(&value)?
        } else if key_str == "Date" {
            if is_date_like(py, &value)? {
                value.call_method1("strftime", ("%Y-%m-%d",))?
            } else if value.is_instance_of::<PyString>() {
                if valid_date_string(py, &value)? {
                    value.clone()
                } else {
                    log(
                        py,
                        "error",
                        format!("Invalid date string for update: {}", s(&value)?),
                    )?;
                    return Ok(false);
                }
            } else {
                py.None().into_bound(py)
            }
        } else if isna(py, &value)? {
            py.None().into_bound(py)
        } else {
            value
        };
        values.set_item(ph, cell)?;
    }
    if set_clauses.is_empty() {
        log(py, "warning", "Update transaction: No data provided.")?;
        return Ok(false);
    }
    let sql = format!(
        "UPDATE transactions SET {} WHERE id = :id_placeholder",
        set_clauses.join(", ")
    );
    values.set_item("id_placeholder", transaction_id)?;

    let updated = (|| -> PyResult<i64> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.execute(&pystr(py, &sql), values.as_any())?;
        conn.commit()?;
        Ok(rows.rowcount())
    })();
    match updated {
        Ok(0) => {
            log(
                py,
                "warning",
                format!(
                    "Update transaction: No row found with ID: {}",
                    s(transaction_id)?
                ),
            )?;
            Ok(false)
        }
        Ok(_) => {
            log(
                py,
                "info",
                format!(
                    "Successfully updated transaction ID: {}.",
                    s(transaction_id)?
                ),
            )?;
            Ok(true)
        }
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            log_exc(
                py,
                "error",
                format!(
                    "Error updating transaction ID {}. Data: {} - Error: {}",
                    s(transaction_id)?,
                    s(&new_data_dict)?,
                    es(py, &e)
                ),
                &e,
            )?;
            rollback_quietly(db_conn);
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

/// `delete_transaction_from_db`
#[pyfunction]
pub fn db_delete_transaction<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    transaction_id: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let deleted = (|| -> PyResult<i64> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "DELETE FROM transactions WHERE id = ?",
            vec![transaction_id.clone()],
        )?;
        conn.commit()?;
        Ok(rows.rowcount())
    })();
    match deleted {
        Ok(0) => {
            log(
                py,
                "warning",
                format!(
                    "Delete transaction: No row found with ID: {}",
                    s(transaction_id)?
                ),
            )?;
            Ok(false)
        }
        Ok(_) => {
            log(
                py,
                "info",
                format!(
                    "Successfully deleted transaction ID: {}",
                    s(transaction_id)?
                ),
            )?;
            Ok(true)
        }
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            log_exc(
                py,
                "error",
                format!(
                    "Error deleting transaction ID {}: {}",
                    s(transaction_id)?,
                    es(py, &e)
                ),
                &e,
            )?;
            rollback_quietly(db_conn);
            Ok(false)
        }
        Err(e) => Err(e),
    }
}

// --- watchlists --------------------------------------------------------------

fn sqlite_or<T>(
    py: Python<'_>,
    r: PyResult<T>,
    on_error: impl FnOnce(&PyErr) -> PyResult<T>,
) -> PyResult<T> {
    match r {
        Ok(v) => Ok(v),
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => on_error(&e),
        Err(e) => Err(e),
    }
}

/// `get_all_watchlists`
#[pyfunction]
pub fn db_get_all_watchlists<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let _ = user_id;
    let r = (|| -> PyResult<Bound<'py, PyList>> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "SELECT id, name, created_at FROM watchlists ORDER BY created_at ASC",
            vec![],
        )?;
        let out = PyList::empty(py);
        for row in rows.all(py) {
            let row = row.bind(py);
            let d = PyDict::new(py);
            d.set_item("id", row.get_item(0)?)?;
            d.set_item("name", row.get_item(1)?)?;
            d.set_item("created_at", row.get_item(2)?)?;
            out.append(d)?;
        }
        Ok(out)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!("Error fetching watchlists: {}", es(py, e)),
        )?;
        Ok(PyList::empty(py))
    })
}

/// `create_watchlist`
#[pyfunction]
pub fn db_create_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    name: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let r = (|| -> PyResult<Py<PyAny>> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "INSERT INTO watchlists (name, created_at, user_id) VALUES (?, ?, ?)",
            vec![name.clone(), pystr(py, &now_iso(py)?), user_id.clone()],
        )?;
        conn.commit()?;
        let id = rows.lastrowid(py);
        log(
            py,
            "info",
            format!("Created watchlist '{}' with ID {}", s(name)?, s(&id)?),
        )?;
        Ok(id.unbind())
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!("Error creating watchlist '{}': {}", s(name)?, es(py, e)),
        )?;
        Ok(py.None())
    })
}

/// `get_or_create_favorites_watchlist`
#[pyfunction]
pub fn db_get_or_create_favorites<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let m = py.import("db_utils")?;
    for watchlist in m
        .getattr("get_all_watchlists")?
        .call1((db_conn,))?
        .try_iter()?
    {
        let watchlist = watchlist?;
        if m.getattr("is_favorites_name")?
            .call1((watchlist.get_item("name")?,))?
            .is_truthy()?
        {
            return Ok(watchlist.unbind());
        }
    }
    let name = m.getattr("FAVORITES_WATCHLIST_NAME")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("user_id", user_id)?;
    let new_id = m
        .getattr("create_watchlist")?
        .call((db_conn, &name), Some(&kwargs))?;
    if !new_id.is_truthy()? {
        return Ok(py.None());
    }
    let d = PyDict::new(py);
    d.set_item("id", new_id)?;
    d.set_item("name", name)?;
    d.set_item("created_at", now_iso(py)?)?;
    Ok(d.into_any().unbind())
}

/// `get_watchlist_ids_for_symbol`
#[pyfunction]
pub fn db_watchlist_ids_for_symbol<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
) -> PyResult<Py<PyAny>> {
    let r = (|| -> PyResult<Py<PyAny>> {
        let conn = PyConn::new(db_conn)?;
        let wanted = symbol.call_method0("strip")?.call_method0("upper")?;
        let rows = conn.run(
            "SELECT DISTINCT i.watchlist_id FROM watchlist_items i JOIN watchlists w ON w.id = i.watchlist_id WHERE i.symbol = ?",
            vec![wanted],
        )?;
        let builtins = py.import("builtins")?;
        let ids = PyList::empty(py);
        for row in rows.all(py) {
            ids.append(
                builtins
                    .getattr("int")?
                    .call1((row.bind(py).get_item(0)?,))?,
            )?;
        }
        Ok(builtins.getattr("sorted")?.call1((ids,))?.unbind())
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error fetching watchlist membership for {}: {}",
                s(symbol)?,
                es(py, e)
            ),
        )?;
        Ok(PyList::empty(py).into_any().unbind())
    })
}

/// `rename_watchlist`
#[pyfunction]
pub fn db_rename_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    watchlist_id: &Bound<'py, PyAny>,
    new_name: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let _ = user_id;
    let r = (|| -> PyResult<bool> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "UPDATE watchlists SET name = ? WHERE id = ?",
            vec![new_name.clone(), watchlist_id.clone()],
        )?;
        conn.commit()?;
        Ok(rows.rowcount() > 0)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error renaming watchlist {}: {}",
                s(watchlist_id)?,
                es(py, e)
            ),
        )?;
        Ok(false)
    })
}

/// `delete_watchlist`: the list, then its items explicitly (the connections
/// do not enable foreign keys, so ON DELETE CASCADE never fires).
#[pyfunction]
pub fn db_delete_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    watchlist_id: &Bound<'py, PyAny>,
    user_id: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let _ = user_id;
    let r = (|| -> PyResult<bool> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "DELETE FROM watchlists WHERE id = ?",
            vec![watchlist_id.clone()],
        )?;
        let deleted = rows.rowcount() > 0;
        conn.run(
            "DELETE FROM watchlist_items WHERE watchlist_id = ?",
            vec![watchlist_id.clone()],
        )?;
        conn.commit()?;
        Ok(deleted)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error deleting watchlist {}: {}",
                s(watchlist_id)?,
                es(py, e)
            ),
        )?;
        Ok(false)
    })
}

/// `add_to_watchlist`: update the note if the symbol is already listed,
/// insert it otherwise.
#[pyfunction]
pub fn db_add_to_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    note: &Bound<'py, PyAny>,
    watchlist_id: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let r = (|| -> PyResult<bool> {
        let conn = PyConn::new(db_conn)?;
        let found = conn.run(
            "SELECT id FROM watchlist_items WHERE watchlist_id = ? AND symbol = ?",
            vec![watchlist_id.clone(), symbol.clone()],
        )?;
        if found.first(py).is_some() {
            conn.run(
                "UPDATE watchlist_items SET note = ?, added_on = ? WHERE watchlist_id = ? AND symbol = ?",
                vec![note.clone(), pystr(py, &now_iso(py)?), watchlist_id.clone(), symbol.clone()],
            )?;
        } else {
            conn.run(
                "INSERT INTO watchlist_items (watchlist_id, symbol, note, added_on) VALUES (?, ?, ?, ?)",
                vec![watchlist_id.clone(), symbol.clone(), note.clone(), pystr(py, &now_iso(py)?)],
            )?;
        }
        conn.commit()?;
        log(
            py,
            "info",
            format!(
                "Successfully added/updated {} in watchlist {}.",
                s(symbol)?,
                s(watchlist_id)?
            ),
        )?;
        Ok(true)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error adding {} to watchlist {}: {}",
                s(symbol)?,
                s(watchlist_id)?,
                es(py, e)
            ),
        )?;
        Ok(false)
    })
}

/// `remove_from_watchlist`
#[pyfunction]
pub fn db_remove_from_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    symbol: &Bound<'py, PyAny>,
    watchlist_id: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    let r = (|| -> PyResult<bool> {
        let conn = PyConn::new(db_conn)?;
        conn.run(
            "DELETE FROM watchlist_items WHERE watchlist_id = ? AND symbol = ?",
            vec![watchlist_id.clone(), symbol.clone()],
        )?;
        conn.commit()?;
        log(
            py,
            "info",
            format!(
                "Successfully removed {} from watchlist {}.",
                s(symbol)?,
                s(watchlist_id)?
            ),
        )?;
        Ok(true)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error removing {} from watchlist {}: {}",
                s(symbol)?,
                s(watchlist_id)?,
                es(py, e)
            ),
        )?;
        Ok(false)
    })
}

/// `get_watchlist`
#[pyfunction]
pub fn db_get_watchlist<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    watchlist_id: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyList>> {
    let r = (|| -> PyResult<Bound<'py, PyList>> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run(
            "SELECT symbol, note, added_on FROM watchlist_items WHERE watchlist_id = ? ORDER BY added_on DESC",
            vec![watchlist_id.clone()],
        )?;
        let out = PyList::empty(py);
        for row in rows.all(py) {
            let row = row.bind(py);
            let d = PyDict::new(py);
            d.set_item("Symbol", row.get_item(0)?)?;
            d.set_item("Note", row.get_item(1)?)?;
            d.set_item("AddedOn", row.get_item(2)?)?;
            out.append(d)?;
        }
        Ok(out)
    })();
    sqlite_or(py, r, |e| {
        log(
            py,
            "error",
            format!(
                "Error fetching watchlist {}: {}",
                s(watchlist_id)?,
                es(py, e)
            ),
        )?;
        Ok(PyList::empty(py))
    })
}

// --- loading -----------------------------------------------------------------

const LOAD_QUERY: &str = r#"
        SELECT id as original_index, Date, Type, Symbol, Quantity, "Price/Share",
               "Total Amount", Commission, Account, "Split Ratio", Note, "Local Currency", "To Account", Tags, ExternalID, user_id
        FROM transactions
        ORDER BY Date, original_index;
        "#;

/// `load_all_transactions_from_db`: (frame, True), or (None, False) after
/// logging the failure.
#[pyfunction]
pub fn db_load_all_transactions<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    account_currency_map: &Bound<'py, PyAny>,
    default_currency: &Bound<'py, PyAny>,
) -> PyResult<(Py<PyAny>, bool)> {
    log(py, "info", "Loading all transactions from the database.")?;
    let loaded = (|| -> PyResult<Bound<'py, PyAny>> {
        let pd = py.import("pandas")?;
        let np = py.import("numpy")?;
        let parse_dates = PyList::new(py, ["Date"])?;
        let df = crate::pyconn::read_frame(
            db_conn,
            &pystr(py, LOAD_QUERY),
            None,
            Some(parse_dates.as_any()),
        )?;
        let columns = df.getattr("columns")?;
        if columns.contains("Local Currency")? && columns.contains("Account")? {
            let local = df.get_item("Local Currency")?;
            let empties = PyList::new(
                py,
                [
                    py.None().into_bound(py),
                    np.getattr("nan")?,
                    pd.getattr("NA")?,
                    pystr(py, ""),
                ],
            )?;
            let mut is_empty = local.call_method1("isin", (empties,))?;
            if pd
                .getattr("api")?
                .getattr("types")?
                .call_method1("is_string_dtype", (&local,))?
                .is_truthy()?
            {
                let na_text = local
                    .call_method1("astype", (py.import("builtins")?.getattr("str")?,))?
                    .getattr("str")?
                    .call_method0("upper")?
                    .getattr("str")?
                    .call_method0("strip")?
                    .call_method1("__eq__", ("<NA>",))?;
                is_empty = is_empty.call_method1("__or__", (na_text,))?;
            }
            if is_empty.call_method0("any")?.is_truthy()? {
                let loc = df.getattr("loc")?;
                let fill = loc
                    .get_item((&is_empty, "Account"))?
                    .call_method1("map", (account_currency_map,))?
                    .call_method1("fillna", (default_currency,))?;
                loc.set_item((&is_empty, "Local Currency"), fill)?;
            }
            let upper = df
                .get_item("Local Currency")?
                .getattr("str")?
                .call_method0("upper")?;
            crate::pyhelp::setitem(&df, "Local Currency", upper)?;
        }
        if columns.contains("original_index")? {
            let kwargs = PyDict::new(py);
            kwargs.set_item("errors", "coerce")?;
            let ids = pd
                .call_method(
                    "to_numeric",
                    (df.get_item("original_index")?,),
                    Some(&kwargs),
                )?
                .call_method1("astype", ("Int64",))?;
            crate::pyhelp::setitem(&df, "original_index", ids)?;
        }
        for col in [
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Split Ratio",
        ] {
            if df.getattr("columns")?.contains(col)? {
                let kwargs = PyDict::new(py);
                kwargs.set_item("errors", "coerce")?;
                let values = pd.call_method("to_numeric", (df.get_item(col)?,), Some(&kwargs))?;
                crate::pyhelp::setitem(&df, col, values)?;
            }
        }
        log(
            py,
            "info",
            format!(
                "Successfully loaded {} transactions from the database.",
                df.len()?
            ),
        )?;
        Ok(df)
    })();
    match loaded {
        Ok(df) => Ok((df.unbind(), true)),
        Err(e) => {
            log_exc(
                py,
                "error",
                format!("Error loading transactions from database: {}", es(py, &e)),
                &e,
            )?;
            Ok((py.None(), false))
        }
    }
}

/// `check_if_db_empty_and_csv_exists`
#[pyfunction]
pub fn db_check_empty_and_csv_exists<'py>(
    py: Python<'py>,
    db_conn: &Bound<'py, PyAny>,
    csv_file_path: &Bound<'py, PyAny>,
) -> PyResult<bool> {
    if !db_conn.is_truthy()? {
        return Ok(false);
    }
    let r = (|| -> PyResult<bool> {
        let conn = PyConn::new(db_conn)?;
        let rows = conn.run("SELECT COUNT(*) FROM transactions", vec![])?;
        let count = match rows.first(py) {
            Some(row) => row.bind(py).get_item(0)?,
            None => 0i64.into_pyobject(py)?.into_any(),
        };
        let path = py.import("os")?.getattr("path")?;
        let csv_exists = path.call_method1("exists", (csv_file_path,))?.is_truthy()?
            && path.call_method1("isfile", (csv_file_path,))?.is_truthy()?;
        let zero = 0i64.into_pyobject(py)?;
        if count.eq(&zero)? && csv_exists {
            log(
                py,
                "info",
                format!(
                    "Database is empty and CSV '{}' exists. Migration may be needed.",
                    s(csv_file_path)?
                ),
            )?;
            return Ok(true);
        } else if count.gt(&zero)? {
            log(
                py,
                "info",
                format!(
                    "Database is not empty (contains {} transactions). Migration prompt for '{}' might be skipped.",
                    s(&count)?,
                    s(csv_file_path)?
                ),
            )?;
        } else if !csv_exists {
            log(
                py,
                "info",
                format!(
                    "Migration CSV '{}' does not exist. No migration prompt.",
                    s(csv_file_path)?
                ),
            )?;
        }
        Ok(false)
    })();
    sqlite_or(py, r, |e| {
        log_exc(
            py,
            "error",
            format!("Error checking database count: {}", es(py, e)),
            e,
        )?;
        Ok(false)
    })
}
