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

// --- paths, connections, schema ------------------------------------------------

fn config(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("config")
}

fn dbu(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    py.import("db_utils")
}

fn os_path(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("os")?.getattr("path")
}

/// `get_database_path`: the app data dir's copy if it exists, then the
/// working directory's, else the app data path (created later).
#[pyfunction]
pub fn db_get_database_path<'py>(
    py: Python<'py>,
    db_filename: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let path = os_path(py)?;
    let cfg = config(py)?;
    let app_data_dir = path.call_method1(
        "join",
        (
            cfg.getattr("get_app_data_dir")?.call0()?,
            cfg.getattr("DB_DIR")?,
        ),
    )?;
    let app_data_path = path.call_method1("join", (&app_data_dir, db_filename))?;
    if path
        .call_method1("exists", (&app_data_path,))?
        .is_truthy()?
    {
        return Ok(app_data_path);
    }
    let cwd_path = path.call_method1(
        "join",
        (py.import("os")?.call_method0("getcwd")?, db_filename),
    )?;
    if path.call_method1("exists", (&cwd_path,))?.is_truthy()? {
        return Ok(cwd_path);
    }
    Ok(app_data_path)
}

/// `get_global_screener_db_path`
#[pyfunction]
pub fn db_get_global_screener_db_path(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let path = os_path(py)?;
    let cfg = config(py)?;
    let dir = path.call_method1(
        "join",
        (
            cfg.getattr("get_app_data_dir")?.call0()?,
            cfg.getattr("SCREENER_DIR")?,
        ),
    )?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("exist_ok", true)?;
    py.import("os")?
        .call_method("makedirs", (&dir,), Some(&kwargs))?;
    path.call_method1("join", (dir, cfg.getattr("SCREENER_DB_FILENAME")?))
}

const SCREENER_SCHEMA: &str = "
        CREATE TABLE IF NOT EXISTS screener_cache (
            symbol TEXT NOT NULL,
            name TEXT,
            price REAL,
            intrinsic_value REAL,
            margin_of_safety REAL,
            pe_ratio REAL,
            market_cap REAL,
            sector TEXT,
            ai_moat REAL,
            ai_financial_strength REAL,
            ai_predictability REAL,
            ai_growth REAL,
            ai_moat_analysis TEXT,
            ai_financial_strength_analysis TEXT,
            ai_predictability_analysis TEXT,
            ai_growth_perspective_analysis TEXT,
            ai_summary TEXT,
            ai_sentiment REAL,
            ai_catalysts TEXT,
            last_fiscal_year_end INTEGER,
            most_recent_quarter INTEGER,
            universe TEXT NOT NULL DEFAULT 'manual',
            updated_at TEXT NOT NULL,
            valuation_details TEXT,
            user_id INTEGER,
            PRIMARY KEY (symbol, universe)
        );
        ";

const ANALYSIS_COLUMNS: [&str; 4] = [
    "ai_moat_analysis TEXT",
    "ai_financial_strength_analysis TEXT",
    "ai_predictability_analysis TEXT",
    "ai_growth_perspective_analysis TEXT",
];

/// `_ensure_screener_schema`
#[pyfunction]
pub fn db_ensure_screener_schema<'py>(py: Python<'py>, conn: &Bound<'py, PyAny>) -> PyResult<()> {
    let pc = PyConn::new(conn)?;
    pc.run(SCREENER_SCHEMA, vec![])?;
    pc.run(
        "CREATE INDEX IF NOT EXISTS idx_screener_cache_universe ON screener_cache(universe);",
        vec![],
    )?;
    for col_def in ANALYSIS_COLUMNS {
        let col_name = col_def.split_whitespace().next().unwrap_or_default();
        match PyConn::new(conn)?.run(
            &format!("ALTER TABLE screener_cache ADD COLUMN {col_def};"),
            vec![],
        ) {
            Ok(_) => {}
            Err(e) if crate::pyhelp::is_sqlite(py, &e, "OperationalError")? => {
                if !es(py, &e).to_lowercase().contains("duplicate column name") {
                    log(
                        py,
                        "warning",
                        format!("Could not add {col_name} to screener_cache: {}", es(py, &e)),
                    )?;
                }
            }
            Err(e) => return Err(e),
        }
    }
    pc.commit()
}

/// `open_screener_db_conn`: a fresh connection to the global screener DB,
/// whose schema is ensured once per process.
#[pyfunction]
pub fn db_open_screener_db_conn(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let m = dbu(py)?;
    let path = m.getattr("get_global_screener_db_path")?.call0()?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("use_cache", false)?;
    let conn = m
        .getattr("get_db_connection")?
        .call((path,), Some(&kwargs))?;
    if conn.is_none() {
        return Ok(conn);
    }
    if !m.getattr("_SCREENER_SCHEMA_INITED")?.is_truthy()? {
        match m.getattr("_ensure_screener_schema")?.call1((&conn,)) {
            Ok(_) => m.setattr("_SCREENER_SCHEMA_INITED", true)?,
            Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
                log(
                    py,
                    "warning",
                    format!("Could not ensure screener schema: {}", es(py, &e)),
                )?;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(conn)
}

/// `is_path_on_cloud_drive`
#[pyfunction]
pub fn db_is_path_on_cloud_drive(path: &Bound<'_, PyAny>) -> PyResult<bool> {
    if !path.is_truthy()? {
        return Ok(false);
    }
    let lower: String = path.call_method0("lower")?.extract()?;
    Ok([
        "cloudstorage",
        "googledrive",
        "google drive",
        "dropbox",
        "onedrive",
        "icloud",
    ]
    .iter()
    .any(|i| lower.contains(i)))
}

/// Python's `str(float)` for a retry delay (Rust prints 1.0 as "1").
fn py_float_str(py: Python<'_>, x: f64) -> PyResult<String> {
    s(&PyFloat::new(py, x).into_any())
}

/// `get_db_connection`: a connection with Investa's PRAGMAs, cached per
/// thread, retried on a locked or busy file.
#[pyfunction]
pub fn db_get_db_connection<'py>(
    py: Python<'py>,
    db_path: &Bound<'py, PyAny>,
    check_same_thread: &Bound<'py, PyAny>,
    use_cache: &Bound<'py, PyAny>,
    timeout: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = dbu(py)?;
    let db_path = if db_path.is_none() {
        m.getattr("get_database_path")?.call0()?
    } else {
        db_path.clone()
    };
    let cache_key = PyTuple::new(py, [&db_path, check_same_thread])?;
    let local = m.getattr("_DB_CONN_CACHE")?;
    if !local.hasattr("connections")? {
        local.setattr("connections", PyDict::new(py))?;
    }
    let use_cache_b = use_cache.is_truthy()?;
    if use_cache_b && local.getattr("connections")?.contains(&cache_key)? {
        let connections = local.getattr("connections")?;
        match PyConn::new(&connections.get_item(&cache_key)?)
            .and_then(|c| c.run("SELECT 1", vec![]))
        {
            Ok(_) => return connections.get_item(&cache_key),
            Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
                connections.del_item(&cache_key)?;
            }
            Err(e) => return Err(e),
        }
    }
    let attempt = (|| -> PyResult<Bound<'py, PyAny>> {
        let path = os_path(py)?;
        let db_dir = path.call_method1("dirname", (&db_path,))?;
        if db_dir.is_truthy()? {
            let kwargs = PyDict::new(py);
            kwargs.set_item("exist_ok", true)?;
            py.import("os")?
                .call_method("makedirs", (&db_dir,), Some(&kwargs))?;
        }
        let sqlite3 = py.import("sqlite3")?;
        let retries = 5;
        let mut conn = py.None().into_bound(py);
        for i in 0..retries {
            let opened = (|| -> PyResult<Bound<'py, PyAny>> {
                let kwargs = PyDict::new(py);
                kwargs.set_item("timeout", timeout)?;
                kwargs.set_item("check_same_thread", check_same_thread)?;
                let c = sqlite3.call_method("connect", (&db_path,), Some(&kwargs))?;
                conn = c.clone();
                let pc = PyConn::new(&c)?;
                pc.run("PRAGMA busy_timeout = 30000;", vec![])?;
                pc.run("PRAGMA foreign_keys = ON;", vec![])?;
                let modes = (|| -> PyResult<()> {
                    let pc = PyConn::new(&c)?;
                    let row = pc.run("PRAGMA journal_mode;", vec![])?;
                    let current: String = row
                        .first(py)
                        .ok_or_else(|| {
                            pyo3::exceptions::PyTypeError::new_err(
                                "'NoneType' object is not subscriptable",
                            )
                        })?
                        .bind(py)
                        .get_item(0)?
                        .call_method0("upper")?
                        .extract()?;
                    let is_cloud = m
                        .getattr("is_path_on_cloud_drive")?
                        .call1((&db_path,))?
                        .is_truthy()?;
                    if is_cloud {
                        if current != "WAL" {
                            log(
                                py,
                                "info",
                                format!(
                                    "Cloud drive detected for {}. Using safe journal mode (DELETE).",
                                    s(&db_path)?
                                ),
                            )?;
                            pc.run("PRAGMA journal_mode=DELETE;", vec![])?;
                        }
                        pc.run("PRAGMA synchronous=FULL;", vec![])?;
                    } else {
                        if current != "WAL" {
                            pc.run("PRAGMA journal_mode=WAL;", vec![])?;
                        }
                        pc.run("PRAGMA synchronous=NORMAL;", vec![])?;
                    }
                    Ok(())
                })();
                match modes {
                    Ok(()) => {}
                    Err(e) if crate::pyhelp::is_sqlite(py, &e, "OperationalError")? => {
                        if es(py, &e).to_lowercase().contains("database is locked") {
                            log(
                                py,
                                "warning",
                                format!(
                                    "Could not set PRAGMA journal_mode/synchronous on {} (locked). Continuing with current settings.",
                                    s(&db_path)?
                                ),
                            )?;
                        } else {
                            return Err(e);
                        }
                    }
                    Err(e) => return Err(e),
                }
                Ok(c)
            })();
            match opened {
                Ok(_) => break,
                Err(e) if crate::pyhelp::is_sqlite(py, &e, "OperationalError")? => {
                    if i < retries - 1 {
                        let delay = 0.2 * f64::from(i + 1);
                        log(
                            py,
                            "warning",
                            format!(
                                "Database access issue ({}) for {}. Retrying in {}s... (attempt {}/{})",
                                es(py, &e),
                                s(&db_path)?,
                                py_float_str(py, delay)?,
                                i + 1,
                                retries
                            ),
                        )?;
                        py.import("time")?.call_method1("sleep", (delay,))?;
                        continue;
                    }
                    log(
                        py,
                        "error",
                        format!(
                            "Failed to connect to database after {retries} attempts: {} - Error: {}",
                            s(&db_path)?,
                            es(py, &e)
                        ),
                    )?;
                    return Err(e);
                }
                Err(e) => return Err(e),
            }
        }
        if conn.is_truthy()? {
            if use_cache_b {
                local.getattr("connections")?.set_item(&cache_key, &conn)?;
            }
            log(
                py,
                "debug",
                format!(
                    "Successfully connected to database ({}): {} (check_same_thread={})",
                    if use_cache_b { "cached" } else { "fresh" },
                    s(&db_path)?,
                    s(check_same_thread)?
                ),
            )?;
            return Ok(conn);
        }
        Ok(py.None().into_bound(py))
    })();
    match attempt {
        Ok(c) => Ok(c),
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            log_exc(
                py,
                "error",
                format!(
                    "Error connecting to database at {}: {}",
                    s(&db_path)?,
                    es(py, &e)
                ),
                &e,
            )?;
            Ok(py.None().into_bound(py))
        }
        Err(e) if e.is_instance_of::<pyo3::exceptions::PyOSError>(py) => {
            log_exc(
                py,
                "error",
                format!(
                    "OS error setting up database path {}: {}",
                    s(&db_path)?,
                    es(py, &e)
                ),
                &e,
            )?;
            Ok(py.None().into_bound(py))
        }
        Err(e) => Err(e),
    }
}

const TRANSACTIONS_SCHEMA: &str = r#"
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT NOT NULL,
        Type TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        Quantity REAL,
        "Price/Share" REAL,
        "Total Amount" REAL,
        Commission REAL,
        Account TEXT NOT NULL,
        "Split Ratio" REAL,
        Note TEXT,
        "Local Currency" TEXT NOT NULL,
        "To Account" TEXT,
        "Tags" TEXT,
        "ExternalID" TEXT
    );
    "#;

const VERSION_SCHEMA: &str = "
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_on TEXT NOT NULL
    );
    ";

const STAMP_SQL: &str = "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)";

/// One migration step's failure, as the original's `except` clauses read it.
enum Step {
    Ok,
    Duplicate,
}

/// Run `sql`; an OperationalError whose text holds "duplicate column name"
/// (compared as the original does, optionally lower-cased) is `Duplicate`,
/// anything else raises.
fn alter(
    py: Python<'_>,
    pc: &PyConn<'_>,
    sql: &str,
    lower: bool,
    any_error: bool,
) -> PyResult<Step> {
    match pc.run(sql, vec![]) {
        Ok(_) => Ok(Step::Ok),
        Err(e) => {
            let class = if any_error {
                "Error"
            } else {
                "OperationalError"
            };
            if crate::pyhelp::is_sqlite(py, &e, class)? {
                let text = es(py, &e);
                let text = if lower { text.to_lowercase() } else { text };
                if text.contains("duplicate column name") {
                    return Ok(Step::Duplicate);
                }
            }
            Err(e)
        }
    }
}

fn stamp(py: Python<'_>, pc: &PyConn<'_>, version: i64) -> PyResult<()> {
    pc.run(
        STAMP_SQL,
        vec![
            version.into_pyobject(py)?.into_any(),
            pystr(py, &now_iso(py)?),
        ],
    )?;
    Ok(())
}

/// The migrations of `create_transactions_table`, inside its `try`.
fn migrate(py: Python<'_>, conn: &Bound<'_, PyAny>, is_new: bool) -> PyResult<()> {
    let m = dbu(py)?;
    let current_version: i64 = m.getattr("DB_SCHEMA_VERSION")?.extract()?;
    let pc = PyConn::new(conn)?;
    pc.run(TRANSACTIONS_SCHEMA, vec![])?;
    if is_new {
        let receipts = py
            .import("server.transaction_writes")?
            .getattr("RECEIPTS_SCHEMA")?;
        pc.execute(&receipts, PyTuple::empty(py).as_any())?;
    }
    pc.run(VERSION_SCHEMA, vec![])?;
    let row = pc.run(
        "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1",
        vec![],
    )?;
    let db_version: i64 = match row.first(py) {
        Some(r) => r.bind(py).get_item(0)?.extract()?,
        None => 0,
    };

    if db_version < 1 {
        stamp(py, &pc, current_version)?;
        log(
            py,
            "info",
            format!("Initialized database schema to version {current_version}."),
        )?;
    }
    if db_version < 2 {
        log(
            py,
            "info",
            "Schema version is less than 2. Applying 'To Account' column migration.",
        )?;
        match alter(
            py,
            &pc,
            "ALTER TABLE transactions ADD COLUMN \"To Account\" TEXT;",
            false,
            false,
        )? {
            Step::Ok => log(
                py,
                "info",
                "Added 'To Account' column to transactions table.",
            )?,
            Step::Duplicate => log(py, "info", "'To Account' column already exists.")?,
        }
        let moved = pc.run(
            "UPDATE transactions SET \"To Account\" = SUBSTR(Note, INSTR(Note, ':') + 2) WHERE Type = 'Transfer' AND Note LIKE 'To:%' AND \"To Account\" IS NULL;",
            vec![],
        )?;
        log(
            py,
            "info",
            format!(
                "Migrated {} existing transfer transactions to use the 'To Account' column.",
                moved.rowcount()
            ),
        )?;
        stamp(py, &pc, current_version)?;
    }
    pc.run(
        "
        CREATE TABLE IF NOT EXISTS watchlist (
            Symbol TEXT PRIMARY KEY,
            Note TEXT,
            AddedOn TEXT NOT NULL
        );
        ",
        vec![],
    )?;
    if db_version < 3 {
        log(
            py,
            "info",
            "Schema version is less than 3. Watchlist table created (if not exists).",
        )?;
        stamp(py, &pc, current_version)?;
    }
    if db_version < 4 {
        log(
            py,
            "info",
            "Schema version is less than 4. Applying 'Tags' column migration.",
        )?;
        match alter(
            py,
            &pc,
            "ALTER TABLE transactions ADD COLUMN Tags TEXT;",
            false,
            false,
        )? {
            Step::Ok => log(py, "info", "Added 'Tags' column to transactions table.")?,
            Step::Duplicate => log(py, "info", "'Tags' column already exists.")?,
        }
    }
    if db_version < 5 {
        log(
            py,
            "info",
            "Schema version is less than 5. Applying Multiple Watchlists migration.",
        )?;
        let v5 = (|| -> PyResult<()> {
            pc.run(
                "
                    CREATE TABLE IF NOT EXISTS watchlists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                ",
                vec![],
            )?;
            let count = pc.run("SELECT count(*) FROM watchlists", vec![])?;
            let n: i64 = count
                .first(py)
                .map(|r| r.bind(py).get_item(0))
                .transpose()?
                .map(|v| v.extract())
                .transpose()?
                .unwrap_or(0);
            let default_id = if n == 0 {
                let made = pc.run(
                    "INSERT INTO watchlists (name, created_at) VALUES (?, ?)",
                    vec![pystr(py, "My Watchlist"), pystr(py, &now_iso(py)?)],
                )?;
                made.lastrowid(py)
            } else {
                let first = pc.run("SELECT id FROM watchlists LIMIT 1", vec![])?;
                first.first(py).expect("a row").bind(py).get_item(0)?
            };
            pc.run(
                "
                    CREATE TABLE IF NOT EXISTS watchlist_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        watchlist_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        note TEXT,
                        added_on TEXT NOT NULL,
                        FOREIGN KEY(watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                    );
                ",
                vec![],
            )?;
            let old = pc.run(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist';",
                vec![],
            )?;
            if old.first(py).is_some() {
                log(py, "info", "Migrating existing watchlist items...")?;
                let rows = pc.run("SELECT Symbol, Note, AddedOn FROM watchlist", vec![])?;
                let items = rows.all(py);
                for r in &items {
                    let r = r.bind(py);
                    pc.run(
                        "INSERT INTO watchlist_items (watchlist_id, symbol, note, added_on) VALUES (?, ?, ?, ?)",
                        vec![default_id.clone(), r.get_item(0)?, r.get_item(1)?, r.get_item(2)?],
                    )?;
                }
                pc.run("DROP TABLE watchlist", vec![])?;
                log(
                    py,
                    "info",
                    format!(
                        "Migrated {} items and dropped old watchlist table.",
                        items.len()
                    ),
                )?;
            }
            Ok(())
        })();
        if let Err(e) = v5 {
            if crate::pyhelp::is_sqlite(py, &e, "Error")? {
                log(
                    py,
                    "error",
                    format!("Error during migration v5: {}", es(py, &e)),
                )?;
            }
            return Err(e);
        }
        log(py, "info", "Updating schema_version to 5...")?;
        stamp(py, &pc, 5)?;
        log(py, "info", "Updated schema_version to 5 (pending commit).")?;
    }
    if db_version < 6 {
        log(
            py,
            "info",
            "Schema version is less than 6. Creating screener_cache table.",
        )?;
        let v6 = (|| -> PyResult<()> {
            pc.run(
                "
                    CREATE TABLE IF NOT EXISTS screener_cache (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        price REAL,
                        intrinsic_value REAL,
                        margin_of_safety REAL,
                        pe_ratio REAL,
                        market_cap REAL,
                        sector TEXT,
                        ai_moat REAL,
                        ai_financial_strength REAL,
                        ai_predictability REAL,
                        ai_growth REAL,
                        ai_summary TEXT,
                        last_fiscal_year_end INTEGER,
                        most_recent_quarter INTEGER,
                        updated_at TEXT NOT NULL
                    );
                ",
                vec![],
            )?;
            stamp(py, &pc, 6)?;
            log(py, "info", "Updated schema_version to 6.")
        })();
        logged_raise(py, v6, "v6")?;
    }
    if db_version < 7 {
        log(
            py,
            "info",
            "Schema version is less than 7. Adding 'universe' column to screener_cache.",
        )?;
        let v7 = (|| -> PyResult<()> {
            pc.run(
                "ALTER TABLE screener_cache ADD COLUMN universe TEXT;",
                vec![],
            )?;
            stamp(py, &pc, 7)?;
            log(py, "info", "Updated schema_version to 7.")
        })();
        duplicate_or_raise(py, v7, "'universe' column already exists.", "v7")?;
    }
    if db_version < 8 {
        log(
            py,
            "info",
            "Schema version is less than 8. Migrating screener_cache to composite PK (symbol, universe).",
        )?;
        let v8 = (|| -> PyResult<()> {
            pc.run(
                "ALTER TABLE screener_cache RENAME TO screener_cache_old;",
                vec![],
            )?;
            pc.run(
                "
                    CREATE TABLE screener_cache (
                        symbol TEXT NOT NULL,
                        name TEXT,
                        price REAL,
                        intrinsic_value REAL,
                        margin_of_safety REAL,
                        pe_ratio REAL,
                        market_cap REAL,
                        sector TEXT,
                        ai_moat REAL,
                        ai_financial_strength REAL,
                        ai_predictability REAL,
                        ai_growth REAL,
                        ai_summary TEXT,
                        last_fiscal_year_end INTEGER,
                        most_recent_quarter INTEGER,
                        universe TEXT NOT NULL DEFAULT 'manual',
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, universe)
                    );
                ",
                vec![],
            )?;
            pc.run(
                "
                    INSERT INTO screener_cache (
                        symbol, name, price, intrinsic_value, margin_of_safety,
                        pe_ratio, market_cap, sector, ai_moat, ai_financial_strength,
                        ai_predictability, ai_growth, ai_summary, last_fiscal_year_end,
                        most_recent_quarter, universe, updated_at
                    )
                    SELECT 
                        symbol, name, price, intrinsic_value, margin_of_safety,
                        pe_ratio, market_cap, sector, ai_moat, ai_financial_strength,
                        ai_predictability, ai_growth, ai_summary, last_fiscal_year_end,
                        most_recent_quarter, COALESCE(universe, 'manual'), updated_at
                    FROM screener_cache_old;
                ",
                vec![],
            )?;
            pc.run("DROP TABLE screener_cache_old;", vec![])?;
            stamp(py, &pc, 8)?;
            log(py, "info", "Updated schema_version to 8.")
        })();
        logged_raise(py, v8, "v8")?;
    }
    if db_version < 9 {
        log(
            py,
            "info",
            "Schema version is less than 9. Adding 'valuation_details' column to screener_cache.",
        )?;
        let v9 = (|| -> PyResult<()> {
            pc.run(
                "ALTER TABLE screener_cache ADD COLUMN valuation_details TEXT;",
                vec![],
            )?;
            stamp(py, &pc, 9)?;
            log(py, "info", "Updated schema_version to 9.")
        })();
        duplicate_or_raise(py, v9, "'valuation_details' column already exists.", "v9")?;
    }
    if db_version < 10 {
        log(
            py,
            "info",
            "Schema version is less than 10. Adding users table and user_id column.",
        )?;
        for table in ["transactions", "watchlists", "screener_cache"] {
            alter(
                py,
                &pc,
                &format!("ALTER TABLE {table} ADD COLUMN user_id INTEGER;"),
                false,
                false,
            )?;
        }
        stamp(py, &pc, 10)?;
        log(py, "info", "Updated schema_version to 10.")?;
    }
    if db_version < 11 {
        log(
            py,
            "info",
            "Schema version is less than 11. Adding 'ExternalID' column to transactions.",
        )?;
        let v11 = (|| -> PyResult<()> {
            pc.run(
                "ALTER TABLE transactions ADD COLUMN ExternalID TEXT;",
                vec![],
            )?;
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_external_id ON transactions (ExternalID);",
                vec![],
            )?;
            stamp(py, &pc, 11)?;
            log(py, "info", "Updated schema_version to 11.")
        })();
        duplicate_or_raise(py, v11, "'ExternalID' column already exists.", "v11")?;
    }
    if db_version < 12 {
        log(
            py,
            "info",
            "Schema version is less than 12. Creating 'pending_transactions' table.",
        )?;
        let v12 = (|| -> PyResult<()> {
            pc.run(
                r#"
                    CREATE TABLE IF NOT EXISTS pending_transactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        Date TEXT NOT NULL,
                        Type TEXT NOT NULL,
                        Symbol TEXT NOT NULL,
                        Quantity REAL,
                        "Price/Share" REAL,
                        "Total Amount" REAL,
                        Commission REAL,
                        Account TEXT NOT NULL,
                        "Split Ratio" REAL,
                        Note TEXT,
                        "Local Currency" TEXT NOT NULL,
                        "To Account" TEXT,
                        "Tags" TEXT,
                        "ExternalID" TEXT,
                        user_id INTEGER
                    );
                "#,
                vec![],
            )?;
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_pending_external_id ON pending_transactions (ExternalID);",
                vec![],
            )?;
            stamp(py, &pc, 12)?;
            log(py, "info", "Updated schema_version to 12.")
        })();
        logged_raise(py, v12, "v12")?;
    }
    if db_version < 13 {
        log(
            py,
            "info",
            "Schema version is less than 13. Creating indexes and portfolio_snapshots table.",
        )?;
        let v13 = (|| -> PyResult<()> {
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_transactions_symbol ON transactions (Symbol);",
                vec![],
            )?;
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions (Date);",
                vec![],
            )?;
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_transactions_account ON transactions (Account);",
                vec![],
            )?;
            pc.run(
                "
                    CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        snapshot_date TEXT NOT NULL,
                        account TEXT NOT NULL,
                        total_value REAL NOT NULL,
                        total_cost REAL,
                        total_gain REAL,
                        total_return_pct REAL,
                        twr REAL,
                        irr REAL,
                        created_at TEXT NOT NULL
                    );
                ",
                vec![],
            )?;
            pc.run(
                "CREATE INDEX IF NOT EXISTS idx_portfolio_snapshots_date_account ON portfolio_snapshots (snapshot_date, account);",
                vec![],
            )?;
            stamp(py, &pc, 13)?;
            log(py, "info", "Updated schema_version to 13.")
        })();
        logged_raise(py, v13, "v13")?;
    }
    if db_version < 14 {
        log(
            py,
            "info",
            "Schema version is less than 14. Adding 'ai_sentiment' and 'ai_catalysts' to screener_cache.",
        )?;
        let v14 = (|| -> PyResult<()> {
            pc.run(
                "ALTER TABLE screener_cache ADD COLUMN ai_sentiment REAL;",
                vec![],
            )?;
            pc.run(
                "ALTER TABLE screener_cache ADD COLUMN ai_catalysts TEXT;",
                vec![],
            )?;
            stamp(py, &pc, 14)?;
            log(py, "info", "Updated schema_version to 14.")
        })();
        logged_raise(py, v14, "v14")?;
    }
    if db_version < 15 {
        log(
            py,
            "info",
            "Schema version is less than 15. Adding AI per-dimension analysis text columns to screener_cache.",
        )?;
        for col_def in ANALYSIS_COLUMNS {
            match pc.run(
                &format!("ALTER TABLE screener_cache ADD COLUMN {col_def};"),
                vec![],
            ) {
                Ok(_) => {}
                Err(e) if crate::pyhelp::is_sqlite(py, &e, "OperationalError")? => {
                    if !es(py, &e).to_lowercase().contains("duplicate column name") {
                        log(
                            py,
                            "error",
                            format!("Error during migration v15 ({col_def}): {}", es(py, &e)),
                        )?;
                        return Err(e);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        stamp(py, &pc, 15)?;
        log(py, "info", "Updated schema_version to 15.")?;
    }
    pc.commit()?;
    log(
        py,
        "info",
        "Transactions, watchlist, and schema_version tables checked/created/updated successfully.",
    )
}

/// `except sqlite3.Error as e: logging.error(f"Error during migration {v}: {e}"); raise`
fn logged_raise(py: Python<'_>, r: PyResult<()>, version: &str) -> PyResult<()> {
    if let Err(e) = r {
        if crate::pyhelp::is_sqlite(py, &e, "Error")? {
            log(
                py,
                "error",
                format!("Error during migration {version}: {}", es(py, &e)),
            )?;
        }
        return Err(e);
    }
    Ok(())
}

/// The v7/v9/v11 handler: a duplicate column is noted, any other sqlite
/// error logged and raised.
fn duplicate_or_raise(py: Python<'_>, r: PyResult<()>, note: &str, version: &str) -> PyResult<()> {
    match r {
        Ok(()) => Ok(()),
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            if es(py, &e).contains("duplicate column name") {
                log(py, "info", note)
            } else {
                log(
                    py,
                    "error",
                    format!("Error during migration {version}: {}", es(py, &e)),
                )?;
                Err(e)
            }
        }
        Err(e) => Err(e),
    }
}

/// `create_transactions_table`
#[pyfunction]
pub fn db_create_transactions_table<'py>(
    py: Python<'py>,
    conn: &Bound<'py, PyAny>,
) -> PyResult<()> {
    let is_new = PyConn::new(conn)?
        .run(
            "SELECT 1 FROM sqlite_master WHERE name='transactions'",
            vec![],
        )?
        .first(py)
        .is_none();
    match migrate(py, conn, is_new) {
        Ok(()) => Ok(()),
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "Error")? => {
            log_exc(
                py,
                "error",
                format!("Error creating/updating tables: {}", es(py, &e)),
                &e,
            )?;
            conn.call_method0("rollback").map(|_| ())
        }
        Err(e) => Err(e),
    }
}

/// `initialize_database`
#[pyfunction]
pub fn db_initialize_database<'py>(
    py: Python<'py>,
    db_path: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let m = dbu(py)?;
    let conn = m.getattr("get_db_connection")?.call1((db_path,))?;
    if conn.is_truthy()? {
        m.getattr("create_transactions_table")?.call1((&conn,))?;
    }
    Ok(conn)
}

/// `initialize_global_database`: the users table in the global DB.
#[pyfunction]
pub fn db_initialize_global_database(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    let cfg = config(py)?;
    let path = os_path(py)?.call_method1(
        "join",
        (
            cfg.getattr("get_app_data_dir")?.call0()?,
            cfg.getattr("DB_DIR")?,
            cfg.getattr("GLOBAL_DB_FILENAME")?,
        ),
    )?;
    let conn = dbu(py)?.getattr("get_db_connection")?.call1((path,))?;
    if conn.is_truthy()? {
        let pc = PyConn::new(&conn)?;
        pc.run(
            "
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                hashed_password TEXT NOT NULL,
                alias TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_at TEXT NOT NULL
            );
        ",
            vec![],
        )?;
        match alter(
            py,
            &pc,
            "ALTER TABLE users ADD COLUMN alias TEXT;",
            false,
            false,
        )? {
            Step::Ok => log(py, "info", "Added 'alias' column to users table.")?,
            Step::Duplicate => {}
        }
        pc.commit()?;
    }
    Ok(conn)
}

/// `connect_readonly`: `mode=ro`, or `query_only` where a WAL file has no
/// `-shm` yet and mode=ro cannot create it.
#[pyfunction]
pub fn db_connect_readonly<'py>(
    py: Python<'py>,
    db_path: &Bound<'py, PyAny>,
    timeout: &Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let sqlite3 = py.import("sqlite3")?;
    let strict = (|| -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("uri", true)?;
        kwargs.set_item("timeout", timeout)?;
        let conn = sqlite3.call_method(
            "connect",
            (format!("file:{}?mode=ro", s(db_path)?),),
            Some(&kwargs),
        )?;
        PyConn::new(&conn)?.run("SELECT 1", vec![])?;
        Ok(conn)
    })();
    match strict {
        Ok(conn) => Ok(conn),
        Err(e) if crate::pyhelp::is_sqlite(py, &e, "OperationalError")? => {
            let kwargs = PyDict::new(py);
            kwargs.set_item("timeout", timeout)?;
            let conn = sqlite3.call_method("connect", (db_path,), Some(&kwargs))?;
            PyConn::new(&conn)?.run("PRAGMA query_only = 1", vec![])?;
            Ok(conn)
        }
        Err(e) => Err(e),
    }
}
