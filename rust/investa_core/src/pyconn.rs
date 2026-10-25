//! A Python `sqlite3.Connection` driven from Rust.
//!
//! The storage modules' Rust code (and `src/rust_sql.py`) run statements
//! through `PyConn`, which does what Python's `execute` does before a
//! statement and then runs it with `sql::Statement` on the connection's own
//! `sqlite3*`:
//! - the thread and closed checks, by calling `conn.cursor()`, which raises
//!   exactly what `execute` raises;
//! - positional and named binding, with CPython 3.12's errors;
//! - the implicit BEGIN;
//! - each error raised as the class, message and code Python gives it.
//!
//! A connection Rust must not touch (a `row_factory`, a custom
//! `text_factory`, explicit autocommit control, an adapter on a base type, a
//! layout it has not verified) and any parameter it does not bind exactly as
//! Python would go through the connection's own Python methods instead, so
//! the behaviour is the same either way.

use std::sync::OnceLock;

use pyo3::exceptions::PyStopIteration;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple, PyType};

use crate::sql::{Failure, Outcome, Statement, Value};

static LAYOUT_OK: OnceLock<bool> = OnceLock::new();

/// `rust_sql.bound()`: investa_core reaches Python's SQLite and a
/// connection's handle sits where it is read. Asked once per process.
fn layout_ok(py: Python<'_>) -> PyResult<bool> {
    if let Some(ok) = LAYOUT_OK.get() {
        return Ok(*ok);
    }
    let ok: bool = py.import("rust_sql")?.call_method0("bound")?.extract()?;
    Ok(*LAYOUT_OK.get_or_init(|| ok))
}

enum Route {
    Rust { db: usize, begin: Option<String> },
    Python,
}

pub struct PyConn<'py> {
    py: Python<'py>,
    obj: Bound<'py, PyAny>,
    route: Route,
}

/// A parameter as Python's sqlite3 binds it without adapting it, or None.
pub fn bindable(v: &Bound<'_, PyAny>) -> Option<Value> {
    if v.is_none() {
        return Some(Value::Null);
    }
    if v.is_exact_instance_of::<PyBool>() {
        return Some(Value::Int(i64::from(v.is_truthy().ok()?)));
    }
    if v.is_exact_instance_of::<PyInt>() {
        return v.extract::<i64>().ok().map(Value::Int);
    }
    if v.is_instance_of::<PyFloat>() {
        if !v.is_exact_instance_of::<PyFloat>() && v.hasattr("__conform__").unwrap_or(true) {
            return None;
        }
        return v.extract::<f64>().ok().map(Value::Real);
    }
    if v.is_exact_instance_of::<PyString>() {
        return v
            .cast::<PyString>()
            .ok()?
            .to_str()
            .ok()
            .map(|s| Value::Text(s.as_bytes().to_vec()));
    }
    if v.is_exact_instance_of::<PyBytes>() {
        return Some(Value::Blob(v.cast::<PyBytes>().ok()?.as_bytes().to_vec()));
    }
    None
}

/// CPython 3.12 `get_exception_class` and `_pysqlite_seterror`.
pub fn sqlite_error(py: Python<'_>, failure: Failure) -> PyErr {
    let raised = (|| -> PyResult<PyErr> {
        let m = py.import("sqlite3")?;
        Ok(match failure {
            Failure::Sqlite { code, message } => {
                let class = match code & 0xFF {
                    2 | 12 => "InternalError",
                    7 => return Ok(pyo3::exceptions::PyMemoryError::new_err(())),
                    1 | 3 | 4 | 5 | 6 | 8 | 9 | 10 | 13 | 14 | 15 | 16 | 17 => "OperationalError",
                    11 => "DatabaseError",
                    18 => "DataError",
                    19 | 20 => "IntegrityError",
                    21 | 25 => "InterfaceError",
                    _ => "DatabaseError",
                };
                let exc = m.getattr(class)?.call1((message,))?;
                exc.setattr("sqlite_errorcode", code)?;
                exc.setattr("sqlite_errorname", error_name(py, code)?)?;
                PyErr::from_value(exc)
            }
            Failure::Programming(message) => {
                PyErr::from_value(m.getattr("ProgrammingError")?.call1((message,))?)
            }
            Failure::Decline(why) => PyErr::from_value(m.getattr("InterfaceError")?.call1((why,))?),
        })
    })();
    raised.unwrap_or_else(|e| e)
}

/// `sqlite_errorname`: the module's name for an error code, "unknown" without one.
fn error_name(py: Python<'_>, code: i32) -> PyResult<String> {
    static NAMES: OnceLock<Vec<(i32, String)>> = OnceLock::new();
    if NAMES.get().is_none() {
        let m = py.import("sqlite3")?;
        let primary = [
            "SQLITE_OK",
            "SQLITE_ERROR",
            "SQLITE_INTERNAL",
            "SQLITE_PERM",
            "SQLITE_ABORT",
            "SQLITE_BUSY",
            "SQLITE_LOCKED",
            "SQLITE_NOMEM",
            "SQLITE_READONLY",
            "SQLITE_INTERRUPT",
            "SQLITE_IOERR",
            "SQLITE_CORRUPT",
            "SQLITE_NOTFOUND",
            "SQLITE_FULL",
            "SQLITE_CANTOPEN",
            "SQLITE_PROTOCOL",
            "SQLITE_EMPTY",
            "SQLITE_SCHEMA",
            "SQLITE_TOOBIG",
            "SQLITE_CONSTRAINT",
            "SQLITE_MISMATCH",
            "SQLITE_MISUSE",
            "SQLITE_NOLFS",
            "SQLITE_AUTH",
            "SQLITE_FORMAT",
            "SQLITE_RANGE",
            "SQLITE_NOTADB",
            "SQLITE_NOTICE",
            "SQLITE_WARNING",
            "SQLITE_ROW",
            "SQLITE_DONE",
        ];
        let mut names = Vec::new();
        for p in primary {
            if let Ok(v) = m.getattr(p).and_then(|v| v.extract::<i32>()) {
                names.push((v, p.to_string()));
            }
        }
        let dir: Vec<String> = m.dir()?.extract()?;
        for n in dir {
            if n.starts_with("SQLITE_")
                && primary.iter().any(|p| n.starts_with(&format!("{p}_")))
                && let Ok(v) = m.getattr(n.as_str()).and_then(|v| v.extract::<i32>())
                && v > 255
            {
                names.push((v, n));
            }
        }
        let _ = NAMES.set(names);
    }
    // Later entries win, as a dict built in the same order would.
    Ok(NAMES
        .get()
        .and_then(|names| {
            names
                .iter()
                .rev()
                .find(|(c, _)| *c == code)
                .map(|(_, n)| n.clone())
        })
        .unwrap_or_else(|| "unknown".to_string()))
}

fn adapters_reach_base_types(py: Python<'_>, sqlite3: &Bound<'_, PyModule>) -> PyResult<bool> {
    let builtins = py.import("builtins")?;
    let bases = PyTuple::new(
        py,
        ["int", "float", "str", "bytes", "bytearray"]
            .iter()
            .map(|n| builtins.getattr(*n))
            .collect::<PyResult<Vec<_>>>()?,
    )?;
    let adapters = sqlite3.getattr("adapters")?;
    for key in adapters.try_iter()? {
        let key = key?;
        let first = key.get_item(0)?;
        if let Ok(t) = first.cast::<PyType>()
            && t.is_subclass(&bases)?
        {
            return Ok(true);
        }
    }
    Ok(false)
}

impl<'py> PyConn<'py> {
    /// The connection as `rust_sql.handle` sees it: the Rust route, or
    /// Python's own methods. Raises what `conn.execute` would raise first
    /// (wrong thread, closed).
    pub fn new(obj: &Bound<'py, PyAny>) -> PyResult<PyConn<'py>> {
        let py = obj.py();
        let python = |obj: &Bound<'py, PyAny>| PyConn {
            py,
            obj: obj.clone(),
            route: Route::Python,
        };
        let sqlite3 = py.import("sqlite3")?;
        if !obj.get_type().is(&sqlite3.getattr("Connection")?) || !layout_ok(py)? {
            return Ok(python(obj));
        }
        obj.call_method0("cursor")?; // Python's thread and closed checks
        let str_type = py.import("builtins")?.getattr("str")?;
        if !obj.getattr("row_factory")?.is_none()
            || !obj.getattr("text_factory")?.is(&str_type)
            || !obj
                .getattr("autocommit")?
                .eq(sqlite3.getattr("LEGACY_TRANSACTION_CONTROL")?)?
            || adapters_reach_base_types(py, &sqlite3)?
        {
            return Ok(python(obj));
        }
        // CPython 3.12 `pysqlite_Connection`: PyObject_HEAD, then `sqlite3 *db`.
        let db = unsafe {
            *(obj
                .as_ptr()
                .cast::<u8>()
                .add(std::mem::size_of::<pyo3::ffi::PyObject>()) as *const usize)
        };
        if db == 0 {
            return Ok(python(obj));
        }
        let level = obj.getattr("isolation_level")?;
        let begin = if level.is_none() {
            None
        } else {
            Some(format!("BEGIN {}", level.extract::<String>()?))
        };
        Ok(PyConn {
            py,
            obj: obj.clone(),
            route: Route::Rust { db, begin },
        })
    }

    pub fn object(&self) -> &Bound<'py, PyAny> {
        &self.obj
    }

    pub fn commit(&self) -> PyResult<()> {
        self.obj.call_method0("commit").map(|_| ())
    }

    pub fn rollback(&self) -> PyResult<()> {
        self.obj.call_method0("rollback").map(|_| ())
    }

    fn via_python(
        &self,
        method: &str,
        sql: &Bound<'py, PyAny>,
        params: &Bound<'py, PyAny>,
    ) -> PyResult<SqlRows> {
        let cursor = self.obj.call_method1(method, (sql, params))?;
        SqlRows::from_cursor(&cursor)
    }

    /// `conn.execute(sql, params)` followed by `fetchall()`.
    pub fn execute(
        &self,
        sql: &Bound<'py, PyAny>,
        params: &Bound<'py, PyAny>,
    ) -> PyResult<SqlRows> {
        let py = self.py;
        let Route::Rust { db, begin } = &self.route else {
            return self.via_python("execute", sql, params);
        };
        let Ok(text) = sql.cast_exact::<PyString>() else {
            return self.via_python("execute", sql, params);
        };
        let text = text.to_str()?.to_owned();
        let (db, begin) = (*db, begin.clone());
        let result = if params.is_exact_instance_of::<PyTuple>()
            || params.is_exact_instance_of::<PyList>()
        {
            let mut values = Vec::new();
            for v in params.try_iter()? {
                match bindable(&v?) {
                    Some(x) => values.push(x),
                    None => return self.via_python("execute", sql, params),
                }
            }
            py.detach(|| {
                let st = Statement::prepare(db, &text)?;
                st.begin(begin.as_deref())?;
                st.bind_positional(&values)?;
                st.run()
            })
        } else if params.is_exact_instance_of::<PyDict>() {
            let dict = params.cast::<PyDict>()?;
            let st = match py.detach(|| {
                let st = Statement::prepare(db, &text)?;
                st.begin(begin.as_deref())?;
                Ok(st)
            }) {
                Ok(st) => st,
                Err(Failure::Decline(_)) => return self.via_python("execute", sql, params),
                Err(f) => return Err(sqlite_error(py, f)),
            };
            let values = match named_values(&st, dict)? {
                Named::Values(v) => v,
                Named::Error(f) => return Err(sqlite_error(py, f)),
                Named::Decline => return self.via_python("execute", sql, params),
            };
            py.detach(|| {
                st.bind_in_order(&values)?;
                st.run()
            })
        } else {
            return self.via_python("execute", sql, params);
        };
        match result {
            Ok(outcome) => SqlRows::from_outcome(py, outcome, true),
            Err(Failure::Decline(_)) => self.via_python("execute", sql, params),
            Err(f) => Err(sqlite_error(py, f)),
        }
    }

    /// `conn.execute` with a Rust-built parameter tuple.
    pub fn run(&self, sql: &str, params: Vec<Bound<'py, PyAny>>) -> PyResult<SqlRows> {
        let sql = PyString::new(self.py, sql).into_any();
        let params = PyTuple::new(self.py, params)?.into_any();
        self.execute(&sql, &params)
    }

    /// `conn.executemany(sql, seq_of_params)`.
    pub fn executemany(
        &self,
        sql: &Bound<'py, PyAny>,
        seq: &Bound<'py, PyAny>,
    ) -> PyResult<SqlRows> {
        let py = self.py;
        let items: Vec<Bound<'py, PyAny>> = seq.try_iter()?.collect::<PyResult<_>>()?;
        let materialised = PyList::new(py, &items)?.into_any();
        let Route::Rust { db, begin } = &self.route else {
            return self.via_python("executemany", sql, &materialised);
        };
        let Ok(text) = sql.cast_exact::<PyString>() else {
            return self.via_python("executemany", sql, &materialised);
        };
        let text = text.to_str()?.to_owned();
        let (db, begin) = (*db, begin.clone());
        let st = match py.detach(|| Statement::prepare(db, &text)) {
            Ok(st) => st,
            Err(Failure::Decline(_)) => return self.via_python("executemany", sql, &materialised),
            Err(f) => return Err(sqlite_error(py, f)),
        };
        if !st.is_dml() {
            return Err(sqlite_error(
                py,
                Failure::Programming("executemany() can only execute DML statements.".into()),
            ));
        }
        // Every item is converted before any is run: one Rust cannot bind
        // sends the whole call to Python while nothing has happened yet. A
        // binding error is Python's at that item, after the earlier ones ran.
        let mut sets: Vec<(Vec<Value>, bool)> = Vec::with_capacity(items.len());
        let mut stop: Option<Failure> = None;
        for item in &items {
            if item.is_exact_instance_of::<PyTuple>() || item.is_exact_instance_of::<PyList>() {
                let mut values = Vec::new();
                for v in item.try_iter()? {
                    match bindable(&v?) {
                        Some(x) => values.push(x),
                        None => return self.via_python("executemany", sql, &materialised),
                    }
                }
                sets.push((values, false));
            } else if item.is_exact_instance_of::<PyDict>() {
                match named_values(&st, item.cast::<PyDict>()?)? {
                    Named::Values(v) => sets.push((v, true)),
                    Named::Error(f) => {
                        stop = Some(f);
                        break;
                    }
                    Named::Decline => return self.via_python("executemany", sql, &materialised),
                }
            } else {
                return self.via_python("executemany", sql, &materialised);
            }
        }
        let result = py.detach(|| -> Result<i64, Failure> {
            st.begin(begin.as_deref())?;
            let mut rowcount = 0;
            for (values, named) in &sets {
                rowcount += st.run_once(values, *named)?;
            }
            match stop {
                Some(f) => Err(f),
                None => Ok(rowcount),
            }
        });
        match result {
            Ok(rowcount) => Ok(SqlRows {
                description: py.None(),
                rowcount,
                lastrowid: py.None(),
                arraysize: 1,
                rows: Vec::new(),
                at: 0,
            }),
            Err(f) => Err(sqlite_error(py, f)),
        }
    }
}

enum Named {
    Values(Vec<Value>),
    Error(Failure),
    Decline,
}

/// CPython 3.12 `bind_parameters` for a dict: each placeholder's name, less
/// its first character, looked up in order.
fn named_values(st: &Statement, dict: &Bound<'_, PyDict>) -> PyResult<Named> {
    let mut values = Vec::new();
    for (i, name) in st.param_names().into_iter().enumerate() {
        let Some(name) = name else {
            return Ok(Named::Error(Failure::Programming(format!(
                "Binding {} has no name, but you supplied a dictionary (which has only names).",
                i + 1
            ))));
        };
        let key = String::from_utf8_lossy(&name[1..]).into_owned();
        match dict.get_item(&key)? {
            None => {
                return Ok(Named::Error(Failure::Programming(format!(
                    "You did not supply a value for binding parameter :{key}."
                ))));
            }
            Some(v) => match bindable(&v) {
                Some(x) => values.push(x),
                None => return Ok(Named::Decline),
            },
        }
    }
    Ok(Named::Values(values))
}

/// What a caller reads from a cursor after `execute`: the rows (all fetched),
/// `description`, `rowcount`, `lastrowid`.
#[pyclass(module = "investa_core")]
pub struct SqlRows {
    #[pyo3(get)]
    description: Py<PyAny>,
    #[pyo3(get)]
    rowcount: i64,
    #[pyo3(get)]
    lastrowid: Py<PyAny>,
    #[pyo3(get, set)]
    arraysize: usize,
    rows: Vec<Py<PyAny>>,
    at: usize,
}

impl SqlRows {
    fn from_outcome(py: Python<'_>, o: Outcome, with_rowid: bool) -> PyResult<SqlRows> {
        use crate::sql::Value as V;
        let description = if o.columns.is_empty() {
            py.None()
        } else {
            let cols: Vec<Bound<'_, PyTuple>> = o
                .columns
                .iter()
                .map(|c| {
                    PyTuple::new(
                        py,
                        [
                            PyString::new(py, c).into_any(),
                            py.None().into_bound(py),
                            py.None().into_bound(py),
                            py.None().into_bound(py),
                            py.None().into_bound(py),
                            py.None().into_bound(py),
                            py.None().into_bound(py),
                        ],
                    )
                })
                .collect::<PyResult<_>>()?;
            PyTuple::new(py, cols)?.into_any().unbind()
        };
        let mut rows = Vec::with_capacity(o.rows.len());
        for row in o.rows {
            let mut cells: Vec<Bound<'_, PyAny>> = Vec::with_capacity(row.len());
            for (c, v) in row.into_iter().enumerate() {
                cells.push(match v {
                    V::Null => py.None().into_bound(py),
                    V::Int(x) => x.into_pyobject(py)?.into_any(),
                    V::Real(x) => x.into_pyobject(py)?.into_any(),
                    V::Blob(b) => PyBytes::new(py, &b).into_any(),
                    V::Text(t) => match String::from_utf8(t) {
                        Ok(s) => s.into_pyobject(py)?.into_any(),
                        Err(e) => {
                            // "Could not decode to UTF-8 column '%s' with text '%s'",
                            // the text cut at its first NUL.
                            let bytes = e.into_bytes();
                            let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
                            let name = o.columns.get(c).cloned().unwrap_or_default();
                            let text = String::from_utf8_lossy(&bytes[..end]);
                            let exc = py.import("sqlite3")?.getattr("OperationalError")?.call1((format!(
                                "Could not decode to UTF-8 column '{name}' with text '{text}'"
                            ),))?;
                            return Err(PyErr::from_value(exc));
                        }
                    },
                });
            }
            rows.push(PyTuple::new(py, cells)?.into_any().unbind());
        }
        Ok(SqlRows {
            description,
            rowcount: o.rowcount,
            lastrowid: if with_rowid {
                o.lastrowid.into_pyobject(py)?.into_any().unbind()
            } else {
                py.None()
            },
            arraysize: 1,
            rows,
            at: 0,
        })
    }

    fn from_cursor(cursor: &Bound<'_, PyAny>) -> PyResult<SqlRows> {
        let rows: Vec<Py<PyAny>> = cursor
            .call_method0("fetchall")?
            .try_iter()?
            .map(|r| r.map(|r| r.unbind()))
            .collect::<PyResult<_>>()?;
        Ok(SqlRows {
            description: cursor.getattr("description")?.unbind(),
            rowcount: cursor.getattr("rowcount")?.extract()?,
            lastrowid: cursor.getattr("lastrowid")?.unbind(),
            arraysize: 1,
            rows,
            at: 0,
        })
    }

    pub fn all(&self, py: Python<'_>) -> Vec<Py<PyAny>> {
        self.rows[self.at..]
            .iter()
            .map(|r| r.clone_ref(py))
            .collect()
    }

    pub fn first(&self, py: Python<'_>) -> Option<Py<PyAny>> {
        self.rows.get(self.at).map(|r| r.clone_ref(py))
    }

    pub fn rowcount(&self) -> i64 {
        self.rowcount
    }

    pub fn lastrowid<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.lastrowid.bind(py).clone()
    }

    pub fn description<'py>(&self, py: Python<'py>) -> Bound<'py, PyAny> {
        self.description.bind(py).clone()
    }
}

#[pymethods]
impl SqlRows {
    fn fetchone(&mut self, py: Python<'_>) -> Py<PyAny> {
        match self.rows.get(self.at) {
            Some(r) => {
                self.at += 1;
                r.clone_ref(py)
            }
            None => py.None(),
        }
    }

    #[pyo3(signature = (size=None))]
    fn fetchmany(&mut self, py: Python<'_>, size: Option<usize>) -> Vec<Py<PyAny>> {
        let n = size.unwrap_or(self.arraysize);
        let end = (self.at + n).min(self.rows.len());
        let out = self.rows[self.at..end]
            .iter()
            .map(|r| r.clone_ref(py))
            .collect();
        self.at = end;
        out
    }

    fn fetchall(&mut self, py: Python<'_>) -> Vec<Py<PyAny>> {
        let out = self.all(py);
        self.at = self.rows.len();
        out
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self.rows.get(self.at) {
            Some(r) => {
                self.at += 1;
                Ok(r.clone_ref(py))
            }
            None => Err(PyStopIteration::new_err(())),
        }
    }

    fn close(&self) {}
}

/// `conn.execute(sql, params)` with the rows fetched (`rust_sql.execute`).
#[pyfunction]
#[pyo3(signature = (conn, sql, params=None))]
pub fn conn_execute<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &Bound<'py, PyAny>,
    params: Option<&Bound<'py, PyAny>>,
) -> PyResult<SqlRows> {
    let py = conn.py();
    let empty = PyTuple::empty(py).into_any();
    PyConn::new(conn)?.execute(sql, params.unwrap_or(&empty))
}

/// `conn.executemany(sql, seq_of_params)` (`rust_sql.executemany`).
#[pyfunction]
pub fn conn_executemany<'py>(
    conn: &Bound<'py, PyAny>,
    sql: &Bound<'py, PyAny>,
    seq_of_params: &Bound<'py, PyAny>,
) -> PyResult<SqlRows> {
    PyConn::new(conn)?.executemany(sql, seq_of_params)
}
