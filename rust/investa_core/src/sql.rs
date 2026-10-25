//! SQL on a connection Python already holds.
//!
//! The storage modules hand this a `sqlite3*` taken from a Python
//! `sqlite3.Connection` (see `src/rust_sql.py`), so a statement run here is
//! inside whatever transaction the caller has open, sees its uncommitted
//! writes, and is committed or rolled back with it. The library is the one
//! Python's module calls (`sqlite_read::bind`).
//!
//! What Python's `sqlite3` does around a statement, CPython 3.12
//! (`Modules/_sqlite/cursor.c`, `statement.c`), is reproduced:
//! - under legacy transaction control, a DML statement (INSERT, UPDATE,
//!   DELETE, REPLACE after any whitespace and comments) run while the
//!   connection is in autocommit first executes the caller's `BEGIN`;
//! - a second statement after the first is refused;
//! - the binding count must match;
//! - `rowcount` is the changes of DML statements, `lastrowid` the last
//!   inserted rowid;
//! - errors carry SQLite's extended code and message, for the shim to raise
//!   as the class Python would.
//!
//! The caller (the shim) only sends what this binds exactly as Python would;
//! anything else stays on Python's own `execute`.

use std::ffi::{CStr, c_char, c_int};

use crate::sqlite_read::{Api, Db, Stmt, api};

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_INTEGER: c_int = 1;
const SQLITE_FLOAT: c_int = 2;
const SQLITE_TEXT: c_int = 3;
const SQLITE_BLOB: c_int = 4;
const SQLITE_TRANSIENT: isize = -1;

/// A value bound to, or read from, a statement.
#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    Null,
    Int(i64),
    Real(f64),
    /// UTF-8 text when bound; read back as SQLite's bytes, which the caller
    /// decodes (and reports when they are not UTF-8, as Python does).
    Text(Vec<u8>),
    Blob(Vec<u8>),
}

/// Why a statement did not run to completion.
#[derive(Debug, PartialEq)]
pub enum Failure {
    /// SQLite refused it: extended result code and `sqlite3_errmsg`.
    Sqlite { code: c_int, message: String },
    /// Python raises ProgrammingError with this message.
    Programming(String),
    /// Not something the port reproduces; the caller lets Python run it.
    Decline(&'static str),
}

pub struct Outcome {
    /// Column names (`cursor.description`), empty for a statement without
    /// result columns.
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    /// `cursor.rowcount`: the DML changes, or -1.
    pub rowcount: i64,
    pub lastrowid: i64,
}

/// CPython's `lstrip_sql`: skip whitespace, `--` line comments and `/* */`
/// block comments. None when nothing else remains (or a block comment never
/// closes).
pub fn lstrip_sql(sql: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i < sql.len() {
        match sql[i] {
            b' ' | b'\t' | b'\x0c' | b'\n' | b'\r' => i += 1,
            b'-' if sql.get(i + 1) == Some(&b'-') => {
                while i < sql.len() && sql[i] != b'\n' {
                    i += 1;
                }
            }
            b'/' if sql.get(i + 1) == Some(&b'*') => {
                let rest = &sql[i + 2..];
                let close = rest.windows(2).position(|w| w == b"*/")?;
                i += 2 + close + 2;
            }
            _ => return Some(&sql[i..]),
        }
    }
    None
}

/// CPython's DML test, which decides whether a statement opens a transaction.
pub fn is_dml(sql: &[u8]) -> bool {
    let Some(p) = lstrip_sql(sql) else {
        return false;
    };
    let starts = |kw: &[u8]| p.len() >= kw.len() && p[..kw.len()].eq_ignore_ascii_case(kw);
    starts(b"insert") || starts(b"update") || starts(b"delete") || starts(b"replace")
}

fn sqlite_failure(api: &Api, db: *mut Db) -> Failure {
    unsafe {
        let code = (api.extended_errcode)(db);
        let p = (api.errmsg)(db);
        let message = if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        Failure::Sqlite { code, message }
    }
}

struct Prepared<'a> {
    api: &'a Api,
    stmt: *mut Stmt,
}

impl Drop for Prepared<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.finalize)(self.stmt) };
    }
}

/// Prepare exactly one statement, as Python's statement cache would.
fn prepare<'a>(api: &'a Api, db: *mut Db, sql: &str) -> Result<Prepared<'a>, Failure> {
    let bytes = sql.as_bytes();
    if bytes.contains(&0) {
        return Err(Failure::Decline("a NUL in the query"));
    }
    let Ok(len) = c_int::try_from(bytes.len()) else {
        return Err(Failure::Decline("query too long"));
    };
    let mut stmt: *mut Stmt = std::ptr::null_mut();
    let mut tail: *const c_char = std::ptr::null();
    let rc = unsafe { (api.prepare_v2)(db, bytes.as_ptr().cast(), len, &mut stmt, &mut tail) };
    if rc != SQLITE_OK {
        if !stmt.is_null() {
            unsafe { (api.finalize)(stmt) };
        }
        return Err(sqlite_failure(api, db));
    }
    if stmt.is_null() {
        return Err(Failure::Decline("an empty statement"));
    }
    let prepared = Prepared { api, stmt };
    if !tail.is_null() {
        let consumed = unsafe { tail.offset_from(bytes.as_ptr().cast()) } as usize;
        if lstrip_sql(&bytes[consumed.min(bytes.len())..]).is_some() {
            return Err(Failure::Programming(
                "You can only execute one statement at a time.".into(),
            ));
        }
    }
    Ok(prepared)
}

fn bind_all(p: &Prepared<'_>, db: *mut Db, params: &[Value]) -> Result<(), Failure> {
    let api = p.api;
    let expected = unsafe { (api.bind_parameter_count)(p.stmt) } as usize;
    if expected != params.len() {
        return Err(Failure::Programming(format!(
            "Incorrect number of bindings supplied. The current statement uses {expected}, and there are {} supplied.",
            params.len()
        )));
    }
    // A named placeholder given a sequence: CPython 3.12 warns (DeprecationWarning).
    if param_names(p)
        .iter()
        .any(|n| n.as_ref().is_some_and(|n| n.first() != Some(&b'?')))
    {
        return Err(Failure::Decline("a named placeholder bound by position"));
    }
    bind_values(p, db, params)
}

/// The names of a statement's placeholders, `None` for a bare `?`.
fn param_names(p: &Prepared<'_>) -> Vec<Option<Vec<u8>>> {
    let n = unsafe { (p.api.bind_parameter_count)(p.stmt) };
    (1..=n)
        .map(|i| unsafe {
            let ptr = (p.api.bind_parameter_name)(p.stmt, i);
            (!ptr.is_null()).then(|| CStr::from_ptr(ptr).to_bytes().to_vec())
        })
        .collect()
}

fn bind_values(p: &Prepared<'_>, db: *mut Db, params: &[Value]) -> Result<(), Failure> {
    let api = p.api;
    for (k, v) in params.iter().enumerate() {
        let i = (k + 1) as c_int;
        let rc = unsafe {
            match v {
                Value::Null => (api.bind_null)(p.stmt, i),
                Value::Int(x) => (api.bind_int64)(p.stmt, i, *x),
                Value::Real(x) => (api.bind_double)(p.stmt, i, *x),
                Value::Text(t) => {
                    let Ok(n) = c_int::try_from(t.len()) else {
                        return Err(Failure::Decline("a string too long to bind"));
                    };
                    (api.bind_text)(p.stmt, i, t.as_ptr().cast(), n, SQLITE_TRANSIENT)
                }
                Value::Blob(b) => {
                    let Ok(n) = c_int::try_from(b.len()) else {
                        return Err(Failure::Decline("a blob too long to bind"));
                    };
                    (api.bind_blob)(p.stmt, i, b.as_ptr().cast(), n, SQLITE_TRANSIENT)
                }
            }
        };
        if rc != SQLITE_OK {
            return Err(sqlite_failure(api, db));
        }
    }
    Ok(())
}

fn read_row(p: &Prepared<'_>, n: c_int) -> Vec<Value> {
    let api = p.api;
    (0..n)
        .map(|c| unsafe {
            match (api.column_type)(p.stmt, c) {
                SQLITE_INTEGER => Value::Int((api.column_int64)(p.stmt, c)),
                SQLITE_FLOAT => Value::Real((api.column_double)(p.stmt, c)),
                SQLITE_TEXT => {
                    let ptr = (api.column_text)(p.stmt, c);
                    let len = (api.column_bytes)(p.stmt, c);
                    if ptr.is_null() || len <= 0 {
                        Value::Text(Vec::new())
                    } else {
                        Value::Text(std::slice::from_raw_parts(ptr, len as usize).to_vec())
                    }
                }
                SQLITE_BLOB => {
                    let ptr = (api.column_blob)(p.stmt, c);
                    let len = (api.column_bytes)(p.stmt, c);
                    if ptr.is_null() || len <= 0 {
                        Value::Blob(Vec::new())
                    } else {
                        Value::Blob(
                            std::slice::from_raw_parts(ptr.cast::<u8>(), len as usize).to_vec(),
                        )
                    }
                }
                _ => Value::Null,
            }
        })
        .collect()
}

fn column_names(p: &Prepared<'_>, n: c_int) -> Vec<String> {
    (0..n)
        .map(|c| unsafe {
            let ptr = (p.api.column_name)(p.stmt, c);
            if ptr.is_null() {
                String::new()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        })
        .collect()
}

/// Python's implicit `BEGIN` before DML in autocommit, under legacy control.
fn maybe_begin(api: &Api, db: *mut Db, sql: &str, begin: Option<&str>) -> Result<(), Failure> {
    let Some(begin) = begin else {
        return Ok(());
    };
    if !is_dml(sql.as_bytes()) || unsafe { (api.get_autocommit)(db) } == 0 {
        return Ok(());
    }
    let p = prepare(api, db, begin)?;
    match unsafe { (api.step)(p.stmt) } {
        SQLITE_DONE | SQLITE_ROW => Ok(()),
        _ => Err(sqlite_failure(api, db)),
    }
}

/// The handle a shim verified (see `rust_sql.py`); never null.
fn handle(db: usize) -> Result<(&'static Api, *mut Db), Failure> {
    let api = api().ok_or(Failure::Decline("SQLite is not bound"))?;
    if db == 0 {
        return Err(Failure::Decline("no connection handle"));
    }
    Ok((api, db as *mut Db))
}

/// `cursor.execute(sql, params)` followed by `fetchall()`.
pub fn execute(
    db: usize,
    sql: &str,
    params: &[Value],
    begin: Option<&str>,
) -> Result<Outcome, Failure> {
    let (api, db) = handle(db)?;
    let p = prepare(api, db, sql)?;
    // CPython opens the transaction before binding: a binding error leaves it open.
    maybe_begin(api, db, sql, begin)?;
    bind_all(&p, db, params)?;
    let n = unsafe { (api.column_count)(p.stmt) };
    let columns = if n > 0 {
        column_names(&p, n)
    } else {
        Vec::new()
    };
    let mut rows = Vec::new();
    loop {
        match unsafe { (api.step)(p.stmt) } {
            SQLITE_ROW => rows.push(read_row(&p, n)),
            SQLITE_DONE => break,
            _ => return Err(sqlite_failure(api, db)),
        }
    }
    let rowcount = if is_dml(sql.as_bytes()) {
        unsafe { (api.changes64)(db) }
    } else {
        -1
    };
    Ok(Outcome {
        columns,
        rows,
        rowcount,
        lastrowid: unsafe { (api.last_insert_rowid)(db) },
    })
}

/// `cursor.executemany(sql, seq)`: one prepared statement, stepped once per
/// parameter set, DML only.
pub fn executemany(
    db: usize,
    sql: &str,
    param_sets: &[Vec<Value>],
    begin: Option<&str>,
) -> Result<Outcome, Failure> {
    let (api, db) = handle(db)?;
    let p = prepare(api, db, sql)?;
    if !is_dml(sql.as_bytes()) {
        return Err(Failure::Programming(
            "executemany() can only execute DML statements.".into(),
        ));
    }
    maybe_begin(api, db, sql, begin)?;
    let mut rowcount: i64 = 0;
    for params in param_sets {
        unsafe {
            (api.reset)(p.stmt);
            (api.clear_bindings)(p.stmt);
        }
        bind_all(&p, db, params)?;
        match unsafe { (api.step)(p.stmt) } {
            SQLITE_DONE => rowcount += unsafe { (api.changes64)(db) },
            SQLITE_ROW => {
                return Err(Failure::Programming(
                    "executemany() can only execute DML statements.".into(),
                ));
            }
            _ => return Err(sqlite_failure(api, db)),
        }
    }
    Ok(Outcome {
        columns: Vec::new(),
        rows: Vec::new(),
        rowcount,
        lastrowid: unsafe { (api.last_insert_rowid)(db) },
    })
}

/// `sqlite3_db_filename(db, "main")`, to check a handle belongs to the
/// connection the shim thinks it does.
pub fn db_filename(db: usize) -> Option<String> {
    let (api, db) = handle(db).ok()?;
    let p = unsafe { (api.db_filename)(db, c"main".as_ptr()) };
    if p.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

/// One prepared statement, driven a step at a time by the PyO3 layer
/// (`pyconn.rs`), which looks named parameters up in a Python dict between
/// `prepare` and `bind`.
pub struct Statement {
    api: &'static Api,
    db: *mut Db,
    stmt: *mut Stmt,
    dml: bool,
}

// The handle and statement are used only while the caller's Python call is
// running, as CPython's own module does when it releases the GIL.
unsafe impl Send for Statement {}
unsafe impl Sync for Statement {}

impl Drop for Statement {
    fn drop(&mut self) {
        unsafe { (self.api.finalize)(self.stmt) };
    }
}

impl Statement {
    pub fn prepare(db: usize, sql: &str) -> Result<Statement, Failure> {
        let (api, db) = handle(db)?;
        let p = prepare(api, db, sql)?;
        let stmt = p.stmt;
        std::mem::forget(p); // ownership moves to Statement
        Ok(Statement {
            api,
            db,
            stmt,
            dml: is_dml(sql.as_bytes()),
        })
    }

    fn prepared(&self) -> std::mem::ManuallyDrop<Prepared<'static>> {
        std::mem::ManuallyDrop::new(Prepared {
            api: self.api,
            stmt: self.stmt,
        })
    }

    pub fn is_dml(&self) -> bool {
        self.dml
    }

    pub fn param_names(&self) -> Vec<Option<Vec<u8>>> {
        param_names(&self.prepared())
    }

    /// Python's implicit BEGIN, before any binding.
    pub fn begin(&self, begin: Option<&str>) -> Result<(), Failure> {
        let Some(begin) = begin else {
            return Ok(());
        };
        if !self.dml || unsafe { (self.api.get_autocommit)(self.db) } == 0 {
            return Ok(());
        }
        let p = prepare(self.api, self.db, begin)?;
        match unsafe { (self.api.step)(p.stmt) } {
            SQLITE_DONE | SQLITE_ROW => Ok(()),
            _ => Err(sqlite_failure(self.api, self.db)),
        }
    }

    /// A sequence of parameters, with Python's count check.
    pub fn bind_positional(&self, params: &[Value]) -> Result<(), Failure> {
        bind_all(&self.prepared(), self.db, params)
    }

    /// Values already matched to the placeholders by name.
    pub fn bind_in_order(&self, params: &[Value]) -> Result<(), Failure> {
        bind_values(&self.prepared(), self.db, params)
    }

    /// Step to the end, collecting rows.
    pub fn run(&self) -> Result<Outcome, Failure> {
        let p = self.prepared();
        let n = unsafe { (self.api.column_count)(self.stmt) };
        let columns = if n > 0 {
            column_names(&p, n)
        } else {
            Vec::new()
        };
        let mut rows = Vec::new();
        loop {
            match unsafe { (self.api.step)(self.stmt) } {
                SQLITE_ROW => rows.push(read_row(&p, n)),
                SQLITE_DONE => break,
                _ => return Err(sqlite_failure(self.api, self.db)),
            }
        }
        let rowcount = if self.dml {
            unsafe { (self.api.changes64)(self.db) }
        } else {
            -1
        };
        Ok(Outcome {
            columns,
            rows,
            rowcount,
            lastrowid: unsafe { (self.api.last_insert_rowid)(self.db) },
        })
    }

    /// One `executemany` round: reset, bind, step once; the changes it made.
    pub fn run_once(&self, params: &[Value], named: bool) -> Result<i64, Failure> {
        unsafe {
            (self.api.reset)(self.stmt);
            (self.api.clear_bindings)(self.stmt);
        }
        if named {
            self.bind_in_order(params)?;
        } else {
            self.bind_positional(params)?;
        }
        match unsafe { (self.api.step)(self.stmt) } {
            SQLITE_DONE => Ok(unsafe { (self.api.changes64)(self.db) }),
            SQLITE_ROW => Err(Failure::Programming(
                "executemany() can only execute DML statements.".into(),
            )),
            _ => Err(sqlite_failure(self.api, self.db)),
        }
    }

    pub fn last_insert_rowid(&self) -> i64 {
        unsafe { (self.api.last_insert_rowid)(self.db) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lstrip_and_dml() {
        assert_eq!(
            lstrip_sql(b"  -- x\n /* y */ SELECT 1"),
            Some(&b"SELECT 1"[..])
        );
        assert_eq!(lstrip_sql(b"  -- only a comment"), None);
        assert_eq!(lstrip_sql(b"/* never closes"), None);
        assert!(is_dml(b"\n  insert into t values (1)"));
        assert!(is_dml(b"/* c */ REPLACE INTO t VALUES (1)"));
        assert!(is_dml(b"Update t set a = 1"));
        assert!(!is_dml(b"SELECT 1"));
        assert!(!is_dml(
            b"WITH x AS (SELECT 1) INSERT INTO t SELECT * FROM x"
        ));
        assert!(!is_dml(b"CREATE TABLE t (a)"));
        assert!(!is_dml(b""));
    }
}
