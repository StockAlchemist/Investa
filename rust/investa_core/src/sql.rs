//! SQL on a connection Python already holds.
//!
//! The storage modules hand this a `sqlite3*` taken from a Python
//! `sqlite3.Connection` (see `src/rust_sql.py`), so a statement run here is
//! inside whatever transaction the caller has open, sees its uncommitted
//! writes, and is committed or rolled back with it. The handle is borrowed
//! through `rusqlite` (`Connection::from_handle`, which neither reconfigures
//! nor closes it), and the library is the one Python's module calls
//! (`sqlite_read::bind`).
//!
//! What Python's `sqlite3` does around a statement, CPython 3.12
//! (`Modules/_sqlite/cursor.c`, `statement.c`), is reproduced:
//! - under legacy transaction control, a DML statement (INSERT, UPDATE,
//!   DELETE, REPLACE after any whitespace and comments) run while the
//!   connection is in autocommit first executes the caller's `BEGIN`;
//! - a second statement after the first is refused, by CPython's own test
//!   (anything but whitespace and comments after it), not rusqlite's;
//! - the binding count must match;
//! - `rowcount` is the changes of DML statements, `lastrowid` the last
//!   inserted rowid;
//! - errors carry SQLite's extended code and message (Python's connections
//!   leave extended result codes off, so the code is read from the handle,
//!   as CPython reads it), for the shim to raise as the class Python would.
//!
//! The caller (the shim) only sends what this binds exactly as Python would;
//! anything else stays on Python's own `execute`.

use std::cell::RefCell;

use rusqlite::types::{ToSqlOutput, ValueRef};
use rusqlite::{Connection, ffi};

use crate::sqlite_read::bound;

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
    Sqlite { code: i32, message: String },
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

/// The extended code and message SQLite holds for the handle's last failure.
fn failure_of(conn: &Connection) -> Failure {
    // SAFETY: the handle is live for as long as `conn` is, and both calls
    // only read the connection's error state.
    unsafe {
        let db = conn.handle();
        let code = ffi::sqlite3_extended_errcode(db);
        let p = ffi::sqlite3_errmsg(db);
        let message = if p.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        };
        Failure::Sqlite { code, message }
    }
}

/// A rusqlite error as the shim reports it: SQLite's own failures from the
/// handle, anything rusqlite raised itself declined to Python.
fn failure(conn: &Connection, e: &rusqlite::Error) -> Failure {
    match e {
        rusqlite::Error::SqliteFailure(..) | rusqlite::Error::SqlInputError { .. } => {
            failure_of(conn)
        }
        _ => Failure::Decline("a rusqlite error outside SQLite"),
    }
}

/// Where the first statement of `sql` ends (`sqlite3_prepare_v2`'s tail),
/// and whether the text holds a statement at all. SQLite ends a statement
/// only at a `;` or the end of the input, so text without one needs no
/// probe; otherwise the statement is prepared once here, for its tail, and
/// finalized.
fn first_statement(conn: &Connection, sql: &str) -> Result<(usize, bool), Failure> {
    if !sql.as_bytes().contains(&b';') {
        return Ok((sql.len(), lstrip_sql(sql.as_bytes()).is_some()));
    }
    let Ok(len) = i32::try_from(sql.len()) else {
        return Err(Failure::Decline("query too long"));
    };
    let mut stmt: *mut ffi::sqlite3_stmt = std::ptr::null_mut();
    let mut tail: *const std::ffi::c_char = std::ptr::null();
    // SAFETY: the text outlives the call, and the statement is finalized
    // before this returns.
    unsafe {
        let rc = ffi::sqlite3_prepare_v2(
            conn.handle(),
            sql.as_ptr().cast(),
            len,
            &mut stmt,
            &mut tail,
        );
        let present = !stmt.is_null();
        if present {
            ffi::sqlite3_finalize(stmt);
        }
        if rc != ffi::SQLITE_OK {
            return Err(failure_of(conn));
        }
        let end = if tail.is_null() {
            sql.len()
        } else {
            (tail.offset_from(sql.as_ptr().cast()) as usize).min(sql.len())
        };
        Ok((end, present))
    }
}

/// One prepared statement on a borrowed handle, driven a step at a time by
/// the PyO3 layer (`pyconn.rs`), which looks named parameters up in a Python
/// dict between `prepare` and `bind`.
pub struct Statement {
    // Declared first, so it is finalized before the connection it borrows.
    stmt: RefCell<rusqlite::Statement<'static>>,
    conn: Box<Connection>,
    dml: bool,
}

// The handle and statement are used only while the caller's Python call is
// running, on that call's thread (`py.detach` releases the GIL but stays on
// the thread), as CPython's own module does when it releases the GIL.
unsafe impl Send for Statement {}
unsafe impl Sync for Statement {}

impl Statement {
    /// Prepare exactly one statement on the handle a shim verified (see
    /// `rust_sql.py`), as Python's statement cache would.
    pub fn prepare(db: usize, sql: &str) -> Result<Statement, Failure> {
        if !bound() {
            return Err(Failure::Decline("SQLite is not bound"));
        }
        if db == 0 {
            return Err(Failure::Decline("no connection handle"));
        }
        if sql.as_bytes().contains(&0) {
            return Err(Failure::Decline("a NUL in the query"));
        }
        // SAFETY: the shim hands over the live `sqlite3*` of a connection it
        // holds; a borrowed connection is never closed by rusqlite.
        let conn = Box::new(
            unsafe { Connection::from_handle(db as *mut ffi::sqlite3) }
                .map_err(|_| Failure::Decline("the handle could not be borrowed"))?,
        );
        let (end, present) = first_statement(&conn, sql)?;
        if !present {
            return Err(Failure::Decline("an empty statement"));
        }
        if lstrip_sql(&sql.as_bytes()[end..]).is_some() {
            return Err(Failure::Programming(
                "You can only execute one statement at a time.".into(),
            ));
        }
        let stmt = conn.prepare(&sql[..end]).map_err(|e| failure(&conn, &e))?;
        // SAFETY: the statement borrows the boxed connection, whose address
        // never moves, and the field order drops the statement first.
        let stmt: rusqlite::Statement<'static> = unsafe { std::mem::transmute(stmt) };
        Ok(Statement {
            stmt: RefCell::new(stmt),
            conn,
            dml: is_dml(sql.as_bytes()),
        })
    }

    pub fn is_dml(&self) -> bool {
        self.dml
    }

    /// The names of the placeholders, `None` for a bare `?`.
    pub fn param_names(&self) -> Vec<Option<Vec<u8>>> {
        let stmt = self.stmt.borrow();
        (1..=stmt.parameter_count())
            .map(|i| stmt.parameter_name(i).map(|n| n.as_bytes().to_vec()))
            .collect()
    }

    /// Python's implicit BEGIN, before any binding.
    pub fn begin(&self, begin: Option<&str>) -> Result<(), Failure> {
        let Some(begin) = begin else {
            return Ok(());
        };
        if !self.dml || !self.conn.is_autocommit() {
            return Ok(());
        }
        let mut b = self
            .conn
            .prepare(begin)
            .map_err(|e| failure(&self.conn, &e))?;
        let mut rows = b.raw_query();
        match rows.next() {
            Ok(_) => Ok(()),
            Err(e) => Err(failure(&self.conn, &e)),
        }
    }

    /// A sequence of parameters, with Python's count check.
    pub fn bind_positional(&self, params: &[Value]) -> Result<(), Failure> {
        let expected = self.stmt.borrow().parameter_count();
        if expected != params.len() {
            return Err(Failure::Programming(format!(
                "Incorrect number of bindings supplied. The current statement uses {expected}, and there are {} supplied.",
                params.len()
            )));
        }
        // A named placeholder given a sequence: CPython 3.12 warns (DeprecationWarning).
        if self
            .param_names()
            .iter()
            .any(|n| n.as_ref().is_some_and(|n| n.first() != Some(&b'?')))
        {
            return Err(Failure::Decline("a named placeholder bound by position"));
        }
        self.bind_in_order(params)
    }

    /// Values already matched to the placeholders by name.
    pub fn bind_in_order(&self, params: &[Value]) -> Result<(), Failure> {
        let mut stmt = self.stmt.borrow_mut();
        for (k, v) in params.iter().enumerate() {
            let value = match v {
                Value::Null => ValueRef::Null,
                Value::Int(x) => ValueRef::Integer(*x),
                Value::Real(x) => ValueRef::Real(*x),
                Value::Text(t) => ValueRef::Text(t),
                Value::Blob(b) => ValueRef::Blob(b),
            };
            stmt.raw_bind_parameter(k + 1, ToSqlOutput::Borrowed(value))
                .map_err(|e| failure(&self.conn, &e))?;
        }
        Ok(())
    }

    /// Step to the end, collecting rows.
    pub fn run(&self) -> Result<Outcome, Failure> {
        let mut stmt = self.stmt.borrow_mut();
        let n = stmt.column_count();
        let mut columns = Vec::with_capacity(n);
        for c in 0..n {
            match stmt.column_name(c) {
                Ok(name) => columns.push(name.to_owned()),
                Err(_) => return Err(Failure::Decline("a column without a name")),
            }
        }
        let mut out = Vec::new();
        {
            let mut rows = stmt.raw_query();
            loop {
                // The failure is read before the rows are dropped (and the
                // statement reset).
                let row = match rows.next() {
                    Ok(Some(row)) => row,
                    Ok(None) => break,
                    Err(e) => return Err(failure(&self.conn, &e)),
                };
                let mut values = Vec::with_capacity(n);
                for c in 0..n {
                    let v = row.get_ref(c).map_err(|e| failure(&self.conn, &e))?;
                    values.push(match v {
                        ValueRef::Null => Value::Null,
                        ValueRef::Integer(x) => Value::Int(x),
                        ValueRef::Real(x) => Value::Real(x),
                        ValueRef::Text(t) => Value::Text(t.to_vec()),
                        ValueRef::Blob(b) => Value::Blob(b.to_vec()),
                    });
                }
                out.push(values);
            }
        }
        let rowcount = if self.dml {
            self.conn.changes() as i64
        } else {
            -1
        };
        Ok(Outcome {
            columns,
            rows: out,
            rowcount,
            lastrowid: self.conn.last_insert_rowid(),
        })
    }

    /// One `executemany` round: clear, bind, step once; the changes it made.
    pub fn run_once(&self, params: &[Value], named: bool) -> Result<i64, Failure> {
        self.stmt.borrow_mut().clear_bindings();
        if named {
            self.bind_in_order(params)?;
        } else {
            self.bind_positional(params)?;
        }
        let mut stmt = self.stmt.borrow_mut();
        let mut rows = stmt.raw_query();
        match rows.next() {
            Ok(None) => Ok(self.conn.changes() as i64),
            Ok(Some(_)) => Err(Failure::Programming(
                "executemany() can only execute DML statements.".into(),
            )),
            Err(e) => Err(failure(&self.conn, &e)),
        }
    }

    pub fn last_insert_rowid(&self) -> i64 {
        self.conn.last_insert_rowid()
    }
}

/// `cursor.execute(sql, params)` followed by `fetchall()`.
pub fn execute(
    db: usize,
    sql: &str,
    params: &[Value],
    begin: Option<&str>,
) -> Result<Outcome, Failure> {
    let st = Statement::prepare(db, sql)?;
    // CPython opens the transaction before binding: a binding error leaves it open.
    st.begin(begin)?;
    st.bind_positional(params)?;
    st.run()
}

/// `cursor.executemany(sql, seq)`: one prepared statement, stepped once per
/// parameter set, DML only.
pub fn executemany(
    db: usize,
    sql: &str,
    param_sets: &[Vec<Value>],
    begin: Option<&str>,
) -> Result<Outcome, Failure> {
    let st = Statement::prepare(db, sql)?;
    if !st.is_dml() {
        return Err(Failure::Programming(
            "executemany() can only execute DML statements.".into(),
        ));
    }
    st.begin(begin)?;
    let mut rowcount: i64 = 0;
    for params in param_sets {
        rowcount += st.run_once(params, false)?;
    }
    Ok(Outcome {
        columns: Vec::new(),
        rows: Vec::new(),
        rowcount,
        lastrowid: st.last_insert_rowid(),
    })
}

/// `sqlite3_db_filename(db, "main")`, to check a handle belongs to the
/// connection the shim thinks it does.
pub fn db_filename(db: usize) -> Option<String> {
    if !bound() || db == 0 {
        return None;
    }
    // SAFETY: as in `Statement::prepare`; the borrow is dropped at once.
    let conn = unsafe { Connection::from_handle(db as *mut ffi::sqlite3) }.ok()?;
    conn.path().map(str::to_owned)
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
