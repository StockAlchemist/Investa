//! Reading the daily price archive (`market_db.get_ohlcv_batch`) without a
//! Python object per cell.
//!
//! The original runs the query through `pd.read_sql_query`, which builds a
//! Python tuple per row and a Python object per cell before pandas infers each
//! column's dtype, and then parses every date string. Here the rows are stepped
//! in C and written straight into column arrays.
//!
//! **One SQLite in the process.** Two copies of SQLite in one process can
//! corrupt a database: POSIX locks belong to the process, so one copy closing
//! a file drops the locks the other still relies on. So this module links no
//! SQLite of its own. `bind` is handed the path of Python's `_sqlite3`
//! extension, opens that already-loaded image (`RTLD_NOLOAD`, so nothing new is
//! loaded) and resolves the C API through it — the very library Python's
//! `sqlite3` module calls. If it cannot, the reader stays unbound and Python
//! keeps its own path.
//!
//! **Only what pandas would read the same way.** A column is returned as
//! int64 when every cell is an INTEGER, as float64 (NULL as NaN) otherwise —
//! the dtype `read_sql_query` infers followed by the original's `to_numeric`.
//! Dates must be `YYYY-MM-DD` text inside the nanosecond range, which
//! `pd.to_datetime` parses to midnight. Anything else (text in a price, another
//! date format, a symbol that is not UTF-8) is `Unsupported`, and the caller
//! runs the original.

use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::sync::OnceLock;

const SQLITE_OK: c_int = 0;
const SQLITE_ROW: c_int = 100;
const SQLITE_DONE: c_int = 101;
const SQLITE_INTEGER: c_int = 1;
const SQLITE_FLOAT: c_int = 2;
const SQLITE_TEXT: c_int = 3;
const SQLITE_NULL: c_int = 5;
const SQLITE_OPEN_READWRITE: c_int = 0x2;
/// `SQLITE_TRANSIENT`: SQLite copies a bound string before the call returns.
const SQLITE_TRANSIENT: isize = -1;
/// The busy timeout Python's connections set (`PRAGMA busy_timeout = 30000`).
const BUSY_TIMEOUT_MS: c_int = 30_000;

#[cfg(target_os = "macos")]
const RTLD_NOLOAD: c_int = 0x10;
#[cfg(not(target_os = "macos"))]
const RTLD_NOLOAD: c_int = 0x4;
const RTLD_LAZY: c_int = 0x1;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

type Db = c_void;
type Stmt = c_void;

#[allow(non_snake_case)]
struct Api {
    open_v2: unsafe extern "C" fn(*const c_char, *mut *mut Db, c_int, *const c_char) -> c_int,
    close_v2: unsafe extern "C" fn(*mut Db) -> c_int,
    busy_timeout: unsafe extern "C" fn(*mut Db, c_int) -> c_int,
    prepare_v2: unsafe extern "C" fn(
        *mut Db,
        *const c_char,
        c_int,
        *mut *mut Stmt,
        *mut *const c_char,
    ) -> c_int,
    bind_text: unsafe extern "C" fn(*mut Stmt, c_int, *const c_char, c_int, isize) -> c_int,
    step: unsafe extern "C" fn(*mut Stmt) -> c_int,
    finalize: unsafe extern "C" fn(*mut Stmt) -> c_int,
    column_type: unsafe extern "C" fn(*mut Stmt, c_int) -> c_int,
    column_int64: unsafe extern "C" fn(*mut Stmt, c_int) -> i64,
    column_double: unsafe extern "C" fn(*mut Stmt, c_int) -> f64,
    column_text: unsafe extern "C" fn(*mut Stmt, c_int) -> *const u8,
    column_bytes: unsafe extern "C" fn(*mut Stmt, c_int) -> c_int,
    errmsg: unsafe extern "C" fn(*mut Db) -> *const c_char,
}

// The function pointers are plain C entry points; SQLite itself is built
// threadsafe (Python's module requires it), and each read opens its own
// connection.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

static API: OnceLock<Option<Api>> = OnceLock::new();

unsafe fn sym<T>(handle: *mut c_void, name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let p = unsafe { dlsym(handle, cname.as_ptr()) };
    if p.is_null() {
        None
    } else {
        // SAFETY: T is the matching `extern "C" fn` type for this symbol.
        Some(unsafe { std::mem::transmute_copy::<*mut c_void, T>(&p) })
    }
}

fn load(module_path: &str) -> Option<Api> {
    let path = CString::new(module_path).ok()?;
    // SAFETY: NOLOAD returns a handle only to an image already in the process.
    let handle = unsafe { dlopen(path.as_ptr(), RTLD_LAZY | RTLD_NOLOAD) };
    if handle.is_null() {
        return None;
    }
    unsafe {
        Some(Api {
            open_v2: sym(handle, "sqlite3_open_v2")?,
            close_v2: sym(handle, "sqlite3_close_v2")?,
            busy_timeout: sym(handle, "sqlite3_busy_timeout")?,
            prepare_v2: sym(handle, "sqlite3_prepare_v2")?,
            bind_text: sym(handle, "sqlite3_bind_text")?,
            step: sym(handle, "sqlite3_step")?,
            finalize: sym(handle, "sqlite3_finalize")?,
            column_type: sym(handle, "sqlite3_column_type")?,
            column_int64: sym(handle, "sqlite3_column_int64")?,
            column_double: sym(handle, "sqlite3_column_double")?,
            column_text: sym(handle, "sqlite3_column_text")?,
            column_bytes: sym(handle, "sqlite3_column_bytes")?,
            errmsg: sym(handle, "sqlite3_errmsg")?,
        })
    }
}

/// Resolve SQLite through Python's `_sqlite3` extension at `module_path`.
/// The first call decides; later calls report that outcome.
pub fn bind(module_path: &str) -> bool {
    API.get_or_init(|| load(module_path)).is_some()
}

fn api() -> Option<&'static Api> {
    API.get().and_then(|a| a.as_ref())
}

/// One price column: int64 when every cell was an INTEGER, else float64.
pub enum Column {
    Int(Vec<i64>),
    Real(Vec<f64>),
}

#[derive(Default)]
struct ColumnBuilder {
    real: Vec<f64>,
    int: Vec<i64>,
    all_int: bool,
}

impl ColumnBuilder {
    fn new() -> Self {
        Self {
            all_int: true,
            ..Self::default()
        }
    }

    fn finish(self) -> Column {
        if self.all_int && !self.int.is_empty() {
            Column::Int(self.int)
        } else {
            Column::Real(self.real)
        }
    }
}

pub struct Batch {
    /// Symbols in the order the query returned them (`ORDER BY symbol`).
    pub symbols: Vec<String>,
    /// Row range of each symbol: `offsets[i]..offsets[i + 1]`.
    pub offsets: Vec<i64>,
    /// Each row's date as nanoseconds since the epoch, midnight.
    pub dates_ns: Vec<i64>,
    /// open, high, low, close, adj_close, volume.
    pub columns: Vec<Column>,
}

pub enum Outcome {
    Batch(Batch),
    /// The rows hold something only the original reads correctly.
    Unsupported(String),
}

#[derive(Debug)]
pub struct ReadError(pub String);

/// Days from 1970-01-01 to a proleptic Gregorian date (Howard Hinnant's
/// `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

const DAY_NS: i64 = 86_400_000_000_000;

/// `YYYY-MM-DD` to nanoseconds, or None for anything `pd.to_datetime` might
/// read differently: another shape, an impossible day, or a year outside
/// datetime64[ns] (1677-09-21 .. 2262-04-11; whole years 1678..=2261 kept).
pub fn parse_iso_day(text: &[u8]) -> Option<i64> {
    if text.len() != 10 || text[4] != b'-' || text[7] != b'-' {
        return None;
    }
    let num = |s: &[u8]| -> Option<i64> {
        s.iter().try_fold(0i64, |acc, &c| {
            c.is_ascii_digit().then(|| acc * 10 + i64::from(c - b'0'))
        })
    };
    let (y, m, d) = (num(&text[0..4])?, num(&text[5..7])?, num(&text[8..10])?);
    if !(1678..=2261).contains(&y) || !(1..=12).contains(&m) || d < 1 {
        return None;
    }
    let leap = (y % 4 == 0 && y % 100 != 0) || y % 400 == 0;
    let month_days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if d > month_days[(m - 1) as usize] {
        return None;
    }
    Some(days_from_civil(y, m, d) * DAY_NS)
}

struct Connection<'a> {
    api: &'a Api,
    db: *mut Db,
}

impl Drop for Connection<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.close_v2)(self.db) };
    }
}

struct Statement<'a> {
    api: &'a Api,
    stmt: *mut Stmt,
}

impl Drop for Statement<'_> {
    fn drop(&mut self) {
        unsafe { (self.api.finalize)(self.stmt) };
    }
}

fn error(api: &Api, db: *mut Db, what: &str) -> ReadError {
    let msg = unsafe {
        let p = (api.errmsg)(db);
        if p.is_null() {
            String::new()
        } else {
            CStr::from_ptr(p).to_string_lossy().into_owned()
        }
    };
    ReadError(format!("{what}: {msg}"))
}

/// The rows of
/// `SELECT symbol, date, open, high, low, close, adj_close, volume FROM daily_ohlcv
///  WHERE symbol IN (..) AND interval = ? AND date BETWEEN ? AND ? ORDER BY symbol, date ASC`.
/// `Err` when the database cannot be read (the caller lets the original try,
/// and raise); `Ok(Unsupported)` when it can but holds values the original
/// reads differently.
pub fn read_daily_ohlcv(
    db_path: &str,
    symbols: &[String],
    interval: &str,
    start: &str,
    end: &str,
) -> Result<Outcome, ReadError> {
    let api = api().ok_or_else(|| ReadError("SQLite is not bound".into()))?;
    let cpath = CString::new(db_path).map_err(|_| ReadError("path holds a NUL".into()))?;
    let mut db: *mut Db = std::ptr::null_mut();
    let rc = unsafe {
        (api.open_v2)(
            cpath.as_ptr(),
            &mut db,
            SQLITE_OPEN_READWRITE,
            std::ptr::null(),
        )
    };
    if db.is_null() {
        return Err(ReadError(format!("open failed ({rc})")));
    }
    let conn = Connection { api, db };
    if rc != SQLITE_OK {
        return Err(error(api, conn.db, "open"));
    }
    unsafe { (api.busy_timeout)(conn.db, BUSY_TIMEOUT_MS) };

    let placeholders = vec!["?"; symbols.len()].join(", ");
    let sql = format!(
        "SELECT symbol, date, open, high, low, close, adj_close, volume \
         FROM daily_ohlcv \
         WHERE symbol IN ({placeholders}) AND interval = ? AND date BETWEEN ? AND ? \
         ORDER BY symbol, date ASC"
    );
    let csql = CString::new(sql).map_err(|_| ReadError("query holds a NUL".into()))?;
    let mut raw: *mut Stmt = std::ptr::null_mut();
    let rc =
        unsafe { (api.prepare_v2)(conn.db, csql.as_ptr(), -1, &mut raw, std::ptr::null_mut()) };
    if rc != SQLITE_OK || raw.is_null() {
        return Err(error(api, conn.db, "prepare"));
    }
    let stmt = Statement { api, stmt: raw };

    let params = symbols
        .iter()
        .map(String::as_str)
        .chain([interval, start, end]);
    for (i, p) in params.enumerate() {
        let Ok(len) = c_int::try_from(p.len()) else {
            return Err(ReadError("parameter too long".into()));
        };
        let rc = unsafe {
            (api.bind_text)(
                stmt.stmt,
                (i + 1) as c_int,
                p.as_ptr().cast(),
                len,
                SQLITE_TRANSIENT,
            )
        };
        if rc != SQLITE_OK {
            return Err(error(api, conn.db, "bind"));
        }
    }

    let mut out_symbols: Vec<String> = Vec::new();
    let mut offsets: Vec<i64> = Vec::new();
    let mut dates_ns: Vec<i64> = Vec::new();
    let mut cols: Vec<ColumnBuilder> = (0..6).map(|_| ColumnBuilder::new()).collect();

    let text_of = |col: c_int| -> &[u8] {
        unsafe {
            let p = (api.column_text)(stmt.stmt, col);
            let n = (api.column_bytes)(stmt.stmt, col);
            if p.is_null() || n <= 0 {
                &[]
            } else {
                std::slice::from_raw_parts(p, n as usize)
            }
        }
    };

    let mut row: i64 = 0;
    loop {
        match unsafe { (api.step)(stmt.stmt) } {
            SQLITE_ROW => {}
            SQLITE_DONE => break,
            _ => return Err(error(api, conn.db, "step")),
        }
        if unsafe { (api.column_type)(stmt.stmt, 0) } != SQLITE_TEXT {
            return Ok(Outcome::Unsupported("a symbol that is not text".into()));
        }
        let symbol = text_of(0);
        if out_symbols.last().map(String::as_bytes) != Some(symbol) {
            let Ok(s) = std::str::from_utf8(symbol) else {
                return Ok(Outcome::Unsupported("a symbol that is not UTF-8".into()));
            };
            out_symbols.push(s.to_owned());
            offsets.push(row);
        }
        if unsafe { (api.column_type)(stmt.stmt, 1) } != SQLITE_TEXT {
            return Ok(Outcome::Unsupported("a date that is not text".into()));
        }
        let Some(ns) = parse_iso_day(text_of(1)) else {
            return Ok(Outcome::Unsupported("a date that is not YYYY-MM-DD".into()));
        };
        dates_ns.push(ns);
        for (k, col) in cols.iter_mut().enumerate() {
            let c = (k + 2) as c_int;
            match unsafe { (api.column_type)(stmt.stmt, c) } {
                SQLITE_INTEGER => {
                    let v = unsafe { (api.column_int64)(stmt.stmt, c) };
                    col.real.push(v as f64);
                    if col.all_int {
                        col.int.push(v);
                    }
                }
                SQLITE_FLOAT => {
                    col.real.push(unsafe { (api.column_double)(stmt.stmt, c) });
                    col.all_int = false;
                }
                SQLITE_NULL => {
                    col.real.push(f64::NAN);
                    col.all_int = false;
                }
                _ => {
                    return Ok(Outcome::Unsupported(
                        "a price that is text or a blob".into(),
                    ));
                }
            }
        }
        row += 1;
    }
    offsets.push(row);

    Ok(Outcome::Batch(Batch {
        symbols: out_symbols,
        offsets,
        dates_ns,
        columns: cols.into_iter().map(ColumnBuilder::finish).collect(),
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iso_days() {
        assert_eq!(parse_iso_day(b"1970-01-01"), Some(0));
        assert_eq!(parse_iso_day(b"2024-02-29"), Some(19_782 * DAY_NS));
        assert_eq!(parse_iso_day(b"2023-02-29"), None);
        assert_eq!(parse_iso_day(b"1900-02-29"), None);
        assert_eq!(parse_iso_day(b"2000-02-29"), Some(11_016 * DAY_NS));
        assert_eq!(parse_iso_day(b"1677-12-31"), None);
        assert_eq!(parse_iso_day(b"2262-01-01"), None);
        assert_eq!(parse_iso_day(b"2024-1-02"), None);
        assert_eq!(parse_iso_day(b"2024-01-02 "), None);
        assert_eq!(parse_iso_day(b"2024-13-01"), None);
        assert_eq!(parse_iso_day(b"+024-01-01"), None);
    }
}
