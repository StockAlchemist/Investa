//! SQLite for the crate: the one-library check, and the daily price archive
//! read (`market_db.get_ohlcv_batch`) without a Python object per cell.
//!
//! **One SQLite in the process.** Two copies of SQLite in one process can
//! corrupt a database: POSIX locks belong to the process, so one copy closing
//! a file drops the locks the other still relies on. The crate uses
//! `rusqlite` linked against the system's shared SQLite (no bundled copy),
//! which is the library Python's `_sqlite3` extension links on macOS, Debian
//! and Ubuntu; the dynamic loader then maps a single image for both. `bind`
//! proves it at start-up: it opens Python's already-loaded extension
//! (`RTLD_NOLOAD`), resolves `sqlite3_libversion` through it, and compares
//! that address with the crate's own. If they differ the crate never opens a
//! database — every Rust SQL path declines and Python keeps its own, as
//! before — so a mismatched build costs speed, not data.
//!
//! **Only what pandas would read the same way.** A column is returned as
//! int64 when every cell is an INTEGER, as float64 (NULL as NaN) otherwise —
//! the dtype `read_sql_query` infers followed by the original's `to_numeric`.
//! Dates must be `YYYY-MM-DD` text inside the nanosecond range, which
//! `pd.to_datetime` parses to midnight. Anything else (text in a price, another
//! date format, a symbol that is not UTF-8) is `Unsupported`, and the caller
//! runs the original.

use std::ffi::{CString, c_char, c_int, c_void};
use std::sync::OnceLock;
use std::time::Duration;

use rusqlite::types::ValueRef;
use rusqlite::{Connection, OpenFlags, ffi};

/// The busy timeout Python's connections set (`PRAGMA busy_timeout = 30000`).
const BUSY_TIMEOUT: Duration = Duration::from_millis(30_000);

#[cfg(target_os = "macos")]
const RTLD_NOLOAD: c_int = 0x10;
#[cfg(not(target_os = "macos"))]
const RTLD_NOLOAD: c_int = 0x4;
const RTLD_LAZY: c_int = 0x1;

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flag: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
}

static SAME_LIBRARY: OnceLock<bool> = OnceLock::new();

/// Whether the SQLite Python's extension at `module_path` calls is the one
/// this crate links: the same `sqlite3_libversion`, at the same address.
fn same_library(module_path: &str) -> bool {
    let Ok(path) = CString::new(module_path) else {
        return false;
    };
    // SAFETY: NOLOAD returns a handle only to an image already in the
    // process, and dlsym only reads its symbol table.
    let theirs = unsafe {
        let handle = dlopen(path.as_ptr(), RTLD_LAZY | RTLD_NOLOAD);
        if handle.is_null() {
            return false;
        }
        dlsym(handle, c"sqlite3_libversion".as_ptr())
    };
    let ours = ffi::sqlite3_libversion as *const c_void;
    !theirs.is_null() && std::ptr::eq(theirs.cast_const(), ours)
}

/// Check once that the crate and Python's `_sqlite3` extension (at
/// `module_path`) share one SQLite. The first call decides; later calls
/// report that outcome.
pub fn bind(module_path: &str) -> bool {
    *SAME_LIBRARY.get_or_init(|| same_library(module_path))
}

/// Whether the crate may touch a database at all (see `bind`).
pub(crate) fn bound() -> bool {
    SAME_LIBRARY.get().copied().unwrap_or(false)
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

/// `"{what}: {SQLite's message}"`, as the reader reports a failure.
fn error(what: &str, e: rusqlite::Error) -> ReadError {
    let msg = match e {
        rusqlite::Error::SqliteFailure(_, Some(m)) => m,
        rusqlite::Error::SqlInputError { msg, .. } => msg,
        other => other.to_string(),
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
    if !bound() {
        return Err(ReadError("SQLite is not bound".into()));
    }
    let conn = Connection::open_with_flags(db_path, OpenFlags::SQLITE_OPEN_READ_WRITE)
        .map_err(|e| error("open", e))?;
    conn.busy_timeout(BUSY_TIMEOUT)
        .map_err(|e| error("open", e))?;

    let placeholders = vec!["?"; symbols.len()].join(", ");
    let sql = format!(
        "SELECT symbol, date, open, high, low, close, adj_close, volume \
         FROM daily_ohlcv \
         WHERE symbol IN ({placeholders}) AND interval = ? AND date BETWEEN ? AND ? \
         ORDER BY symbol, date ASC"
    );
    let mut stmt = conn.prepare(&sql).map_err(|e| error("prepare", e))?;
    let params = symbols
        .iter()
        .map(String::as_str)
        .chain([interval, start, end]);
    for (i, p) in params.enumerate() {
        stmt.raw_bind_parameter(i + 1, p)
            .map_err(|e| error("bind", e))?;
    }

    let mut out_symbols: Vec<String> = Vec::new();
    let mut offsets: Vec<i64> = Vec::new();
    let mut dates_ns: Vec<i64> = Vec::new();
    let mut cols: Vec<ColumnBuilder> = (0..6).map(|_| ColumnBuilder::new()).collect();

    let mut rows = stmt.raw_query();
    let mut n: i64 = 0;
    while let Some(row) = rows.next().map_err(|e| error("step", e))? {
        let cell = |i: usize| row.get_ref(i).map_err(|e| error("step", e));
        let ValueRef::Text(symbol) = cell(0)? else {
            return Ok(Outcome::Unsupported("a symbol that is not text".into()));
        };
        if out_symbols.last().map(String::as_bytes) != Some(symbol) {
            let Ok(s) = std::str::from_utf8(symbol) else {
                return Ok(Outcome::Unsupported("a symbol that is not UTF-8".into()));
            };
            out_symbols.push(s.to_owned());
            offsets.push(n);
        }
        let ValueRef::Text(day) = cell(1)? else {
            return Ok(Outcome::Unsupported("a date that is not text".into()));
        };
        let Some(ns) = parse_iso_day(day) else {
            return Ok(Outcome::Unsupported("a date that is not YYYY-MM-DD".into()));
        };
        dates_ns.push(ns);
        for (k, col) in cols.iter_mut().enumerate() {
            match cell(k + 2)? {
                ValueRef::Integer(v) => {
                    col.real.push(v as f64);
                    if col.all_int {
                        col.int.push(v);
                    }
                }
                ValueRef::Real(v) => {
                    col.real.push(v);
                    col.all_int = false;
                }
                ValueRef::Null => {
                    col.real.push(f64::NAN);
                    col.all_int = false;
                }
                ValueRef::Text(_) | ValueRef::Blob(_) => {
                    return Ok(Outcome::Unsupported(
                        "a price that is text or a blob".into(),
                    ));
                }
            }
        }
        n += 1;
    }
    offsets.push(n);

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
