//! Small helpers that reproduce Python semantics the ported kernels rely on.

use std::fmt;

/// An index that came from Python and does not fit the array it addresses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexOutOfRange {
    pub what: &'static str,
    pub index: i64,
    pub len: usize,
}

impl fmt::Display for IndexOutOfRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} index {} is out of range for length {}",
            self.what, self.index, self.len
        )
    }
}

impl std::error::Error for IndexOutOfRange {}

/// Why a kernel refused its inputs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KernelError {
    Index(IndexOutOfRange),
    /// A per-transaction array shorter or longer than the transaction count.
    /// Numba would read past the end; the port refuses instead.
    Length {
        what: &'static str,
        expected: usize,
        got: usize,
    },
}

impl fmt::Display for KernelError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelError::Index(e) => e.fmt(f),
            KernelError::Length {
                what,
                expected,
                got,
            } => write!(f, "{what} has length {got}, expected {expected}"),
        }
    }
}

impl std::error::Error for KernelError {}

impl From<IndexOutOfRange> for KernelError {
    fn from(e: IndexOutOfRange) -> Self {
        KernelError::Index(e)
    }
}

/// Every per-transaction array must have exactly `expected` entries.
pub fn check_len(what: &'static str, got: usize, expected: usize) -> Result<(), KernelError> {
    if got == expected {
        Ok(())
    } else {
        Err(KernelError::Length {
            what,
            expected,
            got,
        })
    }
}

/// Resolve a Python-side index the way NumPy (and Numba) do: a negative index
/// counts from the end. Unlike Numba with bounds checking off, an index that is
/// still out of range is an error rather than a read of whatever lies there.
#[inline]
pub fn wrap(what: &'static str, index: i64, len: usize) -> Result<usize, IndexOutOfRange> {
    let n = len as i64;
    let resolved = if index < 0 { index + n } else { index };
    if resolved < 0 || resolved >= n {
        Err(IndexOutOfRange { what, index, len })
    } else {
        Ok(resolved as usize)
    }
}

/// What a Python number was, which decides how `sum()` adds it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyNum {
    /// An exact `float`: summed with Neumaier compensation.
    Float,
    /// An `int` (or `bool`).
    Int,
    /// Any other float type, e.g. `numpy.float64`: drops the sum out of the
    /// compensated fast path into plain sequential addition.
    Other,
}

impl PyNum {
    pub fn from_code(code: u8) -> Self {
        match code {
            0 => PyNum::Float,
            1 => PyNum::Int,
            _ => PyNum::Other,
        }
    }
}

/// CPython 3.12's built-in `sum(items)` with the default start of `0`, for
/// items that are all small ints or floats (Python/bltinmodule.c,
/// builtin_sum_impl). Returns `None` for an empty input, where Python returns
/// the int `0`.
///
/// While the running total and every item are exact floats it uses Neumaier
/// compensated summation; the first non-exact float folds the compensation in
/// and continues with plain addition, as CPython does.
pub fn py_sum(items: &[(f64, PyNum)]) -> Option<f64> {
    enum State {
        Int,
        Float { f: f64, c: f64 },
        Plain(f64),
    }
    if items.is_empty() {
        return None;
    }
    let mut state = State::Int;
    let mut int_total = 0.0_f64;
    for &(x, kind) in items {
        state = match state {
            // int + int stays int; int + float becomes 0.0 + x in that type.
            State::Int => match kind {
                PyNum::Int => {
                    int_total += x;
                    State::Int
                }
                PyNum::Float => State::Float {
                    f: int_total + x,
                    c: 0.0,
                },
                PyNum::Other => State::Plain(int_total + x),
            },
            State::Float { f, c } => match kind {
                PyNum::Float => {
                    let t = f + x;
                    let c = if f.abs() >= x.abs() {
                        c + ((f - t) + x)
                    } else {
                        c + ((x - t) + f)
                    };
                    State::Float { f: t, c }
                }
                PyNum::Int => State::Float { f: f + x, c },
                PyNum::Other => {
                    let mut f = f;
                    if c != 0.0 && c.is_finite() {
                        f += c;
                    }
                    State::Plain(f + x)
                }
            },
            State::Plain(g) => State::Plain(g + x),
        };
    }
    Some(match state {
        State::Int => int_total,
        State::Float { f, c } => {
            if c != 0.0 && c.is_finite() {
                f + c
            } else {
                f
            }
        }
        State::Plain(g) => g,
    })
}

/// Python's two-argument `min`: the first argument unless the second is
/// strictly smaller. `f64::min` differs when either side is NaN.
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 {
    if b < a { b } else { a }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wrap_matches_numpy() {
        assert_eq!(wrap("x", 0, 3), Ok(0));
        assert_eq!(wrap("x", 2, 3), Ok(2));
        assert_eq!(wrap("x", -1, 3), Ok(2));
        assert_eq!(wrap("x", -3, 3), Ok(0));
        assert!(wrap("x", 3, 3).is_err());
        assert!(wrap("x", -4, 3).is_err());
        assert!(wrap("x", 0, 0).is_err());
    }

    #[test]
    fn py_min_matches_python() {
        assert_eq!(py_min(1.0, 2.0), 1.0);
        assert_eq!(py_min(2.0, 1.0), 1.0);
        // min(nan, 1.0) is nan in Python; min(1.0, nan) is 1.0.
        assert!(py_min(f64::NAN, 1.0).is_nan());
        assert_eq!(py_min(1.0, f64::NAN), 1.0);
    }
}
