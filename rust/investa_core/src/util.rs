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
