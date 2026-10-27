"""How exactly a Rust float must match the frozen Python original.

The port was made bit for bit on Apple silicon, where NumPy's float64 exp,
log, log1p and power are the C library's, the same functions Rust's f64
methods call. On x86-64, NumPy dispatches those ufuncs to its own SIMD
kernels when the CPU has them (AVX-512 on CI's runners). The kernels differ
from glibc in the last bit, so the reference itself answers differently on
different CPUs. There, CI saw results 1 ulp apart, and up to 1.3e-14 relative
after cancellation. A port error is orders of magnitude larger, so off Apple
silicon a value may differ by at most 1e-12, relative or absolute.
"""

import math
import platform
import struct
import sys

import numpy as np

BITWISE = sys.platform == "darwin" and platform.machine() == "arm64"
TOLERANCE = 1e-12


def _dispatch() -> str:
    try:
        from numpy._core._multiarray_umath import __cpu_features__

        return ",".join(k for k, on in __cpu_features__.items() if on)
    except ImportError:  # NumPy 1.x
        return "unknown"


def floats_match(a: float, b: float) -> bool:
    """`a` and `b` are the same float, or close enough where bits cannot be."""
    if math.isnan(a) and math.isnan(b):
        return True
    if struct.pack("<d", a) == struct.pack("<d", b):
        return True
    return not BITWISE and math.isclose(a, b, rel_tol=TOLERANCE, abs_tol=TOLERANCE)


def mismatch(path: str, a: float, b: float) -> str:
    rule = "bitwise" if BITWISE else f"within {TOLERANCE:g}"
    return f"{path}: {a!r} vs {b!r} ({rule}; {platform.machine()}, NumPy {np.__version__}, CPU {_dispatch()})"
