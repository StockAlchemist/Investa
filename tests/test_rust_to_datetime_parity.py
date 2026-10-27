"""The `pd.to_datetime(..., utc=True)` fast path (rust/investa_core/src/pyhelp.rs)
against pandas itself.

Holding datetimes already, pandas only converts the time zone, but first it
probes the leading elements for repeats in a Python set, and that probe is the
whole cost. The Rust helper runs the probe on the int64 values and does the
conversion itself, so every case here must come out exactly as pandas returns
it: type, values, dtype, time zone, freq, name and index, or the same exception.
The inputs cover the shapes the fast path takes (DatetimeIndex and datetime
Series; naive, UTC and other zones; s/ms/us/ns; NaT; freq; names; lengths
around 50 and 5,000; repeats either side of the 70% cache threshold) and the
shapes it must hand back to pandas (strings, objects, arrays, scalars, lists,
and any keyword but `utc=True` and `errors`).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402

KWARGS = [
    {"utc": True},
    {"utc": True, "errors": "coerce"},
    {"utc": True, "errors": "raise"},
    {"errors": "coerce", "utc": True},
    {"utc": 1},
    {"utc": False},
    {},
    {"errors": "coerce"},
    {"utc": True, "format": "%Y-%m-%d"},
    {"utc": True, "cache": False},
]


def _values(rng, n, distinct_share=None, nat=False):
    base = np.datetime64("2001-01-01", "ns") + np.arange(n) * np.timedelta64(1, "D")
    if distinct_share is not None and n:
        k = max(1, int(round(n * distinct_share)))
        base = base[rng.integers(0, k, n)] if distinct_share < 1 else base
    if nat and n:
        base = base.copy()
        base[rng.integers(0, n, max(1, n // 7))] = np.datetime64("NaT")
    return base


def _threshold_values(distinct):
    """1,000 values whose first 100 (the probe) hold exactly `distinct` dates."""
    head = np.datetime64("2010-01-01", "ns") + (np.arange(100) % distinct) * np.timedelta64(1, "D")
    tail = np.datetime64("2015-01-01", "ns") + np.arange(900) * np.timedelta64(1, "D")
    return np.concatenate([head, tail])


def _inputs():
    rng = np.random.default_rng(31000)
    out = []
    for n in (0, 1, 50, 51, 120, 4999, 5000, 5001, 7000):
        for tz in (None, "UTC", "Asia/Bangkok", "America/New_York"):
            for unit in ("ns", "us", "s"):
                for share, nat in ((None, False), (None, True), (0.05, False), (0.6, True)):
                    vals = _values(rng, n, share, nat).astype(f"M8[{unit}]")
                    idx = pd.DatetimeIndex(vals, name=["Date", None, 0][int(rng.integers(0, 3))])
                    if tz:
                        idx = idx.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
                    out.append(idx)
                    out.append(pd.Series(idx, index=pd.RangeIndex(n) * 2, name=idx.name))
    for distinct in (69, 70, 71, 100):
        for tz in (None, "UTC", "Asia/Bangkok"):
            idx = pd.DatetimeIndex(_threshold_values(distinct))
            idx = idx.tz_localize(tz) if tz else idx
            out += [idx, pd.Series(idx, name="d")]
    out.append(pd.date_range("2020-01-01", periods=400, freq="D", tz="UTC", name="Date"))
    out.append(pd.date_range("2020-01-01", periods=400, freq="h"))
    out.append(pd.date_range("2020-03-01", periods=400, freq="B", tz="America/New_York"))
    frame = pd.DataFrame({"v": range(300)}, index=pd.date_range("2021-01-01", periods=300, tz="UTC"))
    out += [frame.index, frame["v"].index, pd.Series(frame.index)]
    # Shapes that must go to pandas.
    strings = pd.Series([f"2024-01-{d:02d}" for d in range(1, 29)] * 4)
    out += [strings, pd.Index(strings), strings.astype(object), pd.Series(["2024-01-01", "x", None] * 30)]
    out += [pd.Series(list(pd.date_range("2024-01-01", periods=80, tz="UTC")), dtype=object)]
    out += [pd.date_range("2024-01-01", periods=80, tz="UTC").array, pd.Timestamp("2024-01-01", tz="Asia/Bangkok")]
    out += [list(pd.date_range("2024-01-01", periods=80)), "2024-05-06", None, pd.Series([1.5e18, 1.6e18] * 40)]
    try:
        import pyarrow  # noqa: F401 -- optional; CI does not install it
    except ImportError:
        return out
    out += [pd.Series(pd.date_range("2024-01-01", periods=80), dtype="timestamp[ns][pyarrow]")]
    return out


def _outcome(fn):
    try:
        return "ok", fn()
    except BaseException as exc:  # noqa: BLE001 -- the exception is the result
        return "raised", type(exc).__name__, str(exc)


def _same(a, b, arg):
    assert a[0] == b[0], (a, b)
    if a[0] == "raised":
        assert a == b
        return
    x, y = a[1], b[1]
    assert type(x) is type(y)
    assert (x is arg) == (y is arg)
    if isinstance(x, pd.Index):
        pd.testing.assert_index_equal(x, y, exact=True)
        assert x.freq == y.freq and repr(x.dtype) == repr(y.dtype) and x.name == y.name
    elif isinstance(x, pd.Series):
        pd.testing.assert_series_equal(x, y, check_freq=True, check_index_type=True)
        assert repr(x.dtype) == repr(y.dtype) and x.index.equals(y.index) and x.index is y.index
    elif x is None or isinstance(x, (pd.Timestamp, type(pd.NaT))):
        assert x == y or (x is y)
    else:
        assert np.array_equal(np.asarray(x), np.asarray(y)) and repr(x.dtype) == repr(y.dtype)


@pytest.mark.parametrize("kwargs", KWARGS, ids=lambda k: ",".join(f"{a}={b}" for a, b in k.items()) or "none")
def test_matches_pandas(kwargs):
    for arg in _inputs():
        expected = _outcome(lambda: pd.to_datetime(arg, **kwargs))
        got = _outcome(lambda: investa_core.to_datetime(arg, **kwargs))
        _same(expected, got, arg)


def test_fast_path_leaves_its_input_alone():
    idx = pd.date_range("2020-01-01", periods=600, tz="Asia/Bangkok", name="Date")
    before = idx.copy()
    out = investa_core.to_datetime(idx, utc=True)
    pd.testing.assert_index_equal(idx, before, exact=True)
    assert str(out.tz) == "UTC" and out is not idx
