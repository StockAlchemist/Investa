"""The history's stabilization fill (rust/investa_core/src/ph_performance.rs,
`stabilized`) against the pandas sequence it replaces:

    new_index = pd.Index([full_start_ts]).union(v.index)
    v = v.reindex(new_index).sort_index()
    v = v.interpolate(method="time").bfill().ffill()

On random price and FX tables (daily with weekend and holiday gaps, business
days, weekly, month ends, irregular intraday stamps, a regular daily index with
a freq; one to three float columns, an int64 Volume as the archive stores it
(up to 2**62, past exact float range) or every column int64; missing cells
inside, at either end and in whole columns; infinities; starts from a day to
decades early, as a date-made second-resolution stamp or a nanosecond one), the
two must agree bit for bit on every cell, and exactly on the index (values,
dtype, zone, freq, name), the columns and the dtypes. Tables the fill must hand
to pandas (a bool or text column, a Bangkok or naive index, duplicate, unsorted or NaT stamps, a
named index, attrs, a start that does not sort first) must come back as pandas
returns them.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402


def _pandas(v, ts):
    new_index = pd.Index([ts]).union(v.index)
    v = v.reindex(new_index).sort_index()
    return v.interpolate(method="time").bfill().ffill()


def _index(rng, n):
    start = pd.Timestamp("2015-03-02") + pd.Timedelta(days=int(rng.integers(0, 2000)))
    kind = int(rng.integers(0, 6))
    if kind == 0:
        idx = pd.date_range(start, periods=n * 2, freq="D")
        idx = idx[idx.dayofweek < 5][:n]  # weekdays, gaps at weekends
        idx = idx.delete(sorted(set(rng.integers(0, len(idx), n // 20)))) if n > 20 else idx
    elif kind == 1:
        idx = pd.bdate_range(start, periods=n)
    elif kind == 2:
        idx = pd.date_range(start, periods=n, freq="W-FRI")
    elif kind == 3:
        idx = pd.date_range(start, periods=n, freq="ME")
    elif kind == 4:
        offsets = np.cumsum(rng.integers(1, 5 * 86_400, n)) * 1_000_000_000
        idx = pd.DatetimeIndex(start.value + offsets)
    else:
        idx = pd.date_range(start, periods=n, freq="D")  # keeps its freq
    unit = ["s", "ms", "us", "ns"][int(rng.integers(0, 4))]  # pandas 3 picks it from the input
    return idx.tz_localize("UTC").as_unit(unit)


def _frame(rng):
    n = int(rng.integers(1, 400))
    idx = _index(rng, n)
    n = len(idx)
    cols = [["price"], ["price", "Volume"], ["a", "b", "c"]][int(rng.integers(0, 3))]
    data = {c: rng.normal(100, 20, n) for c in cols}
    for c in cols:
        r = rng.random()
        if r < 0.25 and n > 2:
            data[c][rng.integers(0, n, max(1, n // 8))] = np.nan
        elif r < 0.35:
            data[c][: int(rng.integers(1, n + 1))] = np.nan  # leading
        elif r < 0.45:
            data[c][int(rng.integers(0, n)) :] = np.nan  # trailing
        elif r < 0.5:
            data[c][:] = np.nan
        elif r < 0.55 and n > 1:
            data[c][int(rng.integers(0, n))] = np.inf
    if "Volume" in data and rng.random() < 0.7:  # as the archive stores it
        data["Volume"] = rng.integers(0, 2**62 if rng.random() < 0.2 else 10**9, n)  # past 2**53 too
    if rng.random() < 0.05:
        data = {c: rng.integers(-(10**6), 10**6, n) for c in cols}  # every column int64
    return pd.DataFrame(data, index=idx)


def _start(rng, first):
    days = int(rng.choice([1, 2, 7, 400, 9000]))
    day = (first - pd.Timedelta(days=days)).date()
    if rng.random() < 0.7:
        return pd.Timestamp(day, tz="UTC")  # second resolution, as the history makes it
    return pd.Timestamp(str(day), tz="UTC").as_unit("ns")


def _odd(rng, v):
    r = int(rng.integers(0, 10))
    if r == 0:
        return v.assign(**{v.columns[0]: np.arange(len(v)) % 2 == 0})  # a bool column
    if r == 1:
        return v.assign(note="x")
    if r == 2:
        return v.tz_convert("Asia/Bangkok")
    if r == 3:
        return v.tz_localize(None)
    if r == 4 and len(v) > 1:
        return pd.concat([v, v.iloc[:1]])  # a duplicate, out of order
    if r == 5 and len(v) > 1:
        return v.iloc[::-1]
    if r == 6:
        return v.rename_axis("Date")
    if r == 7:
        v = v.copy()
        v.attrs["source"] = "db"
        return v
    if r == 8 and len(v) > 1:
        idx = v.index.tolist()
        idx[-1] = pd.NaT
        return v.set_axis(pd.DatetimeIndex(idx))
    return v.iloc[:0]


def _same(a, b):
    pd.testing.assert_index_equal(a.index, b.index, exact=True)
    assert a.index.freq == b.index.freq and a.index.name == b.index.name and repr(a.index.tz) == repr(b.index.tz)
    pd.testing.assert_index_equal(a.columns, b.columns, exact=True)
    assert list(a.dtypes) == list(b.dtypes)
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype == np.float64:
            assert np.array_equal(x.view(np.int64), y.view(np.int64)), c  # bit for bit
        else:
            assert (pd.isna(x) == pd.isna(y)).all() and (x[~pd.isna(x)] == y[~pd.isna(y)]).all(), c


@pytest.mark.parametrize("seed", range(8))
def test_fill_matches_pandas(seed):
    rng = np.random.default_rng(33000 + seed)
    for _ in range(150):
        v = _frame(rng)
        ts = _start(rng, v.index[0])
        _same(_pandas(v.copy(), ts), investa_core.ph_stabilize(v.copy(), ts))


PATTERNS = [  # (index, starts that keep or break its rhythm)
    (pd.date_range("2020-01-02", periods=60, freq="D"), [1, 2, 30]),
    (pd.bdate_range("2020-01-06", periods=60), [3, 1, 2, 10]),  # a Monday: 3 days back is a Friday
    (pd.date_range("2020-01-03", periods=60, freq="W-FRI"), [7, 14, 6, 400]),
    (pd.date_range("2020-01-01", periods=40, freq="WOM-3FRI"), [28, 35, 7]),
    (pd.date_range("2020-02-01", periods=40, freq="MS"), [31, 29, 62]),
    (pd.date_range("2020-03-31", periods=40, freq="QE"), [91, 92, 90]),
    (pd.date_range("2020-01-01", periods=20, freq="YS"), [365, 366, 731]),
    (pd.date_range("2020-01-06 09:00", periods=60, freq="bh"), [1, 2]),
    (pd.date_range("2020-01-06 09:00", periods=60, freq="h"), [1, 3]),
    (pd.date_range("2020-01-06", periods=2, freq="D"), [1]),
]


@pytest.mark.parametrize("which", range(len(PATTERNS)))
def test_inferred_freq_matches_pandas(which):
    idx, backs = PATTERNS[which]
    v = pd.DataFrame({"price": np.arange(len(idx), dtype=float)}, index=idx.tz_localize("UTC"))
    for back in backs:
        for step in ("D", "h"):
            ts = (v.index[0] - pd.Timedelta(back, unit=step)).floor(step)
            _same(_pandas(v.copy(), ts), investa_core.ph_stabilize(v.copy(), ts))


@pytest.mark.parametrize("seed", range(4))
def test_declined_tables_match_pandas(seed):
    rng = np.random.default_rng(33100 + seed)
    for _ in range(80):
        v = _odd(rng, _frame(rng))
        first = v.index.dropna().min() if len(v.index.dropna()) else pd.Timestamp("2020-01-01", tz="UTC")
        if first is pd.NaT or first.tz is None:
            first = pd.Timestamp(first if first is not pd.NaT else "2020-01-01").tz_localize("UTC")
        ts = _start(rng, first)
        if rng.random() < 0.1:
            ts = first.tz_convert("UTC") if first.tz else first  # a start that does not sort first
        try:
            expected = ("ok", _pandas(v.copy(), ts))
        except Exception as exc:  # noqa: BLE001 -- the exception is the answer
            expected = ("raised", type(exc).__name__, str(exc))
        try:
            got = ("ok", investa_core.ph_stabilize(v.copy(), ts))
        except Exception as exc:  # noqa: BLE001
            got = ("raised", type(exc).__name__, str(exc))
        assert expected[0] == got[0], (expected, got)
        if expected[0] == "ok":
            _same(expected[1], got[1])
        else:
            assert expected == got


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_real_shape_takes_the_fast_path(unit, monkeypatch):
    v = pd.DataFrame({"price": [1.0, np.nan, 3.0], "Volume": [10, 20, 30]},
                     index=pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-05"], tz="UTC").as_unit(unit))  # fmt: skip
    ts = pd.Timestamp(dt.date(2023, 12, 1), tz="UTC")
    expected = _pandas(v, ts)

    def refuse(*args, **kwargs):
        raise AssertionError("the fill fell back to pandas")

    monkeypatch.setattr(pd.DataFrame, "interpolate", refuse)  # the fill must not need it
    out = investa_core.ph_stabilize(v, ts)
    monkeypatch.undo()
    _same(expected, out)
