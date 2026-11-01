"""`align_prices_to_grid` in Rust (rust/investa_core/src/ph_small.rs) against
the pandas it replaces:

    raw_prices = pd.DataFrame(price_series_by_id).sort_index()
    fill_index = raw_prices.index.union(date_range)
    return raw_prices.reindex(fill_index).ffill().bfill().reindex(date_range)

Random grids (daily, business-day, weekly, intraday, with and without a freq)
and random symbols (daily closes, weekly or monthly NAVs, intraday bars between
grid points, prices starting before, inside or after the grid or ending early,
missing prices inside or at either end, all missing, infinities, names) must
come back bit for bit, with the same index (values, dtype, zone, freq, name),
columns and dtypes. Inputs the sweep must hand to pandas (an empty dict, an
integer series, a naive or Bangkok index, duplicate or unsorted stamps, a
grid with NaT or in another zone, a DataFrame among the series) must come back
as pandas returns them, or raise as it raises.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402


def _pandas(series_by_id, date_range):
    raw_prices = pd.DataFrame(series_by_id).sort_index()
    fill_index = raw_prices.index.union(date_range)
    return raw_prices.reindex(fill_index).ffill().bfill().reindex(date_range)


def _grid(rng):
    unit = ["s", "ms", "us", "ns"][int(rng.integers(0, 4))]  # pandas 3 picks it from the input
    return _grid_ns(rng).as_unit(unit)


def _grid_ns(rng):
    start = pd.Timestamp("2018-01-01", tz="UTC") + pd.Timedelta(days=int(rng.integers(0, 900)))
    n = int(rng.integers(1, 500))
    kind = int(rng.integers(0, 5))
    if kind == 0:
        return pd.date_range(start, periods=n, freq="D")
    if kind == 1:
        return pd.bdate_range(start, periods=n, tz=None).tz_localize(None).tz_localize("UTC")
    if kind == 2:
        return pd.date_range(start, periods=n, freq="W-FRI")
    if kind == 3:
        return pd.date_range(start + pd.Timedelta(hours=14, minutes=30), periods=n, freq="30min")
    g = pd.date_range(start, periods=n, freq="D")
    return pd.DatetimeIndex(g.as_unit("ns").asi8, tz="UTC", name="Date")  # no freq, a name


def _series(rng, grid):
    span = grid[-1] - grid[0]
    start = grid[0] + span * float(rng.uniform(-0.5, 1.2)) - pd.Timedelta(days=int(rng.integers(0, 30)))
    n = int(rng.integers(1, 300))
    step = [pd.Timedelta(days=1), pd.Timedelta(days=7), pd.Timedelta(days=30), pd.Timedelta(minutes=45)][int(rng.integers(0, 4))]
    idx = pd.DatetimeIndex(start + step * np.arange(n)).floor("min")
    idx = idx[~idx.duplicated()]
    values = rng.normal(100, 30, len(idx))
    r = rng.random()
    if r < 0.2:
        values[rng.integers(0, len(values), max(1, len(values) // 6))] = np.nan
    elif r < 0.3:
        values[: int(rng.integers(1, len(values) + 1))] = np.nan
    elif r < 0.4:
        values[int(rng.integers(0, len(values))) :] = np.nan
    elif r < 0.45:
        values[:] = np.nan
    elif r < 0.5:
        values[int(rng.integers(0, len(values)))] = -np.inf
    idx = idx.as_unit(["s", "ms", "us", "ns"][int(rng.integers(0, 4))])  # units may differ from the grid's
    return pd.Series(values, index=idx, name=["price", None, "Close"][int(rng.integers(0, 3))])


def _same(a, b):
    pd.testing.assert_index_equal(a.index, b.index, exact=True)
    assert a.index.freq == b.index.freq and a.index.name == b.index.name
    pd.testing.assert_index_equal(a.columns, b.columns, exact=True)
    assert list(a.dtypes) == list(b.dtypes)
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype == np.float64:
            assert np.array_equal(x.view(np.int64), y.view(np.int64)), c  # bit for bit
        else:
            assert (pd.isna(x) == pd.isna(y)).all(), c


def _outcome(fn):
    try:
        return ("ok", fn())
    except Exception as exc:  # noqa: BLE001 -- the exception is the answer
        return ("raised", type(exc).__name__, str(exc))


@pytest.mark.parametrize("seed", range(8))
def test_alignment_matches_pandas(seed):
    rng = np.random.default_rng(34000 + seed)
    for _ in range(60):
        grid = _grid(rng)
        series = {int(k): _series(rng, grid) for k in rng.choice(200, int(rng.integers(1, 12)), replace=False)}
        _same(_pandas(dict(series), grid), investa_core.ph_align_prices_to_grid(dict(series), grid))


def _odd(rng, series, grid):
    r = int(rng.integers(0, 9))
    k = next(iter(series))
    s = series[k]
    if r == 0:
        return {}, grid
    if r == 1:
        series[k] = pd.Series(np.arange(len(s)), index=s.index)
    elif r == 2:
        series[k] = pd.Series(s.to_numpy(), index=s.index.tz_localize(None))  # naive
    elif r == 3:
        series[k] = s.tz_convert("Asia/Bangkok")
    elif r == 4 and len(s) > 1:
        series[k] = pd.concat([s, s.iloc[:1]])  # a duplicate, out of order
    elif r == 5 and len(s) > 1:
        series[k] = s.iloc[::-1]
    elif r == 6:
        grid = grid.tz_convert("Asia/Bangkok")
    elif r == 7 and len(grid) > 1:
        grid = pd.DatetimeIndex(list(grid[:-1]) + [pd.NaT], tz="UTC")
    else:
        series[k] = s.to_frame()
    return series, grid


@pytest.mark.parametrize("seed", range(4))
def test_declined_inputs_match_pandas(seed):
    rng = np.random.default_rng(34100 + seed)
    for _ in range(60):
        grid = _grid(rng)
        series = {int(k): _series(rng, grid) for k in rng.choice(200, int(rng.integers(1, 6)), replace=False)}
        series, grid = _odd(rng, series, grid)
        a = _outcome(lambda: _pandas(dict(series), grid))
        b = _outcome(lambda: investa_core.ph_align_prices_to_grid(dict(series), grid))
        assert a[0] == b[0], (a, b)
        if a[0] == "ok":
            _same(a[1], b[1])
        else:
            assert a == b


@pytest.mark.parametrize("grid_unit,series_unit", [("ns", "ns"), ("s", "us"), ("us", "s"), ("ms", "ns")])
def test_real_shape_takes_the_sweep(grid_unit, series_unit, monkeypatch):
    grid = pd.date_range("2024-01-01", periods=5, freq="D", tz="UTC").as_unit(grid_unit)
    s = pd.Series([1.0, np.nan, 3.0], index=pd.DatetimeIndex(["2024-01-02 21:00", "2024-01-03 21:00", "2024-01-08"], tz="UTC").as_unit(series_unit), name="price")
    expected = _pandas({7: s}, grid)

    def refuse(*args, **kwargs):
        raise AssertionError("the alignment fell back to pandas")

    monkeypatch.setattr(pd.DataFrame, "ffill", refuse)  # only the pandas path fills
    out = investa_core.ph_align_prices_to_grid({7: s}, grid)
    monkeypatch.undo()
    assert out.index is grid
    assert out[7].tolist() == [1.0, 1.0, 1.0, 1.0, 1.0]  # Jan 3's NaN carries Jan 2's close; Jan 8 is past the grid
    _same(expected, out)
