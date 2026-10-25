"""Projections (rust/investa_core/src/projection.rs) against the original.

compute_projection, the walk-forward backtest, its summary and the end-to-end
backtest_projection must match tests/reference/projections.py bitwise, on
random wealth series: calendar-daily, business-daily and weekly dates, a plain
integer index, gaps, crashes past the clip, and partial benchmarks.
"""

import math
import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import projections as new  # noqa: E402
from reference import projections as ref  # noqa: E402


def _same(a, b, path="result"):
    if isinstance(a, pd.DataFrame):
        assert list(a.columns) == list(b.columns), path
        assert len(a) == len(b), f"{path}: {len(a)} vs {len(b)} rows"
        for col in a.columns:
            for i, (x, y) in enumerate(zip(a[col].tolist(), b[col].tolist())):
                _same(x, y, f"{path}[{col}][{i}]")
        return
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, list):
        assert len(a) == len(b), f"{path}: length"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _wealth(rng, years=None):
    years = float(rng.uniform(0.05, 14)) if years is None else years
    freq = str(rng.choice(["D", "B", "W"]))
    per_year = {"D": 365, "B": 252, "W": 52}[freq]
    n = max(2, int(years * per_year))
    steps = rng.normal(float(rng.normal(0.0003, 0.0004)), float(rng.choice([0.005, 0.012, 0.03])), n)
    if rng.random() < 0.2:
        steps[rng.integers(0, n)] = float(rng.choice([-0.95, 1.5]))  # past the clip
    values = 100.0 * np.cumprod(1.0 + steps)
    idx = pd.date_range("2008-01-02", periods=n, freq=freq)
    s = pd.Series(values, index=idx)
    if rng.random() < 0.15:
        keep = np.ones(n, dtype=bool)
        keep[rng.integers(0, n, max(1, n // 20))] = False  # holes in the history
        s = s[keep]
    if rng.random() < 0.1:
        s = s.reset_index(drop=True)  # no dates: positional horizons
    return s


def _bench(rng, wealth):
    if rng.random() < 0.3:
        return None
    b = _wealth(rng, years=max(0.1, len(wealth) / 252))
    if isinstance(wealth.index, pd.DatetimeIndex) and isinstance(b.index, pd.DatetimeIndex):
        return b
    return None


@pytest.mark.parametrize("seed", range(4))
def test_compute_projection_matches(seed):
    rng = np.random.default_rng(5100 + seed)
    for _ in range(300):
        w = _wealth(rng)
        cv = float(rng.choice([1.0, 1234.5, 2.5e6]))
        bench = None if rng.random() < 0.4 else float(rng.normal(0.07, 0.05))
        horizons = None if rng.random() < 0.6 else [k / 12.0 for k in range(1, int(rng.integers(2, 40)))]
        _same(new.compute_projection(w, cv, horizons, bench), ref.compute_projection(w, cv, horizons, bench))


@pytest.mark.parametrize("seed", range(3))
def test_walk_forward_and_summary_match(seed):
    rng = np.random.default_rng(5200 + seed)
    for _ in range(12):
        w = _wealth(rng, years=float(rng.uniform(5.5, 16)))
        bench = _bench(rng, w)
        horizons = sorted(rng.choice([1, 3, 5, 10], int(rng.integers(1, 5)), replace=False).tolist())
        step = None if rng.random() < 0.7 else int(rng.integers(1, 40))
        e_new = new.walk_forward_errors(w, bench, horizons, step=step)
        e_ref = ref.walk_forward_errors(w, bench, horizons, step=step)
        _same(e_new, e_ref, "errors")
        _same(new.summarize_errors(e_new, horizons), ref.summarize_errors(e_ref, horizons), "summary")


@pytest.mark.parametrize("seed", range(2))
def test_backtest_projection_matches(seed):
    rng = np.random.default_rng(5300 + seed)
    for _ in range(8):
        w = _wealth(rng, years=float(rng.uniform(3, 16)))
        bench = _bench(rng, w)
        values = None if rng.random() < 0.5 else w * float(rng.uniform(100, 1e5))
        # A series without dates plus a value series makes the original raise
        # (it subtracts a Timedelta from an integer position); the port must
        # fail the same way, not quietly succeed.
        outcomes = []
        for mod in (new, ref):
            try:
                outcomes.append(("ok", mod.backtest_projection(w, bench, values)))
            except Exception as exc:
                outcomes.append(("raised", type(exc)))
        assert outcomes[0][0] == outcomes[1][0], outcomes
        if outcomes[0][0] == "ok":
            _same(outcomes[0][1], outcomes[1][1])
        else:
            assert outcomes[0][1] is outcomes[1][1]
