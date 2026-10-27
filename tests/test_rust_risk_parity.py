"""risk_metrics (Rust arithmetic) against the frozen original: bitwise.

calculate_beta keeps pandas' covariance (np.cov's BLAS dot product cannot be
reproduced), so Beta and Alpha are bitwise equal too.
"""

import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402
import risk_metrics as new  # noqa: E402
from reference import risk_metrics as ref  # noqa: E402



def _same(a, b):
    a, b = float(a), float(b)
    return (np.isnan(a) and np.isnan(b)) or struct.pack("<d", a) == struct.pack("<d", b)


def _series(rng, n=None, start="2015-01-01"):
    n = int(rng.integers(0, 900)) if n is None else n
    freq = str(rng.choice(["D", "B", "W"]))
    idx = pd.date_range(start, periods=n, freq=freq)
    steps = rng.normal(0.0004, float(rng.choice([0.001, 0.01, 0.05, 0.4])), n)
    values = 1000.0 * np.cumprod(1.0 + steps)
    r = rng.random()
    if r < 0.1 and n:
        values[rng.random(n) < 0.05] = np.nan
    elif r < 0.15 and n:
        values[rng.integers(0, n)] = 0.0  # a zero: inf return, then NaN
    elif r < 0.2:
        values = -values  # a negative "value" series
    s = pd.Series(values, index=idx)
    if rng.random() < 0.05:
        s = s.round().astype("int64") if not s.isna().any() else s
    return s


@pytest.mark.parametrize("seed", range(4))
def test_all_risk_metrics_match_reference(seed):
    rng = np.random.default_rng(600 + seed)
    for _ in range(150):
        port = _series(rng)
        bench = None
        if rng.random() < 0.7:
            offset = pd.Timedelta(days=int(rng.integers(-200, 200)))
            bench = _series(rng, start=str((pd.Timestamp("2015-01-01") + offset).date()))
        ppy = None if rng.random() < 0.7 else int(rng.choice([12, 52, 252, 365]))
        rf = float(rng.choice([0.0, 0.02, 0.05]))
        want = ref.calculate_all_risk_metrics(port, rf, ppy, bench)
        got = new.calculate_all_risk_metrics(port, rf, ppy, bench)
        assert set(got) == set(want)
        for key in want:
            assert _same(got[key], want[key]), (key, got[key], want[key])


@pytest.mark.parametrize("seed", range(2))
def test_drawdown_series_matches_reference(seed):
    rng = np.random.default_rng(700 + seed)
    for _ in range(300):
        s = _series(rng)
        want = ref.calculate_drawdown_series(s)
        got = new.calculate_drawdown_series(s)
        assert got.index.equals(want.index)
        assert got.to_numpy(dtype=float).tobytes() == want.to_numpy(dtype=float).tobytes()
        assert _same(new.calculate_max_drawdown(s), ref.calculate_max_drawdown(s))


@pytest.mark.parametrize("seed", range(2))
def test_scoreboard_matches_reference(seed):
    rng = np.random.default_rng(800 + seed)
    for _ in range(80):
        port = _series(rng, n=int(rng.integers(0, 700)))
        benches = {
            f"B{i}": _series(rng, n=int(rng.integers(0, 700)))
            for i in range(int(rng.integers(0, 4)))
        }
        want = ref.calculate_benchmark_scoreboard(port, benches)
        got = new.calculate_benchmark_scoreboard(port, benches)
        assert [r["name"] for r in got] == [r["name"] for r in want]
        for g, w in zip(got, want):
            for key in w:
                if key != "name":
                    assert _same(g[key], w[key]), (key, g[key], w[key])


def test_single_metrics_match_reference():
    rng = np.random.default_rng(900)
    for _ in range(400):
        r = pd.Series(rng.normal(0.0005, 0.02, int(rng.integers(0, 300))))
        if rng.random() < 0.2:
            r[r.abs() < 0.005] = 0.0
        if rng.random() < 0.1:
            r[:] = abs(r)  # no downside at all
        for fn in ("calculate_volatility",):
            assert _same(getattr(new, fn)(r, 252), getattr(ref, fn)(r, 252))
        assert _same(new.calculate_sharpe_ratio(r, 0.02, 365), ref.calculate_sharpe_ratio(r, 0.02, 365))
        assert _same(
            new.calculate_sortino_ratio(r, 0.03, 252, 0.0),
            ref.calculate_sortino_ratio(r, 0.03, 252, 0.0),
        )


def test_pandas_reductions_match_pandas():
    rng = np.random.default_rng(1000)
    for _ in range(2000):
        # Past 8192 NumPy reduces buffer by buffer; long arrays must agree too.
        n = int(rng.integers(0, 700)) if rng.random() < 0.8 else int(rng.integers(8000, 30000))
        v = rng.normal(0, 10 ** float(rng.integers(-6, 9)), n)
        if rng.random() < 0.3 and n:
            v[rng.random(n) < 0.2] = np.nan
        s = pd.Series(v)
        assert _same(investa_core.pd_mean(v), s.mean())
        assert _same(investa_core.pd_var(v, 1), s.var())
        assert _same(investa_core.pd_std(v, 0), s.std(ddof=0))
        assert _same(investa_core.pd_min(v), s.min())
        assert _same(investa_core.np_sum(v), np.add.reduce(v))
    for v in ([-0.0], [-0.0] * 9, [-0.0] * 9000, [-0.0, 1.0, -1.0]):
        a = np.array(v)
        assert _same(investa_core.np_sum(a), np.add.reduce(a))
        assert _same(investa_core.pd_mean(a), pd.Series(a).mean())
