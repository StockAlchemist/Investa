"""Phase 5b: valuation bands, the trend signal and stress response (Rust) against
the originals frozen in tests/reference/.

Bitwise and type-for-type: the band statistics on random positive
populations, every field of evaluate_trend_signal on random month-end
histories and window lengths (including the one-month average, whose
`iloc[-0:]` is the whole history), and stress_response on random annual
series with gaps, losses, NaN and ties.
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

import investa_core  # noqa: E402
import strategies as new_strategies  # noqa: E402
import track_record as new_track  # noqa: E402
from reference import strategies as ref_strategies  # noqa: E402
from reference import track_record as ref_track  # noqa: E402


def _same(a, b, path="result"):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys"
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


def test_band_stats_match_numpy():
    rng = np.random.default_rng(6100)
    for _ in range(3000):
        n = int(rng.integers(1, 40))
        pop = [float(x) for x in np.abs(rng.lognormal(2.5, 0.6, n))]
        if rng.random() < 0.2:
            pop = [float(rng.choice([10.0, 12.0, 15.0])) for _ in range(n)]  # ties
        now = float(rng.choice(pop)) if rng.random() < 0.3 else float(rng.lognormal(2.5, 0.8))
        want = (
            float(np.median(pop)), float(np.percentile(pop, 25)), float(np.percentile(pop, 75)),
            float(np.min(pop)), float(np.max(pop)),
            100.0 * sum(1 for v in pop if v <= now) / len(pop),
        )  # fmt: skip
        _same(list(investa_core.band_stats(pop, now)), list(want), "band_stats")


@pytest.mark.parametrize("seed", range(4))
def test_trend_signal_matches(seed):
    rng = np.random.default_rng(6200 + seed)
    for _ in range(150):
        n_days = int(rng.integers(20, 2500))
        idx = pd.bdate_range("2014-01-02", periods=n_days)
        prices = pd.Series(400 * np.cumprod(1 + rng.normal(0.0004, 0.012, n_days)), index=idx)
        if rng.random() < 0.1:
            prices.iloc[rng.integers(0, n_days, 5)] = np.nan
        sma_months = int(rng.choice([1, 2, 3, 10, 12]))
        today = (idx[-1] + pd.Timedelta(days=int(rng.integers(0, 40)))).date()
        _same(
            new_strategies.evaluate_trend_signal(prices, sma_months, today),
            ref_strategies.evaluate_trend_signal(prices, sma_months, today),
            "trend",
        )


def _annual(rng):
    years = sorted(set(int(y) for y in rng.integers(2004, 2026, int(rng.integers(0, 20)))))
    out = {}
    for y in years:
        r = rng.random()
        out[f"{y}-12-31"] = (
            float("nan") if r < 0.03 else float(rng.choice([1e8, -2e7])) if r < 0.2 else float(rng.normal(1e9, 4e8))
        )
    return out


@pytest.mark.parametrize("seed", range(4))
def test_stress_response_matches(seed):
    rng = np.random.default_rng(6300 + seed)
    for _ in range(600):
        values = {
            "revenue": _annual(rng),
            "net_income": _annual(rng),
            "operating_cash_flow": _annual(rng),
            "capex": {k: float(abs(v)) * 0.3 for k, v in _annual(rng).items()},
        }
        _same(new_track.stress_response(values), ref_track.stress_response(values), "stress")
