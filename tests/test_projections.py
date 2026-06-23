"""Tests for the portfolio-value projection model (projections.compute_projection).

Covers the two backtested refinements: parameter-uncertainty band widening and
drift shrinkage toward a benchmark prior.
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from projections import compute_projection  # noqa: E402


def _wealth(years=8, mu=0.10, sigma=0.15, seed=0):
    """A synthetic daily wealth index with ~mu drift and ~sigma annual vol."""
    rng = np.random.default_rng(seed)
    n = int(years * 252)
    daily = rng.normal(mu / 252, sigma / math.sqrt(252), n)
    return pd.Series((1.0 + daily).cumprod() * 100.0)


def _median(res, h):
    return next(p for p in res["horizons"] if p["years"] == h)["median_value"]


def test_unavailable_for_short_or_missing_data():
    assert compute_projection(None, 100.0)["available"] is False
    assert compute_projection(_wealth(), None)["available"] is False
    assert compute_projection(_wealth(), -5.0)["available"] is False
    assert compute_projection(pd.Series([1.0, 2.0, 3.0]), 100.0)["available"] is False


def test_bands_are_ordered_and_median_consistent():
    res = compute_projection(_wealth(seed=1), 100.0)
    assert res["available"] is True
    for p in res["horizons"]:
        assert p["p10"] < p["p25"] < p["median_value"] < p["p75"] < p["p90"]
        # median_return_pct and median_value describe the same point.
        assert p["median_return_pct"] == pytest.approx((p["median_value"] / 100.0 - 1.0) * 100.0)


def test_parameter_uncertainty_widens_faster_than_sqrt_t():
    # The predictive spread is sigma*sqrt(t*(1+t/N)), so the band half-width grows
    # FASTER than the naive sqrt(t) — that's the overconfidence fix.
    res = compute_projection(_wealth(years=5, seed=2), 100.0)
    by_year = {p["years"]: p for p in res["horizons"]}

    def half(h):  # band half-width in log space ∝ the predictive sd
        return math.log(by_year[h]["p90"] / by_year[h]["median_value"])

    # With ~5y of history, spread(10)/spread(1) must exceed sqrt(10) (pure process scaling).
    assert half(10) / half(1) > math.sqrt(10)


def test_drift_shrinkage_pulls_median_toward_benchmark():
    w = _wealth(years=8, mu=0.20, sigma=0.15, seed=3)
    v0 = float(w.iloc[-1])
    base = compute_projection(w, v0)  # no benchmark -> raw drift
    raw = math.log(1.0 + base["annual_return_pct"] / 100.0)
    low = compute_projection(w, v0, benchmark_log_return=raw - 0.10)
    high = compute_projection(w, v0, benchmark_log_return=raw + 0.10)
    # A lower prior drags the long-horizon median down; a higher prior lifts it.
    assert _median(low, 10) < _median(base, 10) < _median(high, 10)


def test_more_history_means_less_shrinkage():
    # With the same data pattern, a longer history trusts the data more, so the
    # shrunk drift sits closer to the raw drift (less pull toward the prior).
    short = _wealth(years=4, mu=0.18, sigma=0.15, seed=4)
    long = _wealth(years=20, mu=0.18, sigma=0.15, seed=4)
    prior = 0.0  # shrink toward zero drift
    raw_s = math.log(1.0 + compute_projection(short, 100.0)["annual_return_pct"] / 100.0)
    raw_l = math.log(1.0 + compute_projection(long, 100.0)["annual_return_pct"] / 100.0)
    sh_s = math.log(1.0 + compute_projection(short, 100.0, benchmark_log_return=prior)["annual_return_pct"] / 100.0)
    sh_l = math.log(1.0 + compute_projection(long, 100.0, benchmark_log_return=prior)["annual_return_pct"] / 100.0)
    # Fraction of the raw drift retained after shrinkage is larger for the long history.
    assert (sh_l / raw_l) > (sh_s / raw_s)


def test_annualization_infers_periods_per_year_from_dates():
    # The portfolio TWR series is calendar-daily (~365/yr); raw market data is
    # trading-daily (~252/yr). The model must infer the spacing from the index so
    # the annualized drift is the same either way (not off by 365/252 ~ 1.45x).
    for freq in ("D", "B"):  # calendar-daily vs business-daily
        idx = pd.date_range("2002-01-01", "2018-01-01", freq=freq)  # 16 years
        n = len(idx)
        g = math.log(4.0) / (n - 1)  # constant growth: 4x total over 16y
        wealth = pd.Series(100.0 * np.exp(np.arange(n) * g), index=idx)
        res = compute_projection(wealth, float(wealth.iloc[-1]))
        # 4x over 16y == 9.05%/yr, regardless of daily vs business-day sampling.
        assert res["annual_return_pct"] == pytest.approx(4 ** (1 / 16) * 100 - 100, abs=0.5)


def test_no_benchmark_leaves_drift_unshrunk():
    w = _wealth(years=8, mu=0.15, seed=5)
    a = compute_projection(w, 100.0)
    b = compute_projection(w, 100.0, benchmark_log_return=None)
    assert a["annual_return_pct"] == pytest.approx(b["annual_return_pct"])
