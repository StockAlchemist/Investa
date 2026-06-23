# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pytest

from risk_metrics import (
    calculate_max_drawdown,
    calculate_volatility,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_all_risk_metrics,
    calculate_benchmark_scoreboard,
)


def _wealth(daily_returns, start="2002-01-01", freq="D"):
    idx = pd.date_range(start, periods=len(daily_returns) + 1, freq=freq)
    vals = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + np.asarray(daily_returns))])
    return pd.Series(vals, index=idx)


def test_scoreboard_identical_series_is_neutral():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0004, 0.01, 1500)
    bench = _wealth(r)
    rows = calculate_benchmark_scoreboard(bench, {"S&P 500": bench.copy()})
    assert len(rows) == 1
    s = rows[0]
    assert s["beta"] == pytest.approx(1.0, abs=1e-6)
    assert s["r2"] == pytest.approx(1.0, abs=1e-6)
    assert s["alpha"] == pytest.approx(0.0, abs=1e-6)
    assert s["tracking_error"] == pytest.approx(0.0, abs=1e-6)
    assert s["excess_return"] == pytest.approx(0.0, abs=1e-6)


def test_scoreboard_recovers_known_beta():
    rng = np.random.default_rng(1)
    rb = rng.normal(0.0003, 0.01, 1500)
    rp = 2.0 * rb  # portfolio is a 2x-levered version of the benchmark
    rows = calculate_benchmark_scoreboard(_wealth(rp), {"B": _wealth(rb)})
    assert rows[0]["beta"] == pytest.approx(2.0, rel=1e-3)
    assert rows[0]["r2"] == pytest.approx(1.0, abs=1e-6)


def test_scoreboard_annualization_uses_inferred_period():
    # Same daily outperformance, sampled calendar-daily (~365/yr) vs business-daily
    # (~252/yr): alpha must scale with the inferred period, not a hardcoded 252.
    rng = np.random.default_rng(2)
    rb = rng.normal(0.0003, 0.008, 2000)
    rp = rb + 0.0002  # constant daily active return
    a_cal = calculate_benchmark_scoreboard(_wealth(rp, freq="D"), {"B": _wealth(rb, freq="D")})[0]["alpha"]
    a_biz = calculate_benchmark_scoreboard(_wealth(rp, freq="B"), {"B": _wealth(rb, freq="B")})[0]["alpha"]
    # Calendar-daily annualizes by ~365 vs ~252 -> ~45% larger alpha.
    assert a_cal > a_biz * 1.3

def test_max_drawdown():
    # Case 1: Simple drawdown
    # 100 -> 110 -> 99 -> 120
    # Peak 110, Trough 99. Drawdown = (99-110)/110 = -0.1 (-10%)
    values = pd.Series([100, 110, 99, 120])
    mdd = calculate_max_drawdown(values)
    assert np.isclose(mdd, -0.1)

    # Case 2: No drawdown (strictly increasing)
    values = pd.Series([100, 101, 102, 103])
    mdd = calculate_max_drawdown(values)
    assert mdd == 0.0

    # Case 3: Constant
    values = pd.Series([100, 100, 100])
    mdd = calculate_max_drawdown(values)
    assert mdd == 0.0

def test_volatility():
    # Case 1: Constant returns -> 0 volatility
    returns = pd.Series([0.01, 0.01, 0.01, 0.01])
    vol = calculate_volatility(returns)
    assert vol == 0.0
    
    # Case 2: Known std dev
    # -1%, +1% alternating. 
    returns = pd.Series([-0.01, 0.01, -0.01, 0.01])
    # Std dev of population is 0.01, sample std dev slightly higher
    std_sample = returns.std()
    expected_vol = std_sample * np.sqrt(252)
    vol = calculate_volatility(returns)
    assert np.isclose(vol, expected_vol)

def test_sharpe_ratio():
    # Case 1: Returns = Risk Free Rate -> Sharpe 0
    # Annual RF = 0.02, Daily RF approx 0.02/252
    rf = 0.02
    daily_rf = rf / 252
    returns = pd.Series([daily_rf] * 10)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=rf)
    # Std dev is 0, so our function returns 0.0 to avoid div/0
    assert sharpe == 0.0
    
    # Case 2: High returns, low vol
    returns = pd.Series([0.01, 0.012, 0.01, 0.012]) # Avg > rf
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
    assert sharpe > 0

def test_sortino_ratio():
    # Case 1: No negative returns -> Infinite Sortino (or handled large number)
    # Our implementation returns 'inf' if downside deviation is 0 and mean > 0
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    sortino = calculate_sortino_ratio(returns)
    assert sortino == float('inf')
    
    # Case 2: Some negative returns
    returns = pd.Series([0.01, -0.02, 0.01, 0.03])
    sortino = calculate_sortino_ratio(returns)
    assert sortino > 0 and sortino != float('inf')

def test_calculate_all_metrics():
    values = pd.Series([100, 105, 102, 110, 108, 115])
    metrics = calculate_all_risk_metrics(values)
    
    assert "Max Drawdown" in metrics
    assert "Volatility (Ann.)" in metrics
    assert "Sharpe Ratio" in metrics
    assert "Sortino Ratio" in metrics
    
    assert metrics["Max Drawdown"] < 0
    assert metrics["Volatility (Ann.)"] > 0
