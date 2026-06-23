# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          risk_metrics.py
 Purpose:       Calculate portfolio risk metrics (Drawdown, Sharpe, Volatility).

 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional

from finutils import infer_periods_per_year

def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    """
    Calculates the drawdown series (percentage decline from peak) for a value series.

    Args:
        series (pd.Series): Time series of portfolio values.

    Returns:
        pd.Series: Series of drawdown values (negative percentages, e.g., -0.05 for 5% down).
                   Returns empty series if input is empty.
    """
    if series.empty:
        return pd.Series(dtype=float)
    
    cumulative_max = series.cummax()
    drawdown = (series - cumulative_max) / cumulative_max
    return drawdown

def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown (MDD) of a value series.
    MDD is the maximum observed loss from a peak to a trough.

    Args:
        series (pd.Series): Time series of portfolio values or cumulative returns.

    Returns:
        float: The maximum drawdown as a positive percentage (e.g., 0.20 for 20% drawdown).
               Returns 0.0 if the series is strictly increasing or empty.
    """
    drawdown = calculate_drawdown_series(series)
    if drawdown.empty:
        return 0.0
    
    mdd = drawdown.min()
    return float(mdd) if pd.notna(mdd) else 0.0

def calculate_volatility(returns_series: pd.Series, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized volatility (standard deviation) of returns.

    Args:
        returns_series (pd.Series): Time series of periodic returns (e.g., daily).
        periods_per_year (int): Number of periods in a year (default 252 for daily).

    Returns:
        float: Annualized volatility.
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0
    
    std_dev = returns_series.std()
    annualized_vol = std_dev * np.sqrt(periods_per_year)
    return float(annualized_vol)

def calculate_sharpe_ratio(
    returns_series: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252
) -> float:
    """
    Calculates the Sharpe Ratio.
    Sharpe = (Mean Return - Risk Free Rate) / Volatility

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        risk_free_rate (float): Annualized risk-free rate (default 0.02 for 2%).
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: Sharpe Ratio.
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0
    
    # Convert annualized risk-free rate to periodic
    # Simple approximation: rf_daily = rf_annual / periods
    rf_periodic = risk_free_rate / periods_per_year
    
    excess_returns = returns_series - rf_periodic
    mean_excess_return = excess_returns.mean()
    std_dev_returns = returns_series.std()
    
    if std_dev_returns == 0:
        return 0.0
        
    # Annualize the ratio
    # Sharpe_annual = Sharpe_daily * sqrt(periods)
    sharpe_daily = mean_excess_return / std_dev_returns
    sharpe_annual = sharpe_daily * np.sqrt(periods_per_year)
    
    return float(sharpe_annual)

def calculate_sortino_ratio(
    returns_series: pd.Series, 
    risk_free_rate: float = 0.02, 
    periods_per_year: int = 252,
    target_return: float = 0.0
) -> float:
    """
    Calculates the Sortino Ratio.
    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        risk_free_rate (float): Annualized risk-free rate.
        periods_per_year (int): Number of periods in a year.
        target_return (float): Minimum acceptable return (MAR) for downside deviation. 
                               Usually 0 or risk_free_rate. We use 0 by default for "loss".

    Returns:
        float: Sortino Ratio.
    """
    if returns_series.empty or len(returns_series) < 2:
        return 0.0
        
    rf_periodic = risk_free_rate / periods_per_year
    excess_returns = returns_series - rf_periodic
    mean_excess_return = excess_returns.mean()
    
    # Downside deviation: Standard deviation of negative returns (relative to target)
    # Usually target is 0 for "losing money"
    downside_returns = returns_series[returns_series < target_return]
    
    if downside_returns.empty:
        return float('inf') if mean_excess_return > 0 else 0.0
        
    # Calculate downside deviation (using 0 as the target for deviation calculation usually, 
    # or the mean of downside? Standard is root mean squared of deviations below target)
    # Formula: sqrt( sum( min(0, r - target)^2 ) / N )
    
    # Let's use the standard definition where we penalize returns below target_return
    underperformance = returns_series - target_return
    underperformance[underperformance > 0] = 0
    squared_underperformance = underperformance ** 2
    downside_deviation = np.sqrt(squared_underperformance.mean())
    
    if downside_deviation == 0:
        return float('inf') if mean_excess_return > 0 else 0.0
        
    sortino_daily = mean_excess_return / downside_deviation
    sortino_annual = sortino_daily * np.sqrt(periods_per_year)
    
    return float(sortino_annual)

def calculate_beta(returns_series: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Beta of the portfolio relative to a benchmark.
    Beta = Cov(Rp, Rm) / Var(Rm)

    Args:
        returns_series (pd.Series): Portfolio periodic returns.
        benchmark_returns (pd.Series): Benchmark periodic returns.

    Returns:
        float: Portfolio Beta.
    """
    if returns_series.empty or benchmark_returns.empty:
        return 1.0
        
    # Align the series by date
    aligned = pd.concat([returns_series, benchmark_returns], axis=1).dropna()
    if len(aligned) < 2:
        return 1.0
        
    cov = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
    var = aligned.iloc[:, 1].var()
    
    if var == 0:
        return 1.0
        
    return float(cov / var)

def calculate_alpha(
    portfolio_return_ann: float, 
    benchmark_return_ann: float, 
    beta: float, 
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculates Jensen's Alpha.
    Alpha = Rp - [Rf + Beta * (Rm - Rf)]

    Args:
        portfolio_return_ann (float): Annualized portfolio return.
        benchmark_return_ann (float): Annualized benchmark return.
        beta (float): Portfolio Beta.
        risk_free_rate (float): Annualized risk-free rate.

    Returns:
        float: Portfolio Alpha.
    """
    expected_return = risk_free_rate + beta * (benchmark_return_ann - risk_free_rate)
    alpha = portfolio_return_ann - expected_return
    return float(alpha)

def calculate_all_risk_metrics(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None,
    benchmark_values: Optional[pd.Series] = None
) -> Dict[str, float]:
    """
    Wrapper to calculate all risk metrics from a series of portfolio values.
    
    Args:
        portfolio_values (pd.Series): Time series of portfolio total value.
        risk_free_rate (float): Annualized risk-free rate.
        periods_per_year (int): Periods per year.
        benchmark_values (pd.Series, optional): Time series of benchmark values (e.g. S&P 500).

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics.
    """
    if portfolio_values.empty:
        return {}
        
    # Calculate returns
    returns = portfolio_values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    
    # --- ROBUSTNESS: Handle extreme outliers that ruin metrics ---
    # Daily returns > 1000% or < -95% are almost always data artifacts (missing flows, splits, etc.)
    # especially for diversified portfolios. We clip them to prevent infinity/extreme volatility.
    if not returns.empty:
        # Check for suspected artifacts
        outliers = returns[(returns > 1.0) | (returns < -0.90)]
        if not outliers.empty:
            # We clip while keeping the sign
            returns = returns.clip(lower=-0.90, upper=1.0)
            
    # MDD ignores returns and works on value series (more robust)
    mdd = calculate_max_drawdown(portfolio_values)
    
    # If we have very few data points, metrics like Vol/Sharpe are statistically meaningless
    # and highly subject to initialization bias.
    if len(returns) < 5:
        return {
            "Max Drawdown": mdd,
            "Volatility (Ann.)": 0.0,
            "Sharpe Ratio": 0.0,
            "Sortino Ratio": 0.0,
            "insufficient_data": True
        }

    # Calendar-daily portfolio series have ~365 obs/yr, not 252 — infer from the
    # dates so vol/Sharpe/Sortino/Alpha aren't under-annualized by ~365/252
    # (keeps the risk card consistent with the projection model). An explicit
    # periods_per_year still overrides.
    ppy = periods_per_year if periods_per_year is not None else infer_periods_per_year(returns.index)

    vol = calculate_volatility(returns, ppy)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, ppy)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, ppy)
    
    metrics = {
        "Max Drawdown": mdd,
        "Volatility (Ann.)": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    }

    # Add Alpha & Beta if benchmark provided
    if benchmark_values is not None and not benchmark_values.empty:
        bench_returns = benchmark_values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
        if not bench_returns.empty:
            beta = calculate_beta(returns, bench_returns)
            
            # Annualized returns for Alpha
            port_ann_ret = returns.mean() * ppy
            bench_ann_ret = bench_returns.mean() * ppy
            
            alpha = calculate_alpha(port_ann_ret, bench_ann_ret, beta, risk_free_rate)
            
            metrics["Beta"] = beta
            metrics["Alpha"] = alpha
            
    return metrics


def calculate_benchmark_scoreboard(
    portfolio_values: pd.Series,
    benchmark_values: Dict[str, pd.Series],
) -> list:
    """Per-benchmark active-management stats: alpha, beta, R², tracking error,
    information ratio, and cumulative excess return.

    Single source of truth for the "Benchmark Scoreboard" shown on web + native.
    Uses population moments and infers the period from the dates (the calendar-
    daily TWR series is ~365/yr, not 252), so alpha/TE/IR are correctly annualized
    and identical across clients. ``benchmark_values`` maps a display name to that
    benchmark's accumulated-value (wealth) series.
    """
    out = []
    if portfolio_values is None or portfolio_values.empty:
        return out

    def _returns(s: pd.Series) -> pd.Series:
        return (
            s.pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
            .clip(lower=-0.90, upper=1.0)
        )

    port_ret = _returns(portfolio_values)
    if len(port_ret) < 20:
        return out
    ppy = infer_periods_per_year(port_ret.index)
    sqrt_ppy = np.sqrt(ppy)

    for name, bvals in benchmark_values.items():
        if bvals is None or bvals.empty:
            continue
        aligned = pd.concat([port_ret, _returns(bvals)], axis=1, join="inner").dropna()
        if len(aligned) < 20:
            continue
        rp = aligned.iloc[:, 0].to_numpy()
        rb = aligned.iloc[:, 1].to_numpy()
        mp, mb = rp.mean(), rb.mean()
        cov = ((rp - mp) * (rb - mb)).mean()
        var_b = ((rb - mb) ** 2).mean()
        var_p = ((rp - mp) ** 2).mean()
        beta = cov / var_b if var_b > 0 else 0.0
        alpha = (mp - beta * mb) * ppy * 100.0
        corr = cov / np.sqrt(var_p * var_b) if var_p > 0 and var_b > 0 else 0.0
        diffs = rp - rb
        m_diff = diffs.mean()
        te_daily = np.sqrt(((diffs - m_diff) ** 2).mean())
        te = te_daily * sqrt_ppy * 100.0
        ir = (m_diff * ppy) / (te_daily * sqrt_ppy) if te_daily > 0 else 0.0
        port_total = (float(portfolio_values.iloc[-1]) / float(portfolio_values.iloc[0]) - 1.0) * 100.0
        bench_total = (float(bvals.iloc[-1]) / float(bvals.iloc[0]) - 1.0) * 100.0
        out.append({
            "name": name,
            "alpha": float(alpha),
            "beta": float(beta),
            "r2": float(corr * corr),
            "tracking_error": float(te),
            "information_ratio": float(ir),
            "excess_return": float(port_total - bench_total),
        })
    return out

