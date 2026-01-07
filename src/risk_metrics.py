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
from typing import Dict, Optional, Union

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

def calculate_all_risk_metrics(
    portfolio_values: pd.Series, 
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Wrapper to calculate all risk metrics from a series of portfolio values.
    
    Args:
        portfolio_values (pd.Series): Time series of portfolio total value.
        risk_free_rate (float): Annualized risk-free rate.
        periods_per_year (int): Periods per year.

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics.
    """
    if portfolio_values.empty:
        return {}
        
    # Calculate returns
    # Calculate returns and replace inf with NaN so they can be dropped or handled
    returns = portfolio_values.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    mdd = calculate_max_drawdown(portfolio_values)
    vol = calculate_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    return {
        "Max Drawdown": mdd,
        "Volatility (Ann.)": vol,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino
    }
