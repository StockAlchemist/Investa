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

import investa_core
from finutils import infer_periods_per_year  # noqa: F401 -- read by the Rust port at call time

# The arithmetic runs in Rust (rust/investa_core/src/stats.rs), reproducing
# pandas' reductions bit for bit; this module keeps the pandas side — index
# alignment, dropping NaN, inferring the period from the dates — and the one
# BLAS-backed call, the covariance in calculate_beta. The originals are frozen
# in tests/reference/risk_metrics.py.


def _values(series: pd.Series) -> np.ndarray:
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_values(series)


def calculate_drawdown_series(series: pd.Series) -> pd.Series:
    """
    Calculates the drawdown series (percentage decline from peak) for a value series.

    Args:
        series (pd.Series): Time series of portfolio values.

    Returns:
        pd.Series: Series of drawdown values (negative percentages, e.g., -0.05 for 5% down).
                   Returns empty series if input is empty.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_drawdown_series(series)


def calculate_max_drawdown(series: pd.Series) -> float:
    """
    Calculates the Maximum Drawdown (MDD) of a value series.
    MDD is the maximum observed loss from a peak to a trough.

    Args:
        series (pd.Series): Time series of portfolio values or cumulative returns.

    Returns:
        float: The maximum drawdown as a negative fraction (e.g., -0.20 for a 20% drawdown).
               Returns 0.0 if the series never falls, or is empty.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_max_drawdown(series)


def calculate_volatility(
    returns_series: pd.Series, periods_per_year: int = 252
) -> float:
    """
    Calculates the annualized volatility (standard deviation) of returns.

    Args:
        returns_series (pd.Series): Time series of periodic returns (e.g., daily).
        periods_per_year (int): Number of periods in a year (default 252 for daily).

    Returns:
        float: Annualized volatility.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_volatility(returns_series, periods_per_year)


def calculate_sharpe_ratio(
    returns_series: pd.Series, risk_free_rate: float = 0.02, periods_per_year: int = 252
) -> float:
    """
    Calculates the Sharpe Ratio.
    Sharpe = (Mean Return - Risk Free Rate) / Volatility, annualized by sqrt(periods).
    The risk-free rate is spread evenly over the periods (rf / periods_per_year).

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        risk_free_rate (float): Annualized risk-free rate (default 0.02 for 2%).
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: Sharpe Ratio.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_sharpe_ratio(
        returns_series, risk_free_rate, periods_per_year
    )


def calculate_sortino_ratio(
    returns_series: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: int = 252,
    target_return: float = 0.0,
) -> float:
    """
    Calculates the Sortino Ratio.
    Sortino = (Mean Return - Risk Free Rate) / Downside Deviation, where the
    downside deviation is sqrt(mean(min(0, r - target)^2)) over all periods.

    Args:
        returns_series (pd.Series): Time series of periodic returns.
        risk_free_rate (float): Annualized risk-free rate.
        periods_per_year (int): Number of periods in a year.
        target_return (float): Minimum acceptable return (MAR) for downside deviation.
                               Usually 0 or risk_free_rate. We use 0 by default for "loss".

    Returns:
        float: Sortino Ratio; infinite when no period fell short of the target
               and the mean excess return is positive.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_sortino_ratio(
        returns_series, risk_free_rate, periods_per_year, target_return
    )


def calculate_beta(returns_series: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Calculates the Beta of the portfolio relative to a benchmark.
    Beta = Cov(Rp, Rm) / Var(Rm), over the dates both series share.

    Args:
        returns_series (pd.Series): Portfolio periodic returns.
        benchmark_returns (pd.Series): Benchmark periodic returns.

    Returns:
        float: Portfolio Beta; 1.0 when there is too little overlap or the
               benchmark never moves.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_beta(returns_series, benchmark_returns)


def calculate_alpha(
    portfolio_return_ann: float,
    benchmark_return_ann: float,
    beta: float,
    risk_free_rate: float = 0.02,
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
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_alpha(
        portfolio_return_ann, benchmark_return_ann, beta, risk_free_rate
    )


def _returns(values: pd.Series) -> pd.Series:
    """`pct_change(fill_method=None)` with infinities dropped as NaN, then dropna."""
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_returns(values)


def calculate_all_risk_metrics(
    portfolio_values: pd.Series,
    risk_free_rate: float = 0.02,
    periods_per_year: Optional[int] = None,
    benchmark_values: Optional[pd.Series] = None,
) -> Dict[str, float]:
    """
    Wrapper to calculate all risk metrics from a series of portfolio values.

    Args:
        portfolio_values (pd.Series): Time series of portfolio total value.
        risk_free_rate (float): Annualized risk-free rate.
        periods_per_year (int): Periods per year; inferred from the dates when
            omitted (calendar-daily series have ~365 a year, not 252).
        benchmark_values (pd.Series, optional): Time series of benchmark values (e.g. S&P 500).

    Returns:
        Dict[str, float]: Dictionary containing all calculated metrics.
    """
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_all_risk_metrics(
        portfolio_values, risk_free_rate, periods_per_year, benchmark_values
    )


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
    # Ported to Rust: rust/investa_core/src/rm.rs
    return investa_core.rm_benchmark_scoreboard(portfolio_values, benchmark_values)
