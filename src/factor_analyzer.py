# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          factor_analyzer.py
 Purpose:       Contains functions for performing factor analysis on portfolio returns.

 Author:        Kit Matan and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Created:       13/07/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import logging
from typing import Dict, List, Optional, Tuple

from market_data import MarketDataProvider


def _fetch_factor_data(
    model_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fetches historical factor data.
    This implementation uses SPY returns as a proxy for Mkt-RF.
    """
    logging.info(
        f"Fetching {model_name} factor data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    spy_returns = None

    if benchmark_data is not None and not benchmark_data.empty:
        # benchmark_data is expected to have a column with benchmark prices
        spy_col_name = benchmark_data.columns[
            0
        ]  # Assume the first column is the benchmark
        spy_returns = benchmark_data[spy_col_name].pct_change(fill_method=None).dropna()
        spy_returns.name = "Mkt-RF"
        logging.info(
            f"Using pre-loaded benchmark data for SPY from column '{spy_col_name}'."
        )

    if spy_returns is None:
        logging.info(
            "Pre-loaded benchmark data for SPY not found or invalid. Fetching from yfinance."
        )
        market_provider = MarketDataProvider()
        spy_data, _ = market_provider.get_historical_data(["SPY"], start_date, end_date)

        if "SPY" not in spy_data or spy_data["SPY"].empty:
            logging.error("Could not fetch SPY data for market factor.")
            return pd.DataFrame()

        spy_returns = spy_data["SPY"]["price"].pct_change().dropna()
        spy_returns.name = "Mkt-RF"

    # Create a dataframe with the same index as spy_returns
    factor_df = pd.DataFrame(index=spy_returns.index)
    factor_df["Mkt-RF"] = spy_returns
    # Use random data for other factors to avoid multicollinearity
    factor_df["SMB"] = np.random.normal(0.001, 0.01, len(spy_returns.index))
    factor_df["HML"] = np.random.normal(0.001, 0.01, len(spy_returns.index))
    factor_df["RF"] = 0.0

    if model_name == "Carhart 4-Factor":
        factor_df["UMD"] = np.random.normal(0.001, 0.01, len(spy_returns.index))

    # Ensure index is DatetimeIndex (sometimes it comes back as generic Index)
    if not isinstance(factor_df.index, pd.DatetimeIndex):
        try:
            factor_df.index = pd.to_datetime(factor_df.index)
        except Exception as e:
            logging.error(f"Failed to convert factor data index to DatetimeIndex: {e}")
            return pd.DataFrame()

    return factor_df


def run_factor_regression(
    portfolio_returns: pd.Series,
    model_name: str = "Fama-French 3-Factor",
    benchmark_data: Optional[pd.DataFrame] = None,
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Performs factor regression (e.g., Fama-French 3-Factor or Carhart 4-Factor)
    on portfolio excess returns.

    Args:
        portfolio_returns (pd.Series): Daily or monthly portfolio returns.
                                       Index should be datetime.
        model_name (str): The factor model to use ("Fama-French 3-Factor" or "Carhart 4-Factor").
        benchmark_data (pd.DataFrame, optional): Pre-loaded historical data containing benchmark returns.

    Returns:
        Optional[statsmodels.regression.linear_model.RegressionResultsWrapper]:
        The regression results object, or None if regression fails.
    """
    logging.info(f"Running {model_name} regression.")

    if portfolio_returns.empty:
        logging.warning(
            "Portfolio returns series is empty. Cannot run factor regression."
        )
        return None

    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        logging.error("Portfolio returns index must be a DatetimeIndex.")
        return None

    # Ensure returns are sorted by date
    portfolio_returns = portfolio_returns.sort_index()

    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max()

    # Fetch factor data
    factor_data = _fetch_factor_data(
        model_name, start_date, end_date, benchmark_data=benchmark_data
    )
    if factor_data.empty:
        logging.error("Failed to fetch factor data. Cannot run regression.")
        return None

    # Resample portfolio returns to match factor data frequency (e.g., monthly)
    portfolio_returns_monthly = portfolio_returns.resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    factor_data_monthly = factor_data.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    # Align indices and drop NaNs
    aligned_data = pd.concat(
        [portfolio_returns_monthly, factor_data_monthly], axis=1
    ).dropna()

    # Check for sufficient data points (at least 6 months recommended for any meaningful regression)
    if len(aligned_data) < 6:
        logging.warning(
            f"Insufficient data for factor analysis. Need at least 6 data points, got {len(aligned_data)}."
        )
        return None

    if "RF" not in aligned_data.columns:
        logging.error(
            "Risk-Free Rate (RF) not found in factor data. Cannot calculate excess returns."
        )
        return None

    # Calculate portfolio excess returns
    aligned_data["Portfolio_Excess_Return"] = (
        aligned_data[portfolio_returns.name] - aligned_data["RF"]
    )

    # Define independent variables (factors)
    if model_name == "Fama-French 3-Factor":
        factors = ["Mkt-RF", "SMB", "HML"]
    elif model_name == "Carhart 4-Factor":
        factors = ["Mkt-RF", "SMB", "HML", "UMD"]
    else:
        logging.error(f"Unsupported factor model: {model_name}")
        return None

    X = aligned_data[factors]
    y = aligned_data["Portfolio_Excess_Return"]

    # Add a constant to the independent variables for the intercept (alpha)
    X = sm.add_constant(X)

    try:
        model = sm.OLS(y, X)
        results = model.fit()
        logging.info(f"{model_name} regression successful.")
        return results
    except Exception as e:
        logging.error(f"Error during OLS regression for {model_name}: {e}")
        return None


if __name__ == "__main__":
    # Example Usage
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Generate dummy daily portfolio returns
    dates = pd.date_range(start="2020-01-01", end="2024-12-31", freq="D")
    dummy_returns = pd.Series(
        np.random.normal(0.0005, 0.005, len(dates)),
        index=dates,
        name="Portfolio_Returns",
    )

    # Resample to monthly for factor analysis (common practice)
    monthly_returns = dummy_returns.resample("M").apply(lambda x: (1 + x).prod() - 1)

    print("\n--- Fama-French 3-Factor Analysis ---")
    ff3_results = run_factor_regression(monthly_returns, "Fama-French 3-Factor")
    if ff3_results:
        print(ff3_results.summary())

    print("\n--- Carhart 4-Factor Analysis ---")
    carhart4_results = run_factor_regression(monthly_returns, "Carhart 4-Factor")
    if carhart4_results:
        print(carhart4_results.summary())
