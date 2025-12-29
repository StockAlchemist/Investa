# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          factor_analyzer.py
 Purpose:       Contains functions for performing factor analysis on portfolio returns.

 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import numpy as np
# import statsmodels.api as sm # Lazy loaded
sm = None
import logging
from typing import Dict, List, Optional, Tuple, Any

from market_data import MarketDataProvider


def _fetch_factor_data(
    model_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    benchmark_data: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Fetches historical factor data using ETF proxies.
    
    Proxies:
    - Mkt-RF: SPY (S&P 500)
    - SMB: IWM (Russell 2000 - Small Cap) - SPY (Large Cap)
    - HML: VTV (Value) - VUG (Growth)
    - UMD: MTUM (Momentum) - SPY (Market)
    - RF: Treasuries (Using ^TNX or similar, but for simplicity/reliability we'll use a constant or fetch a bond ETF like SHV for now, or just 0 as it's often small daily). 
      Actually, Fama-French uses risk-free rate. We'll use a small constant for daily RF or 0 to approximate 'Market' as pure SPY return for now to keep it simple and robust, or better:
      Use ^IRX (13 week treasury bill) if available, else 0.
      For this implementation, we will use a simplified assumption:
         RF = 0 (for daily excess returns, this is often acceptable for rough estimation).
         Mkt-RF ~ SPY (Assuming beta=1 to market).
    """
    logging.info(
        f"Fetching {model_name} factor data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    # Define proxies
    tickers = ["SPY", "IWM", "VTV", "VUG"]
    if model_name == "Carhart 4-Factor":
        tickers.append("MTUM")
    
    # We might already have SPY in benchmark_data
    spy_returns = None
    if benchmark_data is not None and not benchmark_data.empty:
        # Check if we can identify SPY column
        # existing logic...
        pass 
    
    # Fetch all needed tickers
    market_provider = MarketDataProvider()
    
    # Check what we need to fetch
    tickers_to_fetch = [t for t in tickers] 
    
    # If we have benchmark_data (likely SPY), we might skip SPY, but for consistency let's just fetch all to ensure alignment
    # Or rely on yfinance caching.
    
    data_dict, _ = market_provider.get_historical_data(tickers_to_fetch, start_date, end_date)
    
    # Process returns
    returns_df = pd.DataFrame()
    
    for ticker in tickers_to_fetch:
        if ticker in data_dict and not data_dict[ticker].empty:
             # Calculate daily returns
             returns_df[ticker] = data_dict[ticker]['price'].pct_change()
        else:
            logging.error(f"Could not fetch data for {ticker}. Factor analysis may be inaccurate.")
            
    returns_df.dropna(inplace=True)
    
    if returns_df.empty:
        logging.error("No factor data available after fetching.")
        return pd.DataFrame()

    # Construct Factors
    factor_df = pd.DataFrame(index=returns_df.index)
    
    # Mkt-RF (Proxy: SPY)
    if "SPY" in returns_df.columns:
        factor_df["Mkt-RF"] = returns_df["SPY"]
    else:
        # Fallback if SPY failed but others worked? Unlikely but handle it
        loading.error("SPY data missing for Mkt-RF.")
        return pd.DataFrame()

    # SMB (Small Minus Big): IWM - SPY
    if "IWM" in returns_df.columns:
        factor_df["SMB"] = returns_df["IWM"] - returns_df["SPY"]
    else:
        factor_df["SMB"] = 0.0

    # HML (High Minus Low): VTV - VUG
    if "VTV" in returns_df.columns and "VUG" in returns_df.columns:
        factor_df["HML"] = returns_df["VTV"] - returns_df["VUG"]
    else:
        factor_df["HML"] = 0.0
        
    # UMD (Momentum): MTUM - SPY (Excess return of momentum over market)
    # Standard UMD is Winners - Losers. MTUM is a long-only momentum ETF.
    # So MTUM - SPY is a reasonable proxy for "Momentum Factor Exposure".
    if model_name == "Carhart 4-Factor":
        if "MTUM" in returns_df.columns:
            factor_df["UMD"] = returns_df["MTUM"] - returns_df["SPY"]
        else:
            factor_df["UMD"] = 0.0
            
    # Risk Free Rate
    # For now, we assume 0 for daily simplification or could fetch ^IRX
    # Let's stick to 0 to avoid another network call dependent failure point
    factor_df["RF"] = 0.0

    # Ensure index is DatetimeIndex (as market_data might return generic Index)
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
) -> Optional[Any]:
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
    
    # Lazy load statsmodels
    global sm
    if sm is None:
        try:
             import statsmodels.api as _sm
             sm = _sm
        except ImportError:
             logging.error("statsmodels not installed. Cannot run factor regression.")
             return None

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
