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

# Placeholder for fetching factor data (e.g., Fama-French)
def _fetch_factor_data(model_name: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetches historical factor data (placeholder).
    In a real application, this would fetch data from a source like Ken French's data library.
    """
    logging.info(f"Fetching {model_name} factor data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    # Dummy data for demonstration
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    data = {
        'Mkt-RF': np.random.normal(0.01, 0.05, len(dates)),
        'SMB': np.random.normal(0.005, 0.02, len(dates)),
        'HML': np.random.normal(0.003, 00.01, len(dates)),
        'RF': np.random.normal(0.0001, 0.0005, len(dates)) # Risk-Free Rate
    }
    if model_name == "Carhart 4-Factor":
        data['UMD'] = np.random.normal(0.002, 0.015, len(dates)) # Momentum factor
    
    df = pd.DataFrame(data, index=dates)
    df.index.name = 'Date'
    return df

def run_factor_regression(
    portfolio_returns: pd.Series,
    model_name: str = "Fama-French 3-Factor"
) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    """
    Performs factor regression (e.g., Fama-French 3-Factor or Carhart 4-Factor)
    on portfolio excess returns.

    Args:
        portfolio_returns (pd.Series): Daily or monthly portfolio returns.
                                       Index should be datetime.
        model_name (str): The factor model to use ("Fama-French 3-Factor" or "Carhart 4-Factor").

    Returns:
        Optional[statsmodels.regression.linear_model.RegressionResultsWrapper]:
        The regression results object, or None if regression fails.
    """
    logging.info(f"Running {model_name} regression.")

    if portfolio_returns.empty:
        logging.warning("Portfolio returns series is empty. Cannot run factor regression.")
        return None
    
    if not isinstance(portfolio_returns.index, pd.DatetimeIndex):
        logging.error("Portfolio returns index must be a DatetimeIndex.")
        return None

    # Ensure returns are sorted by date
    portfolio_returns = portfolio_returns.sort_index()

    start_date = portfolio_returns.index.min()
    end_date = portfolio_returns.index.max()

    # Fetch factor data
    factor_data = _fetch_factor_data(model_name, start_date, end_date)
    if factor_data.empty:
        logging.error("Failed to fetch factor data. Cannot run regression.")
        return None

    # Align dates and calculate excess returns
    # Assuming portfolio_returns are already total returns, convert to excess returns
    # by subtracting the risk-free rate (RF) from factor data.
    
    # Resample portfolio returns to match factor data frequency (e.g., monthly)
    # For simplicity, let's assume monthly for now.
    # In a real scenario, you'd need to handle daily vs. monthly data carefully.
    
    # Align indices and drop NaNs
    aligned_data = pd.concat([portfolio_returns, factor_data], axis=1).dropna()
    
    if 'RF' not in aligned_data.columns:
        logging.error("Risk-Free Rate (RF) not found in factor data. Cannot calculate excess returns.")
        return None

    # Calculate portfolio excess returns
    # Assuming portfolio_returns are already in percentage or decimal form consistent with factors
    aligned_data['Portfolio_Excess_Return'] = aligned_data[portfolio_returns.name] - aligned_data['RF']

    # Define independent variables (factors)
    if model_name == "Fama-French 3-Factor":
        factors = ['Mkt-RF', 'SMB', 'HML']
    elif model_name == "Carhart 4-Factor":
        factors = ['Mkt-RF', 'SMB', 'HML', 'UMD']
    else:
        logging.error(f"Unsupported factor model: {model_name}")
        return None

    X = aligned_data[factors]
    y = aligned_data['Portfolio_Excess_Return']

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

if __name__ == '__main__':
    # Example Usage
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Generate dummy daily portfolio returns
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    dummy_returns = pd.Series(np.random.normal(0.0005, 0.005, len(dates)), index=dates, name='Portfolio_Returns')

    # Resample to monthly for factor analysis (common practice)
    monthly_returns = dummy_returns.resample('M').apply(lambda x: (1 + x).prod() - 1)

    print("\n--- Fama-French 3-Factor Analysis ---")
    ff3_results = run_factor_regression(monthly_returns, "Fama-French 3-Factor")
    if ff3_results:
        print(ff3_results.summary())

    print("\n--- Carhart 4-Factor Analysis ---")
    carhart4_results = run_factor_regression(monthly_returns, "Carhart 4-Factor")
    if carhart4_results:
        print(carhart4_results.summary())
