
import sys
import os
import pandas as pd
import numpy as np
import logging

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from factor_analyzer import run_factor_regression

from unittest.mock import patch

# Configure logging to capture warnings
logging.basicConfig(level=logging.INFO)

def mock_fetch_factor_data(model_name, start_date, end_date, benchmark_data=None):
    # Create dummy factor data with DatetimeIndex matching the request
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame(index=dates)
    df["Mkt-RF"] = np.random.normal(0.001, 0.01, len(dates))
    df["SMB"] = np.random.normal(0.001, 0.01, len(dates))
    df["HML"] = np.random.normal(0.001, 0.01, len(dates))
    df["RF"] = 0.0
    return df

@patch('factor_analyzer._fetch_factor_data', side_effect=mock_fetch_factor_data)
def test_insufficient_data(mock_fetch):
    print("\n--- Testing Insufficient Data ---")
    # Only 2 data points
    dates = pd.date_range(start="2024-01-01", periods=2, freq="ME")
    returns = pd.Series([0.01, 0.02], index=dates, name="Portfolio_Returns")
    # Ensure index is DatetimeIndex
    returns.index = pd.to_datetime(returns.index)
    
    result = run_factor_regression(returns, "Fama-French 3-Factor")
    if result is None:
        print("PASS: Handled insufficient data gracefully (returned None).")
    else:
        print("FAIL: Did not return None for insufficient data.")

@patch('factor_analyzer._fetch_factor_data', side_effect=mock_fetch_factor_data)
def test_constant_data(mock_fetch):
    print("\n--- Testing Constant Data (Zero Variance) ---")
    # Constant returns
    dates = pd.date_range(start="2024-01-01", periods=10, freq="ME")
    returns = pd.Series([0.01] * 10, index=dates, name="Portfolio_Returns")
    # Ensure index is DatetimeIndex
    returns.index = pd.to_datetime(returns.index)
    
    result = run_factor_regression(returns, "Fama-French 3-Factor")
    if result is None:
        print("PASS: Handled constant data gracefully (returned None).")
    else:
        print("FAIL: Did not return None for constant data.")

if __name__ == "__main__":
    test_insufficient_data()
    test_constant_data()
