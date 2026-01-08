
import pandas as pd
import numpy as np
import sys
import os
from datetime import date, timedelta

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from portfolio_logic import _calculate_accumulated_gains_and_resample

def test_drawdown_calculation():
    print("Testing Drawdown Calculation...")
    
    # Create sample daily data: Up, Up, Peak, Drop, Drop, Recover
    dates = pd.date_range(start="2024-01-01", periods=10, freq="D")
    
    # Returns: +10%, +10%, -10%, -10%, +5%, ...
    returns = [0.1, 0.1, -0.1, -0.1, 0.05, 0.05, 0.1, 0.1, 0.0, 0.0]
    
    # Gains: 1.1, 1.21, 1.089, 0.9801, ...
    
    daily_df = pd.DataFrame({
        "daily_return": returns,
        "value": [1000] * 10, # Dummy
        "net_flow": [0] * 10 # Dummy
    }, index=dates)
    
    benchmark_symbols = []
    
    result_df, twr, status = _calculate_accumulated_gains_and_resample(
        daily_df=daily_df,
        benchmark_symbols_yf=benchmark_symbols,
        interval="D",
        start_date_filter=dates[0].date(),
        end_date_filter=dates[-1].date()
    )
    
    print("\nResult Columns:", result_df.columns.tolist())
    
    if "drawdown" in result_df.columns:
        print("\nDrawdown values:")
        for dt, row in result_df.iterrows():
            twr = row.get("Portfolio Accumulated Gain")
            dd = row.get("drawdown")
            print(f"{dt.date()}: TWR={twr:.4f}, Drawdown={dd:.4f}%")
            
        # Assertion
        # Day 0: Ret 0.1 -> Factor 1.1. Max 1.1. DD 0.
        # Day 1: Ret 0.1 -> Factor 1.21. Max 1.21. DD 0.
        # Day 2: Ret -0.1 -> Factor 1.089. Max 1.21. DD (1.089/1.21 - 1) = -0.1 -> -10%
        
        day2_dd = result_df.iloc[2]["drawdown"]
        expected = (1.089 / 1.21 - 1) * 100
        print(f"\nDay 2 Expected: {expected:.4f}%, Got: {day2_dd:.4f}%")
        
        if abs(day2_dd - expected) < 0.01:
            print("SUCCESS: Drawdown calculation is correct.")
        else:
            print("FAILURE: Drawdown calculation mismatch.")
    else:
        print("FAILURE: 'drawdown' column missing from result.")

if __name__ == "__main__":
    test_drawdown_calculation()
