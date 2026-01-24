import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from financial_ratios import estimate_growth_rate

def test_3y_average_growth():
    print("\n--- Testing 3-Year Average Growth Rate ---")
    
    # Case 1: Exactly 4 years of data (3 YoY periods)
    # 100 -> 110 (10%)
    # 110 -> 132 (20%)
    # 132 -> 145.2 (10%)
    # Total Growth = 1.452
    # CAGR = (1.452 ^ (1/3)) - 1 = 13.238%
    financials_4y = pd.DataFrame({
        "2020-12-31": [100.0],
        "2021-12-31": [110.0],
        "2022-12-31": [132.0],
        "2023-12-31": [145.2]
    }, index=["Net Income"])
    
    growth_4y = estimate_growth_rate(financials_4y, item_name="Net Income", years=5)
    print(f"4 Years Data (3 periods): Expected ~0.1324, Got: {growth_4y:.4f}")
    assert abs(growth_4y - 0.1324) < 0.01

    # Case 2: 2 years of data (1 YoY period)
    # 100 -> 120 (20%)
    # Avg = 20%
    financials_2y = pd.DataFrame({
        "2022-12-31": [100.0],
        "2023-12-31": [120.0]
    }, index=["Net Income"])
    
    growth_2y = estimate_growth_rate(financials_2y, item_name="Net Income", years=5)
    print(f"2 Years Data (1 period) : Expected 0.200, Got: {growth_2y:.4f}")
    assert abs(growth_2y - 0.20) < 0.01

    # Case 3: More than 4 years, should still only take last 3 periods (4 years)
    # 50 -> 60 (ignored)
    # 100 -> 110 (10%)
    # 110 -> 121 (10%)
    # 121 -> 133.1 (10%)
    # Avg = 10%
    financials_5y = pd.DataFrame({
        "2019-12-31": [50.0], # Should be ignored if we strictly take last 4
        "2020-12-31": [100.0],
        "2021-12-31": [110.0],
        "2022-12-31": [121.0],
        "2023-12-31": [133.1]
    }, index=["Net Income"])
    
    growth_5y = estimate_growth_rate(financials_5y, item_name="Net Income", years=5)
    print(f"5 Years Data (last 3): Expected 0.100, Got: {growth_5y:.4f}")
    assert abs(growth_5y - 0.10) < 0.01

    # Case 4: Negative growth (now returned as is, no floor)
    financials_neg = pd.DataFrame({
        "2022-12-31": [100.0],
        "2023-12-31": [80.0]
    }, index=["Net Income"])
    growth_neg = estimate_growth_rate(financials_neg, item_name="Net Income", years=5)
    print(f"Negative Growth        : Expected -0.200, Got: {growth_neg:.4f}")
    assert abs(growth_neg - (-0.20)) < 0.01

    # Case 5: Extremely high growth (now returned as is, no cap)
    financials_high = pd.DataFrame({
        "2022-12-31": [100.0],
        "2023-12-31": [350.0]
    }, index=["Net Income"])
    growth_high = estimate_growth_rate(financials_high, item_name="Net Income", years=5)
    print(f"High Growth            : Expected 2.500, Got: {growth_high:.4f}")
    assert abs(growth_high - 2.50) < 0.01

    print("--- All Tests Passed! ---\n")

if __name__ == "__main__":
    test_3y_average_growth()
