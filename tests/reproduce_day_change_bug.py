# tests/reproduce_day_change_bug.py

import sys
import os
import pandas as pd
import numpy as np
from datetime import date, datetime

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from portfolio_analyzer import _build_summary_rows

def test_day_change_with_buy():
    # Setup: 100 shares yesterday, buy 50 today
    # Price yesterday: 100. Price today: 110. Change: +10.
    # Market Gain should be 100 * 10 = 1000.
    
    report_date = date(2026, 4, 10)
    
    holdings = {
        ("AAPL", "Acc1"): {
            "qty": 150.0,
            "total_cost_local": 15500.0, # 100*100 + 50*110
            "local_currency": "USD",
            "total_cost_display_historical_fx": 15500.0,
        }
    }
    
    current_stock_data = {
        "AAPL": {
            "price": 110.0,
            "change": 10.0,
            "changesPercentage": 10.0
        }
    }
    
    current_fx_rates_vs_usd = {"USD": 1.0}
    current_fx_prev_close_vs_usd = {"USD": 1.0}
    
    # Transactions today
    tx_data = {
        "Date": [pd.Timestamp(report_date)],
        "Symbol": ["AAPL"],
        "Quantity": [50.0],
        "Type": ["Buy"],
        "Price/Share": [110.0],
        "Account": ["Acc1"],
        "Local Currency": ["USD"],
        "Commission": [0.0],
        "Total Amount": [5500.0]
    }
    transactions_df = pd.DataFrame(tx_data)
    
    rows, _, _, _ = _build_summary_rows(
        holdings=holdings,
        current_stock_data=current_stock_data,
        current_fx_rates_vs_usd=current_fx_rates_vs_usd,
        current_fx_prev_close_vs_usd=current_fx_prev_close_vs_usd,
        display_currency="USD",
        default_currency="USD",
        transactions_df=transactions_df,
        report_date=report_date,
        shortable_symbols=set(),
        user_excluded_symbols=set(),
        user_symbol_map={},
        manual_prices_dict={}
    )
    
    aapl_row = rows[0]
    day_change = aapl_row["Day Change (USD)"]
    
    print(f"AAPL Day Change: {day_change}")
    # Current behavior will result in 150 * 110 - 150 * 100 = 1500.
    # Correct behavior should be 100 * 10 = 1000.
    
    if abs(day_change - 1000) < 1e-9:
        print("SUCCESS: Bug is fixed!")
    else:
        print(f"FAILURE: Day change is {day_change}, expected 1000")

if __name__ == "__main__":
    test_day_change_with_buy()
