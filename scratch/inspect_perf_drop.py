import pandas as pd
import sqlite3
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath('src'))

from portfolio_logic import calculate_historical_performance
from data_loader import load_and_clean_transactions
from config import DEFAULT_CURRENCY

def inspect_holdings_at_drop():
    db_path = 'data/users/kitmatan/portfolio.db'
    
    # Load transactions
    res = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)
    df = res[0]
    
    # Calculate performance to get daily_df and holdings
    # We use a wide window to see the context
    perf_df, total_return, status = calculate_historical_performance(
        df, 
        start_date=pd.Timestamp('2016-01-01'),
        end_date=pd.Timestamp('2017-12-31'),
        benchmark_symbols=['S&P 500']
    )
    
    if perf_df.empty:
        print("Error: perf_df is empty")
        return

    # Find the drop dates (e.g. Dec 30, 2016)
    drop_dates = ['2016-12-28', '2016-12-29', '2016-12-30', '2016-12-31', '2017-01-01', '2017-01-02']
    
    print("\n--- Daily Performance Around Drop ---")
    cols = ['value', 'net_flow', 'daily_gain', 'daily_return']
    for d in drop_dates:
        if d in perf_df.index:
            row = perf_df.loc[d]
            ret = row['daily_return'] * 100 if not np.isnan(row['daily_return']) else 0
            print(f"{d} | Value: {row['value']:10.2f} | Flow: {row['net_flow']:10.2f} | Gain: {row['daily_gain']:10.2f} | Ret: {ret:6.2f}%")

if __name__ == "__main__":
    inspect_holdings_at_drop()
