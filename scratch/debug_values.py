import pandas as pd
import numpy as np
import os
import sys
from datetime import date

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_clean_transactions
from config import DEFAULT_CURRENCY
import portfolio_logic

def debug_portfolio_values():
    user_id = 'kitmatan'
    db_path = f'data/users/{user_id}/portfolio.db'
    
    # 1. Load transactions
    res = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)
    df = res[0]
    
    start_date = date(2016, 12, 20)
    end_date = date(2017, 1, 15)
    
    # Calculate historical performance
    # We need to mock some inputs for calculate_historical_performance
    daily_df, _, _, _ = portfolio_logic.calculate_historical_performance(
        all_transactions_df_cleaned=df,
        original_transactions_df_for_ignored=df,
        ignored_indices_from_load=set(),
        ignored_reasons_from_load={},
        start_date=start_date,
        end_date=end_date,
        interval='D',
        benchmark_symbols_yf=[],
        display_currency='USD',
        account_currency_map={'SET': 'THB', 'E*TRADE': 'USD'},
        default_currency='USD'
    )
    
    print("\n--- Daily Portfolio Metrics ---")
    cols = ['Portfolio Value', 'Absolute ROI (%)', 'Cumulative Net Flow']
    # Check if 'drawdown' or other cols exist
    available_cols = [c for c in cols if c in daily_df.columns]
    print(daily_df[available_cols].to_string())

    # Now let's look at the raw daily_df from calculate_historical_performance if possible
    # We might need to look at the 'net_flow' and 'value' before resampling
    
if __name__ == "__main__":
    debug_portfolio_values()
