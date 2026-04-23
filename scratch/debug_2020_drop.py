import pandas as pd
import numpy as np
import logging
import os
import sys
from datetime import date

# Add src to path
sys.path.append(os.path.abspath('src'))

import portfolio_logic
# Mock logging
class MockLogger:
    def info(self, msg): pass
    def warning(self, msg): pass
    def error(self, msg): pass
    def debug(self, msg): pass

portfolio_logic.logging = MockLogger()

from portfolio_logic import _calculate_daily_net_cash_flow_vectorized
from data_loader import load_and_clean_transactions
from config import DEFAULT_CURRENCY

def debug_flow():
    user_id = 'kitmatan'
    db_path = f'data/users/{user_id}/portfolio.db'
    
    # 1. Load transactions
    res = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)
    df = res[0]
    
    # 2. Define target dates
    target_dates = ['2014-06-09', '2020-08-31', '2020-09-01', '2020-09-02', '2020-09-03']
    
    date_range = pd.date_range(start='2010-01-01', end='2026-01-01', freq='D', tz='UTC')
    
    flow_series, _ = _calculate_daily_net_cash_flow_vectorized(
        date_range=date_range,
        transactions_df=df,
        target_currency='USD',
        historical_fx_yf={},
        default_currency=DEFAULT_CURRENCY,
        included_accounts=None
    )
    
    for d_str in target_dates:
        d = pd.to_datetime(d_str, utc=True)
        val = flow_series.get(d, 0)
        print(f"\n--- Flow on {d_str}: {val} ---")
        txs = df[df['Date'].dt.date == d.date()]
        print(txs[['Date', 'Type', 'Symbol', 'Quantity', 'Price/Share', 'Total Amount', 'Account', 'Note']])

if __name__ == "__main__":
    debug_flow()
