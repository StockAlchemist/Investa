import pandas as pd
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_clean_transactions
from config import DEFAULT_CURRENCY

def inspect_drop():
    db_path = 'data/users/kitmatan/portfolio.db'
    
    # 1. Load transactions
    res = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)
    df = res[0]
    
    # Filter for dates around the drop: 2016-12-20 to 2017-01-15
    mask = (df['Date'] >= '2016-12-15') & (df['Date'] <= '2017-01-15')
    drop_txs = df[mask].sort_values('Date')
    
    print("\n--- Transactions around the drop (Dec 2016 - Jan 2017) ---")
    print(drop_txs[['Date', 'Type', 'Symbol', 'Quantity', 'Price/Share', 'Total Amount', 'Account', 'Note']].to_string())

    # Also check for symbols involved in splits or major transactions before this
    print("\n--- Symbols involved in those dates ---")
    symbols = drop_txs['Symbol'].unique()
    print(symbols)

if __name__ == "__main__":
    inspect_drop()
