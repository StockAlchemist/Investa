import pandas as pd
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_clean_transactions
from config import DEFAULT_CURRENCY, CASH_SYMBOL_CSV

def check_cash_tracking():
    db_path = 'data/users/kitmatan/portfolio.db'
    res = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)
    df = res[0]
    
    accounts = df['Account'].unique()
    print(f"\n--- Cash tracking check (CASH_SYMBOL_CSV={CASH_SYMBOL_CSV}) ---")
    for acc in accounts:
        has_cash = (df[(df['Account'] == acc) & (df['Symbol'] == CASH_SYMBOL_CSV)]).empty == False
        print(f"Account: {acc:20} | Has Cash Tx: {has_cash}")

if __name__ == "__main__":
    check_cash_tracking()
