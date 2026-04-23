import pandas as pd
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.abspath('src'))

from data_loader import load_and_clean_transactions

def check_rimm_holdings():
    db_path = 'data/users/kitmatan/portfolio.db'
    res = load_and_clean_transactions(db_path, {}, 'USD', is_db_source=True)
    df = res[0]
    
    rimm_df = df[df['Symbol'] == 'RIMM'].sort_values('Date')
    if rimm_df.empty:
        print("No RIMM transactions found.")
        return
        
    print("\n--- RIMM Transactions ---")
    print(rimm_df[['Date', 'Type', 'Quantity', 'Account']])
    
    qty = 0
    for _, row in rimm_df.iterrows():
        t = row['Type'].lower()
        q = row['Quantity']
        if t == 'buy' or t == 'buy to cover':
            qty += q
        elif t == 'sell' or t == 'short sell':
            qty -= q
        print(f"Date: {row['Date']} | Type: {t:15} | Qty Change: {q:10.2f} | Running Qty: {qty:10.4f}")

if __name__ == "__main__":
    check_rimm_holdings()
