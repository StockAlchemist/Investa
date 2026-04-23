import sqlite3
import os
import pandas as pd
from datetime import datetime

def heal_etrade_cash():
    db_path = 'data/users/kitmatan/portfolio.db'
    backup_path = db_path + '.backup'
    
    # Backup
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Backed up database to {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Load all E*TRADE non-cash transactions
    query = "SELECT id, Date, Type, Symbol, Quantity, [Price/Share], [Total Amount], Commission, Account FROM transactions WHERE Account='E*TRADE' AND Symbol != '$CASH' ORDER BY Date"
    df = pd.read_sql_query(query, conn)
    
    # Load existing cash transactions for E*TRADE to avoid duplicates
    cash_query = "SELECT Date, Type, [Total Amount] FROM transactions WHERE Account='E*TRADE' AND Symbol = '$CASH'"
    cash_df = pd.read_sql_query(cash_query, conn)
    
    new_rows = []
    
    for _, row in df.iterrows():
        tx_date = row['Date']
        tx_type = row['Type'].lower()
        symbol = row['Symbol']
        amount = abs(row['Total Amount'])
        
        # Check if matching settlement exists (approx amount)
        # For simplicity, we just check if ANY cash transaction exists on that date
        # Since we found E*TRADE has ZERO cash transactions in that window, 
        # this is mostly safe.
        
        existing = cash_df[cash_df['Date'] == tx_date]
        if not existing.empty:
            continue # Already has some cash tracking on this day
        
        if tx_type == 'buy':
            # Missing settlement for buy
            # 1. Deposit (External flow)
            new_rows.append((
                tx_date, 'deposit', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Cash deposit for {symbol} buy (Healing)"
            ))
            # 2. Sell (Internal settlement)
            new_rows.append((
                tx_date, 'sell', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Cash settlement for {symbol} buy (Healing)"
            ))
            print(f"Added settlement for BUY {symbol} on {tx_date}")
            
        elif tx_type == 'sell':
            # Missing settlement for sell
            # 1. Buy (Internal settlement)
            new_rows.append((
                tx_date, 'buy', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Cash received from {symbol} sell (Healing)"
            ))
            # 2. Withdrawal (External flow)
            new_rows.append((
                tx_date, 'withdrawal', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Cash withdrawal from {symbol} sell proceeds (Healing)"
            ))
            print(f"Added settlement for SELL {symbol} on {tx_date}")
            
        elif tx_type == 'dividend':
            # Missing settlement for dividend
            # 1. Buy (Internal settlement)
            new_rows.append((
                tx_date, 'buy', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Dividend received for {symbol} (Healing)"
            ))
            # 2. Withdrawal (External flow)
            new_rows.append((
                tx_date, 'withdrawal', '$CASH', amount, 1.0, amount, 0.0, 'E*TRADE', 
                f"Auto-generated: Dividend withdrawal for {symbol} (Healing)"
            ))
            print(f"Added settlement for DIVIDEND {symbol} on {tx_date}")

    if new_rows:
        print(f"\nInserting {len(new_rows)} new cash transactions...")
        insert_query = """
            INSERT INTO transactions 
            (Date, Type, Symbol, Quantity, [Price/Share], [Total Amount], Commission, Account, Note, [Local Currency], user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'USD', 1)
        """
        cursor.executemany(insert_query, new_rows)
        conn.commit()
        print("Done.")
    else:
        print("No missing settlements found for E*TRADE.")
        
    conn.close()

if __name__ == "__main__":
    heal_etrade_cash()
