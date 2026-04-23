import sqlite3
import os
import pandas as pd

def heal_all_cash():
    db_path = 'data/users/kitmatan/portfolio.db'
    backup_path = db_path + '.backup'
    
    # Backup
    if not os.path.exists(backup_path):
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"Backed up database to {backup_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Load all non-cash transactions
    query = "SELECT id, Date, Type, Symbol, Quantity, [Price/Share], [Total Amount], Commission, Account, [Local Currency], user_id FROM transactions WHERE Symbol != '$CASH' ORDER BY Date"
    df = pd.read_sql_query(query, conn)
    
    # Load existing cash transactions to avoid duplicates
    cash_query = "SELECT Date, Type, [Total Amount], Account FROM transactions WHERE Symbol = '$CASH'"
    cash_df = pd.read_sql_query(cash_query, conn)
    
    new_rows = []
    
    # Track days processed to avoid double adding on same day/account
    # (Though checking existing cash is better)
    
    for _, row in df.iterrows():
        tx_date = row['Date']
        tx_type = row['Type'].lower().strip()
        symbol = row['Symbol']
        amount = abs(row['Total Amount'])
        acc = row['Account']
        currency = row['Local Currency'] or 'USD'
        user_id = row['user_id'] or 1
        
        # Check if ANY cash transaction exists in this account on this day
        existing = cash_df[(cash_df['Date'] == tx_date) & (cash_df['Account'] == acc)]
        if not existing.empty:
            continue
        
        if tx_type == 'buy':
            new_rows.append((tx_date, 'deposit', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Cash deposit for {symbol} buy (Healing)", currency, user_id))
            new_rows.append((tx_date, 'sell', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Cash settlement for {symbol} buy (Healing)", currency, user_id))
        elif tx_type == 'sell':
            new_rows.append((tx_date, 'buy', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Cash received from {symbol} sell (Healing)", currency, user_id))
            new_rows.append((tx_date, 'withdrawal', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Cash withdrawal from {symbol} sell proceeds (Healing)", currency, user_id))
        elif tx_type == 'dividend':
            new_rows.append((tx_date, 'buy', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Dividend received for {symbol} (Healing)", currency, user_id))
            new_rows.append((tx_date, 'withdrawal', '$CASH', amount, 1.0, amount, 0.0, acc, f"Auto-generated: Dividend withdrawal for {symbol} (Healing)", currency, user_id))

    if new_rows:
        print(f"\nInserting {len(new_rows)} new cash transactions for all missing accounts...")
        insert_query = """
            INSERT INTO transactions 
            (Date, Type, Symbol, Quantity, [Price/Share], [Total Amount], Commission, Account, Note, [Local Currency], user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        cursor.executemany(insert_query, new_rows)
        conn.commit()
        print("Done.")
    else:
        print("No missing settlements found.")
        
    conn.close()

if __name__ == "__main__":
    heal_all_cash()
