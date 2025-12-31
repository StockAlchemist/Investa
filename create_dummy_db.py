
import sqlite3
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

import db_utils
import config

DB_FILE = 'investa_transactions.db'
CSV_FILE = 'tests/sample_transactions.csv'

def create_dummy_db():
    if os.path.exists(DB_FILE):
        print(f"Removing existing {DB_FILE}...")
        os.remove(DB_FILE)
    
    print(f"Creating new {DB_FILE}...")
    conn = sqlite3.connect(DB_FILE)
    
    print("Initializing tables...")
    db_utils.create_transactions_table(conn)
    
    print(f"Importing dummy data from {CSV_FILE}...")
    # Using defaults for currency map and default currency
    db_utils.migrate_csv_to_db(CSV_FILE, conn, {}, 'USD')
    
    conn.commit()
    conn.close()
    print("Dummy database created successfully.")

if __name__ == "__main__":
    create_dummy_db()
