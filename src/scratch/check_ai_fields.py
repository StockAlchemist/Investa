import sys
import os
import sqlite3
import json
from datetime import datetime

# Path to the database
db_path = "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/My Drive/Finance/Investa/data/db/investa_transactions.db"

def check_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check AAPL
    symbol = "AAPL"
    cursor.execute("SELECT ai_sentiment, ai_catalysts, intrinsic_value, updated_at FROM screener_cache WHERE symbol = ?", (symbol,))
    rows = cursor.fetchall()
    
    print(f"Results for {symbol}:")
    for row in rows:
        sentiment, catalysts, iv, updatedAt = row
        print(f"  Sentiment: {sentiment}")
        print(f"  Catalysts: {catalysts}")
        print(f"  Intrinsic Value: {iv}")
        print(f"  Updated At: {updatedAt}")
        
    conn.close()

if __name__ == "__main__":
    check_db()
