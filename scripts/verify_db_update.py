
import sys
import os
import logging
import sqlite3

# Convert to absolute path to find src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from db_utils import get_db_connection, get_database_path

def verify():
    path = get_database_path()
    print(f"DB Path: {path}")
    
    conn = get_db_connection()
    if not conn:
        print("Failed to connect.")
        return

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, intrinsic_value, universe, updated_at, ai_summary FROM screener_cache WHERE symbol IN ('APD', 'ALB')")
        rows = cursor.fetchall()
        print(f"Found {len(rows)} rows.")
        for row in rows:
            print(row)
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    verify()
