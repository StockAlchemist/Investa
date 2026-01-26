import os
import sys
import sqlite3

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from db_utils import get_db_connection

def main():
    # 1. Check DB
    conn = get_db_connection()
    db_count = 0
    if conn:
        try:
            cursor = conn.cursor()
            # Check if table exists first
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='screener_cache';")
            if cursor.fetchone():
                cursor.execute("SELECT count(*) FROM screener_cache WHERE ai_summary IS NOT NULL AND ai_summary != '';")
                db_count = cursor.fetchone()[0]
                print(f"Database AI Reviews: {db_count}")
            else:
                print("Database table 'screener_cache' does not exist.")
        finally:
            conn.close()
    else:
        print("Could not connect to database.")

    # 2. Check File Cache
    cache_dir = os.path.join(config.get_app_data_dir(), "ai_analysis_cache")
    file_count = 0
    if os.path.exists(cache_dir):
        files = [f for f in os.listdir(cache_dir) if f.endswith("_analysis.json")]
        file_count = len(files)
        print(f"File Cache AI Reviews: {file_count}")
    else:
        print("File cache directory does not exist.")

if __name__ == "__main__":
    main()
