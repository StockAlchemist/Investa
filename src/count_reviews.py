import os
import sys
import sqlite3
from datetime import datetime

# Add current directory (src) to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

import config
from db_utils import get_db_connection

def main():
    print(f"--- Investa AI Review Analytics ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---")
    
    # 1. Check DB
    conn = get_db_connection()
    db_count = 0
    if conn:
        try:
            cursor = conn.cursor()
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='screener_cache';")
            if cursor.fetchone():
                cursor.execute("SELECT count(*) FROM screener_cache WHERE ai_summary IS NOT NULL AND ai_summary != '';")
                db_count = cursor.fetchone()[0]
                print(f"Database AI Reviews: {db_count}")
            else:
                print("Database table 'screener_cache' does not exist.")
        except Exception as e:
            print(f"Error querying database: {e}")
        finally:
            conn.close()
    else:
        print("Could not connect to database.")

    # 2. Check File Cache
    cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")
    file_count = 0
    if os.path.exists(cache_dir):
        files = [f for f in os.listdir(cache_dir) if f.endswith("_analysis.json")]
        file_count = len(files)
        print(f"File Cache AI Reviews: {file_count}")
        print(f"Location: {cache_dir}")
    else:
        print(f"File cache directory does not exist: {cache_dir}")

    print("-" * 50)

if __name__ == "__main__":
    main()
