
import os
import sqlite3
import json
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import config

def fix_db_cache():
    print("Fixing database screener_cache...")
    db_path = os.path.join(config.get_app_data_dir(), "investa_transactions.db")
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. Delete entries with NULL intrinsic_value
        cursor.execute("DELETE FROM screener_cache WHERE intrinsic_value IS NULL")
        deleted_null_iv = cursor.rowcount
        
        # 2. Delete entries with NULL identifiers (poisoned data)
        cursor.execute("DELETE FROM screener_cache WHERE last_fiscal_year_end IS NULL OR most_recent_quarter IS NULL")
        deleted_null_id = cursor.rowcount
        
        conn.commit()
        conn.close()
        print(f"  Deleted {deleted_null_iv} entries with NULL intrinsic_value.")
        print(f"  Deleted {deleted_null_id} entries with missing financial identifiers.")
    except Exception as e:
        print(f"  Error fixing database: {e}")

def fix_json_cache():
    print("Fixing JSON fundamentals cache...")
    cache_dir = os.path.join(config.get_app_data_dir(), "fundamentals_cache")
    if not os.path.exists(cache_dir):
        print(f"Cache directory not found at {cache_dir}")
        return

    deleted_count = 0
    files = [f for f in os.listdir(cache_dir) if f.endswith(".json")]
    
    for filename in files:
        file_path = os.path.join(cache_dir, filename)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            
            # Check the "data" part of the cache
            # The structure in market_data.py is: {"timestamp": ..., "data": {...}}
            data = content.get("data", {})
            
            is_poisoned = False
            if not data or len(data) <= 8:
                is_poisoned = True
            
            # Specifically for Equities, check identifiers
            if data and data.get("quoteType", "").upper() == "EQUITY":
                if not data.get("lastFiscalYearEnd") and not data.get("mostRecentQuarter"):
                    is_poisoned = True
            
            if is_poisoned:
                os.remove(file_path)
                deleted_count += 1
                # print(f"  Deleted poisoned cache for {filename}")
        except Exception as e:
            print(f"  Error processing {filename}: {e}")

    print(f"  Deleted {deleted_count} poisoned JSON cache files.")

if __name__ == "__main__":
    fix_db_cache()
    fix_json_cache()
    print("Done.")
