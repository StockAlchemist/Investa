
import sys
import os
import logging
import time
import sqlite3

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/server")))

from screener_service import screen_stocks
from db_utils import get_db_connection
import config

def repopulate_universe(universe_type, db_conn=None):
    print(f"  - Universe: {universe_type}")
    
    start_time = time.time()
    
    # Process the universe. 
    # fast_mode=False triggers full enrichment for items not already valid in cache.
    # Since we cleaned the poisoned cache, this targets exactly what we need.
    results = screen_stocks(universe_type=universe_type, db_conn=db_conn, fast_mode=False)
    
    duration = time.time() - start_time
    complete = sum(1 for r in results if r.get("intrinsic_value") is not None)
    print(f"    Completed in {duration:.2f}s. Valuations: {complete} / {len(results)}")

def get_user_databases():
    """Finds all user portfolio.db files."""
    data_dir = config.get_app_data_dir()
    users_dir = os.path.join(data_dir, "users")
    db_paths = []
    
    if os.path.exists(users_dir):
        for user_name in os.listdir(users_dir):
            user_path = os.path.join(users_dir, user_name)
            if os.path.isdir(user_path):
                db_path = os.path.join(user_path, "portfolio.db")
                if os.path.exists(db_path):
                    db_paths.append((user_name, db_path))
    
    # Also include the global DB for completeness
    global_db = os.path.join(data_dir, "investa_transactions.db")
    if os.path.exists(global_db):
        db_paths.append(("GLOBAL", global_db))
        
    return db_paths

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    
    universes = ["sp500", "sp400", "russell2000"]
    db_configs = get_user_databases()
    
    print("Starting Screener Repopulation (Multi-User Aware)...")
    print(f"Goal: Sync screener cache for {len(db_configs)} databases.")
    
    for user_name, db_path in db_configs:
        print(f"\nProcessing database for user: {user_name} ({db_path})")
        try:
            conn = sqlite3.connect(db_path)
            for uni in universes:
                repopulate_universe(uni, db_conn=conn)
            conn.close()
        except Exception as e:
            print(f"Error processing database for {user_name}: {e}")
            
    print("\nRepopulation Task Finished.")
