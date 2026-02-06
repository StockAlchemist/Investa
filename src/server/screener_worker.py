# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          screener_worker.py
 Purpose:       Standalone worker to pre-warm the market screener cache for the "All Universes" view.
                This runs periodically to ensure that when a user selects "All Market",
                the heavy intrinsic value calculations are already done.

 Usage:         Run from the 'src' directory or with 'src' in PYTHONPATH.
                python screener_worker.py

 Author:        Google Gemini

 Copyright:     (c) Investa Contributors 2026
 Licence:       MIT
 -------------------------------------------------------------------------------
"""
import time
import logging
import os
import sys
import datetime

# Ensure 'src' is in the python path if running from within src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    # If we are in src/server, we need to go up one level
    # Actually, the convention in this project effectively puts src in path or runs from root.
    # Let's try to add the parent of the directory containing this script.
    sys.path.append(os.path.abspath(os.path.join(current_dir, "..")))

# Import necessary modules
try:
    import config
    from server.screener_service import screen_stocks
    from db_utils import get_db_connection
except ImportError as e:
    print(f"CRITICAL: Failed to import modules. Make sure you are running with 'src' in PYTHONPATH. Error: {e}")
    sys.exit(1)

# Configuration
UPDATE_INTERVAL_SECONDS = 4 * 3600  # 4 Hours
RETRY_DELAY_SECONDS = 300           # 5 Minutes

def run_screener_update():
    """
    Runs the screener for the "all" universe to populated the cache.
    """
    logging.info("Starting scheduled 'All Market' screener update...")
    start_time = time.time()
    
    try:
        # We don't need to pass a specific db_conn, screen_stocks handles it (using shared global or per-user logic if adapted)
        # Note: screen_stocks currently caches to the GLOBAL cache / DB tables usually.
        # If user-specific logic is needed, we might need to iterate users like in repopulate_screener.py
        # But for now, let's assume the "all" view relies on a shared metadata cache (or we just populate the main app DB).
        
        # Based on screener_service, it uses `get_db_connection()` which usually points to the main app DB if not specified ??
        # Wait, `get_db_connection` in `db_utils` might need a user context if it's per-user.
        # Let's check `db_utils.get_db_connection`. 
        # Typically in this codebase, there is a global cache or a specific user cache.
        # `repopulate_screener.py` iterates over user DBs. 
        # CAUTION: If we run this as a generic worker, we should probably update ALL user DBs?
        # Or is there a shared cache? 
        # screener_service: `upsert_screener_results` writes to the provided connection.
        # If we use `get_db_connection()` without arguments, what does it verify?
        
        # Let's mimic `repopulate_screener.py` logic to be safe and update for ALL found users.
        # This ensures every user gets the fast experience.
        
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
        
        # Also include the global DB for completeness if it exists
        global_db = os.path.join(data_dir, "investa_transactions.db") # Fallback default
        if os.path.exists(global_db):
            # Check if this is already covered? No, users have their own.
             db_paths.append(("GLOBAL", global_db))

        if not db_paths:
            logging.warning("No user databases found to update.")
            return

        logging.info(f"Found {len(db_paths)} databases to update.")

        # Import sqlite3 here to avoid global dep if not needed
        import sqlite3

        for user_name, db_path in db_paths:
            logging.info(f"  > Updating cache for user: {user_name}")
            try:
                conn = sqlite3.connect(db_path)
                # fast_mode=False forces a refresh of missing/stale data
                results = screen_stocks(universe_type="all", db_conn=conn, fast_mode=False)
                
                # specific verification
                valid_count = sum(1 for r in results if r.get("intrinsic_value") is not None)
                logging.info(f"    - Updated {len(results)} symbols. Valid Valuations: {valid_count}")
                conn.close()
            except Exception as e:
                logging.error(f"    - Failed to update user {user_name}: {e}")

        duration = time.time() - start_time
        logging.info(f"Completed 'All Market' update cycle in {duration:.2f} seconds.")

    except Exception as e:
        logging.error(f"Error during screener update: {e}", exc_info=True)

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - [ScreenerWorker] - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("Screener Background Worker Initialized.")
    logging.info(f"Target Update Interval: Every {UPDATE_INTERVAL_SECONDS/3600:.1f} hours.")

    while True:
        try:
            run_screener_update()
            
            logging.info(f"Sleeping for {UPDATE_INTERVAL_SECONDS/3600:.1f} hours...")
            time.sleep(UPDATE_INTERVAL_SECONDS)
            
        except KeyboardInterrupt:
            logging.info("Worker stopped by user.")
            sys.exit(0)
        except Exception as e:
            logging.error(f"Global worker loop exception: {e}")
            logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
            time.sleep(RETRY_DELAY_SECONDS)

if __name__ == "__main__":
    main()
