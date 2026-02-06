# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          screener_worker.py
 Purpose:       Standalone worker to pre-warm the market screener cache for all users.
                Runs periodically to ensure that when a user selects a screener view,
                the heavy intrinsic value calculations are already done.

 Usage:         Run from the 'src' directory or with 'src' in PYTHONPATH.
                python src/screener_worker.py

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
import sqlite3

# Ensure 'src' is in the python path if running from within src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

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

def get_user_databases():
    """Finds all user portfolio.db files and the global DB."""
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
    global_db = os.path.join(data_dir, "investa_transactions.db")
    if os.path.exists(global_db):
         db_paths.append(("GLOBAL", global_db))
         
    return db_paths

def run_screener_update():
    """
    Runs the screener for standard universes to populate the cache for all users.
    """
    logging.info("Starting scheduled Screener Cache Warm-up...")
    start_time = time.time()
    
    try:
        db_paths = get_user_databases()

        if not db_paths:
            logging.warning("No user databases found to update.")
            return

        logging.info(f"Found {len(db_paths)} databases to update.")
        
        # Standard universes to pre-warm
        universes = ["sp500", "sp400", "russell2000"]

        for user_name, db_path in db_paths:
            logging.info(f"  > Updating cache for user: {user_name}")
            try:
                # We open a dedicated connection for this operation to ensure thread safety
                # and avoid interference with any shared cache state if running concurrently (though this is single threaded)
                conn = sqlite3.connect(db_path)
                
                for uni in universes:
                    uni_start = time.time()
                    # fast_mode=False forces a refresh of missing/stale data (calculates intrinsic value)
                    results = screen_stocks(universe_type=uni, db_conn=conn, fast_mode=False)
                    
                    # specific verification
                    valid_count = sum(1 for r in results if r.get("intrinsic_value") is not None)
                    uni_duration = time.time() - uni_start
                    logging.info(f"    - {uni}: {valid_count}/{len(results)} valid. ({uni_duration:.1f}s)")
                
                conn.close()
            except Exception as e:
                logging.error(f"    - Failed to update user {user_name}: {e}")

        duration = time.time() - start_time
        logging.info(f"Completed Screener update cycle in {duration:.2f} seconds.")

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
