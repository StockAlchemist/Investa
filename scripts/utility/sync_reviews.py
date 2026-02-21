# -*- coding: utf-8 -*-
import os
import sys
import json
import logging
import time

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import config
from db_utils import get_db_connection
from market_data import get_shared_mdp
from ai_review_worker import save_review_to_screener

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("Starting Sync of Missing Reviews...")
    
    # 1. Get File Cache Symbols
    cache_dir = os.path.join(config.get_app_data_dir(), "ai_analysis_cache")
    if not os.path.exists(cache_dir):
        logging.error("Cache directory not found.")
        return

    file_files = [f for f in os.listdir(cache_dir) if f.endswith("_analysis.json")]
    file_symbols = {f.split("_")[0].upper() for f in file_files}
    logging.info(f"Found {len(file_symbols)} symbols in file cache.")

    # 2. Get DB Symbols
    conn = get_db_connection()
    db_symbols = set()
    if conn:
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM screener_cache WHERE ai_summary IS NOT NULL AND ai_summary != ''")
            rows = cursor.fetchall()
            db_symbols = {r[0].upper() for r in rows}
            logging.info(f"Found {len(db_symbols)} symbols in DB with AI reviews.")
        finally:
            conn.close()
    
    # 3. Identify Missing
    missing_symbols = file_symbols - db_symbols
    logging.info(f"Identified {len(missing_symbols)} missing symbols to sync: {missing_symbols}")

    if not missing_symbols:
        logging.info("No missing reviews to sync.")
        return

    # 4. Sync
    mdp = get_shared_mdp()
    
    for i, symbol in enumerate(missing_symbols):
        logging.info(f"Syncing {symbol} ({i+1}/{len(missing_symbols)})...")
        
        try:
            # Load Review
            cache_path = os.path.join(cache_dir, f"{symbol}_analysis.json")
            with open(cache_path, "r") as f:
                data = json.load(f)
                review = data.get("analysis", {})
                
            if not review:
                logging.warning(f"Empty review data for {symbol}, assuming invalid.")
                continue

            # Fetch Metadata (Price, Name)
            # We need this to make the screener entry useful
            fund_data = mdp.get_fundamental_data(symbol)
            if not fund_data:
                logging.warning(f"Could not fetch live fund data for {symbol}, using minimal placeholders.")
                fund_data = {"shortName": symbol} # Fallback

            # Save to DB
            # We use 'sp500' or 'manual'? 
            # If it's in cache, it might have come from anywhere. 
            # But let's assume 'sp500' if it's S&P500, or just 'manual' to be safe?
            # User said "Add to screener table". 
            # Let's check if it's in S&P 500 list?
            # Actually, save_review_to_screener defaults to "sp500". 
            # It's probably safer to stick with "sp500" if we want them to show up there, 
            # or maybe "manual" if we are unsure.
            # Given the context of "worker adding AL reviews" (likely S&P500 worker), 'sp500' seems appropriate.
            
            save_review_to_screener(symbol, review, fund_data, universe="sp500")
            
            # Rate limit/politeness check for basic data fetching
            # Although get_fundamental_data is usually cached or lightweight if batching... 
            # But here we do one by one.
            time.sleep(0.5) 
            
        except Exception as e:
            logging.error(f"Failed to sync {symbol}: {e}")

    logging.info("Sync complete.")

if __name__ == "__main__":
    main()
