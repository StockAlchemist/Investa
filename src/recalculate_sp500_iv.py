import sys
import os
import time
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from market_data import get_shared_mdp
from server.screener_service import get_sp500_tickers
from financial_ratios import get_intrinsic_value_for_symbol
from db_utils import get_db_connection, update_intrinsic_value_in_cache
from config_manager import ConfigManager

def recalculate_sp500():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    
    logging.info("Starting S&P 500 Intrinsic Value Recalculation...")
    
    mdp = get_shared_mdp()
    cm = ConfigManager(config.get_app_data_dir())
    tickers = get_sp500_tickers()
    
    if not tickers:
        logging.error("No tickers found for S&P 500.")
        return

    logging.info(f"Processing {len(tickers)} symbols...")
    
    conn = get_db_connection()
    if not conn:
        logging.error("Failed to connect to database.")
        return

    success_count = 0
    fail_count = 0
    
    # Process in chunks to avoid blocking and allow checking progress
    for i, symbol in enumerate(tickers):
        try:
            logging.info(f"[{i+1}/{len(tickers)}] Recalculating {symbol}...")
            
            # Use the high-quality centralized helper (Always fetches statements)
            iv_results = get_intrinsic_value_for_symbol(symbol, mdp, cm)
            
            if "error" in iv_results:
                logging.warning(f"  Calculation failed for {symbol}: {iv_results['error']}")
                fail_count += 1
                continue

            # Fetch info for cache sync (required for timestamps)
            # This uses the mdp cache if already fetched by the helper
            info = mdp.get_fundamental_data(symbol)
            
            # Update cache
            update_intrinsic_value_in_cache(
                conn,
                symbol,
                iv_results.get("average_intrinsic_value"),
                iv_results.get("margin_of_safety_pct"),
                info.get("lastFiscalYearEnd") if info else None,
                info.get("mostRecentQuarter") if info else None,
                info=info
            )
            
            success_count += 1
            
            # Throttle a bit if not using cache
            # if i % 10 == 0:
            #     time.sleep(1)
                
        except Exception as e:
            logging.error(f"  Error processing {symbol}: {e}")
            fail_count += 1
            
    conn.close()
    
    logging.info("--- Recalculation Complete ---")
    logging.info(f"Successfully processed: {success_count}")
    logging.info(f"Failed: {fail_count}")

if __name__ == "__main__":
    recalculate_sp500()
