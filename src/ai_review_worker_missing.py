import sys
import os
import time
import logging

# Ensure 'src' is in the python path if running from within src
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from ai_review_worker import process_ticker_list, _open_screener_conn
from market_data import get_shared_mdp

def get_missing_reviews():
    """Fetches symbols from all universes that do not have an AI summary."""
    conn = _open_screener_conn()
    if not conn:
        return {}
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT symbol, universe FROM screener_cache WHERE ai_summary IS NULL OR trim(ai_summary) = ''")
        rows = cursor.fetchall()
        missing = {}
        for symbol, universe in rows:
            if universe not in missing:
                missing[universe] = []
            missing[universe].append(symbol)
        return missing
    finally:
        conn.close()

def main():
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logging.info("Starting AI Review Worker for Missing Reviews...")
    
    missing = get_missing_reviews()
    total_missing = sum(len(tickers) for tickers in missing.values())
    logging.info(f"Found {total_missing} stocks missing AI reviews across {len(missing)} universes.")
    
    if total_missing == 0:
        logging.info("No missing reviews found. Exiting.")
        return

    # Get Market Data Provider
    mdp = get_shared_mdp()
    
    try:
        for universe, tickers in missing.items():
            process_ticker_list(mdp, tickers, universe)
        logging.info("Completed processing missing reviews.")
    except KeyboardInterrupt:
        logging.info("Worker stopped by user.")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Global worker exception: {e}", exc_info=True)

if __name__ == "__main__":
    main()
