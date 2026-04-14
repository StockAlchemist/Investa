import sys
import os
import logging
import sqlite3

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/My Drive/Finance/Investa"
src_dir = os.path.join(project_root, "src")
sys.path.insert(0, src_dir)

import config
from market_data import get_shared_mdp
from financial_ratios import get_current_valuation_ratios
from server.ai_analyzer import generate_stock_review
from db_utils import get_db_connection

logging.basicConfig(level=logging.INFO)

symbols = ["AAPL", "AMZN", "ASML", "GLD", "GOOG", "GOOGL", "MA"]

def refresh_symbols():
    mdp = get_shared_mdp()
    conn = get_db_connection()
    
    if not conn:
        print("Could not connect to database")
        return

    for sym in symbols:
        print(f"Refreshing {sym}...")
        try:
            # Fetch data
            fund = mdp.get_ticker_details(sym)
            ratios = get_current_valuation_ratios(sym, mdp)
            
            # This will fetch new AI analysis and save to DB
            # We use force_refresh=True to overwrite the previous incomplete data
            review = generate_stock_review(sym, fund, ratios, force_refresh=True)
            print(f"  Success: Sentiment={review.get('sentiment')}")
        except Exception as e:
            print(f"  Failed {sym}: {e}")
            
    conn.close()

if __name__ == "__main__":
    refresh_symbols()
