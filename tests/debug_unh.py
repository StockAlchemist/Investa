
import sys
import os
import logging
import json
import yfinance as yf

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from market_data import MarketDataProvider

def debug_unh():
    symbol = "UNH"
    
    print(f"--- Debugging {symbol} ---")
    
    # 1. Raw YFinance
    print("Fetching Raw YFinance info...")
    t = yf.Ticker(symbol)
    info = t.info
    print(f"Raw dividendYield: {info.get('dividendYield')}")
    print(f"Raw yield: {info.get('yield')}")
    print(f"Raw dividendRate: {info.get('dividendRate')}")
    print(f"Raw currentPrice: {info.get('currentPrice')}")
    print(f"Raw regularMarketPrice: {info.get('regularMarketPrice')}")
    
    # 2. MDP Logic
    # Clear cache for UNH first to ensure we test logic
    mdp = MarketDataProvider()
    cache_path = os.path.join(mdp.fundamentals_cache_dir, f"{symbol}.json")
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print("Cleared cache.")
        
    print("Fetching via MDP...")
    # Pass empty maps
    results = mdp.get_fundamentals_batch([symbol], {}, set())
    res = results.get(symbol, {})
    
    print(f"MDP Result dividendYield: {res.get('dividendYield')}")
    print(f"MDP Result dividendRate: {res.get('dividendRate')}")

if __name__ == "__main__":
    debug_unh()
