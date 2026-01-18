
import sys
import os
import logging
from datetime import datetime
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

# Clean up any existing handlers to avoid duplicates
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(level=logging.INFO)

from market_data import MarketDataProvider

def debug_fundamentals():
    mdp = MarketDataProvider()
    symbols = ["AAPL", "GOOG", "VHT"]
    
    print(f"Fetching fundamentals for: {symbols}")
    
    # Force fetch (ignore cache if possible, or print cache path)
    # The method uses cache, so if I run it once it might write to cache.
    # I'll check if cache exists first.
    
    results = mdp.get_fundamentals_batch(symbols)
    
    print("\nResults:")
    print(json.dumps(results, indent=2, default=str))
    
    # Check cache directory
    print(f"\nCache Dir: {mdp.fundamentals_cache_dir}")
    for sym in symbols:
        p = os.path.join(mdp.fundamentals_cache_dir, f"{sym}.json")
        if os.path.exists(p):
            print(f"Cache for {sym} exists at {p}")
            with open(p, 'r') as f:
                print(f.read()[:200] + "...")
        else:
            print(f"Cache for {sym} DOES NOT exist.")

if __name__ == "__main__":
    debug_fundamentals()
