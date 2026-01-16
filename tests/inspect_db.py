import yaml
import sqlite3
import pandas as pd
from datetime import date, timedelta
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))
from market_data import MarketDataProvider
import config

def check_db():
    print(f"Checking DB at: {config.get_app_data_dir()}")
    # ... (skipping direct sqlite check for brevity, focusing on provider)
    
    provider = MarketDataProvider()
    end = date.today()
    start = end - timedelta(days=10)
    
    print("\n--- Testing Benchmark Data ---")
    data, failed = provider.get_historical_data(["^GSPC", "^DJI", "^IXIC"], start, end)
    for sym in ["^GSPC", "^DJI", "^IXIC"]:
        if sym in data and not data[sym].empty:
            print(f"{sym} Data Retrieved: {len(data[sym])} rows")
            print(f"Columns: {data[sym].columns}")
            print(data[sym].head(1))
        else:
            print(f"{sym} Not returned by provider!")

if __name__ == "__main__":
    check_db()
