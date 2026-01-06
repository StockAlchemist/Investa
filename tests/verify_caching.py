import sys
import os
import asyncio
import time
import pandas as pd
import logging

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), "src")
sys.path.append(src_dir)

# Mock config
import config
config.DEFAULT_CURRENCY = "USD"

from server.api import _calculate_portfolio_summary_internal, _PORTFOLIO_SUMMARY_CACHE

# Setup Logging
logging.basicConfig(level=logging.INFO)

async def test_caching():
    print("--- Starting Cache Verification ---")
    
    # 1. Setup Mock Data
    df = pd.DataFrame([{
        "Date": pd.to_datetime("2023-01-01"),
        "Symbol": "AAPL",
        "Type": "Buy",
        "Quantity": 10.0,
        "Price/Share": 150.0,
        "Total Amount": 1500.0,
        "Account": "TestAcct",
        "Local Currency": "USD",
        "Currency": "USD",
        "Commission": 0.0,
        "Split Ratio": 1.0,
        "Action": "Buy",
        "original_index": 0,
        "Name": "Apple Inc.",
        "Sector": "Technology",
        "Note": ""
    }])
    
    manual_overrides = {}
    user_symbol_map = {}
    user_excluded_symbols = set()
    account_currency_map = {"TestAcct": "USD"}
    db_path = "/tmp/test.db"
    db_mtime = 100.0
    
    data = (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        db_path,
        db_mtime
    )
    
    # Clear Cache initially
    _PORTFOLIO_SUMMARY_CACHE.clear()
    
    # 2. First Call (Expected Cache Miss)
    t0 = time.time()
    result1 = await _calculate_portfolio_summary_internal(
        currency="USD",
        include_accounts=None,
        show_closed_positions=True,
        data=data
    )
    t1 = time.time()
    duration1 = t1 - t0
    print(f"1st Call Duration: {duration1:.4f}s")
    
    # Verify Cache Entry Exists
    cache_key = ("USD", "ALL", True, db_path, db_mtime)
    if cache_key in _PORTFOLIO_SUMMARY_CACHE:
        print("SUCCESS: Cache entry created.")
    else:
        print("FAILURE: Cache entry NOT created.")
        print(f"Current Cache Keys: {_PORTFOLIO_SUMMARY_CACHE.keys()}")

    # 3. Second Call (Expected Cache Hit)
    t2 = time.time()
    result2 = await _calculate_portfolio_summary_internal(
        currency="USD",
        include_accounts=None,
        show_closed_positions=True,
        data=data
    )
    t3 = time.time()
    duration2 = t3 - t2
    print(f"2nd Call Duration: {duration2:.4f}s")
    
    # 4. Assertions
    if duration2 < duration1:
         print(f"SUCCESS: 2nd call was faster ({duration2:.4f} vs {duration1:.4f})")
    else:
         print(f"WARNING: 2nd call was NOT faster (Simulated data might be too small to show diff)")
         
    # 5. Test Evaluation Invalidation (Change mtime)
    print("\n--- Testing Invalidation ---")
    new_mtime = 200.0
    data_new = (
        df,
        manual_overrides,
        user_symbol_map,
        user_excluded_symbols,
        account_currency_map,
        db_path,
        new_mtime
    )
    
    t4 = time.time()
    result3 = await _calculate_portfolio_summary_internal(
        currency="USD",
        include_accounts=None,
        show_closed_positions=True,
        data=data_new
    )
    t5 = time.time()
    duration3 = t5 - t4
    print(f"3rd Call (New MTime) Duration: {duration3:.4f}s")
    
    new_cache_key = ("USD", "ALL", True, db_path, new_mtime)
    if new_cache_key in _PORTFOLIO_SUMMARY_CACHE:
         print("SUCCESS: New cache entry created for new mtime.")
    else:
         print("FAILURE: New cache entry NOT created.")

if __name__ == "__main__":
    asyncio.run(test_caching())
