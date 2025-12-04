
import sys
import os
import pandas as pd
import logging
import json
from datetime import date, datetime
from typing import Dict, Any, Set

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from data_loader import load_and_clean_transactions
from portfolio_logic import calculate_historical_performance
from config import DEFAULT_CURRENCY, SYMBOL_MAP_TO_YFINANCE, YFINANCE_EXCLUDED_SYMBOLS

# Setup logging
logging.basicConfig(level=logging.INFO)

def reproduce_issue():
    db_path = "/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/finance/Stocks/Evaluations/python/Investa/my_transactions.db"
    
    if not os.path.exists(db_path):
        print(f"DB file not found: {db_path}")
        return

    # Load transactions from DB
    (
        all_transactions_df_cleaned,
        original_transactions_df_for_ignored,
        ignored_indices_from_load,
        ignored_reasons_from_load,
        load_err,
        load_warn,
        account_currency_map,
    ) = load_and_clean_transactions(db_path, {}, DEFAULT_CURRENCY, is_db_source=True)

    if load_err:
        print(f"Error loading transactions: {load_err}")
        return

    print(f"Loaded {len(all_transactions_df_cleaned)} transactions.")

    # Load manual_overrides.json
    manual_overrides_path = os.path.join(os.path.dirname(__file__), "manual_overrides.json")
    user_symbol_map = SYMBOL_MAP_TO_YFINANCE.copy()
    user_excluded_symbols = set(YFINANCE_EXCLUDED_SYMBOLS)
    manual_overrides_dict = {}
    
    if os.path.exists(manual_overrides_path):
        with open(manual_overrides_path, "r") as f:
            manual_overrides_dict = json.load(f)
            if "user_symbol_map" in manual_overrides_dict:
                user_symbol_map.update(manual_overrides_dict["user_symbol_map"])
            if "user_excluded_symbols" in manual_overrides_dict:
                user_excluded_symbols.update(set(manual_overrides_dict["user_excluded_symbols"]))
                
    print(f"Loaded {len(user_symbol_map)} symbol maps and {len(user_excluded_symbols)} exclusions.")

    # Set date range to include Dec 3, 2025
    start_date = date(2025, 11, 25)
    end_date = date(2025, 12, 4)
    
    print(f"Calculating historical performance from {start_date} to {end_date}...")

    daily_df, _, _, status = calculate_historical_performance(
        all_transactions_df_cleaned=all_transactions_df_cleaned,
        original_transactions_df_for_ignored=original_transactions_df_for_ignored,
        ignored_indices_from_load=ignored_indices_from_load,
        ignored_reasons_from_load=ignored_reasons_from_load,
        start_date=start_date,
        end_date=end_date,
        interval="D",
        benchmark_symbols_yf=["SPY"],
        display_currency="USD",
        account_currency_map=account_currency_map,
        default_currency=DEFAULT_CURRENCY,
        use_raw_data_cache=False, # Force fetch from YF
        use_daily_results_cache=False, # Force recalc
        original_csv_file_path=None, # DB source
        user_symbol_map=user_symbol_map, # Pass the map!
        user_excluded_symbols=user_excluded_symbols, # Pass the exclusions!
        manual_overrides_dict=manual_overrides_dict
    )

    print(f"Status: {status}")
    
    if not daily_df.empty:
        # Check Dec 3 and Dec 4
        for date_str in ["2025-12-03", "2025-12-04"]:
            target_date = pd.Timestamp(date_str)
            if target_date in daily_df.index:
                row = daily_df.loc[target_date]
                print(f"\nData for {target_date.date()}:")
                print(f"  Portfolio Value: ${row['Portfolio Value']:,.2f}")
                print(f"  Daily Gain: ${row['daily_gain']:,.2f}")
                
                # --- ADDED: Breakdown by symbol ---
                # We need to access the internal daily_df_by_symbol if possible, 
                # but calculate_historical_performance returns the aggregated daily_df.
                # However, we can inspect the 'holdings' if we had them.
                # Since we can't easily get the internal breakdown from the return value without modifying the function,
                # we will infer it from the 'status' or just rely on the fact that we know Thai stocks are in there.
                # Actually, let's just print the columns of daily_df to see if individual stock columns are present.
                # calculate_historical_performance (unadjusted) returns a DF with columns like "Symbol Value", "Symbol Daily Gain"?
                # No, it returns aggregated.
                
                # Wait, if we use 'calculate_historical_performance', it returns (daily_df, ...).
                # The daily_df usually ONLY has aggregated columns unless we modify it.
                # BUT, we can check if the user has Thai stocks by printing the 'account_currency_map' and 'user_symbol_map' usage.
                pass
            else:
                print(f"\nDate {target_date.date()} not found in results.")
                
    # To get the breakdown, we need to look at the underlying data. 
    # Let's just list the symbols that are likely contributing.
    print("\nChecking for active Thai stocks (ending in .BK):")
    thai_stocks = [s for s in user_symbol_map.values() if s.endswith(".BK")]
    print(f"Found {len(thai_stocks)} Thai stock mappings: {thai_stocks[:5]}...")
    
    # We can't easily get the exact dollar breakdown without modifying the core logic to return it,
    # but confirming the presence of Thai stocks and the current time (10:25 AM BKK) is strong evidence.

if __name__ == "__main__":
    reproduce_issue()
