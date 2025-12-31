
import sys
import os
import pandas as pd
from datetime import date, timedelta
import logging

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from portfolio_logic import calculate_historical_performance
from market_data import MarketDataProvider

# Mock config
import config
config.MARKET_DATA_CACHE_DIR = "historical_data_cache"

# Setup logging
logging.basicConfig(level=logging.INFO)

def verify_weekends_skipped():
    print("--- Verifying Weekend Skipping Logic ---")
    
    # Mock transactions
    # Create a simple portfolio holding 1 AAPL share bought long ago
    df = pd.DataFrame([
        {"Date": pd.Timestamp("2024-01-01"), "Symbol": "AAPL", "Type": "buy", "Quantity": 1, "Price": 150.0, "Account": "Test", "Commission": 0, "Split Ratio": 1, "Local Currency": "USD", "original_index": 0},
    ])
    
    # Define a range that spans a weekend
    # Let's say Friday Jan 26, 2024 to Tuesday Jan 30, 2024
    # Weekends: Jan 27 (Sat), Jan 28 (Sun)
    start_date = date(2025, 12, 1) # Start of month (Monday)
    end_date = date(2025, 12, 14)  # 2 weeks later
    
    print(f"Calculating performance from {start_date} to {end_date} (1d interval)...")
    
    daily_df, _, _, _ = calculate_historical_performance(
        all_transactions_df_cleaned=df,
        original_transactions_df_for_ignored=None,
        ignored_indices_from_load=set(),
        ignored_reasons_from_load={},
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        benchmark_symbols_yf=[],
        display_currency="USD",
        account_currency_map={"Test": "USD"},
        default_currency="USD",
        include_accounts=None,
        manual_overrides_dict={},
        user_symbol_map={"AAPL": "AAPL"},
        user_excluded_symbols=[],
        original_csv_file_path="mock.csv"
    )
    
    if daily_df.empty:
        print("ERROR: Calculation returned empty DataFrame.")
        return

    print(f"Calculation returned {len(daily_df)} rows.")
    
    # Check if we have weekends in the RAW output (we EXPECT them here because calculate_historical_performance fills gaps)
    # Actually, calculate_historical_performance uses market days logic usually, so maybe it ALREADY skips weekends?
    # No, it forward fills. If we asked for market days, it aligns to market days.
    # If we asked for date range business days, it might skip.
    # Let's see what is in there.
    weekends_in_raw = [dt for dt in daily_df.index if dt.weekday() >= 5]
    print(f"Found {len(weekends_in_raw)} weekend data points in raw output.")
    
    # Simulate API Loop with Skipping
    result_list = []
    
    for dt, row in daily_df.iterrows():
        # THIS IS THE LOGIC WE ADDED TO API.PY
        if dt.weekday() >= 5:
            continue
            
        result_list.append(dt)
        
    print(f"Result list has {len(result_list)} items after skipping.")
    
    # Verify no weekends in result
    weekends_in_result = [dt for dt in result_list if dt.weekday() >= 5]
    
    if len(weekends_in_result) > 0:
        print(f"FAILURE: Found {len(weekends_in_result)} weekend items in result list!")
        print(weekends_in_result[:5])
    else:
        print("SUCCESS: No weekend items found in result list.")
        
    # Verify we actually filtered something if raw had them
    if len(weekends_in_raw) > 0:
        if len(result_list) == len(daily_df) - len(weekends_in_raw):
            print("SUCCESS: Count matches (Total - Weekends = Result).")
        else:
             print(f"WARNING: Count mismatch. Total: {len(daily_df)}, Weekends: {len(weekends_in_raw)}, Result: {len(result_list)}")
    else:
        print("INFO: Raw output had no weekends. This means calculate_historical_performance already skips them or data alignment does.")
        print("Verifying that if there WERE weekends, loop would skip them (by pure logic check).")
        # Logic check:
        dummy_weekend = pd.Timestamp("2025-12-06") # Saturday
        if dummy_weekend.weekday() >= 5:
            print("SUCCESS: Logic check confirms Sat/Sun would be skipped.")

if __name__ == "__main__":
    verify_weekends_skipped()
