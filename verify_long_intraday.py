
import pandas as pd
from datetime import date, timedelta
from src.portfolio_logic import calculate_historical_performance
from src.market_data import MarketDataProvider
import logging
import os

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def test_long_intraday_history():
    # Mock transactions going back to 2002
    df = pd.DataFrame([
        {"Date": pd.Timestamp("2002-06-29"), "Symbol": "AAPL", "Type": "buy", "Quantity": 10, "Price": 1.0, "Account": "Test", "Commission": 0, "Split Ratio": 1, "Local Currency": "USD", "original_index": 0},
        {"Date": pd.Timestamp("2025-12-25"), "Symbol": "AAPL", "Type": "sell", "Quantity": 5, "Price": 200.0, "Account": "Test", "Commission": 0, "Split Ratio": 1, "Local Currency": "USD", "original_index": 1},
    ])
    
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    
    print(f"Testing 5D period with transactions since 2002. Requested interval: 1h")
    
    try:
        # This will now fetch Daily from 2002 and Hourly from 7 days ago.
        daily_df, _, _, status = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=None,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=start_date,
            end_date=end_date,
            interval="1h",
            benchmark_symbols_yf=["SPY"],
            display_currency="USD",
            account_currency_map={"Test": "USD"},
            default_currency="USD",
            include_accounts=None,
            user_symbol_map={"AAPL": "AAPL"}
        )
        
        print(f"Status: {status}")
        if not daily_df.empty:
            print(f"Produced {len(daily_df)} rows.")
            # We expect rows for the entire history or at least the active range?
            # calculate_historical_performance returns the grouped results.
            # If period is 5D, it should return hourly points for the last week.
            print("First 5 rows:")
            print(daily_df.head())
            print("Last 5 rows:")
            print(daily_df.tail())
            
            # Check if it has hourly points (freq should be H or there should be many points)
            if len(daily_df) > 24:
                 print("SUCCESS: Hourly points produced without crashing on 2002 data.")
            else:
                 print("INFO: Only daily points or few points produced. Check grouping logic.")
        else:
            print("Produced empty DataFrame (expected if network is restricted, but it shouldn't CRASH).")
            
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_long_intraday_history()
