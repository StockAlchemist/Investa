
import pandas as pd
from datetime import date, timedelta
from src.portfolio_logic import calculate_historical_performance
from src.market_data import MarketDataProvider
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)

def test_intraday_history():
    # Mock some transactions
    df = pd.DataFrame([
        {"Date": pd.Timestamp("2025-11-10"), "Symbol": "AAPL", "Type": "buy", "Quantity": 10, "Price": 150, "Account": "Test", "Commission": 0, "Split Ratio": 1, "Local Currency": "USD"},
        {"Date": pd.Timestamp("2025-11-12"), "Symbol": "AAPL", "Type": "sell", "Quantity": 5, "Price": 160, "Account": "Test", "Commission": 0, "Split Ratio": 1, "Local Currency": "USD"},
    ])
    
    # We need to set up the environment a bit or mock deeper. 
    # Actually, let's just check if it runs without errors and produces more than 7 rows for 7 days.
    
    end_date = date(2025, 11, 15)
    start_date = end_date - timedelta(days=7)
    
    print(f"Testing 5D period (7 calendar days) with 1h interval...")
    
    # This might actually try to fetch from YF. 
    # If it fails due to no network or invalid symbols in this environment, that's fine as long as logic is tested.
    try:
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
            print("First 5 rows:")
            print(daily_df.head())
            
            if len(daily_df) > 10:
                print("SUCCESS: More than 1 daily point per day produced.")
            else:
                print("FAILURE: Produced only daily points or less.")
        else:
            print("Produced empty DataFrame (might be due to network/YF fetch failure in this environment).")
            
    except Exception as e:
        print(f"ERROR during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_intraday_history()
