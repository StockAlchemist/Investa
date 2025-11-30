import sys
import os
import pandas as pd
import numpy as np
from datetime import date, datetime
from unittest.mock import MagicMock, patch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mock config before importing portfolio_logic
sys.modules['config'] = MagicMock()
sys.modules['config'].DEFAULT_CURRENCY = 'USD'
sys.modules['config'].HISTORICAL_CALC_METHOD = 'numba_chrono'
sys.modules['config'].HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = 'test_raw_cache'
sys.modules['config'].DAILY_RESULTS_CACHE_PATH_PREFIX = 'test_daily_cache'
sys.modules['config'].STOCK_QUANTITY_CLOSE_TOLERANCE = 1e-6
sys.modules['config'].DEBUG_DATE_VALUE = None
sys.modules['config'].HISTORICAL_DEBUG_DATE_VALUE = None
sys.modules['config'].HISTORICAL_DEBUG_SYMBOL = None
sys.modules['config'].LOGGING_LEVEL = 'INFO'

# Mock finutils
sys.modules['finutils'] = MagicMock()
sys.modules['finutils']._get_file_hash.return_value = 'dummy_hash'

# Mock market_data
sys.modules['market_data'] = MagicMock()

# Now import portfolio_logic
import portfolio_logic
print(f"Portfolio Logic File: {portfolio_logic.__file__}")
from portfolio_logic import calculate_historical_performance

def test_graph_range_all_accounts():
    print("--- Testing Graph Range for All Accounts ---")
    
    # 1. Setup Transactions
    # Account A: Starts 2020
    # Account B: Starts 2024
    data = {
        "Date": [pd.Timestamp("2020-01-01"), pd.Timestamp("2024-01-01")],
        "Account": ["Account A", "Account B"],
        "Symbol": ["AAPL", "AAPL"],
        "Type": ["BUY", "BUY"],
        "Quantity": [10, 10],
        "Split Ratio": [np.nan, np.nan],
        "To Account": [np.nan, np.nan],
        "Price": [100.0, 150.0],
        "Commission": [0.0, 0.0],
        "Currency": ["USD", "USD"],
        "Fx Rate": [1.0, 1.0],
        "Local Currency": ["USD", "USD"],
        "original_index": [0, 1],
        "Amount": [1000.0, 1500.0]
    }
    df = pd.DataFrame(data)
    
    # 2. Mock MarketDataProvider
    with patch('portfolio_logic.MarketDataProvider') as MockProvider:
        instance = MockProvider.return_value
        
        # Mock get_historical_data
        def mock_get_historical_data(symbols_yf, start_date, end_date, **kwargs):
            print(f"Mock get_historical_data called with start_date={start_date}, end_date={end_date}")
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            prices = pd.DataFrame(index=dates)
            prices['AAPL'] = 150.0 # Constant price
            prices['SPY'] = 400.0 # Benchmark
            return {'AAPL': prices[['AAPL']].rename(columns={'AAPL': 'price'}), 
                    'SPY': prices[['SPY']].rename(columns={'SPY': 'price'})}, {}
        
        instance.get_historical_data.side_effect = mock_get_historical_data
        
        # Mock get_historical_fx_rates
        instance.get_historical_fx_rates.return_value = ({}, {})
        
        # 3. Run Calculation
        # Case 1: include_accounts = None (All)
        print("\nCase 1: include_accounts=None")
        daily_df_1, _, _, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=date(2020, 1, 1),
            end_date=date(2025, 1, 1),
            interval="D",
            benchmark_symbols_yf=["SPY"],
            display_currency="USD",
            account_currency_map={},
            default_currency="USD",
            include_accounts=None,
            use_daily_results_cache=False, # Force calc
            num_processes=1,
            calc_method='numba_chrono'
        )
        
        if not daily_df_1.empty:
            min_date_1 = daily_df_1.index.min().date()
            print(f"Result Start Date: {min_date_1}")
            print(f"Columns: {daily_df_1.columns.tolist()}")
            if min_date_1 <= date(2020, 1, 5): # Allow small margin
                print("PASS: Case 1 starts in 2020")
            else:
                print("FAIL: Case 1 starts too late")
        else:
            print("FAIL: Daily DF empty")

        # Case 2: include_accounts = ['Account A', 'Account B'] (Explicit All)
        print("\nCase 2: include_accounts=['Account A', 'Account B']")
        daily_df_2, _, _, _ = calculate_historical_performance(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            start_date=date(2020, 1, 1),
            end_date=date(2025, 1, 1),
            interval="D",
            benchmark_symbols_yf=["SPY"],
            display_currency="USD",
            account_currency_map={},
            default_currency="USD",
            include_accounts=['Account A', 'Account B'],
            use_daily_results_cache=False, # Force calc
            num_processes=1,
            calc_method='numba_chrono'
        )
        
        if not daily_df_2.empty:
            min_date_2 = daily_df_2.index.min().date()
            print(f"Result Start Date: {min_date_2}")
            if min_date_2 <= date(2020, 1, 5):
                print("PASS: Case 2 starts in 2020")
            else:
                print("FAIL: Case 2 starts too late")
        else:
            print("FAIL: Daily DF empty")

if __name__ == "__main__":
    test_graph_range_all_accounts()
