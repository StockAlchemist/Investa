
import sys
import os
import cProfile
import pstats
import logging
import pandas as pd
import sqlite3
import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))

import config
from db_utils import get_database_path, load_all_transactions_from_db
from workers import PortfolioCalculatorWorker
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
from market_data import MarketDataProvider

# Mock Signals to capture worker output
class MockSignal:
    def __init__(self, name):
        self.name = name

    def emit(self, *args):
        # Print errors to help debugging
        if self.name == "error":
            print(f"SIGNAL ERROR: {args}")
        elif self.name == "result":
            # Just acknowledge result
            pass
        elif self.name == "finished":
            pass

class MockWorkerSignals:
    def __init__(self):
        self.finished = MockSignal("finished")
        self.error = MockSignal("error")
        self.result = MockSignal("result")
        self.progress = MockSignal("progress")
        self.fundamental_data_ready = MockSignal("fundamental_data_ready")

def profile_refresh():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Initialize Qt App for QStandardPaths
    try:
        from PySide6.QtCore import QCoreApplication
        if not QCoreApplication.instance():
            app = QCoreApplication(sys.argv)
        else:
            app = QCoreApplication.instance()
        app.setOrganizationName(config.ORG_NAME)
        app.setApplicationName(config.APP_NAME)
    except ImportError:
        print("PySide6 not found. Path resolution might be incorrect.")

    # DB Connection
    # db_path = get_database_path()
    db_path = "/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/finance/Stocks/Evaluations/python/Investa/src/Investa/investa_transactions.db"
    print(f"Resolved DB Path: {db_path}")
    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    # Load Data
    # Define a default map for profiling
    acc_map = {"SET": "THB", "IBKR": "USD", "Cash": "USD"} 
    def_curr = config.DEFAULT_CURRENCY
    
    print("Loading data from DB...")
    df_all, success = load_all_transactions_from_db(conn, acc_map, def_curr)
    
    if not success or df_all is None or df_all.empty:
        print("Failed to load data or empty DB")
        return

    print(f"Loaded {len(df_all)} transactions.")

    # Prepare Worker Arguments
    # Mimicking main_gui.py refresh_data logic
    # --- 3. Run the refresh logic (get_portfolio_summary) ---
    # This invokes market_data.get_current_quotes internally.
    # To verify FX, we can inspect the market_provider after run (if caching holds) 
    # OR we can just add a direct call here since we have the provider instance.
    
    print("\n--- VERIFYING FX RATES ---")
    provider = MarketDataProvider() # Assuming provider is initialized here or earlier
    q, fx, fx_prev, err, warn = provider.get_current_quotes(
         internal_stock_symbols=["AAPL"], # Dummy
         required_currencies={"THB", "EUR", "JPY", "GBP"},
         user_symbol_map={},
         user_excluded_symbols=set()
    )
    for c, r in fx.items():
        print(f"FX {c}: {r}")
    print("--------------------------\n")

    # This part of the instruction seems to be a misplacement or an example of a different context.
    # The original code continues with `portfolio_kwargs` definition.
    # I will insert the logging command as requested and ensure the original structure is maintained.
    # The `result = calculate_portfolio_summary(...)` call is not part of the `profile_refresh` function's
    # main flow at this point, as it's preparing arguments for a worker.
    # I will assume the user intended to add the FX verification block and then continue with the original `portfolio_kwargs`.

    # Portfolio Summary Args
    portfolio_kwargs = {
        "all_transactions_df_cleaned": df_all,
        "original_transactions_df_for_ignored": df_all.copy(),
        "ignored_indices_from_load": set(),
        "ignored_reasons_from_load": {},
        "display_currency": def_curr, 
        "include_accounts": None, # All accounts
        "manual_overrides_dict": {},
        "user_symbol_map": {},
        "user_excluded_symbols": set(),
        "default_currency": def_curr,
        "account_currency_map": acc_map,
        "all_transactions_df_for_worker": df_all, # used for correlation
        "market_provider": MarketDataProvider() # Injected dependency since latest updates
    }
    
    # Historical Performance Args
    start_date = datetime.date.today() - datetime.timedelta(days=365*5)
    end_date = datetime.date.today()
    
    historical_kwargs = {
        "all_transactions_df_cleaned": df_all,
        "original_transactions_df_for_ignored": df_all.copy(),
        "ignored_indices_from_load": set(),
        "ignored_reasons_from_load": {},
        "start_date": start_date,
        "end_date": end_date,
        "interval": "D",
        "benchmark_symbols_yf": ["^GSPC"],
        "display_currency": def_curr,
        "account_currency_map": acc_map,
        "default_currency": def_curr,
        "include_accounts": None,
        "manual_overrides_dict": {},
        "user_symbol_map": {},
        "user_excluded_symbols": set(),
        # "market_data_provider": ... passed to worker init, not kwargs usually, but let's check worker init again
    }

    print("Initializing Worker...")
    worker = PortfolioCalculatorWorker(
        portfolio_fn=calculate_portfolio_summary,
        portfolio_args=(),
        portfolio_kwargs=portfolio_kwargs,
        historical_fn=calculate_historical_performance,
        historical_args=(),
        historical_kwargs=historical_kwargs,
        worker_signals=MockWorkerSignals(),
        manual_overrides_dict={},
        user_symbol_map={},
        user_excluded_symbols=set(),
        market_data_provider=MarketDataProvider(),
        force_historical_refresh=True, # Force full refresh to profile worst case
        historical_fn_supports_exclude=True
    )
    
    print("Starting Profiling execution...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Run the worker synchronously
    worker.run()
    
    profiler.disable()
    print("Profiling Complete.")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    with open("profile_results.txt", "w") as f:
        stats.stream = f
        stats.print_stats(100)
    stats.dump_stats("refresh_profile.prof")

if __name__ == "__main__":
    profile_refresh()
