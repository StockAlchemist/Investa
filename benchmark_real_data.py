import sys
import os
import time
import logging
import pandas as pd
import numpy as np
from datetime import date, datetime
import shutil
from PySide6.QtCore import QCoreApplication, QStandardPaths

# Set up paths
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
sys.path.insert(0, project_root)

# Set Org/App name for correct cache paths
app = QCoreApplication([])
app.setOrganizationName("StockAlchemist")
app.setApplicationName("Investa")

import config
from market_data import MarketDataProvider
from db_utils import load_all_transactions_from_db, get_database_path, DB_FILENAME
from data_loader import load_and_clean_transactions
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance, DAILY_RESULTS_CACHE_PATH_PREFIX

# Configure Logging
logging.basicConfig(level=logging.WARNING)

def get_cache_dir():
    base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    return os.path.join(base, "StockAlchemist", "Investa")

def clear_daily_cache():
    cache_dir = get_cache_dir()
    if not os.path.exists(cache_dir):
        return
    deleted = 0
    for f in os.listdir(cache_dir):
        if f.startswith(DAILY_RESULTS_CACHE_PATH_PREFIX) and f.endswith(".feather"):
            try:
                os.remove(os.path.join(cache_dir, f))
                deleted += 1
            except Exception as e:
                print(f"Error deleting {f}: {e}")
    print(f"Cleared {deleted} daily cache files.")

def run_benchmark():
    print("--- Setting up Benchmark ---")
    
    # 1. Load Data
    # Priority: CWD > Config (not loaded here) > Default AppData
    possible_dbs = ["my_transactions.db", DB_FILENAME]
    db_path = None
    for fname in possible_dbs:
        cwd_path = os.path.join(os.getcwd(), fname)
        if os.path.exists(cwd_path):
            db_path = cwd_path
            break
            
    if not db_path:
         db_path = get_database_path(DB_FILENAME) # Fallback to appdata location
        
    if not os.path.exists(db_path):
        print(f"Error: DB not found. Checked {possible_dbs} in CWD and default path {get_database_path(DB_FILENAME)}")
        return

    print(f"Loading transactions from {db_path}...")
    print(f"Loading transactions from {db_path}...")
    from db_utils import get_db_connection
    conn = get_db_connection(db_path)
    if not conn:
        print("Failed to connect to DB.")
        return

    # Default settings matching typical usage
    account_currency_map = {"SET": "THB"} 
    default_currency = "USD"

    transactions, success = load_all_transactions_from_db(conn, account_currency_map, default_currency)
    conn.close()
    
    if not success or transactions is None or transactions.empty:
        print("No transactions found or failed to load.")
        return
        
    cleaned_tx_df = transactions
    
    # Minimal Manual Cleaning (mimic data_loader)
    cleaned_tx_df["Date"] = pd.to_datetime(cleaned_tx_df["Date"])
    text_cols = ["Type", "Symbol", "Account", "Note", "Local Currency"]
    for col in text_cols:
        if col in cleaned_tx_df.columns:
             cleaned_tx_df[col] = cleaned_tx_df[col].astype(str).str.strip().replace("nan", "", regex=False).replace("None", "", regex=False)
             
    cleaned_tx_df["Local Currency"] = cleaned_tx_df["Local Currency"].str.upper()
    
    print(f"Loaded {len(cleaned_tx_df)} transactions.")
    
    # Filter out known bad symbols causing fetch timeouts
    bad_symbols = [
        "SCBRMS&P500", "NOK.BK", "UOBBC", "ES-FIXED_INCOME", "BECL.BK", 
        "EMV", "ES-SET50", "SCBRM1", "BML.BK", "ES-JUMBO25", "IDMOX", 
        "UOBCG", "BRK.B", "DSV", "ES-TRESURY", "IDBOX", "ES-GQG", 
        "SCBSFF", "SCBCHA-SSF", "IDIOX", "IDLOX", "AAUKY", "RIMM", "KRFT", "SCBRCTECH"
    ]
    # Also strip spaces just in case
    cleaned_tx_df = cleaned_tx_df[~cleaned_tx_df["Symbol"].str.upper().isin(bad_symbols)].copy()
    print(f"DEBUG: NOK.BK in cleaned? {'NOK.BK' in cleaned_tx_df['Symbol'].str.upper().values}")
    print(f"After filtering bad symbols: {len(cleaned_tx_df)} transactions.")
    
    # 2. Initialize Market Data Provider & Warmup
    mdp = MarketDataProvider()
    
    # We need to trigger a fetch to ensure 'prices' and 'fx' are in cache (or memory if we pass mdp)
    # calculate_portfolio_summary uses mdp internally if passed? 
    # Actually calculate_portfolio_summary takes (transactions_df, market_data_provider, ...)
    # It calls _load_or_fetch_raw_historical_data internally using mdp.
    
    # To treat this as "Calculation Benchmark", we want the fetch inside calculate_portfolio_summary
    # to be fast (cached). So we run it once to ensure cache is populated.
    
    print("--- Warming up Market Data (Fetching/Caching) ---")
    # Force use of 'numba_chrono' for warmup to ensure it runs
    config.HISTORICAL_CALC_METHOD = "numba_chrono" 
    import portfolio_logic
    portfolio_logic.HISTORICAL_CALC_METHOD = "numba_chrono"
    
    # We define a standard date range for benchmark
    # Default behavior of app is usually YTD or Max. calculate_portfolio_summary determines dates.
    # We will let it determine dates.
    
    # Dummy values for required positional args
    original_tx_df = cleaned_tx_df.copy()
    ignored_indices = set()
    ignored_reasons = {}

    try:
        # Use tmp cache to force calc (and thus compilation) without clearing real cache
        tmp_cache = "/tmp/investa_warmup_cache.feather"
        if os.path.exists(tmp_cache):
            os.remove(tmp_cache)
            
        calculate_portfolio_summary(
            cleaned_tx_df,
            original_tx_df,
            ignored_indices,
            ignored_reasons,
            market_provider=mdp,
            cache_file_path=tmp_cache,
            calc_method="numba_chrono"
        )
    except Exception as e:
        print(f"Warmup failed (might be expected if data missing): {e}")
        # traceback.print_exc()

    print("\n--- Starting Benchmark (Calculation Only) ---")
    
    methods = ["python", "numba_chrono"]
    
    results = {}
    
    # Prepare arguments for calculate_historical_performance
    start_date = cleaned_tx_df["Date"].min().date()
    # end_date = datetime.now().date()
    end_date = date(2025, 12, 5) # Fix to Friday to avoid weekend issues
    
    for method in methods:
        print(f"\nTesting Method: {method.upper()}")
        
        # Monkeypatch config AND portfolio_logic global
        config.HISTORICAL_CALC_METHOD = method
        import portfolio_logic
        print(f"DEBUG: portfolio_logic file: {portfolio_logic.__file__}")
        portfolio_logic.HISTORICAL_CALC_METHOD = method
        
        # Clear Calculation Cache (but keep Market Data Cache)
        clear_daily_cache()
        
        start_time = time.time()
        try:
            # Call calculate_historical_performance instead of calculate_portfolio_summary
            daily_df, _, _, status = portfolio_logic.calculate_historical_performance(
                all_transactions_df_cleaned=cleaned_tx_df,
                original_transactions_df_for_ignored=original_tx_df,
                ignored_indices_from_load=ignored_indices,
                ignored_reasons_from_load=ignored_reasons,
                start_date=start_date,
                end_date=end_date,
                interval="D",
                benchmark_symbols_yf=["SPY"],
                display_currency="USD",
                account_currency_map=account_currency_map,
                default_currency=default_currency,
                use_raw_data_cache=True,
                use_daily_results_cache=True, # Will trigger calc because we cleared cache
                calc_method=method,
            )
            duration = time.time() - start_time
            
            # Check if result is empty
            if daily_df is None or daily_df.empty:
               print(f"WARNING: {method} returned empty result! Status: {status}")
               # duration = 9999.0 # Don't penalize, just warn
            
            print(f"Time Taken: {duration:.4f} seconds")
            results[method] = duration
            
        except Exception as e:
             print(f"Method {method} FAILED: {e}")
             import traceback
             traceback.print_exc()
             results[method] = None

    print("\n--- Results Summary ---")
    for method, duration in results.items():
        if duration is not None:
            print(f"{method.upper()}: {duration:.4f}s")
        else:
            print(f"{method.upper()}: FAILED")
            
    if 'numba_chrono' in results and 'python' in results:
        t_py = results['python']
        t_nb = results['numba_chrono']
        if t_nb is not None and t_py is not None: # Ensure both are not None before comparison
            if t_nb < t_py:
                 print(f"\nOptimized (Numba) is {t_py/t_nb:.2f}x FASTER.")
            else:
                 print(f"\nOptimized (Numba) is {t_nb/t_py:.2f}x SLOWER.")

if __name__ == "__main__":
    run_benchmark()
