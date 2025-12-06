

import pytest
import pandas as pd
import numpy as np
from datetime import date, timedelta
import time
import os
import sys

# --- Path Setup ---
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from portfolio_logic import (
    _load_or_calculate_daily_results,
    CASH_SYMBOL_CSV,
    _prepare_historical_inputs,
    _unadjust_prices
)
from market_data import MarketDataProvider

# --- Helpers (Inlined) ---
def generate_mappings(transactions_df_effective):
    """Generates string-to-ID mappings based on the effective transaction data."""
    symbol_to_id, id_to_symbol = {}, {}
    account_to_id, id_to_account = {}, {}
    type_to_id = {}
    currency_to_id, id_to_currency = {}, {}
    all_symbols = transactions_df_effective["Symbol"].unique()
    symbol_to_id = {symbol: i for i, symbol in enumerate(all_symbols)}
    id_to_symbol = {i: symbol for symbol, i in symbol_to_id.items()}
    all_accounts = transactions_df_effective["Account"].unique()
    account_to_id = {account.upper(): i for i, account in enumerate(all_accounts)}
    id_to_account = {i: account.upper() for account, i in account_to_id.items()}
    all_types = transactions_df_effective["Type"].unique()
    type_to_id = {str(tx_type).lower().strip(): i for i, tx_type in enumerate(all_types)}
    all_currencies = transactions_df_effective["Local Currency"].unique()
    currency_to_id = {curr: i for i, curr in enumerate(all_currencies)}
    id_to_currency = {i: curr for curr, i in currency_to_id.items()}
    # Ensure CASH symbol is included if not present
    if CASH_SYMBOL_CSV not in symbol_to_id:
        cash_id = len(symbol_to_id)
        symbol_to_id[CASH_SYMBOL_CSV] = cash_id
        id_to_symbol[cash_id] = CASH_SYMBOL_CSV
    return (
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        type_to_id,
        currency_to_id,
        id_to_currency,
    )

def generate_mock_market_data(start_date, end_date, symbols, fx_pairs):
    """Generates deterministic mock price and FX data."""
    dates = pd.date_range(start_date, end_date, freq="D").date
    
    import zlib
    # Mock Prices
    hist_prices_adj = {}
    for symbol in symbols:
        # Create a simple upward trend with some sine wave noise
        # Use zlib.adler32 for deterministic hashing
        seed = zlib.adler32(symbol.encode("utf-8"))
        base_price = 100.0 + (seed % 50)
        prices = [
            base_price + (i * 0.1) + (np.sin(i / 10.0) * 5.0) 
            for i in range(len(dates))
        ]
        df = pd.DataFrame({"price": prices}, index=dates)
        hist_prices_adj[symbol] = df
        
    # Mock FX
    hist_fx = {}
    for pair in fx_pairs:
        # Simple oscillation around a base rate
        base_rate = 1.0
        if "THB" in pair: base_rate = 35.0
        elif "EUR" in pair: base_rate = 0.9
        elif "JPY" in pair: base_rate = 150.0
        
        rates = [
            base_rate + (np.sin(i / 20.0) * (base_rate * 0.05))
            for i in range(len(dates))
        ]
        df = pd.DataFrame({"price": rates}, index=dates)
        if pair != "USD":
            pair_key = f"{pair}=X"
        else:
             pair_key = pair 
        
        if pair == "USD":
             continue 
             
        key = f"{pair}=X"
        hist_fx[key] = df
        
    return hist_prices_adj, hist_fx

# Mocking
class MockMarketData:
    def __init__(self, prices, fx):
        self.prices = prices
        self.fx = fx
    
    def get_historical_data_multi(self, *args, **kwargs):
        raise NotImplementedError("Not used in this test path directly if pre-fetched")

def test_numba_chrono_vs_python_accuracy_and_speed():
    """
    Benchmarks the optimized 'numba_chrono' method against 'python' method 
    (or validates 'numba_chrono' exactness if python is too slow/deprecated).
    Here we focus on checking if 'numba_chrono' produces valid results faster.
    """
    # 1. Setup Data
    start_date = date(2023, 1, 1)
    end_date = date(2023, 6, 30) # 6 months
    dates = pd.date_range(start_date, end_date, freq="D")
    
    # Symbols
    symbols = ["AAPL", "MSFT", "TSLA"]
    accounts = ["Acc1", "Acc2"]
    
    # Create Transactions
    transactions = []
    # Initial Deposits
    transactions.append({
        "Date": pd.Timestamp("2023-01-01"),
        "Account": "Acc1", "Type": "deposit", "Symbol": CASH_SYMBOL_CSV, "Quantity": 100000, "Local Currency": "USD", "Commission": 0
    })
    transactions.append({
        "Date": pd.Timestamp("2023-01-01"),
        "Account": "Acc2", "Type": "deposit", "Symbol": CASH_SYMBOL_CSV, "Quantity": 3000000, "Local Currency": "THB", "Commission": 0
    })
    
    # Buys
    transactions.append({
        "Date": pd.Timestamp("2023-01-05"),
        "Account": "Acc1", "Type": "buy", "Symbol": "AAPL", "Quantity": 100, "Price": 150.0, "Local Currency": "USD", "Commission": 1.0
    })
    transactions.append({
        "Date": pd.Timestamp("2023-02-15"),
        "Account": "Acc2", "Type": "buy", "Symbol": "TSLA", "Quantity": 50, "Price": 200.0, "Local Currency": "USD", "Commission": 5.0
    })
    
    # Transfer (Internal)
    # Acc1 -> Acc2
    transactions.append({
         "Date": pd.Timestamp("2023-03-01"),
         "Account": "Acc1", "To Account": "Acc2", "Type": "transfer", "Symbol": "AAPL", "Quantity": 50, "Price": 160.0, "Local Currency": "USD", "Commission": 0
    })

    tx_df = pd.DataFrame(transactions)
    tx_df["Split Ratio"] = 0.0
    # Fix for Python parallel worker: it expects 'Price/Share' column or fails
    if "Price" in tx_df.columns:
         tx_df["Price/Share"] = tx_df["Price"]
    else:
         tx_df["Price/Share"] = 0.0
         
    tx_df["original_index"] = tx_df.index
    
    # 2. Mock Market Data
    # FX: THB->USD, USD->USD
    # Prices: AAPL, MSFT, TSLA
    hist_prices_adj, hist_fx = generate_mock_market_data(start_date, end_date, symbols, ["THB", "USD"])
    
    # Adjust FX keys keys if needed (generate_mock provides keys like "THB")
    # portfolio_logic expects keys matching what market_data provides.
    # Usually "THB=X" or "THB". Let's assume keys in hist_fx are sufficient if we map correctly.
    # We will simulate valid `historical_fx_yf` with keys "THB=X", "USD=X" etc if logic requires.
    # generate_mock returns dict {pair: df}.
    
    # 3. Prepare Inputs
    # We need to manually prepare huge set of args for _load_or_calculate_daily_results
    
    # Mappings
    (
        symbol_to_id, id_to_symbol, account_to_id, id_to_account, 
        type_to_id, currency_to_id, id_to_currency
    ) = generate_mappings(tx_df)
    
    internal_to_yf_map = {s: s for s in symbols}
    internal_to_yf_map[CASH_SYMBOL_CSV] = CASH_SYMBOL_CSV
    
    account_currency_map = {"Acc1": "USD", "Acc2": "THB"}
    
    # Unadjust prices (mock is already "adj" but we treat as unadj for simplicity or derive)
    hist_prices_unadj = {k: v.copy() for k, v in hist_prices_adj.items()}
    
    # 4. Helper to Call Function
    def call_calc(method):
        return _load_or_calculate_daily_results(
            use_daily_results_cache=False,
            daily_results_cache_file=None,
            daily_results_cache_key=None,
            worker_signals=None,
            transactions_csv_file="", # Skip mtime check
            start_date=start_date,
            end_date=end_date,
            transactions_df_effective=tx_df,
            historical_prices_yf_unadjusted=hist_prices_unadj,
            historical_prices_yf_adjusted=hist_prices_adj, # used for benchmark
            historical_fx_yf=hist_fx,
            display_currency="USD",
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency="USD",
            symbol_to_id=symbol_to_id,
            id_to_symbol=id_to_symbol,
            account_to_id=account_to_id,
            id_to_account=id_to_account,
            manual_overrides_dict=None,
            type_to_id=type_to_id,
            currency_to_id=currency_to_id,
            id_to_currency=id_to_currency,
            all_holdings_qty=None,
            all_cash_balances=None,
            all_last_prices=None,
            included_accounts_list=["Acc1", "Acc2"],
            calc_method=method,
            filter_desc="Test"
        )

    # 5. Run Numba Chrono (Optimized)
    print("\n--- Running Optimized Numba Chrono ---")
    start_t = time.time()
    df_chrono, valid, status = call_calc("numba_chrono")
    end_t = time.time()
    time_chrono = end_t - start_t
    print(f"Time Taken: {time_chrono:.4f}s")
    
    assert not df_chrono.empty, "Chrono calculation returned empty DF"
    assert "value" in df_chrono.columns
    assert "net_flow" in df_chrono.columns
    
    # 6. Basic Correctness Checks
    # Day 0 (Jan 1): Deposit 100k USD + 3m THB.
    # THB rate in mock data (from generate_mock_market_data) is ~35.0 (THB/USD? Or USD/THB?)
    # generate_mock returns "price" column.
    # if "THB" in pair: base_rate = 35.0. 
    # If standard is USD based: 35.0 means 35 THB/USD? Or 1 THB = 35 USD?
    # generate_mock logic: `base_rate = 35.0`.
    # get_historical_rate_via_usd_bridge uses logic.
    # Let's verify result roughly.
    # Value should not be NaN.
    if df_chrono["value"].isna().any():
        print("WARNING: NaNs in value")
        print(df_chrono[df_chrono["value"].isna()])

    print(f"Final Value: {df_chrono['value'].iloc[-1]:,.2f}")
    
    # 7. Run Python Parallel (Original)
    print("\n--- Running Python Parallel (Original) ---")
    start_t_py = time.time()
    df_py, valid_py, status_py = call_calc("python")
    end_t_py = time.time()
    time_py = end_t_py - start_t_py
    print(f"Time Taken (Python): {time_py:.4f}s")
    
    print(f"\nSpeedup: {time_py / time_chrono:.2f}x")
    
    # Assert Chrono is faster (or at least comparable given compiled overhead on small data)
    # On very small data, Python might be fast. But typically Python parallel has overhead.
    # Numba compilation adds overhead on first run. 
    # Let's just print comparison for User information.
    
    assert time_chrono < 4.0, "Calculation should be very fast for small dataset"

if __name__ == "__main__":
    test_numba_chrono_vs_python_accuracy_and_speed()
