# -*- coding: utf-8 -*-
"""
Tests comparing the historical calculation methods (Python vs Numba).
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
import time
import sys
import os

# --- Add project root to sys.path ---
# This assumes the 'tests' directory is directly inside the 'Investa' project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Addition ---

try:
    from portfolio_logic import (
        _calculate_portfolio_value_at_date_unadjusted_python,
        _calculate_portfolio_value_at_date_unadjusted_numba,
        CASH_SYMBOL_CSV,
        SHORTABLE_SYMBOLS,  # Keep if needed by logic
        _prepare_historical_inputs,  # To get symbols, currencies, splits, etc.
        _unadjust_prices,  # To derive unadjusted prices
    )
    from data_loader import load_and_clean_transactions  # To load real data
    from market_data import MarketDataProvider  # To fetch real data
    from finutils import (
        get_historical_price,
        get_historical_rate_via_usd_bridge,
        map_to_yf_symbol,  # Keep if needed, though _prepare might handle it
    )

    # Import config constants if needed, or define mocks
    # from config import ...
except ImportError as e:
    pytest.fail(f"Failed to import necessary modules: {e}")


# --- Helper to generate mappings (can be used directly in test) ---
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
    account_to_id = {account: i for i, account in enumerate(all_accounts)}
    id_to_account = {i: account for account, i in account_to_id.items()}
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


# --- Define path to sample CSV for this test file ---
TEST_DIR_HISTORICAL = os.path.dirname(os.path.abspath(__file__))
SAMPLE_CSV_PATH_HISTORICAL = os.path.join(
    TEST_DIR_HISTORICAL, "sample_transactions.csv"
)


# --- Test Function ---


# --- Helper to generate deterministic mock data ---
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
        print(f"DEBUG: Generated mock data for {symbol}. Base: {base_price}, Price[0]: {prices[0]}, Price[58] (Feb 28): {prices[58] if len(prices) > 58 else 'N/A'}")
        
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
        hist_fx[pair] = df
        
    return hist_prices_adj, hist_fx

# --- Test Function ---

def test_compare_python_numba_value_mock_data():
    """
    Compares the portfolio value calculated by the Python and Numba versions
    using sample transaction data and DETERMINISTIC MOCK historical data.
    """
    # --- Configuration ---
    csv_file = SAMPLE_CSV_PATH_HISTORICAL
    start_date = date(2023, 1, 1)
    end_date = date(2024, 1, 1)
    target_currency = "USD"
    account_currency_map_load = {"IBKR": "USD", "SET": "THB"}
    default_currency_load = "USD"
    benchmarks = ["SPY"]

    # --- 1. Prepare Inputs using portfolio_logic helper ---
    (
        loaded_tx_df,
        loaded_orig_df,
        loaded_ignored_indices,
        loaded_ignored_reasons,
        _,
        _,
        _,
    ) = load_and_clean_transactions(
        csv_file, account_currency_map_load, default_currency_load
    )
    assert loaded_tx_df is not None, "Failed to load transactions"

    (
        transactions_df_effective,
        original_transactions_df,
        ignored_indices,
        ignored_reasons,
        all_available_accounts_list,
        included_accounts_list_sorted,
        excluded_accounts_list_sorted,
        symbols_for_stocks_and_benchmarks_yf,
        fx_pairs_for_api_yf,
        internal_to_yf_map,
        yf_to_internal_map_hist,
        splits_by_internal_symbol,
        raw_data_cache_file,
        raw_data_cache_key,
        daily_results_cache_file,
        daily_results_cache_key,
        filter_desc,
    ) = _prepare_historical_inputs(
        preloaded_transactions_df=loaded_tx_df,
        original_transactions_df_for_ignored=loaded_orig_df,
        ignored_indices_from_load=loaded_ignored_indices,
        ignored_reasons_from_load=loaded_ignored_reasons,
        account_currency_map=account_currency_map_load,
        default_currency=default_currency_load,
        include_accounts=None,
        exclude_accounts=None,
        start_date=start_date,
        end_date=end_date,
        benchmark_symbols_yf=benchmarks,
        display_currency=target_currency,
        original_csv_file_path=csv_file,
    )

    assert transactions_df_effective is not None

    # --- Get actual account map and default currency ---
    first_account = transactions_df_effective["Account"].iloc[0]
    account_currency_map = {
        first_account: transactions_df_effective["Local Currency"].iloc[0]
    }
    # Populate full map from data
    for _, row in transactions_df_effective.iterrows():
        account_currency_map[row["Account"]] = row["Local Currency"]
        
    default_currency = transactions_df_effective["Local Currency"].iloc[0]

    # --- 2. Generate Mock Historical Data ---
    hist_prices_adj, hist_fx = generate_mock_market_data(
        start_date, end_date, symbols_for_stocks_and_benchmarks_yf, fx_pairs_for_api_yf
    )

    # --- 3. Derive Unadjusted Prices ---
    processed_warnings = set()
    hist_prices_unadj = _unadjust_prices(
        adjusted_prices_yf=hist_prices_adj,
        yf_to_internal_map=yf_to_internal_map_hist,
        splits_by_internal_symbol=splits_by_internal_symbol,
        processed_warnings=processed_warnings,
    )

    # --- 4. Generate Mappings ---
    (
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        type_to_id,
        currency_to_id,
        id_to_currency,
    ) = generate_mappings(transactions_df_effective)

    # --- 5. Generate Test Dates (Month Ends) ---
    test_dates = pd.date_range(start_date, end_date, freq="ME").date

    # --- 6. Perform Calculations and Compare ---
    total_time_py = 0.0
    total_time_nb = 0.0
    dates_tested = 0

    print(f"\nComparing Python vs Numba for {len(test_dates)} month-end dates...")

    for target_date in test_dates:
        # --- Python Calculation ---
        start_py = time.time()
        value_py, failed_py = _calculate_portfolio_value_at_date_unadjusted_python(
            target_date=target_date,
            transactions_df=transactions_df_effective,
            historical_prices_yf_unadjusted=hist_prices_unadj,
            historical_fx_yf=hist_fx,
            target_currency=target_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            manual_overrides_dict=None,
            processed_warnings=processed_warnings.copy(),
        )
        time_py = time.time() - start_py
        total_time_py += time_py

        # --- Numba Calculation ---
        start_nb = time.time()
        value_nb, failed_nb = _calculate_portfolio_value_at_date_unadjusted_numba(
            target_date=target_date,
            transactions_df=transactions_df_effective,
            historical_prices_yf_unadjusted=hist_prices_unadj,
            historical_fx_yf=hist_fx,
            target_currency=target_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            manual_overrides_dict=None,
            processed_warnings=processed_warnings.copy(),
            symbol_to_id=symbol_to_id,
            id_to_symbol=id_to_symbol,
            account_to_id=account_to_id,
            id_to_account=id_to_account,
            type_to_id=type_to_id,
            currency_to_id=currency_to_id,
            id_to_currency=id_to_currency,
        )
        time_nb = time.time() - start_nb
        total_time_nb += time_nb

        # --- Assertions for this date ---
        assert not failed_py, f"Python calculation reported failure for {target_date}"
        assert not failed_nb, f"Numba calculation reported failure for {target_date}"
        
        # Use a slightly looser tolerance for floating point differences in complex calcs
        assert np.isclose(
            value_py, value_nb, atol=1e-4, equal_nan=True
        ), f"Values differ for {target_date}: Python={value_py}, Numba={value_nb}, Diff={value_py-value_nb}"
        
        dates_tested += 1

    # --- 7. Print Final Timing Comparison ---
    print(f"\n--- Timing Comparison ({dates_tested} dates) ---")
    print(f"Total Python Time: {total_time_py:.4f} seconds")
    print(f"Total Numba Time : {total_time_nb:.4f} seconds")
    if total_time_py > 0:
        speedup = total_time_py / total_time_nb if total_time_nb > 0 else float("inf")
        print(f"Numba Speedup    : {speedup:.2f}x")
    assert dates_tested > 0, "No dates were successfully tested."
