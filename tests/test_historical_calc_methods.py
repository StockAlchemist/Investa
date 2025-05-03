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
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    type_to_id = {tx_type: i for i, tx_type in enumerate(all_types)}
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


# --- Test Function ---


def test_compare_python_numba_value_real_data():
    """
    Compares the portfolio value calculated by the Python and Numba versions
    using real transaction data and fetched historical data for month-end dates
    between 2010-01-01 and 2025-05-01. Also compares total execution time.
    """
    # --- Configuration ---
    csv_file = "/Users/kmatan/Library/CloudStorage/OneDrive-MahidolUniversity/finance/Stocks/Evaluations/python/Investa/my_transactions.csv"
    start_date = date(2010, 1, 1)  # Use date() directly
    end_date = date(2025, 5, 1)  # Use date() directly
    target_currency = "USD"
    # Use dummy maps/defaults for loading, _prepare_historical_inputs will get real ones
    account_currency_map_load = {"SET": "THB"}
    default_currency_load = "USD"
    # Benchmarks aren't strictly needed for value calc, but _prepare needs it
    benchmarks = ["SPY"]

    # --- 1. Prepare Inputs using portfolio_logic helper ---
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
        transactions_csv_file=csv_file,
        account_currency_map=account_currency_map_load,  # Use dummy for now
        default_currency=default_currency_load,  # Use dummy for now
        include_accounts=None,  # Test all accounts
        exclude_accounts=None,
        start_date=start_date,
        end_date=end_date,
        benchmark_symbols_yf=benchmarks,
        display_currency=target_currency,
    )

    assert transactions_df_effective is not None, "Failed to load/prepare transactions"
    assert (
        not transactions_df_effective.empty
    ), "Transaction DataFrame is empty after preparation"

    # --- Get actual account map and default currency from loaded data ---
    # This assumes the first transaction's currency or a config default is sufficient
    # A more robust way might involve scanning all accounts if needed.
    # For now, let's use the first account's currency mapping if available.
    first_account = transactions_df_effective["Account"].iloc[0]
    account_currency_map = {
        first_account: transactions_df_effective["Local Currency"].iloc[0]
    }
    default_currency = transactions_df_effective["Local Currency"].iloc[
        0
    ]  # Or use a fixed default like 'USD'

    # --- 2. Fetch Real Historical Data ---
    market_provider = MarketDataProvider()
    hist_prices_adj, fetch_failed_prices = market_provider.get_historical_data(
        symbols_yf=symbols_for_stocks_and_benchmarks_yf,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,  # Use cache for speed
        cache_file=raw_data_cache_file,
        cache_key=raw_data_cache_key,
    )
    hist_fx, fetch_failed_fx = market_provider.get_historical_fx_rates(
        fx_pairs_yf=fx_pairs_for_api_yf,
        start_date=start_date,
        end_date=end_date,
        use_cache=True,  # Use cache for speed
        cache_file=raw_data_cache_file,
        cache_key=raw_data_cache_key,
    )
    assert not fetch_failed_prices, "Failed to fetch historical prices"
    assert not fetch_failed_fx, "Failed to fetch historical FX rates"

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
    test_dates = pd.date_range(
        start_date, end_date, freq="ME"
    ).date  # Use 'ME' for Month End

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
        assert np.isclose(
            value_py, value_nb, atol=1e-6, equal_nan=True  # Allow NaN comparison
        ), f"Values differ significantly for {target_date}: Python={value_py}, Numba={value_nb}"
        dates_tested += 1

    # --- 7. Print Final Timing Comparison ---
    print(f"\n--- Timing Comparison ({dates_tested} dates) ---")
    print(f"Total Python Time: {total_time_py:.4f} seconds")
    print(f"Total Numba Time : {total_time_nb:.4f} seconds")
    if total_time_py > 0:
        speedup = total_time_py / total_time_nb if total_time_nb > 0 else float("inf")
        print(f"Numba Speedup    : {speedup:.2f}x")
    assert dates_tested > 0, "No dates were successfully tested."
