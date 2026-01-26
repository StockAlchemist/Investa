# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_logic.py
 Purpose:       Core logic for portfolio calculations, data fetching, and analysis.
                Handles transaction processing, current summary, and historical performance.
                Uses MarketDataProvider for external market data.


 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
--------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

# --- START OF MODIFIED portfolio_logic.py ---
import sys
import os
import config # Added config import
import math # Added math import
from datetime import datetime, date, timedelta
import pytz
from utils_time import get_est_today, get_latest_trading_date
import json
from typing import List, Dict, Tuple, Optional, Set, Any, Union


import pandas as pd
import numpy as np
import logging
import traceback
from collections import defaultdict  # Still needed for historical parts
import time
from io import StringIO
import multiprocessing
from functools import partial
import calendar
import hashlib
import logging
import numba  # <-- ADD

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    # This is primarily for type hinting and making the code runnable without kernprof
    # The actual profiling happens when run via kernprof
    # If line_profiler is not installed, the @profile decorator will be a no-op
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


def _normalize_series(series):
    """Normalizes a pandas Series to uppercase and trimmed strings."""
    if series.empty:
        return series
    return series.astype(str).str.upper().str.strip()


# --- END ADDED ---

# --- Configure Logging ---
# Assumes logging is configured elsewhere (e.g., main_gui.py or config.py)
# If running standalone, uncomment and configure basicConfig here.
# logging.basicConfig(level=logging.DEBUG, format="...")

# --- Import Configuration and Utilities ---
try:
    from config import (
        LOGGING_LEVEL,
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENT_CACHE_FILE_PATH,  # Still used by MarketDataProvider default
        HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        DAILY_RESULTS_CACHE_PATH_PREFIX,
        YFINANCE_CACHE_DURATION_HOURS,
        YFINANCE_INDEX_TICKER_MAP,
        YFINANCE_EXCLUDED_SYMBOLS,
        SHORTABLE_SYMBOLS,
        DEFAULT_CURRENCY,
        HISTORICAL_DEBUG_USD_CONVERSION,
        HISTORICAL_DEBUG_SET_VALUE,
        STOCK_QUANTITY_CLOSE_TOLERANCE,
        DEBUG_DATE_VALUE,
        HISTORICAL_DEBUG_DATE_VALUE,
        HISTORICAL_DEBUG_SYMBOL,
        HISTORICAL_CALC_METHOD,
        HISTORICAL_COMPARE_METHODS,
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from config.py. Exiting.")
    raise

try:
    from finutils import (
        _get_file_hash,
        calculate_npv,
        calculate_irr,
        get_cash_flows_for_symbol_account,
        get_cash_flows_for_mwr,
        get_conversion_rate,
        get_historical_price,
        get_historical_rate_via_usd_bridge,
        map_to_yf_symbol,
        is_cash_symbol,  # ADDED
        safe_sum,
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from finutils.py. Exiting.")
    raise

try:
    from data_loader import (
        load_and_clean_transactions,
    )  # Still needed for __main__ and historical prep fallback
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from data_loader.py. Exiting.")
    raise

# --- Import QStandardPaths for cache directory ---
try:
    from PySide6.QtCore import QStandardPaths
except ImportError:
    logging.warning(
        "PySide6.QtCore.QStandardPaths not found. Cache paths might be relative."
    )
    QStandardPaths = None  # Fallback

# --- Import the NEW Market Data Provider ---
try:
    from market_data import MarketDataProvider, get_shared_mdp

    MARKET_PROVIDER_AVAILABLE = True
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import MarketDataProvider from market_data.py. Exiting."
    )
    MARKET_PROVIDER_AVAILABLE = False
    raise

# --- Import the NEW Portfolio Analyzer Functions ---
try:
    from portfolio_analyzer import (
        _process_transactions_to_holdings,
        _build_summary_rows,
        _calculate_aggregate_metrics,
        calculate_periodic_returns,
        calculate_correlation_matrix,
        extract_realized_capital_gains_history,
        calculate_fifo_lots_and_gains,  # NEW IMPORT
        extract_dividend_history, # NEW IMPORT
    )

    ANALYZER_FUNCTIONS_AVAILABLE = True
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import helper functions from portfolio_analyzer.py. Exiting."
    )
    ANALYZER_FUNCTIONS_AVAILABLE = False
    raise


# --- Helper Functions (IRR/NPV, Cash Flow Gen - Already in finutils.py) ---
# --- Keep functions specific to processing/calculation logic HERE ---
# --- NOTE: The following functions have been MOVED to portfolio_analyzer.py ---
# _process_transactions_to_holdings
# _calculate_cash_balances
# _build_summary_rows
# _calculate_aggregate_metrics


# --- Main Calculation Function (Current Portfolio Summary) ---
# @profile
def calculate_portfolio_summary(
    all_transactions_df_cleaned: Optional[pd.DataFrame],  # MODIFIED: Accept DataFrame
    original_transactions_df_for_ignored: Optional[
        pd.DataFrame
    ],  # MODIFIED: For ignored rows context
    ignored_indices_from_load: Set[int],  # MODIFIED: Pass from initial load
    ignored_reasons_from_load: Dict[int, str],  # MODIFIED: Pass from initial load
    fmp_api_key: Optional[str] = None,
    display_currency: str = "USD",
    show_closed_positions: bool = False,
    # account_currency_map: Dict = {"SET": "THB"}, # These will be passed via portfolio_kwargs if needed by sub-functions
    # default_currency: str = "USD", # These will be passed via portfolio_kwargs if needed by sub-functions
    cache_file_path: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    include_accounts: Optional[List[str]] = None,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    user_symbol_map: Optional[Dict[str, str]] = None,
    user_excluded_symbols: Optional[Set[str]] = None,
    # --- ADDED default_currency and account_currency_map as explicit args again ---
    # because _process_transactions_to_holdings and _calculate_cash_balances need them directly.
    # They are not easily passable via a generic **kwargs if those helpers are called directly.
    default_currency: str = DEFAULT_CURRENCY,
    account_currency_map: Optional[Dict[str, str]] = None,
    market_provider: Optional[Any] = None,  # Added for dependency injection
    calc_method: Optional[str] = None, # Added for benchmarking override
    account_interest_rates: Optional[Dict[str, float]] = None,
    interest_free_thresholds: Optional[Dict[str, float]] = None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[Dict[str, Dict[str, Any]]],
    Optional[Dict[str, Dict[str, float]]],
    Set[int],
    Dict[int, str],
    str,
]:
    """
    Calculates the current portfolio summary using MarketDataProvider for market data
    and helper functions from portfolio_analyzer.py for calculations.
    Accepts a pre-loaded and cleaned transactions DataFrame.

    Args:
        all_transactions_df_cleaned (Optional[pd.DataFrame]): Pre-loaded and cleaned DataFrame of all transactions.
        original_transactions_df_for_ignored (Optional[pd.DataFrame]): The raw DataFrame read from CSV,
            used for providing context to ignored rows if `all_transactions_df_cleaned` itself
            doesn't contain all original columns or if `original_index` isn't from the raw CSV.
            If `all_transactions_df_cleaned` is comprehensive, this might be the same.
        ignored_indices_from_load (Set[int]): Set of 'original_index' values from rows skipped
                                              during the initial load_and_clean_transactions step.
        ignored_reasons_from_load (Dict[int, str]): Reasons for rows skipped during initial load.
        fmp_api_key (Optional[str], optional): Unused. API key for Financial Modeling Prep. Defaults to None.
        display_currency (str, optional): The currency in which to display all values. Defaults to "USD".
        show_closed_positions (bool, optional): Whether to include closed positions in the summary. Defaults to False.
        cache_file_path (str, optional): Path to the cache file used by MarketDataProvider. Defaults to DEFAULT_CURRENT_CACHE_FILE_PATH.
        include_accounts (Optional[List[str]], optional): A list of account names to include. Defaults to None (all).
        manual_overrides_dict (Optional[Dict[str, Dict[str, Any]]], optional): Manual overrides. Defaults to None.
        user_symbol_map (Optional[Dict[str, str]], optional): User-defined symbol map. Defaults to None.
        user_excluded_symbols (Optional[Set[str]], optional): User-defined excluded symbols. Defaults to None.
        market_provider (MarketDataProvider): An instance of the market data provider.
        market_provider (MarketDataProvider): An instance of the market data provider.
        default_currency (str, optional): Default currency. Defaults to DEFAULT_CURRENCY.
        account_currency_map (Optional[Dict[str, str]], optional): Account to currency map. Defaults to None (uses default).
        account_interest_rates (Optional[Dict[str, float]], optional): Account interest rates. Defaults to None.
        interest_free_thresholds (Optional[Dict[str, float]], optional): Account interest thresholds. Defaults to None.

    Returns:
        Tuple: overall_summary_metrics, summary_df_final, account_level_metrics,
               combined_ignored_indices, combined_ignored_reasons, final_status.
    """
    logging.info(
        f"Starting Portfolio Summary Calculation (Display: {display_currency})"
    )
    start_time_summary = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []

    from utils_time import get_est_today, get_latest_trading_date # Import local helper

    # Use the passed-in ignored data from the initial load
    report_date = get_latest_trading_date()  # Defined early for use in default metrics

    # --- Define default metrics structures ---
    def get_default_metrics_dict(
        display_curr_arg,
        report_date_arg,
        available_accs_list_arg,
        is_empty_data_case=False,
    ):
        metrics = {
            "market_value": 0.0 if is_empty_data_case else np.nan,
            "cost_basis_held": 0.0 if is_empty_data_case else np.nan,
            "unrealized_gain": 0.0 if is_empty_data_case else np.nan,
            "realized_gain": 0.0 if is_empty_data_case else np.nan,
            "dividends": 0.0 if is_empty_data_case else np.nan,
            "commissions": 0.0 if is_empty_data_case else np.nan,
            "total_gain": 0.0 if is_empty_data_case else np.nan,
            "total_cost_invested": 0.0 if is_empty_data_case else np.nan,
            "total_buy_cost": 0.0 if is_empty_data_case else np.nan,
            "portfolio_mwr": np.nan,
            "day_change_display": 0.0 if is_empty_data_case else np.nan,
            "day_change_percent": np.nan,
            "report_date": report_date_arg.strftime("%Y-%m-%d"),
            "display_currency": display_curr_arg,
            "cumulative_investment": 0.0 if is_empty_data_case else np.nan,
            "total_return_pct": np.nan,
            "est_annual_income_display": (
                0.0 if is_empty_data_case else np.nan
            ),  # Key change here
            "fx_gain_loss_display": 0.0 if is_empty_data_case else np.nan,
            "fx_gain_loss_pct": np.nan,
            "_available_accounts": available_accs_list_arg,
        }
        return metrics

    # --- End Define default metrics ---

    combined_ignored_indices = (
        ignored_indices_from_load.copy() if ignored_indices_from_load else set()
    )
    combined_ignored_reasons = (
        ignored_reasons_from_load.copy() if ignored_reasons_from_load else {}
    )

    manual_overrides_effective = (
        manual_overrides_dict if manual_overrides_dict is not None else {}
    )
    effective_user_symbol_map = user_symbol_map if user_symbol_map is not None else {}
    effective_user_excluded_symbols = (
        user_excluded_symbols if user_excluded_symbols is not None else set()
    )
    effective_account_currency_map = (
        account_currency_map if account_currency_map is not None else {}
    )

    if not MARKET_PROVIDER_AVAILABLE or not ANALYZER_FUNCTIONS_AVAILABLE:
        msg = "Error: Critical dependencies (MarketDataProvider or Analyzer Functions) not available."
        logging.error(msg)
        return (
            get_default_metrics_dict(
                display_currency, report_date, [], is_empty_data_case=False
            ),  # MODIFIED
            None,  # summary_df_final
            None,  # account_level_metrics
            combined_ignored_indices,
            combined_ignored_reasons,
            f"Finished with Errors [{msg}]",
        )

    # --- 1. Use Pre-loaded Transactions ---

    if all_transactions_df_cleaned is None or all_transactions_df_cleaned.empty:
        msg = "Error: No transaction data provided or data is empty."
        logging.error(msg)
        status_parts.insert(0, "No Data")
        final_status = f"Finished with Errors [{'; '.join(status_parts)}]"
        # If df_cleaned is None, df_original_raw_for_ignored might also be None or not useful
        # We return the initially passed ignored sets as they are the most relevant at this stage.
        return (
            get_default_metrics_dict(
                display_currency, report_date, [], is_empty_data_case=True
            ),  # MODIFIED
            None,  # summary_df_final
            None,  # account_level_metrics
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )

    # Ensure 'original_index' exists for subsequent processing if it doesn't already
    # This should ideally be handled by load_and_clean_transactions
    if "original_index" not in all_transactions_df_cleaned.columns:
        logging.warning(
            "'original_index' column not found in provided DataFrame. Adding it from DataFrame index."
        )
        all_transactions_df_cleaned["original_index"] = (
            all_transactions_df_cleaned.index
        )
        has_warnings = True
        status_parts.append("original_index missing")

    # Determine available accounts early if all_transactions_df_cleaned is valid
    available_accounts_for_errors = []
    if (
        all_transactions_df_cleaned is not None
        and "Account" in all_transactions_df_cleaned.columns
        and not all_transactions_df_cleaned.empty
    ):
        available_accounts_for_errors = sorted(
            list(all_transactions_df_cleaned["Account"].unique())
        )

    # --- 2. Filter Transactions ---
    transactions_df_filtered = pd.DataFrame()
    filter_desc = "All Accounts"
    if not include_accounts:  # None or empty list means all accounts
        transactions_df_filtered = all_transactions_df_cleaned.copy()
    elif isinstance(include_accounts, list):
        # Ensure 'Account' column exists
        if "Account" not in all_transactions_df_cleaned.columns:
            logging.error(
                "CRITICAL: 'Account' column missing in provided transaction data. Cannot filter."
            )
            status_parts.append("Filter Error: No Account Column")
            # Return the combined ignored indices from the load phase
            return (
                get_default_metrics_dict(
                    display_currency,
                    report_date,
                    available_accounts_for_errors,
                    is_empty_data_case=False,
                ),  # MODIFIED
                None,  # summary_df_final
                None,  # account_level_metrics
                combined_ignored_indices,
                combined_ignored_reasons,
                f"Finished with Errors [{'; '.join(status_parts)}]",
            )

        available_accounts_in_df = set(all_transactions_df_cleaned["Account"].unique())
        # --- MODIFIED: Also consider accounts in the "To Account" column for filtering ---
        if "To Account" in all_transactions_df_cleaned.columns:
            available_accounts_in_df.update(
                all_transactions_df_cleaned["To Account"].dropna().unique()
            )

        valid_include = [
            acc for acc in include_accounts if acc in available_accounts_in_df
        ]
        if valid_include:
            # --- MODIFIED: Robust filtering for transfers. This logic ensures that if a user
            # selects an account that received a transfer, the entire history of the transferred
            # asset from its source account is also included, which is crucial for correct
            # cost basis calculation. ---

            # 1. Get all transactions where the *primary* account matches
            from_account_mask = all_transactions_df_cleaned["Account"].isin(
                valid_include
            )

            # 2. Get all transactions where the *destination* account matches (transfers-in)
            to_account_mask = pd.Series(False, index=all_transactions_df_cleaned.index)
            if "To Account" in all_transactions_df_cleaned.columns:
                to_account_mask = (
                    all_transactions_df_cleaned["To Account"]
                    .isin(valid_include)
                    .fillna(False)
                )

            # 3. Find the preceding transactions for these transfers-in
            preceding_tx_mask = pd.Series(
                False, index=all_transactions_df_cleaned.index
            )
            transfers_in_df = all_transactions_df_cleaned[to_account_mask]

            if not transfers_in_df.empty:
                # Get unique (Symbol, From_Account, Date) tuples from the transfers-in
                transfer_events = transfers_in_df[
                    ["Symbol", "Account", "Date"]
                ].drop_duplicates()

                # Create a list of masks to combine
                masks_to_combine = []

                for _, event_row in transfer_events.iterrows():
                    transfer_symbol = event_row["Symbol"]
                    transfer_from_account = event_row["Account"]
                    transfer_date = event_row["Date"]

                    # Create a mask for all preceding/concurrent tx for this specific asset
                    mask = (
                        (all_transactions_df_cleaned["Symbol"] == transfer_symbol)
                        & (
                            all_transactions_df_cleaned["Account"]
                            == transfer_from_account
                        )
                        & (all_transactions_df_cleaned["Date"] <= transfer_date)
                    )
                    masks_to_combine.append(mask)

                if masks_to_combine:
                    preceding_tx_mask = pd.concat(masks_to_combine, axis=1).any(axis=1)

            # 4. Final filter is the union of all three masks
            # --- MODIFIED: Ensure SPLIT transactions are always included ---
            # Splits are corporate actions that affect the stock globally, regardless of which account recorded it.
            # If we exclude the split transaction because it's in a different account (e.g. Sharebuilder vs E*TRADE),
            # the holdings calculation will be incorrect (missing the quantity adjustment).
            split_mask = (
                all_transactions_df_cleaned["Type"].astype(str).str.lower().isin(["split", "stock split"])
            )
            
            # --- MODIFIED: Ensure "All Accounts" transactions are always included ---
            # This is critical because some global actions (like splits, or manual adjustments)
            # are recorded under "All Accounts" and must not be filtered out when viewing a specific account.
            all_accounts_mask = (
                all_transactions_df_cleaned["Account"].astype(str).str.lower() == "all accounts"
            )
            
            final_combined_mask = (
                from_account_mask | to_account_mask | preceding_tx_mask | split_mask | all_accounts_mask
            )

            transactions_df_filtered = all_transactions_df_cleaned[
                final_combined_mask
            ].copy()
            filter_desc = f"Accounts: {', '.join(sorted(valid_include))}"
        else:
            transactions_df_filtered = pd.DataFrame(
                columns=all_transactions_df_cleaned.columns
            )  # Empty DF with same schema
            filter_desc = "No Valid Accounts Included"
            status_parts.append("No valid accounts after filter")
            has_warnings = (
                True  # This is a warning, not a critical error to stop all processing.
            )
    else:  # Should not happen if GUI validates include_accounts
        transactions_df_filtered = all_transactions_df_cleaned.copy()
        filter_desc = "All Accounts (Invalid Filter Type)"
        status_parts.append("Invalid include_accounts filter type")
        has_warnings = True
    logging.info(f"Processing transactions for scope: {filter_desc}")

    if transactions_df_filtered.empty and (
        not has_warnings or "No valid accounts after filter" not in status_parts
    ):
        # If filtering resulted in empty and it wasn't due to "No valid accounts" (which is a warning scenario),
        # it might be an issue or simply no transactions for the selected scope.
        logging.info("No transactions after filtering for the selected scope.")
        # Proceed, might result in empty summary, which is valid.
        # status_parts.append("No Tx Post-Filter") # Optional status part

    # --- ADDED: Prepare historical FX rates for _process_transactions_to_holdings ---
    historical_fx_for_processing: Dict[Tuple[date, str], float] = {}
    if not transactions_df_filtered.empty:
        logging.debug(
            "Preparing historical FX rates for transaction processing (current summary)..."
        )
        unique_dates = sorted(transactions_df_filtered["Date"].dt.date.unique())

        # Collect all relevant currencies for historical FX fetching
        # FIX: Use all_transactions_df_cleaned instead of transactions_df_filtered to ensure
        # we get currencies for ALL accounts, even excluded ones (needed for FIFO calc of transfers).
        currencies_for_hist_fx_fetch = set(
            all_transactions_df_cleaned["Local Currency"].unique()
        )
        currencies_for_hist_fx_fetch.add(display_currency)
        currencies_for_hist_fx_fetch.add(
            default_currency
        )  # default_currency is an arg to calculate_portfolio_summary

        # Clean the collected currencies
        cleaned_currencies_for_hist_fx = {
            str(c).strip().upper()
            for c in currencies_for_hist_fx_fetch
            if pd.notna(c)
            and isinstance(str(c).strip(), str)
            and str(c).strip() not in ["", "<NA>", "NAN", "NONE", "N/A"]
            and len(str(c).strip()) == 3
        }

        min_tx_date = all_transactions_df_cleaned["Date"].min().date()
        fx_pairs_to_fetch_hist = [
            f"{lc.upper()}=X"
            for lc in cleaned_currencies_for_hist_fx  # Use the cleaned and expanded set
            if lc and lc.upper() != "USD" and pd.notna(lc) and str(lc).strip() != ""
        ]

        if market_provider:
            market_provider_for_hist_fx = market_provider
        else:
            market_provider_for_hist_fx = MarketDataProvider(
                current_cache_file=cache_file_path
            )  # Use same cache config

        # Fetch historical FX rates (local_curr vs USD)
        # FIX: Use date.today() instead of report_date (latest trading date)
        # The Capital Gains API uses date.today(), so if we use an earlier date (e.g. Friday),
        # recent weekend transactions (crypto) or today's transactions will fail FX lookup
        # and result in NaN gain (0 assumed), causing the Dashboard to be lower.
        fx_fetch_end_date = date.today()

        historical_fx_data_usd_based, fx_fetch_err_hist = (
            market_provider_for_hist_fx.get_historical_fx_rates(
                fx_pairs_yf=list(set(fx_pairs_to_fetch_hist)),
                start_date=min_tx_date,
                end_date=fx_fetch_end_date,
                use_cache=True,
                # Use stable cache key (no dates) to allow incremental updates
                # The market_data provider will handle date range checks
                cache_key=f"PROC_FX_HIST_{'_'.join(sorted(list(set(fx_pairs_to_fetch_hist))))}",
            )
        )
        if fx_fetch_err_hist:
            logging.warning(
                "Failed to fetch some historical FX rates needed for precise FX G/L calculation in current summary. FX G/L might be inaccurate."
            )
            has_warnings = True

        # --- OPTIMIZED: Vectorized FX Rate Calculation ---
        # Instead of iterating dates and currencies, we build a master DataFrame.
        
        fx_series_list = []
        if historical_fx_data_usd_based:
            for pair, df in historical_fx_data_usd_based.items():
                if df is None or df.empty:
                    continue
                curr_code = pair.replace("=X", "")
                # Ensure index is datetime/date
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                
                # Just take the price column
                series = df['price'].rename(curr_code)
                # Ensure index is date (not datetime with time)
                series.index = series.index.date
                # Remove duplicates in index if any
                series = series[~series.index.duplicated(keep='last')]
                fx_series_list.append(series)

        if fx_series_list:
            # Concat into a wide DataFrame
            master_fx_df = pd.concat(fx_series_list, axis=1)
            master_fx_df.sort_index(inplace=True)
            
            # 2. Reindex to cover all transaction dates (ffill AND bfill)
            # We need the union of existing FX dates and transaction dates
            all_needed_dates = sorted(list(set(master_fx_df.index) | set(unique_dates)))
            # ADDED: bfill() to handle cases where transactions start before FX history
            master_fx_df = master_fx_df.reindex(all_needed_dates).ffill().bfill()
            
            # --- ADDED: Fallback for completely missing currencies ---
            # If a currency was requested but is not in master_fx_df (e.g. history fetch failed completely),
            # we try to fetch the CURRENT rate and use it as a constant fallback.
            existing_cols = set(master_fx_df.columns)
            missing_currencies = [
                c for c in cleaned_currencies_for_hist_fx 
                if c and c.upper() != "USD" and c not in existing_cols
            ]
            
            if missing_currencies:
                logging.warning(f"Historical FX missing for {missing_currencies}. Attempting to fetch current rates as fallback.")
                missing_pairs = [f"{c.upper()}=X" for c in missing_currencies]
                try:
                    # Fetch current quotes for missing pairs
                    _, current_fx_rates, _, _ = market_provider_for_hist_fx.get_current_quotes(
                        stock_tickers=[], fx_pairs=missing_pairs
                    )
                    
                    for curr in missing_currencies:
                        curr_upper = curr.upper()
                        if curr_upper in current_fx_rates and pd.notna(current_fx_rates[curr_upper]):
                            rate = current_fx_rates[curr_upper]
                            logging.info(f"Using current FX rate {rate} for {curr} as historical fallback (constant).")
                            master_fx_df[curr] = rate
                        else:
                            logging.error(f"Could not get even current FX rate for {curr}. FX conversion will fail/default.")
                            
                except Exception as e_fallback:
                    logging.error(f"Error fetching current FX fallback: {e_fallback}")

            # 3. Add USD column (always 1.0)
            master_fx_df['USD'] = 1.0
            
            # 4. Calculate Cross Rates (Display / Local)
            display_curr_col = display_currency.upper()
            if display_curr_col not in master_fx_df.columns:
                if display_curr_col == 'USD':
                     master_fx_df[display_curr_col] = 1.0
                else:
                     logging.warning(f"Display currency {display_curr_col} not found in FX data. FX conversion may fail.")
            
            if display_curr_col in master_fx_df.columns:
                display_rates = master_fx_df[display_curr_col]
                
                # Iterate over all available currencies to calculate cross rates
                for curr_col in master_fx_df.columns:
                    try:
                        # Calculate cross rate vector: Rate = Display / Local
                        # Note: master_fx_df contains Base/USD rates (e.g. EUR/USD ~ 0.92, THB/USD ~ 34.0)
                        # We want Display/Local.
                        # Display/Local = (Display/USD) / (Local/USD)
                        # Example: Display=USD (1.0), Local=EUR (0.92). Rate = 1.0 / 0.92 = 1.08 USD/EUR. Correct.
                        # Example: Display=THB (34.0), Local=USD (1.0). Rate = 34.0 / 1.0 = 34.0 THB/USD. Correct.
                        
                        cross_rates = display_rates / master_fx_df[curr_col]
                        
                        # Filter for relevant dates and update dict
                        relevant_rates = cross_rates.loc[cross_rates.index.isin(unique_dates)]
                        
                        # Update the dictionary
                        for d, r in relevant_rates.items():
                            if pd.notna(r):
                                historical_fx_for_processing[(d, curr_col)] = float(r)
                                
                    except Exception as e:
                        logging.warning(f"Error calculating vectorized FX for {curr_col}: {e}")
        
        # Fallback/Fill for USD if not present
        for d in unique_dates:
            if (d, 'USD') not in historical_fx_for_processing:
                historical_fx_for_processing[(d, 'USD')] = 1.0

    # --- END ADDED ---

    # --- 3. Process Stock/ETF Transactions ---
    holdings, _, _, _, ignored_indices_proc, ignored_reasons_proc, transfer_costs, warn_proc = (
        _process_transactions_to_holdings(
            transactions_df=transactions_df_filtered,  # Pass filtered DataFrame
            default_currency=default_currency,
            shortable_symbols=SHORTABLE_SYMBOLS,
            historical_fx_lookup=historical_fx_for_processing,  # NEW ARG
            display_currency_for_hist_fx=display_currency,  # NEW ARG
            report_date=report_date,
        )
    )

    # --- NEW: Patch Transfer Prices for Summary Calculation ---
    # We want the summary builder (IRR calc) to see the "Cost Price" for transfers,
    # so that the cash flow is generated correctly (as a flow at cost).
    transactions_for_summary = transactions_df_filtered.copy()
    if transfer_costs:
        logging.debug(f"Patching {len(transfer_costs)} transfer transactions with cost-based prices for IRR...")
        # We can iterate and set, or use map if index aligns.
        # Since transfer_costs keys are 'original_index', and transactions_for_summary has 'original_index' column:
        
        # Create a mapping series
        transfer_cost_series = pd.Series(transfer_costs)
        
        # Find rows where original_index is in the map
        mask = transactions_for_summary["original_index"].isin(transfer_costs.keys())
        
        # For these rows, update 'Price/Share'
        # We need to map the original_index to the price.
        # set_index temporarily to map easily
        temp_df = transactions_for_summary.set_index("original_index")
        temp_df.update(pd.DataFrame({"Price/Share": transfer_cost_series}))
        
        # Restore index/structure (update modifies in place but we need to be careful with index)
        # Actually, simpler loop might be safer to avoid index mess if original_index is not unique (though it should be)
        for orig_idx, cost_price in transfer_costs.items():
             transactions_for_summary.loc[
                transactions_for_summary["original_index"] == orig_idx, "Price/Share"
            ] = cost_price
    # --- END NEW ---
    combined_ignored_indices.update(ignored_indices_proc)
    combined_ignored_reasons.update(ignored_reasons_proc)
    if warn_proc:
        has_warnings = True
    if ignored_reasons_proc:
        status_parts.append(f"Processing Issues: {len(ignored_reasons_proc)}")

    # --- 4. (Obsolete) Cash Balance Calculation ---
    # Cash is now processed as a regular holding in _process_transactions_to_holdings.
    # The cash_summary dict is no longer needed.
    # Removed obsolete cash balance calculation block.

    # --- 5. Fetch Current Market Data ---
    all_stock_symbols_internal = list(set(key[0] for key in holdings.keys()))
    # 2. Determine required currencies
    required_currencies: Set[str] = set([display_currency, default_currency])
    for data in holdings.values():
        required_currencies.add(data.get("local_currency", default_currency))
    
    # --- ADDED: Scan all transactions for used currencies ---
    # This ensures we have rates even for currencies not in the current account map (e.g. historical/closed)
    if isinstance(all_transactions_df_cleaned, pd.DataFrame) and "Local Currency" in all_transactions_df_cleaned.columns:
        unique_tx_currencies = all_transactions_df_cleaned["Local Currency"].dropna().unique()
        for curr in unique_tx_currencies:
            if isinstance(curr, str) and len(curr.strip()) == 3:
                required_currencies.add(curr.strip().upper())
    # -------------------------------------------------------

    required_currencies.discard(None)  # type: ignore
    # --- ADDED: More robust cleaning for required_currencies ---
    cleaned_required_currencies = {
        str(c).strip().upper()
        for c in required_currencies
        if pd.notna(c)
        and isinstance(str(c).strip(), str)
        and str(c).strip() not in ["", "<NA>", "NAN", "NONE", "N/A"]
        and len(str(c).strip()) == 3
    }
    required_currencies = cleaned_required_currencies

    if market_provider is None:
        market_provider = MarketDataProvider(current_cache_file=cache_file_path)
    current_stock_data_internal, current_fx_rates_vs_usd, current_fx_prev_close_vs_usd, err_fetch, warn_fetch = (
        market_provider.get_current_quotes(
            internal_stock_symbols=all_stock_symbols_internal,
            required_currencies=required_currencies,  # Pass as set
            user_symbol_map=effective_user_symbol_map,
            user_excluded_symbols=effective_user_excluded_symbols,
        )
    )
    if err_fetch:  # If get_current_quotes signals a critical error
        has_errors = True  # Treat critical fetch error as overall error
        status_parts.append("Fetch Failed Critically")
    
    if warn_fetch:
        has_warnings = True
        status_parts.append("Fetch Warnings")


    # If fetch failed critically, we might not be able to proceed meaningfully.
    if has_errors and "Fetch Failed Critically" in status_parts:
        msg = "Error: Price/FX fetch failed critically via MarketDataProvider. Proceeding with partial data/fallbacks."
        logging.error(f"WARNING: {msg}") # Downgrade to warning for execution flow
        # final_status_prefix = "Finished with Errors"
        # final_status = f"{final_status_prefix} ({filter_desc})" + (
        #     f" [{'; '.join(status_parts)}]" if status_parts else ""
        # )
        # return (
        #     get_default_metrics_dict(
        #         display_currency,
        #         report_date,
        #         available_accounts_for_errors,
        #         is_empty_data_case=False,
        #     ),  # MODIFIED
        #     None,  # summary_df_final
        #     None,  # account_level_metrics
        #     combined_ignored_indices,
        #     combined_ignored_reasons,
        #     final_status,
        # )
        pass # Continue execution


    # --- 6. Build Detailed Summary Rows ---
    # --- ADDED: Calculate Realized Gains using FIFO (Capital Gains Logic) ---
    # This ensures consistency between Dashboard and Capital Gains tab.
    # We calculate it separately and then override the values in 'holdings'.
    fifo_realized_gains_df = pd.DataFrame()
    open_lots_dict = {}  # NEW: To store open lots
    try:
        logging.info("Calculating FIFO Realized Gains & Lots for Dashboard...")
        
        # Ensure transactions are sorted chronologically and by original index for stable FIFO
        # This matches the logic in strict capital gains extraction
        fifo_input_df = all_transactions_df_cleaned.copy()
        if "original_index" in fifo_input_df.columns:
             fifo_input_df.sort_values(by=["Date", "original_index"], inplace=True)
        else:
             fifo_input_df.sort_values(by=["Date"], inplace=True)

        # Call the new function that returns both gains and lots
        # FAILSAFE CORRECTION: The API endpoint for Capital Gains (get_capital_gains) DOES pass 'current_fx_rates_vs_usd'.
        # Previously, we thought it didn't and passed None, which caused a discrepancy (Dashboard lower by ~$9k)
        # because the fallback to current FX for missing historical rates was blocked.
        fifo_realized_gains_df, open_lots_dict = calculate_fifo_lots_and_gains(
            transactions_df=fifo_input_df, # Use SORTED FULL history
            display_currency=display_currency,
            historical_fx_yf=historical_fx_data_usd_based,
            default_currency=default_currency,
            shortable_symbols=SHORTABLE_SYMBOLS,
            stock_quantity_close_tolerance=STOCK_QUANTITY_CLOSE_TOLERANCE,
            current_fx_rates_vs_usd=current_fx_rates_vs_usd, # Pass the available rates!
        )
        
        # --- DEBUG LOGGING ---
        logging.info(f"FIFO DF Shape: {fifo_realized_gains_df.shape}")
        logging.info(f"Open Lots Count: {len(open_lots_dict)}")
        if not fifo_realized_gains_df.empty:
            if "Realized Gain (Display)" in fifo_realized_gains_df.columns:
                nans_disp = fifo_realized_gains_df["Realized Gain (Display)"].isna().sum()
                logging.info(f"NaNs in Realized Gain (Display): {nans_disp}")
        # ---------------------

        # Aggregate FIFO gains by (Symbol, Account)
        if not fifo_realized_gains_df.empty:
            fifo_gains_agg = (
                fifo_realized_gains_df.groupby(["Symbol", "Account"])[
                    ["Realized Gain (Display)", "Realized Gain (Local)"]
                ]
                .sum()
                .to_dict(orient="index")
            )
        else:
            fifo_gains_agg = {}

        # Override realized gains and attach lots in holdings
        for (sym, acct), holding_data in holdings.items():
            sym_key_upper = str(sym).upper().strip()
            acct_key_upper = str(acct).upper().strip()
            lookup_key = (sym_key_upper, acct_key_upper)
            
            # 1. Override Realized Gains
            if lookup_key in fifo_gains_agg:
                fifo_gain_display = fifo_gains_agg[lookup_key].get("Realized Gain (Display)", 0.0)
                fifo_gain_local = fifo_gains_agg[lookup_key].get("Realized Gain (Local)", 0.0)
                
                holding_data["realized_gain_display"] = fifo_gain_display
                holding_data["realized_gain_local"] = fifo_gain_local
                logging.debug(f"Overrode realized gain for {sym}/{acct} with FIFO value: {fifo_gain_display}")
            else:
                # Same fallback logic as before
                # ... (keep existing fallback checks if needed, or simplify)
                # If no FIFO record found, set to 0 unless shortable/cash special case
                 is_shortable = sym in SHORTABLE_SYMBOLS
                 is_cash = sym == CASH_SYMBOL_CSV or sym == "$CASH"
                 avg_cost_gain = holding_data.get("realized_gain_display", 0.0)
                 
                 if abs(avg_cost_gain) > 1e-9 and not is_shortable and not is_cash:
                      # If avg cost logic had a gain but FIFO doesn't, it might mean FIFO failed or logic diff.
                      # Ideally we set to 0 for consistency, but let's be safe.
                      pass

            # 2. Attach and Calculate Lots
            if lookup_key in open_lots_dict:
                lots = open_lots_dict[lookup_key]
                processed_lots = []
                
                # We need current price/FX to calc Lot Market Value & Unrealized Gain
                # Try to get them from pre-fetched stock data
                current_price = 0.0
                if sym in current_stock_data_internal:
                    current_price = current_stock_data_internal[sym].get("price", 0.0)
                    if pd.isna(current_price): current_price = 0.0
                
                # We also need conversion rate from Local -> Display
                # holding_data has 'local_currency'
                local_curr = holding_data.get("local_currency", default_currency)
                fx_rate_curr = get_conversion_rate(
                    local_curr, display_currency, current_fx_rates_vs_usd
                )
                if pd.isna(fx_rate_curr): fx_rate_curr = 0.0 # Safety

                for lot in lots:
                    # Lot fields needed: Date, Quantity, Cost Basis (Display), Mkt Val (Display), Unreal Gain (Display)
                    # Lot struct from analyzer: 'qty', 'cost_per_share_local_net', 'purchase_date', 'purchase_fx_to_display'
                    
                    l_qty = lot["qty"]
                    l_date = lot["purchase_date"]
                    l_cost_local = lot["cost_per_share_local_net"]
                    l_purch_fx = lot["purchase_fx_to_display"]
                    
                    if pd.isna(l_purch_fx): l_purch_fx = 0.0 # Should not happen if data good
                    
                    # Cost Basis Display = Qty * Cost_Local * Purchase_FX
                    l_cost_basis_display = l_qty * l_cost_local * l_purch_fx
                    
                    # Market Value Display = Qty * Current_Price_Local * Current_FX
                    l_mkt_val_display = l_qty * current_price * fx_rate_curr
                    
                    l_unreal_gain_display = l_mkt_val_display - l_cost_basis_display
                    l_unreal_gain_pct = (l_unreal_gain_display / l_cost_basis_display * 100) if abs(l_cost_basis_display) > 1e-9 else 0.0

                    processed_lots.append({
                        "Date": l_date.strftime("%Y-%m-%d") if isinstance(l_date, (date, datetime)) else str(l_date),
                        "Quantity": l_qty,
                        "Cost Basis": l_cost_basis_display,
                        "Market Value": l_mkt_val_display,
                        "Unreal. Gain": l_unreal_gain_display,
                        "Unreal. Gain %": l_unreal_gain_pct,
                        "purchase_fx": l_purch_fx, # Debug
                        "cost_local": l_cost_local # Debug
                    })
                
                holding_data["lots"] = processed_lots

    except Exception as e_fifo:
        logging.error(f"Error calculating FIFO realized gains & lots for Dashboard: {e_fifo}")
        logging.error(traceback.format_exc())
        # Fallback: Do nothing, keep Average Cost values
    # --- END ADDED ---

    manual_prices_for_build_rows = {}
    if manual_overrides_effective:
        for symbol_key, override_values_dict in manual_overrides_effective.items():
            if (
                isinstance(override_values_dict, dict)
                and "price" in override_values_dict
            ):
                price_val = override_values_dict["price"]
                if price_val is not None and pd.notna(price_val):
                    try:
                        manual_prices_for_build_rows[symbol_key] = float(price_val)
                    except (ValueError, TypeError):
                        logging.warning(
                            f"Could not convert manual price for {symbol_key} to float: {price_val}"
                        )

    (
        portfolio_summary_rows,
        account_local_currency_map,
        err_build,
        warn_build,
    ) = _build_summary_rows(
        holdings=holdings,
        current_stock_data=(
            current_stock_data_internal
            if current_stock_data_internal is not None
            else {}
        ),
        current_fx_rates_vs_usd=(
            current_fx_rates_vs_usd if current_fx_rates_vs_usd is not None else {}
        ),
        current_fx_prev_close_vs_usd=(
            current_fx_prev_close_vs_usd if current_fx_prev_close_vs_usd is not None else {}
        ),
        display_currency=display_currency,
        default_currency=default_currency,
        transactions_df=transactions_for_summary,  # Pass PATCHED DataFrame
        report_date=report_date,
        shortable_symbols=SHORTABLE_SYMBOLS,
        user_excluded_symbols=effective_user_excluded_symbols,
        user_symbol_map=effective_user_symbol_map,
        manual_prices_dict=manual_prices_for_build_rows,
        account_interest_rates=account_interest_rates, # NEW
        interest_free_thresholds=interest_free_thresholds, # NEW
    )

    if err_build:
        has_errors = True
    if warn_build:
        has_warnings = True

    if has_errors:  # Critical error during summary row building (likely FX)
        msg = "Error: Failed critically during summary row building (likely FX)."
        logging.error(msg)
        status_parts.append("Summary Build Error")
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            get_default_metrics_dict(
                display_currency,
                report_date,
                available_accounts_for_errors,
                is_empty_data_case=False,
            ),  # MODIFIED
            None,  # summary_df_final
            {},    # holdings dictionary (empty) <-- ADDED
            None,  # account_level_metrics
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )
    elif not portfolio_summary_rows and (holdings):  # Generated no rows but had input
        msg = "Warning: Failed to generate any summary rows (FX or other issue during build)."
        logging.warning(msg)
        has_warnings = True
        status_parts.append("Summary Build Generated No Rows")
        # Don't return early, let it try to create empty aggregates.

    # The LotsViewerDialog expects keys like "Quantity", "Avg Cost" (Title Case).
    # We back-populate these from the calculated summary rows into the raw holdings dict.
    if portfolio_summary_rows and holdings:
        # Create a normalized map for case-insensitive lookup: (upper_sym, upper_acct) -> original_key
        holdings_map_norm = {
            (str(k[0]).upper().strip(), str(k[1]).upper().strip()): k 
            for k in holdings.keys()
        }
        
        for row in portfolio_summary_rows:
            r_sym = row.get("Symbol")
            r_acct = row.get("Account")
            if r_sym and r_acct:
                # Reconstruct lookup key (upper)
                h_key_upper = (str(r_sym).upper().strip(), str(r_acct).upper().strip())
                
                # Look up original key
                original_key = holdings_map_norm.get(h_key_upper)
                
                if original_key and original_key in holdings:
                    holdings[original_key]["Symbol"] = r_sym
                    holdings[original_key]["Account"] = r_acct
                    holdings[original_key]["Quantity"] = row.get("Quantity", 0)
                    holdings[original_key]["Avg Cost"] = row.get(f"Avg Cost ({display_currency})", 0)
                    holdings[original_key]["Market Value"] = row.get(f"Market Value ({display_currency})", 0)

    # --- Add Sector/Geo information ---
    # logging.info("Categorizing transactions by symbol...")
    if portfolio_summary_rows:
        summary_df_unfiltered_temp = pd.DataFrame(portfolio_summary_rows)
        if "Symbol" in summary_df_unfiltered_temp.columns:
            symbols_in_summary = summary_df_unfiltered_temp["Symbol"].unique()
            sector_map, quote_type_map, country_map, industry_map = {}, {}, {}, {}

            for internal_symbol in symbols_in_summary:
                symbol_overrides = manual_overrides_effective.get(
                    internal_symbol.upper(), {}
                )
                manual_asset_type = symbol_overrides.get("asset_type", "").strip()
                manual_sector = symbol_overrides.get("sector", "").strip()
                manual_geography = symbol_overrides.get("geography", "").strip()
                manual_industry = symbol_overrides.get("industry", "").strip()

                if internal_symbol == CASH_SYMBOL_CSV or internal_symbol.startswith(
                    "Cash ("
                ):
                    sector_map[internal_symbol] = "Cash"
                    quote_type_map[internal_symbol] = "CASH"
                    country_map[internal_symbol] = "Cash"
                    industry_map[internal_symbol] = "Cash"
                    continue

                yf_ticker_for_sector = map_to_yf_symbol(
                    internal_symbol,
                    effective_user_symbol_map,
                    effective_user_excluded_symbols,
                )
                sector_map[internal_symbol] = (
                    manual_sector if manual_sector else "N/A (No YF/Manual)"
                )
                quote_type_map[internal_symbol] = (
                    manual_asset_type if manual_asset_type else "N/A (No YF/Manual)"
                )
                country_map[internal_symbol] = (
                    manual_geography if manual_geography else "N/A (No YF/Manual)"
                )
                industry_map[internal_symbol] = (
                    manual_industry if manual_industry else "N/A (No YF/Manual)"
                )

                if yf_ticker_for_sector and (
                    not manual_sector
                    or not manual_asset_type
                    or not manual_geography
                    or not manual_industry
                ):
                    fundamental_info = market_provider.get_fundamental_data(
                        yf_ticker_for_sector
                    )
                    if fundamental_info and isinstance(fundamental_info, dict):
                        if not manual_sector:
                            sector_map[internal_symbol] = fundamental_info.get(
                                "sector", "Unknown Sector"
                            )
                        if not manual_asset_type:
                            quote_type_map[internal_symbol] = fundamental_info.get(
                                "quoteType", "UNKNOWN"
                            )
                        if not manual_geography:
                            country_map[internal_symbol] = fundamental_info.get(
                                "country", "Unknown Region"
                            )
                        if not manual_industry:
                            industry_map[internal_symbol] = fundamental_info.get(
                                "industry", "Unknown Industry"
                            )
                    else:  # Fetch failed for this symbol
                        if not manual_sector:
                            sector_map[internal_symbol] = "N/A (Fetch Error)"
                        if not manual_asset_type:
                            quote_type_map[internal_symbol] = "N/A (Fetch Error)"
                        if not manual_geography:
                            country_map[internal_symbol] = "N/A (Fetch Error)"
                        if not manual_industry:
                            industry_map[internal_symbol] = "N/A (Fetch Error)"

            summary_df_unfiltered_temp["Sector"] = (
                summary_df_unfiltered_temp["Symbol"]
                .map(sector_map)
                .fillna("Unknown Sector")
            )
            summary_df_unfiltered_temp["quoteType"] = (
                summary_df_unfiltered_temp["Symbol"]
                .map(quote_type_map)
                .fillna("UNKNOWN")
            )
            summary_df_unfiltered_temp["Country"] = (
                summary_df_unfiltered_temp["Symbol"]
                .map(country_map)
                .fillna("Unknown Region")
            )
            summary_df_unfiltered_temp["Industry"] = (
                summary_df_unfiltered_temp["Symbol"]
                .map(industry_map)
                .fillna("Unknown Industry")
            )
            portfolio_summary_rows = summary_df_unfiltered_temp.to_dict(
                orient="records"
            )

    # --- 7. Create DataFrame & Calculate Aggregates ---
    summary_df_unfiltered = pd.DataFrame()
    overall_summary_metrics: Dict[str, Any] = {}  # Ensure it's a dict
    account_level_metrics: Dict[str, Dict[str, float]] = {}  # Ensure it's a dict

    if not portfolio_summary_rows:  # If still empty after sector enrichment
        logging.warning(
            "Portfolio summary list is empty. Returning empty results for aggregates."
        )
        has_warnings = True
        # Define default empty structure for overall_summary_metrics
        overall_summary_metrics = {
            "market_value": 0.0,
            "cost_basis_held": 0.0,
            "unrealized_gain": 0.0,
            "realized_gain": 0.0,
            "dividends": 0.0,
            "commissions": 0.0,
            "total_gain": 0.0,
            "total_cost_invested": 0.0,
            "total_buy_cost": 0.0,
            "portfolio_mwr": np.nan,
            "day_change_display": 0.0,
            "day_change_percent": np.nan,
            "report_date": report_date.strftime("%Y-%m-%d"),
            "display_currency": display_currency,
            "cumulative_investment": 0.0,
            "total_return_pct": np.nan,
            "est_annual_income_display": 0.0,  # This is for empty portfolio_summary_rows
        }
        # account_level_metrics remains empty dict
        summary_df_unfiltered = pd.DataFrame()  # Ensure it's an empty DataFrame
    else:
        full_summary_df = pd.DataFrame(portfolio_summary_rows)
        # ... (numeric conversion and sorting as before) ...
        try:
            money_cols_display = [
                c for c in full_summary_df.columns if f"({display_currency})" in c
            ]
            percent_cols = [
                "Unreal. Gain %",
                "Total Return %",
                "IRR (%)",
                "Day Change %",
                "Div. Yield (Cost) %",
                "Div. Yield (Current) %",
            ]
            numeric_cols_to_convert = ["Quantity"] + money_cols_display + percent_cols
            for col in numeric_cols_to_convert:
                if col in full_summary_df.columns:
                    full_summary_df[col] = pd.to_numeric(
                        full_summary_df[col], errors="coerce"
                    )
        except Exception as e:
            has_warnings = True
            logging.warning(
                f"Warning: Error during numeric conversion of summary columns: {e}"
            )
        try:
            full_summary_df.sort_values(
                by=["Account", f"Market Value ({display_currency})"],
                ascending=[True, False],
                na_position="last",
                inplace=True,
            )
        except KeyError:
            has_warnings = True
            logging.warning(
                "Warning: Could not sort summary DataFrame by Account/Market Value."
            )

        overall_summary_metrics_temp, account_level_metrics_temp, err_agg, warn_agg = (
            _calculate_aggregate_metrics(
                full_summary_df=full_summary_df,
                display_currency=display_currency,
                report_date=report_date,
                include_accounts=include_accounts,
                all_available_accounts=available_accounts_for_errors,
            )
        )
        # Ensure types before assignment
        overall_summary_metrics = (
            overall_summary_metrics_temp
            if isinstance(overall_summary_metrics_temp, dict)
            else {}
        )
        account_level_metrics = (
            account_level_metrics_temp
            if isinstance(account_level_metrics_temp, dict)
            else {}
        )
        

        if err_agg:
            has_errors = True
        if warn_agg:
            has_warnings = True

        # Price source warnings check (as before)
        price_source_warnings = False
        if "Price Source" in full_summary_df.columns:
            non_cash_holdings_agg = full_summary_df[
                full_summary_df["Symbol"] != CASH_SYMBOL_CSV
            ]
            if not non_cash_holdings_agg.empty:
                if (
                    non_cash_holdings_agg["Price Source"]
                    .str.contains(
                        "Fallback|Excluded|Invalid|Error|Zero", case=False, na=False
                    )
                    .any()
                ):
                    price_source_warnings = True
        if price_source_warnings:
            has_warnings = True
            status_parts.append("Fallback Prices Used")
        summary_df_unfiltered = full_summary_df

    # --- MOVED OUTSIDE if/else: Override Dividends (Consistency with Dividends Tab) ---
    # This ensures that even if 'summary_df_unfiltered' is empty (closed account), we still calculate correct totals.
    try:
            # Extract dividend history using the authoritative function
            div_history_df = extract_dividend_history(
                all_transactions_df=transactions_df_filtered, # Use filtered tx
                display_currency=display_currency,
                historical_fx_yf=historical_fx_data_usd_based,
                default_currency=default_currency,
                include_accounts=include_accounts
            )

            if not div_history_df.empty:
                total_dividends_override = div_history_df["DividendAmountDisplayCurrency"].sum()
                
                # Override Overall Dividends
                overall_summary_metrics["dividends"] = total_dividends_override
                overall_summary_metrics["total_dividends_display"] = total_dividends_override
                
                # Recalculate Total Gain (using UNMODIFIED realized/unrealized for now - realized will be updated next)
                unrealized = overall_summary_metrics.get("unrealized_gain", 0.0)
                realized = overall_summary_metrics.get("realized_gain", 0.0)
                commissions = overall_summary_metrics.get("commissions", 0.0)
                
                new_total_gain_div = realized + unrealized + total_dividends_override - commissions
                overall_summary_metrics["total_gain"] = new_total_gain_div
                
                # Recalc Total Return %
                total_buy_cost = overall_summary_metrics.get("total_buy_cost", 0.0)
                if abs(total_buy_cost) > 1e-9:
                    overall_summary_metrics["total_return_pct"] = (new_total_gain_div / total_buy_cost) * 100.0
                elif abs(new_total_gain_div) <= 1e-9:
                    overall_summary_metrics["total_return_pct"] = 0.0

                # Override Account-Level Dividends
                divs_by_account = div_history_df.groupby("Account")["DividendAmountDisplayCurrency"].sum().to_dict()
                
                # FIX: Ensure all accounts with dividends exist in metrics, creating them if needed (for closed accounts)
                for acct, div_amt in divs_by_account.items():
                    if acct not in account_level_metrics:
                         account_level_metrics[acct] = {
                             "total_realized_gain_display": 0.0,
                             "total_unrealized_gain_display": 0.0,
                             "total_dividends_display": 0.0,
                             "total_commissions_display": 0.0,
                             "total_gain_display": 0.0,
                             "total_buy_cost_display": 0.0,
                             "total_return_pct": 0.0,
                             "market_value": 0.0
                         }
                    
                    metrics = account_level_metrics[acct]
                    metrics["total_dividends_display"] = div_amt
                    
                    # Recalculate Account Total Gain
                    acct_realized = metrics.get("total_realized_gain_display", 0.0)
                    acct_unrealized = metrics.get("total_unrealized_gain_display", 0.0)
                    acct_commissions = metrics.get("total_commissions_display", 0.0)
                    
                    acct_new_total_gain = acct_realized + acct_unrealized + div_amt - acct_commissions
                    metrics["total_gain_display"] = acct_new_total_gain
                    
                    acct_buy_cost = metrics.get("total_buy_cost_display", 0.0)
                    if abs(acct_buy_cost) > 1e-9:
                            metrics["total_return_pct"] = (acct_new_total_gain / acct_buy_cost) * 100.0
                    elif abs(acct_new_total_gain) <= 1e-9:
                            metrics["total_return_pct"] = 0.0
                
                logging.info(f"Overrode Dashboard Total Dividends: {total_dividends_override} (Legacy Extraction)")
            else:
                logging.info("Dividend extraction returned empty. Using default aggregation.")
                
    except Exception as e_div:
            logging.error(f"Error overriding dividends: {e_div}")
            logging.error(traceback.format_exc())

    # --- MOVED OUTSIDE if/else: Override Aggregate Realized Gains with FIFO Totals ---
    if not fifo_realized_gains_df.empty:
        try:
            fifo_df_for_sum = fifo_realized_gains_df.copy()

            # Filter by Account if specific accounts are requested
            if include_accounts and isinstance(include_accounts, list):
                    # Ensure we match account names correctly (case-insensitive usually preferred but stick to strict match if data is clean)
                    # fifo_realized_gains_df usually has Normalized Upper case accounts? 
                    # calculate_fifo_lots_and_gains normalizes accounts to upper.
                    # include_accounts should be normalized too.
                    # Let's normalize both for safety.
                    include_norm = [str(a).strip().upper() for a in include_accounts]
                    fifo_df_for_sum = fifo_df_for_sum[fifo_df_for_sum['Account'].isin(include_norm)]
                    
            # 1. Overall Override
            total_fifo_gain = fifo_df_for_sum["Realized Gain (Display)"].sum()
            
            # DEBUG LOGGING for User Issue
            if include_accounts is None:
                logging.info(f"debug_agg: All Accounts View. Total FIFO Gain: {total_fifo_gain}")
                # Log top contributors
                breakdown = fifo_df_for_sum.groupby("Account")["Realized Gain (Display)"].sum().sort_values(ascending=False)
                logging.info(f"debug_agg: Breakdown by Account: {breakdown.to_dict()}")
            else:
                logging.info(f"debug_agg: Filtered View ({include_accounts}). Total FIFO Gain: {total_fifo_gain}")
            
            # Update realized gain
            old_realized = overall_summary_metrics.get("realized_gain", 0.0)
            overall_summary_metrics["realized_gain"] = total_fifo_gain
            overall_summary_metrics["total_realized_gain_display"] = total_fifo_gain
            
            # Update Total Gain (Realized + Unrealized + Div - Comm)
            # Re-calculate to ensure consistency
            unrealized = overall_summary_metrics.get("unrealized_gain", 0.0)
            dividends = overall_summary_metrics.get("dividends", 0.0) # Already overridden (if applicable)
            commissions = overall_summary_metrics.get("commissions", 0.0)
            
            new_total_gain = total_fifo_gain + unrealized + dividends - commissions
            overall_summary_metrics["total_gain"] = new_total_gain
            
            # Recalculate Total Return %
            total_buy_cost = overall_summary_metrics.get("total_buy_cost", 0.0)
            if abs(total_buy_cost) > 1e-9:
                overall_summary_metrics["total_return_pct"] = (new_total_gain / total_buy_cost) * 100.0
            elif abs(new_total_gain) <= 1e-9:
                overall_summary_metrics["total_return_pct"] = 0.0
            
            logging.info(f"Overrode Dashboard Total Realized Gain: {old_realized} -> {total_fifo_gain} (FIFO Local)")

            # 2. Account-Level Override
            fifo_gains_by_account = fifo_realized_gains_df.groupby("Account")["Realized Gain (Display)"].sum().to_dict()
            
            # FIX: Ensure all accounts with gains exist in metrics (Closed Accounts)
            for acct, acct_fifo_gain in fifo_gains_by_account.items():
                if include_accounts and isinstance(include_accounts, list):
                     if str(acct).strip().upper() not in [str(a).strip().upper() for a in include_accounts]:
                         continue # Skip if not in requested filter

                if acct not in account_level_metrics:
                         account_level_metrics[acct] = {
                             "total_realized_gain_display": 0.0,
                             "total_unrealized_gain_display": 0.0,
                             "total_dividends_display": 0.0,
                             "total_commissions_display": 0.0,
                             "total_gain_display": 0.0,
                             "total_buy_cost_display": 0.0,
                             "total_return_pct": 0.0,
                             "market_value": 0.0
                         }

                metrics = account_level_metrics[acct]
                
                old_acct_realized = metrics.get("total_realized_gain_display", 0.0)
                metrics["total_realized_gain_display"] = acct_fifo_gain
                
                # Update Total Gain for account
                acct_unrealized = metrics.get("total_unrealized_gain_display", 0.0)
                acct_dividends = metrics.get("total_dividends_display", 0.0)
                acct_commissions = metrics.get("total_commissions_display", 0.0)
                
                acct_new_total_gain = acct_fifo_gain + acct_unrealized + acct_dividends - acct_commissions
                metrics["total_gain_display"] = acct_new_total_gain
                
                # Update Account Total Return %
                acct_buy_cost = metrics.get("total_buy_cost_display", 0.0)
                if abs(acct_buy_cost) > 1e-9:
                        metrics["total_return_pct"] = (acct_new_total_gain / acct_buy_cost) * 100.0
                elif abs(acct_new_total_gain) <= 1e-9:
                        metrics["total_return_pct"] = 0.0
                
                # logging.debug(f"Overrode Account {acct} Realized Gain: {old_acct_realized} -> {acct_fifo_gain}")
                    
        except Exception as e_override:
            logging.error(f"Error overriding aggregate realized gains with FIFO: {e_override}")

    # --- 8. Filter Closed Positions ---
    summary_df_final = pd.DataFrame()
    if summary_df_unfiltered.empty:
        summary_df_final = summary_df_unfiltered  # Will be empty DF
    elif not show_closed_positions:
        if (
            "Quantity" in summary_df_unfiltered.columns
            and "Symbol" in summary_df_unfiltered.columns
        ):
            held_mask = (
                (
                    summary_df_unfiltered["Quantity"].abs()
                    >= STOCK_QUANTITY_CLOSE_TOLERANCE
                )
                | (summary_df_unfiltered["Symbol"] == CASH_SYMBOL_CSV)
                | (summary_df_unfiltered["Symbol"].str.startswith("Cash (", na=False))
            )
            summary_df_final = summary_df_unfiltered[held_mask].copy()
        else:  # Columns missing for filtering
            summary_df_final = summary_df_unfiltered
            logging.warning(
                "Could not filter closed positions, Quantity/Symbol columns missing from summary_df_unfiltered."
            )
            has_warnings = True
    else:  # show_closed_positions is True
        summary_df_final = summary_df_unfiltered

    # --- 9a. Calculate Contribution % ---
    if not summary_df_final.empty and overall_summary_metrics:
        total_inv = overall_summary_metrics.get("total_cost_invested", 0.0)
        total_gain_col = f"Total Gain ({display_currency})"
        
        # Calculate Contribution % (Total Gain / Total Cost Invested)
        if total_gain_col in summary_df_final.columns and abs(total_inv) > 1e-9:
             summary_df_final["Contribution %"] = (summary_df_final[total_gain_col] / total_inv) * 100.0
        else:
             summary_df_final["Contribution %"] = 0.0

    # --- 9b. Calculate % of Total ---
    if not summary_df_final.empty and overall_summary_metrics:
        total_mkt_val = overall_summary_metrics.get("market_value", 0.0)
        mkt_val_col = f"Market Value ({display_currency})"
        
        if total_mkt_val and abs(total_mkt_val) > 1e-9 and mkt_val_col in summary_df_final.columns:
            # Calculate % using the market value of each holding vs total portfolio market value
            summary_df_final["pct_of_total"] = (summary_df_final[mkt_val_col] / total_mkt_val) * 100.0
        else:
            summary_df_final["pct_of_total"] = 0.0

    # --- 10. Add Metadata to Overall Summary ---
    if overall_summary_metrics is None:
        overall_summary_metrics = {}  # Ensure it's a dict
    # Use all_transactions_df_cleaned for available accounts, as it's the full dataset for this run
    unique_available_accounts = set()
    if all_transactions_df_cleaned is not None:
        if "Account" in all_transactions_df_cleaned.columns:
            unique_available_accounts.update(all_transactions_df_cleaned["Account"].dropna().unique())
        if "To Account" in all_transactions_df_cleaned.columns:
            unique_available_accounts.update(all_transactions_df_cleaned["To Account"].dropna().unique())
    
    # Filter out empty strings and None values explicitly
    unique_available_accounts = {acc for acc in unique_available_accounts if acc and isinstance(acc, str) and acc.strip()}
    overall_summary_metrics["_available_accounts"] = sorted(list(unique_available_accounts))
    if display_currency != default_currency and current_fx_rates_vs_usd:
        rate_to_display = get_conversion_rate(
            default_currency, display_currency, current_fx_rates_vs_usd
        )
        overall_summary_metrics["exchange_rate_to_display"] = (
            rate_to_display if pd.notna(rate_to_display) else None
        )

    # --- 9. Determine Final Status ---
    end_time_summary = time.time()
    status_prefix = "Success"
    if has_errors:
        status_prefix = "Finished with Errors"
    elif has_warnings:
        status_prefix = "Finished with Warnings"
    final_status = f"{status_prefix} ({filter_desc})"
    if status_parts:
        final_status += f" [{'; '.join(status_parts)}]"

    # logging.info(f"Portfolio Summary Calculation Finished ({filter_desc})")
    logging.info(
        f"Total Summary Calc Time: {end_time_summary - start_time_summary:.2f} seconds"
    )

    return (
        overall_summary_metrics,
        summary_df_final,
        holdings, # holdings dictionary (with lots) <-- ADDED
        dict(account_level_metrics),  # Ensure it's a dict
        combined_ignored_indices,
        combined_ignored_reasons,
        final_status,
    )


# =======================================================================
# --- SECTION: HISTORICAL PERFORMANCE CALCULATION FUNCTIONS (REVISED) ---
# =======================================================================
# --- NOTE: These functions remain in portfolio_logic.py as they were not moved ---


# --- Function to Unadjust Prices (Keep as is) ---
@profile
def _unadjust_prices(
    adjusted_prices_yf: Dict[str, pd.DataFrame],
    yf_to_internal_map: Dict[str, str],
    splits_by_internal_symbol: Dict[str, List[Dict]],
    processed_warnings: set,
) -> Dict[str, pd.DataFrame]:
    """
    Calculates the current portfolio summary using MarketDataProvider for market data
    and helper functions from portfolio_analyzer.py for calculations.

    This function orchestrates the calculation of the current portfolio summary. It loads
    and cleans transactions, calculates holdings and cash balances, fetches current market
    data, builds detailed summary rows, and calculates aggregate metrics.  It relies on
    helper functions from `portfolio_analyzer.py` for the core calculations and
    `MarketDataProvider` for real-time price and FX data.

    Args:
        transactions_csv_file (str): Path to the CSV file containing transaction data.
        fmp_api_key (Optional[str], optional): Unused. API key for Financial Modeling Prep. Defaults to None.
        display_currency (str, optional): The currency in which to display all values. Defaults to "USD".
        show_closed_positions (bool, optional): Whether to include closed positions in the summary. Defaults to False.
        account_currency_map (Dict, optional): Mapping of account names to their local currencies. Defaults to {"SET": "THB"}.
        default_currency (str, optional): The default currency used when a transaction doesn't specify one. Defaults to "USD".
        cache_file_path (str, optional): Path to the cache file used by MarketDataProvider. Defaults to DEFAULT_CURRENT_CACHE_FILE_PATH.
        include_accounts (Optional[List[str]], optional): A list of account names to include in the calculation. If None, includes all accounts. Defaults to None.
        manual_prices_dict (Optional[Dict[str, float]], optional): Dictionary of manual price overrides (symbol: price). Defaults to None.

    Returns:
        Tuple[Optional[Dict[str, Any]], Optional[pd.DataFrame], Optional[Dict[str, Dict[str, float]]], Set[int], Dict[int, str], str]:
            - overall_summary_metrics (Optional[Dict[str, Any]]): Dictionary of overall portfolio metrics.
            - summary_df_final (Optional[pd.DataFrame]]): DataFrame containing detailed summary rows.
            - account_level_metrics (Optional[Dict[str, Dict[str, float]]]): Dictionary of metrics aggregated per account.
            - combined_ignored_indices (Set[int]): Set of 'original_index' values from rows skipped during load or processing.
            - combined_ignored_reasons (Dict[int, str]): Maps 'original_index' to a string describing the reason the row was ignored.
            - final_status (str): Overall status message indicating success, warnings, or errors.
    """
    # logging.info("--- Starting Price Unadjustment ---")
    unadjusted_prices_yf = {}
    unadjusted_count = 0

    for yf_symbol, adj_price_df in adjusted_prices_yf.items():
        # --- DEBUG FLAG ---
        IS_DEBUG_SYMBOL = yf_symbol == "AAPL"
        


        # --- Handle Empty/Invalid Input DataFrame ---
        col_to_use = None
        if not adj_price_df.empty:
            if "price" in adj_price_df.columns: col_to_use = "price"
            elif "Adj Close" in adj_price_df.columns: col_to_use = "Adj Close"
            elif "Close" in adj_price_df.columns: col_to_use = "Close"
        
        if not col_to_use:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    "    Skipping: Adjusted price DataFrame is empty or missing 'price'/'Adj Close'."
                )
            unadjusted_prices_yf[yf_symbol] = (
                adj_price_df.copy()
            )  # Return copy of input
            continue
        
        # Renaming for internal consistency within this function
        # We will work with a 'price' column locally
        adj_price_df_working = adj_price_df.copy()
        if col_to_use != "price":
            adj_price_df_working["price"] = adj_price_df_working[col_to_use]
        
        # --- END Handle Empty/Invalid Input ---

        # --- Get Splits ---
        internal_symbol = yf_to_internal_map.get(yf_symbol)
        symbol_splits = (
            splits_by_internal_symbol.get(internal_symbol) if internal_symbol else None
        )

        # --- Handle No Splits ---
        if not symbol_splits:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    f"    No splits found for internal symbol '{internal_symbol}'. Copying adjusted prices."
                )
            unadjusted_prices_yf[yf_symbol] = (
                adj_price_df_working.copy()
            )  # Return copy of input
            continue
        # --- END Handle No Splits ---
        else:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    f"    Found splits for '{internal_symbol}'. Proceeding with unadjustment."
                )

        # --- Prepare DataFrame for Unadjustment ---
        unadj_df = adj_price_df_working.copy()
        # FIX: Ensure index is UTC Timestamp for precise comparison and preservation
        unadj_df.index = pd.to_datetime(unadj_df.index, utc=True)

        if unadj_df.empty:  # Check if empty after index conversion/filtering
            unadjusted_prices_yf[yf_symbol] = unadj_df
            continue
        unadj_df.sort_index(inplace=True)  # Ensure chronological order

        # --- Calculate Forward Split Factor ---
        forward_split_factor = pd.Series(1.0, index=unadj_df.index, dtype=float)
        sorted_splits_desc = sorted(
            symbol_splits, key=lambda x: x.get("Date", date.min), reverse=True
        )
        if IS_DEBUG_SYMBOL:
            logging.debug(f"    Splits (newest first): {sorted_splits_desc}")

        # --- Process Splits (Robust Loop) ---
        for split_info in sorted_splits_desc:
            try:
                split_date_raw = split_info.get("Date")
                split_ratio = float(split_info["Split Ratio"])  # Convert to float early
                if split_date_raw is None:
                    raise ValueError("Split info missing 'Date'")
                # Convert split date to date object
                split_date = (
                    split_date_raw.date()
                    if isinstance(split_date_raw, datetime)
                    else (split_date_raw if isinstance(split_date_raw, date) else None)
                )
                if split_date is None:
                    raise TypeError(f"Invalid split date type: {type(split_date_raw)}")
                # Validate split ratio
                if split_ratio <= 0:
                    if IS_DEBUG_SYMBOL:
                        logging.warning(
                            f"    Invalid split ratio {split_ratio} on {split_date}. Skipping."
                        )
                    continue  # Skip this invalid split

                # FIX: Convert split_date to UTC Timestamp for comparison
                split_ts = pd.Timestamp(split_date).tz_localize('UTC')
                mask = forward_split_factor.index < split_ts
                if IS_DEBUG_SYMBOL and split_date == date(
                    2020, 8, 31
                ):  # Example debug date
                    logging.debug(
                        f"    Applying ratio {split_ratio} for split on {split_date}. Mask sum (dates before): {mask.sum()}"
                    )
                forward_split_factor.loc[mask] *= split_ratio

            except (KeyError, ValueError, TypeError, AttributeError) as e:
                # Catch specific errors related to bad split data format
                warn_key = (
                    f"unadjust_split_proc_{yf_symbol}_{split_info.get('Date', 'N/A')}"
                )
                if warn_key not in processed_warnings:
                    logging.warning(
                        f"Hist WARN: Error processing split for {yf_symbol} around {split_info.get('Date', 'N/A')}: {e}"
                    )
                    processed_warnings.add(warn_key)
                if IS_DEBUG_SYMBOL:
                    logging.warning(
                        f"    Error processing split around {split_info.get('Date', 'N/A')}: {e}"
                    )
                continue  # Skip this problematic split, but continue with others
        # --- END Process Splits ---

        # --- Apply Factor to Prices ---
        original_prices = unadj_df["price"].copy()
        # Align factor series with price series (handles potential missing dates)
        aligned_factor, aligned_prices = forward_split_factor.align(
            unadj_df["price"],
            join="right",
            fill_value=1.0,  # Fill missing factor dates with 1.0
        )
        unadj_df["unadjusted_price"] = aligned_prices * aligned_factor

        # --- Debug Output (Conditional) ---
        if IS_DEBUG_SYMBOL:
            # ... (debug logging remains the same) ...
            pass
        # --- END Debug Output ---

        # --- Prepare Output ---
        if not unadj_df["unadjusted_price"].equals(
            original_prices.reindex_like(unadj_df["unadjusted_price"])
        ):
            unadjusted_count += 1
        # Return DataFrame with only the unadjusted price column, renamed to 'price'
        result_df = unadj_df[["unadjusted_price"]].rename(columns={"unadjusted_price": "price"})
        unadjusted_prices_yf[yf_symbol] = result_df
        

    logging.info(
        f"--- Finished Price Unadjustment ({unadjusted_count} symbols processed with splits) ---"
    )
    return unadjusted_prices_yf


# --- Helper Functions for Point-in-Time Historical Calculation (Keep as is) ---
# _calculate_daily_net_cash_flow
# _calculate_portfolio_value_at_date_unadjusted
# _calculate_daily_metrics_worker
# (Implementations remain the same as provided previously)


# --- Helper Functions for Point-in-Time Historical Calculation ---

# --- VECTORIZED CASH FLOW CALCULATION ---
def _calculate_daily_net_cash_flow_vectorized(
    date_range: pd.DatetimeIndex,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,
    internal_to_yf_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.Series, bool]:
    """
    Vectorized calculation of daily net external cash flow (deposits/withdrawals + transfers).
    Returns a Series indexed by Date with the net flow in target_currency.
    """
    # Initialize result series (0.0 for all days)
    daily_net_flow = pd.Series(0.0, index=date_range)
    has_lookup_errors = False
    
    if transactions_df.empty:
        return daily_net_flow, False

    # Filter transactions up to the end date (Capture historical flows for 'Initial Flow')
    # FIX: Ensure transaction dates are UTC for comparison with UTC date_range
    tx_dates = pd.to_datetime(transactions_df["Date"], utc=True)
    mask_date = (tx_dates <= date_range[-1])
    df_period = transactions_df[mask_date].copy()
    
    if df_period.empty:
        return daily_net_flow, False

    # --- PROACTIVE TIMEZONE NORMALIZATION ---
    # Ensure transaction dates match the target date_range timezone EXACTLY.
    # This prevents "tz-naive vs tz-aware" errors during later aggregation.
    target_tz = date_range.tz
    
    # 1. Convert to datetime and FORCIBLY NORMALIZE to naive to avoid mixed-awareness object dtypes
    # This is critical for Transfers which often have timezones from external exports.
    df_period["Date"] = pd.to_datetime(df_period["Date"], utc=True).dt.tz_localize(None)
    
    # 2. Align Timezone (Now that it's clean and naive)
    if target_tz is not None:
        # Target is Aware
        df_period["Date"] = df_period["Date"].dt.tz_localize(target_tz)
    # else: Already naive from the normalization above

    # Normalize included accounts
    included_set = set()
    if included_accounts:
        included_set = {str(a).upper().strip() for a in included_accounts}
    
    # --- 1. CASH SYMBOL FLOWS (Deposits/Withdrawals) ---
    cash_mask = (df_period["Symbol"] == CASH_SYMBOL_CSV) & (df_period["Type"].isin(["deposit", "withdrawal"]))
    df_cash = df_period[cash_mask].copy()

    # --- FIX: Filter cash flows by included accounts ---
    if included_accounts and not df_cash.empty:
        df_cash = df_cash[
            df_cash["Account"].astype(str).str.upper().str.strip().isin(included_set)
        ]

    
    flows_list = [] # Store partial dataframes (Date, Flow_Local, Currency)

    if not df_cash.empty:
        # Vectorized calculation of local flow
        # We need to handle Commission/Qty safely
        qty = pd.to_numeric(df_cash["Quantity"], errors="coerce").fillna(0.0)
        comm = pd.to_numeric(df_cash["Commission"], errors="coerce").fillna(0.0)
        
        # 1. Deposit
        is_dep = df_cash["Type"].str.lower() == "deposit"
        # 2. Withdrawal
        is_wd = df_cash["Type"].str.lower() == "withdrawal"
        
        local_flow = pd.Series(0.0, index=df_cash.index)
        local_flow[is_dep] = qty[is_dep].abs() - comm[is_dep]
        local_flow[is_wd] = -qty[is_wd].abs() - comm[is_wd]
        
        df_cash["_flow_local"] = local_flow
        flows_list.append(df_cash[["Date", "_flow_local", "Local Currency"]])

    # --- 2. ASSET TRANSFERS (In/Out) ---
    if included_accounts:
        # Transfers are flows if they cross the boundary of "included_accounts".
        trans_mask = (df_period["Type"].str.lower().str.strip() == "transfer") & (df_period["Symbol"] != CASH_SYMBOL_CSV)
        df_trans = df_period[trans_mask].copy()
                
        if not df_trans.empty:
            # Determine direction multiplier
            src_in = df_trans["Account"].astype(str).str.upper().str.strip().isin(included_set)
            dest_in = pd.Series(False, index=df_trans.index)
            if "To Account" in df_trans.columns:
                 dest_in = df_trans["To Account"].astype(str).str.upper().str.strip().isin(included_set)
            
            # Multiplier: src_in & ~dest_in -> -1 (OUT); ~src_in & dest_in -> +1 (IN)
            multiplier = pd.Series(0.0, index=df_trans.index)
            multiplier[src_in & ~dest_in] = -1.0
            multiplier[~src_in & dest_in] = 1.0
                        
            # Filter only relevant transfers
            relevant_trans = df_trans[multiplier != 0.0].copy()
            multiplier = multiplier[multiplier != 0.0]
            
            if not relevant_trans.empty:
                # Calculate Value: Qty * Price
                qty = pd.to_numeric(relevant_trans["Quantity"], errors="coerce").fillna(0.0).abs()
                price = pd.to_numeric(relevant_trans["Price/Share"], errors="coerce").fillna(0.0)
                                
                # FALLBACK PRICE LOOKUP (Vectorized-ish)
                missing_price_mask = (price <= 1e-9)
                
                if missing_price_mask.any():
                    relevant_trans_missing = relevant_trans[missing_price_mask]
                    unique_syms = relevant_trans_missing["Symbol"].unique()
                    
                    for sym in unique_syms:
                        yf_sym = internal_to_yf_map.get(sym) if internal_to_yf_map else None
                        if yf_sym and historical_prices_yf_unadjusted and yf_sym in historical_prices_yf_unadjusted:
                             price_series = historical_prices_yf_unadjusted[yf_sym]["price"]
                             sym_mask = relevant_trans_missing["Symbol"] == sym
                             needed_dates = relevant_trans_missing.loc[sym_mask, "Date"]
                             # Lookups - distinct fix: use asof to get last available price for off-market days
                             # Ensure price_series is sorted
                             if not price_series.index.is_monotonic_increasing:
                                 price_series = price_series.sort_index()
                                 
                             # Fix 2: Ensure index is proper DatetimeIndex and NORMALIZE to naive
                             if not isinstance(price_series.index, pd.DatetimeIndex):
                                 price_series.index = pd.to_datetime(price_series.index)
                             
                             if price_series.index.tz is not None:
                                 price_series.index = price_series.index.tz_localize(None)
                             
                             # Fix: Ensure needed_dates are Timestamps to match price_series index
                             # Force both to be definitely naive to avoid "Cannot compare tz-naive and tz-aware"
                             try:
                                 # Helper to safely strip timezone
                                 def strip_tz(x):
                                     if hasattr(x, 'dt'):
                                         return x.dt.tz_localize(None) if x.dt.tz is not None else x
                                     if hasattr(x, 'tz'):
                                         return x.tz_localize(None) if x.tz is not None else x
                                     return x

                                 needed_dates_ts = strip_tz(pd.to_datetime(needed_dates)).astype("datetime64[ns]")
                                 normalized_price_index = strip_tz(pd.to_datetime(price_series.index)).astype("datetime64[ns]")
                                 
                                 # Work on a copy with naive index for asof call
                                 price_series_naive = price_series.copy()
                                 price_series_naive.index = normalized_price_index
                                 
                                 found_prices = price_series_naive.asof(needed_dates_ts)
                                 # Ensure we have a Series/Values even if asof returns something else
                                 if hasattr(found_prices, "values"):
                                     found_prices = found_prices.values

                             except Exception as e_lookup:
                                 logging.error(f"Lookup crash for {sym}: {e_lookup}")
                                 # Fallback to current behavior if shield fails, but with better error trace
                                 raise ValueError(f"Lookup failed for {sym} on dates {list(needed_dates)[:2]}: {e_lookup}")

                             # Update
                             idxs_to_update = relevant_trans_missing.loc[sym_mask].index
                             price.loc[idxs_to_update] = pd.Series(found_prices, index=idxs_to_update).fillna(0.0)

                value = qty * price
                local_flow = value * multiplier
                
                relevant_trans["_flow_local"] = local_flow
                flows_list.append(relevant_trans[["Date", "_flow_local", "Local Currency"]])

    if not flows_list:
        return daily_net_flow, False

    # --- 3. CONSOLIDATE AND FX CONVERT ---
    all_flows = pd.concat(flows_list)
    flows_grouped = all_flows.groupby(["Date", "Local Currency"])["_flow_local"].sum().reset_index()
    
    unique_currs = flows_grouped["Local Currency"].unique()
    final_flows = []
    
    for curr in unique_currs:
        mask_c = flows_grouped["Local Currency"] == curr
        sub_df = flows_grouped[mask_c].copy()
        
        if curr == target_currency:
             final_flows.append(sub_df.set_index("Date")["_flow_local"])
        else:
             # FIX: Robust Vectorized Lookup using Continuous Series
             # Instead of picking single points which might fail or differ from valuation,
             # we construct the full cross-rate series for the date_range and lookup.
             
             # 1. Build Target Rate Series (Target/USD)
             target_rate_s = None
             if target_currency.upper() == "USD":
                 target_rate_s = pd.Series(1.0, index=date_range)
             else:
                 t_pair = f"{target_currency.upper()}=X"
                 if t_pair in historical_fx_yf:
                      t_df = historical_fx_yf[t_pair]
                      if not t_df.empty and "price" in t_df.columns:
                           t_s = t_df["price"]
                           t_s.index = pd.to_datetime(t_s.index, utc=True)
                           # AGGRESSIVE BACKFILL
                           target_rate_s = t_s.reindex(date_range).ffill().bfill()
             if target_rate_s is None:
                  target_rate_s = pd.Series(np.nan, index=date_range)

             # 2. Build Local Rate Series (Local/USD)
             local_rate_s = None
             if curr.upper() == "USD":
                 local_rate_s = pd.Series(1.0, index=date_range)
             else:
                 l_pair = f"{curr.upper()}=X"
                 if l_pair in historical_fx_yf:
                      l_df = historical_fx_yf[l_pair]
                      if not l_df.empty and "price" in l_df.columns:
                           l_s = l_df["price"]
                           l_s.index = pd.to_datetime(l_s.index, utc=True)
                           # AGGRESSIVE BACKFILL
                           local_rate_s = l_s.reindex(date_range).ffill().bfill()
             if local_rate_s is None:
                  local_rate_s = pd.Series(np.nan, index=date_range)
             
             # 3. Compute Cross Rate Series
             with np.errstate(divide='ignore', invalid='ignore'):
                 cross_rate_s = target_rate_s / local_rate_s
                 # Fill any remaining NaNs (if one series was totally missing)
                 if cross_rate_s.isna().any():
                      cross_rate_s = cross_rate_s.ffill().bfill()
                      # Final fallback to 1.0 ONLY if totally empty (catastrophic)
                      cross_rate_s = cross_rate_s.fillna(1.0)

             # 4. Map transactions to this series
             # Transactions are "Date" (naive or aware). 
             # Series is date_range (likely aware UTC).
             # We need to map safely. index of cross_rate_s is DatetimeIndex.
             
             # Create a lookup map: Date(date) -> Rate
             # This is fast enough for <10k days.
             daily_rate_map = pd.Series(cross_rate_s.values, index=cross_rate_s.index.date).to_dict()
             
             factors = sub_df["Date"].dt.date.map(daily_rate_map)
             
             # Check for unmapped?
             if factors.isna().any():
                 # Should not happen with bfilled series unless date out of range?
                 # If date out of range (snapping might put it inside range?), fallback to series mean or 1.0
                 factors = factors.fillna(1.0) 
                 has_lookup_errors = True

             converted = sub_df["_flow_local"] * factors
             converted.index = sub_df["Date"]
             final_flows.append(converted)

    if final_flows:
        total_series = pd.concat(final_flows)
        daily_net_flow_add = total_series.groupby(level=0).sum()
        
        # DYNAMIC TIMEZONE ALIGNMENT
        # Align `daily_net_flow_add` (derived from transactions, originally naive) 
        # to match `daily_net_flow` (which comes from date_range, typically UTC).
        if isinstance(daily_net_flow.index, pd.DatetimeIndex):
            target_tz = daily_net_flow.index.tz
            
            if isinstance(daily_net_flow_add.index, pd.DatetimeIndex):
                source_tz = daily_net_flow_add.index.tz
                
                if target_tz is not None:
                    # Target is Aware (e.g. UTC)
                    if source_tz is None:
                        # Source is Naive -> Localize to Target
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_localize(target_tz)
                    else:
                        # Source is Aware -> Convert to Target
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_convert(target_tz)
                else:
                    # Target is Naive
                    if source_tz is not None:
                        # Source is Aware -> Make Naive
                        daily_net_flow_add.index = daily_net_flow_add.index.tz_localize(None)

        # SNAPPING: Aggregate any transactions occurring BEFORE the start date into a single 'Initial Flow' 
        # slot on the first day of the range. This ensures Cumulative Net Flow (COST) is accurate 
        # and prevents 0-base TWR spikes.
        start_ts = date_range[0]
        pre_range_mask = daily_net_flow_add.index < start_ts
        if pre_range_mask.any():
            initial_flow_total = daily_net_flow_add[pre_range_mask].sum()
            logging.info(f"STABILIZATION: Found {pre_range_mask.sum()} transactions before start date. Adding Initial Flow of {initial_flow_total:.2f} to {start_ts}")
            
            # Remove the old entries and add back to the first slot
            daily_net_flow_add = daily_net_flow_add[~pre_range_mask]
            if start_ts in daily_net_flow_add.index:
                daily_net_flow_add[start_ts] += initial_flow_total
            else:
                daily_net_flow_add[start_ts] = initial_flow_total
             
        daily_net_flow = daily_net_flow.add(daily_net_flow_add, fill_value=0.0)
    
    return daily_net_flow, has_lookup_errors


def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,  # ADDED
    internal_to_yf_map: Optional[Dict[str, str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the net external cash flow for a specific date in the target currency.
    Now includes ASSET TRANSFERS in/out of the included accounts as flows.
    """
    # --- 1. Filter Transactions for the Date ---
    # Filter for transactions on this specific date
    daily_tx = transactions_df[transactions_df["Date"].dt.date == target_date]
    
    if daily_tx.empty:
        return 0.0, False

    fx_lookup_failed = False
    net_flow_target_curr = 0.0
    
    # 1. External Flows (Deposit/Withdrawal)
    external_flow_types = ["deposit", "withdrawal"]
    
    # Capture ALL deposits and withdrawals, now including non-cash assets
    external_flow_tx = daily_tx[
        (daily_tx["Type"].str.lower().str.strip().isin(external_flow_types))
    ].copy()

    # Process External Flows
    if not external_flow_tx.empty:
        for _, row in external_flow_tx.iterrows():
            tx_type = row["Type"].lower().strip()
            symbol = row["Symbol"]
            is_cash = is_cash_symbol(symbol)
            qty = pd.to_numeric(row["Quantity"], errors="coerce")
            commission_local_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission_local = (
                0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
            )
            local_currency = row["Local Currency"]
            flow_local = 0.0

            if pd.isna(qty):
                continue

            if is_cash:
                if tx_type == "deposit":
                    flow_local = abs(qty) - commission_local
                elif tx_type == "withdrawal":
                    flow_local = -abs(qty) - commission_local
            else:
                # Non-cash Asset Contribution/Removal (e.g., Stock Deposit)
                price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
                
                # If price is missing or zero, try market price lookup
                if pd.isna(price_local) or price_local <= 0:
                    if historical_prices_yf_unadjusted and internal_to_yf_map:
                        yf_ticker = internal_to_yf_map.get(symbol)
                        if yf_ticker and yf_ticker in historical_prices_yf_unadjusted:
                            prices_df = historical_prices_yf_unadjusted[yf_ticker]
                            # Try exact date match first
                            if target_date in prices_df.index:
                                for col_name in ["Close", "price", "Adj Close"]:
                                    if col_name in prices_df.columns:
                                        price_local = prices_df.loc[target_date, col_name]
                                        break
                                else:
                                    if not prices_df.empty:
                                        price_local = prices_df.loc[target_date].iloc[0]
                            else:
                                # Try converting to timestamp
                                try:
                                    target_ts = pd.Timestamp(target_date)
                                    if target_ts in prices_df.index:
                                        for col_name in ["Close", "price", "Adj Close"]:
                                            if col_name in prices_df.columns:
                                                price_local = prices_df.loc[target_ts, col_name]
                                                break
                                        else:
                                            if not prices_df.empty:
                                                price_local = prices_df.loc[target_ts].iloc[0]
                                except:
                                    pass

                if not pd.isna(price_local) and price_local > 0:
                    if tx_type == "deposit":
                        flow_local = (abs(qty) * price_local) - commission_local
                    elif tx_type == "withdrawal":
                        flow_local = -(abs(qty) * price_local) - commission_local
                else:
                    logging.warning(f"Flow: Could not determine price for asset flow {symbol} on {target_date}. Skipping.")
                    continue

            flow_target = flow_local
            if local_currency != target_currency:
                fx_rate = get_historical_rate_via_usd_bridge(
                    local_currency, target_currency, target_date, historical_fx_yf
                )
                if pd.isna(fx_rate):
                    logging.warning(
                        f"Hist FX Lookup CRITICAL Failure for cash flow: {local_currency}->{target_currency} on {target_date}. Aborting day's flow."
                    )
                    fx_lookup_failed = True
                    net_flow_target_curr = np.nan
                    break
                else:
                    flow_target = flow_local * fx_rate

            if pd.isna(net_flow_target_curr):
                break
            if pd.notna(flow_target):
                net_flow_target_curr += flow_target
            else:
                net_flow_target_curr = np.nan
                fx_lookup_failed = True
                break

    if fx_lookup_failed:
        return np.nan, True

    # 2. Asset Transfer Flows (Transfer IN/OUT)
    if included_accounts:
        # Normalize included_accounts to uppercase/stripped
        included_set = {acc.strip().upper() for acc in included_accounts}
        transfer_tx = daily_tx[daily_tx["Type"].str.lower().str.strip() == "transfer"].copy()
        
        for _, row in transfer_tx.iterrows():
            symbol = row["Symbol"]
            if is_cash_symbol(symbol): # Skip cash transfers (handled by cash balance logic usually, or ignored if internal)
                 continue
                 
            account = row["Account"]
            to_account = row.get("To Account")
            qty = pd.to_numeric(row["Quantity"], errors="coerce")
            price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
            local_currency = row["Local Currency"]
            
            if pd.isna(qty):
                continue

            # FIX: If price is missing or 0, try to look up market price
            if pd.isna(price_local) or price_local <= 0:
                if historical_prices_yf_unadjusted and internal_to_yf_map:
                    yf_ticker = internal_to_yf_map.get(symbol)
                    if yf_ticker and yf_ticker in historical_prices_yf_unadjusted:
                        prices_df = historical_prices_yf_unadjusted[yf_ticker]
                        
                        # Look for price on target_date
                        # Try exact match first
                        if target_date in prices_df.index:
                            if "Close" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "Close"]
                            elif "price" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "price"]
                            elif "Adj Close" in prices_df.columns:
                                price_local = prices_df.loc[target_date, "Adj Close"]
                            elif not prices_df.empty:
                                # Fallback to first column if standard names missing
                                price_local = prices_df.loc[target_date].iloc[0]
                        else:
                            # Try converting target_date to datetime if index is datetime
                            try:
                                target_ts = pd.Timestamp(target_date)
                                if target_ts in prices_df.index:
                                    if "Close" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "Close"]
                                    elif "price" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "price"]
                                    elif "Adj Close" in prices_df.columns:
                                        price_local = prices_df.loc[target_ts, "Adj Close"]
                                    elif not prices_df.empty:
                                        price_local = prices_df.loc[target_ts].iloc[0]
                            except:
                                pass
                                
            if pd.isna(price_local) or price_local <= 0:
                continue
                
            # Normalize account names from transaction
            account_norm = str(account).strip().upper()
            to_account_norm = str(to_account).strip().upper() if to_account else None
            
            is_from_included = account_norm in included_set
            is_to_included = to_account_norm in included_set if to_account_norm else False
            
            flow_val_local = 0.0
            if is_from_included and not is_to_included:
                # Transfer OUT (Withdrawal)
                flow_val_local = -(abs(qty) * price_local)
            elif not is_from_included and is_to_included:
                # Transfer IN (Deposit)
                flow_val_local = abs(qty) * price_local
            
            if abs(flow_val_local) > 1e-9:
                flow_target = flow_val_local
                if local_currency != target_currency:
                    fx_rate = get_historical_rate_via_usd_bridge(
                        local_currency, target_currency, target_date, historical_fx_yf
                    )
                    if pd.isna(fx_rate):
                        fx_lookup_failed = True
                        net_flow_target_curr = np.nan
                        break
                    flow_target = flow_val_local * fx_rate
                
                if pd.notna(flow_target):
                    net_flow_target_curr += flow_target

    return net_flow_target_curr, fx_lookup_failed


def _calculate_portfolio_value_at_date_unadjusted_python(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    processed_warnings: set,
    included_accounts: Optional[List[str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices (Pure Python version).

    This function simulates the portfolio state up to the `target_date` by processing
    all transactions (buys, sells, splits, shorts, cash movements) chronologically.
    It determines the quantity of each holding (stocks and cash) per account.
    It then uses the derived *unadjusted* historical stock prices and historical FX rates
    for the `target_date` to calculate the market value of each position in the
    `target_currency`. Cash is valued at 1.0 in its local currency. Includes fallback
    logic using the last known transaction price if historical price is unavailable.

    Args:
        target_date (date): The date for which to calculate the portfolio value.
        transactions_df (pd.DataFrame): DataFrame containing all cleaned transactions.
        historical_prices_yf_unadjusted (Dict[str, pd.DataFrame]): Dictionary mapping YF tickers
            to DataFrames containing derived *unadjusted* historical prices.
        historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
            to DataFrames containing historical rates vs USD.
        target_currency (str): The currency code for the output portfolio value.
        internal_to_yf_map (Dict[str, str]): Dictionary mapping internal symbols to YF tickers.
        account_currency_map (Dict[str, str]): Mapping of account names to their local currencies.
        default_currency (str): Default currency if not found.
        manual_overrides_dict (Optional[Dict[str, Dict[str, Any]]]): Manual overrides for price, etc.
        processed_warnings (set): A set used to track and avoid logging duplicate warnings.

    Returns:
        Tuple[float, bool]:
            - total_market_value_display_curr_agg (float): The total portfolio market value
                in the target currency for the date. Returns np.nan if any critical price/FX lookup fails.
            - any_lookup_nan_on_date (bool): True if any required price or FX rate lookup failed critically.
    """
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )
    if IS_DEBUG_DATE:
        logging.debug(f"--- DEBUG VALUE CALC for {target_date} ---")
        logging.debug(f"  Target Currency: {target_currency}")
        logging.debug(f"  Included Accounts: {included_accounts}")
        logging.debug(f"  Transactions count up to date: {len(transactions_df[transactions_df['Date'].dt.date <= target_date])}")

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        if IS_DEBUG_DATE:
            logging.debug(f"  No transactions found up to {target_date}.")
        return 0.0, False

    # --- ADDED: Normalize included_accounts ---
    included_accounts_norm = set()
    if included_accounts:
        included_accounts_norm = {acc.strip().upper() for acc in included_accounts}
    # --- END ADDED ---

    # --- ADDED: Track last known prices for fallback ---
    last_known_prices: Dict[Tuple[str, str], float] = {}
    # --- END ADDED ---

    holdings: Dict[Tuple[str, str], Dict] = {}
    processed_splits: Set[Tuple[str, date, float]] = set()
    for index, row in transactions_til_date.iterrows():
        symbol = str(row.get("Symbol", "UNKNOWN")).strip()
        # Normalize account
        account_raw = str(row.get("Account", "Unknown"))
        account = account_raw.strip().upper()
        local_currency_from_row = str(row.get("Local Currency", default_currency))
        holding_key_from_row = (symbol, account)
        tx_type = str(row.get("Type", "UNKNOWN_TYPE")).lower().strip()
        tx_date_row = row["Date"].date()

        # --- ADDED: Update last known price from transaction ---
        try:
            tx_price = pd.to_numeric(row.get("Price/Share"), errors="coerce")
            if pd.notna(tx_price) and tx_price > 1e-9:
                last_known_prices[holding_key_from_row] = float(tx_price)
        except Exception:
            pass
        # --- END ADDED ---

        if symbol != CASH_SYMBOL_CSV and holding_key_from_row not in holdings:
            holdings[holding_key_from_row] = {
                "qty": 0.0,
                "local_currency": local_currency_from_row,
                "is_stock": True,
            }
        elif (
            symbol != CASH_SYMBOL_CSV
            and holdings[holding_key_from_row]["local_currency"]
            != local_currency_from_row
        ):
            holdings[holding_key_from_row]["local_currency"] = local_currency_from_row
            if IS_DEBUG_DATE:
                logging.debug(
                    f"  WARN (Value Calc): Currency overwritten for {holding_key_from_row} to {local_currency_from_row}"
                )

        if symbol == CASH_SYMBOL_CSV:
            continue

        try:
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            split_ratio = pd.to_numeric(row.get("Split Ratio"), errors="coerce")
            if tx_type in ["split", "stock split"]:
                if pd.notna(split_ratio) and split_ratio > 0:
                    split_event = (symbol, tx_date_row, float(split_ratio))
                    if split_event not in processed_splits:
                        for h_key, h_data in holdings.items():
                            h_symbol, _ = h_key
                            if h_symbol == symbol:
                                old_qty = h_data["qty"]
                                if abs(old_qty) >= 1e-9:
                                    h_data["qty"] *= split_ratio
                                    if IS_DEBUG_DATE:
                                        logging.debug(
                                            f"  Applying global split ratio {split_ratio} to {h_key} (Date: {tx_date_row}) Qty: {old_qty:.4f} -> {h_data['qty']:.4f}"
                                        )
                                    if abs(h_data["qty"]) < 1e-9:
                                        h_data["qty"] = 0.0
                        processed_splits.add(split_event)
                else:
                    if IS_DEBUG_DATE:
                        logging.warning(
                            f"  Skipping invalid split ratio ({split_ratio}) for {symbol} on {tx_date_row}"
                        )
                continue

            holding_to_update = holdings.get(holding_key_from_row)
            if not holding_to_update:
                continue

            if symbol in SHORTABLE_SYMBOLS and tx_type in [
                "short sell",
                "buy to cover",
            ]:
                if pd.isna(qty):
                    continue
                qty_abs = abs(qty)
                if tx_type == "short sell":
                    holding_to_update["qty"] -= qty_abs
                elif tx_type == "buy to cover":
                    current_short_qty_abs = (
                        abs(holding_to_update["qty"])
                        if holding_to_update["qty"] < -1e-9
                        else 0.0
                    )
                    qty_being_covered = min(qty_abs, current_short_qty_abs)
                    holding_to_update["qty"] += qty_being_covered
            elif tx_type == "buy" or tx_type == "deposit":
                if pd.notna(qty) and qty > 0:
                    holding_to_update["qty"] += qty
            elif tx_type == "sell" or tx_type == "withdrawal":
                if pd.notna(qty) and qty > 0:
                    sell_qty = qty
                    held_qty = holding_to_update["qty"]
                    qty_sold = min(sell_qty, held_qty) if held_qty > 1e-9 else 0
                    holding_to_update["qty"] -= qty_sold
            elif tx_type == "transfer":
                to_account_raw = str(row.get("To Account", ""))
                to_account = to_account_raw.strip().upper()
                if pd.notna(qty) and qty > 0:
                    transfer_qty = qty

                    # 1. Deduct from Source Account
                    holding_to_update["qty"] -= transfer_qty

                    # 2. Add to Destination Account
                    if to_account and transfer_qty > 0:
                        to_key = (symbol, to_account)
                        if to_key not in holdings:
                            holdings[to_key] = {
                                "qty": 0.0,
                                "local_currency": local_currency_from_row,
                                "is_stock": True,
                            }
                        holdings[to_key]["qty"] += transfer_qty

                        # --- ADDED: Propagate last known price to destination ---
                        if holding_key_from_row in last_known_prices:
                            last_known_prices[to_key] = last_known_prices[holding_key_from_row]
                        # --- END ADDED ---

                        # This function only calculates market value, not cost basis,
                        # so only the quantity needs to be moved. The Numba version
                        # below is where the cost basis logic is critical.
                        if IS_DEBUG_DATE:
                            logging.debug(
                                f"  Transferring {transfer_qty} of {symbol} from {account} to {to_account}"
                            )
                            logging.debug(
                                f"    New Source Qty: {holding_to_update['qty']:.4f}"
                            )
                            logging.debug(
                                f"    New Dest Qty: {holdings[to_key]['qty']:.4f}"
                            )
        except Exception as e_h:
            if IS_DEBUG_DATE:
                logging.error(
                    f"      ERROR processing holding qty for {holding_key_from_row} on row index {index}: {e_h}"
                )
            pass

    cash_summary: Dict[str, Dict] = {}
    # --- Apply STOCK_QUANTITY_CLOSE_TOLERANCE to stock holdings before valuation ---
    for holding_key_iter, data_iter in holdings.items():
        sym_iter, _ = holding_key_iter
        if sym_iter != CASH_SYMBOL_CSV:  # Only for stocks
            qty_iter = data_iter.get("qty", 0.0)
            if 0 < abs(qty_iter) < STOCK_QUANTITY_CLOSE_TOLERANCE:
                if IS_DEBUG_DATE:
                    logging.debug(
                        f"  Applying tolerance to {holding_key_iter}, qty {qty_iter} -> 0"
                    )
                data_iter["qty"] = 0.0
                # Cost basis is not explicitly tracked here for daily valuation, qty is primary
    cash_transactions = transactions_til_date[
        transactions_til_date["Symbol"] == CASH_SYMBOL_CSV
    ].copy()
    if not cash_transactions.empty:

        def get_signed_quantity_cash(row):
            """Calculates cash flow including commission impact."""
            type_lower = str(row.get("Type", "")).lower()
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            commission_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission = 0.0 if pd.isna(commission_raw) else float(commission_raw)

            return (
                0.0
                if pd.isna(qty)
                else (
                    # Deposit: Increase cash by quantity MINUS commission
                    abs(qty) - commission
                    if type_lower in ["buy", "deposit"]
                    # Withdrawal: Decrease cash by quantity PLUS commission
                    else (
                        -(abs(qty) + commission)
                        if type_lower in ["sell", "withdrawal"]
                        else 0.0
                    )
                )
            )

        cash_transactions["SignedQuantity"] = cash_transactions.apply(
            get_signed_quantity_cash, axis=1
        )
        cash_qty_agg = cash_transactions.groupby("Account")["SignedQuantity"].sum()
        cash_currency_map = cash_transactions.groupby("Account")[
            "Local Currency"
        ].first()
        all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index)
        for acc in all_cash_accounts:
            cash_summary[acc] = {
                "qty": cash_qty_agg.get(acc, 0.0),
                "local_currency": cash_currency_map.get(acc, default_currency),
                "is_stock": False,
            }

    all_positions: Dict[Tuple[str, str], Dict] = {
        **holdings,
        **{(CASH_SYMBOL_CSV, acc): data for acc, data in cash_summary.items()},
    }
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False
    if IS_DEBUG_DATE:
        logging.debug(
            f"  Value Aggregation Start - Combined Positions ({len(all_positions)}): {list(all_positions.keys())}"
        )

    for (internal_symbol, account), data in all_positions.items():
        current_qty = data.get("qty", 0.0)
        local_currency = data.get("local_currency", default_currency)
        is_stock = data.get("is_stock", internal_symbol != CASH_SYMBOL_CSV)
        DO_DETAILED_LOG = IS_DEBUG_DATE
        if DO_DETAILED_LOG:
            logging.debug(
                f"    Value Agg: Processing {internal_symbol}/{account}, Qty: {current_qty:.4f}"
            )
        if abs(current_qty) < 1e-9:
            continue
            
        # --- ADDED: Filter by included_accounts ---
        # Filter by included_accounts
        if included_accounts and account not in included_accounts_norm:
            continue
        # --- END ADDED ---

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        if DO_DETAILED_LOG:
            logging.debug(
                f"      FX Rate ({local_currency}->{target_currency}): {fx_rate}"
            )
        if pd.isna(fx_rate):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            if DO_DETAILED_LOG:
                logging.debug(f"      CRITICAL: FX lookup failed. Aborting.")
            break

        current_price_local = np.nan
        manual_price_override_applied = False

        # --- ADDED: Check for manual price override ---
        if manual_overrides_dict and internal_symbol in manual_overrides_dict:
            symbol_override_data = manual_overrides_dict[internal_symbol]
            manual_price = symbol_override_data.get("price")
            if manual_price is not None and pd.notna(manual_price):
                try:
                    manual_price_float = float(manual_price)
                    if manual_price_float > 1e-9:  # Ensure positive price
                        current_price_local = manual_price_float
                        manual_price_override_applied = True
                        if DO_DETAILED_LOG:
                            logging.debug(
                                f"      Using MANUAL OVERRIDE Price for {internal_symbol}: {current_price_local}"
                            )
                    elif DO_DETAILED_LOG:
                        logging.debug(
                            f"      Manual override price for {internal_symbol} is not positive: {manual_price_float}. Ignoring."
                        )
                except (ValueError, TypeError) as e_manual_price:
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Manual override price for {internal_symbol} ('{manual_price}') is not a valid number: {e_manual_price}. Ignoring."
                        )
        # --- END ADDED ---

        force_fallback = internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
        if (
            pd.isna(current_price_local) and not is_stock
        ):  # If no manual override and it's cash
            current_price_local = 1.0
        elif (
            pd.isna(current_price_local) and not force_fallback and is_stock
        ):  # If no manual override, not forced fallback, and is stock
            yf_symbol_for_lookup = internal_to_yf_map.get(internal_symbol)
            if yf_symbol_for_lookup:
                price_val = get_historical_price(
                    yf_symbol_for_lookup, target_date, historical_prices_yf_unadjusted
                )
                if price_val is not None and pd.notna(price_val) and price_val > 1e-9:
                    current_price_local = float(price_val)
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Using YFinance Price for {internal_symbol}: {current_price_local}"
                        )

        if (pd.isna(current_price_local) or force_fallback) and is_stock:
            # Fallback to last transaction price if still no price, or if yfinance is excluded (and no manual override was applied for it)
            
            # --- ADDED: Check last_known_prices first ---
            if (internal_symbol, account) in last_known_prices:
                last_known = last_known_prices[(internal_symbol, account)]
                if pd.notna(last_known) and last_known > 1e-9:
                    current_price_local = last_known
                    if DO_DETAILED_LOG:
                        logging.debug(
                            f"      Using Last Known Price (tracked): {current_price_local}"
                        )
            # --- END ADDED ---

            if pd.isna(current_price_local):
                try:
                    fallback_tx = transactions_df[
                        (transactions_df["Symbol"] == internal_symbol)
                        & (transactions_df["Account"] == account)
                        & (transactions_df["Price/Share"].notna())
                        & (
                            pd.to_numeric(transactions_df["Price/Share"], errors="coerce")
                            > 1e-9
                        )
                        & (transactions_df["Date"].dt.date <= target_date)
                    ].copy()
                    if not fallback_tx.empty:
                        fallback_tx.sort_values(
                            by=["Date", "original_index"], inplace=True, ascending=True
                        )
                        last_tx_row = fallback_tx.iloc[-1]
                        last_tx_price = pd.to_numeric(
                            last_tx_row["Price/Share"], errors="coerce"
                        )
                        if pd.notna(last_tx_price) and last_tx_price > 1e-9:
                            current_price_local = float(last_tx_price)
                            if DO_DETAILED_LOG:
                                logging.debug(  # This log might be hit if force_fallback=True and no manual price
                                    f"      Using Fallback Price (DF lookup): {current_price_local}"
                                )
                except Exception:
                    pass

        if pd.isna(current_price_local):
            logging.warning(
                f"Missing price for {internal_symbol} on {target_date}. Using 0.0."
            )
            current_price_local = 0.0
        else:
            if DO_DETAILED_LOG:
                logging.debug(f"      Final Local Price: {current_price_local:.4f}")

        market_value_local = current_qty * float(current_price_local)
        market_value_display = market_value_local * fx_rate
        if DO_DETAILED_LOG:
            logging.debug(
                f"      MV Local: {market_value_local:.2f}, MV Display ({target_currency}): {market_value_display:.2f}"
            )
        if pd.isna(market_value_display):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            if DO_DETAILED_LOG:
                logging.debug(f"      CRITICAL: MV Display is NaN. Aborting.")
            break
        else:
            total_market_value_display_curr_agg += market_value_display
            if DO_DETAILED_LOG:
                logging.debug(
                    f"      Running Total MV Display: {total_market_value_display_curr_agg:.2f}"
                )

    if IS_DEBUG_DATE:
        logging.debug(
            f"--- DEBUG VALUE CALC for {target_date} END --- Final Value: {total_market_value_display_curr_agg}, Lookup Failed: {any_lookup_nan_on_date}"
        )
    return total_market_value_display_curr_agg, any_lookup_nan_on_date


# --- START NUMBA HELPER FUNCTION ---
@numba.jit(nopython=True, fastmath=True, cache=True)
def _calculate_holdings_numba(
    target_date_ordinal,
    tx_dates_ordinal_np,
    tx_symbols_np,
    tx_accounts_np,
    tx_to_accounts_np,  # NEW argument
    tx_types_np,
    tx_quantities_np,
    tx_prices_np,
    tx_commissions_np,
    tx_split_ratios_np,
    tx_local_currencies_np,
    num_symbols,
    num_accounts,
    num_currencies,
    split_type_id,
    stock_split_type_id,
    buy_type_id,
    deposit_type_id,
    sell_type_id,
    withdrawal_type_id,
    short_sell_type_id,
    buy_to_cover_type_id,  # type: ignore
    transfer_type_id,  # NEW argument
    fees_type_id,
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
):
    # Initialize state arrays
    holdings_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_cost_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_currency_np = np.full((num_symbols, num_accounts), -1, dtype=np.int64)

    holdings_short_proceeds_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_short_orig_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    cash_balances_np = np.zeros(num_accounts, dtype=np.float64)
    cash_currency_np = np.full(num_accounts, -1, dtype=np.int64)
    
    # --- NEW: Track last prices ---
    last_prices_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    num_transactions = len(tx_dates_ordinal_np)

    for i in range(num_transactions):
        tx_date = tx_dates_ordinal_np[i]
        if tx_date > target_date_ordinal:
            continue

        symbol_id = tx_symbols_np[i]
        account_id = tx_accounts_np[i]
        type_id = tx_types_np[i]
        qty = tx_quantities_np[i]
        price = tx_prices_np[i]
        commission = tx_commissions_np[i]
        split_ratio = tx_split_ratios_np[i]
        currency_id = tx_local_currencies_np[i]

        # --- Handle CASH transactions ---
        if symbol_id == cash_symbol_id:
            if cash_currency_np[account_id] == -1:
                cash_currency_np[account_id] = currency_id

            if type_id == buy_type_id or type_id == deposit_type_id:
                cash_balances_np[account_id] += qty - commission
            elif type_id == sell_type_id or type_id == withdrawal_type_id:
                cash_balances_np[account_id] -= qty + commission
            elif type_id == transfer_type_id:
                dest_account_id = tx_to_accounts_np[i]
                if dest_account_id != -1:
                    # Initialize Dest Currency if needed
                    if cash_currency_np[dest_account_id] == -1:
                        cash_currency_np[dest_account_id] = currency_id

                    # Move Cash: Deduct from Source, Add to Dest
                    # Assuming commission is paid by source
                    cash_balances_np[account_id] -= qty + commission
                    cash_balances_np[dest_account_id] += qty
            continue
        # --- Handle STOCK transactions ---
        if holdings_currency_np[symbol_id, account_id] == -1:
            holdings_currency_np[symbol_id, account_id] = currency_id
            
        # --- NEW: Update Last Price ---
        if price > 1e-9:
            last_prices_np[symbol_id, account_id] = price

        # --- STOCK TRANSFER LOGIC ---
        if type_id == transfer_type_id:
            if qty > 1e-9:
                source_qty = holdings_qty_np[symbol_id, account_id]
                source_cost_basis = holdings_cost_np[symbol_id, account_id]

                # Determine the quantity to transfer. Do not cap it. Trust the transaction.
                transfer_qty = qty
                cost_to_transfer = 0.0

                # Calculate proportional cost to transfer.
                if abs(source_qty) > 1e-9:  # Avoid division by zero
                    # If transferring more than held (e.g., due to same-day buy), transfer 100% of cost.
                    proportion = min(transfer_qty / source_qty, 1.0)
                    cost_to_transfer = source_cost_basis * proportion

                # 1. Deduct from Source Account
                holdings_qty_np[symbol_id, account_id] -= transfer_qty
                holdings_cost_np[symbol_id, account_id] -= cost_to_transfer

                # Zero out if quantity becomes negligible
                if (
                    abs(holdings_qty_np[symbol_id, account_id])
                    < stock_qty_close_tolerance
                ):
                    holdings_qty_np[symbol_id, account_id] = 0.0
                    holdings_cost_np[symbol_id, account_id] = 0.0

                # 2. Add to Destination
                dest_account_id = tx_to_accounts_np[i]
                if dest_account_id != -1:
                    if holdings_currency_np[symbol_id, dest_account_id] == -1:
                        holdings_currency_np[symbol_id, dest_account_id] = currency_id
                    holdings_qty_np[symbol_id, dest_account_id] += transfer_qty
                    holdings_cost_np[symbol_id, dest_account_id] += cost_to_transfer
                    
                    # Also copy last price to destination if available
                    if last_prices_np[symbol_id, account_id] > 1e-9:
                        last_prices_np[symbol_id, dest_account_id] = last_prices_np[symbol_id, account_id]
            continue

        # --- Existing Split Logic ---
        if type_id == split_type_id or type_id == stock_split_type_id:
            if split_ratio > 1e-9:
                for acc_idx in range(num_accounts):
                    if abs(holdings_qty_np[symbol_id, acc_idx]) > 1e-9:
                        holdings_qty_np[symbol_id, acc_idx] *= split_ratio

                        # Handle Shorts
                        is_shortable = False
                        for short_id in shortable_symbol_ids:
                            if symbol_id == short_id:
                                is_shortable = True
                                break
                        if holdings_qty_np[symbol_id, acc_idx] < -1e-9 and is_shortable:
                            holdings_short_orig_qty_np[
                                symbol_id, acc_idx
                            ] *= split_ratio

            if abs(commission) > 1e-9:
                holdings_cost_np[symbol_id, account_id] += commission
            continue

        # --- Existing Shorting Logic ---
        is_shortable_flag = False
        for short_id in shortable_symbol_ids:
            if symbol_id == short_id:
                is_shortable_flag = True
                break

        if is_shortable_flag and (
            type_id == short_sell_type_id or type_id == buy_to_cover_type_id
        ):
            qty_abs = abs(qty)
            if qty_abs > 1e-9:
                if type_id == short_sell_type_id:
                    proceeds = (qty_abs * price) - commission
                    holdings_qty_np[symbol_id, account_id] -= qty_abs
                    holdings_short_proceeds_np[symbol_id, account_id] += proceeds
                    holdings_short_orig_qty_np[symbol_id, account_id] += qty_abs
                    holdings_cost_np[symbol_id, account_id] += commission
                elif type_id == buy_to_cover_type_id:
                    qty_currently_short = (
                        abs(holdings_qty_np[symbol_id, account_id])
                        if holdings_qty_np[symbol_id, account_id] < -1e-9
                        else 0.0
                    )
                    if qty_currently_short > 1e-9:
                        qty_covered = min(qty_abs, qty_currently_short)
                        cost_to_cover = (qty_covered * price) + commission
                        holdings_qty_np[symbol_id, account_id] += qty_covered
                        holdings_cost_np[symbol_id, account_id] += cost_to_cover

                        short_orig = holdings_short_orig_qty_np[symbol_id, account_id]
                        if short_orig > 1e-9:
                            ratio = qty_covered / short_orig
                            holdings_short_proceeds_np[symbol_id, account_id] *= (
                                1.0 - ratio
                            )
                            holdings_short_orig_qty_np[
                                symbol_id, account_id
                            ] -= qty_covered
            continue

        # --- Standard Buy/Sell ---
        if type_id == buy_type_id or type_id == deposit_type_id:
            if qty > 1e-9:
                cost = (qty * price) + commission
                holdings_qty_np[symbol_id, account_id] += qty
                holdings_cost_np[symbol_id, account_id] += cost
        elif type_id == sell_type_id or type_id == withdrawal_type_id:
            if qty > 1e-9:
                held_qty = holdings_qty_np[symbol_id, account_id]
                if held_qty > 1e-9:
                    qty_sold = min(qty, held_qty)
                    cost_basis_held = holdings_cost_np[symbol_id, account_id]
                    cost_sold = qty_sold * (cost_basis_held / held_qty)

                    holdings_qty_np[symbol_id, account_id] -= qty_sold
                    holdings_cost_np[symbol_id, account_id] -= cost_sold

                    if abs(holdings_qty_np[symbol_id, account_id]) < 1e-9:
                        holdings_qty_np[symbol_id, account_id] = 0.0
                        holdings_cost_np[symbol_id, account_id] = 0.0
        elif type_id == fees_type_id:
            if abs(commission) > 1e-9:
                holdings_cost_np[symbol_id, account_id] += commission

    # Apply Final Tolerance
    for s_id in range(num_symbols):
        if s_id == cash_symbol_id:
            continue
        for a_id in range(num_accounts):
            if 0 < abs(holdings_qty_np[s_id, a_id]) < stock_qty_close_tolerance:
                holdings_qty_np[s_id, a_id] = 0.0
                holdings_cost_np[s_id, a_id] = 0.0

    return (
        holdings_qty_np,
        holdings_cost_np,
        holdings_currency_np,
        cash_balances_np,
        cash_currency_np,
        last_prices_np,  # NEW return
    )


# --- END NUMBA HELPER FUNCTION ---


# --- START NEW CHRONOLOGICAL NUMBA HELPER ---
@profile
@numba.jit(nopython=True, fastmath=True, cache=True)
def _calculate_daily_holdings_chronological_numba(
    date_ordinals_np,
    tx_dates_ordinal_np,
    tx_symbols_np,  # type: ignore
    tx_to_accounts_np,  # NEW argument
    tx_accounts_np,
    tx_types_np,
    tx_quantities_np,
    tx_commissions_np,
    tx_split_ratios_np,
    tx_prices_np,  # NEW argument
    num_symbols,
    num_accounts,
    split_type_id,
    stock_split_type_id,
    buy_type_id,
    deposit_type_id,
    sell_type_id,
    withdrawal_type_id,
    short_sell_type_id,
    buy_to_cover_type_id,
    transfer_type_id,  # NEW argument
    cash_symbol_id,
    stock_qty_close_tolerance,
    shortable_symbol_ids,
):
    """
    Calculates holdings and cash balances chronologically for each day in the target range.
    This is much more efficient than recalculating from scratch each day.
    """
    num_days = len(date_ordinals_np)
    # Initialize daily result arrays
    daily_holdings_qty_np = np.zeros(
        (num_days, num_symbols, num_accounts), dtype=np.float64
    )
    daily_cash_balances_np = np.zeros((num_days, num_accounts), dtype=np.float64)
    # --- NEW: Track last transaction prices for fallback ---
    daily_last_prices_np = np.zeros(
        (num_days, num_symbols, num_accounts), dtype=np.float64
    )

    # Initialize current state
    current_holdings_qty = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    current_cash_balances = np.zeros(num_accounts, dtype=np.float64)
    current_last_prices = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    tx_idx = 0
    num_transactions = len(tx_dates_ordinal_np)

    for day_idx in range(num_days):
        current_date_ordinal = date_ordinals_np[day_idx]

        # Process all transactions for this day
        while (
            tx_idx < num_transactions
            and tx_dates_ordinal_np[tx_idx] <= current_date_ordinal
        ):
            symbol_id = tx_symbols_np[tx_idx]
            account_id = tx_accounts_np[tx_idx]
            type_id = tx_types_np[tx_idx]
            qty = tx_quantities_np[tx_idx]
            commission = tx_commissions_np[tx_idx]
            split_ratio = tx_split_ratios_np[tx_idx]
            # --- NEW: Get price for fallback ---
            price = tx_prices_np[tx_idx]

            if symbol_id == cash_symbol_id:
                if type_id == buy_type_id or type_id == deposit_type_id:
                    current_cash_balances[account_id] += abs(qty) - commission
                elif type_id == sell_type_id or type_id == withdrawal_type_id:
                    current_cash_balances[account_id] -= abs(qty) + commission
                elif type_id == transfer_type_id:
                    dest_account_id = tx_to_accounts_np[tx_idx]
                    if dest_account_id != -1:
                        # Move Cash: Deduct from Source, Add to Dest
                        # Assuming commission is paid by source
                        current_cash_balances[account_id] -= abs(qty) + commission
                        current_cash_balances[dest_account_id] += abs(qty)

            else:
                # --- NEW: Update Last Price ---
                if price > 1e-9:
                    current_last_prices[symbol_id, account_id] = price

                if type_id == split_type_id or type_id == stock_split_type_id:
                    if split_ratio > 1e-9:
                        for acc_idx in range(num_accounts):
                            if abs(current_holdings_qty[symbol_id, acc_idx]) > 1e-9:
                                current_holdings_qty[symbol_id, acc_idx] *= split_ratio
                elif type_id == buy_type_id or type_id == deposit_type_id:
                    if qty > 1e-9:
                        current_holdings_qty[symbol_id, account_id] += qty

                elif type_id == sell_type_id or type_id == withdrawal_type_id:
                    if qty > 1e-9:
                        held_qty = current_holdings_qty[symbol_id, account_id]
                        qty_sold = min(qty, held_qty) if held_qty > 1e-9 else 0.0
                        current_holdings_qty[symbol_id, account_id] -= qty_sold

                elif type_id == transfer_type_id:
                    if qty > 1e-9:
                        transfer_qty = qty

                        # 1. Deduct from Source Account
                        current_holdings_qty[symbol_id, account_id] -= transfer_qty

                        # 2. Add to Destination Account
                        dest_account_id = tx_to_accounts_np[tx_idx]
                        if dest_account_id != -1:
                            current_holdings_qty[
                                symbol_id, dest_account_id
                            ] += transfer_qty
                            # Also copy last price to destination if available
                            if current_last_prices[symbol_id, account_id] > 1e-9:
                                current_last_prices[symbol_id, dest_account_id] = current_last_prices[symbol_id, account_id]

                        # Note: This function only tracks quantity, not cost basis.
                        # The cost basis transfer is handled in _calculate_holdings_numba.
                else:
                    is_shortable = False
                    for short_id in shortable_symbol_ids:
                        if symbol_id == short_id:
                            is_shortable = True
                            break
                    if is_shortable:
                        if type_id == short_sell_type_id:
                            current_holdings_qty[symbol_id, account_id] -= abs(qty)
                        elif type_id == buy_to_cover_type_id:
                            qty_currently_short = (
                                abs(current_holdings_qty[symbol_id, account_id])
                                if current_holdings_qty[symbol_id, account_id] < -1e-9
                                else 0.0
                            )
                            qty_covered = min(abs(qty), qty_currently_short)
                            current_holdings_qty[symbol_id, account_id] += qty_covered
            
            tx_idx += 1

        for s_id in range(num_symbols):
            if s_id == cash_symbol_id:
                continue
            for a_id in range(num_accounts):
                qty_val = current_holdings_qty[s_id, a_id]
                if 0 < abs(qty_val) < stock_qty_close_tolerance:
                    current_holdings_qty[s_id, a_id] = 0.0

        daily_holdings_qty_np[day_idx] = current_holdings_qty
        daily_cash_balances_np[day_idx] = current_cash_balances
        # --- NEW: Store daily last prices ---
        daily_last_prices_np[day_idx] = current_last_prices

    return daily_holdings_qty_np, daily_cash_balances_np, daily_last_prices_np


@profile
def _calculate_portfolio_value_at_date_unadjusted_numba(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],
    processed_warnings: set,
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    included_accounts: Optional[List[str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices (Numba version).
    """
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        return 0.0, False

    # --- FIX: Sort transactions to ensure chronological processing ---
    transactions_til_date.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )
    
    # --- ADDED: Support Intraday comparison ---
    # If target_date is a full datetime (and not just midnight), use exact comparison
    mask = None
    if isinstance(target_date, datetime):
        # Check if it has non-zero time or if we are in intraday mode (context dependent)
        # But safest is: if it's a datetime, use full comparison.
        # However, for legacy daily calls, target_date might be datetime(2023,1,1,0,0) but we want all day?
        # Actually daily logic usually passes date() objects.
        # Intraday logic passes datetime() objects.
        mask = transactions_df["Date"] <= target_date
    else:
        # Fallback for date objects (Legacy Daily)
        mask = transactions_df["Date"].dt.date <= target_date
        
    transactions_til_date = transactions_df[mask].copy()
    if transactions_til_date.empty:
        return 0.0, False
        
    # Re-sort again just in case (though filtered subset should preserve order)
    transactions_til_date.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )
    # --- Prepare NumPy Inputs (Step 4: Update Data Prep) ---
    # --- ADDED: Prepare included_account_ids set ---
    included_account_ids = set()
    if included_accounts:
        # Normalize included_accounts to match account_to_id keys (which are normalized in generate_mappings)
        normalized_included = [acc.strip().upper() for acc in included_accounts]
        for acc in normalized_included:
            if acc in account_to_id:
                included_account_ids.add(account_to_id[acc])
    # --- END ADDED ---

    try:
        target_date_ts = pd.Timestamp(target_date)
        if target_date_ts.tz is None: target_date_ts = target_date_ts.tz_localize('UTC')
        target_date_ordinal = target_date_ts.value
        
        # --- FIX: Define tx_types_np earlier for use in debug block ---
        tx_types_series = (
            transactions_til_date["Type"]
            .str.lower()
            .str.strip()
            .map(type_to_id)
            .fillna(-1)
        )
        tx_types_np = tx_types_series.values.astype(np.int64)

        tx_dates_ordinal_np = np.array(pd.to_datetime(transactions_til_date["Date"], utc=True).values.astype('int64'), dtype=np.int64)
        tx_symbols_series = _normalize_series(transactions_til_date["Symbol"]).map(
            symbol_to_id
        )
        tx_symbols_np = tx_symbols_series.fillna(-1).values.astype(np.int64)

        tx_accounts_series = _normalize_series(transactions_til_date["Account"]).map(
            account_to_id
        )
        tx_accounts_np = tx_accounts_series.fillna(-1).values.astype(np.int64)

        # --- ADDED: Map 'To Account' for transfers ---
        if "To Account" in transactions_til_date.columns:
            # Map 'To Account' to IDs, filling NaN with -1
            tx_to_accounts_series = _normalize_series(
                transactions_til_date["To Account"]
            ).map(account_to_id)
        else:
            tx_to_accounts_series = pd.Series(-1, index=transactions_til_date.index)
        tx_to_accounts_np = tx_to_accounts_series.fillna(-1).values.astype(np.int64)
    
        # --- END ADDED ---

        # --- DEBUG BLOCK 2: Check Mapping ---

        # 1. Check ID Map for specific target
        target_acc = "IBKR Acct. 1".upper().strip()
        logging.debug(
            f"Direct ID lookup for '{target_acc}': {account_to_id.get(target_acc, 'NOT FOUND')}"
        )

        # 2. Dump all account keys to check for whitespace issues
        logging.debug("All Account Keys in Map:")
        for k, v in account_to_id.items():
            logging.debug(f"  '{k}' -> {v}")

        # 3. Check Transfer IDs in Arrays
        transfer_id = type_to_id.get("transfer")
        logging.debug(f"Transfer Type ID: {transfer_id}")
        if transfer_id is not None:
            transfer_indices = np.where(tx_types_np == transfer_id)[0]

            if len(transfer_indices) > 0:
                logging.debug(
                    f"Found {len(transfer_indices)} transfers in NumPy arrays."
                )
                for idx in transfer_indices:
                    src_id = tx_accounts_np[idx]
                    dst_id = tx_to_accounts_np[idx]

                    src_name = id_to_account.get(src_id, "UNKNOWN_ID")
                    dst_name = id_to_account.get(dst_id, "UNKNOWN_ID")

                    logging.debug(
                        f"  Tx Index {idx}: From ID {src_id} ('{src_name}') -> To ID {dst_id} ('{dst_name}')"
                    )

                    if dst_id == -1:
                        logging.debug(
                            "  CRITICAL: Destination ID is -1. The Numba engine will IGNORE this transfer."
                        )
            else:
                logging.debug(
                    "CRITICAL: No transfers found in NumPy arrays (tx_types_np)."
                )
        logging.debug("----------------------------\n")
        # --- END ADDED ---

        tx_quantities_np = (
            transactions_til_date["Quantity"].fillna(0.0).values.astype(np.float64)
        )
        tx_prices_np = (
            transactions_til_date["Price/Share"].fillna(0.0).values.astype(np.float64)
        )
        tx_commissions_series = transactions_til_date["Commission"].fillna(0.0)
        tx_commissions_np = tx_commissions_series.values.astype(np.float64)
        tx_split_ratios_series = transactions_til_date["Split Ratio"].fillna(0.0)
        tx_split_ratios_np = tx_split_ratios_series.values.astype(np.float64)
        tx_local_currencies_series = (
            transactions_til_date["Local Currency"].map(currency_to_id).fillna(-1)
        )
        tx_local_currencies_np = tx_local_currencies_series.values.astype(np.int64)

        split_type_id = type_to_id.get("split", -1)
        stock_split_type_id = type_to_id.get("stock split", -1)
        buy_type_id = type_to_id.get("buy", -1)
        deposit_type_id = type_to_id.get("deposit", -1)
        sell_type_id = type_to_id.get("sell", -1)
        withdrawal_type_id = type_to_id.get("withdrawal", -1)
        short_sell_type_id = type_to_id.get("short sell", -1)
        buy_to_cover_type_id = type_to_id.get("buy to cover", -1)
        transfer_type_id = type_to_id.get("transfer", -1)  # ADDED
        fees_type_id = type_to_id.get("fees", -1)
        cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

        shortable_symbol_ids = np.array(
            [symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id],
            dtype=np.int64,
        )
        num_symbols = len(symbol_to_id)
        num_accounts = len(account_to_id)
        num_currencies = len(currency_to_id)

    except Exception as e_np_prep:
        logging.error(f"Numba Prep Error for {target_date}: {e_np_prep}")
        return np.nan, True

    # --- DEBUG NUMBA INPUTS ---
    
    # --- Call Numba Helper ---
    try:
        (
            holdings_qty_np,
            holdings_cost_np,
            holdings_currency_np,
            cash_balances_np,
            cash_currency_np,
            last_prices_np,  # NEW return
        ) = _calculate_holdings_numba(
            target_date_ordinal,
            tx_dates_ordinal_np,
            tx_symbols_np,
            tx_accounts_np,
            tx_to_accounts_np,  # Pass to Numba function
            tx_types_np,
            tx_quantities_np,
            tx_prices_np,
            tx_commissions_np,
            tx_split_ratios_np,
            tx_local_currencies_np,
            num_symbols,
            num_accounts,
            num_currencies,
            split_type_id,
            stock_split_type_id,
            buy_type_id,
            deposit_type_id,
            sell_type_id,
            withdrawal_type_id,
            short_sell_type_id,
            buy_to_cover_type_id,
            transfer_type_id,  # Pass to Numba function
            fees_type_id,
            cash_symbol_id,
            STOCK_QUANTITY_CLOSE_TOLERANCE,
            shortable_symbol_ids,
        )
        
    except Exception as e_numba_call:
        logging.error(f"Numba Call Error for {target_date}: {e_numba_call}")
        return np.nan, True

    # --- Valuation Loop (using results from Numba) ---
    # [The rest of the valuation loop logic remains identical to the existing file]
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False

    # Iterate through stock holdings
    stock_indices = np.argwhere(np.abs(holdings_qty_np) > STOCK_QUANTITY_CLOSE_TOLERANCE)
    
    for idx_tuple in stock_indices:
        symbol_id = idx_tuple[0]
        account_id = idx_tuple[1]
        
        # --- ADDED: Filter by included_accounts ---
        if included_accounts and account_id not in included_account_ids:
            continue
        # --- END ADDED ---

        current_qty = holdings_qty_np[symbol_id, account_id]
        last_price = last_prices_np[symbol_id, account_id]
        
        internal_symbol = id_to_symbol[symbol_id]
        account = id_to_account[account_id]
        
        # 3. Get Price
        current_price_local = np.nan
        try:
            # Correctly map internal symbol to YF symbol first
            yf_symbol = internal_to_yf_map.get(internal_symbol, internal_symbol)
            current_price_local = get_historical_price(
                yf_symbol,
                target_date,
                historical_prices_yf_unadjusted,
            )
        except Exception:
            pass
            
        # --- NEW: Fallback to last transaction price ---
        if pd.isna(current_price_local):
            if last_price > 1e-9:
                current_price_local = last_price

        currency_id = holdings_currency_np[symbol_id, account_id]
        local_currency = id_to_currency.get(currency_id, default_currency)

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        if pd.isna(fx_rate):
            # Hist Fix: Default to 1.0 if missing FX
            fx_rate = 1.0
            # any_lookup_nan_on_date = True
            # total_market_value_display_curr_agg = np.nan
            # break
            
        # Calculate local market value
        market_value_local = current_qty * current_price_local
        
        market_value_display = market_value_local * fx_rate

        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Stock {internal_symbol}/{account}, Qty: {current_qty}, Price: {current_price_local}, MV: {market_value_display}")

        if pd.isna(market_value_display):
            if current_qty == 0:
                # Zero qty should not cause NaN
                market_value_display = 0.0
            else:
                 any_lookup_nan_on_date = True
        else:
            total_market_value_display_curr_agg += market_value_display

    if IS_DEBUG_DATE:
        logging.debug(f"    Numba Val Agg: Pre-Cash Total: {total_market_value_display_curr_agg}")

    # --- ADDED: Aggregate Cash Balances from Numba ---
    # Note: If any_lookup_nan_on_date is true from stocks, we might still want to sum cash?
    # Original logic breaks early. Keep consistent?
    # If we patched stock FX, we likely won't break early.
    
    cash_indices = np.argwhere(np.abs(cash_balances_np) > 1e-9)
    for acc_id_tuple in cash_indices:
        acc_id = acc_id_tuple[0]
        
        # --- ADDED: Filter by included_accounts ---
        if included_accounts and acc_id not in included_account_ids:
            continue
        # --- END ADDED ---

        account = id_to_account.get(acc_id)
        if account is None:
            continue

        current_qty = cash_balances_np[acc_id]
        currency_id = cash_currency_np[acc_id]
        local_currency = id_to_currency.get(currency_id, default_currency)

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        
        if pd.isna(fx_rate):
             # Hist Fix: Default to 1.0
             fx_rate = 1.0

        cash_val_display = current_qty * fx_rate
        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Cash {account}, Qty: {current_qty}, FX: {fx_rate}, MV: {cash_val_display}")

        total_market_value_display_curr_agg += cash_val_display
        if IS_DEBUG_DATE:
            logging.debug(f"    Numba Val Agg: Running Total after Cash {account}: {total_market_value_display_curr_agg}")

    if IS_DEBUG_DATE:
        logging.debug(
            f"--- DEBUG VALUE CALC for {target_date} END --- Final Value: {total_market_value_display_curr_agg}, Lookup Failed: {any_lookup_nan_on_date}"
        )
    return total_market_value_display_curr_agg, any_lookup_nan_on_date


# --- Dispatcher Function ---
def _calculate_portfolio_value_at_date_unadjusted(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    processed_warnings: set,
    # --- ADD MAPPINGS and METHOD ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    method: str = HISTORICAL_CALC_METHOD,
    included_accounts: Optional[List[str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Dispatcher function to calculate portfolio value using either Python or Numba method.
    """
    if method == "numba":
        return _calculate_portfolio_value_at_date_unadjusted_numba(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
            symbol_to_id,
            id_to_symbol,
            account_to_id,
            id_to_account,
            type_to_id,
            currency_to_id,
            id_to_currency,
            included_accounts=included_accounts,  # Pass included_accounts
        )
    elif method == "python":
        return _calculate_portfolio_value_at_date_unadjusted_python(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
            included_accounts=included_accounts,  # Pass included_accounts
        )
    else:
        import traceback
        traceback.print_stack()
        logging.error(
            f"Invalid calculation method specified: {method}. Defaulting to python."
        )
        return _calculate_portfolio_value_at_date_unadjusted_python(
            target_date,
            transactions_df,
            historical_prices_yf_unadjusted,
            historical_fx_yf,
            target_currency,
            internal_to_yf_map,
            account_currency_map,
            default_currency,
            manual_overrides_dict,  # Pass through
            processed_warnings,
        )


@profile
def _calculate_daily_metrics_worker(
    eval_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    # benchmark_symbols_yf removed
    # --- ADD MAPPINGS and METHOD --- # type: ignore
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],  # type: ignore
    calc_method: str = HISTORICAL_CALC_METHOD,  # Use config default
    included_accounts: Optional[List[str]] = None,  # ADDED
) -> Optional[Dict]:
    """
    Worker function (for multiprocessing) to calculate key metrics for a single date.

    Calls `_calculate_portfolio_value_at_date_unadjusted` to get the portfolio market value
    and `_calculate_daily_net_cash_flow` to get external cash flows for the `eval_date`.
    Also retrieves the closing prices for specified benchmark symbols using *adjusted*
    historical prices for accurate benchmark comparison. Returns results in a dictionary.
    Handles exceptions internally and returns a specific structure on failure.

    Args:
        eval_date (date): The date for which metrics are calculated.
        transactions_df (pd.DataFrame): DataFrame of all cleaned transactions.
        historical_prices_yf_unadjusted (Dict): Unadjusted stock prices for portfolio value calc.
        historical_prices_yf_adjusted (Dict): Adjusted stock/benchmark prices for benchmark lookup.
        historical_fx_yf (Dict): Historical FX rates vs USD.
        target_currency (str): The currency for portfolio value and cash flow results.
        internal_to_yf_map (Dict): Mapping from internal symbols to YF tickers.
        account_currency_map (Dict): Mapping of accounts to their local currencies.
        default_currency (str): Default currency.
        manual_overrides_dict (Optional[Dict[str, Dict[str, Any]]]): Manual overrides for price, etc.
        benchmark_symbols_yf (List[str]): List of YF tickers for benchmark symbols.
        symbol_to_id (Dict): Map internal symbol string to int ID.
        account_to_id (Dict): Map account string to int ID.
        type_to_id (Dict): Map transaction type string to int ID.
        currency_to_id (Dict): Map currency string to int ID.
        calc_method (str): The calculation method to use ('python' or 'numba').

    Returns:
        Optional[Dict]: A dictionary containing calculated metrics for the date:
            {'Date': date, 'value': float|np.nan, 'net_flow': float|np.nan,
             'value_lookup_failed': bool, 'flow_lookup_failed': bool,
             'bench_lookup_failed': bool, '{bm_symbol} Price': float|np.nan, ...}
            Returns a dictionary with error flags set on critical failure within the worker.
            Returns None only if the function entry itself fails unexpectedly.
    """
    # ... (Function body remains unchanged) ...
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s [%(levelname)-8s] PID:%(process)d {%(module)s:%(lineno)d} %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )
    try:
        start_time = time.time()
        dummy_warnings_set = set()

        portfolio_value_main, val_lookup_failed_main = (
            _calculate_portfolio_value_at_date_unadjusted(
                eval_date,
                transactions_df,
                historical_prices_yf_unadjusted,
                historical_fx_yf,
                target_currency,
                internal_to_yf_map,
                account_currency_map,
                default_currency,
                manual_overrides_dict,  # Pass through
                dummy_warnings_set,
                # --- PASS MAPPINGS and METHOD ---
                symbol_to_id,
                id_to_symbol,
                account_to_id,
                id_to_account,
                type_to_id,
                currency_to_id,
                id_to_currency,
                # STOCK_QUANTITY_CLOSE_TOLERANCE is implicitly passed to numba version via _calculate_daily_metrics_worker
                method=calc_method,
                included_accounts=included_accounts,  # Pass included_accounts
            )
        )
        end_time_main = time.time()

        # --- Comparison Logic (Optional) ---
        if HISTORICAL_COMPARE_METHODS and calc_method == "numba":
            try:
                start_time_py = time.time()
                portfolio_value_py, val_lookup_failed_py = (
                    _calculate_portfolio_value_at_date_unadjusted_python(
                        eval_date,
                        transactions_df,
                        historical_prices_yf_unadjusted,
                        historical_fx_yf,
                        target_currency,
                        internal_to_yf_map,
                        account_currency_map,
                        default_currency,
                        manual_overrides_dict,  # Pass through to Python version for comparison
                        dummy_warnings_set,
                    )
                )
                end_time_py = time.time()
                diff = (
                    abs(portfolio_value_main - portfolio_value_py)
                    if pd.notna(portfolio_value_main) and pd.notna(portfolio_value_py)
                    else np.nan
                )
                
                if diff > 1e-3:  # Log significant differences
                    logging.warning(
                        f"COMPARE {eval_date}: Numba={portfolio_value_main:.4f} ({end_time_main-start_time:.4f}s), Python={portfolio_value_py:.4f} ({end_time_py-start_time_py:.4f}s), Diff={diff:.4f}"
                    )
                elif eval_date.day == 1:  # Log first day of month for timing check
                    logging.info(
                        f"COMPARE {eval_date}: Numba={portfolio_value_main:.4f} ({end_time_main-start_time:.4f}s), Python={portfolio_value_py:.4f} ({end_time_py-start_time_py:.4f}s), Diff={diff:.4f}"
                    )

            except Exception as e_comp:
                logging.error(
                    f"Error during comparison calculation for {eval_date}: {e_comp}"
                )
        elif HISTORICAL_COMPARE_METHODS and calc_method == "python":
            # If primary is python, comparison logic would go here if needed
            pass
        # --- End Comparison Logic ---

        portfolio_value = portfolio_value_main
        val_lookup_failed = val_lookup_failed_main

        if val_lookup_failed_main:
            logging.warning(f"Valuation failed for date: {eval_date}")
        if pd.isna(portfolio_value):
            net_cash_flow = np.nan
            flow_lookup_failed = True  # If value failed, flow is irrelevant/failed
        else:
            net_cash_flow, flow_lookup_failed = _calculate_daily_net_cash_flow(
                eval_date,
                transactions_df,
                target_currency,
                historical_fx_yf,
                account_currency_map,
                default_currency,
                dummy_warnings_set,
                included_accounts=included_accounts,  # FIX: Pass the actual included_accounts list
                historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
                internal_to_yf_map=internal_to_yf_map,
            )

        # Benchmark processing moved to main process
        bench_lookup_failed = False

        result_row = {
            "Date": eval_date,
            "value": portfolio_value,
            "net_flow": net_cash_flow,
            "value_lookup_failed": val_lookup_failed,
            "flow_lookup_failed": flow_lookup_failed,
            "bench_lookup_failed": bench_lookup_failed,
        }
        # result_row.update(benchmark_prices) # Removed
        return result_row
    except Exception as e:
        logging.critical(
            f"!!! CRITICAL ERROR in worker process for date {eval_date}: {e}"
        )
        logging.exception("Worker Traceback:")
        failed_row = {"Date": eval_date, "value": np.nan, "net_flow": np.nan}
        # Benchmark columns removed from failure row
        failed_row["value_lookup_failed"] = True
        failed_row["flow_lookup_failed"] = True
        failed_row["bench_lookup_failed"] = True
        failed_row["worker_error"] = True
        failed_row["worker_error_msg"] = str(e)
        return failed_row


# --- Helper Function for Historical Input Preparation ---
@profile
def _prepare_historical_inputs(
    # transactions_csv_file: str, # REMOVED
    preloaded_transactions_df: Optional[pd.DataFrame],  # ADDED
    original_transactions_df_for_ignored: Optional[
        pd.DataFrame
    ],  # ADDED for ignored context
    ignored_indices_from_load: Set[int],  # ADDED
    ignored_reasons_from_load: Dict[int, str],  # ADDED
    account_currency_map: Dict,
    default_currency: str,
    include_accounts: Optional[List[str]],
    exclude_accounts: Optional[List[str]],
    start_date: date,
    end_date: date,
    benchmark_symbols_yf: List[str],
    display_currency: str,
    current_hist_version: str = "v10",
    raw_cache_prefix: str = HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
    daily_cache_prefix: str = DAILY_RESULTS_CACHE_PATH_PREFIX,
    # preloaded_transactions_df: Optional[pd.DataFrame] = None, # Already added above
    user_symbol_map: Optional[Dict[str, str]] = None,
    user_excluded_symbols: Optional[Set[str]] = None,
    # ADDED: original_csv_file_path for daily_results_cache_key hash generation
    original_csv_file_path: Optional[str] = None,
    interval: str = "1d",
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],  # This will be original_transactions_df_for_ignored
    Set[int],
    Dict[int, str],
    List[str],
    List[str],
    List[str],
    List[str],
    List[str],
    Dict[str, str],
    Dict[str, str],
    Dict[str, List[Dict]],
    str,
    str,
    Optional[str],
    Optional[str],
    str,
]:
    # logging.info("Preparing inputs for historical calculation...")
    empty_tuple_return = (
        None,
        None,
        set(),
        {},
        [],
        [],
        [],
        [],
        [],
        {},
        {},
        {},
        "",
        "",
        None,
        None,
        "",
    )

    effective_user_symbol_map = user_symbol_map if user_symbol_map is not None else {}
    effective_user_excluded_symbols = (
        user_excluded_symbols if user_excluded_symbols is not None else set()
    )

    # Use preloaded DataFrame
    all_transactions_df = preloaded_transactions_df
    # Use the passed-in ignored data directly
    ignored_indices = (
        ignored_indices_from_load.copy() if ignored_indices_from_load else set()
    )
    ignored_reasons = (
        ignored_reasons_from_load.copy() if ignored_reasons_from_load else {}
    )

    if all_transactions_df is None:
        logging.error(
            "ERROR in _prepare_historical_inputs: No preloaded transaction data provided."
        )
        return empty_tuple_return  # type: ignore

    all_available_accounts_list = (
        sorted(all_transactions_df["Account"].unique().tolist())
        if "Account" in all_transactions_df.columns and not all_transactions_df.empty
        else []
    )
    available_accounts_set = set(all_available_accounts_list)
    transactions_df_effective = pd.DataFrame()
    included_accounts_list_sorted = []
    excluded_accounts_list_sorted = []
    filter_desc = "All Accounts"

    if not include_accounts:  # None or empty list means all
        transactions_df_included = all_transactions_df.copy()
        included_accounts_list_sorted = sorted(list(available_accounts_set))
    else:
        # Normalize the DataFrame's Account columns to ensure consistent matching
        # Generate normalized account sets from the DataFrame
        df_accounts_normalized = _normalize_series(all_transactions_df["Account"]).unique()
        available_accounts_in_df = set(df_accounts_normalized)
        
        if "To Account" in all_transactions_df.columns:
            df_to_accounts_normalized = _normalize_series(
                all_transactions_df["To Account"]
            ).dropna().unique()
            available_accounts_in_df.update(df_to_accounts_normalized)

        # Normalize include_accounts to match
        normalized_include_accounts = [acc.strip().upper() for acc in include_accounts]
        
        valid_include_accounts = [
            acc for acc in normalized_include_accounts if acc in available_accounts_in_df
        ]
        if not valid_include_accounts:
            logging.warning(
                "WARN in _prepare_historical_inputs: No valid accounts to include."
            )
            return empty_tuple_return  # type: ignore

        # --- Robust Filtering Logic (Matches calculate_portfolio_summary) ---
        # 1. Primary Account Match - use normalized account column
        from_account_mask = _normalize_series(all_transactions_df["Account"]).isin(valid_include_accounts)

        # 2. Destination Account Match (Transfers In) - use normalized to account column
        to_account_mask = pd.Series(False, index=all_transactions_df.index)
        if "To Account" in all_transactions_df.columns:
            to_account_mask = (
                _normalize_series(all_transactions_df["To Account"])
                .isin(valid_include_accounts)
                .fillna(False)
            )

        # 3. Preceding Transactions for Transfers In
        preceding_tx_mask = pd.Series(False, index=all_transactions_df.index)
        transfers_in_df = all_transactions_df[to_account_mask]

        if not transfers_in_df.empty:
            # Get unique (Symbol, From_Account, Date) tuples from the transfers-in
            transfer_events = transfers_in_df[
                ["Symbol", "Account", "Date"]
            ].drop_duplicates()

            masks_to_combine = []
            for _, event_row in transfer_events.iterrows():
                transfer_symbol = event_row["Symbol"]
                transfer_from_account = event_row["Account"]
                transfer_date = event_row["Date"]

                # Create a mask for all preceding/concurrent tx for this specific asset
                mask = (
                    (all_transactions_df["Symbol"] == transfer_symbol)
                    & (all_transactions_df["Account"] == transfer_from_account)
                    & (all_transactions_df["Date"] <= transfer_date)
                )
                masks_to_combine.append(mask)

            if masks_to_combine:
                preceding_tx_mask = pd.concat(masks_to_combine, axis=1).any(axis=1)

        # --- MODIFIED: Ensure SPLIT transactions and "All Accounts" are always included ---
        # Splits are corporate actions that affect the stock globally.
        # "All Accounts" acts as a catch-all for global or manual adjustment transactions.
        split_mask = (
            all_transactions_df["Type"].astype(str).str.lower().isin(["split", "stock split"])
        )
        all_accounts_mask = (
            _normalize_series(all_transactions_df["Account"]) == "ALL ACCOUNTS"
        )

        final_combined_mask = (
            from_account_mask | to_account_mask | preceding_tx_mask | split_mask | all_accounts_mask
        )
        transactions_df_included = all_transactions_df[final_combined_mask].copy()

        included_accounts_list_sorted = sorted(valid_include_accounts)
        filter_desc = f"Included: {', '.join(included_accounts_list_sorted)}"

    if not exclude_accounts or not isinstance(exclude_accounts, list):
        transactions_df_effective = transactions_df_included.copy()
    else:
        # Check if accounts exist in "Account" OR "To Account"
        available_accounts_in_df = set(all_transactions_df["Account"].unique())
        if "To Account" in all_transactions_df.columns:
            available_accounts_in_df.update(
                all_transactions_df["To Account"].dropna().unique()
            )

        valid_exclude_accounts = [
            acc for acc in exclude_accounts if acc in available_accounts_in_df
        ]
        if valid_exclude_accounts:
            logging.info(
                f"Hist Prep: Excluding accounts: {', '.join(sorted(valid_exclude_accounts))}"
            )
            
            # FIX: If include_accounts is set, we only exclude accounts that are explicitly in the include list
            # (resolving contradiction). We do NOT exclude dependencies (like transfer sources) that are not in the include list.
            if include_accounts:
                accounts_to_exclude_really = set(valid_exclude_accounts).intersection(set(valid_include_accounts))
                if accounts_to_exclude_really:
                     transactions_df_effective = transactions_df_included[
                        ~transactions_df_included["Account"].isin(accounts_to_exclude_really)
                    ].copy()
                else:
                    transactions_df_effective = transactions_df_included.copy()
            else:
                # Original logic for "All Accounts"
                transactions_df_effective = transactions_df_included[
                    ~transactions_df_included["Account"].isin(valid_exclude_accounts)
                ].copy()
                
            excluded_accounts_list_sorted = sorted(valid_exclude_accounts)
            if include_accounts:
                filter_desc += (
                    f" (Excluding: {', '.join(excluded_accounts_list_sorted)})"
                )
            else:
                filter_desc = f"All Accounts (Excluding: {', '.join(excluded_accounts_list_sorted)})"
        else:
            transactions_df_effective = transactions_df_included.copy()

    if transactions_df_effective.empty:
        logging.warning(
            "WARN in _prepare_historical_inputs: No transactions remain after filtering."
        )
        # Return structure must match
        return (
            transactions_df_effective,
            original_transactions_df_for_ignored,  # Pass this through
            ignored_indices,
            ignored_reasons,
            all_available_accounts_list,
            included_accounts_list_sorted,
            excluded_accounts_list_sorted,
            [],
            [],
            {},
            {},
            {},
            "",
            "",
            None,
            None,
            filter_desc,
        )

    # Extract split history from the *unfiltered* `all_transactions_df`
    # to ensure all splits are considered, even if an account with a split tx is filtered out.
    split_transactions = all_transactions_df[
        all_transactions_df["Type"].str.lower().isin(["split", "stock split"])
        & all_transactions_df["Split Ratio"].notna()
        & (
            pd.to_numeric(all_transactions_df["Split Ratio"], errors="coerce") > 0
        )  # Ensure valid ratio
    ].sort_values(by="Date", ascending=True)

    splits_by_internal_symbol: Dict[str, List[Dict[str, Any]]] = {}
    if not split_transactions.empty:
        splits_by_internal_symbol = {
            symbol: group[["Date", "Split Ratio"]]
            .apply(
                lambda r: {
                    "Date": r["Date"].date(),  # Assuming Date is datetime
                    "Split Ratio": float(r["Split Ratio"]),
                },
                axis=1,
            )
            .tolist()
            for symbol, group in split_transactions.groupby("Symbol")
        }

    # Symbols for stocks (from effective/filtered transactions)
    # --- SIMPLIFICATION: Include ALL symbols in the effective transaction set ---
    # Previous optimization attempted to only fetch symbols "active" or "held" in the window,
    # but the "held" check (approximate quantity sum) was fragile and missed edge cases 
    # (e.g. Dividend Reinvestment not being counted as Buy), leading to missing prices/value 
    # for long-term holdings in short-term views (like 1M graph).
    # Since yfinance calls are cached and the number of symbols is usually manageable (<500),
    # it is safer to just include all symbols present in the filtered account's history.
    
    all_symbols_internal_effective = transactions_df_effective["Symbol"].unique().tolist()
    # -------------------------------------------------------------------------
    symbols_to_fetch_yf_portfolio = []
    internal_to_yf_map: Dict[str, str] = {}  # Ensure type
    for internal_sym in all_symbols_internal_effective:
        if is_cash_symbol(
            internal_sym
        ):  # MODIFIED: Use helper to catch all cash symbols
            continue
        yf_sym = map_to_yf_symbol(
            internal_sym, effective_user_symbol_map, effective_user_excluded_symbols
        )
        if yf_sym:
            symbols_to_fetch_yf_portfolio.append(yf_sym)
            internal_to_yf_map[internal_sym] = yf_sym

    yf_to_internal_map_hist: Dict[str, str] = {
        v: k for k, v in internal_to_yf_map.items()
    }  # Ensure type
    symbols_to_fetch_yf_portfolio = sorted(list(set(symbols_to_fetch_yf_portfolio)))
    symbols_for_stocks_and_benchmarks_yf = sorted(
        list(set(symbols_to_fetch_yf_portfolio) | set(benchmark_symbols_yf))
    )

    all_currencies_in_tx_effective = set(
        transactions_df_effective["Local Currency"].unique()
    )
    all_currencies_needed = all_currencies_in_tx_effective.union(
        {display_currency, default_currency}
    )
    # --- ADDED: More robust cleaning for all_currencies_needed ---
    cleaned_all_currencies_needed = {
        str(c).strip().upper()
        for c in all_currencies_needed
        if pd.notna(c)
        and isinstance(str(c).strip(), str)
        and str(c).strip() not in ["", "<NA>", "NAN", "NONE", "N/A"]
        and len(str(c).strip()) == 3
    }
    all_currencies_needed = cleaned_all_currencies_needed
    # --- END ADDED ---
    fx_pairs_for_api_yf = sorted(
        list(
            {f"{curr}=X" for curr in all_currencies_needed if curr != "USD" and curr}
        )  # Added 'and curr' to ensure not empty
    )

    # Cache paths
    app_cache_dir = None
    if QStandardPaths:
        cache_dir_base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        if cache_dir_base:
            # FIX: Ensure we use the proper app structure (Org/App)
            # QStandardPaths usually returns ~/Library/Caches on macOS for python scripts not in a bundle
            if config.APP_NAME not in cache_dir_base:
                app_cache_dir = os.path.join(cache_dir_base, config.ORG_NAME, config.APP_NAME)
            else:
                app_cache_dir = cache_dir_base
            
            os.makedirs(app_cache_dir, exist_ok=True)

    raw_data_cache_file_name = (
        f"{raw_cache_prefix}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    )
    raw_data_cache_file = (
        os.path.join(app_cache_dir, raw_data_cache_file_name)
        if app_cache_dir
        else raw_data_cache_file_name
    )

    # Use stable cache key (no dates) to allow incremental updates
    # The market_data provider will handle date range checks
    raw_data_cache_key = f"ADJUSTED_v7::{'_'.join(sorted(symbols_for_stocks_and_benchmarks_yf))}::{'_'.join(fx_pairs_for_api_yf)}"

    daily_results_cache_file: Optional[str] = None
    daily_results_cache_key: Optional[str] = None
    try:
        # Use placeholder if original_csv_file_path is not available (loading from DB)
        tx_file_hash_component = (
            _get_file_hash(original_csv_file_path)
            if original_csv_file_path and os.path.exists(original_csv_file_path)
            else "FROM_DB_OR_DATAFRAME"
        )
        if original_csv_file_path and not os.path.exists(original_csv_file_path):
            logging.warning(
                f"Hist Prep: Original CSV path '{original_csv_file_path}' for hash not found. Using placeholder."
            )

        acc_map_str = json.dumps(account_currency_map, sort_keys=True)
        included_accounts_str = json.dumps(
            included_accounts_list_sorted
        )  # Use effectively included
        excluded_accounts_str = json.dumps(
            excluded_accounts_list_sorted
        )  # Use effectively excluded

        # Add user_symbol_map and user_excluded_symbols to cache key
        user_map_str = json.dumps(effective_user_symbol_map, sort_keys=True)
        user_excluded_str = json.dumps(sorted(list(effective_user_excluded_symbols)))

        daily_results_cache_key_str_content = (
            f"DAILY_RES_{current_hist_version}::"
            f"{start_date.isoformat()}::{end_date.isoformat()}::"
            f"{tx_file_hash_component}::"
            f"{display_currency}::"  # Removed benchmarks from key
            f"{acc_map_str}::{default_currency}::"
            f"{included_accounts_str}::{excluded_accounts_str}::"
            f"{user_map_str}::{user_excluded_str}::"  # Added user settings
            f"{HISTORICAL_CALC_METHOD}::"  # Added calc method
            f"{interval}"  # Added interval to prevent hourly/daily collision
        )
        daily_results_cache_key = hashlib.sha256(
            daily_results_cache_key_str_content.encode()
        ).hexdigest()[
            :32
        ]  # Longer hash for more uniqueness

        daily_results_filename = (
            f"{daily_cache_prefix}_{daily_results_cache_key}.feather"
        )
        daily_results_cache_file = (
            os.path.join(app_cache_dir, daily_results_filename)
            if app_cache_dir
            else daily_results_filename
        )
        logging.info(
            f"Hist Prep: Daily results cache file set to: {daily_results_cache_file}"
        )
    except Exception as e_key:
        logging.warning(
            f"Hist Prep WARN: Could not generate daily results cache key/filename: {e_key}."
        )

    return (
        transactions_df_effective,
        original_transactions_df_for_ignored,  # Pass this through
        ignored_indices,  # Pass this through
        ignored_reasons,  # Pass this through
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
    )


# --- Daily Results Calculation (Keep as is) ---
@profile

# --- VECTORIZED VALUATION HELPER ---
def _value_daily_holdings_vectorized(
    date_range: pd.DatetimeIndex,
    daily_holdings_qty_np: np.ndarray,
    daily_last_prices_np: Optional[np.ndarray],
    daily_cash_balances_np: np.ndarray,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    account_currency_map: Dict[str, str],
    id_to_symbol: Dict[int, str],
    id_to_account: Dict[int, str],
    internal_to_yf_map: Dict[str, str],
    account_ids_to_include_set: Set[int],
    default_currency: str,
) -> Tuple[pd.Series, bool, str]:
    """
    Vectorized valuation of holdings and cash.
    Returns (daily_value_series, has_errors, status_msg).
    """
    num_days, num_symbols, num_accounts = daily_holdings_qty_np.shape
    has_errors = False
    status_msg = ""
    
    # 1. Prepare aligned PRICES array (N_days, N_symbols)
    # Initialize with NaNs
    daily_prices_aligned = np.full((num_days, num_symbols), np.nan, dtype=np.float64)
    

    
    # Iterate symbols to fill prices
    # This loop is cheap (N_symbols ~ hundreds)
    for sym_id in range(num_symbols):
        symbol = id_to_symbol.get(sym_id)
        if symbol == CASH_SYMBOL_CSV:
            continue
            
        yf_symbol = internal_to_yf_map.get(symbol)
        if yf_symbol and yf_symbol in historical_prices_yf_unadjusted:
            price_df = historical_prices_yf_unadjusted[yf_symbol]
            if not price_df.empty:
                col_to_use = None
                if "price" in price_df.columns:
                    col_to_use = "price"
                elif "Adj Close" in price_df.columns:
                    col_to_use = "Adj Close"
                elif "Close" in price_df.columns:
                    col_to_use = "Close"
                
                if col_to_use:
                    # Reindex to date_range
                    price_series = price_df[col_to_use]
                    # FIX: Ensure UTC awareness and DO NOT Normalize (preserves intraday/hourly timestamps)
                    price_series.index = pd.to_datetime(price_series.index, utc=True)
                
                # Reindex using ffill
                aligned_series = price_series.reindex(date_range, method='ffill')
                daily_prices_aligned[:, sym_id] = aligned_series.values
                


    # 2. Check for missing prices and fill with Last Price if valid
    # We will handle fallback during calculation.

    # 3. Prepare aligned FX array (N_days, N_accounts)
    daily_fx_aligned = np.full((num_days, num_accounts), np.nan, dtype=np.float64)
    
    # Cache FX series to avoid re-fetching for same currency
    currency_fx_series_cache = {}
    
    # Pre-fetch Target Currency Rate vs USD (Series)
    # Target Rate = Target / USD
    target_rate_series = None
    target_curr_upper = target_currency.upper()
    
    if target_curr_upper == "USD":
        target_rate_series = pd.Series(1.0, index=date_range)
    else:
        # Fetch Target=X (e.g. THB=X means THB per USD)
        target_pair = f"{target_curr_upper}=X"
        if target_pair in historical_fx_yf:
            t_df = historical_fx_yf[target_pair]
            if not t_df.empty:
                col_to_use = None
                if "price" in t_df.columns: col_to_use = "price"
                elif "Adj Close" in t_df.columns: col_to_use = "Adj Close"
                elif "Close" in t_df.columns: col_to_use = "Close"
                elif "rate" in t_df.columns: col_to_use = "rate"

                if col_to_use:
                    # Reindex to date_range with ffill
                    # Reindex to date_range with ffill then bfill to cover start of history
                    t_series = t_df[col_to_use]
                t_series.index = pd.to_datetime(t_series.index, utc=True)
                # FIX: Ensure we have a value for the first day by bfilling, but use interpolation
                target_rate_series = t_series.reindex(date_range).interpolate(method='linear').ffill().bfill()
    
    # If target rate missing, we can't convert anything (unless local==target)
    # Ensure it's not None for math consistency, fill with NaN if missing
    if target_rate_series is None:
         target_rate_series = pd.Series(np.nan, index=date_range)

    for acc_id in range(num_accounts):
        if acc_id not in account_ids_to_include_set:
            continue
            
        acc_name = id_to_account.get(acc_id)
        local_curr = account_currency_map.get(acc_name, default_currency)
        local_curr_upper = local_curr.upper()
        
        if local_curr in currency_fx_series_cache:
            daily_fx_aligned[:, acc_id] = currency_fx_series_cache[local_curr]
            continue
            
        # Build aligned series
        if local_curr == target_currency:
            rates = np.ones(num_days, dtype=np.float64)
            daily_fx_aligned[:, acc_id] = rates
            currency_fx_series_cache[local_curr] = rates
            continue
        
        # Calculate Cross Rate: (Target / USD) / (Local / USD)
        # We need Local Rate = Local / USD
        local_rate_series = None
        if local_curr_upper == "USD":
            local_rate_series = pd.Series(1.0, index=date_range)
        else:
            local_pair = f"{local_curr_upper}=X"
            if local_pair in historical_fx_yf:
                 l_df = historical_fx_yf[local_pair]
                 if not l_df.empty:
                    col_to_use = None
                    if "price" in l_df.columns: col_to_use = "price"
                    elif "Adj Close" in l_df.columns: col_to_use = "Adj Close"
                    elif "Close" in l_df.columns: col_to_use = "Close"
                    elif "rate" in l_df.columns: col_to_use = "rate"
                    
                    if col_to_use:
                        l_series = l_df[col_to_use]
                    # FIX: Ensure UTC awareness and DO NOT Normalize (preserves intraday/hourly timestamps)
                    l_series.index = pd.to_datetime(l_series.index, utc=True)
                    # FIX: Aggressive backfill for local rates too, but smooth it
                    local_rate_series = l_series.reindex(date_range).interpolate(method='linear').ffill().bfill()

        if local_rate_series is None:
             # Missing local rate data - Default to 1.0 to prevent NaN propagation
             rates = np.ones(num_days, dtype=np.float64)
             # logging.warning(f"Hist Val: Missing FX rates for {local_curr}, defaulting to 1.0")
             has_errors = True
        else:
              # Compute cross rate
              # Handle division by zero or NaN
              with np.errstate(divide='ignore', invalid='ignore'):
                  # aligned_target / aligned_local
                  cross_rates = target_rate_series / local_rate_series
                  
                  # Final safety: If any cross rate is STILL NaN, it means one of the series was all NaN 
                  # or had persistent gaps. Try to bfill again to be sure.
                  if pd.Series(cross_rates).isna().any():
                       cross_rates = pd.Series(cross_rates).bfill().ffill().values
              
              rates = cross_rates
              # Check for NaNs and ONLY fill with 1.0 as a catastrophic last resort
              # if both target and local rates were completely missing.
              mask_nan_fx = np.isnan(rates)
              if mask_nan_fx.any():
                   rates[mask_nan_fx] = 1.0
        
        daily_fx_aligned[:, acc_id] = rates
        currency_fx_series_cache[local_curr] = rates
    
    
    # 4. Calculate Market Value (Stocks)
    # Broadcast Arrays to (Days, Syms, Accs)
    # P_3d = (D, S, 1)
    P_3d = daily_prices_aligned[:, :, np.newaxis]
    
    # Handle Price Fallback in 3D
    if daily_last_prices_np is not None:
        # P_full = (D, S, A)
        P_full = np.broadcast_to(P_3d, daily_holdings_qty_np.shape).copy()
        mask_nan = np.isnan(P_full)
        
        # --- NEW: Identify and Log symbols that are missing market data ---
        if mask_nan.any():
            nan_indices = np.where(mask_nan)
            # Just log once per symbol if it has significant gaps
            unique_nan_sym_ids = np.unique(nan_indices[1])
            for s_id in unique_nan_sym_ids:
                sym_name = id_to_symbol.get(s_id, "Unknown")
                if sym_name != CASH_SYMBOL_CSV:
                    logging.debug(f"Valuation: Symbol '{sym_name}' missing market data. Using fallback.")
        
        P_full[mask_nan] = daily_last_prices_np[mask_nan]
        
        # --- NEW: Final fallback for symbols with NO market data and NO transaction price ---
        # (Though buy transactions should always have a price)
        # If still NaN, we might have a transfer or split with missing price.
        # Use 1.0 as a catastrophic fallback to avoid total value NaN if possible, 
        # but only if quantity > 0. Actually, better to let it be NaN and ffill later.
        
        V_stocks = daily_holdings_qty_np * P_full * daily_fx_aligned[:, np.newaxis, :]
    else:
        V_stocks = daily_holdings_qty_np * P_3d * daily_fx_aligned[:, np.newaxis, :]
    
    # --- CRITICAL FIX: Prevent 0.0 * NaN = NaN ---
    # Ensure zero quantity always results in zero value, regardless of Price or FX NaNs
    V_stocks[daily_holdings_qty_np == 0] = 0.0

    # Mask out excluded accounts
    acc_mask = np.zeros(num_accounts, dtype=bool)
    for acc_id in account_ids_to_include_set:
        acc_mask[acc_id] = True
    
    # Zeros for excluded
    V_stocks[:, :, ~acc_mask] = 0.0
    
    # Sum over Symbols and Accounts
    total_stock_value = np.sum(V_stocks, axis=(1, 2))

    # 5. Calculate Cash Value
    V_cash = daily_cash_balances_np * daily_fx_aligned
    # --- CRITICAL FIX: Prevent 0.0 * NaN = NaN for Cash ---
    V_cash[daily_cash_balances_np == 0] = 0.0
    
    V_cash[:, ~acc_mask] = 0.0
    
    total_cash_value = np.sum(V_cash, axis=1)

    # 6. Total
    total_value = total_stock_value + total_cash_value
    
    daily_value_series = pd.Series(total_value, index=date_range)
    
    # --- FIX: Avoid zeroing out portfolio value due to missing data ---
    # Instead of nan_to_num(nan=0.0), use forward-fill to maintain TWR integrity.
    if daily_value_series.isna().any():
        status_msg += " (partial data missing, f-filled)"
        # Identify which dates are missing
        missing_dates = daily_value_series.index[daily_value_series.isna()]
        logging.warning(f"Valuation: Missing data for {len(missing_dates)} days. Using forward-fill.")
        daily_value_series = daily_value_series.ffill().fillna(0.0) # fillna(0) only for very beginning
    
    return daily_value_series, has_errors, status_msg

def _load_or_calculate_daily_results(
    use_daily_results_cache: bool,
    daily_results_cache_file: Optional[str],
    daily_results_cache_key: Optional[str],
    interval: str,
    worker_signals: Optional[Any],  # <-- ADDED: For progress reporting
    transactions_csv_file: str,  # <-- ADDED: Need path for mtime check
    start_date: date,
    end_date: date,
    transactions_df_effective: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    display_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    # clean_benchmark_symbols_yf removed
    # --- MOVED MAPPINGS HERE ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],  # ADDED
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    # --- NEW: Add pre-calculated holdings and account filter list ---
    all_holdings_qty: Optional[np.ndarray],
    all_cash_balances: Optional[np.ndarray],
    all_last_prices: Optional[np.ndarray],  # NEW argument
    all_holdings_start_date: Optional[date],  # NEW argument for L1 offset calc
    included_accounts_list: List[str],
    # --- Parameters with defaults follow ---
    num_processes: Optional[int] = None,
    current_hist_version: str = "v13",
    filter_desc: str = "All Accounts",
    calc_method: str = HISTORICAL_CALC_METHOD,
) -> Tuple[pd.DataFrame, bool, str]:  # type: ignore
    """
    Loads calculated daily results from cache or calculates them using parallel processing.

    Checks if a valid daily results cache file exists based on the provided key.
    If valid, loads the pre-calculated DataFrame.
    If not valid or caching is disabled, it determines the relevant market days within
    the date range, distributes the calculation of daily portfolio value (using unadjusted prices),
    net cash flow, and benchmark prices (using adjusted prices) across multiple processes
    using `_calculate_daily_metrics_worker`. It then aggregates the results, calculates
    daily gain and daily return (TWR), and saves the results to cache if enabled.

    Args:
        use_daily_results_cache (bool): Whether to use the daily results cache.
        daily_results_cache_file (Optional[str]): Path to the daily results cache file.
        daily_results_cache_key (Optional[str]): Validation key for the daily results cache.
        worker_signals (Optional[Any]): The signals object from the GUI worker (or None).
        transactions_csv_file (str): Path to the source transactions CSV file (for mtime check).
        start_date (date): Start date of the analysis period.
        end_date (date): End date of the analysis period.
        transactions_df_effective (pd.DataFrame): Filtered transactions for the scope.
        historical_prices_yf_unadjusted (Dict): Unadjusted prices for portfolio value.
        historical_prices_yf_adjusted (Dict): Adjusted prices for benchmarks.
        historical_fx_yf (Dict): Historical FX rates.
        display_currency (str): Target currency for results.
        internal_to_yf_map (Dict): Internal symbol to YF ticker map.
        account_currency_map (Dict): Account to currency map.
        default_currency (str): Default currency.
        manual_overrides_dict (Optional[Dict[str, Dict[str, Any]]]): Manual overrides for price, etc.
        clean_benchmark_symbols_yf (List[str]): List of YF benchmark tickers.
        num_processes (Optional[int]): Number of parallel processes to use. Defaults to CPU count - 1.
        current_hist_version (str): Version string used in cache key generation.
        filter_desc (str): Description of the account scope for logging.
        calc_method (str): The calculation method to use ('python', 'numba', 'numba_chrono').
        symbol_to_id (Dict): Map internal symbol string to int ID.
        account_to_id (Dict): Map account string to int ID.
        type_to_id (Dict): Map transaction type string to int ID.
        currency_to_id (Dict): Map currency string to int ID.
        id_to_currency (Dict): Map currency ID back to string.

    Returns:
        Tuple[pd.DataFrame, bool, str]:
            - daily_df (pd.DataFrame): DataFrame indexed by date, containing columns like
              'value', 'net_flow', 'daily_gain', 'daily_return', and benchmark prices.
              Empty DataFrame on critical failure.
            - cache_valid_daily_results (bool): True if the data was loaded from a valid cache.
            - status_update (str): String describing the outcome (cache load, calculation, errors).
    """
    # ... (Function body remains unchanged) ...
    import pandas as pd  # Explicit import locally to fix UnboundLocalError
    daily_df = pd.DataFrame()
    t0 = time.time()
    cache_valid_daily_results = False
    status_update = ""
    dummy_warnings_set = set()


    # --- Chronological Calculation (numba_chrono) ---
    logging.debug(
        f"[_load_or_calculate_daily_results] Received date range: {start_date} to {end_date}"
    )
    t1_setup = time.time()
    # --- END ADDED ---

    # --- ADDED: Define metadata filename ---
    metadata_cache_file = None
    if daily_results_cache_file:
        metadata_cache_file = daily_results_cache_file.replace(".feather", ".meta.json")
    # --- END ADDED ---

    if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key:
        # --- MODIFIED: Check for both feather and metadata files ---
        if (
            os.path.exists(daily_results_cache_file)
            and metadata_cache_file
            and os.path.exists(metadata_cache_file)
        ):
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Found daily cache files (.feather & .meta.json). Validating..."
            )
            try:
                # --- ADDED: Load metadata and perform validation ---
                with open(metadata_cache_file, "r") as f_meta:
                    metadata = json.load(f_meta)

                # 1. Check Cache Key
                if metadata.get("cache_key") != daily_results_cache_key:
                    logging.warning(
                        f"Hist WARN (Scope: {filter_desc}): Daily cache key MISMATCH in metadata. Recalculating."
                    )
                    raise ValueError("Cache key mismatch")  # Treat as invalid cache

                # 2. Check Transaction File Modification Time
                # Only perform mtime check if the cache metadata itself contains a 'csv_mtime'.
                # This implies the cache was originally created from a CSV.
                if metadata.get("csv_mtime") is not None:
                    # Cache expects a CSV mtime. Now check if a valid CSV path is provided for current validation.
                    if (
                        transactions_csv_file
                        and isinstance(transactions_csv_file, str)
                        and transactions_csv_file.strip()
                    ):
                        # A non-empty CSV path is provided.
                        try:
                            if os.path.exists(transactions_csv_file):
                                csv_mtime_current = os.path.getmtime(
                                    transactions_csv_file
                                )
                                csv_mtime_cached = metadata.get(
                                    "csv_mtime"
                                )  # Already checked it's not None
                                if csv_mtime_current > csv_mtime_cached:
                                    logging.warning(
                                        f"Hist WARN (Scope: {filter_desc}): Transactions CSV file '{transactions_csv_file}' modified since cache was created. Recalculating."
                                    )
                                    raise ValueError("CSV file modified")
                            else:  # Provided CSV path does not exist, but cache expected one.
                                logging.warning(
                                    f"Hist WARN (Scope: {filter_desc}): Transactions CSV file '{transactions_csv_file}' (for mtime check) not found, but cache expected one. Recalculating."
                                )
                                raise ValueError("CSV file for mtime check not found")
                        except (
                            OSError
                        ) as e_mtime:  # Handles issues with os.path.exists or os.path.getmtime
                            logging.warning(
                                f"Hist WARN (Scope: {filter_desc}): Could not access transactions CSV file '{transactions_csv_file}' for mtime check: {e_mtime}. Recalculating."
                            )
                            raise ValueError("CSV mtime check failed (OS error)")
                    else:  # No valid CSV path provided now (None or empty), but cache expected one.
                        logging.warning(
                            f"Hist WARN (Scope: {filter_desc}): Cache metadata expects 'csv_mtime' but no valid CSV path provided for current validation (path: '{transactions_csv_file}'). Recalculating."
                        )
                        raise ValueError(
                            "Cache expected CSV mtime, but no CSV path provided now."
                        )
                # If metadata does not contain "csv_mtime", then the cache was likely not from a CSV (e.g., from DB).
                # In this case, we don't need to perform an mtime check against transactions_csv_file.
                # The cache_key check is sufficient.
                # --- END ADDED: Metadata validation ---

                # --- If validation passed, load Feather file ---
                logging.info(
                    f"Hist Daily (Scope: {filter_desc}): Cache metadata valid. Loading Feather data..."
                )
                # --- CHANGE: Load directly using read_feather ---
                daily_df = pd.read_feather(daily_results_cache_file)
                # --- END CHANGE ---

                if not isinstance(daily_df, pd.DataFrame):
                    raise ValueError("Loaded Feather cache is not a DataFrame.")

                # Handle index restoration
                if "Date" in daily_df.columns:
                    daily_df["Date"] = pd.to_datetime(daily_df["Date"], utc=True)
                    daily_df.set_index("Date", inplace=True)
                elif "index" in daily_df.columns:
                    # Fallback: 'reset_index' might have named it 'index' if original name was None
                    daily_df.rename(columns={"index": "Date"}, inplace=True)
                    daily_df["Date"] = pd.to_datetime(daily_df["Date"], utc=True)
                    daily_df.set_index("Date", inplace=True)
                elif not isinstance(daily_df.index, pd.DatetimeIndex):
                    # Fallback: try to convert index if it looks like dates (unlikely for feather if reset)
                    # But if we didn't reset index on save, feather might have saved it if supported.
                    # If it's RangeIndex, this is dangerous, so we log warning.
                    if isinstance(daily_df.index, pd.RangeIndex):
                         logging.warning("Loaded Feather cache has RangeIndex and no Date column. Cache might be invalid.")
                         raise ValueError("Invalid cache structure (missing Date index/column)")
                    
                    # Force UTC conversion to ensure compatibility with benchmark data
                    daily_df.index = pd.to_datetime(daily_df.index, errors="coerce", utc=True)
                    daily_df = daily_df[pd.notnull(daily_df.index)]

                daily_df.sort_index(inplace=True)
                
                # --- FIX: Filter cached data by requested date range ---
                # The cache might contain more data than requested (e.g. if key collision or logic change)
                # or if we want to be absolutely sure.
                # --- FIX REVERTED: Filtering moved to calculate_historical_performance ---
                # daily_df = daily_df[(daily_df.index >= ts_start) & (daily_df.index <= ts_end)]
                # --- END FIX REVERTED ---

                required_cols = [
                    "value",
                    "net_flow",
                    "daily_return",
                    "daily_gain",
                ]
                missing_cols = [c for c in required_cols if c not in daily_df.columns]

                if not missing_cols and not daily_df.empty:
                    logging.info(
                        f"Hist Daily (Scope: {filter_desc}): Loaded {len(daily_df)} rows from Feather cache."
                    )
                    cache_valid_daily_results = True
                    status_update = " Daily results loaded from cache."
                else:
                    logging.warning(
                        f"Hist WARN (Scope: {filter_desc}): Feather cache missing columns ({missing_cols}), empty, or failed validation. Recalculating."
                    )
                    daily_df = pd.DataFrame()

            # --- MODIFIED: Catch specific validation errors or general load errors ---
            except (
                ValueError,  # Catches our explicit raises for cache invalidity
                json.JSONDecodeError,  # For manifest.json issues
            ) as e_validate:
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): Daily cache validation failed: {e_validate}. Recalculating."
                )
                daily_df = pd.DataFrame()  # Ensure df is reset
            except (
                Exception
            ) as e_load_cache:  # Catch other errors (Feather load, file IO etc.)
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): Error reading/validating daily cache: {e_load_cache}. Recalculating."
                )
                daily_df = pd.DataFrame()  # Ensure df is reset
        else:
            logging.info(  # Log if either file is missing
                f"Hist Daily (Scope: {filter_desc}): Daily cache file not found. Calculating."
            )
    elif not use_daily_results_cache:
        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Daily results caching disabled. Calculating."
        )
    else:
        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Daily cache file/key invalid. Calculating."
        )

    if not cache_valid_daily_results:
        if calc_method == "numba_chrono":
            # This date range is needed for both cached and non-cached paths for valuation loop
            first_tx_date = (
                transactions_df_effective["Date"].min().date()
                if not transactions_df_effective.empty
                else start_date
                if not transactions_df_effective.empty
                else start_date
            )
            # FIX: ALWAYS calculate from the earliest possible history to ensure
            # Cumulative Net Flow (COST) and TWR context are preserved, 
            # even if the user only requested a recent slice.
            calc_start_date = first_tx_date
            calc_end_date = end_date
            
            # Determine calculation frequency range
            # If interval is intraday, we use daily for history and intraday frequency for active range
            valid_intraday = ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]
            is_intraday_request = interval in valid_intraday
            if is_intraday_request:
                freq_map = {
                    "1h": "h", "60m": "h",
                    "1m": "1min", "2m": "2min", "5m": "5min",
                    "15m": "15min", "30m": "30min", "90m": "90min"
                }
                active_freq = freq_map.get(interval, "D")
                
                # Use daily for history before start_date
                range_historical = pd.date_range(start=calc_start_date, end=start_date - timedelta(days=1), freq="D")
                # Fix: ensure historical range is UTC aware before append
                if not range_historical.empty:
                    range_historical = range_historical.tz_localize('UTC')

                # Use intraday for active range
                # FIX: Ensure range_active covers the FULL end_date day for intraday intervals
                ts_end_of_period = pd.Timestamp(calc_end_date, tz='UTC') + timedelta(days=1)
                
                # FIX 2: Check current time to prevent generating empty "future" points that get ffilled.
                # User requested: "Do not plot a flat line where the data are not available yet"
                now_utc = pd.Timestamp.now(tz='UTC')
                # If the theoretical end of period is in the future, clip it to now (plus small buffer)
                if ts_end_of_period > now_utc:
                    # Clip to now. Ceil to next interval?
                    # Simply using 'now' works because inclusive='left' in date_range loops until it hits end.
                    # We remove the buffer to strictly stop at 'now' and avoid any future flatline.
                    # Actually, if we are at 10:00:01, inclusive='left' with freq='2min' 
                    # starting at 09:30 might generate 10:00:00 tick.
                    # If data for 10:00:00 is not yet available/complete, it gets ffilled.
                    # Safer to lag slightly behind 'now' to ensure we only show fully elapsed intervals.
                    # FORCE CLIP: -5 minutes to guarantee no future flatline.
                    active_end_bound = now_utc - timedelta(minutes=5)
                    logging.info(f"[_load_or_calculate] 1D Graph Clip: now={now_utc}, bound={active_end_bound}")
                else:
                    active_end_bound = ts_end_of_period
                
                heading_into_future = False # logic no longer needed, we fill what we have
                
                range_active = pd.date_range(
                    start=pd.Timestamp(start_date, tz='UTC'), 
                    end=active_end_bound, 
                    freq=active_freq,
                    inclusive='left'
                )

                # FIX for "5D graph showing afterhours":
                # We need to filter this range to only include market hours (9:30 - 16:00 ET).
                # Otherwise, ffill() will bridge the overnight gap with a flat line.
                # Only apply this if we are in an intraday mode that expects market hours.
                if is_intraday_request and not range_active.empty:
                     # Convert to NY time to filter by clock time
                     range_active_ny = range_active.tz_convert('America/New_York')
                     keep_mask = range_active_ny.indexer_between_time("09:30", "16:00")
                     range_active = range_active[keep_mask]
                
                if range_historical.empty and range_active.empty:
                     date_range_for_calc = pd.DatetimeIndex([])
                elif range_historical.empty:
                     date_range_for_calc = range_active
                elif range_active.empty:
                     date_range_for_calc = range_historical
                else:
                     # Append and unique handles UTC-aware indices correctly now
                     date_range_for_calc = range_historical.append(range_active).unique().sort_values()
            else:
                date_range_for_calc = pd.date_range(
                    start=calc_start_date, end=calc_end_date, freq="D"
                ).tz_localize('UTC')
            
            # Final check - ensure UTC
            if date_range_for_calc.tz is None:
                date_range_for_calc = date_range_for_calc.tz_localize('UTC')
            # logging.debug(f"DEBUG HIST: calc_start_date={calc_start_date}, first_tx_date={first_tx_date}, range_len={len(date_range_for_calc)}")

            # Determine which set of holdings to use (L1 cached or calculate now)
            # L1 Cache is purely DAILY. We cannot use it if we are calculating Intraday (mismatch in array length/mapping).
            is_intraday = interval in valid_intraday
            use_l1_cache = (
                all_holdings_qty is not None 
                and all_cash_balances is not None
                and not is_intraday # DISABLE L1 cache for intraday
                and not included_accounts_list  # Only use cache if no account filtering
            )
        
            daily_holdings_qty_to_use = None
            daily_cash_balances_to_use = None
            daily_last_prices_to_use = None

            if use_l1_cache:
                logging.info(
                    f"Hist Daily (Scope: {filter_desc}): Using L1 cached holdings..."
                )
                status_update = " Using pre-calculated daily values..."
            
                # Calculate offset if we are starting later than the first transaction
                # FIX: Use passed all_holdings_start_date if available (covers expanded ranges)
                l1_cache_start_date = (
                    all_holdings_start_date 
                    if all_holdings_start_date 
                    else transactions_df_effective["Date"].min().date()
                )
                l1_offset = (calc_start_date - l1_cache_start_date).days
                if l1_offset < 0: l1_offset = 0
            
                # Slice L1 cache
                # Ensure we don't go out of bounds
                len_needed = len(date_range_for_calc)
                daily_holdings_qty_to_use = all_holdings_qty[l1_offset : l1_offset + len_needed]
                daily_cash_balances_to_use = all_cash_balances[l1_offset : l1_offset + len_needed]
                if all_last_prices is not None:
                    daily_last_prices_to_use = all_last_prices[l1_offset : l1_offset + len_needed]
            
                logging.info(f"Hist Daily: Using L1 cache with offset {l1_offset} days.")
            else:
                logging.info(
                    f"Hist Daily (Scope: {filter_desc}): L1 cache miss. Calculating daily holdings chronologically (numba_chrono)..."
                )
                status_update = " Calculating daily values (chrono)..."
                status_update = " Calculating daily values (chrono)..."
                sorted_df = transactions_df_effective.sort_values(by=["Date", "original_index"]).copy()

                # Prepare inputs for chronological Numba function
                # Use nanosecond timestamps (np.int64) for sub-daily precision support
                # date_range_for_calc is already UTC-aware Timestamps
                date_ordinals_np = np.array(date_range_for_calc.values.astype('int64'), dtype=np.int64)
                
                tx_dates_ordinal_np = np.array(pd.to_datetime(sorted_df["Date"], utc=True).values.astype('int64'), dtype=np.int64)
                tx_symbols_np = sorted_df["Symbol"].map(symbol_to_id).values.astype(np.int64)
            
                account_ids_series = _normalize_series(sorted_df["Account"]).map(account_to_id)
                if account_ids_series.isna().any():
                    valid_mask = account_ids_series.notna()
                    sorted_df = sorted_df[valid_mask]
                    account_ids_series = account_ids_series[valid_mask]

                tx_accounts_np = account_ids_series.values.astype(np.int64)
            
                if "To Account" in sorted_df.columns:
                    tx_to_accounts_series = _normalize_series(sorted_df["To Account"]).map(account_to_id)
                else:
                    tx_to_accounts_series = pd.Series(-1, index=sorted_df.index)
                tx_to_accounts_np = tx_to_accounts_series.fillna(-1).values.astype(np.int64)
            
                tx_types_np = sorted_df["Type"].str.lower().str.strip().map(type_to_id).fillna(-1).values.astype(np.int64)
                tx_quantities_np = sorted_df["Quantity"].fillna(0.0).values.astype(np.float64)
                tx_commissions_np = sorted_df["Commission"].fillna(0.0).values.astype(np.float64)
                tx_split_ratios_np = sorted_df["Split Ratio"].fillna(0.0).values.astype(np.float64)
            
                price_col = "Price/Share" if "Price/Share" in sorted_df.columns else "Price"
                tx_prices_np = sorted_df[price_col].fillna(0.0).values.astype(np.float64) if price_col in sorted_df.columns else np.zeros(len(sorted_df), dtype=np.float64)

                split_type_id = type_to_id.get("split", -1)
                stock_split_type_id = type_to_id.get("stock split", -1)
                buy_type_id = type_to_id.get("buy", -1)
                deposit_type_id = type_to_id.get("deposit", -1)
                sell_type_id = type_to_id.get("sell", -1)
                withdrawal_type_id = type_to_id.get("withdrawal", -1)
                short_sell_type_id = type_to_id.get("short sell", -1)
                buy_to_cover_type_id = type_to_id.get("buy to cover", -1)
                transfer_type_id = type_to_id.get("transfer", -1)
                cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

                shortable_symbol_ids = np.array([symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id], dtype=np.int64)
                num_symbols = len(symbol_to_id)
                num_accounts = len(account_to_id)

                num_accounts = len(account_to_id)
            
                num_accounts = len(account_to_id)
            
                # Call Numba function
                daily_holdings_qty_to_use, daily_cash_balances_to_use, daily_last_prices_to_use = _calculate_daily_holdings_chronological_numba(
                    date_ordinals_np,
                    tx_dates_ordinal_np,
                    tx_symbols_np,
                    tx_to_accounts_np, # DEST
                    tx_accounts_np,    # SOURCE
                    tx_types_np,
                    tx_quantities_np,
                    tx_commissions_np,
                    tx_split_ratios_np,
                    tx_prices_np,
                    num_symbols,
                    num_accounts,
                    split_type_id,
                    stock_split_type_id,
                    buy_type_id,
                    deposit_type_id,
                    sell_type_id,
                    withdrawal_type_id,
                    short_sell_type_id,
                    buy_to_cover_type_id,
                    transfer_type_id,
                    cash_symbol_id,
                    STOCK_QUANTITY_CLOSE_TOLERANCE,
                    shortable_symbol_ids,
                )
                
        
            # Determine included account IDs
            if included_accounts_list:
                account_ids_to_include_set = {account_to_id.get(str(acc).upper().strip()) for acc in included_accounts_list}
                account_ids_to_include_set.discard(None)
            else:
                # Include ALL accounts if list is empty (Global Scope)
                account_ids_to_include_set = set(account_to_id.values())

            # --- VECTORIZED VALUATION ---
            daily_value_series, val_errors, val_status = _value_daily_holdings_vectorized(
                date_range=date_range_for_calc,
                daily_holdings_qty_np=daily_holdings_qty_to_use,
                daily_last_prices_np=daily_last_prices_to_use,
                daily_cash_balances_np=daily_cash_balances_to_use,
                historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
                historical_fx_yf=historical_fx_yf,
                target_currency=display_currency,
                account_currency_map=account_currency_map,
                id_to_symbol=id_to_symbol,
                id_to_account=id_to_account,
                internal_to_yf_map=internal_to_yf_map,
                account_ids_to_include_set=account_ids_to_include_set,
                default_currency=default_currency,
            )
            if val_errors:
                status_update += val_status

            # --- VECTORIZED CASH FLOW ---
            daily_net_flow_series, flow_errors = _calculate_daily_net_cash_flow_vectorized(
                date_range=date_range_for_calc,
                transactions_df=transactions_df_effective,
                target_currency=display_currency,
                historical_fx_yf=historical_fx_yf,
                default_currency=default_currency,
                included_accounts=included_accounts_list,
                historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
                internal_to_yf_map=internal_to_yf_map,
            )
            if flow_errors:
                status_update += " (flow errors)"
        
            # --- CONSTRUCT DAILY DF ---
            daily_df = pd.DataFrame({
                "value": daily_value_series,
                "net_flow": daily_net_flow_series
            }, index=date_range_for_calc)
        
        if daily_df.empty:
            status_update = " Calculating daily values..."
            # FIX: Ensure fallback calculation also uses full history for Cumulative context
            if not transactions_df_effective.empty:
                calc_start_date = transactions_df_effective["Date"].min().date()
            else:
                calc_start_date = start_date
            
            calc_end_date = end_date
            market_day_source_symbol = "SPY"
            if "SPY" not in historical_prices_yf_adjusted:
                if historical_prices_yf_adjusted:
                    market_day_source_symbol = next(iter(historical_prices_yf_adjusted.keys()))
                else:
                    market_day_source_symbol = None
            

            market_days_index = pd.Index([], dtype="object")
            logging.debug(
                f"[_load_or_calculate_daily_results] Attempting to use '{market_day_source_symbol}' for market days."
            )  # ADDED LOG
            if (
                market_day_source_symbol
                and market_day_source_symbol in historical_prices_yf_adjusted
            ):
                bench_df = historical_prices_yf_adjusted[market_day_source_symbol]
                if not bench_df.empty and isinstance(
                    bench_df.index, (pd.DatetimeIndex, pd.Index)
                ):  # ADDED: Check if index is valid before conversion
                    try:
                        # Ensure index is datetime first
                        datetime_index = pd.to_datetime(bench_df.index, errors="coerce", utc=True)
                        valid_datetime_index = datetime_index.dropna()
                        if not valid_datetime_index.empty:
                            # --- MODIFIED: Preserve intraday resolution if interval implies it ---
                            if interval in ["1m", "5m", "15m", "30m", "60m", "1h", "90m"]:
                                market_days_index = valid_datetime_index.unique() # Keep Timestamps
                            else:
                                market_days_index = pd.Index(
                                    valid_datetime_index.date
                                ).unique()  # Get unique dates (Daily)
                            logging.info(
                                f"  Successfully created market_days_index from '{market_day_source_symbol}' ({len(market_days_index)} points)."
                            )  # ADDED LOG
                        else:
                            logging.warning(
                                f"  Benchmark '{market_day_source_symbol}' index could not be converted to valid datetimes."
                            )  # ADDED LOG
                    except Exception as e_idx:
                        logging.warning(
                            f"WARN: Failed converting benchmark index for market days: {e_idx}"
                        )
                else:  # ADDED: Log if benchmark df is empty or has bad index
                    logging.warning(
                        f"  Benchmark df '{market_day_source_symbol}' is empty or has invalid index type."
                    )
            if market_days_index.empty:
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): No market days found. Using business day range."
                )
                all_dates_to_process = pd.date_range(
                    start=calc_start_date, end=calc_end_date, freq="B"
                ).date.tolist()
            else:
                # Filter the market days index by the calculation start/end dates
                all_dates_to_process = market_days_index[
                    (market_days_index >= calc_start_date)
                    & (market_days_index <= calc_end_date)
                ].tolist()
                # ADDED: Log the result of filtering market days
                logging.debug(
                    f"  Filtered market days index to range {calc_start_date} - {calc_end_date}. Found {len(all_dates_to_process)} dates."
                )
                # --- END ADDED ---

            if not all_dates_to_process:
                logging.error(
                    f"Hist ERROR (Scope: {filter_desc}): No calculation dates found in range {calc_start_date} to {calc_end_date}."
                )
                return pd.DataFrame(), False, status_update + " No calculation dates found."
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Determined {len(all_dates_to_process)} calculation dates from {min(all_dates_to_process)} to {max(all_dates_to_process)}"
            )
            # --- ADDED: Log the determined calculation dates ---
            if all_dates_to_process:  # FIX: Correct indentation
                logging.debug(
                    f"  First 5 calc dates: {all_dates_to_process[:5]}, Last 5 calc dates: {all_dates_to_process[-5:]}"
                )  # FIX: Correct indentation
            # --- END ADDED ---
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Calculating {len(all_dates_to_process)} daily metrics parallel..."
            )

            # Prepare arguments for parallel processing
            # Note: We must pass all arguments required by _calculate_daily_metrics_worker
            worker_partial = partial(
                _calculate_daily_metrics_worker,
                transactions_df=transactions_df_effective,
                historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
                historical_prices_yf_adjusted=historical_prices_yf_adjusted,
                historical_fx_yf=historical_fx_yf,
                target_currency=display_currency,
                internal_to_yf_map=internal_to_yf_map,
                account_currency_map=account_currency_map,
                default_currency=default_currency,
                manual_overrides_dict=manual_overrides_dict,
                # benchmark_symbols_yf removed
                symbol_to_id=symbol_to_id,
                id_to_symbol=id_to_symbol,
                account_to_id=account_to_id,
                id_to_account=id_to_account,
                type_to_id=type_to_id,
                currency_to_id=currency_to_id,
                id_to_currency=id_to_currency,
                calc_method=HISTORICAL_CALC_METHOD,  # Pass method from config
                included_accounts=included_accounts_list,  # Pass included_accounts_list
            )
            daily_results_list = []
            pool_start_time = time.time()
            if num_processes is None:
                try:
                    num_processes = max(1, os.cpu_count() - 1)
                except NotImplementedError:
                    num_processes = 1
            num_processes = max(1, num_processes)

            try:
                chunksize = max(1, len(all_dates_to_process) // (num_processes * 4))
            
                # --- ADDED: Progress Reporting ---
                total_dates = len(all_dates_to_process)
                last_reported_percent = -1

                # Define helper to consume results (avoids code duplication)
                def consume_results(iterator):
                    nonlocal last_reported_percent
                    for i, result in enumerate(iterator):
                        if i % 100 == 0 and i > 0:
                            logging.info(
                                f"  Processed {i}/{len(all_dates_to_process)} days..."
                            )
                        if result:
                            daily_results_list.append(result)

                        # --- Emit progress signal ---
                        if worker_signals and hasattr(worker_signals, "progress"):
                            try:
                                percent_done = int(((i + 1) / total_dates) * 100)
                                if (
                                    percent_done > last_reported_percent
                                ):  # Avoid emitting too often
                                    worker_signals.progress.emit(percent_done)
                                    last_reported_percent = percent_done
                            except Exception as e_prog:
                                logging.warning(
                                    f"Warning: Failed to emit progress: {e_prog}"
                                )
                    logging.info(
                        f"  Finished processing all {len(all_dates_to_process)} days."
                    )

                if num_processes == 1:
                    # Run synchronously in the main process
                    results_iterator = map(worker_partial, all_dates_to_process)
                    consume_results(results_iterator)
                else:
                    with multiprocessing.Pool(processes=num_processes) as pool:
                        results_iterator = pool.imap_unordered(
                            worker_partial, all_dates_to_process, chunksize=chunksize
                        )
                        consume_results(results_iterator)
            except Exception as e_pool:
                logging.error(
                    f"Hist CRITICAL (Scope: {filter_desc}): Pool failed: {e_pool}"
                )
                traceback.print_exc()
                return pd.DataFrame(), False, status_update + " Multiprocessing failed."
            finally:
                pool_end_time = time.time()
                logging.info(
                    f"Hist Daily (Scope: {filter_desc}): Pool finished. Received {len(daily_results_list)} results total."
                )
                # --- ADDED: Log sample results ---
                if daily_results_list:
                    logging.debug(f"  Sample result [0]: {daily_results_list[0]}")
                    if len(daily_results_list) > 1:  # FIX: Use > instead of &gt;
                        logging.debug(f"  Sample result [-1]: {daily_results_list[-1]}")
                # --- END ADDED ---
                # FIX: Correct indentation for this log message
                logging.info(
                    f"Hist Daily (Scope: {filter_desc}): Pool finished in {pool_end_time - pool_start_time:.2f}s."
                )

            successful_results = [
                r for r in daily_results_list if not r.get("worker_error", False)
            ]
            failed_count = len(all_dates_to_process) - len(successful_results)
            if failed_count > 0:
                status_update += f" ({failed_count} dates failed in worker)."
                # Log first failure
                for r in daily_results_list:
                    if r.get("worker_error", False):
                        logging.error(f"Worker Error Sample: {r.get('error_message', 'Unknown Error')}")
                        break
            if not successful_results:
                # --- ADDED: Log why results are empty ---
                logging.error(
                    f"Hist ERROR (Scope: {filter_desc}): No successful results from workers. daily_results_list size: {len(daily_results_list)}"
                )
                # --- END ADDED ---

                return (
                    pd.DataFrame(),
                    False,
                    status_update + " All daily calculations failed in worker.",
                )

            try:
                daily_df = pd.DataFrame(successful_results)
                daily_df["Date"] = pd.to_datetime(daily_df["Date"], utc=True)
                daily_df.set_index("Date", inplace=True)
                daily_df.sort_index(inplace=True)
                cols_to_numeric = ["value", "net_flow"]
                # Benchmarks removed from here, added later in main function
                for col in cols_to_numeric:
                    daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
                # Skip: Drop rows where 'value' is NaN (failed lookup)
                # FIX: Do NOT drop rows. This causes history truncation and incorrect TWR annualization.
                # instead, let 'value' be NaN/0 and handle in gain calculations.
                # daily_df.dropna(subset=["value"], inplace=True)
                rows_dropped = 0
                if daily_df.empty:
                    return (
                        pd.DataFrame(),
                        False,
                        status_update + " All rows dropped due to NaN portfolio value.",
                    )

            except Exception as e_proc_res:
                 logging.error(
                    f"Hist ERROR (Scope: {filter_desc}): Post-processing failed: {e_proc_res}"
                 )
                 traceback.print_exc()
                 return pd.DataFrame(), False, status_update + " Error processing results."

    # --- SHARED: Derived Metrics (Gain/Return) ---
    # Runs for both Numba and Python paths, provided daily_df is not empty.
    # Runs regardless of whether data was calculated or loaded from cache.
    if not daily_df.empty:
        # Check if derived cols missing (e.g. from Numba path or stale cache)
        # Or force recalculation if we suspect stale cache
        if "daily_gain" not in daily_df.columns or "daily_return" not in daily_df.columns:
            try:
                previous_value = daily_df["value"].shift(1)
                net_flow_filled = daily_df["net_flow"].fillna(0.0)
                daily_df["daily_gain"] = (
                    daily_df["value"] - previous_value - net_flow_filled
                )
                daily_df["daily_return"] = np.nan
            
                # FIX: Adjusted Denominator for TWR
                # Treat positive net flows (Deposits) as Start-of-Day (participating in gain)
                # Treat negative net flows (Withdrawals) as End-of-Day (capital was at risk)
                # Denom = PrevValue + max(0, NetFlow)
                adjusted_prev_value = previous_value + net_flow_filled.clip(lower=0.0)
            
                valid_denom_mask = adjusted_prev_value.notna() & (
                    abs(adjusted_prev_value) > 1e-9
                )
                daily_df.loc[valid_denom_mask, "daily_return"] = (
                    daily_df.loc[valid_denom_mask, "daily_gain"]
                    / adjusted_prev_value.loc[valid_denom_mask]
                )

                anomalies = daily_df[
                    (daily_df["value"] < 1.0) & 
                    (daily_df["net_flow"] > 1000.0) & 
                    (daily_df["daily_return"] < -0.99)
                ]
                if not anomalies.empty:
                    logging.warning("CRITICAL TWR ANOMALY DETECTED: Deposits made but Portfolio Value is Zero.")
                    logging.warning(anomalies[["value", "net_flow", "daily_return"]].head().to_string())
                zero_gain_mask = daily_df["daily_gain"].notna() & (
                    abs(daily_df["daily_gain"]) < 1e-9
                )
                zero_prev_value_mask = previous_value.notna() & (
                    abs(previous_value) <= 1e-9
                )
                daily_df.loc[zero_gain_mask & zero_prev_value_mask, "daily_return"] = 0.0
                if not daily_df.empty:
                    first_idx = daily_df.index[0]
                    daily_df.loc[first_idx, "daily_gain"] = np.nan
                    daily_df.loc[first_idx, "daily_return"] = np.nan
                
                # Validation
                if (
                    "daily_gain" not in daily_df.columns
                    or "daily_return" not in daily_df.columns
                ):
                        logging.error("Failed to calculate derived daily metrics.")
                        
                # Don't fail the whole thing, but log error
            except Exception as e_deriv:
                logging.error(f"Error calculating derived metrics: {e_deriv}")
    # --- END SHARED ---

    # --- SAVE CACHE (Only if calculated new) ---
    if not cache_valid_daily_results:
        status_update += f" {len(daily_df)} days calculated."

        if (
            use_daily_results_cache
            and daily_results_cache_file
            and daily_results_cache_key
            and not daily_df.empty
        ):
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Saving daily results to cache: {daily_results_cache_file}"
            )
            try:
                cache_dir = os.path.dirname(daily_results_cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)

                # --- Save metadata file ---
                try:
                    # Only get mtime if transactions_csv_file is a valid, existing file path
                    if (
                        transactions_csv_file
                        and isinstance(transactions_csv_file, str)
                        and transactions_csv_file.strip()
                        and os.path.exists(transactions_csv_file)
                    ):
                        csv_mtime_current = os.path.getmtime(transactions_csv_file)
                    else:
                        csv_mtime_current = None
                except OSError:
                    csv_mtime_current = None  # Cannot save mtime if file access fails
                metadata_content = {
                    "cache_key": daily_results_cache_key,
                    "cache_timestamp": datetime.now().isoformat(),
                    "csv_mtime": csv_mtime_current,
                }
                with open(metadata_cache_file, "w") as f_meta:
                    json.dump(metadata_content, f_meta, indent=2)

                # --- Save directly using to_feather ---
                daily_df.index.name = "Date"
                daily_df.reset_index().to_feather(daily_results_cache_file)
            except Exception as e_save_cache:
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): Error writing daily cache: {e_save_cache}"
                )

    return daily_df, cache_valid_daily_results, status_update


@profile
def _get_or_calculate_all_daily_holdings(
    all_transactions_df: pd.DataFrame,
    start_date: date,
    end_date: date,
    symbol_to_id: Dict,
    account_to_id: Dict,
    type_to_id: Dict,
):
    """
    Layer 1 Cache: Calculates or loads from cache the daily holdings for ALL transactions.
    """
    tx_hash = hashlib.sha256(
        pd.util.hash_pandas_object(all_transactions_df, index=True).values
    ).hexdigest()
    # Version bump for cache key
    cache_key = (
        f"ALL_HOLDINGS_v1.7_{tx_hash}_{start_date.isoformat()}_{end_date.isoformat()}"
    )

    cache_dir_base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
    if not cache_dir_base:
        logging.warning(
            "Could not find cache directory. Layer 1 caching will be disabled."
        )
        return None, None, None

    holdings_cache_dir = os.path.join(cache_dir_base, "all_holdings_cache_new")
    os.makedirs(holdings_cache_dir, exist_ok=True)

    key_file = os.path.join(holdings_cache_dir, f"{cache_key}.key")
    qty_file = os.path.join(holdings_cache_dir, f"{cache_key}_qty.npy")
    cash_file = os.path.join(holdings_cache_dir, f"{cache_key}_cash.npy")
    prices_file = os.path.join(holdings_cache_dir, f"{cache_key}_prices.npy")

    if (
        os.path.exists(key_file)
        and os.path.exists(qty_file)
        and os.path.exists(cash_file)
        and os.path.exists(prices_file)
    ):
        logging.info(
            "L1 Cache HIT: Loading pre-calculated daily holdings for all accounts."
        )
        try:
            daily_holdings_qty = np.load(qty_file)
            daily_cash_balances = np.load(cash_file)
            daily_last_prices = np.load(prices_file)
            return daily_holdings_qty, daily_cash_balances, daily_last_prices
        except Exception as e:
            logging.warning(f"L1 Cache Load Error: Failed to load numpy arrays: {e}")

    # logging.info("L1 Cache MISS: Calculating daily holdings for all accounts...")

    # --- FIX: Create a sorted copy to ensure chronological processing ---
    sorted_tx_df = all_transactions_df.sort_values(by=["Date", "original_index"]).copy()

    # Create date range for calculation - Daily freq for this L1 cache
    date_range_for_calc = pd.date_range(start=start_date, end=end_date, freq="D")
    
    # Standardize range to UTC
    if date_range_for_calc.tz is None:
        date_range_for_calc = date_range_for_calc.tz_localize('UTC')
    date_ordinals_np = np.array(date_range_for_calc.values.astype('int64'), dtype=np.int64)
    
    tx_dates_ordinal_np = np.array(pd.to_datetime(sorted_tx_df["Date"], utc=True).values.astype('int64'), dtype=np.int64)
    tx_symbols_np = (
        sorted_tx_df["Symbol"].map(symbol_to_id).fillna(-1).values.astype(np.int64)
    )
    tx_accounts_np = (
        sorted_tx_df["Account"]
        .astype(str)
        .str.upper()
        .str.strip()
        .map(account_to_id)
        .fillna(-1)
        .values.astype(np.int64)
    )

    # --- ADDED: Map 'To Account' for transfers ---
    if "To Account" in sorted_tx_df.columns:
        tx_to_accounts_series = (
            sorted_tx_df["To Account"]
            .astype(str)
            .str.upper()
            .str.strip()
            .map(account_to_id)
        )
    else:
        tx_to_accounts_series = pd.Series(-1, index=sorted_tx_df.index)
    tx_to_accounts_np = tx_to_accounts_series.fillna(-1).values.astype(np.int64)
    # --- END ADDED ---

    tx_types_np = (
        sorted_tx_df["Type"]
        .str.lower()
        .str.strip()
        .map(type_to_id)
        .fillna(-1)
        .values.astype(np.int64)
    )
    tx_quantities_np = sorted_tx_df["Quantity"].fillna(0.0).values.astype(np.float64)
    tx_commissions_np = sorted_tx_df["Commission"].fillna(0.0).values.astype(np.float64)
    tx_split_ratios_np = (
        sorted_tx_df["Split Ratio"].fillna(0.0).values.astype(np.float64)
    )
    # --- NEW: Prepare prices for fallback ---
    price_col = "Price/Share" if "Price/Share" in sorted_tx_df.columns else "Price"
    tx_prices_np = (
        sorted_tx_df[price_col].fillna(0.0).values.astype(np.float64)
        if price_col in sorted_tx_df.columns
        else np.zeros(len(sorted_tx_df), dtype=np.float64)
    )

    split_type_id = type_to_id.get("split", -1)
    stock_split_type_id = type_to_id.get("stock split", -1)
    buy_type_id = type_to_id.get("buy", -1)
    deposit_type_id = type_to_id.get("deposit", -1)
    sell_type_id = type_to_id.get("sell", -1)
    withdrawal_type_id = type_to_id.get("withdrawal", -1)
    short_sell_type_id = type_to_id.get("short sell", -1)
    buy_to_cover_type_id = type_to_id.get("buy to cover", -1)
    transfer_type_id = type_to_id.get("transfer", -1)  # ADDED
    cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)


    shortable_symbol_ids = np.array(
        [symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id],
        dtype=np.int64,
    )
    num_symbols = len(symbol_to_id)
    num_accounts = len(account_to_id)

    daily_holdings_qty, daily_cash_balances, daily_last_prices = (
        _calculate_daily_holdings_chronological_numba(
            date_ordinals_np,
            tx_dates_ordinal_np,
            tx_symbols_np,
            tx_to_accounts_np, # DEST (Arg 4)
            tx_accounts_np,    # SOURCE (Arg 5)
            tx_types_np,
            tx_quantities_np,
            tx_commissions_np,
            tx_split_ratios_np,
            tx_prices_np,  # NEW argument
            num_symbols,
            num_accounts,
            split_type_id,
            stock_split_type_id,
            buy_type_id,
            deposit_type_id,
            sell_type_id,
            withdrawal_type_id,
            short_sell_type_id,
            buy_to_cover_type_id,
            transfer_type_id,
            cash_symbol_id,
            STOCK_QUANTITY_CLOSE_TOLERANCE,
            shortable_symbol_ids,
        )
    )

    try:
        np.save(qty_file, daily_holdings_qty)
        np.save(cash_file, daily_cash_balances)
        np.save(prices_file, daily_last_prices)
        with open(key_file, "w") as f:
            f.write("valid")
        # logging.info("L1 Cache SAVE: Saved calculated daily holdings to cache.")
    except Exception as e:
        logging.error(f"L1 Cache SAVE Error: Failed to save numpy arrays: {e}")

    return daily_holdings_qty, daily_cash_balances, daily_last_prices


# --- Accumulated Gain and Resampling (Keep as is) ---
def _calculate_accumulated_gains_and_resample(
    daily_df: pd.DataFrame,
    benchmark_symbols_yf: List[str],
    interval: str,
    start_date_filter: date,  # ADDED
    end_date_filter: date,  # ADDED
) -> Tuple[pd.DataFrame, float, str]:  # Return signature unchanged for now
    """
    Calculates accumulated gains for portfolio and benchmarks, and optionally resamples the data.

    Takes the daily results DataFrame (with 'daily_return' calculated) and computes the
    cumulative gain factor (Time-Weighted Return) for the portfolio. It also calculates
    the cumulative gain factor for each benchmark based on their daily price changes
    (using adjusted prices). If the specified `interval` is 'W' (Weekly) or 'ME' (Month End),
    it resamples the data to that frequency, recalculating accumulated gains based on
    the resampled period returns.

    Args:
        daily_df (pd.DataFrame): DataFrame containing daily results, including 'value',
            'daily_return', and benchmark prices. Must be indexed by date.
        benchmark_symbols_yf (List[str]): List of YF benchmark tickers present in daily_df.
        interval (str): The desired output interval ('D', 'W', or 'ME').
        start_date_filter (date): The start date requested by the user for filtering the final output.
        end_date_filter (date): The end date requested by the user for filtering the final output.

    Returns:
        Tuple[pd.DataFrame, float, str]:
            - final_df_output (pd.DataFrame): DataFrame indexed by the chosen interval's dates,
              containing columns like 'Portfolio Value', 'Portfolio Daily Gain' (summed if resampled),
              'Portfolio Accumulated Gain', and benchmark prices/accumulated gains.
            - final_twr_factor (float): The final cumulative TWR factor for the portfolio over the
              entire period (based on daily data before resampling). np.nan if unavailable.
            - status_update (str): String describing the outcome (calculation, resampling, errors).
    """
    if daily_df.empty:
        return pd.DataFrame(), np.nan, " (No daily data for final calcs)"
    if "daily_return" not in daily_df.columns:
        return pd.DataFrame(), np.nan, " (Daily return column missing)"

    status_update = ""
    final_twr_factor = np.nan
    results_df = daily_df.copy()
    # --- ADDED: Log input DataFrame shape and tail ---
    logging.debug(
        f"[_calculate_accumulated_gains_and_resample] Input daily_df shape: {results_df.shape}"
    )
    logging.debug(f"Input daily_df tail:\n{results_df.tail().to_string()}")

    try:
        gain_factors_portfolio = 1 + results_df["daily_return"].fillna(0.0)
        # --- FIX: Prevent TWR "Death" by Zero ---
        # If any daily return is -1.0 (-100%), the factor becomes 0.0.
        # This causes the cumprod to stay 0.0 forever, ruining the graph for all future dates.
        # We replace factors near 0.0 with 1.0 (neutral 0% return) to bridge these data gaps/errors.
        zero_factor_mask = gain_factors_portfolio < 1e-6
        if zero_factor_mask.any():
             gain_factors_portfolio[zero_factor_mask] = 1.0
        
        results_df["Portfolio Accumulated Gain Daily"] = (
            gain_factors_portfolio.cumprod()
        )
        if not results_df.empty and pd.isna(results_df["daily_return"].iloc[0]):
            results_df.iloc[
                0, results_df.columns.get_loc("Portfolio Accumulated Gain Daily")
            ] = np.nan
        
        # --- MODIFIED: Calculate final_twr_factor ONLY for the requested range ---
        if not results_df.empty:
            ts_start = pd.Timestamp(start_date_filter).tz_localize(results_df.index.tz)
            ts_end = pd.Timestamp(end_date_filter).tz_localize(results_df.index.tz) + pd.Timedelta(hours=23, minutes=59)
            
            # Slice the gain factors for the requested period
            period_gain_factors = gain_factors_portfolio.loc[ts_start:ts_end]
            if not period_gain_factors.empty:
                final_twr_factor = period_gain_factors.prod()
            else:
                # Fallback to the last point if slicing yielded nothing (shouldn't happen with valid data)
                final_twr_factor = results_df["Portfolio Accumulated Gain Daily"].dropna().iloc[-1] if not results_df.empty else np.nan

        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
            if price_col in results_df.columns:
                bench_prices_no_na = results_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    bench_daily_returns = (
                        bench_prices_no_na.pct_change(fill_method=None)
                        .reindex(results_df.index)
                        .ffill()
                        .fillna(0.0)
                    )
                    gain_factors_bench = 1 + bench_daily_returns
                    accum_gains_bench = gain_factors_bench.cumprod()
                    results_df[accum_col_daily] = accum_gains_bench
                    if not results_df.empty:
                        results_df.iloc[
                            0, results_df.columns.get_loc(accum_col_daily)
                        ] = np.nan
                else:
                    results_df[accum_col_daily] = np.nan
            else:
                results_df[accum_col_daily] = np.nan
        status_update += " Daily accum gain calc complete."

        # --- NEW: Calculate Absolute Gain / ROI on Daily Basis ---
        # This provides a money-weighted perspective for all views.
        if "value" in results_df.columns and "net_flow" in results_df.columns:
            results_df["Cumulative Net Flow"] = results_df["net_flow"].fillna(0.0).cumsum()
            results_df["Absolute Gain ($)"] = (
                results_df["value"] - results_df["Cumulative Net Flow"]
            )
            denom_roi = results_df["Cumulative Net Flow"]
            # Set ROI to NaN if Cost Basis is <= 0 to avoid massive misleading spikes
            # or undefined math (infinite return).
            roi_series = (results_df["Absolute Gain ($)"] / denom_roi) * 100.0
            roi_series[denom_roi <= 0] = np.nan
            results_df["Absolute ROI (%)"] = roi_series

        # --- NEW: Calculate Drawdown from Portfolio Accumulated Gain Daily ---
        if "Portfolio Accumulated Gain Daily" in results_df.columns:
            # 1. Calculate Running Max
            running_max = results_df["Portfolio Accumulated Gain Daily"].cummax()
            # 2. Drawdown = (Current / Running Max) - 1
            # Handle potential NaNs or zeros if needed (though cummax usually safe)
            results_df["drawdown"] = (results_df["Portfolio Accumulated Gain Daily"] / running_max) - 1
            # 3. Convert to percentage? Frontend expects percentage number (e.g. -5.5 for -5.5%)?
            # Looking at PerformanceGraph.tsx, it renders `dataPoint.drawdown.toFixed(2) + '%'`.
            # If the value is -0.05 (for -5%), then * 100
            results_df["drawdown"] = results_df["drawdown"] * 100.0
            # Fill NaNs with 0.0
            results_df["drawdown"] = results_df["drawdown"].fillna(0.0)

        if interval in ["D", "1d"]:
            final_df_resampled = results_df
            final_df_resampled.rename(
                columns={
                    "Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"
                },
                inplace=True,
            )
            for bm_symbol in benchmark_symbols_yf:
                accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                accum_col_final = f"{bm_symbol} Accumulated Gain"
                if accum_col_daily in final_df_resampled.columns:
                    final_df_resampled.rename(
                        columns={accum_col_daily: accum_col_final}, inplace=True
                    )
            
            # --- FIX: Shift timestamp to Market Close (16:00 EST) ---
            # Standard YF data is 00:00 UTC, which equals 19:00 EST of the PREVIOUS day.
            # This causes the frontend to display Tuesday's data as Monday.
            # We shift it to 16:00 EST (21:00 UTC) so it falls correctly on the trading day.
            if not final_df_resampled.empty:
                 try:
                     dates = final_df_resampled.index.date
                     naive_dti = pd.DatetimeIndex(dates)
                     ny_midnight = naive_dti.tz_localize('America/New_York')
                     ny_close = ny_midnight + pd.Timedelta(hours=16)
                     final_df_resampled.index = ny_close.tz_convert('UTC')
                     logging.info("Shifted Daily index to 16:00 EST to correct visual date alignment.")
                 except Exception as e_shift:
                     logging.error(f"Failed to shift daily index: {e_shift}")

        elif (
            interval in ["W", "M", "ME"] and not results_df.empty
        ):  # <-- ADD 'M' to the check
            # logging.info(f"Hist Final: Resampling to interval '{interval}'...")
            try:
                # --- ADDED: Log before resampling ---
                logging.debug(
                    f"Resampling '{interval}': Input results_df shape: {results_df.shape}"
                )
                # --- FIX: Determine correct resampling frequency ---
                resample_freq = interval
                if interval == "M":
                    resample_freq = "ME"  # Use Month End for 'M' interval
                # --- END FIX ---

                # --- MODIFIED: Include absolute investment tracking ---
                resampling_agg = {
                    "value": "last",
                    "daily_gain": "sum",
                    "net_flow": "sum",
                    "Portfolio Accumulated Gain Daily": "last",
                    "Cumulative Net Flow": "last",
                    "Absolute Gain ($)": "last",
                    "drawdown": "last",  # Preserve drawdown
                }
                for bm_symbol in benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"
                    accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                    if price_col in results_df.columns:
                        resampling_agg[price_col] = "last"
                    if accum_col_daily in results_df.columns:
                        resampling_agg[accum_col_daily] = "last"

                final_df_resampled = results_df.resample(resample_freq).agg(
                    resampling_agg
                )

                # --- ADDED: Log after resampling ---
                logging.debug(
                    f"Resampling '{interval}': Output final_df_resampled shape: {final_df_resampled.shape}"
                )

                # --- FIX: Use Daily Accumulated Gain directly ---
                # This ensures TWR is consistent across all timeframes.
                final_df_resampled.rename(
                    columns={"Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"},
                    inplace=True
                )
                for bm_symbol in benchmark_symbols_yf:
                    accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                    accum_col_final = f"{bm_symbol} Accumulated Gain"
                    if accum_col_daily in final_df_resampled.columns:
                        final_df_resampled.rename(
                            columns={accum_col_daily: accum_col_final}, inplace=True
                        )

                # --- NEW: NO LONGER recalculating Absolute ROI here, it's summed/lasted from daily ---
                # But we do need to handle ROI on resampled data because simply taking 'last' ROI 
                # or summing ROI doesn't make sense.
                # However, Absolute Gain is additive (if using sum of gains) or 'last' (if using cumulative flow).
                # Actually, using 'last' for Absolute Gain and Cumulative Flow IS correct because they are status-at-time.
                # But let's re-calculate ROI on the resampled values to be perfectly accurate.
                denom_roi_res = final_df_resampled["Cumulative Net Flow"].replace(0, np.nan)
                final_df_resampled["Absolute ROI (%)"] = (
                    (final_df_resampled["Absolute Gain ($)"] / denom_roi_res.abs()) * 100.0
                )
                
                status_update += f" Resampled to '{interval}' with Absolute metrics."
            except Exception as e_resample:
                logging.warning(
                    f"Hist WARN: Failed resampling to interval '{interval}': {e_resample}. Returning daily results."
                )
                status_update += f" Resampling failed ('{interval}')."
                final_df_resampled = results_df
        else:
            final_df_resampled = results_df
            final_df_resampled.rename(
                columns={
                    "Portfolio Accumulated Gain Daily": "Portfolio Accumulated Gain"
                },
                inplace=True,
            )
            for bm_symbol in benchmark_symbols_yf:
                accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
                accum_col_final = f"{bm_symbol} Accumulated Gain"
                if accum_col_daily in final_df_resampled.columns:
                    final_df_resampled.rename(
                        columns={accum_col_daily: accum_col_final}, inplace=True
                    )

        columns_to_keep = [
            "value",
            "daily_gain",
            "daily_return",
            "Portfolio Accumulated Gain",
            "Absolute Gain ($)",
            "Absolute ROI (%)",
            "Cumulative Net Flow",
            "drawdown",
        ]
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_final = f"{bm_symbol} Accumulated Gain"
            if price_col in final_df_resampled.columns:
                columns_to_keep.append(price_col)
            if accum_col_final in final_df_resampled.columns:
                columns_to_keep.append(accum_col_final)
        
        # Ensure we only keep columns that actually exist
        columns_to_keep = [
            col for col in columns_to_keep if col in final_df_resampled.columns
        ]
        
        final_df_output = final_df_resampled[columns_to_keep].copy()
        final_df_output.rename(
            columns={"value": "Portfolio Value", "daily_gain": "Portfolio Daily Gain"},
            inplace=True,
        )
        # --- ADDED: Log final output DataFrame shape and tail ---
        logging.debug(
            f"[_calculate_accumulated_gains_and_resample] Final final_df_output shape: {final_df_output.shape}"
        )
        logging.debug(
            f"Final final_df_output tail:\n{final_df_output.tail().to_string()}"
        )
        if interval != "D" and "daily_return" in final_df_output.columns:
            final_df_output.drop(columns=["daily_return"], inplace=True)

        # --- ADDED: Filter final output based on requested date range ---
        if start_date_filter and end_date_filter:
            try:
                pd_start = pd.Timestamp(start_date_filter)
                pd_end = pd.Timestamp(end_date_filter) + pd.Timedelta(hours=23, minutes=59, seconds=59)
                
                # Ensure index is timezone-naive before comparison if needed
                if final_df_output.index.tz is not None:
                    final_df_output.index = final_df_output.index.tz_localize(None)
                # --- NEW BASELINE LOGIC ($t_{-1}$) ---
                # Instead of normalizing by the first visible point (t0), 
                # we want to normalize by the point immediately preceding it (t-1).
                # This ensures the first point (t0) already shows the return of that first day.
                
                # 1. Prepare a TZ-naive copy of the full resampled data for lookup
                resampled_naive = final_df_resampled.copy()
                if resampled_naive.index.tz is not None:
                    resampled_naive.index = resampled_naive.index.tz_localize(None)
                
                # 2. Identify first date in the visible range
                available_dates = final_df_output.index.sort_values()
                visible_mask = (available_dates >= pd_start) & (available_dates <= pd_end)
                visible_dates = available_dates[visible_mask]
                
                if not visible_dates.empty:
                    # 3. Find the first valid baseline in the visible range
                    # Instead of just taking [0], we search for the first point where value != 0 and not NaN
                    # This prevents "flatlining" or jumps if the start date falls on a zero-value period
                    for col in final_df_output.columns:
                        if "Accumulated Gain" in col:
                            # Search for first valid baseline
                            divisor = 0.0
                            found_valid = False
                            
                            # Try t0 first
                            t0_val = final_df_output.loc[visible_dates[0], col]
                            if pd.notnull(t0_val) and t0_val != 0:
                                divisor = t0_val
                                found_valid = True
                            else:
                                # t0 is invalid, search forwards
                                logging.debug(f"t0 baseline invalid for {col} (val={t0_val}). Searching forward...")
                                for d in visible_dates:
                                    val = final_df_output.loc[d, col]
                                    if pd.notnull(val) and val != 0:
                                        divisor = val
                                        found_valid = True
                                        logging.debug(f"Found valid baseline for {col} at {d}: {divisor}")
                                        break
                            
                            if found_valid and divisor != 0:
                                final_df_output[col] = final_df_output[col] / divisor
                            else:
                                logging.warning(f"Could not find ANY valid normalization baseline for {col} in range.")

                # finally slice the output
                final_df_output = final_df_output.loc[pd_start:pd_end]
                
                logging.debug(
                    f"Filtered and normalized final output to range: {start_date_filter} - {end_date_filter}"
                )
            except Exception as e_final_filter:
                logging.warning(f"Could not apply final date filter: {e_final_filter}")
        # --- END ADDED ---
    except Exception as e_accum:
        logging.exception(f"Hist CRITICAL: Accum gain/resample calc error")
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update

    return final_df_output, final_twr_factor, status_update


# --- Main Historical Performance Calculation Function ---
@profile
def calculate_historical_performance(
    all_transactions_df_cleaned: Optional[pd.DataFrame],  # MODIFIED: Accept DataFrame
    original_transactions_df_for_ignored: Optional[pd.DataFrame],  # MODIFIED
    ignored_indices_from_load: Set[int],  # MODIFIED
    ignored_reasons_from_load: Dict[int, str],  # MODIFIED
    start_date: date,
    end_date: date,
    interval: str,  # 'D', 'W', 'M', 'ME' for grouping, or data interval like '1h'
    benchmark_symbols_yf: List[str],  # Expects YF tickers
    display_currency: str,
    account_currency_map: Dict,
    default_currency: str,
    use_raw_data_cache: bool = True,
    use_daily_results_cache: bool = True,
    num_processes: Optional[int] = None,
    include_accounts: Optional[List[str]] = None,
    worker_signals: Optional[Any] = None,
    exclude_accounts: Optional[
        List[str]
    ] = None,  # Kept for signature, used by _prepare_historical_inputs
    # all_transactions_df_cleaned is now the first arg
    user_symbol_map: Optional[Dict[str, str]] = None,
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]] = None,
    user_excluded_symbols: Optional[Set[str]] = None,
    # ADDED: original_csv_file_path for daily_results_cache_key hash generation
    # This is needed because we no longer pass the CSV path directly to this function for loading
    original_csv_file_path: Optional[str] = None,
    calc_method: Optional[str] = None, # ADDED for benchmarking
) -> Tuple[
    pd.DataFrame,  # daily_df
    Dict[str, pd.DataFrame],  # historical_prices_yf_adjusted
    Dict[str, pd.DataFrame],  # historical_fx_yf
    str,  # final_status_str
    # pd.DataFrame, # key_ratios_df - Ratios are not calculated here
    # Dict[str, Any] # current_valuation_ratios - Ratios are not calculated here
]:
    # --- AUTO-CACHE INVALIDATION ---
    def _get_self_hash():
        try:
            with open(__file__, "rb") as f:
                return hashlib.md5(f.read()).hexdigest()[:8]
        except Exception:
            return "UNKNOWN"

    CURRENT_HIST_VERSION = f"v2.0_AUTO_{_get_self_hash()}"
    # -------------------------------
    start_time_hist = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []

    processed_warnings = set()

    # --- NEW: Generate Mappings for ALL transactions at the start ---
    (
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        type_to_id,
        currency_to_id,
        id_to_currency,
    ) = ({}, {}, {}, {}, {}, {}, {})
    try:
        if (
            all_transactions_df_cleaned is not None
            and not all_transactions_df_cleaned.empty
        ):
            (
                symbol_to_id,
                id_to_symbol,
                account_to_id,
                id_to_account,
                type_to_id,
                currency_to_id,
                id_to_currency,
            ) = generate_mappings(all_transactions_df_cleaned)
            logging.info(
                "Generated full string-to-ID mappings for historical calculation."
            )
    except Exception as e_map:
        logging.error(f"CRITICAL ERROR creating full string-to-ID mappings: {e_map}")
        return pd.DataFrame(), {}, {}, f"Error: Mapping creation failed: {e_map}"
    # --- END NEW ---

    final_twr_factor = np.nan
    daily_df = pd.DataFrame()
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}  # Ensure type
    historical_fx_yf: Dict[str, pd.DataFrame] = {}  # Ensure type

    if not MARKET_PROVIDER_AVAILABLE:
        return pd.DataFrame(), {}, {}, "Error: MarketDataProvider not available."
    if start_date >= end_date:
        return pd.DataFrame(), {}, {}, "Error: Start date must be before end date."
    if interval not in ["D", "W", "M", "ME", "1d", "1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        return pd.DataFrame(), {}, {}, f"Error: Invalid interval '{interval}'."

    clean_benchmark_symbols_yf = (
        [
            b.upper().strip()
            for b in benchmark_symbols_yf
            if isinstance(b, str) and b.strip()
        ]
        if benchmark_symbols_yf and isinstance(benchmark_symbols_yf, list)
        else []
    )
    if not benchmark_symbols_yf:
        logging.info(
            f"Hist INFO ({CURRENT_HIST_VERSION}): No benchmark symbols provided."
        )
    elif not clean_benchmark_symbols_yf and benchmark_symbols_yf:
        logging.warning(
            f"Hist WARN ({CURRENT_HIST_VERSION}): Invalid benchmark_symbols_yf type. Ignoring benchmarks."
        )
        has_warnings = True

    # --- HARDENING: Enforce original_csv_file_path for Caching ---
    # If using daily results cache, we MUST have the file path to generate a valid hash
    # that changes when the file (DB or CSV) changes.
    if use_daily_results_cache and not original_csv_file_path:
        logging.critical("CRITICAL WARNING: `calculate_historical_performance` called with `use_daily_results_cache=True` but NO `original_csv_file_path`. Disabling cache to prevent stale data.")
        use_daily_results_cache = False
    
    # --- 1. Prepare Inputs (Uses preloaded DataFrame) ---
    prep_result = _prepare_historical_inputs(
        preloaded_transactions_df=all_transactions_df_cleaned,  # PASS DF
        original_transactions_df_for_ignored=original_transactions_df_for_ignored,
        ignored_indices_from_load=ignored_indices_from_load,
        ignored_reasons_from_load=ignored_reasons_from_load,
        account_currency_map=account_currency_map,
        default_currency=default_currency,
        include_accounts=include_accounts,
        exclude_accounts=exclude_accounts,
        start_date=start_date,
        end_date=end_date,
        benchmark_symbols_yf=clean_benchmark_symbols_yf,
        display_currency=display_currency,
        current_hist_version=CURRENT_HIST_VERSION,
        raw_cache_prefix=HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        daily_cache_prefix=DAILY_RESULTS_CACHE_PATH_PREFIX,
        user_symbol_map=user_symbol_map,
        user_excluded_symbols=user_excluded_symbols,
        original_csv_file_path=original_csv_file_path,  # Pass for cache key
        interval=interval,
    )
    (
        transactions_df_effective,
        _,  # original_transactions_df_for_ignored is not needed further here
        combined_ignored_indices_prep,  # Renamed to avoid conflict if used later
        combined_ignored_reasons_prep,  # Renamed
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
    ) = prep_result

    if transactions_df_effective is None:  # Check if prep failed
        status_msg = "Error: Failed to prepare inputs from preloaded DataFrame."
        return pd.DataFrame(), {}, {}, status_msg

    status_parts.append(f"Inputs prepared ({filter_desc})")
    if combined_ignored_reasons_prep:  # Use reasons from prep
        status_parts.append(
            f"{len(combined_ignored_reasons_prep)} tx ignored (load/prep)"
        )

    # --- Determine FULL date range from the EFFECTIVE transactions ---
    try:
        if transactions_df_effective.empty:
            raise ValueError("Effective transaction DataFrame is empty.")
        # Ensure 'Date' column is datetime
        if not pd.api.types.is_datetime64_any_dtype(transactions_df_effective["Date"]):
            transactions_df_effective["Date"] = pd.to_datetime(
                transactions_df_effective["Date"], errors="coerce", utc=True
            )
            transactions_df_effective.dropna(
                subset=["Date"], inplace=True
            )  # Drop rows where date conversion failed

        if transactions_df_effective.empty:
            raise ValueError(
                "Effective transaction DataFrame became empty after date conversion."
            )

        # FIX: Ensure full_start_date covers the requested start_date
        # This prevents shape mismatch in _value_daily_holdings_vectorized when L1 cache is used.
        full_start_date = min(start_date, transactions_df_effective["Date"].min().date())
        # Ensure we have at least a few days of history for benchmarks and normalization
        # specially for intraday where we need the previous close.
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
            full_start_date = full_start_date - timedelta(days=3)
        
        full_end_date_tx = transactions_df_effective["Date"].max().date()
        # FIX REVERTED: MarketDataProvider handles +1 day for yfinance exclusivity.
        # We pass the INCLUSIVE end date (today).
        fetch_end_date = max(end_date, full_end_date_tx)
        
        logging.info(
            f"Determined full transaction range: {full_start_date} to {full_end_date_tx}. Fetching data up to {fetch_end_date} (requested start_date: {start_date}, interval: {interval})."
        )
    except Exception as e_range:
        logging.error(
            f"Could not determine full date range from transactions: {e_range}. Using original UI start/end."
        )
        full_start_date = start_date
        fetch_end_date = end_date

    # --- FIX: Clamp end_date to latest_trading_date for Daily/Weekly/Monthly to avoid flat lines "Today" ---
    # Only clamp if the user asked for a future/today date that hasn't closed yet,
    # AND there are no transactions forcing us to show that future date.
    clamped_end_date = end_date
    if interval in ["D", "1d", "W", "M", "ME"]:
         latest_trading = get_latest_trading_date()
         if end_date > latest_trading and full_end_date_tx <= latest_trading:
             logging.info(f"Clamping end_date from {end_date} to {latest_trading} (Latest Trading Date).")
             clamped_end_date = latest_trading
    
    # fetch_end_date must be at least clamped_end_date + 1 day (because YF is exclusive)
    # AND cover any transactions
    min_fetch_end = max(clamped_end_date + timedelta(days=1), full_end_date_tx + timedelta(days=1))
    # We also respect the original end_date if it was larger (e.g. for some reason needed?)
    # But usually we just need to cover the clamped range + buffer.
    # Let's just ensure we capture up to 'today' if needed for checking.
    # Actually, simplistic logic:
    fetch_end_date = max(end_date, date.today()) # Ensure we fetch up to today to populate cache correctly?
    # Better:
    fetch_end_date = max(end_date, full_end_date_tx)
    if fetch_end_date <= clamped_end_date:
         fetch_end_date = clamped_end_date + timedelta(days=1)
    
    logging.info(f"Date Range Config: Request={start_date}..{end_date}, ClampedEnd={clamped_end_date}, FetchEnd={fetch_end_date}")

    market_provider = get_shared_mdp(
        hist_data_cache_dir_name="historical_data_cache"
    )

    # This calculates daily holdings purely from transactions (no market data yet), or loads L1 cache
    # But it returns just the arrays/maps needed for *daily* scope.
    # For intraday, we might re-calculate inside _load_or_calculate if needed (cache miss logic handles it).
    all_holdings_qty, all_cash_balances, all_last_prices = _get_or_calculate_all_daily_holdings(
        all_transactions_df=all_transactions_df_cleaned,
        start_date=full_start_date,
        end_date=fetch_end_date,
        symbol_to_id=symbol_to_id,
        account_to_id=account_to_id,
        type_to_id=type_to_id,
    )

    if all_holdings_qty is None:
        status_parts.append("L1 Cache Error")
        has_errors = True  # This is a critical failure



    # --- 3. Load or Fetch ADJUSTED Historical Raw Data ---
    # We always fetch daily data for the FULL range to support valuation baseline.
    historical_prices_yf_adjusted, fetch_failed_prices = market_provider.get_historical_data(
        symbols_yf=symbols_for_stocks_and_benchmarks_yf,
        start_date=full_start_date,
        end_date=fetch_end_date,
        interval="1d",
        # Force DISABLE raw cache for intraday intervals to ensure we get fresh daily close/open for the day
        use_cache=use_raw_data_cache if interval == "1d" else False,
        # Append fetch_end_date to key to ensure we don't use stale cache for different fetch range
        cache_key=f"{raw_data_cache_key}_{fetch_end_date.isoformat()}",
    )
    
    # --- INTRADAY PATCH: Fetch high-res data for active range ---
    # If the user requested an intraday interval (e.g. 5m), we must ensure 
    # historical_prices_yf_adjusted contains these intraday timestamps for the active period.
    # Otherwise, the calculation loop (which runs at 5m resolution) will just find the same
    # 'daily' price for every 5m step, resulting in a flat line for the day.
    if interval in ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        logging.info(f"Intraday interval '{interval}' detected. Fetching coverage for {len(symbols_for_stocks_and_benchmarks_yf)} symbols...")
        try:
            # Fetch intraday data. yfinance handles 60d limit for 5m, 730d for 1h etc.
            # We use start_date from the request to capture the relevant active window.
            # CRITICAL: fetch_end_date is exclusive in yfinance. To include "Today" (if fetch_end_date is today),
            # we must extend it by 1 day.
            intraday_end_date = fetch_end_date
            if intraday_end_date >= get_est_today():
                 intraday_end_date = get_est_today() + timedelta(days=1)

            intraday_prices_adj, _ = market_provider.get_historical_data(
                symbols_yf=symbols_for_stocks_and_benchmarks_yf,
                start_date=max(start_date, get_est_today() - timedelta(days=59)) if "m" in interval else start_date,
                end_date=intraday_end_date,
                interval=interval,
                use_cache=False, # FORCE FALSE for intraday patch to ensure live data
            )
            
            if intraday_prices_adj:
                logging.info(f"Merging {len(intraday_prices_adj)} intraday price series into main price history...")
                for sym, intra_df in intraday_prices_adj.items():
                    if intra_df is None or intra_df.empty:
                        continue
                        
                    # 1. Normalize Intraday Index to UTC
                    if not isinstance(intra_df.index, pd.DatetimeIndex) or intra_df.index.tz is None:
                        intra_df.index = pd.to_datetime(intra_df.index, utc=True)
                    
                    # 2. Merge into existing Daily DF
                    # We combine them. If index overlaps, we prefer Intraday? 
                    # Actually, Daily data is at 00:00 UTC (usually). Intraday is at 09:30, 09:35 etc.
                    # So they don't overlap in time. We can just concat and sort.
                    # This gives the "step" function for history (daily) and "curve" for active day (intraday).
                    if sym in historical_prices_yf_adjusted:
                        daily_df_sym = historical_prices_yf_adjusted[sym]
                         # Normalize Daily Index to UTC if not already
                        if not isinstance(daily_df_sym.index, pd.DatetimeIndex) or daily_df_sym.index.tz is None:
                            daily_df_sym.index = pd.to_datetime(daily_df_sym.index, utc=True)
                        
                        # Concat and Sort
                        # Filter out intraday range from daily to avoid duplication if daily has 'today' row?
                        # Daily usually has 'today' 00:00 or similar.
                        # Safest is to keep daily history OLDER than intraday start, and append full intraday.
                        intra_start = intra_df.index.min()
                        daily_mask = daily_df_sym.index < intra_start
                        
                        merged_df = pd.concat([daily_df_sym.loc[daily_mask], intra_df])
                        merged_df = merged_df[~merged_df.index.duplicated(keep='last')].sort_index()
                        historical_prices_yf_adjusted[sym] = merged_df
                    else:
                        historical_prices_yf_adjusted[sym] = intra_df
        except Exception as e_intra:
            logging.error(f"Failed to fetch/merge intraday data: {e_intra}")

    
    
    # If intraday interval requested, fetch intraday data for the RECENT/ACTIVE range
    if interval in ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        # Determined strictly by loop logic in market_data, but we pass full range here.
        # Market data provider handles limits (e.g. 730d for 1h, 30d for 1m).
        # We start from max(start_date, limit) usually, but let's just pass full range and let provider clip if needed?
        # Provider clips start date. So we can just pass start_date.
        # BUT, to be efficient, we might only want intraday for the *relevant* recent period if the user asked for "All" but visualized as 1h?
        # No, if user asks for 1h, they get 1h for as long as possible.
        
        intraday_prices, _ = market_provider.get_historical_data(
            symbols_yf=symbols_for_stocks_and_benchmarks_yf,
            start_date=start_date, # Let provider clip
            end_date=fetch_end_date,
            interval=interval,
            use_cache=use_raw_data_cache,
        )
        # Merge intraday data into the adjusted map
        # Use a case-insensitive lookup for matching symbols
        adj_prices_upper = {k.upper(): k for k in historical_prices_yf_adjusted.keys()}
        

        for s, h_df in intraday_prices.items():
            s_upper = s.upper()
            if s_upper in adj_prices_upper:
                original_key = adj_prices_upper[s_upper]
                d_df = historical_prices_yf_adjusted[original_key]
                # Force standardized UTC DatetimeIndex
                d_df.index = pd.to_datetime(d_df.index, utc=True)
                
                h_df_proc = h_df.copy()
                h_df_proc.index = pd.to_datetime(h_df_proc.index, utc=True)
                
                if not h_df_proc.empty:
                    h_start_ts = h_df_proc.index.min()
                    
                    combined = pd.concat([d_df[d_df.index < h_start_ts], h_df_proc]).sort_index()
                    final_df = combined[~combined.index.duplicated(keep='last')]
                    historical_prices_yf_adjusted[original_key] = final_df
            else:
                historical_prices_yf_adjusted[s] = h_df

    logging.info(
        f"Fetching/Loading historical FX rates ({len(fx_pairs_for_api_yf)} pairs)..."
    )
    historical_fx_yf, fetch_failed_fx = market_provider.get_historical_fx_rates(
        fx_pairs_yf=fx_pairs_for_api_yf,
        start_date=full_start_date,
        end_date=fetch_end_date,
        interval="1d",
        use_cache=use_raw_data_cache,
        cache_key=raw_data_cache_key,
    )
    
    if interval in ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        intraday_fx, _ = market_provider.get_historical_fx_rates(
            fx_pairs_yf=fx_pairs_for_api_yf,
            start_date=start_date, # Let provider clip
            end_date=fetch_end_date,
            interval=interval,
            use_cache=use_raw_data_cache,
        )
        for p, h_df in intraday_fx.items():
            if p in historical_fx_yf:
                d_df = historical_fx_yf[p]
                d_df.index = pd.to_datetime(d_df.index, utc=True)
                
                h_df_proc = h_df.copy()
                h_df_proc.index = pd.to_datetime(h_df_proc.index, utc=True)
                
                if not h_df_proc.empty:
                    h_start_ts = h_df_proc.index.min()
                    
                    combined = pd.concat([d_df[d_df.index < h_start_ts], h_df_proc]).sort_index()
                    historical_fx_yf[p] = combined[~combined.index.duplicated(keep='last')]
            else:
                historical_fx_yf[p] = h_df

    fetch_failed = fetch_failed_prices or fetch_failed_fx
    if fetch_failed:
        # has_errors = True  # Treat critical fetch error as overall error
        # RELAXED: Don't fail the whole graph just because some symbols (e.g. BLV, DAL) are missing.
        # We want to show partial data.
        logging.warning("History: Some symbols failed to fetch, but proceeding with available data.")
        has_errors = False
    status_parts.append("Raw adjusted data loaded/fetched")

    if has_errors:
        status_msg = (
            "Error: Failed fetching critical historical FX/Price data via Provider."
        )
        status_parts.append("Fetch Error")
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            pd.DataFrame(),
            historical_prices_yf_adjusted,
            historical_fx_yf,
            final_status,
        )

    # --- GLOBAL STANDARDIZATION: Ensure ALL dataframes have UTC index and are backfilled ---
    # This prevents massive TWR spikes when data starts late by backfilling the first known 
    # value to the very beginning of the portfolio history.
    full_start_ts = pd.Timestamp(full_start_date, tz='UTC')
    for s_map in [historical_prices_yf_adjusted, historical_fx_yf]:
        for k, v in list(s_map.items()): # Use list to avoid mutation issues
            if not isinstance(v.index, pd.DatetimeIndex) or v.index.tz is None:
                v.index = pd.to_datetime(v.index, utc=True)
            
            # STABILIZATION: Use linear interpolation for smoothness, fallback to bfill/ffill for edges
            if v is not None and not v.empty:
                min_idx = v.index.min()
                if min_idx > full_start_ts:
                    logging.info(f"STABILIZATION: Smart-Filling {k} from {min_idx} to {full_start_ts}")
                    # Create a copy with the extended index
                    new_index = pd.Index([full_start_ts]).union(v.index)
                    v = v.reindex(new_index).sort_index()
                    
                    # Interpolate gaps linearly (requires numerical/time index)
                    # Limit direction both fills leading gaps too? No, it handles interpolation. 
                    # If we only have extensive future data and 1 start point, interpolate creates a line.
                    # But if start point is NaN (from reindex), we can't interpolate from nothing.
                    # So we use bfill() BUT we might check if the gap is "too large".
                    # However, to avoid "Jumps", backfilling constant price is better than 0.
                    # The "Jump" issue (Step 72) showed 120k jump.
                    
                    # Revised Strategy: 
                    # 1. Interpolate internal gaps (method='time').
                    # 2. bfill() leading edge to ensure valid price exists, so value != 0.
                    # 3. ffill() trailing edge.
                    v = v.interpolate(method='time').bfill().ffill()
                    
                    s_map[k] = v

    # --- 4. Derive Unadjusted Prices ---
    # logging.info("Deriving unadjusted prices using split data...")
    historical_prices_yf_unadjusted = _unadjust_prices(
        adjusted_prices_yf=historical_prices_yf_adjusted,
        yf_to_internal_map=yf_to_internal_map_hist,
        splits_by_internal_symbol=splits_by_internal_symbol,
        processed_warnings=processed_warnings,
    )
    status_parts.append("Unadjusted prices derived")
    if processed_warnings:
        has_warnings = True  # If _unadjust_prices logged warnings

    # --- Create String-to-ID Mappings (for Numba) ---

    # --- 5 & 6. Load or Calculate Daily Results ---
    # If it's None (e.g., data loaded from DB), the mtime check in daily results cache will be skipped or handled.
    
    # 1. Attempt Load from Cache
    daily_results_cache_file_path = (
        daily_results_cache_file if daily_results_cache_file else None
    )
    daily_df, cache_was_valid_daily, status_update_daily = (
        _load_or_calculate_daily_results(
            daily_results_cache_file=daily_results_cache_file_path,
            daily_results_cache_key=daily_results_cache_key,
            interval=interval,
            worker_signals=worker_signals,
            transactions_csv_file=(
                original_csv_file_path  # Pass None or actual path for mtime check
            ),
            transactions_df_effective=transactions_df_effective,
            start_date=start_date,  # FIX: Pass requested view start (2026), not loaded history start (2002)
            end_date=clamped_end_date,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            display_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            symbol_to_id=symbol_to_id,
            id_to_symbol=id_to_symbol,
            account_to_id=account_to_id,
            id_to_account=id_to_account,
            manual_overrides_dict=manual_overrides_dict,
            type_to_id=type_to_id,
            currency_to_id=currency_to_id,
            id_to_currency=id_to_currency,
            all_holdings_qty=all_holdings_qty,
            all_cash_balances=all_cash_balances,
            all_last_prices=all_last_prices,
            all_holdings_start_date=full_start_date, # PASS the expanded start date
            use_daily_results_cache=use_daily_results_cache and interval not in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"], # DISABLE CACHE if intraday
            included_accounts_list=include_accounts,
            current_hist_version=CURRENT_HIST_VERSION,
            filter_desc=filter_desc,
            calc_method=calc_method or HISTORICAL_CALC_METHOD,
        )
    )
    if status_update_daily:
        status_parts.append(status_update_daily.strip())
    if daily_df is None or daily_df.empty:
        if "Error" in status_update_daily or "failed" in status_update_daily.lower():
            has_errors = True
        else:
            has_warnings = True
        status_parts.append("Daily calc failed/empty")
    if "Error" in status_update_daily or "failed" in status_update_daily.lower():
        has_errors = True
    elif "WARN" in status_update_daily.upper():
        has_warnings = True

    # --- NEW: Fetch and Merge Benchmark Data (Optimized) ---
    # We do this AFTER getting the portfolio daily_df (cached or calc'd)
    # so that benchmark changes don't invalidate portfolio cache.
    if not daily_df.empty and clean_benchmark_symbols_yf:
        # --- DEFENSIVE: Ensure daily_df index is UTC before alignment ---
        if not isinstance(daily_df.index, pd.DatetimeIndex):
            daily_df.index = pd.to_datetime(daily_df.index, utc=True)
        if daily_df.index.tz is None:
            daily_df.index = daily_df.index.tz_localize('UTC')
            
        # DEDUPLICATION: Ensure portfolio index is unique before benchmark alignment
        if daily_df.index.duplicated().any():
            logging.info(f"Deduplicating portfolio daily_df.index: {len(daily_df)} -> {len(daily_df[~daily_df.index.duplicated()])}")
            daily_df = daily_df[~daily_df.index.duplicated(keep='last')]
        # FIX: For intraday intervals, don't fetch from full_start_date (too long)
        # Instead, fetch from start_date - 1 day to get a baseline for returns.
        benchmark_fetch_start = full_start_date
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
             benchmark_fetch_start = max(full_start_date, start_date - timedelta(days=5))

        # Use the same market_provider instance
        bench_prices_adj, _ = market_provider.get_historical_data(
            symbols_yf=clean_benchmark_symbols_yf,
            start_date=benchmark_fetch_start,
            end_date=fetch_end_date,
            use_cache=True,
            interval=interval,  # FIX: Pass interval to ensure intraday data is fetched for 1d view
            # We don't pass a monolithic cache_key here, allowing per-symbol caching
        )
        
        if bench_prices_adj:
            logging.info(f"Fetched {len(bench_prices_adj)} benchmarks. Keys: {list(bench_prices_adj.keys())}")
            for bm in clean_benchmark_symbols_yf:
                if bm in bench_prices_adj:
                    bm_df = bench_prices_adj[bm]
                    
                    if bm_df is not None and not bm_df.empty:
                        col_to_use = None
                        if "price" in bm_df.columns: col_to_use = "price"
                        elif "Adj Close" in bm_df.columns: col_to_use = "Adj Close"
                        elif "Close" in bm_df.columns: col_to_use = "Close"
                        
                        if col_to_use:
                            logging.info(f"Benchmark {bm}: Using col '{col_to_use}', count={len(bm_df)}, first={bm_df.index[0]}")
                            bm_series = bm_df[col_to_use].copy()
                        bm_series.name = f"{bm} Price"
                        
                        # Ensure index is datetime for merging
                        if not isinstance(bm_series.index, pd.DatetimeIndex) or bm_series.index.tz is None:
                            bm_series.index = pd.to_datetime(bm_series.index, utc=True)
                        
                        # DEDUPLICATION: Ensure benchmark series index is unique before reindexing
                        if bm_series.index.duplicated().any():
                            logging.info(f"Deduplicating benchmark {bm} index: {len(bm_series)} -> {len(bm_series[~bm_series.index.duplicated()])}")
                            bm_series = bm_series[~bm_series.index.duplicated(keep='last')]
                        # Merge into daily_df using reindex with 'ffill' to handle timestamp mismatches
                        # and overnight gaps. method='nearest' with tight tolerance often misses morning data.
                        try:
                            # Use generous tolerance to bridge overnight/weekend gaps
                            # MLK Day holiday requires > 72h (Fri 4pm to Tue 9:30am is ~90h)
                            tol = pd.Timedelta('120h')

                            bm_series_aligned = bm_series.reindex(
                                daily_df.index, 
                                method='ffill', 
                                tolerance=tol
                            )
                            daily_df[f"{bm} Price"] = bm_series_aligned
                            # ADDED: Diagnostic logging for Tuesday data
                            num_tuesday = daily_df.loc[daily_df.index.date == date(2026, 1, 20), f"{bm} Price"].notna().sum()
                            logging.info(f"Benchmark {bm}: Aligned. Tuesday points: {num_tuesday}")
                            
                        except Exception as e_align:
                            logging.warning(f"Benchmark alignment failed for {bm}: {e_align}. Falling back to strict join.")
                            # Fallback to strict alignment (will likely be NaN if mismatch exists)
                            daily_df[f"{bm} Price"] = bm_series
                        
                    else:
                        logging.warning(f"No price data found for benchmark {bm}")
                        daily_df[f"{bm} Price"] = np.nan
                else:
                    daily_df[f"{bm} Price"] = np.nan

    # --- 7. Resample and Calculate Final TWR & Absolute Metrics ---
    # This call handles TWR, benchmarks, resampling, normalization, and absolute metrics.
    # It replaces the manual filtering and calculation that was previously here.
    daily_df, final_twr_factor, status_update_resample = _calculate_accumulated_gains_and_resample(
        daily_df=daily_df,
        benchmark_symbols_yf=clean_benchmark_symbols_yf,
        interval=interval,
        start_date_filter=start_date,
        end_date_filter=clamped_end_date,
    )
    if status_update_resample:
        status_parts.append(status_update_resample.strip())

    # --- 8. Final Status and Return ---
    # Note: final_twr_factor is already calculated by the helper
    end_time_hist = time.time()
    logging.info(
        f"Total Historical Calc Time: {end_time_hist - start_time_hist:.2f} seconds"
    )

    final_status_prefix = "Success"
    if has_errors:
        final_status_prefix = "Finished with Errors"
    elif has_warnings:
        final_status_prefix = "Finished with Warnings"
    
    final_status_str = (
        f"{final_status_prefix} ({filter_desc})"
    )
    if status_parts:
        final_status_str += f" [{'; '.join(status_parts)}]"
    
    # Ensure TWR factor is correctly formatted in status string for UI components
    final_status_str += (
        f"|||TWR_FACTOR:{final_twr_factor:.6f}"
        if pd.notna(final_twr_factor)
        else "|||TWR_FACTOR:NaN"
    )

    return daily_df, historical_prices_yf_adjusted, historical_fx_yf, final_status_str


# --- Helper to generate mappings (Needed for standalone profiling) ---
def generate_mappings(transactions_df_effective):
    """Generates string-to-ID mappings based on the effective transaction data."""
    symbol_to_id, id_to_symbol = {}, {}
    account_to_id, id_to_account = {}, {}
    type_to_id = {}
    currency_to_id, id_to_currency = {}, {}

    all_symbols = _normalize_series(transactions_df_effective["Symbol"]).unique()
    symbol_to_id = {symbol: i for i, symbol in enumerate(all_symbols)}
    id_to_symbol = {i: symbol for symbol, i in symbol_to_id.items()}

    # --- MODIFIED: Include "To Account" in unique account list --- # type: ignore
    accounts_source = set(
        _normalize_series(transactions_df_effective["Account"]).unique()
    )
    accounts_dest = set()
    if "To Account" in transactions_df_effective.columns:
        accounts_dest = set(
            _normalize_series(transactions_df_effective["To Account"]).dropna().unique()
        )
    # --- END MODIFIED ---
    all_accounts = sorted(list(accounts_source | accounts_dest))
    # --- END MODIFIED ---
    account_to_id = {account: i for i, account in enumerate(all_accounts)}

    id_to_account = {i: account for account, i in account_to_id.items()}

    all_types = transactions_df_effective["Type"].unique()
    type_to_id = {tx_type.lower().strip(): i for i, tx_type in enumerate(all_types)}

    all_currencies = transactions_df_effective["Local Currency"].unique()
    currency_to_id = {curr: i for i, curr in enumerate(all_currencies)}
    id_to_currency = {i: curr for curr, i in currency_to_id.items()}

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


# --- Example Usage (Main block for testing this file directly) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s [%(levelname)-8s] %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Ensure this config takes precedence
    )
    multiprocessing.freeze_support()  # For PyInstaller
    # logging.info("Running portfolio_logic.py tests...")

    # --- Define test parameters ---
    test_csv_file_main = "my_transactions.csv"  # Keep a distinct name for main test CSV
    test_display_currency_main = "EUR"
    test_account_currency_map_main = {
        "SET": "THB",
        "IBKR": "USD",
        "Fidelity": "USD",
        "E*TRADE": "USD",
        "TD Ameritrade": "USD",
        "Sharebuilder": "USD",
        "Unknown": "USD",
        "Dime!": "USD",
        "ING Direct": "USD",
        "Penson": "USD",
    }
    test_default_currency_main = "USD"
    test_manual_overrides_main = {
        "XYZ": {"price": 123.45, "sector": "Tech"}
    }  # Example manual override

    # --- Load data once for all tests in __main__ ---
    (
        loaded_tx_df,
        loaded_orig_df,
        loaded_ignored_indices,
        loaded_ignored_reasons,
        load_err,
        load_warn,
        _,
    ) = load_and_clean_transactions(
        test_csv_file_main, test_account_currency_map_main, test_default_currency_main
    )

    if load_err or loaded_tx_df is None:
        logging.error(
            f"CRITICAL: Failed to load test CSV '{test_csv_file_main}'. Cannot run tests."
        )
    else:
        # logging.info(f"\n--- Testing Current Portfolio Summary (with DataFrame) ---")
        (
            summary_metrics,
            holdings_df,
            account_metrics,
            ignored_idx_summary,
            ignored_rsn_summary,
            status_summary,
        ) = calculate_portfolio_summary(
            all_transactions_df_cleaned=(
                loaded_tx_df.copy() if loaded_tx_df is not None else None
            ),
            original_transactions_df_for_ignored=(
                loaded_orig_df.copy() if loaded_orig_df is not None else None
            ),
            ignored_indices_from_load=loaded_ignored_indices,
            ignored_reasons_from_load=loaded_ignored_reasons,
            display_currency=test_display_currency_main,
            account_currency_map=test_account_currency_map_main,  # Pass explicitly
            default_currency=test_default_currency_main,  # Pass explicitly
            include_accounts=None,
            manual_overrides_dict=test_manual_overrides_main,
            user_symbol_map={},
            user_excluded_symbols=set(),
        )
        # logging.info(f"Current Summary Status (All Accounts): {status_summary}")
        if summary_metrics:
            # logging.info(f"Overall Metrics: {summary_metrics}")
            pass
        if holdings_df is not None and not holdings_df.empty:
            # logging.info(f"Holdings DF Head:\n{holdings_df.head().to_string()}")
            pass
        # ... (rest of summary logging)

        logging.info(
            "\n--- Testing Historical Performance Calculation (with DataFrame) ---"
        )
        test_start_hist = date(2023, 1, 1)
        test_end_hist = date(2024, 6, 30)
        test_interval_hist = "M"
        test_benchmarks_hist = ["SPY", "QQQ"]

        start_time_run_hist = time.time()
        hist_df, raw_prices, raw_fx, hist_status = calculate_historical_performance(
            all_transactions_df_cleaned=(
                loaded_tx_df.copy() if loaded_tx_df is not None else None
            ),
            original_transactions_df_for_ignored=(
                loaded_orig_df.copy() if loaded_orig_df is not None else None
            ),
            ignored_indices_from_load=loaded_ignored_indices,
            ignored_reasons_from_load=loaded_ignored_reasons,
            start_date=test_start_hist,
            end_date=test_end_hist,
            interval=test_interval_hist,
            benchmark_symbols_yf=test_benchmarks_hist,
            display_currency=test_display_currency_main,
            account_currency_map=test_account_currency_map_main,
            default_currency=test_default_currency_main,
            include_accounts=None,
            original_csv_file_path=test_csv_file_main,  # Pass the original CSV path for cache key context
            # ... other args like user_symbol_map, manual_overrides_dict etc.
        )
        end_time_run_hist = time.time()
        # logging.info(f"Test 'All Accounts' Hist Status: {hist_status}")
        logging.info(
            f"Test 'All Accounts' Hist Exec Time: {end_time_run_hist - start_time_run_hist:.2f} seconds"
        )
        if hist_df is not None and not hist_df.empty:
            logging.info(
                f"Test 'All Accounts' Hist DF tail:\n{hist_df.tail().to_string()}"
            )
        else:
            logging.info(f"Test 'All Accounts' Hist Result: Empty DataFrame")

    logging.info("Finished portfolio_logic.py tests.")

    # --- Profiling Section (remains largely the same, uses preloaded data) ---
    logging.info("\n--- Preparing for Single Date Profiling Run (with DataFrame) ---")
    profile_target_date = date(2024, 4, 1)
    profile_display_currency = "USD"

    if loaded_tx_df is None:
        logging.error("Cannot profile: Transaction data failed to load initially.")
    else:
        logging.info(f"Profiling target date: {profile_target_date}")
        prep_result_profile = _prepare_historical_inputs(
            preloaded_transactions_df=loaded_tx_df.copy(),  # Use preloaded
            original_transactions_df_for_ignored=(
                loaded_orig_df.copy() if loaded_orig_df is not None else None
            ),
            ignored_indices_from_load=loaded_ignored_indices,
            ignored_reasons_from_load=loaded_ignored_reasons,
            account_currency_map=test_account_currency_map_main,
            default_currency=test_default_currency_main,
            include_accounts=None,
            exclude_accounts=None,
            start_date=date(2010, 1, 1),
            end_date=profile_target_date,
            benchmark_symbols_yf=[],
            display_currency=profile_display_currency,
            user_symbol_map={},
            user_excluded_symbols=set(),
            original_csv_file_path=test_csv_file_main,  # Pass original path
        )
        (
            profile_tx_df_effective,
            _,
            _,
            _,
            _,
            _,
            _,
            profile_symbols_yf,
            profile_fx_pairs_yf,
            profile_internal_to_yf,
            profile_yf_to_internal,
            profile_splits,
            profile_raw_cache_file,
            profile_raw_cache_key,
            _,
            _,
            _,
        ) = prep_result_profile

        if profile_tx_df_effective is not None and not profile_tx_df_effective.empty:
            profile_market_provider = get_shared_mdp()
            # ... (rest of data fetching and unadjustment as before) ...
            profile_prices_adj, _ = profile_market_provider.get_historical_data(
                symbols_yf=profile_symbols_yf,
                start_date=date(2010, 1, 1),
                end_date=profile_target_date,
                use_cache=True,
                cache_key=profile_raw_cache_key,
            )
            profile_fx, _ = profile_market_provider.get_historical_fx_rates(
                fx_pairs_yf=profile_fx_pairs_yf,
                start_date=date(2010, 1, 1),
                end_date=profile_target_date,
                use_cache=True,
                cache_key=profile_raw_cache_key,
            )
            profile_prices_unadj = _unadjust_prices(
                profile_prices_adj, profile_yf_to_internal, profile_splits, set()
            )
            (
                prof_sym_id,
                prof_id_sym,
                prof_acc_id,
                prof_id_acc,
                prof_type_id,
                prof_curr_id,
                prof_id_curr,
            ) = generate_mappings(profile_tx_df_effective)

            logging.info(
                f"Calling _calculate_portfolio_value_at_date_unadjusted_numba (WARM-UP) for {profile_target_date}..."
            )
            _calculate_portfolio_value_at_date_unadjusted_numba(
                target_date=profile_target_date,
                transactions_df=profile_tx_df_effective,  # Pass effective df
                historical_prices_yf_unadjusted=profile_prices_unadj,
                historical_fx_yf=profile_fx,
                target_currency=profile_display_currency,
                internal_to_yf_map=profile_internal_to_yf,
                account_currency_map=test_account_currency_map_main,
                default_currency=test_default_currency_main,
                manual_overrides_dict=None,
                processed_warnings=set(),
                symbol_to_id=prof_sym_id,
                id_to_symbol=prof_id_sym,
                account_to_id=prof_acc_id,
                id_to_account=prof_id_acc,
                type_to_id=prof_type_id,
                currency_to_id=prof_curr_id,
                id_to_currency=prof_id_curr,
            )
            logging.info("Warm-up call finished.")
            logging.info(
                f"Calling _calculate_portfolio_value_at_date_unadjusted_numba (PROFILED) for {profile_target_date}..."
            )
            _calculate_portfolio_value_at_date_unadjusted_numba(
                target_date=profile_target_date,
                transactions_df=profile_tx_df_effective,  # Pass effective df
                historical_prices_yf_unadjusted=profile_prices_unadj,
                historical_fx_yf=profile_fx,
                target_currency=profile_display_currency,
                internal_to_yf_map=profile_internal_to_yf,
                account_currency_map=test_account_currency_map_main,
                default_currency=test_default_currency_main,
                manual_overrides_dict=None,
                processed_warnings=set(),
                symbol_to_id=prof_sym_id,
                id_to_symbol=prof_id_sym,
                account_to_id=prof_acc_id,
                id_to_account=prof_id_acc,
                type_to_id=prof_type_id,
                currency_to_id=prof_curr_id,
                id_to_currency=prof_id_curr,
            )
            logging.info("Profiling call finished.")
        else:
            logging.error(
                "Could not prepare data for profiling run (effective transactions empty)."
            )


# --- END OF MODIFIED portfolio_logic.py ---
