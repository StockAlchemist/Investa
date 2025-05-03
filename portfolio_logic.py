# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_logic.py
 Purpose:       Core logic for portfolio calculations, data fetching, and analysis.
                Handles transaction processing, current summary, and historical performance.
                Uses MarketDataProvider for external market data.
                *** MODIFIED: Current summary helpers moved to portfolio_analyzer.py ***

 Author:        Kit Matan
 Author Email:  kittiwit@gmail.com

 Created:       26/04/2025
 Modified:      2025-04-30 # <-- Updated modification date
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

# --- START OF MODIFIED portfolio_logic.py ---
import pandas as pd
from datetime import datetime, date, timedelta, UTC
import os
import json
import numpy as np
from scipy import optimize
from typing import List, Tuple, Dict, Optional, Any, Set
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

# --- Configure Logging ---
# Assumes logging is configured elsewhere (e.g., main_gui.py or config.py)
# If running standalone, uncomment and configure basicConfig here.
# logging.basicConfig(level=logging.DEBUG, format="...")

# --- Import Configuration and Utilities ---
try:
    from config import (
        LOGGING_LEVEL,
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENT_CACHE_FILE_PATH,
        HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        DAILY_RESULTS_CACHE_PATH_PREFIX,
        YFINANCE_CACHE_DURATION_HOURS,
        YFINANCE_INDEX_TICKER_MAP,
        DEFAULT_INDEX_QUERY_SYMBOLS,
        SYMBOL_MAP_TO_YFINANCE,
        YFINANCE_EXCLUDED_SYMBOLS,
        SHORTABLE_SYMBOLS,
        DEFAULT_CURRENCY,
        HISTORICAL_DEBUG_USD_CONVERSION,
        HISTORICAL_DEBUG_SET_VALUE,
        DEBUG_DATE_VALUE,
        HISTORICAL_DEBUG_DATE_VALUE,
        HISTORICAL_DEBUG_SYMBOL,
        HISTORICAL_CALC_METHOD,  # <-- ADD
        HISTORICAL_COMPARE_METHODS,  # <-- ADD
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from config.py. Exiting.")
    raise  # Re-raise to stop execution if config is essential

try:
    from finutils import (
        _get_file_hash,
        calculate_npv,
        calculate_irr,  # Still needed for historical parts
        get_cash_flows_for_symbol_account,  # Still needed for historical parts
        get_cash_flows_for_mwr,  # Still needed for historical parts
        get_conversion_rate,  # Still needed for calculate_portfolio_summary wrapper
        get_historical_price,
        get_historical_rate_via_usd_bridge,
        map_to_yf_symbol,
        safe_sum,  # Still needed for historical parts
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from finutils.py. Exiting.")
    raise

try:
    from data_loader import load_and_clean_transactions
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from data_loader.py. Exiting.")
    raise

# --- Import the NEW Market Data Provider ---
try:
    from market_data import MarketDataProvider

    MARKET_PROVIDER_AVAILABLE = True
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import MarketDataProvider from market_data.py. Exiting."
    )
    MARKET_PROVIDER_AVAILABLE = False
    raise  # Stop if provider is missing

# --- Import the NEW Portfolio Analyzer Functions ---
try:
    from portfolio_analyzer import (
        _process_transactions_to_holdings,
        _calculate_cash_balances,
        _build_summary_rows,
        _calculate_aggregate_metrics,
    )

    ANALYZER_FUNCTIONS_AVAILABLE = True
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import helper functions from portfolio_analyzer.py. Exiting."
    )
    ANALYZER_FUNCTIONS_AVAILABLE = False
    raise  # Stop if analyzer functions are missing


# --- Helper Functions (IRR/NPV, Cash Flow Gen - Already in finutils.py) ---
# --- Keep functions specific to processing/calculation logic HERE ---
# --- NOTE: The following functions have been MOVED to portfolio_analyzer.py ---
# _process_transactions_to_holdings
# _calculate_cash_balances
# _build_summary_rows
# _calculate_aggregate_metrics


# --- Main Calculation Function (Current Portfolio Summary) ---
# --- This function now orchestrates calls to portfolio_analyzer.py ---
def calculate_portfolio_summary(
    transactions_csv_file: str,
    fmp_api_key: Optional[str] = None,  # Unused
    display_currency: str = "USD",
    show_closed_positions: bool = False,
    account_currency_map: Dict = {"SET": "THB"},
    default_currency: str = "USD",
    cache_file_path: str = DEFAULT_CURRENT_CACHE_FILE_PATH,  # Used by provider
    include_accounts: Optional[List[str]] = None,
    manual_prices_dict: Optional[Dict[str, float]] = None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[Dict[str, Dict[str, float]]],
    Set[int],
    Dict[int, str],
    str,
]:
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
    logging.info(
        f"Starting Portfolio Summary Calculation (Display: {display_currency})"
    )
    start_time_summary = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []
    original_transactions_df: Optional[pd.DataFrame] = None
    all_transactions_df: Optional[pd.DataFrame] = None
    combined_ignored_indices = set()
    combined_ignored_reasons = {}
    manual_prices_effective = (
        manual_prices_dict if manual_prices_dict is not None else {}
    )
    report_date = datetime.now().date()

    # --- Check if dependencies are available ---
    if not MARKET_PROVIDER_AVAILABLE or not ANALYZER_FUNCTIONS_AVAILABLE:
        msg = "Error: Critical dependencies (MarketDataProvider or Analyzer Functions) not available."
        logging.error(msg)
        return None, None, None, set(), {}, f"Finished with Errors [{msg}]"

    # --- 1. Load & Clean Transactions ---
    (
        all_transactions_df,
        original_transactions_df,
        ignored_indices_load,
        ignored_reasons_load,
        err_load,
        warn_load,
    ) = load_and_clean_transactions(
        transactions_csv_file, account_currency_map, default_currency
    )
    combined_ignored_indices.update(ignored_indices_load)
    combined_ignored_reasons.update(ignored_reasons_load)
    if err_load:
        has_errors = True
    if warn_load:
        has_warnings = True
    if ignored_reasons_load:
        status_parts.append(f"Load/Clean Issues: {len(ignored_reasons_load)}")

    if has_errors or all_transactions_df is None:
        msg = f"Error: File not found or failed critically during load/clean: {transactions_csv_file}"
        logging.error(msg)
        status_parts.insert(0, "Load Error")
        final_status = f"Finished with Errors [{'; '.join(status_parts)}]"
        return (
            None,
            None,
            None,
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )

    # --- 2. Filter Transactions ---
    transactions_df_filtered = pd.DataFrame()
    filter_desc = "All Accounts"
    if not include_accounts:
        transactions_df_filtered = all_transactions_df.copy()
    elif isinstance(include_accounts, list):
        available_accounts = set(all_transactions_df["Account"].unique())
        valid_include = [acc for acc in include_accounts if acc in available_accounts]
        if valid_include:
            transactions_df_filtered = all_transactions_df[
                all_transactions_df["Account"].isin(valid_include)
            ].copy()
            filter_desc = f"Accounts: {', '.join(sorted(valid_include))}"
        else:
            transactions_df_filtered = pd.DataFrame()
            filter_desc = "No Valid Accounts Included"
            status_parts.append("No valid accounts after filter")
            has_warnings = True
    else:
        transactions_df_filtered = all_transactions_df.copy()
        filter_desc = "All Accounts (Invalid Filter)"
        status_parts.append("Invalid include_accounts filter type")
        has_warnings = True
    logging.info(f"Processing transactions for scope: {filter_desc}")

    # --- 3. Process Stock/ETF Transactions (using imported function) ---
    holdings, _, _, _, ignored_indices_proc, ignored_reasons_proc, warn_proc = (
        _process_transactions_to_holdings(  # <-- CALL TO IMPORTED FUNCTION
            transactions_df=transactions_df_filtered,
            default_currency=default_currency,
            shortable_symbols=SHORTABLE_SYMBOLS,
        )
    )
    combined_ignored_indices.update(ignored_indices_proc)
    combined_ignored_reasons.update(ignored_reasons_proc)
    if warn_proc:
        has_warnings = True
    if ignored_reasons_proc:
        status_parts.append(f"Processing Issues: {len(ignored_reasons_proc)}")

    # --- 4. Calculate $CASH Balances (using imported function) ---
    cash_summary, err_cash, warn_cash = (
        _calculate_cash_balances(  # <-- CALL TO IMPORTED FUNCTION
            transactions_df=transactions_df_filtered, default_currency=default_currency
        )
    )
    if err_cash:
        has_errors = True
        status_parts.append("Cash Calc Error")

    if has_errors:
        msg = "Error: Failed critically during cash balance calculation."
        logging.error(msg)
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            None,
            None,
            None,
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )

    # --- 5. Fetch Current Market Data using MarketDataProvider ---
    all_stock_symbols_internal = list(set(key[0] for key in holdings.keys()))
    required_currencies: Set[str] = set([display_currency, default_currency])
    for data in holdings.values():
        required_currencies.add(data.get("local_currency", default_currency))
    for data in cash_summary.values():
        required_currencies.add(data.get("currency", default_currency))
    required_currencies.discard(None)
    required_currencies.discard("N/A")

    # Instantiate the provider
    market_provider = MarketDataProvider(
        current_cache_file=cache_file_path
    )  # Use cache path from args

    current_stock_data_internal, current_fx_rates_vs_usd, err_fetch, warn_fetch = (
        market_provider.get_current_quotes(
            internal_stock_symbols=all_stock_symbols_internal,
            required_currencies=required_currencies,
        )
    )
    if err_fetch:
        has_errors = True
    if warn_fetch:
        has_warnings = True
    if (
        has_errors
        or current_stock_data_internal is None
        or current_fx_rates_vs_usd is None
    ):
        msg = "Error: Price/FX fetch failed critically via MarketDataProvider."
        logging.error(f"FATAL: {msg}")
        status_parts.append("Fetch Failed")
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            None,
            None,
            None,
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )
    elif has_warnings:
        status_parts.append("Fetch Warnings")

    # --- 6. Build Detailed Summary Rows (using imported function) ---
    (
        portfolio_summary_rows,
        account_market_values_local,
        account_local_currency_map,
        err_build,
        warn_build,
    ) = _build_summary_rows(  # <-- CALL TO IMPORTED FUNCTION
        holdings=holdings,
        cash_summary=cash_summary,
        current_stock_data=current_stock_data_internal,
        current_fx_rates_vs_usd=current_fx_rates_vs_usd,
        display_currency=display_currency,
        default_currency=default_currency,
        transactions_df=transactions_df_filtered,
        report_date=report_date,
        shortable_symbols=SHORTABLE_SYMBOLS,
        excluded_symbols=YFINANCE_EXCLUDED_SYMBOLS,
        manual_prices_dict=manual_prices_effective,
    )
    if err_build:
        has_errors = True
    if warn_build:
        has_warnings = True
    if has_errors:
        msg = "Error: Failed critically during summary row building (likely FX)."
        logging.error(msg)
        status_parts.append("Summary Build Error")
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            None,
            None,
            None,
            combined_ignored_indices,
            combined_ignored_reasons,
            final_status,
        )
    elif not portfolio_summary_rows and (holdings or cash_summary):
        msg = "Warning: Failed to generate summary rows (FX or other issue)."
        logging.warning(msg)
        has_warnings = True
        status_parts.append("Summary Build Failed")

    # --- 7. Create DataFrame & Calculate Aggregates (using imported function) ---
    summary_df_unfiltered = pd.DataFrame()
    overall_summary_metrics = {}
    account_level_metrics = {}

    if not portfolio_summary_rows:
        logging.warning(
            "Portfolio summary list is empty after processing. Returning empty results."
        )
        has_warnings = True
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
        }
        account_level_metrics = {}
        summary_df_unfiltered = pd.DataFrame()
    else:
        full_summary_df = pd.DataFrame(portfolio_summary_rows)
        try:
            money_cols_display = [
                c for c in full_summary_df.columns if f"({display_currency})" in c
            ]
            percent_cols = [
                "Unreal. Gain %",
                "Total Return %",
                "IRR (%)",
                "Day Change %",
            ]
            numeric_cols_to_convert = ["Quantity"] + money_cols_display + percent_cols
            if (
                f"Cumulative Investment ({display_currency})"
                not in numeric_cols_to_convert
            ):
                numeric_cols_to_convert.append(
                    f"Cumulative Investment ({display_currency})"
                )
            if f"Total Buy Cost ({display_currency})" not in numeric_cols_to_convert:
                numeric_cols_to_convert.append(f"Total Buy Cost ({display_currency})")
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

        # --- MODIFIED CALL: Removed transactions_df argument ---
        overall_summary_metrics, account_level_metrics, err_agg, warn_agg = (
            _calculate_aggregate_metrics(  # <-- CALL TO IMPORTED FUNCTION
                full_summary_df=full_summary_df,
                display_currency=display_currency,
                # transactions_df=transactions_df_filtered, # Argument removed
                report_date=report_date,
            )
        )
        # --- END MODIFIED CALL ---
        if err_agg:
            has_errors = True
        if warn_agg:
            has_warnings = True

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

    # --- 8. Filter Closed Positions ---
    summary_df_final = pd.DataFrame()
    if summary_df_unfiltered.empty:
        summary_df_final = summary_df_unfiltered
    elif not show_closed_positions:
        if (
            "Quantity" in summary_df_unfiltered.columns
            and "Symbol" in summary_df_unfiltered.columns
        ):
            held_mask = (
                (summary_df_unfiltered["Quantity"].abs() > 1e-9)
                | (summary_df_unfiltered["Symbol"] == CASH_SYMBOL_CSV)
                | (summary_df_unfiltered["Symbol"].str.startswith("Cash (", na=False))
            )
            summary_df_final = summary_df_unfiltered[held_mask].copy()
        else:
            summary_df_final = summary_df_unfiltered
            logging.warning(
                "Could not filter closed positions, Quantity/Symbol columns missing."
            )
            has_warnings = True
    else:
        summary_df_final = summary_df_unfiltered

    # --- Add Metadata to Overall Summary ---
    if overall_summary_metrics is None:
        overall_summary_metrics = {}
    overall_summary_metrics["_available_accounts"] = (
        sorted(list(all_transactions_df["Account"].unique()))
        if all_transactions_df is not None and "Account" in all_transactions_df
        else []
    )
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

    logging.info(f"Portfolio Summary Calculation Finished ({filter_desc})")
    logging.info(
        f"Total Summary Calc Time: {end_time_summary - start_time_summary:.2f} seconds"
    )

    # --- RETURN MODIFIED (6 items) ---
    return (
        overall_summary_metrics,
        summary_df_final,
        dict(account_level_metrics),
        combined_ignored_indices,
        combined_ignored_reasons,
        final_status,
    )


# =======================================================================
# --- SECTION: HISTORICAL PERFORMANCE CALCULATION FUNCTIONS (REVISED) ---
# =======================================================================
# --- NOTE: These functions remain in portfolio_logic.py as they were not moved ---


# --- Function to Unadjust Prices (Keep as is) ---
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
    logging.info("--- Starting Price Unadjustment ---")
    unadjusted_prices_yf = {}
    unadjusted_count = 0

    for yf_symbol, adj_price_df in adjusted_prices_yf.items():
        # --- DEBUG FLAG (Should be removed/made configurable) ---
        IS_DEBUG_SYMBOL = yf_symbol == "AAPL"
        if IS_DEBUG_SYMBOL:
            logging.debug(f"  Processing unadjustment for DEBUG symbol: {yf_symbol}")
        # --- END DEBUG FLAG ---

        # --- Handle Empty/Invalid Input DataFrame ---
        if adj_price_df.empty or "price" not in adj_price_df.columns:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    "    Skipping: Adjusted price DataFrame is empty or missing 'price'."
                )
            unadjusted_prices_yf[yf_symbol] = (
                adj_price_df.copy()
            )  # Return copy of input
            continue
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
                adj_price_df.copy()
            )  # Return copy of input
            continue
        # --- END Handle No Splits ---
        else:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    f"    Found splits for '{internal_symbol}'. Proceeding with unadjustment."
                )

        # --- Prepare DataFrame for Unadjustment ---
        unadj_df = adj_price_df.copy()
        # --- Robust Index Conversion ---
        if not isinstance(unadj_df.index, pd.DatetimeIndex):
            try:
                # Convert index to date objects for comparison with split dates
                unadj_df.index = pd.to_datetime(unadj_df.index, errors="coerce").date
                unadj_df = unadj_df[
                    pd.notnull(unadj_df.index)
                ]  # Remove rows where conversion failed
            except Exception as e_idx:
                warn_key = f"unadjust_index_conv_{yf_symbol}"
                if warn_key not in processed_warnings:
                    logging.warning(
                        f"Hist WARN: Failed converting index to date for {yf_symbol}: {e_idx}"
                    )
                    processed_warnings.add(warn_key)
                if IS_DEBUG_SYMBOL:
                    logging.warning(
                        "    Failed to convert index to date. Skipping unadjustment."
                    )
                unadjusted_prices_yf[yf_symbol] = (
                    adj_price_df.copy()
                )  # Return original on failure
                continue
        # --- END Robust Index Conversion ---

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

                # Apply factor to dates *before* the split date
                mask = forward_split_factor.index < split_date
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
        unadjusted_prices_yf[yf_symbol] = unadj_df[["unadjusted_price"]].rename(
            columns={"unadjusted_price": "price"}
        )

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
def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,
) -> Tuple[float, bool]:
    """
    Calculates the net external cash flow for a specific date in the target currency.

    Considers only explicit '$CASH' transactions of type 'deposit' or 'withdrawal'.
    Converts the flow amount (Quantity - Commission for deposit, -(Quantity + Commission) for withdrawal)
    from the transaction's local currency to the target_currency using historical FX rates.

    Args:
        target_date (date): The specific date for which to calculate the cash flow.
        transactions_df (pd.DataFrame): DataFrame containing all cleaned transactions.
        target_currency (str): The currency code (e.g., "USD") for the output flow value.
        historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
            (e.g., 'EUR=X') to DataFrames containing historical rates vs USD.
        account_currency_map (Dict[str, str]): Mapping of account names to their local currencies.
        default_currency (str): Default currency if not found in transaction or map.
        processed_warnings (set): A set used to track and avoid logging duplicate warnings.

    Returns:
        Tuple[float, bool]:
            - net_flow_target_curr (float): The net cash flow in the target currency for the date.
                                            Returns np.nan if a critical FX lookup fails.
            - fx_lookup_failed (bool): True if any required FX rate lookup failed critically.
    """
    # ... (Function body remains unchanged) ...
    fx_lookup_failed = False
    net_flow_target_curr = 0.0
    daily_tx = transactions_df[transactions_df["Date"].dt.date == target_date].copy()
    if daily_tx.empty:
        return 0.0, False

    external_flow_types = ["deposit", "withdrawal"]
    cash_flow_tx = daily_tx[
        (daily_tx["Symbol"] == CASH_SYMBOL_CSV)
        & (daily_tx["Type"].isin(external_flow_types))
    ].copy()
    if cash_flow_tx.empty:
        return 0.0, False

    logging.debug(
        f"Found {len(cash_flow_tx)} explicit external $CASH flows for {target_date}"
    )

    for _, row in cash_flow_tx.iterrows():
        tx_type = row["Type"]
        qty = pd.to_numeric(row["Quantity"], errors="coerce")
        commission_local_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
        commission_local = (
            0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        )
        local_currency = row["Local Currency"]
        flow_local = 0.0

        if pd.isna(qty):
            logging.warning(
                f"Skipping external cash flow row on {target_date} due to missing Quantity."
            )
            processed_warnings.add(f"missing_cash_flow_qty_{target_date}")
            continue

        if tx_type == "deposit":
            flow_local = abs(qty) - commission_local
        elif tx_type == "withdrawal":
            flow_local = -abs(qty) - commission_local

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
            logging.warning(
                f"Unexpected NaN cash flow target for {tx_type} on {target_date} after FX conversion."
            )
            net_flow_target_curr = np.nan
            fx_lookup_failed = True
            break

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
    processed_warnings: set,
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
        processed_warnings (set): A set used to track and avoid logging duplicate warnings.

    Returns:
        Tuple[float, bool]:
            - total_market_value_display_curr_agg (float): The total portfolio market value
                in the target currency for the date. Returns np.nan if any critical price/FX lookup fails.
            - any_lookup_nan_on_date (bool): True if any required price or FX rate lookup failed critically.
    """
    # ... (Function body remains unchanged) ...
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )
    if IS_DEBUG_DATE:
        logging.debug(f"--- DEBUG VALUE CALC for {target_date} ---")

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        if IS_DEBUG_DATE:
            logging.debug(f"  No transactions found up to {target_date}.")
        return 0.0, False

    holdings: Dict[Tuple[str, str], Dict] = {}
    for index, row in transactions_til_date.iterrows():
        symbol = str(row.get("Symbol", "UNKNOWN")).strip()
        account = str(row.get("Account", "Unknown"))
        local_currency_from_row = str(row.get("Local Currency", default_currency))
        holding_key_from_row = (symbol, account)
        tx_type = str(row.get("Type", "UNKNOWN_TYPE")).lower().strip()
        tx_date_row = row["Date"].date()

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
                    for h_key, h_data in holdings.items():
                        h_symbol, _ = h_key
                        if h_symbol == symbol:
                            old_qty = h_data["qty"]
                            if abs(old_qty) >= 1e-9:
                                h_data["qty"] *= split_ratio
                                if IS_DEBUG_DATE:
                                    logging.debug(
                                        f"  Applying split ratio {split_ratio} to {h_key} (Date: {tx_date_row}) Qty: {old_qty:.4f} -> {h_data['qty']:.4f}"
                                    )
                                if abs(h_data["qty"]) < 1e-9:
                                    h_data["qty"] = 0.0
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
        except Exception as e_h:
            if IS_DEBUG_DATE:
                logging.error(
                    f"      ERROR processing holding qty for {holding_key_from_row} on row index {index}: {e_h}"
                )
            pass

    cash_summary: Dict[str, Dict] = {}
    cash_transactions = transactions_til_date[
        transactions_til_date["Symbol"] == CASH_SYMBOL_CSV
    ].copy()
    if not cash_transactions.empty:

        def get_signed_quantity_cash(row):
            """Calculates cash flow including commission impact."""  # <-- Corrected Indentation
            type_lower = str(row.get("Type", "")).lower()  # <-- Corrected Indentation
            qty = pd.to_numeric(
                row.get("Quantity"), errors="coerce"
            )  # <-- Corrected Indentation
            commission_raw = pd.to_numeric(row.get("Commission"), errors="coerce")
            commission = 0.0 if pd.isna(commission_raw) else float(commission_raw)

            return (  # <-- Corrected Indentation
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
        force_fallback = internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
        if not is_stock:
            current_price_local = 1.0
        elif not force_fallback:
            yf_symbol_for_lookup = internal_to_yf_map.get(internal_symbol)
            if yf_symbol_for_lookup:
                price_val = get_historical_price(
                    yf_symbol_for_lookup, target_date, historical_prices_yf_unadjusted
                )
                if price_val is not None and pd.notna(price_val) and price_val > 1e-9:
                    current_price_local = float(price_val)
        if (pd.isna(current_price_local) or force_fallback) and is_stock:
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
                            logging.debug(
                                f"      Using Fallback Price: {current_price_local}"
                            )
            except Exception:
                pass

        if pd.isna(current_price_local):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            if DO_DETAILED_LOG:
                logging.debug(
                    f"      CRITICAL: Final price determination failed. Aborting."
                )
            break
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
@numba.jit(nopython=True, fastmath=True)
def _calculate_holdings_numba(
    target_date_ordinal,  # int: date.toordinal()
    tx_dates_ordinal_np,  # int64 array: date.toordinal()
    tx_symbols_np,  # int64 array
    tx_accounts_np,  # int64 array
    tx_types_np,  # int64 array
    tx_quantities_np,  # float64 array
    tx_prices_np,  # float64 array
    tx_commissions_np,  # float64 array
    tx_split_ratios_np,  # float64 array
    tx_local_currencies_np,  # int64 array
    num_symbols,  # int
    num_accounts,  # int
    num_currencies,  # int
    split_type_id,  # int
    stock_split_type_id,  # int
    buy_type_id,  # int
    deposit_type_id,  # int
    sell_type_id,  # int
    withdrawal_type_id,  # int
    short_sell_type_id,  # int
    buy_to_cover_type_id,  # int
    fees_type_id,  # int
    cash_symbol_id,  # int
    shortable_symbol_ids,  # int64 array
):
    # Initialize state arrays (size based on num_symbols, num_accounts)
    # Using a 2D array: index (symbol_id, account_id)
    holdings_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_cost_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    # Store currency ID per holding (symbol, account) - initialize carefully
    holdings_currency_np = np.full(
        (num_symbols, num_accounts), -1, dtype=np.int64
    )  # -1 indicates not set

    # Add arrays for shorting if needed
    holdings_short_proceeds_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)
    holdings_short_orig_qty_np = np.zeros((num_symbols, num_accounts), dtype=np.float64)

    # Cash balances (1D array indexed by account_id)
    cash_balances_np = np.zeros(num_accounts, dtype=np.float64)
    # Store currency ID per cash account
    cash_currency_np = np.full(num_accounts, -1, dtype=np.int64)  # -1 indicates not set

    num_transactions = len(tx_dates_ordinal_np)

    for i in range(num_transactions):
        tx_date = tx_dates_ordinal_np[i]
        # Numba compares dates correctly
        if tx_date > target_date_ordinal:
            continue  # Skip transactions after the target date

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
            # Initialize currency if first time seeing this cash account
            if cash_currency_np[account_id] == -1:
                cash_currency_np[account_id] = currency_id

            if type_id == buy_type_id or type_id == deposit_type_id:
                cash_balances_np[account_id] += (
                    qty - commission
                )  # Assuming qty is positive amount
            elif type_id == sell_type_id or type_id == withdrawal_type_id:
                cash_balances_np[account_id] -= (
                    qty + commission
                )  # Assuming qty is positive amount
            # Ignore dividends/fees for cash balance here, handled by stock side? Or add logic if needed.
            continue  # Move to next transaction

        # --- Handle STOCK transactions ---

        # Initialize currency if first time seeing this stock/account combo
        if holdings_currency_np[symbol_id, account_id] == -1:
            holdings_currency_np[symbol_id, account_id] = currency_id
        # Optional: Check for currency mismatch if already set (more complex in Numba)

        # --- Split Logic ---
        if type_id == split_type_id or type_id == stock_split_type_id:
            if split_ratio > 1e-9:
                # Apply split to ALL accounts holding this symbol_id
                for acc_idx in range(num_accounts):
                    current_qty_split = holdings_qty_np[symbol_id, acc_idx]
                    if abs(current_qty_split) > 1e-9:
                        holdings_qty_np[symbol_id, acc_idx] *= split_ratio
                        # Adjust cost basis? The original code didn't, stick to that for now.
                        # Adjust short original qty if shorting
                        is_shortable = False
                        for short_id in shortable_symbol_ids:
                            if symbol_id == short_id:
                                is_shortable = True
                                break
                        if current_qty_split < -1e-9 and is_shortable:
                            holdings_short_orig_qty_np[
                                symbol_id, acc_idx
                            ] *= split_ratio
                            if (
                                abs(holdings_short_orig_qty_np[symbol_id, acc_idx])
                                < 1e-9
                            ):
                                holdings_short_orig_qty_np[symbol_id, acc_idx] = 0.0

                        # Zero out if qty becomes tiny
                        if abs(holdings_qty_np[symbol_id, acc_idx]) < 1e-9:
                            holdings_qty_np[symbol_id, acc_idx] = 0.0
            # Apply commission if any (rare for splits, but handle)
            if abs(commission) > 1e-9:
                # Add commission to cost basis? Original logic added to 'commissions_local' which isn't tracked here directly for simplicity.
                # For now, let's add it to the cost basis array for the specific account.
                holdings_cost_np[
                    symbol_id, account_id
                ] += commission  # Or handle separately if needed
            continue  # Move to next transaction

        # --- Shorting Logic ---
        is_shortable_flag = False
        for short_id in shortable_symbol_ids:
            if symbol_id == short_id:
                is_shortable_flag = True
                break

        if is_shortable_flag and (
            type_id == short_sell_type_id or type_id == buy_to_cover_type_id
        ):
            qty_abs = abs(qty)
            if qty_abs <= 1e-9:
                continue  # Skip zero qty

            if type_id == short_sell_type_id:
                proceeds = (qty_abs * price) - commission
                holdings_qty_np[symbol_id, account_id] -= qty_abs
                holdings_short_proceeds_np[symbol_id, account_id] += proceeds
                holdings_short_orig_qty_np[symbol_id, account_id] += qty_abs
                # Accumulate commission in cost basis?
                holdings_cost_np[
                    symbol_id, account_id
                ] += commission  # Add commission cost
            elif type_id == buy_to_cover_type_id:
                qty_currently_short = (
                    abs(holdings_qty_np[symbol_id, account_id])
                    if holdings_qty_np[symbol_id, account_id] < -1e-9
                    else 0.0
                )
                if qty_currently_short < 1e-9:
                    continue  # Cannot cover if not short

                qty_covered = min(qty_abs, qty_currently_short)
                cost_to_cover = (qty_covered * price) + commission

                # Realized Gain calculation needs avg proceeds - complex here
                # For simplicity in Numba, let's just update qty and cost basis impact
                holdings_qty_np[symbol_id, account_id] += qty_covered
                # Cost basis impact: covering reduces liability, effectively adds cost
                holdings_cost_np[symbol_id, account_id] += cost_to_cover

                # Adjust short tracking arrays (approximate gain/loss is embedded in cost basis change)
                short_orig_qty_held = holdings_short_orig_qty_np[symbol_id, account_id]
                if short_orig_qty_held > 1e-9:
                    proceeds_ratio = qty_covered / short_orig_qty_held
                    holdings_short_proceeds_np[symbol_id, account_id] *= (
                        1.0 - proceeds_ratio
                    )
                    holdings_short_orig_qty_np[symbol_id, account_id] -= qty_covered
                else:  # Should not happen if qty_currently_short > 0, but safety check
                    holdings_short_proceeds_np[symbol_id, account_id] = 0.0
                    holdings_short_orig_qty_np[symbol_id, account_id] = 0.0

                # Zero out if needed
                if abs(holdings_short_orig_qty_np[symbol_id, account_id]) < 1e-9:
                    holdings_short_proceeds_np[symbol_id, account_id] = 0.0
                    holdings_short_orig_qty_np[symbol_id, account_id] = 0.0
                if abs(holdings_qty_np[symbol_id, account_id]) < 1e-9:
                    holdings_qty_np[symbol_id, account_id] = 0.0
                    holdings_cost_np[symbol_id, account_id] = (
                        0.0  # Reset cost if position closed
                    )

            continue  # Skip standard buy/sell

        # --- Standard Buy/Sell/Deposit/Withdrawal ---
        if type_id == buy_type_id or type_id == deposit_type_id:
            if qty > 1e-9:
                cost = (qty * price) + commission
                holdings_qty_np[symbol_id, account_id] += qty
                holdings_cost_np[symbol_id, account_id] += cost
        elif type_id == sell_type_id or type_id == withdrawal_type_id:
            if qty > 1e-9:
                held_qty = holdings_qty_np[symbol_id, account_id]
                if held_qty > 1e-9:  # Only sell if holding positive qty
                    qty_sold = min(qty, held_qty)
                    cost_basis_held = holdings_cost_np[symbol_id, account_id]
                    cost_sold = 0.0
                    if (
                        abs(held_qty) > 1e-9
                    ):  # Avoid division by zero if held_qty is zero
                        cost_sold = qty_sold * (cost_basis_held / held_qty)

                    holdings_qty_np[symbol_id, account_id] -= qty_sold
                    holdings_cost_np[symbol_id, account_id] -= cost_sold
                    # Add commission cost for sell
                    holdings_cost_np[symbol_id, account_id] += commission

                    # Zero out if position closed
                    if abs(holdings_qty_np[symbol_id, account_id]) < 1e-9:
                        holdings_qty_np[symbol_id, account_id] = 0.0
                        holdings_cost_np[symbol_id, account_id] = 0.0

        # --- Dividend/Fees ---
        # Original code adds these to separate accumulators.
        # To keep Numba simple, we might ignore these inside the Numba loop
        # OR add them to the cost basis (e.g., fees increase cost, dividends decrease cost).
        # Let's add fees to cost basis for now. Dividends are harder as they don't affect basis.
        # We might need separate arrays for realized gain, dividends, commissions if exact match is needed.
        # For now, focus on qty and cost basis.
        elif type_id == fees_type_id:
            if abs(commission) > 1e-9:
                holdings_cost_np[symbol_id, account_id] += commission

        # Ignore other types like dividend for now inside Numba

    # Return the state arrays
    return (
        holdings_qty_np,
        holdings_cost_np,
        holdings_currency_np,
        cash_balances_np,
        cash_currency_np,
    )


# --- END NUMBA HELPER FUNCTION ---


def _calculate_portfolio_value_at_date_unadjusted_numba(
    target_date: date,
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,
    # --- ADD MAPPINGS ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
) -> Tuple[float, bool]:  # type: ignore
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices (Numba version).

    Prepares NumPy arrays from transactions, calls the Numba-compiled helper function
    `_calculate_holdings_numba` to determine holdings, and then performs valuation
    using the results.
    """
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )
    if IS_DEBUG_DATE:
        logging.debug(f"--- DEBUG VALUE CALC (Numba) for {target_date} ---")

    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        if IS_DEBUG_DATE:
            logging.debug(f"  No transactions found up to {target_date}.")
        return 0.0, False

    # --- Prepare NumPy Inputs ---
    try:
        # Convert dates to ordinal integers for Numba
        target_date_ordinal = target_date.toordinal()
        tx_dates_ordinal_np = np.array(
            [d.toordinal() for d in transactions_til_date["Date"].dt.date.values],
            dtype=np.int64,
        )
        # Keep as date objects
        tx_symbols_np = (
            transactions_til_date["Symbol"].map(symbol_to_id).values.astype(np.int64)
        )
        tx_accounts_np = (
            transactions_til_date["Account"].map(account_to_id).values.astype(np.int64)
        )
        tx_types_np = (
            transactions_til_date["Type"].map(type_to_id).values.astype(np.int64)
        )
        tx_quantities_np = (
            transactions_til_date["Quantity"].fillna(0.0).values.astype(np.float64)
        )
        tx_prices_np = (
            transactions_til_date["Price/Share"].fillna(0.0).values.astype(np.float64)
        )
        tx_commissions_np = (
            transactions_til_date["Commission"].fillna(0.0).values.astype(np.float64)
        )
        tx_split_ratios_np = (
            transactions_til_date["Split Ratio"].fillna(0.0).values.astype(np.float64)
        )
        tx_local_currencies_np = (
            transactions_til_date["Local Currency"]
            .map(currency_to_id)
            .values.astype(np.int64)
        )

        # Get IDs for specific types/symbols needed inside Numba
        split_type_id = type_to_id.get("split", -1)
        stock_split_type_id = type_to_id.get("stock split", -1)
        buy_type_id = type_to_id.get("buy", -1)
        deposit_type_id = type_to_id.get("deposit", -1)
        sell_type_id = type_to_id.get("sell", -1)
        withdrawal_type_id = type_to_id.get("withdrawal", -1)
        short_sell_type_id = type_to_id.get("short sell", -1)
        buy_to_cover_type_id = type_to_id.get("buy to cover", -1)
        fees_type_id = type_to_id.get("fees", -1)
        cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

        # Map shortable symbols to IDs
        shortable_symbol_ids = np.array(
            [symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id],
            dtype=np.int64,
        )

        # Determine number of unique symbols/accounts/currencies for array sizing
        num_symbols = len(symbol_to_id)
        num_accounts = len(account_to_id)
        num_currencies = len(currency_to_id)

    except Exception as e_np_prep:
        logging.error(f"Numba Prep Error for {target_date}: {e_np_prep}")
        return np.nan, True  # Indicate failure

    # --- Call Numba Helper ---
    try:
        (
            holdings_qty_np,
            holdings_cost_np,
            holdings_currency_np,
            cash_balances_np,
            cash_currency_np,
        ) = _calculate_holdings_numba(  # type: ignore
            target_date_ordinal,  # <-- Pass ordinal date
            tx_dates_ordinal_np,  # <-- Pass ordinal dates array
            tx_symbols_np,
            tx_accounts_np,
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
            fees_type_id,
            cash_symbol_id,
            shortable_symbol_ids,
        )
    except Exception as e_numba_call:
        logging.error(f"Numba Call Error for {target_date}: {e_numba_call}")
        return np.nan, True  # Indicate failure

    # --- Valuation Loop (using results from Numba) ---
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False

    # Iterate through stock holdings
    stock_indices = np.argwhere(np.abs(holdings_qty_np) > 1e-9)
    for sym_id, acc_id in stock_indices:
        internal_symbol = id_to_symbol.get(sym_id)
        account = id_to_account.get(acc_id)
        if internal_symbol is None or account is None:
            continue  # Should not happen

        current_qty = holdings_qty_np[sym_id, acc_id]
        currency_id = holdings_currency_np[sym_id, acc_id]
        local_currency = id_to_currency.get(currency_id, default_currency)

        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )
        if pd.isna(fx_rate):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            break

        current_price_local = np.nan
        force_fallback = internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
        if not force_fallback:
            yf_symbol_for_lookup = internal_to_yf_map.get(internal_symbol)
            if yf_symbol_for_lookup:
                price_val = get_historical_price(
                    yf_symbol_for_lookup, target_date, historical_prices_yf_unadjusted
                )
                if price_val is not None and pd.notna(price_val) and price_val > 1e-9:
                    current_price_local = float(price_val)

        if pd.isna(current_price_local) or force_fallback:
            # Fallback logic (same as Python version)
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
            except Exception:
                pass

        if pd.isna(current_price_local):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            break

        market_value_local = current_qty * float(current_price_local)
        market_value_display = market_value_local * fx_rate
        if pd.isna(market_value_display):
            any_lookup_nan_on_date = True
            total_market_value_display_curr_agg = np.nan
            break
        else:
            total_market_value_display_curr_agg += market_value_display

    # Iterate through cash balances if no failure yet
    if not any_lookup_nan_on_date:
        cash_indices = np.argwhere(np.abs(cash_balances_np) > 1e-9)
        for acc_id_tuple in cash_indices:
            acc_id = acc_id_tuple[0]  # acc_id is the index from the 1D array
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
                any_lookup_nan_on_date = True
                total_market_value_display_curr_agg = np.nan
                break

            current_price_local = 1.0  # Cash price is 1.0
            market_value_local = current_qty * current_price_local
            market_value_display = market_value_local * fx_rate
            if pd.isna(market_value_display):
                any_lookup_nan_on_date = True
                total_market_value_display_curr_agg = np.nan
                break
            else:
                total_market_value_display_curr_agg += market_value_display

    if IS_DEBUG_DATE:
        logging.debug(
            f"--- DEBUG VALUE CALC (Numba) for {target_date} END --- Final Value: {total_market_value_display_curr_agg}, Lookup Failed: {any_lookup_nan_on_date}"
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
    processed_warnings: set,
    # --- ADD MAPPINGS and METHOD ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    method: str = "numba",  # Default to numba # type: ignore
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
            processed_warnings,
            symbol_to_id,
            id_to_symbol,
            account_to_id,
            id_to_account,
            type_to_id,
            currency_to_id,
            id_to_currency,
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
            processed_warnings,
        )
    else:
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
            processed_warnings,
        )


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
    benchmark_symbols_yf: List[str],
    # --- ADD MAPPINGS and METHOD --- # type: ignore
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    calc_method: str = HISTORICAL_CALC_METHOD,  # Use config default
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
        account_currency_map (Dict): Mapping of accounts to local currencies.
        default_currency (str): Default currency.
        benchmark_symbols_yf (List[str]): List of YF tickers for benchmark symbols.
        symbol_to_id (Dict): Map internal symbol string to int ID.
        account_to_id (Dict): Map account string to int ID.
        type_to_id (Dict): Map transaction type string to int ID.
        currency_to_id (Dict): Map currency string to int ID.
        calc_method (str): Calculation method ('python' or 'numba').

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
                dummy_warnings_set,
                # --- PASS MAPPINGS and METHOD ---
                symbol_to_id,
                id_to_symbol,
                account_to_id,
                id_to_account,
                type_to_id,
                currency_to_id,
                id_to_currency,
                method=calc_method,
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
            )

        benchmark_prices = {}
        bench_lookup_failed = False
        for bm_symbol in benchmark_symbols_yf:
            price = get_historical_price(
                bm_symbol, eval_date, historical_prices_yf_adjusted
            )
            bench_price = float(price) if pd.notna(price) else np.nan
            benchmark_prices[f"{bm_symbol} Price"] = bench_price
            if pd.isna(bench_price):
                bench_lookup_failed = True

        result_row = {
            "Date": eval_date,
            "value": portfolio_value,
            "net_flow": net_cash_flow,
            "value_lookup_failed": val_lookup_failed,
            "flow_lookup_failed": flow_lookup_failed,
            "bench_lookup_failed": bench_lookup_failed,
        }
        result_row.update(benchmark_prices)
        return result_row
    except Exception as e:
        logging.critical(
            f"!!! CRITICAL ERROR in worker process for date {eval_date}: {e}"
        )
        logging.exception("Worker Traceback:")
        failed_row = {"Date": eval_date, "value": np.nan, "net_flow": np.nan}
        for bm_symbol in benchmark_symbols_yf:
            failed_row[f"{bm_symbol} Price"] = np.nan
        failed_row["value_lookup_failed"] = True
        failed_row["flow_lookup_failed"] = True
        failed_row["bench_lookup_failed"] = True
        failed_row["worker_error"] = True
        failed_row["worker_error_msg"] = str(e)
        return failed_row


# --- Historical Input Preparation (Keep as is) ---
def _prepare_historical_inputs(
    transactions_csv_file: str,
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
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
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
    """
    Prepares all necessary inputs for the historical performance calculation.

    Loads and cleans transactions, applies account filtering (include/exclude),
    identifies all unique stock symbols and currencies involved, determines the
    corresponding Yahoo Finance tickers for stocks and required FX pairs, extracts
    split history, and generates cache keys and filenames for both raw historical
    data and calculated daily results based on the inputs.

    Args:
        transactions_csv_file (str): Path to the transaction CSV.
        account_currency_map (Dict): Account to currency mapping.
        default_currency (str): Default currency code.
        include_accounts (Optional[List[str]]): Accounts to include (None for all).
        exclude_accounts (Optional[List[str]]): Accounts to exclude.
        start_date (date): Start date for historical analysis.
        end_date (date): End date for historical analysis.
        benchmark_symbols_yf (List[str]): List of YF benchmark tickers.
        display_currency (str): Target currency for results.
        current_hist_version (str): Version string for daily results cache key.
        raw_cache_prefix (str): Prefix for the raw historical data cache file.
        daily_cache_prefix (str): Prefix for the daily results cache file.

    Returns:
        Tuple containing:
        - transactions_df_effective (Optional[pd.DataFrame]): Filtered transactions. None on critical load failure.
        - original_transactions_df (Optional[pd.DataFrame]): Unfiltered original transactions.
        - ignored_indices (Set[int]): Indices ignored during load/clean.
        - ignored_reasons (Dict[int, str]): Reasons for ignoring.
        - all_available_accounts_list (List[str]): All unique accounts found in the CSV.
        - included_accounts_list_sorted (List[str]): Accounts included after filtering.
        - excluded_accounts_list_sorted (List[str]): Accounts explicitly excluded.
        - symbols_for_stocks_and_benchmarks_yf (List[str]): Sorted list of YF tickers needed (portfolio + benchmarks).
        - fx_pairs_for_api_yf (List[str]): Sorted list of YF FX pair tickers needed (e.g., 'EUR=X').
        - internal_to_yf_map (Dict[str, str]): Mapping from internal symbols to YF tickers.
        - yf_to_internal_map_hist (Dict[str, str]): Reverse mapping (YF to internal).
        - splits_by_internal_symbol (Dict[str, List[Dict]]): Split history per internal symbol.
        - raw_data_cache_file (str): Full path for the raw data cache file.
        - raw_data_cache_key (str): Validation key for the raw data cache.
        - daily_results_cache_file (Optional[str]): Full path for the daily results cache file.
        - daily_results_cache_key (Optional[str]): Validation key for the daily results cache.
        - filter_desc (str): Description of the account filtering applied.
    """
    # ... (Function body remains unchanged) ...
    logging.info("Preparing inputs for historical calculation...")
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

    (
        all_transactions_df,
        original_transactions_df,
        ignored_indices,
        ignored_reasons,
        err_load,
        warn_load,
    ) = load_and_clean_transactions(
        transactions_csv_file, account_currency_map, default_currency
    )
    if all_transactions_df is None:
        logging.error(
            "ERROR in _prepare_historical_inputs: Failed to load/clean transactions."
        )
        return empty_tuple_return

    all_available_accounts_list = (
        sorted(all_transactions_df["Account"].unique().tolist())
        if "Account" in all_transactions_df.columns
        else []
    )
    available_accounts_set = set(all_available_accounts_list)
    transactions_df_effective = pd.DataFrame()
    included_accounts_list_sorted = []
    excluded_accounts_list_sorted = []
    filter_desc = "All Accounts"

    if not include_accounts:
        transactions_df_included = all_transactions_df.copy()
        included_accounts_list_sorted = sorted(list(available_accounts_set))
    else:
        valid_include_accounts = [
            acc for acc in include_accounts if acc in available_accounts_set
        ]
        if not valid_include_accounts:
            logging.warning(
                "WARN in _prepare_historical_inputs: No valid accounts to include."
            )
            return empty_tuple_return
        transactions_df_included = all_transactions_df[
            all_transactions_df["Account"].isin(valid_include_accounts)
        ].copy()
        included_accounts_list_sorted = sorted(valid_include_accounts)
        filter_desc = f"Included: {', '.join(included_accounts_list_sorted)}"

    if not exclude_accounts or not isinstance(exclude_accounts, list):
        transactions_df_effective = transactions_df_included.copy()
    else:
        valid_exclude_accounts = [
            acc for acc in exclude_accounts if acc in available_accounts_set
        ]
        if valid_exclude_accounts:
            logging.info(
                f"Hist Prep: Excluding accounts: {', '.join(sorted(valid_exclude_accounts))}"
            )
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
        return (
            transactions_df_effective,
            original_transactions_df,
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

    split_transactions = all_transactions_df[
        all_transactions_df["Type"].str.lower().isin(["split", "stock split"])
        & all_transactions_df["Split Ratio"].notna()
        & (all_transactions_df["Split Ratio"] > 0)
    ].sort_values(by="Date", ascending=True)
    splits_by_internal_symbol = {
        symbol: group[["Date", "Split Ratio"]]
        .apply(
            lambda r: {
                "Date": r["Date"].date(),
                "Split Ratio": float(r["Split Ratio"]),
            },
            axis=1,
        )
        .tolist()
        for symbol, group in split_transactions.groupby("Symbol")
    }

    all_symbols_internal = list(set(transactions_df_effective["Symbol"].unique()))
    symbols_to_fetch_yf_portfolio = []
    internal_to_yf_map = {}
    yf_to_internal_map_hist = {}
    for internal_sym in all_symbols_internal:
        if internal_sym == CASH_SYMBOL_CSV:
            continue
        yf_sym = map_to_yf_symbol(internal_sym)
        if yf_sym:
            symbols_to_fetch_yf_portfolio.append(yf_sym)
            internal_to_yf_map[internal_sym] = yf_sym
            yf_to_internal_map_hist[yf_sym] = internal_sym
    symbols_to_fetch_yf_portfolio = sorted(list(set(symbols_to_fetch_yf_portfolio)))
    symbols_for_stocks_and_benchmarks_yf = sorted(
        list(set(symbols_to_fetch_yf_portfolio + benchmark_symbols_yf))
    )

    all_currencies_in_tx = set(transactions_df_effective["Local Currency"].unique())
    all_currencies_needed = all_currencies_in_tx.union(
        {display_currency, default_currency}
    )
    all_currencies_needed.discard(None)
    all_currencies_needed.discard("N/A")
    fx_pairs_for_api_yf = sorted(
        list({f"{curr}=X" for curr in all_currencies_needed if curr != "USD"})
    )

    raw_data_cache_file = (
        f"{raw_cache_prefix}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    )
    raw_data_cache_key = f"ADJUSTED_v7::{start_date.isoformat()}::{end_date.isoformat()}::{'_'.join(sorted(symbols_for_stocks_and_benchmarks_yf))}::{'_'.join(fx_pairs_for_api_yf)}"

    daily_results_cache_file = None
    daily_results_cache_key = None
    try:
        tx_file_hash = _get_file_hash(transactions_csv_file)
        acc_map_str = json.dumps(account_currency_map, sort_keys=True)
        included_accounts_str = json.dumps(included_accounts_list_sorted)
        excluded_accounts_str = json.dumps(excluded_accounts_list_sorted)
        daily_results_cache_key = f"DAILY_RES_{current_hist_version}::{start_date.isoformat()}::{end_date.isoformat()}::{tx_file_hash}::{'_'.join(sorted(benchmark_symbols_yf))}::{display_currency}::{acc_map_str}::{default_currency}::{included_accounts_str}::{excluded_accounts_str}"
        cache_key_hash = hashlib.sha256(daily_results_cache_key.encode()).hexdigest()[
            :16
        ]
        daily_results_cache_file = f"{daily_cache_prefix}_{cache_key_hash}.json"
    except Exception as e_key:
        logging.warning(
            f"Hist Prep WARN: Could not generate daily results cache key/filename: {e_key}."
        )

    return (
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
    )


# --- Daily Results Calculation (Keep as is) ---
def _load_or_calculate_daily_results(
    use_daily_results_cache: bool,
    daily_results_cache_file: Optional[str],
    daily_results_cache_key: Optional[str],
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
    clean_benchmark_symbols_yf: List[str],
    # --- MOVED MAPPINGS HERE ---
    symbol_to_id: Dict[str, int],
    id_to_symbol: Dict[int, str],
    account_to_id: Dict[str, int],
    id_to_account: Dict[int, str],
    type_to_id: Dict[str, int],
    currency_to_id: Dict[str, int],
    id_to_currency: Dict[int, str],
    # --- Parameters with defaults follow ---
    num_processes: Optional[int] = None,
    current_hist_version: str = "v10",
    filter_desc: str = "All Accounts",
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
        clean_benchmark_symbols_yf (List[str]): List of YF benchmark tickers.
        num_processes (Optional[int]): Number of parallel processes to use. Defaults to CPU count - 1.
        current_hist_version (str): Version string used in cache key generation.
        filter_desc (str): Description of the account scope for logging.
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
    daily_df = pd.DataFrame()
    cache_valid_daily_results = False
    status_update = ""

    if use_daily_results_cache and daily_results_cache_file and daily_results_cache_key:
        if os.path.exists(daily_results_cache_file):
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Found daily cache file. Checking key..."
            )
            try:
                with open(daily_results_cache_file, "r") as f:
                    cached_daily_data = json.load(f)
                if cached_daily_data.get("cache_key") == daily_results_cache_key:
                    logging.info(
                        f"Hist Daily (Scope: {filter_desc}): Cache MATCH. Loading..."
                    )
                    results_json = cached_daily_data.get("daily_results_json")
                    if results_json:
                        try:
                            daily_df = pd.read_json(
                                StringIO(results_json), orient="split"
                            )
                            if not isinstance(daily_df, pd.DataFrame):
                                raise ValueError(
                                    "Loaded daily cache is not a DataFrame."
                                )
                            if not isinstance(daily_df.index, pd.DatetimeIndex):
                                daily_df.index = pd.to_datetime(
                                    daily_df.index, errors="coerce"
                                )
                                daily_df = daily_df[pd.notnull(daily_df.index)]
                            daily_df.sort_index(inplace=True)
                            required_cols = [
                                "value",
                                "net_flow",
                                "daily_return",
                                "daily_gain",
                            ]
                            missing_cols = [
                                c for c in required_cols if c not in daily_df.columns
                            ]
                            if not missing_cols and not daily_df.empty:
                                logging.info(
                                    f"Hist Daily (Scope: {filter_desc}): Loaded {len(daily_df)} rows from cache."
                                )
                                cache_valid_daily_results = True
                                status_update = " Daily results loaded from cache."
                            else:
                                logging.warning(
                                    f"Hist WARN (Scope: {filter_desc}): Daily cache missing columns ({missing_cols}), empty, or failed validation. Recalculating."
                                )
                                daily_df = pd.DataFrame()
                        except Exception as e_load_df:
                            logging.warning(
                                f"Hist WARN (Scope: {filter_desc}): Error deserializing/validating daily cache DF: {e_load_df}. Recalculating."
                            )
                            daily_df = pd.DataFrame()
                    else:
                        logging.warning(
                            f"Hist WARN (Scope: {filter_desc}): Daily cache missing result data. Recalculating."
                        )
                else:
                    logging.warning(
                        f"Hist WARN (Scope: {filter_desc}): Daily results cache key MISMATCH. Recalculating."
                    )
            except Exception as e_load_cache:
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): Error reading daily cache: {e_load_cache}. Recalculating."
                )
        else:
            logging.info(
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
        status_update = " Calculating daily values..."
        first_tx_date = (
            transactions_df_effective["Date"].min().date()
            if not transactions_df_effective.empty
            else start_date
        )
        calc_start_date = max(start_date, first_tx_date)
        calc_end_date = end_date
        market_day_source_symbol = (
            "SPY"
            if "SPY" in historical_prices_yf_adjusted
            else (clean_benchmark_symbols_yf[0] if clean_benchmark_symbols_yf else None)
        )
        market_days_index = pd.Index([], dtype="object")
        if (
            market_day_source_symbol
            and market_day_source_symbol in historical_prices_yf_adjusted
        ):
            bench_df = historical_prices_yf_adjusted[market_day_source_symbol]
            if not bench_df.empty and isinstance(
                bench_df.index, (pd.DatetimeIndex, pd.Index)
            ):
                try:
                    market_days_index = pd.Index(
                        pd.to_datetime(bench_df.index, errors="coerce").date
                    ).dropna()
                except Exception as e_idx:
                    logging.warning(
                        f"WARN: Failed converting benchmark index for market days: {e_idx}"
                    )
        if market_days_index.empty:
            logging.warning(
                f"Hist WARN (Scope: {filter_desc}): No market days found. Using business day range."
            )
            all_dates_to_process = pd.date_range(
                start=calc_start_date, end=calc_end_date, freq="B"
            ).date.tolist()
        else:
            all_dates_to_process = market_days_index[
                (market_days_index >= calc_start_date)
                & (market_days_index <= calc_end_date)
            ].tolist()

        if not all_dates_to_process:
            logging.error(
                f"Hist ERROR (Scope: {filter_desc}): No calculation dates found in range {calc_start_date} to {calc_end_date}."
            )
            return pd.DataFrame(), False, status_update + " No calculation dates found."
        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Determined {len(all_dates_to_process)} calculation dates from {min(all_dates_to_process)} to {max(all_dates_to_process)}"
        )
        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Calculating {len(all_dates_to_process)} daily metrics parallel..."
        )

        partial_worker = partial(
            _calculate_daily_metrics_worker,
            transactions_df=transactions_df_effective,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            target_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            benchmark_symbols_yf=clean_benchmark_symbols_yf,
            # --- PASS MAPPINGS and METHOD ---
            symbol_to_id=symbol_to_id,
            id_to_symbol=id_to_symbol,
            account_to_id=account_to_id,
            id_to_account=id_to_account,
            type_to_id=type_to_id,
            currency_to_id=currency_to_id,
            id_to_currency=id_to_currency,
            calc_method=HISTORICAL_CALC_METHOD,  # Pass method from config
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
            logging.info(
                f"Hist Daily: Starting pool with {num_processes} processes, chunksize={chunksize}"
            )
            with multiprocessing.Pool(processes=num_processes) as pool:
                results_iterator = pool.imap_unordered(
                    partial_worker, all_dates_to_process, chunksize=chunksize
                )
                for i, result in enumerate(results_iterator):
                    if i % 100 == 0 and i > 0:
                        logging.info(
                            f"  Processed {i}/{len(all_dates_to_process)} days..."
                        )
                    if result:
                        daily_results_list.append(result)
                logging.info(
                    f"  Finished processing all {len(all_dates_to_process)} days."
                )
        except Exception as e_pool:
            logging.error(
                f"Hist CRITICAL (Scope: {filter_desc}): Pool failed: {e_pool}"
            )
            traceback.print_exc()
            return pd.DataFrame(), False, status_update + " Multiprocessing failed."
        finally:
            pool_end_time = time.time()
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Pool finished in {pool_end_time - pool_start_time:.2f}s."
            )

        successful_results = [
            r for r in daily_results_list if not r.get("worker_error", False)
        ]
        failed_count = len(all_dates_to_process) - len(successful_results)
        if failed_count > 0:
            status_update += f" ({failed_count} dates failed in worker)."
        if not successful_results:
            return (
                pd.DataFrame(),
                False,
                status_update + " All daily calculations failed in worker.",
            )

        try:
            daily_df = pd.DataFrame(successful_results)
            daily_df["Date"] = pd.to_datetime(daily_df["Date"])
            daily_df.set_index("Date", inplace=True)
            daily_df.sort_index(inplace=True)
            cols_to_numeric = ["value", "net_flow"] + [
                f"{bm} Price"
                for bm in clean_benchmark_symbols_yf
                if f"{bm} Price" in daily_df.columns
            ]
            for col in cols_to_numeric:
                daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")
            initial_rows_calc = len(daily_df)
            daily_df.dropna(subset=["value"], inplace=True)
            rows_dropped = initial_rows_calc - len(daily_df)
            if rows_dropped > 0:
                status_update += (
                    f" ({rows_dropped} rows dropped post-calc due to NaN value)."
                )
            if daily_df.empty:
                return (
                    pd.DataFrame(),
                    False,
                    status_update + " All rows dropped due to NaN portfolio value.",
                )

            previous_value = daily_df["value"].shift(1)
            net_flow_filled = daily_df["net_flow"].fillna(0.0)
            daily_df["daily_gain"] = (
                daily_df["value"] - previous_value - net_flow_filled
            )
            daily_df["daily_return"] = np.nan
            valid_prev_value_mask = previous_value.notna() & (
                abs(previous_value) > 1e-9
            )
            daily_df.loc[valid_prev_value_mask, "daily_return"] = (
                daily_df.loc[valid_prev_value_mask, "daily_gain"]
                / previous_value.loc[valid_prev_value_mask]
            )
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
            if (
                "daily_gain" not in daily_df.columns
                or "daily_return" not in daily_df.columns
            ):
                return (
                    pd.DataFrame(),
                    False,
                    status_update + " Failed calc daily gain/return.",
                )
            status_update += f" {len(daily_df)} days calculated."
        except Exception as e_df_create:
            logging.error(
                f"Hist CRITICAL (Scope: {filter_desc}): Failed create/process daily DF from results: {e_df_create}"
            )
            traceback.print_exc()
            return pd.DataFrame(), False, status_update + " Error processing results."

        if (
            use_daily_results_cache
            and daily_results_cache_file
            and daily_results_cache_key
            and not daily_df.empty
        ):
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Saving daily results to cache: {daily_results_cache_file}"
            )
            cache_content = {
                "cache_key": daily_results_cache_key,
                "timestamp": datetime.now().isoformat(),
                "daily_results_json": daily_df.to_json(
                    orient="split", date_format="iso"
                ),
            }
            try:
                cache_dir = os.path.dirname(daily_results_cache_file)
                if cache_dir:
                    os.makedirs(cache_dir, exist_ok=True)
                with open(daily_results_cache_file, "w") as f:
                    json.dump(cache_content, f, indent=2)
            except Exception as e_save_cache:
                logging.warning(
                    f"Hist WARN (Scope: {filter_desc}): Error writing daily cache: {e_save_cache}"
                )

    return daily_df, cache_valid_daily_results, status_update


# --- Accumulated Gain and Resampling (Keep as is) ---
def _calculate_accumulated_gains_and_resample(
    daily_df: pd.DataFrame, benchmark_symbols_yf: List[str], interval: str
) -> Tuple[pd.DataFrame, float, str]:
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

    Returns:
        Tuple[pd.DataFrame, float, str]:
            - final_df_output (pd.DataFrame): DataFrame indexed by the chosen interval's dates,
              containing columns like 'Portfolio Value', 'Portfolio Daily Gain' (summed if resampled),
              'Portfolio Accumulated Gain', and benchmark prices/accumulated gains.
            - final_twr_factor (float): The final cumulative TWR factor for the portfolio over the
              entire period (based on daily data before resampling). np.nan if unavailable.
            - status_update (str): String describing the outcome (calculation, resampling, errors).
    """
    # ... (Function body remains unchanged) ...
    if daily_df.empty:
        return pd.DataFrame(), np.nan, " (No daily data for final calcs)"
    if "daily_return" not in daily_df.columns:
        return pd.DataFrame(), np.nan, " (Daily return column missing)"

    status_update = ""
    final_twr_factor = np.nan
    results_df = daily_df.copy()

    try:
        gain_factors_portfolio = 1 + results_df["daily_return"].fillna(0.0)
        results_df["Portfolio Accumulated Gain Daily"] = (
            gain_factors_portfolio.cumprod()
        )
        if not results_df.empty and pd.isna(results_df["daily_return"].iloc[0]):
            results_df.iloc[
                0, results_df.columns.get_loc("Portfolio Accumulated Gain Daily")
            ] = np.nan
        if (
            not results_df.empty
            and "Portfolio Accumulated Gain Daily" in results_df.columns
        ):
            last_valid_twr = (
                results_df["Portfolio Accumulated Gain Daily"].dropna().iloc[-1:]
            )
            if not last_valid_twr.empty:
                final_twr_factor = last_valid_twr.iloc[0]

        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"
            if price_col in results_df.columns:
                bench_prices_no_na = results_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    bench_daily_returns = (
                        bench_prices_no_na.pct_change()
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

        if interval == "D":
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
        elif interval in ["W", "ME"] and not results_df.empty:
            logging.info(f"Hist Final: Resampling to interval '{interval}'...")
            try:
                resampling_agg = {"value": "last", "daily_gain": "sum"}
                for bm_symbol in benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"
                    if price_col in results_df.columns:
                        resampling_agg[price_col] = "last"
                final_df_resampled = results_df.resample(interval).agg(resampling_agg)

                if (
                    "value" in final_df_resampled.columns
                    and not final_df_resampled["value"].dropna().empty
                ):
                    resampled_returns = final_df_resampled["value"].pct_change()
                    resampled_gain_factors = 1 + resampled_returns.fillna(0.0)
                    final_df_resampled["Portfolio Accumulated Gain"] = (
                        resampled_gain_factors.cumprod()
                    )
                    if not final_df_resampled.empty and pd.isna(
                        resampled_returns.iloc[0]
                    ):
                        final_df_resampled.iloc[
                            0,
                            final_df_resampled.columns.get_loc(
                                "Portfolio Accumulated Gain"
                            ),
                        ] = np.nan
                else:
                    final_df_resampled["Portfolio Accumulated Gain"] = np.nan

                for bm_symbol in benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"
                    accum_col = f"{bm_symbol} Accumulated Gain"
                    if (
                        price_col in final_df_resampled.columns
                        and not final_df_resampled[price_col].dropna().empty
                    ):
                        resampled_bench_returns = final_df_resampled[
                            price_col
                        ].pct_change()
                        resampled_bench_gain_factors = (
                            1 + resampled_bench_returns.fillna(0.0)
                        )
                        final_df_resampled[accum_col] = (
                            resampled_bench_gain_factors.cumprod()
                        )
                        if not final_df_resampled.empty and pd.isna(
                            resampled_bench_returns.iloc[0]
                        ):
                            final_df_resampled.iloc[
                                0, final_df_resampled.columns.get_loc(accum_col)
                            ] = np.nan
                    else:
                        final_df_resampled[accum_col] = np.nan
                status_update += f" Resampled to '{interval}'."
            except Exception as e_resample:
                logging.warning(
                    f"Hist WARN: Failed resampling to interval '{interval}': {e_resample}. Returning daily results."
                )
                status_update += f" Resampling failed ('{interval}')."
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

        columns_to_keep = ["value", "daily_gain", "Portfolio Accumulated Gain"]
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_final = f"{bm_symbol} Accumulated Gain"
            if price_col in final_df_resampled.columns:
                columns_to_keep.append(price_col)
            if accum_col_final in final_df_resampled.columns:
                columns_to_keep.append(accum_col_final)
        columns_to_keep = [
            col for col in columns_to_keep if col in final_df_resampled.columns
        ]
        final_df_output = final_df_resampled[columns_to_keep].copy()
        final_df_output.rename(
            columns={"value": "Portfolio Value", "daily_gain": "Portfolio Daily Gain"},
            inplace=True,
        )
        if interval != "D" and "daily_return" in final_df_output.columns:
            final_df_output.drop(columns=["daily_return"], inplace=True)

    except Exception as e_accum:
        logging.exception(f"Hist CRITICAL: Accum gain/resample calc error")
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update

    return final_df_output, final_twr_factor, status_update


# --- Historical Performance Calculation Wrapper Function (Uses MarketDataProvider) ---
def calculate_historical_performance(
    transactions_csv_file: str,
    start_date: date,
    end_date: date,
    interval: str,
    benchmark_symbols_yf: List[str],
    display_currency: str,
    account_currency_map: Dict,
    default_currency: str,
    use_raw_data_cache: bool = True,
    use_daily_results_cache: bool = True,
    num_processes: Optional[int] = None,
    include_accounts: Optional[List[str]] = None,
    exclude_accounts: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], str]:
    """
    Calculates historical portfolio performance using MarketDataProvider for data fetching.

    Orchestrates the entire historical performance calculation process:
    1. Prepares inputs (loads/filters transactions, identifies symbols/FX, gets splits, generates cache keys) using `_prepare_historical_inputs`.
    2. Instantiates `MarketDataProvider`.
    3. Loads/fetches *adjusted* historical stock/benchmark prices and FX rates using the provider.
    4. Derives *unadjusted* historical stock prices using `_unadjust_prices` and split data.
    5. Loads/calculates daily portfolio value (using unadjusted prices), net cash flow, and benchmark prices (using adjusted prices) potentially using cache and parallel processing via `_load_or_calculate_daily_results`.
    6. Calculates daily returns, cumulative TWR for portfolio and benchmarks, and resamples results to the desired interval using `_calculate_accumulated_gains_and_resample`.
    7. Returns the final performance DataFrame, raw price/FX data, and a status message.

    Args:
        transactions_csv_file (str): Path to the transaction CSV.
        start_date (date): Start date for historical analysis.
        end_date (date): End date for historical analysis.
        interval (str): Desired output interval ('D', 'W', 'ME').
        benchmark_symbols_yf (List[str]): List of YF benchmark tickers.
        display_currency (str): Target currency for results.
        account_currency_map (Dict): Account to currency mapping.
        default_currency (str): Default currency code.
        use_raw_data_cache (bool): Flag to use cache for raw price/FX data.
        use_daily_results_cache (bool): Flag to use cache for calculated daily results.
        num_processes (Optional[int]): Number of parallel processes for daily calculation.
        include_accounts (Optional[List[str]]): Accounts to include (None for all).
        exclude_accounts (Optional[List[str]]): Accounts to exclude.

    Returns:
        Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], str]:
            - final_df_filtered (pd.DataFrame): DataFrame containing the calculated historical
              performance data, indexed by date/period end. Empty on critical error.
            - historical_prices_yf_adjusted (Dict): Dictionary of raw *adjusted* historical
              prices fetched/loaded (YF Ticker -> DataFrame).
            - historical_fx_yf (Dict): Dictionary of raw historical FX rates fetched/loaded
              (YF Pair -> DataFrame).
            - final_status (str): Overall status message, including TWR factor if calculated.
    """
    # ... (Function body remains unchanged) ...
    CURRENT_HIST_VERSION = "v1.0"  # Keep version consistent
    start_time_hist = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []

    # --- Initial Checks ---
    if not MARKET_PROVIDER_AVAILABLE:
        return pd.DataFrame(), {}, {}, "Error: MarketDataProvider not available."
    if start_date >= end_date:
        return pd.DataFrame(), {}, {}, "Error: Start date must be before end date."
    if interval not in ["D", "W", "ME"]:
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
            f"Hist WARN ({CURRENT_HIST_VERSION}): Invalid benchmark_symbols_yf type provided: {type(benchmark_symbols_yf)}. Ignoring benchmarks."
        )
        has_warnings = True

    # --- 1. Prepare Inputs ---
    prep_result = _prepare_historical_inputs(
        transactions_csv_file,
        account_currency_map,
        default_currency,
        include_accounts,
        exclude_accounts,
        start_date,
        end_date,
        clean_benchmark_symbols_yf,
        display_currency,
        current_hist_version=CURRENT_HIST_VERSION,
        raw_cache_prefix=HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        daily_cache_prefix=DAILY_RESULTS_CACHE_PATH_PREFIX,
    )
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
    ) = prep_result

    if transactions_df_effective is None:
        status_msg = f"Error: Failed to prepare inputs (load/clean/filter failed)."
        return pd.DataFrame(), {}, {}, status_msg
    status_parts.append(f"Inputs prepared ({filter_desc})")
    if ignored_reasons:
        status_parts.append(f"{len(ignored_reasons)} tx ignored")

    processed_warnings = set()
    final_twr_factor = np.nan
    daily_df = pd.DataFrame()
    historical_prices_yf_adjusted = {}  # Initialize
    historical_fx_yf = {}  # Initialize

    # --- 2. Instantiate MarketDataProvider ---
    market_provider = MarketDataProvider(
        hist_raw_cache_prefix=HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX  # Pass prefix if needed
    )

    # --- 3. Load or Fetch ADJUSTED Historical Raw Data using Provider ---
    logging.info("Fetching/Loading adjusted historical prices...")
    historical_prices_yf_adjusted, fetch_failed_prices = (
        market_provider.get_historical_data(
            symbols_yf=symbols_for_stocks_and_benchmarks_yf,
            start_date=start_date,
            end_date=end_date,
            use_cache=use_raw_data_cache,
            cache_file=raw_data_cache_file,
            cache_key=raw_data_cache_key,
        )
    )
    logging.info("Fetching/Loading historical FX rates...")
    historical_fx_yf, fetch_failed_fx = market_provider.get_historical_fx_rates(
        fx_pairs_yf=fx_pairs_for_api_yf,
        start_date=start_date,
        end_date=end_date,
        use_cache=use_raw_data_cache,
        cache_file=raw_data_cache_file,
        cache_key=raw_data_cache_key,
    )
    fetch_failed = fetch_failed_prices or fetch_failed_fx  # Combine failure flags
    if fetch_failed:
        has_errors = True
    status_parts.append("Raw adjusted data loaded/fetched")

    if has_errors:
        status_msg = (
            f"Error: Failed fetching critical historical FX/Price data via Provider."
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
        )  # Return fetched data even on error

    # --- 4. Derive Unadjusted Prices ---
    logging.info("Deriving unadjusted prices using split data...")
    historical_prices_yf_unadjusted = _unadjust_prices(
        adjusted_prices_yf=historical_prices_yf_adjusted,
        yf_to_internal_map=yf_to_internal_map_hist,
        splits_by_internal_symbol=splits_by_internal_symbol,
        processed_warnings=processed_warnings,
    )
    status_parts.append("Unadjusted prices derived")
    if processed_warnings:
        has_warnings = True

    # --- Create String-to-ID Mappings ---
    # (This should ideally happen only once per historical run)
    symbol_to_id, id_to_symbol = {}, {}
    account_to_id, id_to_account = {}, {}
    type_to_id = {}
    currency_to_id, id_to_currency = {}, {}
    try:
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
        logging.info("Created string-to-ID mappings for Numba.")
    except Exception as e_map:
        logging.error(f"CRITICAL ERROR creating string-to-ID mappings: {e_map}")
        has_errors = True
        # Handle error appropriately, maybe return early

    # --- 5 & 6. Load or Calculate Daily Results ---
    daily_df, cache_was_valid_daily, status_update_daily = (
        _load_or_calculate_daily_results(
            use_daily_results_cache=use_daily_results_cache,
            daily_results_cache_file=daily_results_cache_file,
            daily_results_cache_key=daily_results_cache_key,
            start_date=start_date,
            end_date=end_date,
            transactions_df_effective=transactions_df_effective,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,  # Pass unadjusted
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,  # Pass adjusted for benchmarks
            historical_fx_yf=historical_fx_yf,
            display_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            clean_benchmark_symbols_yf=clean_benchmark_symbols_yf,
            # --- PASS MAPPINGS ---
            symbol_to_id=symbol_to_id,
            id_to_symbol=id_to_symbol,
            account_to_id=account_to_id,
            id_to_account=id_to_account,
            type_to_id=type_to_id,
            currency_to_id=currency_to_id,
            id_to_currency=id_to_currency,
            num_processes=num_processes,
            current_hist_version=CURRENT_HIST_VERSION,
            filter_desc=filter_desc,
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

    if has_errors or daily_df.empty:
        final_status_prefix = (
            "Finished with Errors"
            if has_errors
            else ("Finished with Warnings" if has_warnings else "Success")
        )
        final_status = f"{final_status_prefix} ({filter_desc})" + (
            f" [{'; '.join(status_parts)}]" if status_parts else ""
        )
        return (
            pd.DataFrame(),
            historical_prices_yf_adjusted,
            historical_fx_yf,
            final_status,
        )  # Return fetched data

    # --- 7. Calculate Accumulated Gains and Resample ---
    final_df_filtered, final_twr_factor, status_update_final = (
        _calculate_accumulated_gains_and_resample(
            daily_df=daily_df,
            benchmark_symbols_yf=clean_benchmark_symbols_yf,
            interval=interval,
        )
    )
    if status_update_final:
        status_parts.append(status_update_final.strip())
    if "Error" in status_update_final or "failed" in status_update_final.lower():
        has_errors = True
    elif "WARN" in status_update_final.upper():
        has_warnings = True

    # --- 8. Final Status and Return ---
    end_time_hist = time.time()
    logging.info(f"Historical Performance Calculation Finished (Scope: {filter_desc})")
    logging.info(
        f"Total Historical Calc Time: {end_time_hist - start_time_hist:.2f} seconds"
    )

    final_status_prefix = "Success"
    if has_errors:
        final_status_prefix = "Finished with Errors"
    elif has_warnings:
        final_status_prefix = "Finished with Warnings"
    final_status = f"{final_status_prefix} ({filter_desc})"
    if status_parts:
        final_status += f" [{'; '.join(status_parts)}]"
    final_status += (
        f"|||TWR_FACTOR:{final_twr_factor:.6f}"
        if pd.notna(final_twr_factor)
        else "|||TWR_FACTOR:NaN"
    )

    # Final column ordering (optional)
    if not final_df_filtered.empty:
        final_cols_order = [
            "Portfolio Value",
            "Portfolio Daily Gain",
            "daily_return",
            "Portfolio Accumulated Gain",
        ]
        for bm in clean_benchmark_symbols_yf:
            if f"{bm} Price" in final_df_filtered.columns:
                final_cols_order.append(f"{bm} Price")
            if f"{bm} Accumulated Gain" in final_df_filtered.columns:
                final_cols_order.append(f"{bm} Accumulated Gain")
        final_cols_order = [
            c for c in final_cols_order if c in final_df_filtered.columns
        ]
        try:
            final_df_filtered = final_df_filtered[final_cols_order]
        except KeyError:
            logging.warning("Warning: Could not reorder final columns.")

    # --- RETURN MODIFIED (4 items) ---
    return (
        final_df_filtered,
        historical_prices_yf_adjusted,
        historical_fx_yf,
        final_status,
    )


# --- Example Usage (Main block for testing this file directly) ---
if __name__ == "__main__":
    logging.basicConfig(
        level=LOGGING_LEVEL,
        format="%(asctime)s [%(levelname)-8s] %(module)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    multiprocessing.freeze_support()
    logging.info("Running portfolio_logic.py tests...")
    test_csv_file = "my_transactions.csv"
    test_display_currency = "EUR"
    test_account_currency_map = {
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
    test_default_currency = "USD"
    test_manual_prices = {"XYZ": 123.45}  # Example manual price

    logging.info("\n--- Testing Current Portfolio Summary (Refactored) ---")
    if not os.path.exists(test_csv_file):
        logging.error(
            f"ERROR: Test transactions file '{test_csv_file}' not found. Cannot run summary test."
        )
    else:
        (
            summary_metrics,
            holdings_df,
            account_metrics,
            ignored_idx,
            ignored_rsn,
            status,
        ) = calculate_portfolio_summary(
            transactions_csv_file=test_csv_file,
            display_currency=test_display_currency,
            account_currency_map=test_account_currency_map,
            default_currency=test_default_currency,
            include_accounts=None,  # Test all accounts
            manual_prices_dict=test_manual_prices,
        )
        logging.info(f"Current Summary Status (All Accounts): {status}")
        if summary_metrics:
            logging.info(f"Overall Metrics: {summary_metrics}")
        if holdings_df is not None and not holdings_df.empty:
            logging.info(f"Holdings DF Head:\n{holdings_df.head().to_string()}")
        if account_metrics:
            logging.info(f"Account Metrics: {account_metrics}")
        if ignored_idx:
            logging.warning(f"Ignored Indices ({len(ignored_idx)}): {ignored_idx}")
        if ignored_rsn:
            logging.warning(f"Ignored Reasons: {ignored_rsn}")

    logging.info("\n--- Testing Historical Performance Calculation (Refactored) ---")
    test_start = date(2023, 1, 1)
    test_end = date(2024, 6, 30)
    test_interval = "M"
    test_benchmarks = ["SPY", "QQQ"]
    test_use_raw_cache_flag = True
    test_use_daily_results_cache_flag = True
    test_num_processes = None
    test_accounts_subset1 = ["IBKR", "E*TRADE"]
    test_exclude_set = ["SET"]

    if not os.path.exists(test_csv_file):
        logging.error(
            f"ERROR: Test transactions file '{test_csv_file}' not found. Cannot run historical test."
        )
    else:
        logging.info(f"\n--- Running Historical Test: Scenario='All Accounts' ---")
        start_time_run = time.time()
        hist_df, raw_prices, raw_fx, hist_status = calculate_historical_performance(
            transactions_csv_file=test_csv_file,
            start_date=test_start,
            end_date=test_end,
            interval=test_interval,
            benchmark_symbols_yf=test_benchmarks,
            display_currency=test_display_currency,
            account_currency_map=test_account_currency_map,
            default_currency=test_default_currency,
            use_raw_data_cache=test_use_raw_cache_flag,
            use_daily_results_cache=test_use_daily_results_cache_flag,
            num_processes=test_num_processes,
            include_accounts=None,
            exclude_accounts=None,
        )
        end_time_run = time.time()
        logging.info(f"Test 'All Accounts' Status: {hist_status}")
        logging.info(
            f"Test 'All Accounts' Execution Time: {end_time_run - start_time_run:.2f} seconds"
        )
        if not hist_df.empty:
            logging.info(f"Test 'All Accounts' DF tail:\n{hist_df.tail().to_string()}")
        else:
            logging.info(f"Test 'All Accounts' Result: Empty DataFrame")

    logging.info("Finished portfolio_logic.py tests.")
# --- END OF MODIFIED portfolio_logic.py ---
