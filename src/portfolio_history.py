"""Historical performance pipeline: input prep, daily valuation, caching, resampling.

Split from portfolio_logic.py. Public entry point: calculate_historical_performance
(re-exported from portfolio_logic for backward compatibility).
"""

# ruff: noqa: E402
import hashlib
import json
import logging
import multiprocessing
import os
import time
import traceback
from datetime import datetime, date, timedelta
from functools import partial
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

import config
from config import (
    CASH_SYMBOL_CSV,
    DAILY_RESULTS_CACHE_PATH_PREFIX,
    HISTORICAL_CALC_METHOD,
    HISTORICAL_COMPARE_METHODS,
    HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
    LOGGING_LEVEL,
    SHORTABLE_SYMBOLS,
    STOCK_QUANTITY_CLOSE_TOLERANCE,
)
from corporate_actions import deduplicate_split_transactions as _deduplicate_split_transactions
from finutils import _get_file_hash, is_cash_symbol, map_to_yf_symbol
from market_data import get_shared_mdp

# Hard import above: if market_data is missing the module fails to load,
# matching the original try/except-raise. The flag is kept for the runtime check.
MARKET_PROVIDER_AVAILABLE = True
from portfolio_cashflows import (
    _calculate_daily_net_cash_flow,
    _calculate_daily_net_cash_flow_vectorized,
)
from portfolio_valuation_kernels import (
    _calculate_daily_holdings_chronological_numba,
    _calculate_portfolio_value_at_date_unadjusted,
    _calculate_portfolio_value_at_date_unadjusted_python,
    _normalize_series,
)
from portfolio_version import CURRENT_HIST_VERSION
from utils_time import get_est_today, get_latest_trading_date

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


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
        # Prefer the UNADJUSTED Close column. yfinance is fetched with
        # auto_adjust=False so "Close" is the raw historical price (the price
        # the user actually saw/paid on that date). "Adj Close" includes
        # split + dividend back-adjustment, which systematically deflates
        # historical values for dividend-paying assets — never use it here.
        # "price" is kept first only for legacy cache compatibility (the
        # function may also be passed an already-renamed DataFrame).
        col_to_use = None
        if not adj_price_df.empty:
            if "price" in adj_price_df.columns:
                col_to_use = "price"
            elif "Close" in adj_price_df.columns:
                col_to_use = "Close"
            elif "Adj Close" in adj_price_df.columns:
                col_to_use = "Adj Close"
        
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

                # FIX: Use date-based comparison to ensure precision and avoid off-by-one spikes
                # All dates STRICTLY BEFORE the split_date get unadjusted.
                # The split_date itself uses the already-adjusted (new) price.
                mask = pd.Series(forward_split_factor.index).dt.date < split_date
                mask = mask.values # Convert back to numpy mask for index selection
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
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
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
                account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
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
            # Per-position cause was already logged at WARNING level inside the
            # valuation loop with symbol/account context. Demote this aggregate
            # marker to debug so logs aren't doubled.
            logging.debug(f"Valuation failed for date: {eval_date}")
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
    current_hist_version: str = "v23",
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
        # --- FIX: Deduplicate splits by (Symbol, Date) to prevent double-counting ---
        # This can happen if a split is recorded for multiple accounts.
        # We take the first one found for each day.
        unique_splits = split_transactions.drop_duplicates(subset=["Symbol", "Date"])
        
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
            for symbol, group in unique_splits.groupby("Symbol")
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
    fx_pairs_for_api_yf = []
    for curr in all_currencies_needed:
        if not curr or curr.upper() == "USD":
            continue
        curr_upper = curr.upper()
        if curr_upper == "THB":
            fx_pairs_for_api_yf.append("USDTHB=X")
            fx_pairs_for_api_yf.append("THB=X")
        else:
            fx_pairs_for_api_yf.append(f"{curr_upper}=X")
    fx_pairs_for_api_yf = sorted(list(set(fx_pairs_for_api_yf)))

    # Cache paths
    app_cache_dir = config.get_app_cache_dir()

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
        # Cache must invalidate whenever the transaction data changes. With a
        # CSV path we hash the file; when loaded from DB (no CSV path) we hash
        # the actual DataFrame contents so DB edits force a recompute.
        if original_csv_file_path and os.path.exists(original_csv_file_path):
            tx_file_hash_component = _get_file_hash(original_csv_file_path)
        else:
            if original_csv_file_path:
                logging.warning(
                    f"Hist Prep: Original CSV path '{original_csv_file_path}' for hash not found. Hashing DataFrame contents instead."
                )
            _hash_cols = [
                c for c in ("Date", "Type", "Symbol", "Quantity",
                            "Price/Share", "Total Amount", "Commission",
                            "Account", "Local Currency", "To Account")
                if preloaded_transactions_df is not None and c in preloaded_transactions_df.columns
            ]
            if preloaded_transactions_df is not None and not preloaded_transactions_df.empty and _hash_cols:
                tx_file_hash_component = hashlib.sha256(
                    pd.util.hash_pandas_object(
                        preloaded_transactions_df[_hash_cols], index=False
                    ).values.tobytes()
                ).hexdigest()[:16]
            else:
                tx_file_hash_component = "EMPTY_DATAFRAME"

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
    interval: str,
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
                # Prefer "price" (already set to raw Close by normalize_df), then
                # raw Close, then Adj Close. Adj Close is dividend-adjusted and
                # would deflate historical valuations.
                col_to_use = None
                if "price" in price_df.columns:
                    col_to_use = "price"
                elif "Close" in price_df.columns:
                    col_to_use = "Close"
                elif "Adj Close" in price_df.columns:
                    col_to_use = "Adj Close"
                
                if col_to_use:
                    # Reindex to date_range
                    price_series = price_df[col_to_use]
                    # FIX: Ensure UTC awareness and DO NOT Normalize (preserves intraday/hourly timestamps)
                    price_series.index = pd.to_datetime(price_series.index, utc=True)
                
                # Reindex with interpolation to smooth gaps (especially the 9:30 AM anchor gap)
                # We use method=None in reindex to create NaNs, then interpolate them.
                is_intra = any(x in interval for x in ["m", "h", "min"])
                
                if is_intra:
                    # 1. Reindex to the full index (adds NaNs for missing minutes/hours)
                    # 2. Interpolate linearly based on time
                    # 3. ffill/bfill for any remaining edges
                    aligned_series = price_series.reindex(date_range).interpolate(method='time').ffill().bfill()
                else:
                    # Standard daily/weekly: Use linear interpolation to smooth gaps
                    # instead of simple ffill, which prevents "price restoration" spikes.
                    aligned_series = price_series.reindex(date_range).interpolate(method='linear').ffill().bfill()

                
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
        # FALLBACK: Try USDCURR=X if CURR=X missing, or vice versa
        if target_pair not in historical_fx_yf:
            if target_curr_upper == "THB" and "USDTHB=X" in historical_fx_yf:
                target_pair = "USDTHB=X"
            elif target_curr_upper != "THB" and f"USD{target_curr_upper}=X" in historical_fx_yf:
                target_pair = f"USD{target_curr_upper}=X"

        if target_pair in historical_fx_yf:
            t_df = historical_fx_yf[target_pair]
            if not t_df.empty:
                col_to_use = None
                if "price" in t_df.columns:
                    col_to_use = "price"
                elif "Close" in t_df.columns:
                    col_to_use = "Close"
                elif "Adj Close" in t_df.columns:
                    col_to_use = "Adj Close"
                elif "rate" in t_df.columns:
                    col_to_use = "rate"

                if col_to_use:
                    # Reindex to date_range with ffill
                    # Reindex to date_range with ffill then bfill to cover start of history
                    t_series = t_df[col_to_use].copy()
                    t_series.index = pd.to_datetime(t_series.index, utc=True)

                    # --- ORIENTATION CHECK ---
                    # Ensure rate is Local / USD. 
                    # If it's THB=X or USDTHB=X (~35), it's already Local/USD.
                    # If it was returned as a Major pair (e.g. EURUSD=X ~1.08), it's USD/Local -> Invert.
                    # Note: My analysis shows yfinance EUR=X is already EUR/USD (~0.92).
                    # But we keep this check for robustness if a USD... pair was fetched.
                    if target_pair.endswith("USD=X") and not target_pair.startswith("USD"):
                         # e.g. EURUSD=X -> USD per EUR -> Invert to get EUR per USD
                         t_series = 1.0 / t_series
                    
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
            # FALLBACK
            if local_pair not in historical_fx_yf:
                if local_curr_upper == "THB" and "USDTHB=X" in historical_fx_yf:
                    local_pair = "USDTHB=X"
                elif local_curr_upper != "THB" and f"USD{local_curr_upper}=X" in historical_fx_yf:
                    local_pair = f"USD{local_curr_upper}=X"

            if local_pair in historical_fx_yf:
                 l_df = historical_fx_yf[local_pair]
                 if not l_df.empty:
                    col_to_use = None
                    if "price" in l_df.columns:
                        col_to_use = "price"
                    elif "Close" in l_df.columns:
                        col_to_use = "Close"
                    elif "Adj Close" in l_df.columns:
                        col_to_use = "Adj Close"
                    elif "rate" in l_df.columns:
                        col_to_use = "rate"
                    
                    if col_to_use:
                        l_series = l_df[col_to_use].copy()
                        # FIX: Ensure UTC awareness and DO NOT Normalize (preserves intraday/hourly timestamps)
                        l_series.index = pd.to_datetime(l_series.index, utc=True)

                        # --- ORIENTATION CHECK ---
                        if local_pair.endswith("USD=X") and not local_pair.startswith("USD"):
                             l_series = 1.0 / l_series

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
    # We use linear interpolation to spread gains/losses over gaps, 
    # which prevents "price restoration" spikes from ruining TWR and Volatility.
    if daily_value_series.isna().any():
        # Identify which dates are missing for logging
        missing_dates = daily_value_series.index[daily_value_series.isna()]
        logging.warning(f"Valuation: Missing data for {len(missing_dates)} days. Using linear interpolation to smooth jumps.")
        status_msg += " (partial data interpolated)"
        daily_value_series = daily_value_series.interpolate(method='linear').ffill().fillna(0.0)

    
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
    current_hist_version: str = "v14",
    filter_desc: str = "All Accounts",
    calc_method: str = HISTORICAL_CALC_METHOD,
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
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
    _t0 = time.time()
    cache_valid_daily_results = False
    status_update = ""
    _dummy_warnings_set = set()


    # --- Chronological Calculation (numba_chrono) ---
    logging.debug(
        f"[_load_or_calculate_daily_results] Received date range: {start_date} to {end_date}"
    )
    _t1_setup = time.time()
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
                
                _heading_into_future = False # logic no longer needed, we fill what we have
                
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
                if l1_offset < 0:
                    l1_offset = 0
            
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
                # BUG-06 FIX: Prepare Total Amount for Auto Cash delta alignment
                tx_totals_np = sorted_df["Total Amount"].fillna(0.0).values.astype(np.float64) if "Total Amount" in sorted_df.columns else np.zeros(len(sorted_df), dtype=np.float64)

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

                # AUTO CASH: Resolve type IDs needed by auto-cash logic
                dividend_type_id = type_to_id.get("dividend", -1)
                interest_type_id = type_to_id.get("interest", -1)
                fees_type_id = type_to_id.get("fees", -1)
                tax_type_id = type_to_id.get("tax", -1)

                # AUTO CASH: Build acc_cash_modes array
                _acm = account_cash_mode_map if account_cash_mode_map else {}
                acc_cash_modes_np = np.zeros(num_accounts, dtype=np.int64)
                for _acc_name, _mode_str in _acm.items():
                    _acc_upper = _acc_name.upper().strip()
                    if _acc_upper in account_to_id and _mode_str == "Auto":
                        acc_cash_modes_np[account_to_id[_acc_upper]] = 1
            
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
                    tx_totals_np,  # BUG-06 FIX: Total Amount
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
                    dividend_type_id,
                    interest_type_id,
                    fees_type_id,
                    tax_type_id,
                    cash_symbol_id,
                    STOCK_QUANTITY_CLOSE_TOLERANCE,
                    shortable_symbol_ids,
                    acc_cash_modes_np,
                )
                
        
            # Determine included account IDs
            if included_accounts_list:
                account_ids_to_include_set = {account_to_id.get(str(acc).upper().strip()) for acc in included_accounts_list}
                account_ids_to_include_set.discard(None)
            else:
                # Include ALL accounts if list is empty (Global Scope)
                account_ids_to_include_set = set(account_to_id.values())

            # --- NEW: Split-Adjust Historical Prices to match Ledger Quantities ---
            # Yahoo Finance returns split-adjusted prices for all historical points.
            # Our ledger tracks raw quantities that multiply on split days.
            # To avoid 7x/4x spikes, we must "un-adjust" historical prices by multiplying
            # them by the cumulative split factor that occurs AFTER each historical point.
            historical_prices_yf_raw = {}
            split_txs = transactions_df_effective[transactions_df_effective["Type"].str.lower().str.strip().isin(["split", "stock split"])]
            
            # --- DEDUPLICATE SPLITS (Match Numba core logic) ---
            if not split_txs.empty:
                split_txs = split_txs.copy()
                # Priority 0 for 'All Accounts', 1 for others
                split_txs['__split_priority'] = np.where(split_txs['Account'].astype(str).str.lower() == 'all accounts', 0, 1)
                # Group by Symbol and Month for fuzzy deduplication
                split_txs['__ym'] = pd.to_datetime(split_txs['Date']).dt.to_period('M')
                
                sort_cols = ['Symbol', '__ym', '__split_priority']
                if 'original_index' in split_txs.columns:
                    sort_cols.append('original_index')
                
                split_txs = split_txs.sort_values(by=sort_cols)
                split_txs = split_txs.drop_duplicates(subset=['Symbol', '__ym', 'Split Ratio'])

            for yf_sym, price_df in historical_prices_yf_unadjusted.items():
                if price_df.empty:
                    historical_prices_yf_raw[yf_sym] = price_df
                    continue
                
                # Find ledger symbol(s) matching this YF ticker
                ledger_syms = [k for k, v in internal_to_yf_map.items() if v == yf_sym]
                sym_splits = split_txs[split_txs["Symbol"].isin(ledger_syms)]
                
                if sym_splits.empty:
                    historical_prices_yf_raw[yf_sym] = price_df
                else:
                    new_df = price_df.copy()
                    # Backwards cumulative product of splits
                    factors = pd.Series(1.0, index=new_df.index)
                    sorted_splits = sym_splits.sort_values(by="Date", ascending=False)
                    for _, split_row in sorted_splits.iterrows():
                        s_date = pd.to_datetime(split_row["Date"], utc=True)
                        ratio = pd.to_numeric(split_row.get("Split Ratio"), errors='coerce')
                        qty = pd.to_numeric(split_row.get("Quantity"), errors='coerce')
                        
                        # Fallback: Ratio might be in Quantity column for some importers
                        if (ratio is None or ratio <= 1e-9) and (qty is not None and 0 < qty <= 20.0):
                            ratio = qty
                            
                        if ratio and ratio > 1e-9:
                            factors.index = pd.to_datetime(factors.index, utc=True)
                            factors.loc[factors.index < s_date] *= ratio
                    
                    for col in ["price", "Close", "Adj Close", "Open", "High", "Low"]:
                        if col in new_df.columns:
                            new_df[col] = new_df[col] * factors
                    historical_prices_yf_raw[yf_sym] = new_df

            # --- VECTORIZED VALUATION ---
            daily_value_series, val_errors, val_status = _value_daily_holdings_vectorized(
                date_range=date_range_for_calc,
                daily_holdings_qty_np=daily_holdings_qty_to_use,
                daily_last_prices_np=daily_last_prices_to_use,
                daily_cash_balances_np=daily_cash_balances_to_use,
                historical_prices_yf_unadjusted=historical_prices_yf_raw, # USE RAW
                historical_fx_yf=historical_fx_yf,
                target_currency=display_currency,
                account_currency_map=account_currency_map,
                id_to_symbol=id_to_symbol,
                id_to_account=id_to_account,
                internal_to_yf_map=internal_to_yf_map,
                account_ids_to_include_set=account_ids_to_include_set,
                default_currency=default_currency,
                interval=interval,
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
                historical_prices_yf_unadjusted=historical_prices_yf_raw, # USE RAW
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
                account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
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
                    pool = None
                    try:
                        pool = multiprocessing.Pool(
                            processes=num_processes,
                            maxtasksperchild=50,  # Prevent memory leaks in long-running workers
                        )
                        results_iterator = pool.imap_unordered(
                            worker_partial, all_dates_to_process, chunksize=chunksize
                        )
                        consume_results(results_iterator)
                    finally:
                        # Graceful shutdown: close() stops accepting new tasks,
                        # join() waits for workers to finish sending results.
                        # This prevents BrokenPipeError from terminate() killing
                        # workers while they're still writing to the result pipe.
                        if pool is not None:
                            try:
                                pool.close()
                                pool.join()
                            except Exception as e_pool_shutdown:
                                logging.warning(f"Pool shutdown warning: {e_pool_shutdown}")
                                try:
                                    pool.terminate()
                                except Exception:
                                    pass
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
                _rows_dropped = 0
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
            
                # Robustness: We ignore returns if capital at risk is too small (e.g. < 10 cents)
                # to prevent division-by-nothing blowups.
                valid_denom_mask = adjusted_prev_value.notna() & (
                    abs(adjusted_prev_value) > 0.1
                )
                
                # --- NEW: Log ignored days for transparency ---
                if (~valid_denom_mask).any():
                    ignored_count = (~valid_denom_mask).sum()
                    if ignored_count > 0:
                        logging.debug(f"TWR: Ignoring {ignored_count} days where capital at risk was too small (<$0.10) — likely pre-funding period.")

                daily_df.loc[valid_denom_mask, "daily_return"] = (
                    daily_df.loc[valid_denom_mask, "daily_gain"]
                    / adjusted_prev_value.loc[valid_denom_mask]
                )
                
                # --- BUG-04 FIX: Contextual Spike Guard ---
                # Only cap returns that are clearly flow-driven artifacts, NOT legitimate market moves.
                # A spike is flow-driven if: portfolio value is small AND net flow is large relative to value.
                spike_threshold = 0.5
                spike_mask = (daily_df["daily_return"] > spike_threshold) | (daily_df["daily_return"] < -spike_threshold)
                if spike_mask.any():
                    net_flow_filled_abs = daily_df["net_flow"].fillna(0.0).abs()
                    portfolio_value_abs = daily_df["value"].abs()
                    # Only cap if: value < $1000 AND |net_flow| > 50% of value (flow-driven spike)
                    flow_driven_mask = spike_mask & (
                        (portfolio_value_abs < 1000.0) & (net_flow_filled_abs > portfolio_value_abs * 0.5)
                    )
                    if flow_driven_mask.any():
                        try:
                            spike_info = [f"{d.strftime('%Y-%m-%d')} ({daily_df.at[d, 'daily_return']*100:.1f}%)" for d in daily_df.index[flow_driven_mask]]
                            logging.warning(f"BUG-04 FIX: Capped {len(spike_info)} FLOW-DRIVEN TWR spikes (> {spike_threshold*100:.0f}%): {', '.join(spike_info[:12])}{'...' if len(spike_info) > 12 else ''}")
                        except Exception as e:
                            logging.error(f"Failed to log spike info: {e}")
                        daily_df.loc[flow_driven_mask, "daily_return"] = 0.0

                    # Log but DON'T cap legitimate market-driven spikes
                    market_spikes = spike_mask & ~flow_driven_mask
                    if market_spikes.any():
                        try:
                            market_info = [f"{d.strftime('%Y-%m-%d')} ({daily_df.at[d, 'daily_return']*100:.1f}%)" for d in daily_df.index[market_spikes]]
                            logging.info(f"BUG-04 FIX: Preserved {len(market_info)} legitimate market-driven returns (> {spike_threshold*100:.0f}%): {', '.join(market_info[:12])}{'...' if len(market_info) > 12 else ''}")
                        except Exception:
                            pass

                # --- BUG-05 FIX: Flow-Aware Transfer Healing ---
                # Only heal transfer-day spikes when the spike is clearly caused by the flow, not the market.
                high_vol_mask = (daily_df["daily_return"] > 0.1) | (daily_df["daily_return"] < -0.1)
                if high_vol_mask.any():
                    transfer_days = transactions_df_effective[transactions_df_effective["Type"].str.lower().str.strip() == "transfer"]["Date"].unique()
                    transfer_days_dt = pd.to_datetime(transfer_days).date
                    for idx, row in daily_df[high_vol_mask].iterrows():
                         if idx.date() in transfer_days_dt:
                             # BUG-05 FIX: Only heal if the net flow magnitude exceeds the daily gain
                             net_flow_val = abs(row.get("net_flow", 0.0)) if pd.notna(row.get("net_flow")) else 0.0
                             daily_gain_val = abs(row.get("daily_gain", 0.0)) if pd.notna(row.get("daily_gain")) else 0.0
                             if net_flow_val > daily_gain_val * 0.5:
                                 logging.info(f"TWR HEALING: Zeroed flow-driven artifact on {idx.date()} (Return: {row['daily_return']*100:.1f}%, NetFlow: {net_flow_val:.0f}, Gain: {daily_gain_val:.0f}).")
                                 daily_df.at[idx, "daily_return"] = 0.0

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
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
):
    """
    Layer 1 Cache: Calculates or loads from cache the daily holdings for ALL transactions.
    """
    tx_hash = hashlib.sha256(
        pd.util.hash_pandas_object(all_transactions_df, index=True).values
    ).hexdigest()
    # Version bump for cache key - Now linked to global CURRENT_HIST_VERSION
    cache_key = (
        f"ALL_HOLDINGS_{CURRENT_HIST_VERSION}_{tx_hash}_{start_date.isoformat()}_{end_date.isoformat()}"
    )

    cache_dir_base = config.get_app_cache_dir()
    if cache_dir_base:
        holdings_cache_dir = os.path.join(cache_dir_base, "all_holdings_cache_new")
        os.makedirs(holdings_cache_dir, exist_ok=True)
        key_file = os.path.join(holdings_cache_dir, f"{cache_key}.key")
        qty_file = os.path.join(holdings_cache_dir, f"{cache_key}_qty.npy")
        cash_file = os.path.join(holdings_cache_dir, f"{cache_key}_cash.npy")
        prices_file = os.path.join(holdings_cache_dir, f"{cache_key}_prices.npy")
    else:
        # No writable cache directory available: still compute, just skip the
        # load/save round-trip.
        logging.warning(
            "Could not find cache directory. Layer 1 caching disabled; computing without cache."
        )
        holdings_cache_dir = None
        key_file = qty_file = cash_file = prices_file = None

    if holdings_cache_dir and (
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
    # BUG-06 FIX: Prepare Total Amount for Auto Cash delta alignment
    tx_totals_np = (
        sorted_tx_df["Total Amount"].fillna(0.0).values.astype(np.float64)
        if "Total Amount" in sorted_tx_df.columns
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
    dividend_type_id = type_to_id.get("dividend", -1)    # AUTO CASH
    interest_type_id = type_to_id.get("interest", -1)    # AUTO CASH
    fees_type_id = type_to_id.get("fees", -1)            # AUTO CASH
    tax_type_id = type_to_id.get("tax", -1)              # AUTO CASH
    cash_symbol_id = symbol_to_id.get(CASH_SYMBOL_CSV, -1)

    # AUTO CASH: Build acc_cash_modes array
    num_symbols = len(symbol_to_id)
    num_accounts = len(account_to_id)
    
    _acm2 = account_cash_mode_map if account_cash_mode_map else {}
    acc_cash_modes_np2 = np.zeros(num_accounts, dtype=np.int64)
    for _acc_name2, _mode_str2 in _acm2.items():
        _acc_upper2 = _acc_name2.upper().strip()
        if _acc_upper2 in account_to_id and _mode_str2 == "Auto":
            acc_cash_modes_np2[account_to_id[_acc_upper2]] = 1

    shortable_symbol_ids = np.array(
        [symbol_to_id[s] for s in SHORTABLE_SYMBOLS if s in symbol_to_id],
        dtype=np.int64,
    )

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
            tx_totals_np,  # BUG-06 FIX: Total Amount
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
            dividend_type_id,
            interest_type_id,
            fees_type_id,
            tax_type_id,
            cash_symbol_id,
            STOCK_QUANTITY_CLOSE_TOLERANCE,
            shortable_symbol_ids,
            acc_cash_modes_np2,
        )
    )

    if holdings_cache_dir:
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
                    # 3. Find the normalization baseline
                    for col in final_df_output.columns:
                        if "Accumulated Gain" in col:
                            divisor = 0.0
                            found_valid = False
                            
                            # --- QUICK CHECK: If entire column is NaN or zero, skip search ---
                            col_data = final_df_output[col]
                            if col_data.isna().all() or (col_data == 0).all():
                                logging.debug(f"Normalization skipped for {col}: Column is empty or all zeros.")
                                continue
                            
                            # BUG-10 FIX: Try t-1 (last point BEFORE visible range) from full data
                            if col in resampled_naive.columns:
                                pre_range_data = resampled_naive.loc[resampled_naive.index < visible_dates[0], col].dropna()
                                if not pre_range_data.empty:
                                    t_minus_1_val = pre_range_data.iloc[-1]
                                    if pd.notnull(t_minus_1_val) and t_minus_1_val != 0:
                                        divisor = t_minus_1_val
                                        found_valid = True
                                        logging.debug(f"BUG-10 FIX: Using t-1 baseline for {col}: {divisor}")
                            
                            # Fallback: t0 if t-1 not available
                            if not found_valid:
                                t0_val = final_df_output.loc[visible_dates[0], col]
                                if pd.notnull(t0_val) and t0_val != 0:
                                    divisor = t0_val
                                    found_valid = True
                                    logging.debug(f"BUG-10 FIX: Fallback to t0 baseline for {col}: {divisor}")
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
                                # --- FIX: Safety check for massive spikes due to near-zero divisor ---
                                # If the divisor is extremely small (e.g. 0.0001), it will amplify the 
                                # whole series by 10,000x. We set a threshold for validity.
                                if abs(divisor) < 1e-4:
                                     logging.warning(f"Normalization divisor for {col} is too small ({divisor}). Skipping normalization to prevent spikes.")
                                else:
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
    except Exception:
        logging.exception("Hist CRITICAL: Accum gain/resample calc error")
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update

    return final_df_output, final_twr_factor, status_update


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
    account_cash_mode_map: Optional[Dict[str, str]] = None,  # AUTO CASH
) -> Tuple[
    pd.DataFrame,  # daily_df
    Dict[str, pd.DataFrame],  # historical_prices_yf_adjusted
    Dict[str, pd.DataFrame],  # historical_fx_yf
    str,  # final_status_str
    # pd.DataFrame, # key_ratios_df - Ratios are not calculated here
    # Dict[str, Any] # current_valuation_ratios - Ratios are not calculated here
]:
    # --- CRITICAL FIX: Deduplicate split transactions to prevent double-multiplication ---
    # This ensures both holdings and price unadjustment see the same consistent set of splits.
    all_transactions_df_cleaned = _deduplicate_split_transactions(all_transactions_df_cleaned)
    
    # -------------------------------
    start_time_hist = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []

    _processed_warnings = set()

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
    _min_fetch_end = max(clamped_end_date + timedelta(days=1), full_end_date_tx + timedelta(days=1))
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
        account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
    )

    if all_holdings_qty is None:
        status_parts.append("L1 Cache Error")
        has_errors = True  # This is a critical failure

    # --- NEW: Identify Currently Held Symbols for Integrity Check Optimization ---
    # We only care about integrity (drifts/splits) for symbols we actually hold today,
    # or benchmarks we track. For symbols we no longer hold, small historical 
    # drift doesn't justify a full history re-fetch during routine background syncs.
    active_symbols_yf = set(clean_benchmark_symbols_yf)
    try:
        # all_holdings_qty is indexed by [date_idx, symbol_id]
        # We check the last date index to find non-zero holdings
        last_date_idx = all_holdings_qty.shape[0] - 1
        if last_date_idx >= 0:
            held_sym_ids = np.where(np.abs(all_holdings_qty[last_date_idx]) > 1e-6)[0]
            for sid in held_sym_ids:
                if sid in id_to_symbol:
                    internal_sym = id_to_symbol[sid]
                    yf_sym = internal_to_yf_map.get(internal_sym)
                    if yf_sym:
                        active_symbols_yf.add(yf_sym)
        logging.info(f"Integrity check restricted to {len(active_symbols_yf)} active symbols.")
    except Exception as e_active:
        logging.warning(f"Failed to determine active symbols for integrity check: {e_active}")
        active_symbols_yf = None # Fallback to checking everything if determination fails
    # ----------------------------------------------------------------------------



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
        integrity_check_symbols=active_symbols_yf, # Pass the optimized list
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
                        
                        # --- FIX: Ensure the merged series has NO leading NaNs for the day ---
                        # This prevents the 'jump' from zero if today's first point is missing
                        if not merged_df.empty:
                            merged_df = merged_df.ffill().bfill()
                        
                        historical_prices_yf_adjusted[sym] = merged_df
                    else:
                        historical_prices_yf_adjusted[sym] = intra_df.ffill().bfill()
        except Exception as e_intra:
            logging.error(f"Failed to fetch/merge intraday data: {e_intra}")


    # BUG-11 FIX: Removed duplicate intraday fetch block that was here.
    # The intraday fetch and merge is already handled by the block above (lines ~6540-6600).


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
            historical_prices_yf_unadjusted=historical_prices_yf_adjusted,
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
            account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
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
                        if "price" in bm_df.columns:
                            col_to_use = "price"
                        elif "Adj Close" in bm_df.columns:
                            col_to_use = "Adj Close"
                        elif "Close" in bm_df.columns:
                            col_to_use = "Close"
                        
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

                            # --- FIX: Ensure TZ alignment for reindexing ---
                            if bm_series.index.tz is None and daily_df.index.tz is not None:
                                bm_series.index = bm_series.index.tz_localize(daily_df.index.tz)
                            elif bm_series.index.tz is not None and daily_df.index.tz is None:
                                bm_series.index = bm_series.index.tz_localize(None)

                            is_intra = any(x in interval for x in ["m", "h", "min"])
                            if is_intra:
                                bm_series_aligned = bm_series.reindex(daily_df.index).interpolate(method='time').ffill().bfill()
                            else:
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
