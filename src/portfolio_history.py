"""Historical performance pipeline: input prep, daily valuation, caching, resampling.

Split from portfolio_logic.py. Public entry point: calculate_historical_performance
(re-exported from portfolio_logic for backward compatibility).
"""

# ruff: noqa: E402
import hashlib  # noqa: F401 -- read by the Rust port at call time
import json  # noqa: F401 -- read by the Rust port at call time
import logging
import multiprocessing  # noqa: F401 -- read by the Rust port at call time
import os  # noqa: F401 -- read by the Rust port at call time
import time
from datetime import datetime, date, timedelta  # noqa: F401 -- read by the Rust port at call time
from functools import partial  # noqa: F401 -- read by the Rust port at call time
from typing import Any, Dict, List, Optional, Set, Tuple

import investa_core
import numpy as np
import pandas as pd

import config  # noqa: F401 -- read by the Rust port at call time
from config import (
    CASH_SYMBOL_CSV,  # noqa: F401 -- read by the Rust port at call time
    DAILY_RESULTS_CACHE_PATH_PREFIX,
    HISTORICAL_CALC_METHOD,
    HISTORICAL_COMPARE_METHODS,  # noqa: F401 -- read by the Rust port at call time
    HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
    LOGGING_LEVEL,  # noqa: F401 -- read by the Rust port at call time
    SHORTABLE_SYMBOLS,  # noqa: F401 -- read by the Rust port at call time
    STOCK_QUANTITY_CLOSE_TOLERANCE,  # noqa: F401 -- read by the Rust port at call time
)
from corporate_actions import (
    deduplicate_split_transactions as _deduplicate_split_transactions,
)
from finutils import _get_file_hash, is_cash_symbol, map_to_yf_symbol  # noqa: F401 -- read by the Rust port at call time
from market_data import get_shared_mdp

# Hard import above: if market_data is missing the module fails to load,
# matching the original try/except-raise. The flag is kept for the runtime check.
MARKET_PROVIDER_AVAILABLE = True
from portfolio_cashflows import (
    _calculate_daily_net_cash_flow,  # noqa: F401 -- read by the Rust port at call time
    _calculate_daily_net_cash_flow_vectorized,  # noqa: F401 -- read by the Rust port at call time
)
from portfolio_valuation_kernels import (
    _calculate_daily_holdings_chronological_numba,  # noqa: F401 -- read by the Rust port at call time
    _calculate_portfolio_value_at_date_unadjusted,  # noqa: F401 -- read by the Rust port at call time
    _calculate_portfolio_value_at_date_unadjusted_python,  # noqa: F401 -- read by the Rust port at call time
    _normalize_series,  # noqa: F401 -- read by the Rust port at call time
)
from portfolio_version import CURRENT_HIST_VERSION
from utils_time import get_est_today, get_latest_trading_date, nyse_schedule  # noqa: F401 -- read by the Rust port at call time

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
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_unadjust_prices(
        adjusted_prices_yf, yf_to_internal_map, splits_by_internal_symbol, processed_warnings
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
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_daily_metrics_worker(
        eval_date,
        transactions_df,
        historical_prices_yf_unadjusted,
        historical_prices_yf_adjusted,
        historical_fx_yf,
        target_currency,
        internal_to_yf_map,
        account_currency_map,
        default_currency,
        manual_overrides_dict,
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        type_to_id,
        currency_to_id,
        id_to_currency,
        calc_method,
        included_accounts,
        account_cash_mode_map,
    )


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
    # Ported to Rust: rust/investa_core/src/ph_valuation.rs
    return investa_core.ph_prepare_historical_inputs(
        preloaded_transactions_df,
        original_transactions_df_for_ignored,
        ignored_indices_from_load,
        ignored_reasons_from_load,
        account_currency_map,
        default_currency,
        include_accounts,
        exclude_accounts,
        start_date,
        end_date,
        benchmark_symbols_yf,
        display_currency,
        current_hist_version,
        raw_cache_prefix,
        daily_cache_prefix,
        user_symbol_map,
        user_excluded_symbols,
        original_csv_file_path,
        interval,
    )


def manual_prices_by_symbol_id(
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]],
    id_to_symbol: Dict[int, str],
) -> Dict[int, float]:
    """Resolve the user's manual price overrides to {symbol_id: price}.

    Holdings with no price feed at all — Thai mutual funds, private placements —
    are worth whatever NAV the user entered for them. Keys are matched
    case-insensitively because the engine normalizes symbols but the override
    file keeps whatever case the user typed.
    """
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_manual_prices_by_symbol_id(manual_overrides_dict, id_to_symbol)


def _fund_nav_series(
    symbol_ids: Dict[int, str], date_range: pd.DatetimeIndex
) -> Dict[int, pd.Series]:
    """Published NAV series for any of `symbol_ids` the fund store covers.

    Only symbols with no market feed reach here, so a miss is the ordinary case
    and must stay cheap: one coverage read tells us which codes exist at all,
    and only those are queried. A store that isn't there yet (pre-migration, or
    a fresh clone) returns nothing rather than raising — valuation then falls
    through to the manual override exactly as it did before.
    """
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_fund_nav_series(symbol_ids, date_range)


def align_prices_to_grid(
    price_series_by_id: Dict[int, pd.Series], date_range: pd.DatetimeIndex
) -> pd.DataFrame:
    """Carry each symbol's last observed price onto the calculation grid.

    Never interpolate across a gap: an overnight, weekend or holiday gap would be
    bridged by blending in the *next* real quote, which prices a past timestamp
    with a trade that had not happened yet. The last close is the only price that
    existed at that moment, so a Sunday is worth what Friday closed at — not
    three quarters of the way to Monday's open.

    The union index keeps bars that fall between grid points (a 15:45 close
    against a 16:00 grid point) from being dropped before the fill. A leading
    backfill still covers grid points that precede a symbol's first quote.
    """
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_align_prices_to_grid(price_series_by_id, date_range)


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
    manual_overrides_dict: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[pd.Series, bool, str]:
    """
    Vectorized valuation of holdings and cash.
    Returns (daily_value_series, has_errors, status_msg).
    """
    # Ported to Rust: rust/investa_core/src/ph_valuation.rs
    return investa_core.ph_value_daily_holdings(
        date_range,
        daily_holdings_qty_np,
        daily_last_prices_np,
        daily_cash_balances_np,
        historical_prices_yf_unadjusted,
        historical_fx_yf,
        target_currency,
        account_currency_map,
        id_to_symbol,
        id_to_account,
        internal_to_yf_map,
        account_ids_to_include_set,
        default_currency,
        interval,
        manual_overrides_dict,
    )


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
    # Ported to Rust: rust/investa_core/src/ph_daily.rs
    return investa_core.ph_load_or_calculate_daily_results(
        use_daily_results_cache,
        daily_results_cache_file,
        daily_results_cache_key,
        interval,
        worker_signals,
        transactions_csv_file,
        start_date,
        end_date,
        transactions_df_effective,
        historical_prices_yf_unadjusted,
        historical_prices_yf_adjusted,
        historical_fx_yf,
        display_currency,
        internal_to_yf_map,
        account_currency_map,
        default_currency,
        symbol_to_id,
        id_to_symbol,
        account_to_id,
        id_to_account,
        manual_overrides_dict,
        type_to_id,
        currency_to_id,
        id_to_currency,
        all_holdings_qty,
        all_cash_balances,
        all_last_prices,
        all_holdings_start_date,
        included_accounts_list,
        num_processes,
        current_hist_version,
        filter_desc,
        calc_method,
        account_cash_mode_map,
    )




# Entries embed the transaction hash and an end date anchored to "today", so a
# new entry (~10-90 MB × 3 arrays) is minted every day and on every transaction
# edit. Without pruning the directory grows unbounded (it reached 174 GB and
# filled the disk in July 2026).
_HOLDINGS_CACHE_MAX_AGE_DAYS = 7


def _prune_holdings_cache(
    holdings_cache_dir: str, max_age_days: int = _HOLDINGS_CACHE_MAX_AGE_DAYS
):
    """Deletes L1 holdings-cache files older than max_age_days.

    Runs only on cache-miss saves (roughly once per user per day), so a full
    directory listing is cheap.
    """
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_prune_holdings_cache(holdings_cache_dir, max_age_days)


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
    # Ported to Rust: rust/investa_core/src/ph_valuation.rs
    return investa_core.ph_all_daily_holdings(
        all_transactions_df,
        start_date,
        end_date,
        symbol_to_id,
        account_to_id,
        type_to_id,
        account_cash_mode_map,
    )


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
    # Ported to Rust: rust/investa_core/src/ph_daily.rs
    return investa_core.ph_accumulated_gains_and_resample(
        daily_df, benchmark_symbols_yf, interval, start_date_filter, end_date_filter
    )


@profile
def _intraday_symbols_for_window(
    symbols_yf, benchmark_symbols_yf, holdings_qty, full_start_date, start_date,
    end_date, id_to_symbol, internal_to_yf_map, transactions,
):
    """Include positions held in the window and trades absent from daily closes."""
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_intraday_symbols_for_window(
        symbols_yf,
        benchmark_symbols_yf,
        holdings_qty,
        full_start_date,
        start_date,
        end_date,
        id_to_symbol,
        internal_to_yf_map,
        transactions,
    )


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
    calc_method: Optional[str] = None,  # ADDED for benchmarking
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
    all_transactions_df_cleaned = _deduplicate_split_transactions(
        all_transactions_df_cleaned
    )

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
    if interval not in [
        "D",
        "W",
        "M",
        "ME",
        "1d",
        "1h",
        "1m",
        "2m",
        "5m",
        "15m",
        "30m",
        "60m",
        "90m",
    ]:
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
        logging.critical(
            "CRITICAL WARNING: `calculate_historical_performance` called with `use_daily_results_cache=True` but NO `original_csv_file_path`. Disabling cache to prevent stale data."
        )
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
        full_start_date = min(
            start_date, transactions_df_effective["Date"].min().date()
        )
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
            logging.info(
                f"Clamping end_date from {end_date} to {latest_trading} (Latest Trading Date)."
            )
            clamped_end_date = latest_trading

    # fetch_end_date must be at least clamped_end_date + 1 day (because YF is exclusive)
    # AND cover any transactions
    _min_fetch_end = max(
        clamped_end_date + timedelta(days=1), full_end_date_tx + timedelta(days=1)
    )
    # We also respect the original end_date if it was larger (e.g. for some reason needed?)
    # But usually we just need to cover the clamped range + buffer.
    # Let's just ensure we capture up to 'today' if needed for checking.
    # Actually, simplistic logic:
    fetch_end_date = max(
        end_date, date.today()
    )  # Ensure we fetch up to today to populate cache correctly?
    # Better:
    fetch_end_date = max(end_date, full_end_date_tx)
    if fetch_end_date <= clamped_end_date:
        fetch_end_date = clamped_end_date + timedelta(days=1)

    logging.info(
        f"Date Range Config: Request={start_date}..{end_date}, ClampedEnd={clamped_end_date}, FetchEnd={fetch_end_date}"
    )

    market_provider = get_shared_mdp(hist_data_cache_dir_name="historical_data_cache")

    # This calculates daily holdings purely from transactions (no market data yet), or loads L1 cache
    # But it returns just the arrays/maps needed for *daily* scope.
    # For intraday, we might re-calculate inside _load_or_calculate if needed (cache miss logic handles it).
    all_holdings_qty, all_cash_balances, all_last_prices = (
        _get_or_calculate_all_daily_holdings(
            all_transactions_df=all_transactions_df_cleaned,
            start_date=full_start_date,
            end_date=fetch_end_date,
            symbol_to_id=symbol_to_id,
            account_to_id=account_to_id,
            type_to_id=type_to_id,
            account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
        )
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
        logging.info(
            f"Integrity check restricted to {len(active_symbols_yf)} active symbols."
        )
    except Exception as e_active:
        logging.warning(
            f"Failed to determine active symbols for integrity check: {e_active}"
        )
        active_symbols_yf = (
            None  # Fallback to checking everything if determination fails
        )
    # ----------------------------------------------------------------------------

    # --- 3. Load or Fetch ADJUSTED Historical Raw Data ---
    # We always fetch daily data for the FULL range to support valuation baseline.
    historical_prices_yf_adjusted, fetch_failed_prices = (
        market_provider.get_historical_data(
            symbols_yf=symbols_for_stocks_and_benchmarks_yf,
            start_date=full_start_date,
            end_date=fetch_end_date,
            interval="1d",
            # Force DISABLE raw cache for intraday intervals to ensure we get fresh daily close/open for the day
            use_cache=use_raw_data_cache if interval == "1d" else False,
            # Append fetch_end_date to key to ensure we don't use stale cache for different fetch range
            cache_key=f"{raw_data_cache_key}_{fetch_end_date.isoformat()}",
            integrity_check_symbols=active_symbols_yf,  # Pass the optimized list
        )
    )

    # --- INTRADAY PATCH: Fetch high-res data for active range ---
    # If the user requested an intraday interval (e.g. 5m), we must ensure
    # historical_prices_yf_adjusted contains these intraday timestamps for the active period.
    # Otherwise, the calculation loop (which runs at 5m resolution) will just find the same
    # 'daily' price for every 5m step, resulting in a flat line for the day.
    if interval in ["1h", "1m", "2m", "5m", "15m", "30m", "60m", "90m"]:
        # Only fetch intraday bars for symbols actually held during [start_date, fetch_end_date]
        # plus requested benchmarks. Fetching closed positions from 20 years ago wastes
        # dozens of subprocess batches and triggers retries on delisted tickers.
        intraday_symbols_yf = list(symbols_for_stocks_and_benchmarks_yf)
        try:
            intraday_symbols_yf = _intraday_symbols_for_window(
                symbols_for_stocks_and_benchmarks_yf, clean_benchmark_symbols_yf,
                all_holdings_qty, full_start_date, start_date, fetch_end_date,
                id_to_symbol, internal_to_yf_map, all_transactions_df_cleaned,
            )
        except Exception as e_intra_syms:
            logging.warning(f"Failed to narrow intraday symbol list: {e_intra_syms}")

        logging.info(
            f"Intraday interval '{interval}' detected. Fetching coverage for {len(intraday_symbols_yf)}/{len(symbols_for_stocks_and_benchmarks_yf)} active symbols..."
        )
        try:
            # Fetch intraday data. yfinance handles 60d limit for 5m, 730d for 1h etc.
            # We use start_date from the request to capture the relevant active window.
            # CRITICAL: fetch_end_date is exclusive in yfinance. To include "Today" (if fetch_end_date is today),
            # we must extend it by 1 day.
            intraday_end_date = fetch_end_date
            if intraday_end_date >= get_est_today():
                intraday_end_date = get_est_today() + timedelta(days=1)

            intraday_prices_adj, _ = market_provider.get_historical_data(
                symbols_yf=intraday_symbols_yf,
                start_date=max(start_date, get_est_today() - timedelta(days=59))
                if "m" in interval
                else start_date,
                end_date=intraday_end_date,
                interval=interval,
                use_cache=True,
            )

            if intraday_prices_adj:
                logging.info(
                    f"Merging {len(intraday_prices_adj)} intraday price series into main price history..."
                )
                for sym, intra_df in intraday_prices_adj.items():
                    if intra_df is None or intra_df.empty:
                        continue

                    # 1. Normalize Intraday Index to UTC
                    if (
                        not isinstance(intra_df.index, pd.DatetimeIndex)
                        or intra_df.index.tz is None
                    ):
                        intra_df.index = pd.to_datetime(intra_df.index, utc=True)

                    # 2. Merge into existing Daily DF
                    # We combine them. If index overlaps, we prefer Intraday?
                    # Actually, Daily data is at 00:00 UTC (usually). Intraday is at 09:30, 09:35 etc.
                    # So they don't overlap in time. We can just concat and sort.
                    # This gives the "step" function for history (daily) and "curve" for active day (intraday).
                    if sym in historical_prices_yf_adjusted:
                        daily_df_sym = historical_prices_yf_adjusted[sym]
                        # Normalize Daily Index to UTC if not already
                        if (
                            not isinstance(daily_df_sym.index, pd.DatetimeIndex)
                            or daily_df_sym.index.tz is None
                        ):
                            daily_df_sym.index = pd.to_datetime(
                                daily_df_sym.index, utc=True
                            )

                        # Concat and Sort
                        # Filter out intraday range from daily to avoid duplication if daily has 'today' row?
                        # Daily usually has 'today' 00:00 or similar.
                        # Safest is to keep daily history OLDER than intraday start, and append full intraday.
                        intra_start = intra_df.index.min()
                        daily_mask = daily_df_sym.index < intra_start

                        merged_df = pd.concat([daily_df_sym.loc[daily_mask], intra_df])
                        merged_df = merged_df[
                            ~merged_df.index.duplicated(keep="last")
                        ].sort_index()

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
            start_date=start_date,  # Let provider clip
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

                    combined = pd.concat(
                        [d_df[d_df.index < h_start_ts], h_df_proc]
                    ).sort_index()
                    historical_fx_yf[p] = combined[
                        ~combined.index.duplicated(keep="last")
                    ]
            else:
                historical_fx_yf[p] = h_df

    fetch_failed = fetch_failed_prices or fetch_failed_fx
    if fetch_failed:
        # has_errors = True  # Treat critical fetch error as overall error
        # RELAXED: Don't fail the whole graph just because some symbols (e.g. BLV, DAL) are missing.
        # We want to show partial data.
        logging.warning(
            "History: Some symbols failed to fetch, but proceeding with available data."
        )
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
    full_start_ts = pd.Timestamp(full_start_date, tz="UTC")
    for s_map in [historical_prices_yf_adjusted, historical_fx_yf]:
        for k, v in list(s_map.items()):  # Use list to avoid mutation issues
            if not isinstance(v.index, pd.DatetimeIndex) or v.index.tz is None:
                v.index = pd.to_datetime(v.index, utc=True)

            # STABILIZATION: Use linear interpolation for smoothness, fallback to bfill/ffill for edges
            if v is not None and not v.empty:
                min_idx = v.index.min()
                if min_idx > full_start_ts:
                    logging.info(
                        f"STABILIZATION: Smart-Filling {k} from {min_idx} to {full_start_ts}"
                    )
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
                    v = v.interpolate(method="time").bfill().ffill()

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
            all_holdings_start_date=full_start_date,  # PASS the expanded start date
            use_daily_results_cache=use_daily_results_cache
            and interval
            not in [
                "1m",
                "2m",
                "5m",
                "15m",
                "30m",
                "60m",
                "90m",
                "1h",
            ],  # DISABLE CACHE if intraday
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
            daily_df.index = daily_df.index.tz_localize("UTC")

        # DEDUPLICATION: Ensure portfolio index is unique before benchmark alignment
        if daily_df.index.duplicated().any():
            logging.info(
                f"Deduplicating portfolio daily_df.index: {len(daily_df)} -> {len(daily_df[~daily_df.index.duplicated()])}"
            )
            daily_df = daily_df[~daily_df.index.duplicated(keep="last")]
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
            logging.info(
                f"Fetched {len(bench_prices_adj)} benchmarks. Keys: {list(bench_prices_adj.keys())}"
            )
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
                            logging.info(
                                f"Benchmark {bm}: Using col '{col_to_use}', count={len(bm_df)}, first={bm_df.index[0]}"
                            )
                            bm_series = bm_df[col_to_use].copy()
                        bm_series.name = f"{bm} Price"

                        # Ensure index is datetime for merging
                        if (
                            not isinstance(bm_series.index, pd.DatetimeIndex)
                            or bm_series.index.tz is None
                        ):
                            bm_series.index = pd.to_datetime(bm_series.index, utc=True)

                        # DEDUPLICATION: Ensure benchmark series index is unique before reindexing
                        if bm_series.index.duplicated().any():
                            logging.info(
                                f"Deduplicating benchmark {bm} index: {len(bm_series)} -> {len(bm_series[~bm_series.index.duplicated()])}"
                            )
                            bm_series = bm_series[
                                ~bm_series.index.duplicated(keep="last")
                            ]
                        # Merge into daily_df using reindex with 'ffill' to handle timestamp mismatches
                        # and overnight gaps. method='nearest' with tight tolerance often misses morning data.
                        try:
                            # Use generous tolerance to bridge overnight/weekend gaps
                            # MLK Day holiday requires > 72h (Fri 4pm to Tue 9:30am is ~90h)
                            tol = pd.Timedelta("120h")

                            # --- FIX: Ensure TZ alignment for reindexing ---
                            if (
                                bm_series.index.tz is None
                                and daily_df.index.tz is not None
                            ):
                                bm_series.index = bm_series.index.tz_localize(
                                    daily_df.index.tz
                                )
                            elif (
                                bm_series.index.tz is not None
                                and daily_df.index.tz is None
                            ):
                                bm_series.index = bm_series.index.tz_localize(None)

                            is_intra = any(x in interval for x in ["m", "h", "min"])
                            if is_intra:
                                bm_series_aligned = (
                                    bm_series.reindex(daily_df.index)
                                    .interpolate(method="time")
                                    .ffill()
                                    .bfill()
                                )
                            else:
                                bm_series_aligned = bm_series.reindex(
                                    daily_df.index, method="ffill", tolerance=tol
                                )
                            daily_df[f"{bm} Price"] = bm_series_aligned
                            # ADDED: Diagnostic logging for Tuesday data
                            num_tuesday = (
                                daily_df.loc[
                                    daily_df.index.date == date(2026, 1, 20),
                                    f"{bm} Price",
                                ]
                                .notna()
                                .sum()
                            )
                            logging.info(
                                f"Benchmark {bm}: Aligned. Tuesday points: {num_tuesday}"
                            )

                        except Exception as e_align:
                            logging.warning(
                                f"Benchmark alignment failed for {bm}: {e_align}. Falling back to strict join."
                            )
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
    daily_df, final_twr_factor, status_update_resample = (
        _calculate_accumulated_gains_and_resample(
            daily_df=daily_df,
            benchmark_symbols_yf=clean_benchmark_symbols_yf,
            interval=interval,
            start_date_filter=start_date,
            end_date_filter=clamped_end_date,
        )
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

    final_status_str = f"{final_status_prefix} ({filter_desc})"
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
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_generate_mappings(transactions_df_effective)


