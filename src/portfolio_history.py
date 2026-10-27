"""Historical performance pipeline: input prep, daily valuation, caching, resampling.

Split from portfolio_logic.py. Public entry point: calculate_historical_performance
(re-exported from portfolio_logic for backward compatibility).
"""

# ruff: noqa: E402
import hashlib  # noqa: F401 -- read by the Rust port at call time
import json  # noqa: F401 -- read by the Rust port at call time
import multiprocessing  # noqa: F401 -- read by the Rust port at call time
import os  # noqa: F401 -- read by the Rust port at call time
import time  # noqa: F401 -- read by the Rust port at call time
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
    deduplicate_split_transactions as _deduplicate_split_transactions,  # noqa: F401 -- read by the Rust port at call time
)
from finutils import _get_file_hash, is_cash_symbol, map_to_yf_symbol  # noqa: F401 -- read by the Rust port at call time
from market_data import get_shared_mdp  # noqa: F401 -- read by the Rust port at call time

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
from portfolio_version import CURRENT_HIST_VERSION  # noqa: F401 -- read by the Rust port at call time
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
    # Ported to Rust: rust/investa_core/src/ph_performance.rs
    return investa_core.ph_historical_performance(
        all_transactions_df_cleaned,
        original_transactions_df_for_ignored,
        ignored_indices_from_load,
        ignored_reasons_from_load,
        start_date,
        end_date,
        interval,
        benchmark_symbols_yf,
        display_currency,
        account_currency_map,
        default_currency,
        use_raw_data_cache,
        use_daily_results_cache,
        num_processes,
        include_accounts,
        worker_signals,
        exclude_accounts,
        user_symbol_map,
        manual_overrides_dict,
        user_excluded_symbols,
        original_csv_file_path,
        calc_method,
        account_cash_mode_map,
    )


def generate_mappings(transactions_df_effective):
    """Generates string-to-ID mappings based on the effective transaction data."""
    # Ported to Rust: rust/investa_core/src/ph_small.rs
    return investa_core.ph_generate_mappings(transactions_df_effective)


