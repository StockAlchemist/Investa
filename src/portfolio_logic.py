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
from datetime import datetime, date  # noqa: F401 -- read by the Rust port at call time
from utils_time import get_latest_trading_date, get_est_today  # noqa: F401 -- read by the Rust port at call time
from typing import List, Dict, Tuple, Optional, Set, Any


import investa_core
import pandas as pd
import logging
import threading
from collections import OrderedDict
import time  # noqa: F401 -- read by the Rust port at call time
import multiprocessing

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    # This is primarily for type hinting and making the code runnable without kernprof
    # The actual profiling happens when run via kernprof
    # If line_profiler is not installed, the @profile decorator will be a no-op
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---

# --- Configure Logging ---
# Assumes logging is configured elsewhere (e.g., main_gui.py or config.py)
# If running standalone, uncomment and configure basicConfig here.
# logging.basicConfig(level=logging.DEBUG, format="...")

# --- AUTO-CACHE INVALIDATION ---

# -------------------------------

# --- Import Configuration and Utilities ---
try:
    from config import (
        LOGGING_LEVEL,
        CASH_SYMBOL_CSV,  # noqa: F401 -- read by the Rust port at call time
        DEFAULT_CURRENT_CACHE_FILE_PATH,  # Still used by MarketDataProvider default
        SHORTABLE_SYMBOLS,  # noqa: F401 -- re-exported, and read by the Rust port
        DEFAULT_CURRENCY,
        STOCK_QUANTITY_CLOSE_TOLERANCE,  # noqa: F401 -- read by the Rust port at call time
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from config.py. Exiting.")
    raise

try:
    from finutils import (
        get_conversion_rate,  # noqa: F401 -- read by the Rust port at call time
        map_to_yf_symbol,  # noqa: F401 -- read by the Rust port at call time
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

# --- Import the NEW Market Data Provider ---
try:
    from market_data import MarketDataProvider, get_shared_mdp  # noqa: F401 -- read by the Rust port at call time

    MARKET_PROVIDER_AVAILABLE = True
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import MarketDataProvider from market_data.py. Exiting."
    )
    MARKET_PROVIDER_AVAILABLE = False
    raise

# --- Import the NEW Portfolio Analyzer Functions ---
try:
    from portfolio_analyzer import (  # noqa: F401 -- read by the Rust port at call time
        _process_transactions_to_holdings,
        _build_summary_rows,
        _calculate_aggregate_metrics,
        calculate_fifo_lots_and_gains,  # NEW IMPORT
        extract_dividend_history,  # NEW IMPORT
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


# --- Engine split (2026-06): the functions below moved to focused modules and are
# re-exported here so existing importers (server, GUI, tests) work unchanged. ---
from portfolio_version import CURRENT_HIST_VERSION, _get_self_hash  # noqa: F401
from portfolio_valuation_kernels import (  # noqa: F401
    _calculate_daily_holdings_chronological_numba,
    _calculate_holdings_numba,
    _calculate_portfolio_value_at_date_unadjusted,
    _calculate_portfolio_value_at_date_unadjusted_numba,
    _calculate_portfolio_value_at_date_unadjusted_python,
    _normalize_series,
)
from portfolio_cashflows import (  # noqa: F401
    _calculate_daily_net_cash_flow,
    _calculate_daily_net_cash_flow_vectorized,
)
from portfolio_history import (  # noqa: F401
    _calculate_accumulated_gains_and_resample,
    _calculate_daily_metrics_worker,
    _get_or_calculate_all_daily_holdings,
    _load_or_calculate_daily_results,
    _prepare_historical_inputs,
    _unadjust_prices,
    _value_daily_holdings_vectorized,
    calculate_historical_performance,
    generate_mappings,
)

# FIFO lots and realized gains, keyed by a fingerprint of the transactions they
# were computed from. It used to be keyed on (db_mtime, currency) and hold one
# entry: two users on the server evicted each other on every request, and a
# change that leaves the DB file alone (an account-currency map, which fills
# blank currencies at load) served lots computed from the old transactions.
_FIFO_CACHE: "OrderedDict[Tuple[Any, ...], Tuple[Any, Any]]" = OrderedDict()
_FIFO_CACHE_MAX = 8
_FIFO_CACHE_LOCK = threading.Lock()


def _transactions_fingerprint(df: pd.DataFrame) -> str:
    """A content hash of `df`, values and index: ~2.5 ms for 5,000 rows."""
    # Ported to Rust: rust/investa_core/src/pl_helpers.rs
    return investa_core.pl_transactions_fingerprint(df)



def get_default_metrics_dict(
    display_curr_arg,
    report_date_arg,
    available_accs_list_arg,
    is_empty_data_case=False,
):
    """Build the default/zeroed overall-summary metrics dict.

    Values are 0.0 for the "empty but valid" case and np.nan for the
    "no data / error" case, so a missing portfolio reads as zero while a
    failed calculation reads as unknown.
    """
    # Ported to Rust: rust/investa_core/src/pl_helpers.rs
    return investa_core.pl_default_metrics(
        display_curr_arg, report_date_arg, available_accs_list_arg, is_empty_data_case
    )



def _enrich_summary_rows_with_sector_geo(
    portfolio_summary_rows,
    market_provider,
    manual_overrides_effective,
    effective_user_symbol_map,
    effective_user_excluded_symbols,
):
    """Attach Sector/quoteType/Country/Industry/exchange to each summary row.

    Resolves each held symbol's metadata from manual overrides first, then the
    batch-prefetched market-data metadata, falling back to per-symbol fetch.
    Cash rows are categorized as Cash. Returns the (possibly enriched) rows;
    the input is returned unchanged when empty or lacking a Symbol column.
    """
    # Ported to Rust: rust/investa_core/src/pl_helpers.rs
    return investa_core.pl_enrich_sector_geo(
        portfolio_summary_rows,
        market_provider,
        manual_overrides_effective,
        effective_user_symbol_map,
        effective_user_excluded_symbols,
    )



def _prepare_historical_fx_rates(
    transactions_df_filtered,
    all_transactions_df_cleaned,
    display_currency,
    default_currency,
    market_provider,
    cache_file_path,
):
    """Build the (date, currency) -> display/local FX lookup for tx processing.

    Returns (historical_fx_for_processing, historical_fx_data_usd_based,
    fx_warning); fx_warning is True when some historical FX rates could not be
    fetched (the caller raises has_warnings).
    """
    # Ported to Rust: rust/investa_core/src/pl_helpers.rs
    return investa_core.pl_prepare_historical_fx(
        transactions_df_filtered,
        all_transactions_df_cleaned,
        display_currency,
        default_currency,
        market_provider,
        cache_file_path,
    )



def published_nav_prices(symbols: Set[str]) -> Dict[str, Tuple[str, float]]:
    """Newest published NAV per symbol, as {SYMBOL: (nav_date, nav)}.

    Thai SSF/RMF funds have no ticker and no quote, so they sit on the user's
    exclusion list and never reach a price feed. The graph has valued them from
    the SEC's published series since the fund store landed
    (``portfolio_history._fund_nav_series``); the holdings table did not, and
    fell through to the hand-entered override — a scalar typed once and then
    left, which had SCBRCTECH 15% above its real NAV while the chart beside it
    drew the published number.

    A miss is the ordinary case (every US holding lands here), so this must stay
    cheap: one batched read of the newest row per fund, filtered to the symbols
    actually held. A store that isn't there yet returns nothing rather than
    raising, and valuation falls back to the override exactly as before.
    """
    # Ported to Rust: rust/investa_core/src/pl_helpers.rs
    return investa_core.pl_published_nav_prices(symbols)



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
    calc_method: Optional[str] = None,  # Added for benchmarking override
    account_interest_rates: Optional[Dict[str, float]] = None,
    interest_free_thresholds: Optional[Dict[str, float]] = None,
    account_cash_mode_map: Optional[Dict[str, str]] = None,
    db_mtime: float = 0.0,  # NEW: for caching FIFO results
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
    # Ported to Rust: rust/investa_core/src/pl_summary.rs
    return investa_core.pl_portfolio_summary(
        all_transactions_df_cleaned,
        original_transactions_df_for_ignored,
        ignored_indices_from_load,
        ignored_reasons_from_load,
        fmp_api_key,
        display_currency,
        show_closed_positions,
        cache_file_path,
        include_accounts,
        manual_overrides_dict,
        user_symbol_map,
        user_excluded_symbols,
        default_currency,
        account_currency_map,
        market_provider,
        calc_method,
        account_interest_rates,
        interest_free_thresholds,
        account_cash_mode_map,
        db_mtime,
    )



# =======================================================================
# --- SECTION: HISTORICAL PERFORMANCE CALCULATION FUNCTIONS (REVISED) ---
# =======================================================================
# --- NOTE: These functions remain in portfolio_logic.py as they were not moved ---


# --- Function to Unadjust Prices (Keep as is) ---


# --- Helper Functions for Point-in-Time Historical Calculation (Keep as is) ---
# _calculate_daily_net_cash_flow
# _calculate_portfolio_value_at_date_unadjusted
# _calculate_daily_metrics_worker
# (Implementations remain the same as provided previously)


# --- Helper Functions for Point-in-Time Historical Calculation ---

# --- VECTORIZED CASH FLOW CALCULATION ---


# --- START NUMBA HELPER FUNCTION ---


# --- END NUMBA HELPER FUNCTION ---


# --- START NEW CHRONOLOGICAL NUMBA HELPER ---


# --- Dispatcher Function ---


# --- Helper Function for Historical Input Preparation ---


# --- Daily Results Calculation (Keep as is) ---


# --- Accumulated Gain and Resampling (Keep as is) ---


# --- Main Historical Performance Calculation Function ---


# --- Helper to generate mappings (Needed for standalone profiling) ---


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
            raw_holdings,
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
            logging.info("Test 'All Accounts' Hist Result: Empty DataFrame")

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
