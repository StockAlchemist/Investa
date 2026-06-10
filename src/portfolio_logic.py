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
from datetime import datetime, date
from utils_time import get_latest_trading_date
from typing import List, Dict, Tuple, Optional, Set, Any


import pandas as pd
import numpy as np
import logging
import traceback
import time
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
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENT_CACHE_FILE_PATH,  # Still used by MarketDataProvider default
        SHORTABLE_SYMBOLS,  # re-exported (tests import it from here)
        DEFAULT_CURRENCY,
        STOCK_QUANTITY_CLOSE_TOLERANCE,
    )
except ImportError:
    logging.critical("CRITICAL ERROR: Could not import from config.py. Exiting.")
    raise

try:
    from finutils import (
        get_conversion_rate,
        map_to_yf_symbol,
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
    account_cash_mode_map: Optional[Dict[str, str]] = None,
    db_mtime: float = 0.0, # NEW: for caching FIFO results
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
            "taxes": 0.0 if is_empty_data_case else np.nan,
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
    _effective_account_currency_map = (
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
        fx_pairs_to_fetch_hist = []
        for lc in cleaned_currencies_for_hist_fx:
            if not lc or str(lc).strip() == "" or lc.upper() == "USD" or pd.isna(lc):
                continue
            curr_code = lc.upper()
            if curr_code == "THB":
                fx_pairs_to_fetch_hist.append("USDTHB=X")
            else:
                fx_pairs_to_fetch_hist.append(f"{curr_code}=X")

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
                
                # Just take the price column (prefer 'Close' from Yahoo, fallback to 'price' if internal mock)
                if 'Close' in df.columns:
                    series = df['Close'].rename(curr_code)
                elif 'price' in df.columns:
                    series = df['price'].rename(curr_code)
                else:
                    logging.warning(f"Warning: neither 'Close' nor 'price' column found in historical FX for {pair}. Columns: {df.columns}")
                    continue
                # Ensure index is date (not datetime with time)
                series.index = series.index.date
                # Remove duplicates in index if any
                series = series[~series.index.duplicated(keep='last')]
                fx_series_list.append(series)

        if fx_series_list:
            # Concat into a wide DataFrame
            master_fx_df = pd.concat(fx_series_list, axis=1)
            # --- IMPROVE: Handle NaNs in FX data (holidays, weekends) ---
            master_fx_df.sort_index(inplace=True)
            master_fx_df = master_fx_df.ffill().bfill()

            
            # --- ADDED: Normalize column names (e.g. USDTHB=X -> THB) ---
            # This ensures that 'THB' is found in master_fx_df.columns later.
            rename_map = {}
            for col in master_fx_df.columns:
                # 1. Strip =X suffix if present
                clean_name = col.replace("=X", "").upper()
                
                # 2. Extract local currency if it's a USD pair (e.g. USDTHB or EURUSD)
                if len(clean_name) == 6:
                    if clean_name.startswith("USD"):
                        final_code = clean_name[3:]
                    elif clean_name.endswith("USD"):
                        final_code = clean_name[:3]
                    else:
                        final_code = clean_name
                else:
                    final_code = clean_name
                
                if final_code != col:
                    rename_map[col] = final_code

            
            if rename_map:
                master_fx_df.rename(columns=rename_map, inplace=True)
                logging.debug(f"Normalized FX columns: {rename_map}")

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
                    _, current_fx_rates, _, _, _ = market_provider_for_hist_fx.get_current_quotes(
                        internal_stock_symbols=[],
                        required_currencies=set(missing_pairs),
                        user_symbol_map={},
                        user_excluded_symbols=set()
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
                        # --- MODIFIED: Robust Cross Rate Calculation ---
                        # master_fx_df[curr_col] is the rate fetched from YF.
                        # YF is inconsistent: 
                        # - USDTHB=X (~35) is THB/USD (Local/USD)
                        # - EURUSD=X (~1.08) is USD/EUR (USD/Local)
                        # We want all rates in 'master_fx_df' to effectively be 'Local / USD' (units per 1 USD)
                        # so that: Display/Local = (Display/USD) / (Local/USD)
                        
                        raw_rate = master_fx_df[curr_col]
                        
                        # Heuristic: Determine if rate is Local/USD or USD/Local.
                        # YF symbols like EUR=X, GBP=X usually return Local/USD (~0.9, ~0.8).
                        # But explicit pairs like EURUSD=X return USD/Local (~1.08).
                        # We want all rates to be Local / USD for the divisor logic.
                        
                        rate_to_use = raw_rate
                        
                        # --- IMPROVED HEURISTIC ---
                        # Standardize all rates to "Local per 1 USD" (e.g. THB=35, JPY=150, EUR=0.92)
                        # so that: Rate_to_Target = Target_per_USD / Local_per_USD
                        
                        avg_rate = raw_rate.mean()
                        is_major = curr_col in ["EUR", "GBP", "AUD", "NZD"]
                        
                        # 1. If it's a major currency and > 1.0 (e.g. 1.08), it's USD/Local. Invert to get Local/USD (~0.92).
                        if is_major and avg_rate > 1.0:
                            rate_to_use = 1.0 / raw_rate
                            logging.debug(f"Vectorized FX: Inverting major {curr_col} (Avg={avg_rate:.4f}) to Local/USD.")
                        # 2. If it's THB and < 1.0 (e.g. 0.028), it's USD/THB. Invert to get THB/USD (~35.0).
                        elif curr_col == "THB" and avg_rate < 1.0:
                            rate_to_use = 1.0 / raw_rate
                            logging.debug(f"Vectorized FX: Inverting {curr_col} (Avg={avg_rate:.4f}) to THB/USD.")
                        # 3. General catch-all: if rate is extremely small (< 0.1), it's likely USD/Local
                        elif avg_rate < 0.1:
                            rate_to_use = 1.0 / raw_rate
                            logging.debug(f"Vectorized FX: Inverting low-rate {curr_col} (Avg={avg_rate:.4f}) to Local/USD.")
                        else:
                            rate_to_use = raw_rate

                        
                        cross_rates = display_rates / rate_to_use
                        
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
    holdings, _, _, _, taxes_local, ignored_indices_proc, ignored_reasons_proc, transfer_costs, warn_proc = (
        _process_transactions_to_holdings(
            transactions_df=transactions_df_filtered,  # Pass filtered DataFrame
            default_currency=default_currency,
            shortable_symbols=SHORTABLE_SYMBOLS,
            historical_fx_lookup=historical_fx_for_processing,  # NEW ARG
            display_currency_for_hist_fx=display_currency,  # NEW ARG
            report_date=report_date,
            account_cash_mode_map=account_cash_mode_map,  # AUTO CASH
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

        # BN-08: Cache FIFO results keyed by db_mtime and display_currency
        global _FIFO_CACHE
        if '_FIFO_CACHE' not in globals():
            _FIFO_CACHE = {}
            
        fifo_cache_key = (db_mtime, display_currency)
        if db_mtime > 0 and fifo_cache_key in _FIFO_CACHE:
            logging.info("Using cached FIFO Realized Gains & Lots...")
            fifo_realized_gains_df, open_lots_dict = _FIFO_CACHE[fifo_cache_key]
        else:
            # Call the function that returns both gains and lots
            fifo_realized_gains_df, open_lots_dict = calculate_fifo_lots_and_gains(
                transactions_df=fifo_input_df, # Use SORTED FULL history
                display_currency=display_currency,
                historical_fx_yf=historical_fx_data_usd_based,
                default_currency=default_currency,
                shortable_symbols=SHORTABLE_SYMBOLS,
                stock_quantity_close_tolerance=STOCK_QUANTITY_CLOSE_TOLERANCE,
                current_fx_rates_vs_usd=current_fx_rates_vs_usd, # Pass the available rates!
            )
            
            if db_mtime > 0:
                _FIFO_CACHE.clear() # keep only the latest
                _FIFO_CACHE[fifo_cache_key] = (fifo_realized_gains_df, open_lots_dict)
        
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
                # Manual price overrides win — without this, lots for symbols yfinance
                # can't quote (e.g. SET tickers) get Market Value = 0 even though the
                # rest of the engine uses the override correctly.
                current_price = 0.0
                manual_override_for_sym = manual_overrides_effective.get(sym) if manual_overrides_effective else None
                if isinstance(manual_override_for_sym, dict):
                    manual_price_val = manual_override_for_sym.get("price")
                    if manual_price_val is not None and pd.notna(manual_price_val):
                        try:
                            mp_float = float(manual_price_val)
                            if mp_float > 1e-9:
                                current_price = mp_float
                        except (ValueError, TypeError):
                            pass
                if current_price <= 1e-9 and sym in current_stock_data_internal:
                    fetched_price = current_stock_data_internal[sym].get("price", 0.0)
                    if pd.notna(fetched_price):
                        current_price = fetched_price
                
                # We also need conversion rate from Local -> Display
                # holding_data has 'local_currency'
                local_curr = holding_data.get("local_currency", default_currency)
                fx_rate_curr = get_conversion_rate(
                    local_curr, display_currency, current_fx_rates_vs_usd
                )
                if pd.isna(fx_rate_curr):
                    fx_rate_curr = 0.0 # Safety

                for lot in lots:
                    # Lot fields needed: Date, Quantity, Cost Basis (Display), Mkt Val (Display), Unreal Gain (Display)
                    # Lot struct from analyzer: 'qty', 'cost_per_share_local_net', 'purchase_date', 'purchase_fx_to_display'
                    
                    l_qty = lot["qty"]
                    l_date = lot["purchase_date"]
                    l_cost_local = lot["cost_per_share_local_net"]
                    l_purch_fx = lot["purchase_fx_to_display"]
                    
                    if pd.isna(l_purch_fx):
                        l_purch_fx = 0.0 # Should not happen if data good
                    
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
        include_accounts=include_accounts,
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
            sector_map, quote_type_map, country_map, industry_map, exchange_map = {}, {}, {}, {}, {}

            # PERF FIX (BN-07): Batch pre-fetch metadata for ALL symbols at once.
            # Previously, get_fundamental_data() was called per-symbol inside the loop,
            # causing N sequential cache lookups (and potentially N subprocess calls).
            # Now we fetch all metadata in one batch call and use the results in the loop.
            yf_symbols_for_metadata = set()
            internal_to_yf_for_sector = {}
            for internal_symbol in symbols_in_summary:
                if internal_symbol == CASH_SYMBOL_CSV or internal_symbol.startswith("Cash ("):
                    continue
                yf_ticker = map_to_yf_symbol(
                    internal_symbol,
                    effective_user_symbol_map,
                    effective_user_excluded_symbols,
                )
                if yf_ticker:
                    yf_symbols_for_metadata.add(yf_ticker)
                    internal_to_yf_for_sector[internal_symbol] = yf_ticker
            
            # Single batch fetch — returns cached metadata including sector, industry, country, quoteType
            batch_metadata = {}
            if yf_symbols_for_metadata and MARKET_PROVIDER_AVAILABLE and market_provider:
                try:
                    batch_metadata = market_provider._ensure_metadata_batch(yf_symbols_for_metadata)
                except Exception as e_batch_meta:
                    logging.warning(f"Batch metadata pre-fetch failed: {e_batch_meta}")

            for internal_symbol in symbols_in_summary:
                symbol_overrides = manual_overrides_effective.get(
                    internal_symbol.upper(), {}
                )
                manual_asset_type = symbol_overrides.get("asset_type", "").strip()
                manual_sector = symbol_overrides.get("sector", "").strip()
                manual_geography = symbol_overrides.get("geography", "").strip()
                manual_industry = symbol_overrides.get("industry", "").strip()
                manual_exchange = symbol_overrides.get("exchange", "").strip()

                if internal_symbol == CASH_SYMBOL_CSV or internal_symbol.startswith(
                    "Cash ("
                ):
                    sector_map[internal_symbol] = "Cash"
                    quote_type_map[internal_symbol] = "CASH"
                    country_map[internal_symbol] = "Cash"
                    industry_map[internal_symbol] = "Cash"
                    exchange_map[internal_symbol] = "Cash" # Set Market for Cash
                    
                    if manual_exchange:
                         exchange_map[internal_symbol] = manual_exchange
                    continue

                if internal_symbol == "SCBRM1":
                     logging.info(f"DEBUG_OVERRIDE_SCBRM1: ManualExchange={manual_exchange}, AllOverrides={symbol_overrides}")

                yf_ticker_for_sector = internal_to_yf_for_sector.get(internal_symbol)
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
                if manual_exchange:
                    exchange_map[internal_symbol] = manual_exchange

                if yf_ticker_for_sector and (
                    not manual_sector
                    or not manual_asset_type
                    or not manual_geography
                    or not manual_industry
                ):
                    # PERF FIX (BN-07): Use batch-prefetched metadata instead of per-symbol fetch.
                    # Falls back to get_fundamental_data only if metadata is missing for this symbol.
                    fundamental_info = batch_metadata.get(yf_ticker_for_sector)
                    if not fundamental_info and MARKET_PROVIDER_AVAILABLE and market_provider:
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
                            ) or "Unknown Region"
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
                            country_map[internal_symbol] = "Unknown Region"
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
            # Apply Manual Exchange Overrides
            existing_exchange = summary_df_unfiltered_temp["exchange"] if "exchange" in summary_df_unfiltered_temp.columns else pd.Series([None] * len(summary_df_unfiltered_temp), index=summary_df_unfiltered_temp.index)
            summary_df_unfiltered_temp["exchange"] = (
                summary_df_unfiltered_temp["Symbol"]
                .map(exchange_map)
                .combine_first(existing_exchange)
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
            "taxes": 0.0,
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
                fx_rates=current_fx_rates_vs_usd,
                include_accounts=include_accounts,
                all_available_accounts=available_accounts_for_errors,
                transactions_df=transactions_for_summary,
                historical_fx_rates=historical_fx_for_processing, # ADDED: Historical FX Data
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
                taxes = overall_summary_metrics.get("taxes", 0.0)

                new_total_gain_div = realized + unrealized + total_dividends_override - commissions - taxes
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
                             "total_taxes_display": 0.0,
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
                    acct_taxes = metrics.get("total_taxes_display", 0.0)

                    acct_new_total_gain = acct_realized + acct_unrealized + div_amt - acct_commissions - acct_taxes
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
            taxes = overall_summary_metrics.get("taxes", 0.0)

            new_total_gain = total_fifo_gain + unrealized + dividends - commissions - taxes
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
                             "total_taxes_display": 0.0,
                             "total_gain_display": 0.0,
                             "total_buy_cost_display": 0.0,
                             "total_return_pct": 0.0,
                             "market_value": 0.0
                         }

                metrics = account_level_metrics[acct]
                
                _old_acct_realized = metrics.get("total_realized_gain_display", 0.0)
                metrics["total_realized_gain_display"] = acct_fifo_gain
                
                # Update Total Gain for account
                acct_unrealized = metrics.get("total_unrealized_gain_display", 0.0)
                acct_dividends = metrics.get("total_dividends_display", 0.0)
                acct_commissions = metrics.get("total_commissions_display", 0.0)
                acct_taxes = metrics.get("total_taxes_display", 0.0)

                acct_new_total_gain = acct_fifo_gain + acct_unrealized + acct_dividends - acct_commissions - acct_taxes
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
                summary_df_unfiltered["Quantity"].abs()
                >= STOCK_QUANTITY_CLOSE_TOLERANCE
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
    if not summary_df_final.empty:
        # Re-calculate total market value from the final filtered dataframe to ensure it matches
        mkt_val_col = f"Market Value ({display_currency})"
        if mkt_val_col in summary_df_final.columns:
            actual_total_mkt_val = summary_df_final[mkt_val_col].sum()
            if abs(actual_total_mkt_val) > 1e-9:
                summary_df_final["pct_of_total"] = (summary_df_final[mkt_val_col] / actual_total_mkt_val) * 100.0
            else:
                summary_df_final["pct_of_total"] = 0.0
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
    
    # Store full FX rates list for downstream conversions (e.g. projected income)
    overall_summary_metrics["_fx_rates_vs_usd"] = current_fx_rates_vs_usd

    # --- 11. Aggregate Taxes ---
    total_taxes_display = 0.0
    if taxes_local:
        for curr, amt in taxes_local.items():
            rate = get_conversion_rate(curr, display_currency, current_fx_rates_vs_usd)
            if pd.notna(rate):
                total_taxes_display += amt * rate
    overall_summary_metrics["total_taxes"] = total_taxes_display

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

    # Strip source-only accounts from the holdings dict for parity with the
    # summary_df filter — see _build_summary_rows for the rationale. The
    # engine kept these entries during cost-basis backfill, but they should
    # not leak into the API response when the user explicitly excluded the
    # account (otherwise StockDetailModal lots, the AI review fallback, and
    # any other consumer of holdings_dict will re-introduce the phantom row).
    if include_accounts:
        _include_norm = {str(a).strip().upper() for a in include_accounts}
        holdings = {
            k: v
            for k, v in holdings.items()
            if str(k[1] if isinstance(k, tuple) and len(k) > 1 else "").strip().upper() in _include_norm
        }

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
