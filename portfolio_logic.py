# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_logic.py
 Purpose:       Core logic for portfolio calculations, data fetching, and analysis.
                Handles transaction processing, current summary, and historical performance.

Author:        Kit Matan
 Author Email:  kittiwit@gmail.com

 Created:       26/04/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

# --- START OF MODIFIED portfolio_logic.py ---
# --- Imports needed within this function's scope or globally ---
import pandas as pd
from datetime import datetime, date, timedelta, UTC  # Add UTC here
import requests  # Keep for potential future use or different APIs
import os
import json
import numpy as np
from scipy import optimize
from typing import List, Tuple, Dict, Optional, Any, Set
import traceback  # For detailed error logging
from collections import defaultdict
import time  # For adding slight delays if needed (historical fetch)
from io import StringIO  # For historical cache loading

# --- Multiprocessing Imports ---
import multiprocessing
from functools import partial

# --- End Multiprocessing Imports ---
import calendar  # Added for potential market day checks
import hashlib  # Added for cache key hashing
from collections import defaultdict  # Ensure defaultdict is imported
import logging  # Added for logging

# --- Configure Logging Globally (as early as possible) ---
# Set the desired global level here (e.g., logging.INFO, logging.DEBUG)
LOGGING_LEVEL = logging.INFO  # Or logging.DEBUG for more detail

# ADD THIS near the start of the file for easy toggling
HISTORICAL_DEBUG_USD_CONVERSION = (
    False  # Set to True only when debugging this specific issue
)
HISTORICAL_DEBUG_SET_VALUE = (
    False  # Set to True only when debugging this specific issue
)
DEBUG_DATE_VALUE = date(
    2024, 2, 5
)  # Choose a relevant date within your range where SET should have value

# Add near the top if not already present
HISTORICAL_DEBUG_DATE_VALUE = date(
    2025, 4, 29
)  # Choose a relevant date within your range
HISTORICAL_DEBUG_SYMBOL = None  # Optional: Focus on a specific symbol/account

# --- Finance API Import ---
try:
    import yfinance as yf  # Import yfinance

    YFINANCE_AVAILABLE = True
except ImportError:
    logging.warning(
        "Warning: yfinance library not found. Stock/FX fetching and historical data will fail."
    )
    logging.info("         Please install it: pip install yfinance")
    YFINANCE_AVAILABLE = False

    class DummyYFinance:
        def Tickers(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

        def download(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

        def Ticker(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

    yf = DummyYFinance()

# --- Constants ---
CASH_SYMBOL_CSV = "$CASH"  # Standardized cash symbol

# --- Caching ---
DEFAULT_CURRENT_CACHE_FILE_PATH = "portfolio_cache_yf.json"
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = "yf_portfolio_hist_raw_adjusted"
DAILY_RESULTS_CACHE_PATH_PREFIX = (
    "yf_portfolio_daily_results"  # Cache with daily_return & daily_gain
)
YFINANCE_CACHE_DURATION_HOURS = 4  # Keep for CURRENT data

# --- Yahoo Finance Mappings & Configuration ---
# (Keep existing mappings and configs: YFINANCE_INDEX_TICKER_MAP, SYMBOL_MAP_TO_YFINANCE, YFINANCE_EXCLUDED_SYMBOLS, SHORTABLE_SYMBOLS, DEFAULT_CURRENCY)
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())
SYMBOL_MAP_TO_YFINANCE = {
    "BRK.B": "BRK-B",
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "GOOGL": "GOOGL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "LQD": "LQD",
    "SPY": "SPY",
    "VTI": "VTI",
    "KHC": "KHC",
    "DIA": "DIA",
    "AXP": "AXP",
    "BLV": "BLV",
    "NVDA": "NVDA",
    "PLTR": "PLTR",
    "JNJ": "JNJ",
    "XLE": "XLE",
    "VDE": "VDE",
    "BND": "BND",
    "VWO": "VWO",
    "DPZ": "DPZ",
    "QQQ": "QQQ",
    "BHP": "BHP",
    "DAL": "DAL",
    "QSR": "QSR",
    "ASML": "ASML",
    "NLY": "NLY",
    "ADRE": "ADRE",
    "GS": "GS",
    "EPP": "EPP",
    "EFA": "EFA",
    "IBM": "IBM",
    "VZ": "VZ",
    "BBW": "BBW",
    "CVX": "CVX",
    "NKE": "NKE",
    "KO": "KO",
    "BAC": "BAC",
    "VGK": "VGK",
    "C": "C",  # Add others...
    "TLT": "TLT",
    "AGG": "AGG",
    "^GSPC": "^GSPC",
    "VT": "VT",
    "IWM": "IWM",
}
YFINANCE_EXCLUDED_SYMBOLS = set(
    [
        "BBW",
        "IDBOX",
        "IDIOX",
        "ES-Fixed_Income",
        "GENCO:BKK",
        "UOBBC",
        "ES-JUMBO25",
        "SCBCHA-SSF",
        "ES-SET50",
        "ES-Tresury",
        "UOBCG",
        "ES-GQG",
        "SCBRM1",
        "SCBRMS50",
        "AMARIN:BKK",
        "RIMM",
        "SCBSFF",
        "BANPU:BKK",
        "AAV:BKK",
        "CPF:BKK",
        "EMV",
        "IDMOX",
        "BML:BKK",
        "ZEN:BKK",
        "SCBRCTECH",
        "MBK:BKK",
        "DSV",
        "THAI:BKK",
        "IDLOX",
        "SCBRMS&P500",
        "AOT:BKK",
        "BECL:BKK",
        "TCAP:BKK",
        "KRFT",
        "AAUKY",
        "NOK:BKK",
        "ADRE",
        "SCC:BKK",
        "CPALL:BKK",
        "TRUE:BKK",
        "PTT:BKK",
        "ES-FIXED_INCOME",
        "ES-TRESURY",
        "BEM:BKK",
    ]
)
# YFINANCE_EXCLUDED_SYMBOLS = set()
SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}  # Used RIMM instead of BB
DEFAULT_CURRENCY = "USD"

# --- Helper Functions ---


# --- File Hashing Helper ---
def _get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file.

    Args:
        filepath (str): The path to the file.

    Returns:
        str: The SHA256 hash of the file as a hexadecimal string,
             or a specific error string ('FILE_NOT_FOUND',
             'HASHING_ERROR_PERMISSION', 'HASHING_ERROR_IO',
             'HASHING_ERROR_UNEXPECTED') if an error occurs.
    """
    hasher = hashlib.sha256()
    try:
        with open(filepath, "rb") as file:
            while chunk := file.read(8192):  # Read in chunks
                hasher.update(chunk)
        return hasher.hexdigest()
    except FileNotFoundError:
        logging.warning(f"Warning: File not found for hashing: {filepath}")
        return "FILE_NOT_FOUND"
    # --- Refined Exception Handling ---
    except PermissionError:
        logging.error(f"Permission denied accessing file {filepath} for hashing.")
        return "HASHING_ERROR_PERMISSION"
    except IOError as e:
        logging.error(f"I/O error hashing file {filepath}: {e}")
        return "HASHING_ERROR_IO"
    except Exception as e:
        logging.exception(f"Unexpected error hashing file {filepath}")
        return "HASHING_ERROR_UNEXPECTED"
    # --- End Refined Exception Handling ---


# --- REVISED: load_and_clean_transactions (Correct na_values Scope) ---
def load_and_clean_transactions(
    transactions_csv_file: str,
    account_currency_map: Dict,  # Now required
    default_currency: str,  # Now required
) -> Tuple[
    Optional[pd.DataFrame], Optional[pd.DataFrame], Set[int], Dict[int, str], bool, bool
]:  # Added has_errors, has_warnings
    """
    Loads transactions from CSV, performs cleaning, validation, and adds 'Local Currency'.

    Reads a CSV file containing transaction data, renames columns, cleans data types
    (dates, numerics), validates essential fields based on transaction type, and adds
    a 'Local Currency' column based on the account mapping. Rows with critical errors
    (e.g., unparseable dates, missing essential data for a transaction type) are
    dropped.

    Args:
        transactions_csv_file (str): Path to the transactions CSV file.
        account_currency_map (Dict): Mapping from account name (str) to its local currency (str)
                                     (e.g., {'SET': 'THB'}).
        default_currency (str): Default currency (str) to use if an account is not found in the map.

    Returns:
        Tuple containing:
            - pd.DataFrame | None: Cleaned and validated transactions DataFrame, sorted by Date
                                   and original index. Returns None if a critical loading error occurs.
            - pd.DataFrame | None: Original loaded DataFrame (before cleaning), with an added
                                   'original_index' column. Returns None if loading fails.
            - Set[int]: Set of original 0-based indices of rows ignored due to validation errors
                        or unparseable dates.
            - Dict[int, str]: Dictionary mapping ignored original index (int) to the reason (str)
                              why it was ignored.
            - bool: has_errors - True if critical loading/parsing errors occurred that prevent
                                 further processing; False otherwise.
            - bool: has_warnings - True if recoverable issues occurred (e.g., bad dates ignored,
                                     numeric coercion failures, missing non-critical data); False otherwise.
    """
    original_transactions_df: Optional[pd.DataFrame] = None
    transactions_df: Optional[pd.DataFrame] = None
    ignored_row_indices = set()  # Collects ORIGINAL indices of ALL ignored rows
    ignored_reasons: Dict[int, str] = {}  # Maps ORIGINAL index to first reason ignored
    has_errors = False  # Flag for critical loading errors
    has_warnings = False  # Flag for recoverable issues

    # --- Define na_values at the top level of the function ---
    na_values_list = [
        "",
        "#N/A",
        "#N/A N/A",
        "#NA",
        "-1.#IND",
        "-1.#QNAN",
        "-NaN",
        "-nan",
        "1.#IND",
        "1.#QNAN",
        "<NA>",
        "N/A",
        "NA",
        "NULL",
        "NaN",
        "n/a",
        "nan",
        "null",
    ]

    logging.info(
        f"Helper: Attempting to load transactions from: {transactions_csv_file}"
    )
    try:
        dtype_spec = {
            "Quantity of Units": str,
            "Amount per unit": str,
            "Total Amount": str,
            "Fees": str,
            "Split Ratio (new shares per old share)": str,
        }
        if isinstance(transactions_csv_file, str):
            file_source = transactions_csv_file
        else:
            file_source = transactions_csv_file
        original_transactions_df_no_index = pd.read_csv(
            file_source,
            header=0,
            skipinitialspace=True,
            keep_default_na=True,
            na_values=na_values_list,
            dtype=dtype_spec,
            encoding="utf-8",
        )  # Use na_values_list

        # print("--- RAW DATAFRAME DEBUG ---")
        # print(original_transactions_df_no_index[original_transactions_df_no_index['Stock / ETF Symbol'] == 'AAPL'].to_string())
        # print("--- END RAW DATAFRAME DEBUG ---")

        original_transactions_df = original_transactions_df_no_index.copy()
        # --- Assign 0-based index as original_index ---
        original_transactions_df["original_index"] = original_transactions_df.index
        transactions_df = original_transactions_df.copy()

        logging.info(f"Helper: Successfully loaded {len(transactions_df)} records.")
    # --- (Exception handling for load) ---
    except FileNotFoundError:
        logging.error(f"Transaction file not found: {transactions_csv_file}")
        has_errors = True
        return (
            None,
            None,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except pd.errors.ParserError as e:
        logging.error(f"Error parsing CSV file {transactions_csv_file}: {e}")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except UnicodeDecodeError as e:
        logging.error(
            f"Encoding error reading CSV file {transactions_csv_file}: {e}. Try saving as UTF-8."
        )
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except PermissionError:
        logging.error(f"Permission denied reading CSV file: {transactions_csv_file}")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except Exception as e:
        logging.exception(
            f"Unexpected error loading transactions from {transactions_csv_file}"
        )
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )

    if transactions_df is None or transactions_df.empty:
        logging.warning("Warning: Transactions DataFrame is empty after loading.")
        has_warnings = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )

    try:
        # --- Rename Columns ---
        column_mapping = {
            "Date (MMM DD, YYYY)": "Date",
            "Transaction Type": "Type",
            "Stock / ETF Symbol": "Symbol",
            "Quantity of Units": "Quantity",
            "Amount per unit": "Price/Share",
            "Total Amount": "Total Amount",
            "Fees": "Commission",
            "Split Ratio (new shares per old share)": "Split Ratio",
            "Investment Account": "Account",
            "Note": "Note",
        }
        required_original_cols = [
            "Date (MMM DD, YYYY)",
            "Transaction Type",
            "Stock / ETF Symbol",
            "Investment Account",
        ]
        actual_columns = [col.strip() for col in transactions_df.columns]
        required_stripped_cols = {col: col.strip() for col in required_original_cols}
        missing_original = [
            orig_col
            for orig_col, stripped_col in required_stripped_cols.items()
            if stripped_col not in actual_columns
        ]
        if missing_original:
            raise ValueError(
                f"Missing essential CSV columns: {missing_original}. Found: {transactions_df.columns.tolist()}"
            )

        rename_dict = {
            stripped_csv_header: internal_name
            for csv_header, internal_name in column_mapping.items()
            if (stripped_csv_header := csv_header.strip()) in actual_columns
        }
        transactions_df.columns = actual_columns
        transactions_df.rename(columns=rename_dict, inplace=True)

        # --- Basic Cleaning ---
        transactions_df["Symbol"] = (
            transactions_df["Symbol"]
            .fillna("UNKNOWN_SYMBOL")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        transactions_df.loc[transactions_df["Symbol"] == "", "Symbol"] = (
            "UNKNOWN_SYMBOL"
        )
        transactions_df.loc[transactions_df["Symbol"] == "$CASH", "Symbol"] = (
            CASH_SYMBOL_CSV
        )
        transactions_df["Type"] = (
            transactions_df["Type"]
            .fillna("UNKNOWN_TYPE")
            .astype(str)
            .str.strip()
            .str.lower()
        )
        transactions_df.loc[transactions_df["Type"] == "", "Type"] = "UNKNOWN_TYPE"
        transactions_df["Account"] = (
            transactions_df["Account"].fillna("Unknown").astype(str).str.strip()
        )
        transactions_df.loc[transactions_df["Account"] == "", "Account"] = "Unknown"
        transactions_df["Local Currency"] = (
            transactions_df["Account"]
            .map(account_currency_map)
            .fillna(default_currency)
        )
        logging.info(f"Helper: Added 'Local Currency' column.")

        # Store original date strings before attempting conversion
        original_date_strings = transactions_df["Date"].copy()

        # --- Numeric Conversion ---
        numeric_cols = [
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Split Ratio",
        ]
        for col in numeric_cols:
            if col in transactions_df.columns:
                # ... (numeric conversion logic with debug logging remains the same) ...
                original_col_copy = transactions_df[col].copy()
                if transactions_df[col].dtype == "object":
                    cleaned_strings = (
                        transactions_df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                    )
                    converted_col = pd.to_numeric(cleaned_strings, errors="coerce")
                else:
                    converted_col = pd.to_numeric(transactions_df[col], errors="coerce")
                nan_mask = converted_col.isna() & original_col_copy.notna()
                if nan_mask.any():
                    has_warnings = True
                    logging.warning(
                        f"Warning: Coercion to numeric failed for column '{col}' on {nan_mask.sum()} rows."
                    )
                    # ... (optional detailed debug logging for failed coercion) ...
                transactions_df[col] = converted_col

        # --- Date Parsing and Dropping (DO THIS FIRST) ---
        transactions_df["Date"] = pd.to_datetime(original_date_strings, errors="coerce")
        if transactions_df["Date"].isnull().any():
            # ... (Fallback date parsing logic) ...
            formats_to_try = ["%b %d, %Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%b-%Y", "%Y%m%d"]
            for fmt in formats_to_try:
                if transactions_df["Date"].isnull().any():
                    nat_mask = transactions_df["Date"].isnull()
                    try:
                        parsed = pd.to_datetime(
                            original_date_strings[nat_mask].astype(str),
                            format=fmt,
                            errors="coerce",
                        )
                        transactions_df.loc[nat_mask, "Date"] = parsed
                    except (ValueError, TypeError):
                        pass
            if transactions_df["Date"].isnull().any():
                nat_mask = transactions_df["Date"].isnull()
                try:
                    inferred = pd.to_datetime(
                        original_date_strings[nat_mask],
                        infer_datetime_format=True,
                        errors="coerce",
                    )
                    transactions_df.loc[nat_mask, "Date"] = inferred
                except (ValueError, TypeError):
                    pass

        bad_date_mask = transactions_df["Date"].isnull()
        bad_date_df_indices_to_drop = transactions_df.index[
            bad_date_mask
        ]  # Use a distinct name for clarity

        if not bad_date_df_indices_to_drop.empty:
            has_warnings = True
            # --- CORRECTED LOGGING ---
            logging.warning(
                f"Warning: {len(bad_date_df_indices_to_drop)} rows ignored due to unparseable dates."
            )
            # --- END CORRECTION ---
            orig_indices_bad_date = transactions_df.loc[
                bad_date_df_indices_to_drop, "original_index"
            ].tolist()
            for orig_idx in orig_indices_bad_date:
                if orig_idx not in ignored_reasons:
                    ignored_reasons[orig_idx] = "Invalid/Unparseable Date"
            ignored_row_indices.update(orig_indices_bad_date)
            transactions_df.drop(bad_date_df_indices_to_drop, inplace=True)
            transactions_df.reset_index(drop=True, inplace=True)
            logging.debug(f"DataFrame index reset after date drop.")
        # --- End Date Parsing ---

        # --- Fill Commission NaNs ---
        transactions_df["Commission"] = transactions_df["Commission"].fillna(0.0)

        # --- Flagging Rows to Drop (Collect DF Indices AFTER Reset) ---
        initial_row_count = len(transactions_df)
        df_indices_to_drop = pd.Index([])  # Collect DataFrame indices
        local_has_warnings_flagging = False

        def flag_for_drop_direct(df_indices, reason):
            nonlocal local_has_warnings_flagging
            if not df_indices.empty:
                valid_df_indices = df_indices.intersection(transactions_df.index)
                if not valid_df_indices.empty:
                    local_has_warnings_flagging = True
                    original_indices = transactions_df.loc[
                        valid_df_indices, "original_index"
                    ].tolist()
                    logging.debug(
                        f"FLAGGING DF Indices TO DROP: Reason='{reason}', DF Indices={valid_df_indices.tolist()}, Orig Indices={original_indices}"
                    )
                    for orig_idx in original_indices:
                        if orig_idx not in ignored_reasons:
                            ignored_row_indices.add(orig_idx)
                            ignored_reasons[orig_idx] = reason
                    return valid_df_indices
            return pd.Index([])

        logging.debug("--- START FLAGGING (AFTER DATE DROP & RESET) ---")
        try:
            # Define masks (using the state AFTER date drop and reset)
            is_buy_sell_stock = transactions_df["Type"].isin(
                ["buy", "sell", "deposit", "withdrawal"]
            ) & (transactions_df["Symbol"] != CASH_SYMBOL_CSV)
            is_short_stock = transactions_df["Type"].isin(
                ["short sell", "buy to cover"]
            ) & transactions_df["Symbol"].isin(SHORTABLE_SYMBOLS)
            is_split = transactions_df["Type"].isin(["split", "stock split"])
            is_dividend = transactions_df["Type"] == "dividend"
            is_fees = transactions_df["Type"] == "fees"
            is_cash_tx = (
                transactions_df["Symbol"] == CASH_SYMBOL_CSV
            ) & transactions_df["Type"].isin(["buy", "sell", "deposit", "withdrawal"])
            nan_qty = transactions_df["Quantity"].isnull()
            nan_price = transactions_df["Price/Share"].isnull()
            nan_qty_or_price = nan_qty | nan_price
            # Robust Split Ratio Check
            numeric_split_ratio = pd.to_numeric(
                transactions_df["Split Ratio"], errors="coerce"
            )
            is_nan_split = numeric_split_ratio.isnull()
            is_le_zero_split = pd.Series(False, index=transactions_df.index)
            numeric_mask_split = pd.notna(numeric_split_ratio)
            if numeric_mask_split.any():
                is_le_zero_split.loc[numeric_mask_split] = (
                    numeric_split_ratio.loc[numeric_mask_split] <= 0
                )
            invalid_split = is_nan_split | is_le_zero_split
            missing_div = transactions_df["Total Amount"].isnull() & nan_price
            # Check if original commission value was NaN
            commission_was_nan_mask = pd.Series(False, index=transactions_df.index)
            if (
                original_transactions_df is not None
                and "Commission" in original_transactions_df.columns
                and "original_index" in original_transactions_df.columns
            ):
                aligned_original_commission = original_transactions_df.set_index(
                    "original_index"
                )["Commission"].reindex(transactions_df["original_index"])
                # --- Use na_values_list defined at function start ---
                commission_was_nan_mask = (
                    aligned_original_commission.isnull()
                    | aligned_original_commission.isin(na_values_list)
                )
            is_unknown = (transactions_df["Symbol"] == "UNKNOWN_SYMBOL") | (
                transactions_df["Type"] == "UNKNOWN_TYPE"
            )

            # Flag DataFrame indices for each reason
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_buy_sell_stock & nan_qty_or_price],
                    "Missing Qty/Price Stock",
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_short_stock & nan_qty_or_price],
                    "Missing Qty/Price Short",
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_cash_tx & nan_qty], "Missing $CASH Qty"
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_split & invalid_split],
                    "Missing/Invalid Split Ratio",
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_dividend & missing_div],
                    "Missing Dividend Amt/Price",
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_fees & commission_was_nan_mask],
                    "Missing Fee Commission",
                )
            )
            df_indices_to_drop = df_indices_to_drop.union(
                flag_for_drop_direct(
                    transactions_df.index[is_unknown], "Unknown Symbol/Type"
                )
            )

            logging.debug(
                f"--- END FLAGGING --- Final DF Indices to Drop: {sorted(df_indices_to_drop.tolist())}"
            )

        except Exception as e_mask_flag:
            logging.exception(
                "CRITICAL ERROR during mask creation/flagging after date drop"
            )
            has_errors = True
            return (
                None,
                original_transactions_df,
                ignored_row_indices,
                ignored_reasons,
                has_errors,
                has_warnings,
            )

        # --- FINAL Drop rows using the collected DATAFRAME indices ---
        if not df_indices_to_drop.empty:
            if local_has_warnings_flagging:
                has_warnings = True
            valid_indices_to_drop = df_indices_to_drop.intersection(
                transactions_df.index
            )
            if not valid_indices_to_drop.empty:
                original_indices_ignored = sorted(
                    list(transactions_df.loc[valid_indices_to_drop, "original_index"])
                )
                reasons_for_drop = {
                    idx: ignored_reasons.get(idx, "Unknown validation")
                    for idx in original_indices_ignored
                }
                logging.warning(
                    f"Warning: Dropping {len(valid_indices_to_drop)} final rows (Original Indices: {original_indices_ignored}). Reasons: {reasons_for_drop}"
                )
                transactions_df.drop(valid_indices_to_drop, inplace=True)

        if transactions_df.empty:
            logging.warning(
                "Helper WARN: All transactions dropped during cleaning validation."
            )
            has_warnings = True
            return (
                None,
                original_transactions_df,
                ignored_row_indices,
                ignored_reasons,
                has_errors,
                has_warnings,
            )

        # Sort and Return
        transactions_df.sort_values(
            by=["Date", "original_index"], inplace=True, ascending=True
        )
        return (
            transactions_df,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )

    # --- (Refined Exception Handling for Cleaning Block remains the same) ---
    except ValueError as e:
        logging.error(f"Data integrity error during cleaning: {e}")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except KeyError as e:
        logging.error(f"Missing expected column during cleaning: {e}")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except TypeError as e:
        logging.error(f"Type error during cleaning checks: {e}")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    except Exception as e:
        logging.exception(f"CRITICAL ERROR during data cleaning helper")
        has_errors = True
        return (
            None,
            original_transactions_df,
            ignored_row_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
        )
    # --- End Refined Exception Handling ---


# --- IRR/MWR Calculation Functions ---
def calculate_npv(rate: float, dates: List[date], cash_flows: List[float]) -> float:
    """
    Calculates the Net Present Value (NPV) of a series of cash flows.

    Discounts each cash flow back to the date of the first cash flow using the
    provided discount rate, assuming time is measured in years (days/365.0).
    Handles potential errors like invalid rates, date/flow mismatches, and
    calculation issues (e.g., division by zero, overflow).

    Args:
        rate (float): The discount rate per period (annualized).
        dates (List[date]): A list of dates corresponding to the cash flows. Must be sorted.
        cash_flows (List[float]): A list of cash flows. Must be the same length as dates.

    Returns:
        float: The calculated Net Present Value (NPV). Returns np.nan if inputs are invalid,
               lengths mismatch, dates are unsorted, or calculation errors occur.

    Raises:
        ValueError: If the lengths of `dates` and `cash_flows` do not match.
                    (This is primarily for the IRR solver using this function).
    """
    # Refined Error Handling
    if not isinstance(rate, (int, float)) or not np.isfinite(rate):
        logging.debug("NPV Calculation Error: Invalid rate provided.")
        return np.nan
    if len(dates) != len(cash_flows):
        logging.error("NPV Calculation Error: Dates and cash_flows lengths mismatch.")
        raise ValueError(
            "Dates and cash_flows must have the same length."
        )  # Raise for IRR solver
    if not dates:
        return 0.0
    base = 1.0 + rate
    if base <= 1e-9:
        logging.debug(
            f"NPV Calculation Warning: Base (1+rate) is <= 1e-9 ({base}). Returning NaN."
        )
        return np.nan  # Avoid issues with rate = -1 or less

    start_date = dates[0]
    npv = 0.0
    for i in range(len(cash_flows)):
        try:
            if not isinstance(dates[i], date) or not isinstance(start_date, date):
                logging.debug(f"NPV Calc Error: Invalid date type at index {i}.")
                return np.nan
            time_delta_years = (dates[i] - start_date).days / 365.0
            if not np.isfinite(time_delta_years) or (
                time_delta_years < -1e-9 and i > 0
            ):
                logging.debug(
                    f"NPV Calc Error: Invalid time delta ({time_delta_years}) at index {i}."
                )
                return np.nan
            if not np.isfinite(cash_flows[i]):
                continue  # Skip non-finite flows silently

            if abs(base) < 1e-9 and time_delta_years != 0:
                logging.debug(
                    "NPV Calc Warning: Base is near zero with non-zero time delta. Returning NaN."
                )
                return np.nan

            # Check for negative base with non-integer exponent
            if base < 0 and time_delta_years != int(time_delta_years):
                logging.debug(
                    f"NPV Calc Warning: Negative base ({base}) with non-integer exponent ({time_delta_years}). Returning NaN."
                )
                return np.nan

            denominator = base**time_delta_years
            if not np.isfinite(denominator) or abs(denominator) < 1e-12:
                logging.debug(
                    f"NPV Calc Warning: Invalid denominator ({denominator}) at index {i}. Returning NaN."
                )
                return np.nan

            term_value = cash_flows[i] / denominator
            if not np.isfinite(term_value):
                logging.debug(
                    f"NPV Calc Warning: Non-finite term value ({term_value}) at index {i}. Returning NaN."
                )
                return np.nan
            npv += term_value
        except OverflowError:
            logging.warning(
                f"NPV Calculation OverflowError at index {i} (Rate: {rate}, TimeDelta: {time_delta_years}). Returning NaN."
            )
            return np.nan
        except TypeError as e:
            logging.warning(
                f"NPV Calculation TypeError at index {i}: {e}. Returning NaN."
            )
            return np.nan
        except Exception as e:
            logging.exception(f"Unexpected error in NPV calculation loop at index {i}")
            return np.nan  # Catch any other unexpected calculation errors
    return float(npv) if np.isfinite(npv) else np.nan


def calculate_irr(dates: List[date], cash_flows: List[float]) -> float:
    """
    Calculates the Internal Rate of Return (IRR/MWR) for a series of cash flows.

    Finds the discount rate at which the Net Present Value (NPV) of the cash flows equals zero.
    Uses numerical methods (Newton-Raphson with Brentq fallback) to solve for the rate.
    Requires at least two cash flows, sorted dates, and a valid investment pattern
    (typically starting with a negative flow and having at least one positive flow later).

    Args:
        dates (List[date]): A list of dates corresponding to the cash flows. Must be sorted.
        cash_flows (List[float]): A list of cash flows. Must be the same length as dates.

    Returns:
        float: The calculated Internal Rate of Return (IRR) as a decimal (e.g., 0.1 for 10%),
               or np.nan if the calculation fails, inputs are invalid, dates are unsorted,
               or the cash flow pattern doesn't allow for a standard IRR calculation
               (e.g., all positive flows, all negative flows, first non-zero flow is positive).
    """
    # 1. Basic Input Validation
    if len(dates) < 2 or len(cash_flows) < 2 or len(dates) != len(cash_flows):
        logging.debug("DEBUG IRR: Fail - Length mismatch or < 2")  # Optional Debug
        return np.nan
    if any(
        not isinstance(cf, (int, float)) or not np.isfinite(cf) for cf in cash_flows
    ):
        logging.debug("DEBUG IRR: Fail - Non-finite cash flows")  # Optional Debug
        return np.nan
    # Check dates are valid and sorted
    try:
        # Ensure elements are date objects before comparison
        if not all(isinstance(d, date) for d in dates):
            raise TypeError("Not all elements in dates are date objects")
        for i in range(1, len(dates)):
            if dates[i] < dates[i - 1]:
                raise ValueError("Dates are not sorted")
    except (TypeError, ValueError) as e:
        logging.debug(f"DEBUG IRR: Fail - Date validation error: {e}")  # Optional Debug
        return np.nan

    # 2. Cash Flow Pattern Validation (Stricter)
    first_non_zero_flow = None
    first_non_zero_idx = -1
    non_zero_cfs_list = []  # Also collect non-zero flows

    for idx, cf in enumerate(cash_flows):
        if abs(cf) > 1e-9:
            non_zero_cfs_list.append(cf)
            if first_non_zero_flow is None:
                first_non_zero_flow = cf
                first_non_zero_idx = idx

    if first_non_zero_flow is None:  # All flows are zero or near-zero
        logging.debug("DEBUG IRR: Fail - All flows are zero")  # Optional Debug
        return np.nan

    # Stricter Check 1: First non-zero flow MUST be negative (typical investment)
    if first_non_zero_flow >= -1e-9:
        logging.debug(
            f"DEBUG IRR: Fail - First non-zero flow is non-negative: {first_non_zero_flow} in {cash_flows}"
        )  # Optional Debug
        return np.nan

    # Stricter Check 2: Must have at least one positive flow overall
    has_positive_flow = any(cf > 1e-9 for cf in non_zero_cfs_list)
    if not has_positive_flow:
        logging.debug(
            f"DEBUG IRR: Fail - No positive flows found: {cash_flows}"
        )  # Optional Debug
        return np.nan

    # Check 3 (Redundant but safe): Ensure not ALL flows are negative (covered by check 2)
    # all_negative = all(cf <= 1e-9 for cf in non_zero_cfs_list)
    # if all_negative:
    #     return np.nan

    # 3. Solver Logic (Keep previous robust version)
    irr_result = np.nan
    try:
        # Newton-Raphson with validation
        irr_result = optimize.newton(
            calculate_npv, x0=0.1, args=(dates, cash_flows), tol=1e-6, maxiter=100
        )
        if (
            not np.isfinite(irr_result) or irr_result <= -1.0 or irr_result > 100.0
        ):  # Check range
            raise RuntimeError("Newton result out of reasonable range")
        npv_check = calculate_npv(irr_result, dates, cash_flows)
        if not np.isclose(
            npv_check, 0.0, atol=1e-4
        ):  # Check if it finds the root accurately
            raise RuntimeError("Newton result did not produce zero NPV")

    except (RuntimeError, OverflowError):
        # Brentq fallback
        try:
            lower_bound, upper_bound = -0.9999, 50.0  # Sensible bounds
            npv_low = calculate_npv(lower_bound, dates, cash_flows)
            npv_high = calculate_npv(upper_bound, dates, cash_flows)
            if (
                pd.notna(npv_low) and pd.notna(npv_high) and npv_low * npv_high < 0
            ):  # Check sign change
                irr_result = optimize.brentq(
                    calculate_npv,
                    a=lower_bound,
                    b=upper_bound,
                    args=(dates, cash_flows),
                    xtol=1e-6,
                    rtol=1e-6,
                    maxiter=100,
                )
                # Final check on Brentq result
                if not np.isfinite(irr_result) or irr_result <= -1.0:
                    irr_result = np.nan
            else:  # Bounds don't bracket
                irr_result = np.nan
        except (ValueError, RuntimeError, OverflowError, Exception):
            irr_result = np.nan  # Brentq failed

    # 4. Final Validation and Return
    if not (
        isinstance(irr_result, (float, int))
        and np.isfinite(irr_result)
        and irr_result > -1.0
    ):
        logging.debug(
            f"DEBUG IRR: Fail - Final result invalid: {irr_result}"
        )  # Optional Debug
        return np.nan

    return irr_result


# --- Cash Flow Helpers ---
# --- START OF REVISED get_cash_flows_for_symbol_account ---
def get_cash_flows_for_symbol_account(
    symbol: str,
    account: str,
    transactions: pd.DataFrame,
    final_market_value_local: float,
    end_date: date,
) -> Tuple[List[date], List[float]]:
    """
    Extracts LOCAL currency cash flows for a specific symbol/account pair for IRR calculation.

    Filters transactions for the given symbol and account. Calculates the cash flow
    amount in the holding's local currency for each relevant transaction (buy, sell,
    dividend, fees, shorting actions). Appends the final market value (in local currency)
    as the last cash flow on the end_date.

    Args:
        symbol (str): The stock or ETF symbol.
        account (str): The investment account name.
        transactions (pd.DataFrame): The transactions DataFrame, filtered for the relevant
                                     period and scope. Must contain 'Date', 'Symbol', 'Account',
                                     'Type', 'Quantity', 'Price/Share', 'Commission',
                                     'Total Amount', and 'Local Currency' columns.
        final_market_value_local (float): The final market value of the holding in its
                                          local currency as of the end_date.
        end_date (date): The end date for the calculation period.

    Returns:
        Tuple[List[date], List[float]]: A tuple containing:
            - List[date]: A list of dates for the cash flows, sorted chronologically.
            - List[float]: A list of cash flows in the local currency. Buys/fees are negative,
                           sells/dividends are positive. The final market value is added
                           as a positive flow on the end_date.
            Returns ([], []) if no relevant transactions or final value exist, or if the
            cash flow pattern is invalid for IRR.
    """
    # Assumes 'transactions' df ALREADY contains 'Local Currency' column.
    symbol_account_tx_filtered = transactions[
        (transactions["Symbol"] == symbol) & (transactions["Account"] == account)
    ]
    if symbol_account_tx_filtered.empty:
        return [], []
    symbol_account_tx = symbol_account_tx_filtered.copy()
    dates_flows = defaultdict(float)
    symbol_account_tx.sort_values(
        by=["Date", "original_index"], inplace=True, ascending=True
    )

    for _, row in symbol_account_tx.iterrows():
        tx_type = str(row.get("Type", "")).lower().strip()
        # --- Retrieve values ---
        qty_val = row.get("Quantity")
        price_val = row.get("Price/Share")
        commission_val = row.get("Commission")
        total_amount_val = row.get("Total Amount")
        # --- Convert and handle potential NaNs ---
        qty = pd.to_numeric(qty_val, errors="coerce")
        price_local = pd.to_numeric(price_val, errors="coerce")

        # --- FIX: Handle NaN for scalar commission ---
        commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
        commission_local = (
            0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        )
        # --- END FIX ---

        total_amount_local = pd.to_numeric(total_amount_val, errors="coerce")
        tx_date = row["Date"].date()
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        # --- Calculations (unchanged logic, operates on local currency values) ---
        # ... (Calculations for buy, sell, short, dividend, fees, split remain the same) ...
        if tx_type == "buy" or tx_type == "deposit":
            if pd.notna(qty) and qty > 0 and pd.notna(price_local):
                cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == "sell" or tx_type == "withdrawal":
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = (qty_abs * price_local) - commission_local
        elif tx_type == "short sell" and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = (qty_abs * price_local) - commission_local
        elif tx_type == "buy to cover" and symbol in SHORTABLE_SYMBOLS:
            if pd.notna(price_local) and pd.notna(qty) and qty_abs > 0:
                cash_flow_local = -((qty_abs * price_local) + commission_local)
        elif tx_type == "dividend":
            dividend_amount_local_cf = 0.0
            if pd.notna(total_amount_local):
                dividend_amount_local_cf = total_amount_local
            elif pd.notna(price_local):
                dividend_amount_local_cf = (
                    (qty_abs * price_local)
                    if (pd.notna(qty) and qty_abs > 0)
                    else price_local
                )
            if pd.notna(dividend_amount_local_cf):
                cash_flow_local = dividend_amount_local_cf - commission_local
        elif tx_type == "fees":
            cash_flow_local = -(abs(commission_local))
        elif tx_type in ["split", "stock split"]:
            cash_flow_local = 0.0
            if commission_local != 0:
                cash_flow_local = -abs(commission_local)

        # --- Aggregation (unchanged) ---
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            try:
                flow_to_add = float(cash_flow_local)
                dates_flows[tx_date] += flow_to_add
            except (ValueError, TypeError):
                logging.warning(
                    f"Warning IRR CF Gen ({symbol}/{account}): Could not convert cash_flow_local {cash_flow_local} to float. Skipping flow."
                )

    # --- Final sorting, adding final value, checks (unchanged) ---
    # ... (Rest of the function remains the same) ...
    sorted_dates = sorted(dates_flows.keys())
    if not sorted_dates and abs(final_market_value_local) < 1e-9:
        return [], []
    final_dates = list(sorted_dates)
    final_flows = [float(dates_flows[d]) for d in final_dates]
    final_market_value_local_abs = (
        abs(final_market_value_local) if pd.notna(final_market_value_local) else 0.0
    )
    if final_market_value_local_abs > 1e-9 and isinstance(end_date, date):
        # Ensure we only add final value if there are initial cash flows OR if the final value itself exists
        if not final_dates:
            # No prior cash flows, but a final value exists. Need a dummy start date?
            # Let's assume the first transaction date for the holding is the "start"
            first_tx_date_for_holding = (
                symbol_account_tx["Date"].min().date()
                if not symbol_account_tx.empty
                else end_date
            )
            if end_date >= first_tx_date_for_holding:
                final_dates.append(end_date)
                final_flows.append(final_market_value_local_abs)
            else:  # End date is before the first transaction? Should not happen if holding exists.
                return [], []
        elif end_date >= final_dates[-1]:
            if final_dates[-1] == end_date:
                final_flows[-1] += final_market_value_local_abs
            else:
                final_dates.append(end_date)
                final_flows.append(final_market_value_local_abs)
        # If end_date is before the last cash flow, the final value shouldn't be added here. calculate_irr handles time value.

    if len(final_dates) < 2:
        return (
            [],
            [],
        )  # Need at least two points for IRR (e.g., initial investment + final value)
    non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
    if (
        not non_zero_final_flows
        or all(cf >= -1e-9 for cf in non_zero_final_flows)
        or all(cf <= 1e-9 for cf in non_zero_final_flows)
    ):
        return [], []
    return final_dates, final_flows


# --- END OF REVISED get_cash_flows_for_symbol_account ---


# --- START OF REVISED get_cash_flows_for_mwr ---
def get_cash_flows_for_mwr(
    account_transactions: pd.DataFrame,
    final_account_market_value: float,  # Already in target_currency
    end_date: date,
    target_currency: str,
    fx_rates: Optional[Dict[str, float]],  # Expects standard 'FROM/TO' -> rate format
    display_currency: str,  # Used for warning msg only (REMOVED - fx_rates needed)
) -> Tuple[List[date], List[float]]:
    """
    Calculates cash flows for Money-Weighted Return (MWR) for a specific account in the target currency.

    Processes transactions for a single account, calculating the cash flow impact of each
    transaction (buys, sells, dividends, fees, cash deposits/withdrawals) in its local currency.
    Converts these local currency flows to the `target_currency` using the provided `fx_rates`.
    The final account market value (already in `target_currency`) is added as the terminal flow.
    Note: The sign convention for MWR cash flows is often flipped compared to IRR (deposits/buys
    are positive, withdrawals/sells are negative) before solving. This function applies that flip.

    Args:
        account_transactions (pd.DataFrame): The transactions DataFrame filtered for a specific account.
                                             Must contain 'Date', 'Symbol', 'Type', 'Quantity',
                                             'Price/Share', 'Commission', 'Total Amount', and
                                             'Local Currency' columns.
        final_account_market_value (float): The final market value of the entire account in the
                                            `target_currency` as of the end_date.
        end_date (date): The end date for the calculation period.
        target_currency (str): The target currency for the MWR calculation.
        fx_rates (Optional[Dict[str, float]]): A dictionary of FX rates relative to a base currency
                                               (typically USD, e.g., {'JPY': 150.0, 'EUR': 0.9})
                                               used for currency conversion via `get_conversion_rate`.
        display_currency (str): The display currency used in log messages (informational only).

    Returns:
        Tuple[List[date], List[float]]: A tuple containing:
            - List[date]: A list of dates for the cash flows, sorted chronologically.
            - List[float]: A list of cash flows in the `target_currency`, with signs flipped
                           for MWR calculation (deposits/buys positive, withdrawals/sells negative).
                           The final market value is added as a positive flow on the end_date.
            Returns ([], []) if no relevant transactions or final value exist, or if the
            cash flow pattern is invalid for IRR/MWR.
    """
    # Assumes 'account_transactions' df ALREADY contains 'Local Currency' column.
    if account_transactions.empty:
        return [], []
    acc_tx_copy = account_transactions.copy()
    dates_flows = defaultdict(float)
    acc_tx_copy.sort_values(by=["Date", "original_index"], inplace=True, ascending=True)

    for _, row in acc_tx_copy.iterrows():
        # ... (Get tx_type, symbol, qty, price, total_amount - unchanged) ...
        tx_type = str(row.get("Type", "")).lower().strip()
        symbol = row["Symbol"]
        qty = pd.to_numeric(row["Quantity"], errors="coerce")
        price_local = pd.to_numeric(row["Price/Share"], errors="coerce")
        commission_val = row.get("Commission")
        total_amount_local = pd.to_numeric(row.get("Total Amount"), errors="coerce")

        # --- FIX: Handle NaN for scalar commission ---
        commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
        commission_local = (
            0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
        )
        # --- END FIX ---

        tx_date = row["Date"].date()
        # --- Get Local Currency from the DataFrame ---
        local_currency = row["Local Currency"]
        # -------------------------------------------
        cash_flow_local = 0.0
        qty_abs = abs(qty) if pd.notna(qty) else 0.0

        # --- MWR Flow Logic (unchanged, calculates flow in local currency) ---
        # ... (Calculations for buy, sell, short, dividend, fees, split, cash remain the same) ...
        if symbol != CASH_SYMBOL_CSV:
            if tx_type == "buy":
                if pd.notna(qty) and qty > 0 and pd.notna(price_local):
                    cash_flow_local = -(
                        (qty_abs * price_local) + commission_local
                    )  # OUT (-)
            elif tx_type == "sell":
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = (
                        qty_abs * price_local
                    ) - commission_local  # IN (+)
            elif tx_type == "short sell" and symbol in SHORTABLE_SYMBOLS:
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = (
                        qty_abs * price_local
                    ) - commission_local  # IN (+)
            elif tx_type == "buy to cover" and symbol in SHORTABLE_SYMBOLS:
                if pd.notna(price_local) and qty_abs > 0:
                    cash_flow_local = -(
                        (qty_abs * price_local) + commission_local
                    )  # OUT (-)
            elif tx_type == "dividend":
                dividend_amount_local_cf = 0.0
                if pd.notna(total_amount_local) and total_amount_local != 0:
                    dividend_amount_local_cf = total_amount_local
                elif pd.notna(price_local) and price_local != 0:
                    dividend_amount_local_cf = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                cash_flow_local = dividend_amount_local_cf - commission_local  # IN (+)
            elif tx_type == "fees":
                if pd.notna(commission_local):
                    cash_flow_local = -(abs(commission_local))  # OUT (-)
            elif tx_type in ["split", "stock split"]:
                cash_flow_local = 0.0
                if pd.notna(commission_local) and commission_local != 0:
                    cash_flow_local = -abs(commission_local)  # OUT (-)
        elif symbol == CASH_SYMBOL_CSV:
            if tx_type == "deposit" or tx_type == "buy":
                if pd.notna(qty):
                    cash_flow_local = abs(qty)  # IN (+)
                cash_flow_local -= commission_local
            elif tx_type == "withdrawal" or tx_type == "sell":
                if pd.notna(qty):
                    cash_flow_local = -abs(qty)  # OUT (-)
                cash_flow_local -= commission_local
            elif tx_type == "dividend":
                dividend_amount_local_cf = 0.0
                if pd.notna(total_amount_local) and total_amount_local != 0:
                    dividend_amount_local_cf = total_amount_local
                elif pd.notna(price_local) and price_local != 0:
                    dividend_amount_local_cf = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                cash_flow_local = dividend_amount_local_cf - commission_local  # IN (+)
            elif tx_type == "fees":
                if pd.notna(commission_local):
                    cash_flow_local = -abs(commission_local)  # OUT (-)

        # --- Convert to target currency ---
        cash_flow_target = cash_flow_local
        if pd.notna(cash_flow_local) and abs(cash_flow_local) > 1e-9:
            if local_currency != target_currency:
                # --- Use get_conversion_rate helper ---
                rate = get_conversion_rate(local_currency, target_currency, fx_rates)
                # --------------------------------------
                if (
                    rate == 1.0 and local_currency != target_currency
                ):  # Rate lookup failed or was 1.0 incorrectly
                    logging.warning(
                        f"Warning: MWR calc cannot convert flow on {tx_date} from {local_currency} to {target_currency} (FX rate missing/invalid). Skipping flow."
                    )  # Reduced verbosity
                    cash_flow_target = (
                        0.0  # Assign 0 instead of NaN to prevent downstream errors
                    )
                else:
                    cash_flow_target = cash_flow_local * rate

            if pd.notna(cash_flow_target) and abs(cash_flow_target) > 1e-9:
                dates_flows[tx_date] += cash_flow_target
            # else: Handle potential NaN after conversion if needed, currently skipped implicitly

    # --- Final sorting, adding final value, sign flip, checks (unchanged) ---
    # ... (Rest of the function remains the same) ...
    sorted_dates = sorted(dates_flows.keys())
    # If no cash flows generated and final MV is zero, return empty
    if not sorted_dates and abs(final_account_market_value) < 1e-9:
        return [], []

    final_dates = list(sorted_dates)
    final_flows = [-dates_flows[d] for d in final_dates]  # <<< SIGN FLIP

    # --- Ensure final_account_market_value is treated as float ---
    final_market_value_target = (
        float(final_account_market_value)
        if pd.notna(final_account_market_value)
        else 0.0
    )
    final_market_value_abs = abs(final_market_value_target)

    if final_market_value_abs > 1e-9 and isinstance(end_date, date):
        # If no initial flows, add the first tx date (or end date) and the final value
        if not final_dates:
            first_tx_date_for_account = (
                acc_tx_copy["Date"].min().date() if not acc_tx_copy.empty else end_date
            )
            if end_date >= first_tx_date_for_account:
                # Need a starting point for the MWR calc - usually a deposit or buy
                # If only a final value exists, MWR is undefined. Let's return empty.
                # Or perhaps add a zero flow at the start? Let's return empty for now.
                return [], []  # Cannot calculate MWR with only a final value
            else:
                return [], []
        elif end_date >= final_dates[-1]:
            if final_dates[-1] == end_date:
                final_flows[-1] += final_market_value_abs
            else:
                final_dates.append(end_date)
                final_flows.append(final_market_value_abs)
        # If end_date is before the last cash flow, final value isn't added here.

    if len(final_dates) < 2:
        return [], []
    non_zero_final_flows = [cf for cf in final_flows if abs(cf) > 1e-9]
    if (
        not non_zero_final_flows
        or all(cf >= -1e-9 for cf in non_zero_final_flows)
        or all(cf <= 1e-9 for cf in non_zero_final_flows)
    ):
        return [], []
    return final_dates, final_flows


# --- END OF REVISED get_cash_flows_for_mwr ---


# --- Current Price Fetching etc. ---
def fetch_index_quotes_yfinance(
    query_symbols: List[str] = DEFAULT_INDEX_QUERY_SYMBOLS,
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Fetches near real-time quotes for specified INDEX symbols using yfinance.

    Maps internal index symbols (e.g., '.DJI') to their Yahoo Finance tickers (e.g., '^DJI'),
    fetches data using `yfinance.Tickers`, and extracts key quote information like price,
    change, percentage change, and name.

    Args:
        query_symbols (List[str], optional): A list of internal index symbols to query.
                                             Defaults to DEFAULT_INDEX_QUERY_SYMBOLS.

    Returns:
        Dict[str, Optional[Dict[str, Any]]]: A dictionary where keys are the original
            `query_symbols`. Values are dictionaries containing quote data
            ('price', 'change', 'changesPercentage' (as fraction), 'name', 'symbol', 'yf_ticker')
            if successful, or None if the symbol mapping failed or data fetch/parsing failed
            for that symbol.
    """
    if not query_symbols:
        return {}
    yf_tickers_to_fetch = []
    yf_ticker_to_query_symbol_map = {}
    for q_sym in query_symbols:
        yf_ticker = YFINANCE_INDEX_TICKER_MAP.get(q_sym)
        if yf_ticker:
            if yf_ticker not in yf_ticker_to_query_symbol_map:
                yf_tickers_to_fetch.append(yf_ticker)
            yf_ticker_to_query_symbol_map[yf_ticker] = q_sym
        else:
            logging.warning(f"Warn: No yfinance ticker mapping for index: {q_sym}")
    if not yf_tickers_to_fetch:
        return {q_sym: None for q_sym in query_symbols}
    results: Dict[str, Optional[Dict[str, Any]]] = {}
    logging.info(
        f"Fetching index quotes via yfinance for tickers: {', '.join(yf_tickers_to_fetch)}"
    )  # Reduced verbosity
    try:
        tickers_data = yf.Tickers(" ".join(yf_tickers_to_fetch))
        for yf_ticker, ticker_obj in tickers_data.tickers.items():
            original_query_symbol = yf_ticker_to_query_symbol_map.get(yf_ticker)
            if not original_query_symbol:
                continue
            ticker_info = getattr(ticker_obj, "info", None)
            if ticker_info:
                try:
                    price_raw = ticker_info.get(
                        "regularMarketPrice", ticker_info.get("currentPrice")
                    )
                    prev_close_raw = ticker_info.get("previousClose")
                    name = ticker_info.get(
                        "shortName", ticker_info.get("longName", original_query_symbol)
                    )
                    change_val_raw = ticker_info.get("regularMarketChange")
                    change_pct_val_raw = ticker_info.get(
                        "regularMarketChangePercent"
                    )  # Raw value from yfinance

                    price = float(price_raw) if price_raw is not None else np.nan
                    prev_close = (
                        float(prev_close_raw) if prev_close_raw is not None else np.nan
                    )
                    change = (
                        float(change_val_raw) if change_val_raw is not None else np.nan
                    )

                    # --- FIX: Store raw percentage value, handle fallback without * 100 ---
                    # Get the percentage directly if available
                    changesPercentage = (
                        float(change_pct_val_raw)
                        if change_pct_val_raw is not None
                        else np.nan
                    )

                    # Calculate change if missing
                    if pd.isna(change) and pd.notna(price) and pd.notna(prev_close):
                        change = price - prev_close

                    # Fallback for percentage ONLY if primary fetch failed AND change/prev_close are valid
                    if (
                        pd.isna(changesPercentage)
                        and pd.notna(change)
                        and pd.notna(prev_close)
                        and prev_close != 0
                    ):
                        # changesPercentage = (change / prev_close) * 100 # Scale to % <-- REMOVED * 100
                        changesPercentage = (
                            change / prev_close
                        )  # Store as fraction/decimal
                    # --- END FIX ---

                    if pd.notna(price):
                        results[original_query_symbol] = {
                            "price": price,
                            "change": change if pd.notna(change) else np.nan,
                            "changesPercentage": (
                                changesPercentage
                                if pd.notna(changesPercentage)
                                else np.nan
                            ),  # Use the potentially calculated value
                            "name": name,
                            "symbol": original_query_symbol,
                            "yf_ticker": yf_ticker,
                        }
                    else:
                        results[original_query_symbol] = None
                        logging.warning(
                            f"Warn: Index {original_query_symbol} ({yf_ticker}) missing price."
                        )  # Reduced verbosity
                except (ValueError, TypeError, KeyError, AttributeError) as e:
                    results[original_query_symbol] = None
                    logging.warning(
                        f"Warn: Error parsing index {original_query_symbol} ({yf_ticker}): {e}"
                    )
            else:
                results[original_query_symbol] = None
                logging.warning(
                    f"Warn: No yfinance info for index: {yf_ticker}"
                )  # Reduced verbosity

    except Exception as e:
        logging.error(f"Error fetching yfinance index data: {e}")
        traceback.print_exc()
        results = {q_sym: None for q_sym in query_symbols}
    for q_sym in query_symbols:
        if q_sym not in results:
            results[q_sym] = None
    return results


# --- Current Price Fetching etc. ---
# fetch_index_quotes_yfinance: Error handling seems okay, using specific logging.
# get_cached_or_fetch_yfinance_data: Needs refinement.
def get_cached_or_fetch_yfinance_data(
    internal_stock_symbols: List[str],
    required_currencies: Set[
        str
    ],  # Currencies needed (incl. display_currency, default_currency, local currencies)
    cache_file: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    cache_duration_hours: int = YFINANCE_CACHE_DURATION_HOURS,
) -> Tuple[
    Optional[Dict[str, Dict[str, Optional[float]]]],
    Optional[Dict[str, float]],
    bool,
    bool,
]:  # Added has_errors, has_warnings
    """
    Gets current stock/ETF price/change data and FX rates relative to USD using yfinance, leveraging a cache.

    Checks a JSON cache file for recent stock and FX data. If the cache is missing, outdated,
    or incomplete for the requested symbols and currencies, it fetches the required data
    from Yahoo Finance using `yfinance.Tickers`. Fetched data includes stock prices, changes,
    and FX rates against USD (e.g., THB per USD from THB=X). Updates the cache file upon fetching.

    Args:
        internal_stock_symbols (List[str]): List of internal stock/ETF symbols to fetch data for.
        required_currencies (Set[str]): Set of currency codes (e.g., 'USD', 'EUR', 'THB') for which
                                        FX rates against USD are needed.
        cache_file (str, optional): Path to the JSON cache file. Defaults to DEFAULT_CURRENT_CACHE_FILE_PATH.
        cache_duration_hours (int, optional): Maximum age of the cache in hours before it's considered outdated.
                                              Defaults to YFINANCE_CACHE_DURATION_HOURS.

    Returns:
        Tuple containing:
        - Optional[Dict[str, Dict[str, Optional[float]]]]: Stock Data Dict. Keys are internal symbols.
            Values are dicts with 'price', 'change', 'changesPercentage' (as percentage), 'previousClose'.
            Values within the inner dict can be float or np.nan. Returns None if a critical error occurs.
        - Optional[Dict[str, float]]: FX Rates vs USD Dict. Keys are currency codes (str). Values are
            rates representing units of the currency per 1 USD (float or np.nan). Includes 'USD': 1.0.
            Returns None if a critical error occurs.
        - bool: has_errors - True if critical fetch/load errors occurred (e.g., network error,
                             cache load failure preventing data retrieval); False otherwise.
        - bool: has_warnings - True if recoverable issues occurred (e.g., cache incomplete,
                                data missing for *some* symbols/currencies, timeout error); False otherwise.
    """
    stock_data_internal: Optional[Dict[str, Dict[str, Optional[float]]]] = None
    fx_rates_vs_usd: Optional[Dict[str, float]] = {
        "USD": 1.0
    }  # Initialize with USD base
    cache_needs_update = True
    cached_data = {}
    now = datetime.now()
    has_errors = False
    has_warnings = False

    # --- Cache Loading Logic ---
    if not cache_file:
        logging.warning(
            f"Warning: Invalid cache file path provided ('{cache_file}'). Cache read skipped."
        )
        has_warnings = True
        cache_needs_update = True
    elif os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            cache_timestamp_str = cached_data.get("timestamp")
            cached_stocks = cached_data.get("stock_data_internal")
            cached_fx_vs_usd = cached_data.get("fx_rates_vs_usd")

            if (
                cache_timestamp_str
                and isinstance(cached_stocks, dict)
                and isinstance(cached_fx_vs_usd, dict)
            ):
                cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                cache_age = now - cache_timestamp
                if cache_age <= timedelta(hours=cache_duration_hours):
                    all_stocks_present = all(
                        s in cached_stocks
                        for s in internal_stock_symbols
                        if s not in YFINANCE_EXCLUDED_SYMBOLS and s != CASH_SYMBOL_CSV
                    )
                    stock_format_ok = (
                        all(
                            isinstance(v, dict) and "price" in v
                            for v in cached_stocks.values()
                        )
                        if cached_stocks
                        else True
                    )
                    all_fx_present = required_currencies.issubset(
                        cached_fx_vs_usd.keys()
                    )

                    if all_stocks_present and stock_format_ok and all_fx_present:
                        logging.info(
                            f"Yahoo Finance cache is valid (age: {cache_age}). Using cached data."
                        )  # Info level
                        stock_data_internal = cached_stocks
                        fx_rates_vs_usd = cached_fx_vs_usd
                        cache_needs_update = False
                    else:
                        # ... (existing incomplete cache logic) ...
                        logging.warning(
                            f"Yahoo Finance cache is recent but incomplete/invalid. Will fetch/update."
                        )
                        has_warnings = True
                        stock_data_internal = cached_stocks if stock_format_ok else {}
                        fx_rates_vs_usd = (
                            cached_fx_vs_usd
                            if isinstance(cached_fx_vs_usd, dict)
                            else {"USD": 1.0}
                        )
                        cache_needs_update = True
                else:
                    logging.info(
                        f"Yahoo Finance cache is outdated (age: {cache_age}). Will fetch fresh data."
                    )  # Info level
                    cache_needs_update = True
            else:
                logging.warning(
                    "Yahoo Finance cache is invalid or missing data. Will fetch fresh data."
                )
                has_warnings = True
                cache_needs_update = True
        # --- Refined Cache Loading Exceptions ---
        except json.JSONDecodeError as e:
            logging.warning(
                f"Error decoding JSON from cache file {cache_file}: {e}. Will fetch fresh data."
            )
            has_warnings = True
            cache_needs_update = True
            stock_data_internal = {}
            fx_rates_vs_usd = {"USD": 1.0}
        except (
            FileNotFoundError
        ):  # Should be caught by os.path.exists, but belt-and-suspenders
            logging.error(
                f"Cache file {cache_file} disappeared unexpectedly?. Will fetch fresh data."
            )
            has_errors = True
            cache_needs_update = True
            stock_data_internal = {}
            fx_rates_vs_usd = {"USD": 1.0}
        except Exception as e:
            logging.warning(
                f"Error reading Yahoo Finance cache file {cache_file}: {e}. Will fetch fresh data."
            )
            has_warnings = True
            cache_needs_update = True
            stock_data_internal = {}
            fx_rates_vs_usd = {"USD": 1.0}
        # --- End Refined Cache Loading Exceptions ---
    else:
        logging.info(
            f"Yahoo Finance cache file {cache_file} not found. Will fetch fresh data."
        )  # Info level
        cache_needs_update = True

    # --- Data Fetching Logic ---
    if cache_needs_update:
        logging.info(
            "Fetching/Updating data from Yahoo Finance (Stocks & FX vs USD)..."
        )
        fetched_stocks = stock_data_internal if stock_data_internal is not None else {}
        # fx_rates_vs_usd is already initialized

        # Determine symbols/tickers to fetch
        yf_tickers_to_fetch = set()
        internal_to_yf_map = {}
        yf_to_internal_map = {}
        missing_stock_symbols = []
        explicitly_excluded_count = 0
        # Stocks/ETFs
        for internal_sym in internal_stock_symbols:
            # ... (symbol mapping logic remains the same) ...
            if internal_sym == CASH_SYMBOL_CSV:
                continue
            if internal_sym in YFINANCE_EXCLUDED_SYMBOLS:  # Skip fetch if excluded
                if internal_sym not in fetched_stocks:
                    explicitly_excluded_count += 1
                    fetched_stocks[internal_sym] = {
                        "price": np.nan,
                        "change": np.nan,
                        "changesPercentage": np.nan,
                        "previousClose": np.nan,
                    }
                continue
            yf_ticker = SYMBOL_MAP_TO_YFINANCE.get(internal_sym, internal_sym.upper())
            if yf_ticker and " " not in yf_ticker and ":" not in yf_ticker:
                should_fetch_stock = True
                if internal_sym in fetched_stocks:
                    cached_entry = fetched_stocks[internal_sym]
                    if (
                        isinstance(cached_entry, dict)
                        and pd.notna(cached_entry.get("price"))
                        and cached_entry.get("price") > 1e-9
                    ):
                        should_fetch_stock = False
                if should_fetch_stock:
                    yf_tickers_to_fetch.add(yf_ticker)
                    internal_to_yf_map[internal_sym] = yf_ticker
                    yf_to_internal_map[yf_ticker] = internal_sym
            else:  # Invalid mapping/format
                missing_stock_symbols.append(internal_sym)
                if internal_sym not in fetched_stocks:
                    logging.warning(
                        f"Warning: No valid Yahoo Finance ticker mapping for: {internal_sym}."
                    )
                    has_warnings = True
                    fetched_stocks[internal_sym] = {
                        "price": np.nan,
                        "change": np.nan,
                        "changesPercentage": np.nan,
                        "previousClose": np.nan,
                    }
        if explicitly_excluded_count > 0:
            logging.info(
                f"Info: Explicitly excluded {explicitly_excluded_count} stock symbols."
            )

        # FX Rates vs USD
        fx_tickers_to_fetch = set()
        currencies_to_fetch_vs_usd = required_currencies - set(fx_rates_vs_usd.keys())
        for currency in currencies_to_fetch_vs_usd:
            if currency != "USD" and currency and isinstance(currency, str):
                fx_ticker = f"{currency}=X"
                fx_tickers_to_fetch.add(fx_ticker)
                yf_tickers_to_fetch.add(fx_ticker)

        # Fetching using yfinance
        if not yf_tickers_to_fetch:
            logging.info("No new Stock/FX tickers need fetching.")
        else:
            logging.info(
                f"Fetching/Updating {len(yf_tickers_to_fetch)} tickers from Yahoo: {list(yf_tickers_to_fetch)}"
            )
            fetch_success = True
            all_fetched_data = {}
            try:
                yf_ticker_list = list(yf_tickers_to_fetch)
                fetch_batch_size = 50
                for i in range(0, len(yf_ticker_list), fetch_batch_size):
                    batch = yf_ticker_list[i : i + fetch_batch_size]
                    logging.info(
                        f"  Fetching batch {i//fetch_batch_size + 1} ({len(batch)} tickers)..."
                    )
                    try:
                        tickers_data = yf.Tickers(" ".join(batch))
                        for yf_ticker, ticker_obj in tickers_data.tickers.items():
                            # ... (processing ticker_info remains the same) ...
                            ticker_info = getattr(ticker_obj, "info", None)
                            price, change, pct_change, prev_close = (
                                np.nan,
                                np.nan,
                                np.nan,
                                np.nan,
                            )
                            if ticker_info:
                                price_raw = ticker_info.get(
                                    "regularMarketPrice",
                                    ticker_info.get("currentPrice"),
                                )
                                change_raw = ticker_info.get("regularMarketChange")
                                pct_change_raw = ticker_info.get(
                                    "regularMarketChangePercent"
                                )
                                prev_close_raw = ticker_info.get("previousClose")
                                try:
                                    price = (
                                        float(price_raw)
                                        if price_raw is not None
                                        else np.nan
                                    )
                                except (ValueError, TypeError):
                                    price = np.nan
                                try:
                                    change = (
                                        float(change_raw)
                                        if change_raw is not None
                                        else np.nan
                                    )
                                except (ValueError, TypeError):
                                    change = np.nan
                                try:
                                    pct_change = (
                                        float(pct_change_raw)
                                        if pct_change_raw is not None
                                        else np.nan
                                    )  # Fractional
                                except (ValueError, TypeError):
                                    pct_change = np.nan
                                try:
                                    prev_close = (
                                        float(prev_close_raw)
                                        if prev_close_raw is not None
                                        else np.nan
                                    )
                                except (ValueError, TypeError):
                                    prev_close = np.nan
                                if not (pd.notna(price) and price > 1e-9):
                                    price = np.nan
                                if pd.notna(pct_change):
                                    pct_change *= 100.0  # Convert fraction to percent
                            all_fetched_data[yf_ticker] = {
                                "price": price,
                                "change": change,
                                "changesPercentage": pct_change,
                                "previousClose": prev_close,
                            }
                    # --- Refined Fetching Exceptions ---
                    except requests.exceptions.ConnectionError as conn_err:
                        logging.error(f"  NETWORK ERROR fetching batch: {conn_err}")
                        fetch_success = False
                        has_errors = True
                        break  # Stop batch loop on connection error
                    except requests.exceptions.Timeout as timeout_err:
                        logging.warning(
                            f"  TIMEOUT ERROR fetching batch: {timeout_err}"
                        )
                        fetch_success = False
                        has_warnings = True
                        # Continue if possible? Or break? Let's flag warning and continue for now.
                    except requests.exceptions.HTTPError as http_err:
                        logging.error(f"  HTTP ERROR fetching batch: {http_err}")
                        fetch_success = False
                        has_errors = True
                        # Treat HTTP error as serious
                    except Exception as yf_err:
                        logging.error(f"  YFINANCE ERROR fetching batch info: {yf_err}")
                        fetch_success = False
                        has_warnings = (
                            True  # Treat internal yf error as warning for now
                        )
                    # --- End Refined Fetching Exceptions ---
                    time.sleep(0.1)
            except Exception as e:  # Catch-all for the outer loop
                logging.exception(f"ERROR during Yahoo Finance fetch loop")
                fetch_success = False
                has_errors = True

            # Process fetched data
            # Stocks
            for yf_ticker, data_dict in all_fetched_data.items():
                internal_sym = yf_to_internal_map.get(yf_ticker)
                if internal_sym:
                    fetched_stocks[internal_sym] = data_dict
            # Ensure entries for all requested symbols
            for sym in internal_stock_symbols:
                if (
                    sym != CASH_SYMBOL_CSV
                    and sym not in YFINANCE_EXCLUDED_SYMBOLS
                    and sym not in fetched_stocks
                ):
                    if sym not in missing_stock_symbols:
                        logging.warning(
                            f"Warning: Data for {sym} still not found after fetch."
                        )
                        has_warnings = True
                    fetched_stocks[sym] = {
                        "price": np.nan,
                        "change": np.nan,
                        "changesPercentage": np.nan,
                        "previousClose": np.nan,
                    }
            # FX Rates vs USD
            for currency in currencies_to_fetch_vs_usd:
                if currency == "USD":
                    continue
                fx_ticker = f"{currency}=X"
                fx_data_dict = all_fetched_data.get(fx_ticker)
                price = (
                    fx_data_dict.get("price")
                    if isinstance(fx_data_dict, dict)
                    else np.nan
                )
                if pd.notna(price):
                    fx_rates_vs_usd[currency] = price
                else:
                    logging.warning(
                        f"Warning: Failed to fetch/update FX rate for {fx_ticker}."
                    )
                    has_warnings = True
                    if currency not in fx_rates_vs_usd:
                        fx_rates_vs_usd[currency] = np.nan

        # Assign final results
        stock_data_internal = fetched_stocks

        # Save Cache
        if cache_needs_update:
            if not cache_file:
                logging.error(
                    f"ERROR: Cache file path is invalid ('{cache_file}'). Cannot save cache."
                )
            else:
                logging.info(
                    f"Saving updated Yahoo Finance data to cache: {cache_file}"
                )
                if "USD" not in fx_rates_vs_usd:
                    fx_rates_vs_usd["USD"] = 1.0
                content = {
                    "timestamp": now.isoformat(),
                    "stock_data_internal": stock_data_internal,
                    "fx_rates_vs_usd": fx_rates_vs_usd,
                }
                try:
                    cache_dir = os.path.dirname(cache_file)
                    if cache_dir and not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    with open(cache_file, "w") as f:

                        class NpEncoder(json.JSONEncoder):  # Keep NaN encoder
                            def default(self, obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                if isinstance(obj, np.floating):
                                    return float(obj) if np.isfinite(obj) else None
                                if isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                return super(NpEncoder, self).default(obj)

                        json.dump(content, f, indent=4, cls=NpEncoder)
                # --- Refined Cache Saving Exceptions ---
                except TypeError as e:
                    logging.error(f"TypeError writing cache ('{cache_file}'): {e}")
                except IOError as e:
                    logging.error(f"IOError writing cache ('{cache_file}'): {e}")
                except Exception as e:
                    logging.exception(
                        f"Unexpected error writing cache ('{cache_file}')"
                    )
                # --- End Refined Cache Saving Exceptions ---

    # Final Checks
    if not isinstance(stock_data_internal, dict):
        stock_data_internal = None
        has_errors = True
    if not isinstance(fx_rates_vs_usd, dict):
        fx_rates_vs_usd = None
        has_errors = True
    if isinstance(fx_rates_vs_usd, dict) and "USD" not in fx_rates_vs_usd:
        fx_rates_vs_usd["USD"] = 1.0
    if stock_data_internal is not None:
        for sym in internal_stock_symbols:
            if (
                sym != CASH_SYMBOL_CSV
                and sym not in YFINANCE_EXCLUDED_SYMBOLS
                and sym not in stock_data_internal
            ):
                stock_data_internal[sym] = {
                    "price": np.nan,
                    "change": np.nan,
                    "changesPercentage": np.nan,
                    "previousClose": np.nan,
                }
                logging.warning(
                    f"Warning: Data for symbol {sym} missing in final check."
                )  # Add warning if missed
                has_warnings = True

    # If fetch failed for critical data, mark as error
    if fx_rates_vs_usd is None or stock_data_internal is None:
        has_errors = True

    return stock_data_internal, fx_rates_vs_usd, has_errors, has_warnings


# --- FINAL Version: Simple lookup assuming consistent dictionary ---
# Use this version of get_conversion_rate
def get_conversion_rate(
    from_curr: str, to_curr: str, fx_rates: Optional[Dict[str, float]]
) -> float:
    """
    Gets CURRENT FX conversion rate (units of to_curr per 1 unit of from_curr).

    Calculates cross rates via USD using the provided `fx_rates` dictionary, which is
    assumed to contain rates relative to USD (units of OTHER_CURRENCY per 1 USD).
    Handles direct conversion, conversion via USD, and returns a fallback rate if
    necessary data is missing or invalid.

    Args:
        from_curr (str): The currency code to convert FROM.
        to_curr (str): The currency code to convert TO.
        fx_rates (Optional[Dict[str, float]]): A dictionary containing current FX rates,
            where keys are currency codes (str) and values are the rate per 1 USD (float).
            Example: {'JPY': 150.0, 'EUR': 0.9, 'USD': 1.0}.

    Returns:
        float: The conversion rate (units of `to_curr` for 1 unit of `from_curr`).
               Returns 1.0 if `from_curr` equals `to_curr`, or if the necessary rates
               are missing or invalid in the `fx_rates` dictionary, or if inputs are invalid.
    """
    # --- ADD VALIDATION ---
    if (
        not from_curr
        or not isinstance(from_curr, str)
        or not to_curr
        or not isinstance(to_curr, str)
    ):
        logging.debug("get_conversion_rate: Invalid from_curr or to_curr input.")
        return 1.0  # Return fallback for invalid input types
    # --- END VALIDATION ---

    if from_curr == to_curr:
        return 1.0
    if not isinstance(fx_rates, dict):
        logging.warning(
            f"Warning: get_conversion_rate received invalid fx_rates type. Returning 1.0"
        )  # Optional
        return 1.0  # Fallback

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()

    # fx_rates now holds {CURRENCY: rate_per_USD} e.g., {'JPY': 143.3, 'THB': 33.5}

    # Get intermediate rates: Currency per 1 USD
    rate_A_per_USD = fx_rates.get(from_curr_upper)  # e.g., THB per USD
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0

    rate_B_per_USD = fx_rates.get(to_curr_upper)  # e.g., JPY per USD
    if to_curr_upper == "USD":
        rate_B_per_USD = 1.0

    rate_B_per_A = np.nan  # Initialize rate for B per A (TO / FROM)

    # Formula: TO / FROM = (TO / USD) / (FROM / USD)
    if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
        if abs(rate_A_per_USD) > 1e-9:  # Check denominator (FROM/USD) is not zero
            try:
                rate_B_per_A = rate_B_per_USD / rate_A_per_USD
                logging.debug(
                    f"DEBUG get_conv_rate: {to_curr}/{from_curr} = {rate_B_per_USD} / {rate_A_per_USD} = {rate_B_per_A}"
                )  # DEBUG
            except (ZeroDivisionError, TypeError):
                pass  # Keep NaN
        # else: Denominator is zero/invalid

    # Final check and fallback
    if pd.isna(rate_B_per_A):
        logging.warning(
            f"Warning: Current FX rate lookup failed for {from_curr}->{to_curr}. Returning 1.0"
        )  # Optional Warning
        return 1.0
    else:
        return float(rate_B_per_A)


# --- REVISED: _process_transactions_to_holdings (Split applied to all accounts) ---
def _process_transactions_to_holdings(
    transactions_df: pd.DataFrame, default_currency: str, shortable_symbols: Set[str]
) -> Tuple[
    Dict[Tuple[str, str], Dict],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Set[int],
    Dict[int, str],
    bool,
]:
    """
    Processes stock/ETF transactions to calculate holdings and aggregate metrics in local currencies.

    Iterates through filtered, cleaned non-cash transactions, updating the state of each
    holding (symbol, account pair). Calculates quantity, total cost basis, realized gains,
    dividends received/paid, commissions paid, cumulative investment, and total buy cost,
    all in the holding's local currency. Stock splits are applied globally to all accounts
    holding the affected stock at the time of the split. Short selling and covering are handled.

    Args:
        transactions_df (pd.DataFrame): DataFrame of cleaned and filtered non-$CASH transactions.
                                        Must include 'Symbol', 'Account', 'Type', 'Quantity',
                                        'Price/Share', 'Total Amount', 'Commission', 'Split Ratio',
                                        'Local Currency', 'Date', 'original_index'.
        default_currency (str): The default currency to use if somehow missing.
        shortable_symbols (Set[str]): A set of symbols that allow short selling.

    Returns:
        Tuple containing:
        - holdings (Dict[Tuple[str, str], Dict]): Dictionary keyed by (symbol, account). Values are
            dicts containing holding details ('qty', 'total_cost_local', 'realized_gain_local',
            'dividends_local', 'commissions_local', 'local_currency', 'short_proceeds_local',
            'short_original_qty', 'total_cost_invested_local', 'cumulative_investment_local',
            'total_buy_cost_local').
        - overall_realized_gains_local (Dict[str, float]): Dictionary keyed by currency code, summing
            realized gains across all holdings in that currency.
        - overall_dividends_local (Dict[str, float]): Dictionary keyed by currency code, summing
            net dividends across all holdings in that currency.
        - overall_commissions_local (Dict[str, float]): Dictionary keyed by currency code, summing
            commissions across all holdings in that currency.
        - ignored_row_indices_local (Set[int]): Set of original indices ignored during this processing
            step due to calculation errors or invalid data for the operation.
        - ignored_reasons_local (Dict[int, str]): Dictionary mapping original index to the reason
            it was ignored during this step.
        - has_warnings (bool): True if recoverable issues occurred during processing (e.g., skipping a row
                               due to bad data); False otherwise. Note: Critical errors preventing
                               processing (like missing columns) are handled by returning empty dicts.
    """
    # --- Add initial log ---
    # logging.debug(f"Worker Start: Processing date {eval_date}") # Can be very verbose
    # --- CONFIGURE LOGGING WITHIN WORKER (for debugging) ---
    # This ensures messages from this worker process are output.
    # Use force=True (Python 3.8+) to override potential root logger setup issues.
    # Note: This simple setup might cause issues if you have complex file logging
    #       in the main process that you expect workers to use directly.
    #       For production, consider using a QueueHandler.
    logging.basicConfig(
        level=LOGGING_LEVEL,  # Match the desired level
        format="%(asctime)s [%(levelname)-8s] PID:%(process)d {%(module)s:%(lineno)d} %(message)s",  # Add PID
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override existing config if any
    )
    # --- END WORKER LOGGING CONFIG ---
    holdings: Dict[Tuple[str, str], Dict] = {}
    overall_realized_gains_local: Dict[str, float] = defaultdict(float)
    overall_dividends_local: Dict[str, float] = defaultdict(float)
    overall_commissions_local: Dict[str, float] = defaultdict(float)
    ignored_row_indices_local = set()
    ignored_reasons_local = {}
    has_warnings = False  # Flag for recoverable issues during processing

    logging.debug(
        "Processing filtered stock/ETF transactions (split logic modified)..."
    )

    required_cols = [
        "Symbol",
        "Account",
        "Type",
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Split Ratio",
        "Local Currency",
        "Date",
        "original_index",
    ]
    missing_cols = [col for col in required_cols if col not in transactions_df.columns]
    if missing_cols:
        logging.error(
            f"CRITICAL ERROR in _process_transactions: Input DataFrame missing required columns: {missing_cols}. Cannot proceed."
        )
        # Return empty dicts and error flag
        return (
            {},
            {},
            {},
            {},
            ignored_row_indices_local,
            ignored_reasons_local,
            True,
        )  # Indicate critical error

    # --- Main Processing Loop ---
    for index, row in transactions_df.iterrows():
        if row["Symbol"] == CASH_SYMBOL_CSV:
            continue

        try:
            # --- Safely get values from row (as before) ---
            original_index = row["original_index"]
            symbol = str(row["Symbol"]).strip()
            account = str(row["Account"]).strip()  # Account listed in this specific row
            tx_type = str(row["Type"]).lower().strip()
            local_currency_from_row = str(
                row["Local Currency"]
            ).strip()  # Currency associated with this row's account
            tx_date = row["Date"].date()
            qty = pd.to_numeric(
                row.get("Quantity"), errors="coerce"
            )  # Use .get for robustness
            price_local = pd.to_numeric(row.get("Price/Share"), errors="coerce")
            total_amount_local = pd.to_numeric(row.get("Total Amount"), errors="coerce")
            commission_val = row.get("Commission")
            commission_local_raw = pd.to_numeric(commission_val, errors="coerce")
            # Commission specific to THIS transaction row
            commission_local_for_this_tx = (
                0.0 if pd.isna(commission_local_raw) else float(commission_local_raw)
            )
            split_ratio = pd.to_numeric(row.get("Split Ratio"), errors="coerce")

            if (
                not symbol
                or not account
                or not tx_type
                or not local_currency_from_row
                or pd.isna(tx_date)
            ):
                raise ValueError(
                    "Essential row data (Symbol, Account, Type, Currency, Date) is missing or invalid."
                )

        except (KeyError, ValueError, AttributeError, TypeError) as e:
            error_msg = f"Row Read Error ({type(e).__name__}): {e}"
            row_repr = row.to_string().replace("\n", " ")[:150]
            logging.warning(
                f"WARN in _process_transactions pre-check row {index} (orig: {row.get('original_index', 'N/A')}): {error_msg}. Data: {row_repr}... Skipping row."
            )
            ignored_reasons_local[row.get("original_index", index)] = error_msg
            ignored_row_indices_local.add(row.get("original_index", index))
            has_warnings = True
            continue

        holding_key_from_row = (symbol, account)  # Key based on the account in THIS row
        log_this_row = account == "E*TRADE"  # Flag for general E*TRADE logging

        # --- Initialize holding for the account IN THIS ROW if first time ---
        # This ensures the account exists if it's only mentioned in a split/fee row
        if holding_key_from_row not in holdings:
            holdings[holding_key_from_row] = {
                "qty": 0.0,
                "total_cost_local": 0.0,
                "realized_gain_local": 0.0,
                "dividends_local": 0.0,
                "commissions_local": 0.0,
                "local_currency": local_currency_from_row,
                "short_proceeds_local": 0.0,
                "short_original_qty": 0.0,
                "total_cost_invested_local": 0.0,
                "cumulative_investment_local": 0.0,
                "total_buy_cost_local": 0.0,
            }
        # Check for currency consistency only if the holding was already initialized
        elif (
            holdings[holding_key_from_row]["local_currency"] != local_currency_from_row
        ):
            # This case should be rare if _load_and_clean adds currency correctly based on map
            msg = f"Currency mismatch for {symbol}/{account}"
            logging.warning(
                f"CRITICAL WARN in _process_transactions: {msg} row {original_index}. Holding exists with diff ccy. Skip."
            )
            ignored_reasons_local[original_index] = msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            continue

        # --- MODIFIED SPLIT HANDLING ---
        if tx_type in ["split", "stock split"]:
            split_ratio_raw = row.get("Split Ratio")
            logging.debug(
                f"--- SPLIT ROW DEBUG (Row Index: {index}, Orig: {original_index}) ---"
            )
            logging.debug(f"  Symbol: {symbol}, Account: {account}, Date: {tx_date}")
            logging.debug(
                f"  Raw 'Split Ratio' value from row: '{split_ratio_raw}' (Type: {type(split_ratio_raw)})"
            )
            # --- Now process the split --- sell', 'buy to cover']:
            try:
                if pd.isna(split_ratio) or split_ratio <= 0:
                    raise ValueError(f"Invalid split ratio: {split_ratio}")

                logging.debug(
                    f"Processing SPLIT for {symbol} on {tx_date} (Ratio: {split_ratio}). Applying to all accounts holding it."
                )

                logging.debug("  Holdings state BEFORE applying split:")
                for h_key, h_data in holdings.items():
                    if h_key[0] == symbol:  # Log only AAPL holdings
                        logging.debug(
                            f"    {h_key}: Qty={h_data.get('qty', 'N/A')}, ShortQty={h_data.get('short_original_qty', 'N/A')}"
                        )

                # Iterate through all existing holdings to apply the split quantity adjustment
                affected_accounts = []
                for h_key, h_data in holdings.items():
                    h_symbol, h_account = h_key
                    if h_symbol == symbol:  # Apply split if symbol matches
                        affected_accounts.append(h_account)
                        old_qty = h_data["qty"]
                        if abs(old_qty) >= 1e-9:  # Only adjust if holding exists
                            h_data["qty"] *= split_ratio
                            logging.debug(
                                f"  Applied split to {h_symbol}/{h_account}: Qty {old_qty:.4f} -> {h_data['qty']:.4f}"
                            )
                            # Adjust short original qty IF shorting
                            if old_qty < -1e-9 and symbol in shortable_symbols:
                                h_data["short_original_qty"] *= split_ratio
                                if abs(h_data["short_original_qty"]) < 1e-9:
                                    h_data["short_original_qty"] = 0.0
                                logging.debug(
                                    f"  Adjusted short original qty for {h_symbol}/{h_account}"
                                )

                            # Clean up near-zero quantities
                            if abs(h_data["qty"]) < 1e-9:
                                h_data["qty"] = 0.0
                        else:
                            logging.debug(
                                f"  Skipped split qty adjust for {h_symbol}/{h_account} (Qty near zero: {old_qty:.4f})"
                            )

                logging.debug(
                    f"Split for {symbol} applied to accounts: {affected_accounts}"
                )
                logging.debug("  Holdings state AFTER applying split:")
                for h_key, h_data in holdings.items():
                    if h_key[0] == symbol:
                        logging.debug(f"    {h_key}: Qty={h_data.get('qty', 'N/A')}")

                # --- Apply Commission/Fee ONLY to the account specified in the split row ---
                if commission_local_for_this_tx != 0:
                    holding_for_fee = holdings.get(
                        holding_key_from_row
                    )  # Get holding for account in this row
                    if holding_for_fee:
                        fee_cost = abs(commission_local_for_this_tx)
                        holding_for_fee["commissions_local"] += fee_cost
                        holding_for_fee[
                            "total_cost_invested_local"
                        ] += fee_cost  # Add split fee to invested cost
                        holding_for_fee[
                            "cumulative_investment_local"
                        ] += fee_cost  # Add split fee to cumulative investment
                        overall_commissions_local[
                            local_currency_from_row
                        ] += fee_cost  # Add to overall for this currency
                        logging.debug(
                            f"  Applied split fee {fee_cost:.2f} to specific account {account}"
                        )
                    else:
                        # This case is unlikely if we pre-initialize, but log if it happens
                        logging.warning(
                            f"  WARN: Could not apply split fee to {holding_key_from_row} - holding not found?"
                        )
                        has_warnings = True

                continue  # Skip the rest of the standard transaction processing for this row

            except (ValueError, TypeError, KeyError) as e_split:
                error_msg = (
                    f"Split Processing Error ({type(e_split).__name__}): {e_split}"
                )
                logging.warning(
                    f"WARN in _process_transactions SPLIT row {original_index} ({symbol}): {error_msg}. Skipping row."
                )
                ignored_reasons_local[original_index] = error_msg
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                continue  # Skip to next row
            except Exception as e_split_unexp:
                logging.exception(
                    f"Unexpected error processing SPLIT row {original_index} ({symbol})"
                )
                ignored_reasons_local[original_index] = (
                    "Unexpected Split Processing Error"
                )
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                continue  # Skip to next row
        # --- END MODIFIED SPLIT HANDLING ---

        # --- Standard Processing for Buy/Sell/Dividend/Fee/Short ---
        # (Get holding for the specific account in the row)
        holding = holdings.get(holding_key_from_row)
        # This should always exist now due to pre-initialization, but check just in case
        if not holding:
            logging.error(
                f"CRITICAL LOGIC ERROR: Holding not found for {holding_key_from_row} after initialization. Skipping row {original_index}."
            )
            ignored_reasons_local[original_index] = (
                "Internal Logic Error: Holding not found"
            )
            ignored_row_indices_local.add(original_index)
            has_errors = True
            continue

        commission_for_overall = (
            commission_local_for_this_tx  # Base commission to add to overall total
        )

        try:
            prev_cum_inv = (
                holding.get("cumulative_investment_local", 0.0) if log_this_row else 0
            )
            # --- Validate Numeric Inputs Specific to Transaction Type ---
            if tx_type in [
                "buy",
                "sell",
                "deposit",
                "withdrawal",
                "short sell",
                "buy to cover",
            ]:
                if pd.isna(qty):
                    raise ValueError(f"Missing Quantity for {tx_type}")
                if pd.isna(price_local) and symbol != CASH_SYMBOL_CSV:
                    raise ValueError(f"Missing Price/Share for {tx_type} {symbol}")
            elif tx_type == "dividend":
                if pd.isna(total_amount_local) and pd.isna(price_local):
                    raise ValueError(
                        "Missing both Total Amount and Price/Share for dividend"
                    )
            elif tx_type == "fees":
                if pd.isna(commission_local_raw):
                    raise ValueError("Missing Commission for fees transaction")
            # Split validation is now handled above

            # --- Shorting Logic (Unchanged from previous version) ---
            if symbol in shortable_symbols and tx_type in [
                "short sell",
                "buy to cover",
            ]:
                qty_abs = abs(qty)
                if qty_abs <= 1e-9:
                    raise ValueError(f"{tx_type} qty must be > 0")
                if tx_type == "short sell":
                    proceeds = (qty_abs * price_local) - commission_local_for_this_tx
                    holding["qty"] -= qty_abs
                    holding["short_proceeds_local"] += proceeds
                    holding["short_original_qty"] += qty_abs
                    holding["commissions_local"] += commission_local_for_this_tx
                    holding[
                        "cumulative_investment_local"
                    ] -= proceeds  # Cash IN from short sell decreases investment
                elif tx_type == "buy to cover":
                    qty_currently_short = (
                        abs(holding["qty"]) if holding["qty"] < -1e-9 else 0.0
                    )
                    if qty_currently_short < 1e-9:
                        raise ValueError(
                            f"Not currently short {symbol}/{account} to cover."
                        )
                    qty_covered = min(qty_abs, qty_currently_short)
                    cost = (
                        qty_covered * price_local
                    ) + commission_local_for_this_tx  # Cash OUT to buy cover increases investment
                    if holding["short_original_qty"] <= 1e-9:
                        raise ZeroDivisionError(
                            f"Short original qty is zero/neg for {symbol}/{account}"
                        )
                    avg_proceeds_per_share = (
                        holding["short_proceeds_local"] / holding["short_original_qty"]
                    )
                    proceeds_attributed = qty_covered * avg_proceeds_per_share
                    gain = proceeds_attributed - cost
                    holding["qty"] += qty_covered
                    holding["short_proceeds_local"] -= proceeds_attributed
                    holding["short_original_qty"] -= qty_covered
                    holding["commissions_local"] += commission_local_for_this_tx
                    holding["realized_gain_local"] += gain
                    overall_realized_gains_local[holding["local_currency"]] += gain
                    # Use currency from holding data
                    if abs(holding["short_original_qty"]) < 1e-9:
                        holding["short_proceeds_local"] = 0.0
                        holding["short_original_qty"] = 0.0
                    if abs(holding["qty"]) < 1e-9:
                        holding["qty"] = 0.0
                    holding["cumulative_investment_local"] += cost
                # Add commission to overall total for shorting actions
                if commission_for_overall != 0:
                    overall_commissions_local[holding["local_currency"]] += abs(
                        commission_for_overall
                    )
                continue  # Skip standard processing below

            # --- Standard Buy/Sell/Dividend/Fee (Split handled above) ---
            if tx_type == "buy" or tx_type == "deposit":
                qty_abs = abs(qty)
                if qty_abs <= 1e-9:
                    raise ValueError("Buy/Deposit qty must be > 0")
                cost = (qty_abs * price_local) + commission_local_for_this_tx
                holding["qty"] += qty_abs
                holding["total_cost_local"] += cost
                holding["commissions_local"] += commission_local_for_this_tx
                holding["total_cost_invested_local"] += cost
                holding["cumulative_investment_local"] += cost
                holding["total_buy_cost_local"] += cost

            elif tx_type == "sell" or tx_type == "withdrawal":
                qty_abs = abs(qty)
                held_qty = holding["qty"]
                if held_qty <= 1e-9:
                    msg = f"Sell attempt {symbol}/{account} w/ non-positive long qty ({held_qty:.4f})"
                    logging.warning(
                        f"Warn in _process_transactions: {msg} row {original_index}. Skip."
                    )
                    ignored_reasons_local[original_index] = msg
                    ignored_row_indices_local.add(original_index)
                    has_warnings = True
                    commission_for_overall = 0.0
                    continue
                if qty_abs <= 1e-9:
                    raise ValueError("Sell/Withdrawal qty must be > 0")
                qty_sold = min(qty_abs, held_qty)
                cost_sold = 0.0
                if held_qty > 1e-9 and abs(holding["total_cost_local"]) > 1e-9:
                    if pd.isna(holding["total_cost_local"]):
                        cost_sold = 0.0
                        has_warnings = True
                        logging.warning(
                            f"Warning: total_cost_local is NaN for {symbol}/{account} before selling."
                        )
                    else:
                        cost_sold = qty_sold * (holding["total_cost_local"] / held_qty)
                proceeds = (qty_sold * price_local) - commission_local_for_this_tx
                gain = proceeds - cost_sold
                holding["qty"] -= qty_sold
                holding["total_cost_local"] -= cost_sold
                holding["commissions_local"] += commission_local_for_this_tx
                holding["realized_gain_local"] += gain
                overall_realized_gains_local[holding["local_currency"]] += gain
                holding[
                    "total_cost_invested_local"
                ] -= cost_sold  # Cost basis tracking adjusted on sell
                if abs(holding["qty"]) < 1e-9:
                    holding["qty"] = 0.0
                    holding["total_cost_local"] = 0.0
                holding["cumulative_investment_local"] -= proceeds

            elif tx_type == "dividend":
                div_amt_local = 0.0
                qty_abs = abs(qty) if pd.notna(qty) else 0
                if pd.notna(total_amount_local) and abs(total_amount_local) > 1e-9:
                    div_amt_local = total_amount_local
                elif pd.notna(price_local) and abs(price_local) > 1e-9:
                    div_amt_local = (
                        (qty_abs * price_local) if qty_abs > 0 else price_local
                    )
                else:
                    div_amt_local = 0.0
                # Apply dividend based on long/short status (short pays dividend)
                div_effect = (
                    abs(div_amt_local)
                    if (
                        holding.get("qty", 0.0) >= -1e-9
                        or symbol not in shortable_symbols
                    )
                    else -abs(div_amt_local)
                )
                holding["dividends_local"] += div_effect
                overall_dividends_local[holding["local_currency"]] += div_effect
                holding[
                    "commissions_local"
                ] += (
                    commission_local_for_this_tx  # Add any fee associated with dividend
                )
                # DIVIDEND DOES NOT AFFECT CUMULATIVE INVESTMENT / TOTAL BUY COST

            elif tx_type == "fees":
                fee_cost = abs(commission_local_for_this_tx)
                holding["commissions_local"] += fee_cost
                holding["total_cost_invested_local"] += fee_cost
                # Fees increase invested cost
                holding["cumulative_investment_local"] += fee_cost
                # Fees increase cumulative investment

            else:  # Should not be reachable if split is handled above
                msg = f"Unhandled stock tx type '{tx_type}'"
                logging.warning(
                    f"Warn in _process_transactions: {msg} row {original_index}. Skip."
                )
                ignored_reasons_local[original_index] = msg
                ignored_row_indices_local.add(original_index)
                has_warnings = True
                commission_for_overall = 0.0
                continue

            # --- Log AFTER modification if it's E*TRADE (Unchanged) ---
            if log_this_row:
                current_cum_inv = holding.get("cumulative_investment_local", 0.0)
                cost = 0.0
                proceeds = 0.0
                div_effect = 0.0
                fee_cost = 0.0
                if tx_type in ["buy", "deposit"]:
                    cost = (
                        (abs(qty) * price_local) + commission_local_for_this_tx
                        if pd.notna(qty) and pd.notna(price_local)
                        else 0
                    )
                if tx_type in ["sell", "withdrawal"]:
                    proceeds = (
                        (abs(qty) * price_local) - commission_local_for_this_tx
                        if pd.notna(qty) and pd.notna(price_local)
                        else 0
                    )
                if tx_type == "dividend":
                    div_amt_local_log = (
                        total_amount_local
                        if pd.notna(total_amount_local)
                        else (
                            abs(qty) * price_local
                            if pd.notna(qty) and pd.notna(price_local)
                            else 0
                        )
                    )
                    div_effect = (
                        abs(div_amt_local_log)
                        if (
                            holding.get("qty", 0) >= -1e-9
                            or symbol not in shortable_symbols
                        )
                        else -abs(div_amt_local_log)
                    )
                if tx_type == "fees":
                    fee_cost = (
                        abs(commission_local_for_this_tx)
                        if pd.notna(commission_local_for_this_tx)
                        else 0
                    )
                logging.debug(
                    f"TRACE E*TRADE ({symbol}, {tx_type}, Q:{qty:.2f}, P:{price_local:.2f}, Comm:{commission_local_for_this_tx:.2f}): CumInv: {prev_cum_inv:.2f} -> {current_cum_inv:.2f} | Cost: {cost:.2f}, Proceeds: {proceeds:.2f}, DivEffect: {div_effect:.2f}, Fee: {fee_cost:.2f}"
                )

            # Add commission to overall total if the transaction wasn't skipped or a split
            if commission_for_overall != 0:
                overall_commissions_local[holding["local_currency"]] += abs(
                    commission_for_overall
                )

        # --- Exception Handling for row processing (Unchanged) ---
        except (ValueError, TypeError, ZeroDivisionError) as e:
            error_msg = f"Calculation Error ({type(e).__name__}): {e}"
            logging.warning(
                f"WARN in _process_transactions row {original_index} ({symbol}, {tx_type}): {error_msg}. Skipping row."
            )
            ignored_reasons_local[original_index] = error_msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            if log_this_row:
                logging.error(
                    f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}"
                )
            continue
        except KeyError as e:
            error_msg = f"Internal Holding Data Error: {e}"
            logging.warning(
                f"WARN in _process_transactions row {original_index} ({symbol}, {tx_type}): {error_msg}. Skipping row."
            )
            ignored_reasons_local[original_index] = error_msg
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            if log_this_row:
                logging.error(
                    f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}"
                )
            continue
        except Exception as e:
            logging.exception(
                f"Unexpected error processing row {original_index} ({symbol}, {tx_type})"
            )
            ignored_reasons_local[original_index] = "Unexpected Processing Error"
            ignored_row_indices_local.add(original_index)
            has_warnings = True
            if log_this_row:
                logging.error(
                    f"ERROR during TRACE E*TRADE ({symbol}, {tx_type}): {e}. PrevCumInv: {prev_cum_inv:.2f}"
                )
            continue

    # Clean up holdings with zero quantity if needed (optional, but can reduce final dict size)
    # holdings = {key: data for key, data in holdings.items() if abs(data.get('qty', 0.0)) > 1e-9}

    return (
        holdings,
        dict(overall_realized_gains_local),
        dict(overall_dividends_local),
        dict(overall_commissions_local),
        ignored_row_indices_local,
        ignored_reasons_local,
        has_warnings,  # Return overall warning status
    )


def _calculate_cash_balances(
    transactions_df: pd.DataFrame, default_currency: str
) -> Tuple[Dict[str, Dict], bool, bool]:  # Added has_errors, has_warnings
    """
    Calculates the final cash balance, dividends, and commissions for each account's $CASH position.

    Filters transactions for the '$CASH' symbol. Aggregates quantities based on 'deposit'/'buy' (+)
    and 'withdrawal'/'sell' (-) types. Separately aggregates net dividends received on cash
    and total commissions paid on cash transactions.

    Args:
        transactions_df (pd.DataFrame): The cleaned and filtered transactions DataFrame, including
                                        $CASH transactions. Must contain 'Symbol', 'Account', 'Type',
                                        'Quantity', 'Commission', 'Total Amount', 'Price/Share',
                                        'Local Currency'.
        default_currency (str): The default currency to assign if a cash account's currency
                                cannot be determined.

    Returns:
        Tuple containing:
        - cash_summary (Dict[str, Dict]): Dictionary keyed by account name. Values are dicts
            containing cash details: 'qty' (final balance), 'realized' (always 0.0 for cash),
            'dividends' (net dividends received on cash), 'commissions' (total commissions on cash tx),
            'currency' (local currency of the cash balance).
        - has_errors (bool): True if critical errors occurred during aggregation (e.g., missing columns,
                             type errors preventing calculation); False otherwise.
        - has_warnings (bool): True if non-critical issues occurred (currently always False for this function);
                               False otherwise.
    """
    cash_summary: Dict[str, Dict] = {}
    cash_symbol = CASH_SYMBOL_CSV
    has_errors = False
    has_warnings = False

    try:
        cash_transactions = transactions_df[
            transactions_df["Symbol"] == cash_symbol
        ].copy()
        if not cash_transactions.empty:
            # Define helper function (moved inside for scope, consider global if reused heavily)
            def get_signed_quantity_cash(row):
                type_lower = str(row.get("Type", "")).lower()
                qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
                if pd.isna(qty):
                    return 0.0  # Treat missing quantity as zero flow
                if type_lower in ["buy", "deposit"]:
                    return abs(qty)
                elif type_lower in ["sell", "withdrawal"]:
                    return -abs(qty)
                else:
                    return 0.0  # Other types don't affect cash quantity directly

            cash_transactions["SignedQuantity"] = cash_transactions.apply(
                get_signed_quantity_cash, axis=1
            )

            # Aggregate (keep existing logic)
            grouped_cash = cash_transactions.groupby("Account")
            cash_qty_agg = grouped_cash["SignedQuantity"].sum()
            cash_comm_agg = (
                grouped_cash["Commission"].sum(min_count=1).fillna(0.0)
            )  # Sum ignores NaN, fillna handles empty groups
            cash_currency_map = grouped_cash["Local Currency"].first()

            # Calculate dividends (keep existing logic)
            cash_dividends_tx = cash_transactions[
                cash_transactions["Type"] == "dividend"
            ].copy()
            cash_div_agg = pd.Series(dtype=float)
            if not cash_dividends_tx.empty:

                def get_dividend_amount(r):  # Inner helper is fine here
                    total_amt = pd.to_numeric(r.get("Total Amount"), errors="coerce")
                    price = pd.to_numeric(r.get("Price/Share"), errors="coerce")
                    qty = pd.to_numeric(r.get("Quantity"), errors="coerce")
                    qty_abs = abs(qty) if pd.notna(qty) else 0.0
                    if pd.notna(total_amt) and abs(total_amt) > 1e-9:
                        return total_amt
                    elif pd.notna(price) and abs(price) > 1e-9:
                        return (qty_abs * price) if qty_abs > 0 else price
                    else:
                        return 0.0

                cash_dividends_tx["DividendAmount"] = cash_dividends_tx.apply(
                    get_dividend_amount, axis=1
                )
                cash_dividends_tx["Commission"] = pd.to_numeric(
                    cash_dividends_tx["Commission"], errors="coerce"
                ).fillna(0.0)
                cash_dividends_tx["NetDividend"] = (
                    cash_dividends_tx["DividendAmount"]
                    - cash_dividends_tx["Commission"]
                )
                cash_div_agg = cash_dividends_tx.groupby("Account")["NetDividend"].sum()

            all_cash_accounts = cash_currency_map.index.union(cash_qty_agg.index).union(
                cash_div_agg.index
            )

            # Loop and retrieve (keep existing logic)
            for acc in all_cash_accounts:
                acc_currency = cash_currency_map.get(acc, default_currency)
                acc_balance = (
                    cash_qty_agg.get(acc, 0.0)
                    if isinstance(cash_qty_agg, pd.Series)
                    else (float(cash_qty_agg) if pd.notna(cash_qty_agg) else 0.0)
                )
                acc_commissions = (
                    cash_comm_agg.get(acc, 0.0)
                    if isinstance(cash_comm_agg, pd.Series)
                    else (float(cash_comm_agg) if pd.notna(cash_comm_agg) else 0.0)
                )
                acc_dividends_only = (
                    cash_div_agg.get(acc, 0.0)
                    if isinstance(cash_div_agg, pd.Series)
                    else 0.0
                )

                cash_summary[acc] = {
                    "qty": acc_balance,
                    "realized": 0.0,
                    "dividends": acc_dividends_only,
                    "commissions": acc_commissions,
                    "currency": acc_currency,
                }
        else:
            logging.info(
                "Info in _calculate_cash_balances: No $CASH transactions found."
            )
    # --- Refined Exception Handling ---
    except (TypeError, ValueError) as e:
        logging.error(f"Data type/value error calculating cash balances: {e}")
        has_errors = True  # Treat aggregation errors as critical
    except KeyError as e:
        logging.error(f"Missing expected column calculating cash balances: {e}")
        has_errors = True
    except Exception as e:
        logging.exception(f"Unexpected error calculating cash balances")
        has_errors = True
    # --- End Refined Exception Handling ---

    return cash_summary, has_errors, has_warnings  # Return flags


# --- REVISED: _build_summary_rows (with total_buy_cost) ---
def _build_summary_rows(
    holdings: Dict[Tuple[str, str], Dict],
    cash_summary: Dict[str, Dict],
    current_stock_data: Dict[str, Dict[str, Optional[float]]],
    current_fx_rates_vs_usd: Dict[str, float],
    display_currency: str,
    default_currency: str,
    transactions_df: pd.DataFrame,
    report_date: date,
    shortable_symbols: Set[str],
    excluded_symbols: Set[str],
    manual_prices_dict: Dict[str, float],
) -> Tuple[
    List[Dict[str, Any]], Dict[str, float], Dict[str, str], bool, bool
]:  # Added has_errors, has_warnings
    """
    Builds the detailed list of portfolio summary rows, converting values to the display currency.

    Iterates through processed stock/ETF holdings and cash balances. Fetches the current price
    (using API/cache data, fallback manual prices or fallback to last transaction price). Calculates market value,
    cost basis, unrealized gain/loss, total gain, total return %, and IRR for each position.
    Converts all monetary values from their local currency to the specified `display_currency`
    using the provided current FX rates. Aggregates the total market value for each account
    in its local currency.

    Args:
        holdings (Dict[Tuple[str, str], Dict]): Processed stock/ETF holdings data from
                                                `_process_transactions_to_holdings`.
        cash_summary (Dict[str, Dict]): Processed cash balance data from `_calculate_cash_balances`.
        current_stock_data (Dict[str, Dict[str, Optional[float]]]): Dictionary of current stock prices
                                                                     and changes from `get_cached_or_fetch_yfinance_data`.
        current_fx_rates_vs_usd (Dict[str, float]): Dictionary of current FX rates relative to USD
                                                    from `get_cached_or_fetch_yfinance_data`.
        display_currency (str): The target currency for displaying results.
        default_currency (str): The default currency.
        transactions_df (pd.DataFrame): The cleaned, filtered transactions DataFrame (used for IRR calculation
                                        and price fallback).
        report_date (date): The date for which the summary is being generated (used for IRR).
        shortable_symbols (Set[str]): Set of symbols allowed for shorting.
        excluded_symbols (Set[str]): Set of symbols to exclude from price fetching/use fallback.

    Returns:
        Tuple containing:
        - portfolio_summary_rows (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
            represents a holding (row) in the final summary table, with values calculated in the
            `display_currency`. Includes columns like 'Account', 'Symbol', 'Quantity', 'Market Value (...)',
            'Total Gain (...)', 'Total Return %', 'IRR (%)', etc.
        - account_market_values_local (Dict[str, float]): Dictionary mapping account name to its total
            market value calculated in its *local* currency.
        - account_local_currency_map (Dict[str, str]): Dictionary mapping account name to its determined
            local currency.
        - has_errors (bool): True if critical errors occurred, primarily missing FX rates preventing
                             conversion to the display currency; False otherwise.
        - has_warnings (bool): True if non-critical issues occurred (e.g., fallback price used,
                               IRR calculation failed for a holding); False otherwise.
    """
    portfolio_summary_rows: List[Dict[str, Any]] = []
    account_market_values_local: Dict[str, float] = defaultdict(float)
    account_local_currency_map: Dict[str, str] = {}
    has_errors = False  # Flag for critical issues preventing accurate summary
    has_warnings = (
        False  # Flag for non-critical issues (e.g., fallback price, missing IRR)
    )

    logging.info(f"Calculating final portfolio summary rows in {display_currency}...")

    portfolio_summary_rows: List[Dict[str, Any]] = []
    account_market_values_local: Dict[str, float] = defaultdict(float)
    account_local_currency_map: Dict[str, str] = {}
    has_errors = False
    has_warnings = False

    logging.info(f"Calculating final portfolio summary rows in {display_currency}...")

    # --- Loop 1: Process Stock/ETF Holdings ---
    for holding_key, data in holdings.items():
        symbol, account = holding_key
        current_qty = data.get("qty", 0.0)
        realized_gain_local = data.get("realized_gain_local", 0.0)
        dividends_local = data.get("dividends_local", 0.0)
        commissions_local = data.get("commissions_local", 0.0)
        local_currency = data.get("local_currency", default_currency)
        # ... (get other local values: cost_basis, short_proceeds, etc.) ...
        current_total_cost_local = data.get("total_cost_local", 0.0)
        short_proceeds_local = data.get("short_proceeds_local", 0.0)
        total_cost_invested_local = data.get("total_cost_invested_local", 0.0)
        cumulative_investment_local = data.get("cumulative_investment_local", 0.0)
        total_buy_cost_local = data.get("total_buy_cost_local", 0.0)

        account_local_currency_map[account] = local_currency
        stock_data = current_stock_data.get(symbol, {})
        current_price_local_raw = stock_data.get("price")
        day_change_local_raw = stock_data.get("change")
        day_change_pct_raw = stock_data.get("changesPercentage")

        # --- Price Determination ---
        price_source = "Unknown"
        current_price_local = np.nan  # Start assuming no price
        day_change_local = np.nan
        day_change_pct = np.nan

        is_excluded = symbol in excluded_symbols
        # --- Get potential live price ---
        current_price_local_raw = stock_data.get("price")
        is_yahoo_price_valid = (
            pd.notna(current_price_local_raw)
            and isinstance(current_price_local_raw, (int, float))
            and current_price_local_raw > 1e-9
        )

        # --- Step 1: Try Yahoo Price (if not excluded) ---
        if not is_excluded and is_yahoo_price_valid:
            price_source = "Yahoo API/Cache"
            current_price_local = float(current_price_local_raw)
            # Assign day change only if Yahoo price is valid
            day_change_local_raw = stock_data.get("change")
            day_change_pct_raw = stock_data.get("changesPercentage")
            day_change_local = (
                float(day_change_local_raw)
                if pd.notna(day_change_local_raw)
                else np.nan
            )
            day_change_pct = (
                float(day_change_pct_raw) if pd.notna(day_change_pct_raw) else np.nan
            )
            logging.debug(
                f"Price OK ({symbol}): Using Yahoo price {current_price_local}"
            )
        elif not is_excluded:  # Yahoo price failed for NON-excluded symbol
            logging.warning(
                f"Warning: Yahoo price invalid/missing for {symbol}. Trying fallbacks."
            )
            price_source = "Yahoo Invalid"  # Base source before fallbacks
            has_warnings = True
        elif is_excluded:
            logging.debug(  # Changed to info as this is expected
                f"Info: Symbol {symbol} is excluded. Skipping Yahoo fetch, trying fallbacks."
            )
            price_source = "Excluded"  # Base source before fallbacks
            has_warnings = True

        # --- Step 2: Try Manual Fallback (if price still NaN) ---
        if pd.isna(current_price_local):
            manual_price = manual_prices_dict.get(
                symbol
            )  # Assumes manual_prices_dict is passed
            # Check if manual price is a valid positive number
            if (
                manual_price is not None
                and pd.notna(manual_price)
                and isinstance(manual_price, (int, float))
                and manual_price > 0
            ):
                current_price_local = float(manual_price)
                price_source += " - Manual Fallback"  # Append source info
                logging.debug(
                    f"Info: Used MANUAL fallback price {current_price_local} for {symbol}/{account}"
                )
                # Manual price means no reliable day change
                day_change_local = np.nan
                day_change_pct = np.nan
            else:
                # Log even if manual price key exists but value is invalid/zero
                if symbol in manual_prices_dict:
                    logging.warning(
                        f"Warn: Manual price found for {symbol} but was invalid/zero ({manual_price}). Trying Last TX."
                    )
                else:
                    logging.debug(
                        f"Info: Manual price not found for {symbol}. Trying Last TX."
                    )
                # Don't set price_source here, let Step 3 handle it

        # --- Step 3: Try Last Transaction Fallback (if price still NaN) ---
        if pd.isna(current_price_local):
            price_source += (
                " - No Manual" if "Manual Fallback" not in price_source else ""
            )  # Indicate if manual was missing
            try:
                fallback_tx = transactions_df[
                    (transactions_df["Symbol"] == symbol)
                    & (transactions_df["Account"] == account)
                    & (transactions_df["Price/Share"].notna())
                    & (
                        pd.to_numeric(transactions_df["Price/Share"], errors="coerce")
                        > 1e-9
                    )
                    & (transactions_df["Date"].dt.date <= report_date)
                ].copy()

                if not fallback_tx.empty:
                    fallback_tx.sort_values(
                        by=["Date", "original_index"], inplace=True, ascending=True
                    )
                    last_tx_row = fallback_tx.iloc[-1]
                    last_tx_price = pd.to_numeric(
                        last_tx_row["Price/Share"], errors="coerce"
                    )
                    last_tx_date = last_tx_row["Date"].date()

                    if pd.notna(last_tx_price) and last_tx_price > 1e-9:
                        current_price_local = float(last_tx_price)
                        price_source += (
                            f" - Last TX ({last_tx_price:.2f}@{last_tx_date})"
                        )
                        logging.debug(
                            f"Info: Used Last TX fallback price {current_price_local} for {symbol}/{account}"
                        )
                    else:
                        logging.warning(
                            f"Warn: Fallback TX found for {symbol}/{account} but price invalid ({last_tx_price}). Using 0."
                        )
                        price_source += " - Last TX Invalid/Zero"
                        current_price_local = 0.0
                else:
                    logging.warning(
                        f"Warn: No valid prior TX found for fallback for {symbol}/{account}. Using 0."
                    )
                    price_source += " - No Last TX"
                    current_price_local = 0.0
            except Exception as e_fallback:
                logging.error(
                    f"ERROR during last TX fallback for {symbol}/{account}: {e_fallback}"
                )
                price_source += " - Fallback Error"
                current_price_local = 0.0
            # Ensure day change is NaN if last TX fallback was attempted
            day_change_local = np.nan
            day_change_pct = np.nan

        # --- Step 4: Final Check (if price STILL NaN) ---
        if pd.isna(current_price_local):
            logging.error(
                f"ERROR: All price sources failed for {symbol}/{account}. Forcing 0."
            )
            current_price_local = 0.0
            # Ensure Price Source reflects the ultimate failure state if needed
            if "Using 0" not in price_source and "Error" not in price_source:
                price_source += " - Forced Zero"
            has_warnings = True

        # Ensure price is float
        current_price_local = float(current_price_local)

        # --- Currency Conversion ---
        fx_rate = get_conversion_rate(
            local_currency, display_currency, current_fx_rates_vs_usd
        )
        if pd.isna(fx_rate):
            logging.error(
                f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for {symbol}/{account}."
            )
            has_errors = True
            fx_rate = np.nan  # Use NaN to propagate error

        # --- Calculate Display Currency Values ---
        market_value_local = current_qty * current_price_local
        if pd.notna(market_value_local):
            account_market_values_local[account] += market_value_local

        market_value_display = (
            market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        day_change_value_display = (
            (current_qty * day_change_local * fx_rate)
            if pd.notna(day_change_local) and pd.notna(fx_rate)
            else np.nan
        )
        current_price_display = (
            current_price_local * fx_rate
            if pd.notna(current_price_local) and pd.notna(fx_rate)
            else np.nan
        )
        cost_basis_display = np.nan
        avg_cost_price_display = np.nan
        unrealized_gain_display = np.nan
        unrealized_gain_pct = np.nan

        # Unrealized Gain/Loss (existing logic)
        # ... (no changes needed here) ...
        is_long = current_qty > 1e-9
        is_short = current_qty < -1e-9
        if is_long:
            cost_basis_display_local = max(
                0, current_total_cost_local
            )  # Local cost basis cannot be negative
            cost_basis_display = (
                cost_basis_display_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(cost_basis_display):
                avg_cost_price_display = (
                    (cost_basis_display / current_qty)
                    if abs(current_qty) > 1e-9
                    else np.nan
                )
                if pd.notna(market_value_display):
                    unrealized_gain_display = market_value_display - cost_basis_display
                    if abs(cost_basis_display) > 1e-9:
                        unrealized_gain_pct = (
                            unrealized_gain_display / cost_basis_display
                        ) * 100.0
                    elif abs(market_value_display) > 1e-9:
                        unrealized_gain_pct = np.inf
                    else:
                        unrealized_gain_pct = 0.0
        elif is_short:
            avg_cost_price_display = np.nan
            cost_basis_display = 0.0
            short_proceeds_display = (
                short_proceeds_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            if pd.notna(market_value_display) and pd.notna(short_proceeds_display):
                current_cost_to_cover_display = abs(market_value_display)
                unrealized_gain_display = (
                    short_proceeds_display - current_cost_to_cover_display
                )
                if abs(short_proceeds_display) > 1e-9:
                    unrealized_gain_pct = (
                        unrealized_gain_display / short_proceeds_display
                    ) * 100.0
                elif abs(current_cost_to_cover_display) > 1e-9:
                    unrealized_gain_pct = -np.inf
                else:
                    unrealized_gain_pct = 0.0

        # Other Display Values
        realized_gain_display = (
            realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        dividends_display = dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
        commissions_display = (
            commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        total_cost_invested_display = (
            total_cost_invested_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        cumulative_investment_display = (
            cumulative_investment_local * fx_rate if pd.notna(fx_rate) else np.nan
        )
        total_buy_cost_display = (
            total_buy_cost_local * fx_rate if pd.notna(fx_rate) else np.nan
        )  # <<< Convert new value

        # Total Gain
        unrealized_gain_comp = (
            unrealized_gain_display if pd.notna(unrealized_gain_display) else 0.0
        )
        total_gain_display = (
            (
                realized_gain_display
                + unrealized_gain_comp
                + dividends_display
                - commissions_display
            )
            if all(
                pd.notna(v)
                for v in [realized_gain_display, dividends_display, commissions_display]
            )
            else np.nan
        )

        # --- MODIFIED: Total Return % Calculation ---
        denominator_for_pct = total_buy_cost_display  # <<< USE TOTAL BUY COST
        total_return_pct = np.nan
        if pd.notna(total_gain_display) and pd.notna(denominator_for_pct):
            if abs(denominator_for_pct) > 1e-9:
                total_return_pct = (total_gain_display / denominator_for_pct) * 100.0
            elif abs(total_gain_display) <= 1e-9:
                total_return_pct = 0.0
        # --- End Modification ---

        # Calculate IRR (existing logic)
        # ... (no changes needed here) ...
        stock_irr = np.nan
        try:
            market_value_local_for_irr = (
                abs(market_value_local)
                if abs(current_qty) > 1e-9 and pd.notna(market_value_local)
                else 0.0
            )
            cf_dates, cf_values = get_cash_flows_for_symbol_account(
                symbol,
                account,
                transactions_df,
                market_value_local_for_irr,
                report_date,
            )
            if cf_dates and cf_values:
                stock_irr = calculate_irr(cf_dates, cf_values)
        except Exception as e_irr:
            logging.warning(
                f"Warning: IRR calculation failed for {symbol}/{account}: {e_irr}"
            )
            has_warnings = True
            stock_irr = np.nan
        irr_value_to_store = stock_irr * 100.0 if pd.notna(stock_irr) else np.nan
        if (
            pd.isna(irr_value_to_store) and current_qty != 0
        ):  # Check non-zero qty more strictly
            logging.debug(f"Debug: IRR is NaN for non-zero holding {symbol}/{account}")

        # --- Append row data ---
        portfolio_summary_rows.append(
            {
                "Account": account,
                "Symbol": symbol,
                "Quantity": current_qty,
                f"Avg Cost ({display_currency})": avg_cost_price_display,
                f"Price ({display_currency})": current_price_display,
                f"Cost Basis ({display_currency})": cost_basis_display,
                f"Market Value ({display_currency})": market_value_display,
                f"Day Change ({display_currency})": day_change_value_display,
                "Day Change %": day_change_pct,
                f"Unreal. Gain ({display_currency})": unrealized_gain_display,
                "Unreal. Gain %": unrealized_gain_pct,
                f"Realized Gain ({display_currency})": realized_gain_display,
                f"Dividends ({display_currency})": dividends_display,
                f"Commissions ({display_currency})": commissions_display,
                f"Total Gain ({display_currency})": total_gain_display,
                f"Total Cost Invested ({display_currency})": total_cost_invested_display,  # Legacy, maybe remove later
                "Total Return %": total_return_pct,  # <<< Uses new calculation base
                f"Cumulative Investment ({display_currency})": cumulative_investment_display,  # Keep for potential analysis
                f"Total Buy Cost ({display_currency})": total_buy_cost_display,  # <<< Add new value (optional for output df)
                "IRR (%)": irr_value_to_store,
                "Local Currency": local_currency,
                "Price Source": price_source,
            }
        )
    # --- End Stock/ETF Loop ---

    # --- Loop 2: Process CASH Balances (Add Total Buy Cost column for consistency) ---
    if cash_summary:
        for account, cash_data in cash_summary.items():
            # ... (existing cash setup) ...
            symbol = CASH_SYMBOL_CSV
            current_qty = cash_data.get("qty", 0.0)
            local_currency = cash_data.get("currency", default_currency)
            # ... (other local values) ...
            realized_gain_local = cash_data.get("realized", 0.0)
            dividends_local = cash_data.get("dividends", 0.0)
            commissions_local = cash_data.get("commissions", 0.0)

            account_local_currency_map[account] = local_currency
            fx_rate = get_conversion_rate(
                local_currency, display_currency, current_fx_rates_vs_usd
            )
            if pd.isna(fx_rate):
                has_errors = True
                fx_rate = np.nan
                logging.error(
                    f"CRITICAL ERROR: Failed FX rate {local_currency}->{display_currency} for CASH in {account}."
                )

            # Calculate Display Values
            market_value_local = current_qty * 1.0
            if pd.notna(market_value_local):
                account_market_values_local[account] += market_value_local
            market_value_display = (
                market_value_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            # ... (other cash display value calculations remain the same) ...
            current_price_display = 1.0 * fx_rate if pd.notna(fx_rate) else np.nan
            cost_basis_display = market_value_display
            avg_cost_price_display = current_price_display
            day_change_value_display = 0.0
            day_change_pct = 0.0
            unrealized_gain_display = 0.0
            unrealized_gain_pct = 0.0
            realized_gain_display = (
                realized_gain_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            dividends_display = (
                dividends_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            commissions_display = (
                commissions_local * fx_rate if pd.notna(fx_rate) else np.nan
            )
            total_gain_display = (
                (dividends_display - commissions_display)
                if pd.notna(dividends_display) and pd.notna(commissions_display)
                else np.nan
            )
            cumulative_investment_display = (
                market_value_display if pd.notna(market_value_display) else np.nan
            )
            total_buy_cost_display = (
                market_value_display  # For cash, buy cost ~ market value
            )
            total_return_pct_cash = np.nan
            irr_value_to_store_cash = np.nan

            portfolio_summary_rows.append(
                {
                    "Account": account,
                    "Symbol": symbol,
                    "Quantity": current_qty,
                    f"Avg Cost ({display_currency})": avg_cost_price_display,
                    f"Price ({display_currency})": current_price_display,
                    f"Cost Basis ({display_currency})": cost_basis_display,
                    f"Market Value ({display_currency})": market_value_display,
                    f"Day Change ({display_currency})": day_change_value_display,
                    "Day Change %": day_change_pct,
                    f"Unreal. Gain ({display_currency})": unrealized_gain_display,
                    "Unreal. Gain %": unrealized_gain_pct,
                    f"Realized Gain ({display_currency})": realized_gain_display,
                    f"Dividends ({display_currency})": dividends_display,
                    f"Commissions ({display_currency})": commissions_display,
                    f"Total Gain ({display_currency})": total_gain_display,
                    f"Total Cost Invested ({display_currency})": cost_basis_display,  # Keep legacy name pointing to cost basis
                    "Total Return %": total_return_pct_cash,
                    f"Cumulative Investment ({display_currency})": cumulative_investment_display,
                    f"Total Buy Cost ({display_currency})": total_buy_cost_display,  # <<< Add new value
                    "IRR (%)": irr_value_to_store_cash,
                    "Local Currency": local_currency,
                    "Price Source": "N/A (Cash)",
                }
            )

    return (
        portfolio_summary_rows,
        dict(account_market_values_local),
        dict(account_local_currency_map),
        has_errors,
        has_warnings,
    )


# Revised safe_sum - Attempt 3 (Direct Sum on Subset)
def safe_sum(df, col):
    """
    Safely sums a DataFrame column, handling NaNs and non-numeric types.

    Attempts to convert the specified column to numeric, coercing errors to NaN.
    Fills any remaining NaNs with 0.0 and then calculates the sum. Returns 0.0
    if the column doesn't exist or if any error occurs during the process.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col (str): The name of the column to sum.

    Returns:
        float: The sum of the numeric values in the column, or 0.0 on failure or if empty.
    """
    if col not in df.columns:
        return 0.0  # Column doesn't exist

    try:
        # Select the column, convert errors to NaN, fill NaN with 0, then sum
        # This ensures we are working with a Series before sum
        data_series = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
        total = data_series.sum()
        # Ensure return is a standard float, handle potential lingering NaNs from sum itself
        return float(total) if pd.notna(total) else 0.0
    except Exception as e:
        logging.error(f"Error in safe_sum for column {col}: {e}")  # Optional debug
        return 0.0  # Return 0 on any unexpected error during sum


# --- REVISED: _calculate_aggregate_metrics ---
def _calculate_aggregate_metrics(
    full_summary_df: pd.DataFrame,
    display_currency: str,
    transactions_df: pd.DataFrame,  # Still passed but unused here now
    report_date: date,
    # MWR arguments removed
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]], bool, bool]:
    """
    Calculates account-level and overall portfolio summary metrics.
    Uses 'Total Buy Cost' as denominator for Total Return %.
    Returns: overall_summary_metrics, account_level_metrics, has_errors, has_warnings
    """
    account_level_metrics: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {
            "mwr": np.nan,
            "total_return_pct": np.nan,
            "total_market_value_display": 0.0,
            "total_realized_gain_display": 0.0,
            "total_unrealized_gain_display": 0.0,
            "total_dividends_display": 0.0,
            "total_commissions_display": 0.0,
            "total_gain_display": 0.0,
            "total_cash_display": 0.0,
            "total_cost_invested_display": 0.0,
            "total_buy_cost_display": 0.0,
            "total_day_change_display": 0.0,
            "total_day_change_percent": np.nan,
        }
    )
    overall_summary_metrics = {}  # Initialize
    has_errors = False
    has_warnings = False

    logging.info("Calculating Account-Level & Overall Metrics...")
    # --- DEBUGGING START ---
    logging.debug("--- Inside _calculate_aggregate_metrics ---")
    if full_summary_df is None or full_summary_df.empty:
        logging.warning("Input 'full_summary_df' is empty or None.")
        # Return default empty results if input is bad
        # (Copied from earlier logic for empty input)
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
        return (
            overall_summary_metrics,
            dict(account_level_metrics),
            has_errors,
            True,
        )  # Indicate warning
    else:
        logging.debug(f"Input DataFrame shape: {full_summary_df.shape}")
        logging.debug(f"Input DataFrame Columns: {full_summary_df.columns.tolist()}")
        # Log info specifically about SET account if present
        if "Account" in full_summary_df.columns:
            set_rows = full_summary_df[full_summary_df["Account"] == "SET"]
            if not set_rows.empty:
                logging.debug("SET Account Rows in Input DF to Aggregation:")
                # Select key columns to check values
                cols_to_log = [
                    "Account",
                    "Symbol",
                    f"Market Value ({display_currency})",
                    f"Total Gain ({display_currency})",
                    "Local Currency",
                ]
                cols_exist = [c for c in cols_to_log if c in set_rows.columns]
                logging.debug(set_rows[cols_exist].to_string())
            else:
                logging.debug("SET Account rows NOT found in input DataFrame.")
        else:
            logging.warning("Account column missing in aggregate input DF.")
    # --- DEBUGGING END ---

    # --- Input Validation ---
    required_cols = [
        "Account",
        "Symbol",
        "Quantity",
        f"Market Value ({display_currency})",
        f"Total Gain ({display_currency})",
        f"Total Buy Cost ({display_currency})",
    ]
    missing_cols = [col for col in required_cols if col not in full_summary_df.columns]
    if missing_cols:
        logging.warning(
            f"Warning: Aggregate calculation input missing required columns: {missing_cols}. Totals may be inaccurate."
        )
        has_warnings = True

    # --- Account Level Aggregation (Keep as is) ---
    unique_accounts_in_summary = full_summary_df["Account"].unique()
    for account in unique_accounts_in_summary:
        try:
            account_full_df = full_summary_df[full_summary_df["Account"] == account]
            metrics_entry = account_level_metrics[account]
            metrics_entry["mwr"] = np.nan  # MWR Skipped

            cols_to_sum_display = [
                f"Market Value ({display_currency})",
                f"Realized Gain ({display_currency})",
                f"Unreal. Gain ({display_currency})",
                f"Dividends ({display_currency})",
                f"Commissions ({display_currency})",
                f"Total Gain ({display_currency})",
                f"Total Cost Invested ({display_currency})",
                f"Total Buy Cost ({display_currency})",
                f"Day Change ({display_currency})",
            ]
            for col in cols_to_sum_display:
                if col not in account_full_df.columns:
                    logging.warning(
                        f"Warn: Col '{col}' missing for acc '{account}' agg."
                    )
                    has_warnings = True

            metrics_entry["total_market_value_display"] = safe_sum(
                account_full_df, f"Market Value ({display_currency})"
            )
            metrics_entry["total_realized_gain_display"] = safe_sum(
                account_full_df, f"Realized Gain ({display_currency})"
            )
            # ... (rest of account aggregations) ...
            metrics_entry["total_unrealized_gain_display"] = safe_sum(
                account_full_df, f"Unreal. Gain ({display_currency})"
            )
            metrics_entry["total_dividends_display"] = safe_sum(
                account_full_df, f"Dividends ({display_currency})"
            )
            metrics_entry["total_commissions_display"] = safe_sum(
                account_full_df, f"Commissions ({display_currency})"
            )
            metrics_entry["total_gain_display"] = safe_sum(
                account_full_df, f"Total Gain ({display_currency})"
            )
            cash_mask = (
                account_full_df["Symbol"] == CASH_SYMBOL_CSV
                if "Symbol" in account_full_df.columns
                else pd.Series(False, index=account_full_df.index)
            )
            metrics_entry["total_cash_display"] = (
                safe_sum(
                    account_full_df[cash_mask], f"Market Value ({display_currency})"
                )
                if cash_mask.any()
                else 0.0
            )
            metrics_entry["total_cost_invested_display"] = safe_sum(
                account_full_df, f"Total Cost Invested ({display_currency})"
            )
            acc_total_buy_cost_display = safe_sum(
                account_full_df, f"Total Buy Cost ({display_currency})"
            )
            metrics_entry["total_buy_cost_display"] = acc_total_buy_cost_display
            acc_total_gain = metrics_entry["total_gain_display"]
            acc_denominator = acc_total_buy_cost_display
            acc_total_return_pct = np.nan
            if pd.notna(acc_total_gain) and pd.notna(acc_denominator):
                if abs(acc_denominator) > 1e-9:
                    acc_total_return_pct = (acc_total_gain / acc_denominator) * 100.0
                elif abs(acc_total_gain) <= 1e-9:
                    acc_total_return_pct = 0.0
            metrics_entry["total_return_pct"] = acc_total_return_pct
            acc_total_day_change_display = safe_sum(
                account_full_df, f"Day Change ({display_currency})"
            )
            metrics_entry["total_day_change_display"] = acc_total_day_change_display
            acc_current_mv_display = metrics_entry["total_market_value_display"]
            acc_prev_close_mv_display = np.nan
            if pd.notna(acc_current_mv_display) and pd.notna(
                acc_total_day_change_display
            ):
                acc_prev_close_mv_display = (
                    acc_current_mv_display - acc_total_day_change_display
                )
            metrics_entry["total_day_change_percent"] = np.nan
            if pd.notna(acc_total_day_change_display) and pd.notna(
                acc_prev_close_mv_display
            ):
                if abs(acc_prev_close_mv_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        acc_total_day_change_display / acc_prev_close_mv_display
                    ) * 100.0
                elif abs(acc_total_day_change_display) > 1e-9:
                    metrics_entry["total_day_change_percent"] = (
                        np.inf if acc_total_day_change_display > 0 else -np.inf
                    )
                elif abs(acc_total_day_change_display) < 1e-9:
                    metrics_entry["total_day_change_percent"] = 0.0
        except Exception as e_acc_agg:
            logging.exception(f"Error aggregating metrics for account '{account}'")
            has_warnings = True

    # --- Overall metrics ---
    mkt_val_col = f"Market Value ({display_currency})"
    # ... (define other column names needed) ...
    total_gain_col = f"Total Gain ({display_currency})"
    total_buy_cost_col = f"Total Buy Cost ({display_currency})"
    cost_basis_col = f"Cost Basis ({display_currency})"
    unreal_gain_col = f"Unreal. Gain ({display_currency})"
    real_gain_col = f"Realized Gain ({display_currency})"
    divs_col = f"Dividends ({display_currency})"
    comm_col = f"Commissions ({display_currency})"
    cost_invest_col = f"Total Cost Invested ({display_currency})"
    cum_invest_col = f"Cumulative Investment ({display_currency})"
    day_change_col = f"Day Change ({display_currency})"

    # Check overall availability
    cols_to_check = [
        mkt_val_col,
        total_gain_col,
        total_buy_cost_col,
        cost_basis_col,
        unreal_gain_col,
        real_gain_col,
        divs_col,
        comm_col,
        cost_invest_col,
        cum_invest_col,
        day_change_col,
    ]
    for col in cols_to_check:
        if col not in full_summary_df.columns:
            logging.warning(f"Warning: Column '{col}' missing for overall aggregation.")
            has_warnings = True

    overall_market_value_display = safe_sum(full_summary_df, mkt_val_col)
    # Add specific check after sum
    logging.debug(
        f"Intermediate overall_market_value_display: {overall_market_value_display}"
    )

    # ... (rest of overall calculations using defined column variables) ...
    held_mask = pd.Series(False, index=full_summary_df.index)
    if "Quantity" in full_summary_df.columns and "Symbol" in full_summary_df.columns:
        held_mask = (full_summary_df["Quantity"].abs() > 1e-9) | (
            full_summary_df["Symbol"] == CASH_SYMBOL_CSV
        )
    overall_cost_basis_display = (
        safe_sum(full_summary_df.loc[held_mask], cost_basis_col)
        if held_mask.any()
        else 0.0
    )
    overall_unrealized_gain_display = safe_sum(full_summary_df, unreal_gain_col)
    overall_realized_gain_display_agg = safe_sum(full_summary_df, real_gain_col)
    overall_dividends_display_agg = safe_sum(full_summary_df, divs_col)
    overall_commissions_display_agg = safe_sum(full_summary_df, comm_col)
    overall_total_gain_display = safe_sum(full_summary_df, total_gain_col)
    overall_total_cost_invested_display = safe_sum(full_summary_df, cost_invest_col)
    overall_cumulative_investment_display = safe_sum(full_summary_df, cum_invest_col)
    overall_total_buy_cost_display = safe_sum(full_summary_df, total_buy_cost_col)

    overall_day_change_display = safe_sum(full_summary_df, day_change_col)
    overall_prev_close_mv_display = np.nan
    if pd.notna(overall_market_value_display) and pd.notna(overall_day_change_display):
        overall_prev_close_mv_display = (
            overall_market_value_display - overall_day_change_display
        )
    overall_day_change_percent = np.nan
    if pd.notna(overall_day_change_display) and pd.notna(overall_prev_close_mv_display):
        if abs(overall_prev_close_mv_display) > 1e-9:
            overall_day_change_percent = (
                overall_day_change_display / overall_prev_close_mv_display
            ) * 100.0
        elif abs(overall_day_change_display) > 1e-9:
            overall_day_change_percent = (
                np.inf if overall_day_change_display > 0 else -np.inf
            )
        elif abs(overall_day_change_display) < 1e-9:
            overall_day_change_percent = 0.0

    overall_total_return_pct = np.nan
    overall_denominator = overall_total_buy_cost_display
    if pd.notna(overall_total_gain_display) and pd.notna(overall_denominator):
        if abs(overall_denominator) > 1e-9:
            overall_total_return_pct = (
                overall_total_gain_display / overall_denominator
            ) * 100.0
        elif abs(overall_total_gain_display) <= 1e-9:
            overall_total_return_pct = 0.0

    critical_overall_metrics = [
        overall_market_value_display,
        overall_total_gain_display,
    ]
    if any(pd.isna(m) for m in critical_overall_metrics):
        logging.warning("Warning: Critical overall metrics calculated as NaN.")
        has_warnings = True

    overall_summary_metrics = {
        "market_value": overall_market_value_display,
        "cost_basis_held": overall_cost_basis_display,
        "unrealized_gain": overall_unrealized_gain_display,
        "realized_gain": overall_realized_gain_display_agg,
        "dividends": overall_dividends_display_agg,
        "commissions": overall_commissions_display_agg,
        "total_gain": overall_total_gain_display,
        "total_cost_invested": overall_total_cost_invested_display,
        "total_buy_cost": overall_total_buy_cost_display,
        "portfolio_mwr": np.nan,  # MWR not calculated here
        "day_change_display": overall_day_change_display,
        "day_change_percent": overall_day_change_percent,
        "report_date": report_date.strftime("%Y-%m-%d"),
        "display_currency": display_currency,
        "cumulative_investment": overall_cumulative_investment_display,
        "total_return_pct": overall_total_return_pct,
    }
    logging.debug(
        f"--- Finished Aggregating Metrics. Overall Market Value: {overall_market_value_display} ---"
    )

    return (
        overall_summary_metrics,
        dict(account_level_metrics),
        has_errors,
        has_warnings,
    )


# --- Main Calculation Function (Current Portfolio Summary) ---
def calculate_portfolio_summary(
    transactions_csv_file: str,
    fmp_api_key: Optional[str] = None,  # Currently unused but kept for signature
    display_currency: str = "USD",
    show_closed_positions: bool = False,
    account_currency_map: Dict = {"SET": "THB"},  # Default for signature
    default_currency: str = "USD",  # Default for signature
    cache_file_path: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
    include_accounts: Optional[List[str]] = None,
    manual_prices_dict: Optional[Dict[str, float]] = None,
) -> Tuple[
    Optional[Dict[str, Any]],
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Optional[Dict[str, Dict[str, float]]],
    str,
]:
    """
    Calculates the current portfolio summary using Yahoo Finance for market data.

    Orchestrates the process of calculating a snapshot of the portfolio's current state.
    It loads and cleans transactions, filters by specified accounts, processes transactions
    to determine current holdings and cash balances, fetches current stock/ETF prices and FX rates
    (using a cache), builds detailed summary rows with values converted to the display currency,
    calculates aggregate metrics for accounts and the overall portfolio, and prepares a DataFrame
    of any transactions that were ignored during processing.

    Args:
        transactions_csv_file (str): Path to the transactions CSV file.
        fmp_api_key (Optional[str], optional): Financial Modeling Prep API key (currently unused). Defaults to None.
        display_currency (str, optional): The currency for displaying final results. Defaults to 'USD'.
        show_closed_positions (bool, optional): If True, include positions with zero quantity in the
                                                holdings DataFrame. Defaults to False.
        account_currency_map (Dict, optional): Mapping from account name to its local currency.
                                               Defaults to {'SET': 'THB'}.
        default_currency (str, optional): Default currency for accounts not in the map. Defaults to 'USD'.
        cache_file_path (str, optional): Path to the cache file for current price/FX data.
                                         Defaults to DEFAULT_CURRENT_CACHE_FILE_PATH.
        include_accounts (Optional[List[str]], optional): A list of account names to include in the calculation.
                                                          If None or empty, all accounts found in the transaction
                                                          file are included. Defaults to None.

    Returns:
        Tuple containing:
            - Optional[Dict[str, Any]]: Overall summary metrics for the included accounts (e.g., total market
                                        value, total gain, return %, day change %). Includes metadata like
                                        '_available_accounts'. Returns None on critical failure.
            - Optional[pd.DataFrame]: DataFrame containing detailed holdings information for the included
                                      accounts, with values in the display currency. Returns None on failure.
            - Optional[pd.DataFrame]: DataFrame containing transactions that were ignored during the
                                      loading or processing steps, along with the reason. Returns None on failure.
            - Optional[Dict[str, Dict[str, float]]]: Dictionary containing account-level summary metrics
                                                     for included accounts. Returns None on failure.
            - str: A status message string summarizing the outcome (e.g., "Success", "Finished with Warnings",
                   "Finished with Errors") and providing context about filters or issues encountered.
    """
    logging.info(f"Starting Portfolio Calculation (Yahoo Finance - Current Summary)")
    # ... (parameter logging) ...
    has_errors = False  # Overall error flag for the function
    has_warnings = False  # Overall warning flag for the function
    status_parts = []  # Collect status message parts

    original_transactions_df: Optional[pd.DataFrame] = None
    all_transactions_df: Optional[pd.DataFrame] = None
    ignored_row_indices = set()
    ignored_reasons: Dict[int, str] = {}
    report_date = datetime.now().date()  # Use current date for summary report

    # --- 1. Load & Clean ALL Transactions ---
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
    ignored_row_indices.update(ignored_indices_load)
    ignored_reasons.update(ignored_reasons_load)
    if err_load:
        has_errors = True
    if warn_load:
        has_warnings = True
    if ignored_reasons_load:
        status_parts.append(
            f"Load/Clean Issues: {len(ignored_reasons_load)}"
        )  # Concise msg

    if has_errors:  # Critical error during load
        msg = f"Error: File not found or failed critically during load/clean: {transactions_csv_file}"
        logging.error(msg)
        ignored_df_final = (
            original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
            if ignored_row_indices and original_transactions_df is not None
            else pd.DataFrame()
        )
        return None, None, ignored_df_final, None, msg

    # --- (Get available accounts, Filter transactions - logic remains the same) ---
    all_available_accounts_list = []
    if "Account" in all_transactions_df.columns:
        all_available_accounts_list = sorted(
            all_transactions_df["Account"].unique().tolist()
        )

    transactions_df = pd.DataFrame()
    available_accounts_set = set(all_available_accounts_list)
    included_accounts_list = []
    filter_desc = "All Accounts"
    if not include_accounts:
        logging.info(
            "Info: No specific accounts provided, using all available accounts."
        )
        transactions_df = all_transactions_df.copy()
        included_accounts_list = sorted(list(available_accounts_set))
    else:
        valid_include_accounts = [
            acc for acc in include_accounts if acc in available_accounts_set
        ]
        if not valid_include_accounts:
            msg = "Warning: None of the specified accounts to include were found. No data processed."
            logging.warning(msg)
            has_warnings = True
            status_parts.append("No valid included accounts")
            ignored_df_final = (
                original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
                if ignored_row_indices and original_transactions_df is not None
                else pd.DataFrame()
            )
            empty_summary = {"_available_accounts": all_available_accounts_list}
            final_status = (
                "Finished with Warnings" if has_warnings else "Success"
            )  # Final status logic moved here for early exit
            return (
                empty_summary,
                pd.DataFrame(),
                ignored_df_final,
                {},
                f"{final_status} ({msg})",
            )
        logging.info(
            f"Info: Filtering transactions FOR accounts: {', '.join(sorted(valid_include_accounts))}"
        )
        transactions_df = all_transactions_df[
            all_transactions_df["Account"].isin(valid_include_accounts)
        ].copy()
        included_accounts_list = sorted(valid_include_accounts)
        filter_desc = f"Accounts: {', '.join(included_accounts_list)}"

    if transactions_df.empty:
        msg = f"Warning: No transactions remain after filtering for accounts: {', '.join(sorted(include_accounts if include_accounts else []))}"
        logging.warning(msg)
        has_warnings = True
        status_parts.append("No data after filter")
        ignored_df_final = (
            original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
            if ignored_row_indices and original_transactions_df is not None
            else pd.DataFrame()
        )
        empty_summary = {"_available_accounts": all_available_accounts_list}
        final_status = (
            "Finished with Warnings" if has_warnings else "Success"
        )  # Final status logic moved here
        return (
            empty_summary,
            pd.DataFrame(),
            ignored_df_final,
            {},
            f"{final_status} ({msg})",
        )

    # Use the passed-in manual_prices_dict, default to empty if None
    manual_prices_effective = (
        manual_prices_dict if manual_prices_dict is not None else {}
    )

    # --- 3. Process Stock/ETF Transactions ---
    holdings, _, _, _, ignored_indices_proc, ignored_reasons_proc, warn_proc = (
        _process_transactions_to_holdings(
            transactions_df=transactions_df,
            default_currency=default_currency,
            shortable_symbols=SHORTABLE_SYMBOLS,
        )
    )
    ignored_row_indices.update(ignored_indices_proc)
    ignored_reasons.update(ignored_reasons_proc)
    if warn_proc:
        has_warnings = True  # Update overall flag
    if ignored_reasons_proc:
        status_parts.append(f"Processing Issues: {len(ignored_reasons_proc)}")

    # --- 4. Calculate $CASH Balances ---
    cash_summary, err_cash, warn_cash = _calculate_cash_balances(
        transactions_df=transactions_df, default_currency=default_currency
    )
    if err_cash:
        has_errors = True  # Update overall flag
    # if warn_cash: has_warnings = True # Currently no warnings from cash calc

    if has_errors:  # Check if critical error happened during cash calc
        msg = "Error: Failed critically during cash balance calculation."
        logging.error(msg)
        status_parts.append("Cash Calc Error")
        ignored_df_final = (
            original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
            if ignored_row_indices and original_transactions_df is not None
            else pd.DataFrame()
        )
        empty_summary = {"_available_accounts": all_available_accounts_list}
        final_status = "Finished with Errors"  # Final status logic moved here
        return (
            empty_summary,
            pd.DataFrame(),
            ignored_df_final,
            {},
            f"{final_status} ({msg})",
        )

    # --- 5. Fetch Current Market Data ---
    # ... (logic to determine symbols and currencies remains the same) ...
    all_stock_symbols_internal = list(set(key[0] for key in holdings.keys()))
    required_currencies: Set[str] = set([display_currency, default_currency])
    for data in holdings.values():
        required_currencies.add(data.get("local_currency", default_currency))
    for data in cash_summary.values():
        required_currencies.add(data.get("currency", default_currency))
    required_currencies.discard(None)
    required_currencies.discard("N/A")

    # Capture flags from data fetch
    current_stock_data_internal, current_fx_rates_vs_usd, err_fetch, warn_fetch = (
        get_cached_or_fetch_yfinance_data(
            internal_stock_symbols=all_stock_symbols_internal,
            required_currencies=required_currencies,
            cache_file=cache_file_path,
        )
    )
    if err_fetch:
        has_errors = True  # Update overall flag
    if warn_fetch:
        has_warnings = True  # Update overall flag

    if (
        has_errors
        or current_stock_data_internal is None
        or current_fx_rates_vs_usd is None
    ):
        msg = "Error: Price/FX fetch failed critically via Yahoo Finance. Cannot calculate current values."
        logging.error(f"FATAL: {msg}")
        status_parts.append("Fetch Failed")
        ignored_df_final = (
            original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
            if ignored_row_indices and original_transactions_df is not None
            else pd.DataFrame()
        )
        empty_summary = {"_available_accounts": all_available_accounts_list}
        final_status = "Finished with Errors"  # Final status logic moved here
        return (
            empty_summary,
            pd.DataFrame(),
            ignored_df_final,
            {},
            f"{final_status} ({msg})",
        )
    elif has_warnings:  # Check only warnings now, as errors handled above
        status_parts.append("Fetch Warnings")  # Add context

    # --- 6. Build Detailed Summary Rows ---
    # Capture flags from build step
    (
        portfolio_summary_rows,
        account_market_values_local,
        account_local_currency_map,
        err_build,
        warn_build,
    ) = _build_summary_rows(
        holdings=holdings,
        cash_summary=cash_summary,
        current_stock_data=current_stock_data_internal,
        current_fx_rates_vs_usd=current_fx_rates_vs_usd,
        display_currency=display_currency,
        default_currency=default_currency,
        transactions_df=transactions_df,
        report_date=report_date,
        shortable_symbols=SHORTABLE_SYMBOLS,
        excluded_symbols=YFINANCE_EXCLUDED_SYMBOLS,
        manual_prices_dict=manual_prices_effective,  # <-- PASS IT HERE
    )
    if err_build:
        has_errors = True  # Update overall flag
    if warn_build:
        has_warnings = True  # Update overall flag

    if has_errors:  # Check if critical error (like FX failure) happened during build
        msg = "Error: Failed critically during summary row building (likely FX)."
        logging.error(msg)
        status_parts.append("Summary Build Error")
        ignored_df_final = (
            original_transactions_df.loc[sorted(list(ignored_row_indices))].copy()
            if ignored_row_indices and original_transactions_df is not None
            else pd.DataFrame()
        )
        empty_summary = {"_available_accounts": all_available_accounts_list}
        final_status = "Finished with Errors"  # Final status logic moved here
        return (
            empty_summary,
            pd.DataFrame(),
            ignored_df_final,
            {},
            f"{final_status} ({msg})",
        )
    elif not portfolio_summary_rows and (
        holdings or cash_summary
    ):  # If no rows generated despite data
        msg = "Warning: Failed to generate summary rows (FX or other issue)."
        logging.warning(msg)
        has_warnings = True
        status_parts.append("Summary Build Failed")

    summary_df = pd.DataFrame()
    overall_summary_metrics = {}
    account_level_metrics: Dict[str, Dict[str, float]] = {}

    # --- 7. Create DataFrame & Calculate Aggregates ---
    if not portfolio_summary_rows:
        # ... (logic for empty rows remains same, uses has_warnings flag) ...
        logging.warning("Warning: Portfolio summary list is empty after processing.")
        # has_warnings should already be true if we got here and build failed
        overall_summary_metrics = {  # Populate with zeros/NaNs
            "market_value": 0.0,
            "cost_basis_held": 0.0,
            "unrealized_gain": 0.0,
            "realized_gain": 0.0,
            "dividends": 0.0,
            "commissions": 0.0,
            "total_gain": 0.0,
            "total_cost_invested": 0.0,
            "portfolio_mwr": np.nan,
            "day_change_display": 0.0,
            "day_change_percent": np.nan,
            "report_date": report_date.strftime("%Y-%m-%d"),
            "display_currency": display_currency,
            "cumulative_investment": 0.0,
            "total_return_pct": np.nan,
        }
        summary_df = pd.DataFrame()
    else:
        full_summary_df = pd.DataFrame(portfolio_summary_rows)
        # Convert data types (add try-except)
        try:
            # ... (numeric conversion logic) ...
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
            for col in numeric_cols_to_convert:
                if col in full_summary_df.columns:
                    full_summary_df[col] = pd.to_numeric(
                        full_summary_df[col], errors="coerce"
                    )
        except Exception as e:
            logging.warning(
                f"Warning: Error during numeric conversion of summary columns: {e}"
            )
            has_warnings = True

        # Sort (add try-except)
        try:
            full_summary_df.sort_values(
                by=["Account", f"Market Value ({display_currency})"],
                ascending=[True, False],
                na_position="last",
                inplace=True,
            )
        except KeyError:
            logging.warning("Warning: Could not sort summary DataFrame.")
            has_warnings = True  # Sorting failure is a warning

        # Calculate Aggregates and capture flags
        overall_summary_metrics, account_level_metrics, err_agg, warn_agg = (
            _calculate_aggregate_metrics(
                full_summary_df=full_summary_df,
                display_currency=display_currency,
                transactions_df=transactions_df,
                report_date=report_date,  # Removed MWR args
            )
        )
        if err_agg:
            has_errors = (
                True  # Update overall flag (though currently not set in helper)
            )
        if warn_agg:
            has_warnings = True  # Update overall flag

        # Check price source warning (already updates has_warnings)
        price_source_warnings = False
        if "Price Source" in full_summary_df.columns:
            non_cash_holdings = full_summary_df[
                full_summary_df["Symbol"] != CASH_SYMBOL_CSV
            ]
            if (
                not non_cash_holdings.empty
                and non_cash_holdings["Price Source"]
                .str.contains("Fallback|Excluded|Invalid", na=False)
                .any()
            ):
                price_source_warnings = True  # Local flag for message below
        if price_source_warnings:
            logging.warning("Warning: Some holdings used fallback prices.")
            has_warnings = True
            status_parts.append("Fallback Prices Used")

        # Filter closed positions
        summary_df_for_return = full_summary_df
        if not show_closed_positions:
            held_mask = (full_summary_df["Quantity"].abs() > 1e-9) | (
                full_summary_df["Symbol"] == CASH_SYMBOL_CSV
            )
            summary_df_for_return = full_summary_df[held_mask].copy()
        summary_df = summary_df_for_return

    # --- Add Metadata to Overall Summary ---
    # ... (logic remains the same) ...
    if overall_summary_metrics is None:
        overall_summary_metrics = {}
    overall_summary_metrics["_available_accounts"] = all_available_accounts_list
    if display_currency != default_currency and current_fx_rates_vs_usd:
        base_to_display_rate = get_conversion_rate(
            default_currency, display_currency, current_fx_rates_vs_usd
        )
        if base_to_display_rate != 1.0 and np.isfinite(base_to_display_rate):
            overall_summary_metrics["exchange_rate_to_display"] = base_to_display_rate

    # --- 8. Prepare Ignored Transactions DataFrame ---
    # ... (logic remains the same) ...
    ignored_df_final = pd.DataFrame()
    if ignored_row_indices and original_transactions_df is not None:
        valid_indices = sorted(
            [
                idx
                for idx in ignored_row_indices
                if idx in original_transactions_df.index
            ]
        )
        if valid_indices:
            ignored_df_final = original_transactions_df.loc[valid_indices].copy()
            try:
                ignored_df_final["Reason Ignored"] = ignored_df_final.index.map(
                    ignored_reasons
                ).fillna("Unknown Reason")
            except Exception as e_reason:
                logging.warning(
                    f"Warning: Could not add 'Reason Ignored' column: {e_reason}"
                )

    # --- 9. Determine Final Status (using flags) ---
    status_prefix = "Success"
    if has_errors:
        status_prefix = "Finished with Errors"
    elif has_warnings:
        status_prefix = "Finished with Warnings"
    final_status = f"{status_prefix} ({filter_desc})"
    if status_parts:
        final_status += f" [{'; '.join(status_parts)}]"  # Append collected context

    logging.info(f"Portfolio Calculation Finished ({filter_desc})")
    return (
        overall_summary_metrics,
        summary_df,
        ignored_df_final,
        dict(account_level_metrics),
        final_status,
    )


# =======================================================================
# --- SECTION: HISTORICAL PERFORMANCE CALCULATION FUNCTIONS (REVISED + PARALLEL + CACHE) ---
# =======================================================================


# --- Function to map internal symbols to Yahoo Finance format ---
def map_to_yf_symbol(internal_symbol: str) -> Optional[str]:
    """
    Maps an internal symbol to a Yahoo Finance compatible ticker, handling specific cases.

    Checks for excluded symbols, cash symbol, explicit mappings (SYMBOL_MAP_TO_YFINANCE),
    and converts BKK (Thailand) stock exchange symbols (e.g., 'ADVANC:BKK' -> 'ADVANC.BK').
    Returns None for symbols that should be excluded or cannot be reliably mapped.

    Args:
        internal_symbol (str): The internal symbol used in the transaction data.

    Returns:
        Optional[str]: The corresponding Yahoo Finance ticker (e.g., 'AAPL', 'BRK-B', 'ADVANC.BK'),
                       or None if the symbol is excluded, is cash, or has an invalid format.
    """
    if (
        internal_symbol == CASH_SYMBOL_CSV
        or internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
    ):
        return None
    if internal_symbol in SYMBOL_MAP_TO_YFINANCE:
        return SYMBOL_MAP_TO_YFINANCE[internal_symbol]
    if internal_symbol.endswith(":BKK"):
        base_symbol = internal_symbol[:-4]
        if base_symbol in SYMBOL_MAP_TO_YFINANCE:
            base_symbol = SYMBOL_MAP_TO_YFINANCE[base_symbol]
        if "." in base_symbol or len(base_symbol) == 0:
            logging.warning(
                f"Hist WARN: Skipping potentially invalid BKK conversion: {internal_symbol}"
            )
            return None
        return f"{base_symbol.upper()}.BK"
    if " " in internal_symbol or any(c in internal_symbol for c in [":", ","]):
        logging.warning(
            f"Hist WARN: Skipping potentially invalid symbol format for YF: {internal_symbol}"
        )
        return None
    return internal_symbol.upper()


# --- Yahoo Finance Historical Data Fetching ---
def fetch_yf_historical(
    symbols_yf: List[str], start_date: date, end_date: date
) -> Dict[str, pd.DataFrame]:
    """
    Fetches historical 'Close' data (automatically adjusted for splits AND dividends) for multiple symbols from Yahoo Finance.

    Downloads historical data using `yfinance.download` with `auto_adjust=True`.
    Processes the results, extracts the adjusted 'Close' price, renames it to 'price',
    sets the index to date objects, cleans the data (removes NaNs, non-positive prices),
    and returns a dictionary mapping the Yahoo Finance ticker to its historical price DataFrame.
    Handles potential errors during download and processing for individual symbols or batches.

    Args:
        symbols_yf (List[str]): A list of Yahoo Finance tickers to fetch data for.
        start_date (date): The start date for the historical data period.
        end_date (date): The end date for the historical data period.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are the Yahoo Finance tickers (str)
            and values are pandas DataFrames containing the historical adjusted prices.
            The DataFrame index is composed of date objects, and it contains a single 'price' column.
            Returns an empty dictionary if `yfinance` is unavailable or no symbols are provided.
            Symbols for which data fetching or processing failed will be omitted.
    """
    if not YFINANCE_AVAILABLE:
        logging.error("Error: yfinance not available for historical fetch.")
        return {}
    historical_data: Dict[str, pd.DataFrame] = {}
    if not symbols_yf:
        logging.warning("Hist Fetch: No symbols provided.")
        return historical_data
    logging.info(
        f"Hist Fetch: Fetching historical data (auto-adjusted) for {len(symbols_yf)} symbols from Yahoo Finance ({start_date} to {end_date})..."
    )
    yf_end_date = max(start_date, end_date) + timedelta(days=1)
    yf_start_date = min(start_date, end_date)
    fetch_batch_size = 50
    symbols_processed = 0
    for i in range(0, len(symbols_yf), fetch_batch_size):
        batch_symbols = symbols_yf[i : i + fetch_batch_size]
        try:
            data = yf.download(
                tickers=batch_symbols,
                start=yf_start_date,
                end=yf_end_date,
                progress=False,
                group_by="ticker",
                auto_adjust=True,
                actions=False,
            )
            if data.empty:
                logging.warning(
                    f"  Hist WARN: No data returned for batch: {', '.join(batch_symbols)}"
                )
                # Reduced verbosity
                continue
            for symbol in batch_symbols:
                df_symbol = None
                try:
                    if len(batch_symbols) == 1 and not data.columns.nlevels > 1:
                        df_symbol = data if not data.empty else None
                    elif symbol in data.columns.levels[0]:
                        df_symbol = data[symbol]
                    elif len(batch_symbols) > 1 and not data.columns.nlevels > 1:
                        logging.warning(
                            f"  Hist WARN: Unexpected flat DataFrame structure for multi-ticker batch with auto_adjust=True. Symbol {symbol} might be missing."
                        )
                        # Reduced verbosity
                        continue
                    else:
                        if isinstance(data, pd.Series) and data.name == symbol:
                            df_symbol = pd.DataFrame(data)
                        elif isinstance(data, pd.DataFrame) and symbol in data.columns:
                            df_symbol = data[[symbol]].rename(columns={symbol: "Close"})
                        else:
                            logging.warning(
                                f"  Hist WARN: Symbol {symbol} not found in yfinance download results for this batch (Structure: {data.columns})."
                            )
                            # Reduced verbosity
                            continue
                    if df_symbol is None or df_symbol.empty:
                        continue
                    price_col = "Close"
                    if price_col not in df_symbol.columns:
                        logging.warning(
                            f"  Hist WARN: Expected 'Close' column not found for {symbol}. Columns: {df_symbol.columns}"
                        )
                        # Reduced verbosity
                        continue
                    df_filtered = df_symbol[[price_col]].copy()
                    df_filtered.rename(columns={price_col: "price"}, inplace=True)
                    df_filtered.index = pd.to_datetime(df_filtered.index).date
                    df_filtered["price"] = pd.to_numeric(
                        df_filtered["price"], errors="coerce"
                    )
                    df_filtered = df_filtered.dropna(subset=["price"])
                    df_filtered = df_filtered[df_filtered["price"] > 1e-6]
                    if not df_filtered.empty:
                        historical_data[symbol] = df_filtered.sort_index()
                except Exception as e_sym:
                    logging.warning(
                        f"  Hist ERROR processing symbol {symbol} within batch: {e_sym}"
                    )
        except Exception as e_batch:
            logging.warning(
                f"  Hist ERROR during yf.download for batch starting with {batch_symbols[0]}: {e_batch}"
            )
        symbols_processed += len(batch_symbols)
        time.sleep(0.2)
    logging.info(
        f"Hist Fetch: Finished fetching ({len(historical_data)} symbols successful)."
    )
    return historical_data


# --- Function to Unadjust Prices based on Splits ---
def _unadjust_prices(
    adjusted_prices_yf: Dict[
        str, pd.DataFrame
    ],  # Key: YF symbol, Val: DF with 'price' (ADJUSTED)
    yf_to_internal_map: Dict[str, str],  # Map YF symbol back to internal
    splits_by_internal_symbol: Dict[
        str, List[Dict]
    ],  # Internal symbol -> List of {'Date': date, 'Split Ratio': float}
    processed_warnings: set,  # To avoid spamming warnings
) -> Dict[str, pd.DataFrame]:
    """
    Derives unadjusted prices from Yahoo Finance's adjusted prices using recorded split history.

    Yahoo Finance's `auto_adjust=True` provides prices adjusted for both splits and dividends.
    To get prices adjusted *only* for dividends (needed for accurate portfolio value calculation
    when splits are handled manually), this function reverses the split adjustments.
    It calculates a cumulative forward split factor for each date based on the provided split history
    and multiplies the adjusted price by this factor.

    Formula: Unadjusted_Price(t) = Adjusted_Price(t) * Cumulative_Forward_Split_Factor(t)
    where Cumulative_Forward_Split_Factor(t) is the product of all split ratios occurring *after* date t.

    Args:
        adjusted_prices_yf (Dict[str, pd.DataFrame]): Dictionary mapping Yahoo Finance tickers to
            DataFrames containing 'price' column (adjusted for splits and dividends).
        yf_to_internal_map (Dict[str, str]): Dictionary mapping Yahoo Finance tickers back to internal symbols.
        splits_by_internal_symbol (Dict[str, List[Dict]]): Dictionary mapping internal symbols to a list
            of split events. Each split event is a dict like {'Date': date, 'Split Ratio': float}.
        processed_warnings (set): A set to track warnings already logged to avoid duplicates.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary mapping Yahoo Finance tickers to DataFrames containing
            the calculated unadjusted 'price' (effectively, adjusted only for dividends). The index
            consists of date objects. Symbols without splits are returned with their original adjusted prices.
    """
    logging.info("--- Starting Price Unadjustment ---")  # Add start log
    unadjusted_prices_yf = {}
    unadjusted_count = 0

    for yf_symbol, adj_price_df in adjusted_prices_yf.items():
        # --- Add Debug for specific symbol ---
        IS_DEBUG_SYMBOL = yf_symbol == "AAPL"  # Check if it's AAPL
        if IS_DEBUG_SYMBOL:
            logging.debug(f"  Processing unadjustment for DEBUG symbol: {yf_symbol}")
        # --- End Debug ---

        if adj_price_df.empty or "price" not in adj_price_df.columns:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    "    Skipping: Adjusted price DataFrame is empty or missing 'price'."
                )
            unadjusted_prices_yf[yf_symbol] = adj_price_df.copy()
            continue

        internal_symbol = yf_to_internal_map.get(yf_symbol)
        symbol_splits = None
        if internal_symbol:
            symbol_splits = splits_by_internal_symbol.get(internal_symbol)

        if not symbol_splits:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    f"    No splits found for internal symbol '{internal_symbol}'. Copying adjusted prices."
                )
            unadjusted_prices_yf[yf_symbol] = adj_price_df.copy()
            continue
        else:
            if IS_DEBUG_SYMBOL:
                logging.debug(
                    f"    Found splits for '{internal_symbol}'. Proceeding with unadjustment."
                )

        unadj_df = adj_price_df.copy()
        if not isinstance(unadj_df.index, pd.DatetimeIndex):
            try:
                unadj_df.index = pd.to_datetime(unadj_df.index, errors="coerce").date
                unadj_df = unadj_df[pd.notnull(unadj_df.index)]
            except Exception:
                # ... (warning log) ...
                if IS_DEBUG_SYMBOL:
                    logging.warning(
                        "    Failed to convert index to date. Skipping unadjustment."
                    )
                unadjusted_prices_yf[yf_symbol] = adj_price_df.copy()
                continue

        if unadj_df.empty:
            unadjusted_prices_yf[yf_symbol] = unadj_df
            continue
        unadj_df.sort_index(inplace=True)

        forward_split_factor = pd.Series(1.0, index=unadj_df.index, dtype=float)
        sorted_splits_desc = sorted(
            symbol_splits, key=lambda x: x.get("Date", date.min), reverse=True
        )

        if IS_DEBUG_SYMBOL:
            logging.debug(f"    Splits (newest first): {sorted_splits_desc}")

        for split_info in sorted_splits_desc:
            try:
                split_date_raw = split_info.get("Date")
                if split_date_raw is None:
                    raise ValueError("Split info missing 'Date'")
                if isinstance(split_date_raw, datetime):
                    split_date = split_date_raw.date()
                elif isinstance(split_date_raw, date):
                    split_date = split_date_raw
                else:
                    raise TypeError(f"Invalid split date type: {type(split_date_raw)}")

                split_ratio = float(split_info["Split Ratio"])
                if split_ratio <= 0:
                    if IS_DEBUG_SYMBOL:
                        logging.warning(
                            f"    Invalid split ratio {split_ratio} on {split_date}. Skipping."
                        )
                    continue

                # Apply split factor to dates *before* the split date
                mask = forward_split_factor.index < split_date
                if IS_DEBUG_SYMBOL and split_date == date(2020, 8, 31):
                    logging.debug(
                        f"    Applying ratio {split_ratio} for split on {split_date}. Mask sum (dates before): {mask.sum()}"
                    )
                forward_split_factor.loc[mask] *= split_ratio

            except (KeyError, ValueError, TypeError, AttributeError) as e:
                # ... (warning log) ...
                if IS_DEBUG_SYMBOL:
                    logging.warning(
                        f"    Error processing split around {split_info.get('Date', 'N/A')}: {e}"
                    )
                continue

        original_prices = unadj_df["price"].copy()
        # Align series before multiplication
        aligned_factor, aligned_prices = forward_split_factor.align(
            unadj_df["price"], join="right", fill_value=1.0
        )
        unadj_df["unadjusted_price"] = (
            aligned_prices * aligned_factor
        )  # Use new column name

        # --- Add Debug logging for specific dates ---
        if IS_DEBUG_SYMBOL:
            debug_dates = [
                date(2020, 8, 28),
                date(2020, 8, 31),
                date(2020, 9, 1),
            ]  # Dates around split
            debug_dates_in_index = [d for d in debug_dates if d in unadj_df.index]
            if debug_dates_in_index:
                logging.debug(
                    f"    --- Unadjustment Details for {yf_symbol} around split ---"
                )
                log_df = unadj_df.loc[debug_dates_in_index].copy()
                log_df["original_adjusted"] = original_prices.reindex(
                    debug_dates_in_index
                )
                log_df["forward_factor"] = forward_split_factor.reindex(
                    debug_dates_in_index
                )
                log_df = log_df[
                    ["original_adjusted", "forward_factor", "unadjusted_price"]
                ]  # Keep relevant columns
                logging.debug(f"\n{log_df.to_string()}")
            else:
                logging.debug(f"    Could not find debug dates {debug_dates} in index.")
        # --- End Debug Logging ---

        if not unadj_df["unadjusted_price"].equals(
            original_prices.reindex_like(unadj_df["unadjusted_price"])
        ):
            unadjusted_count += 1

        # --- IMPORTANT: Return the correct column ---
        unadjusted_prices_yf[yf_symbol] = unadj_df[["unadjusted_price"]].rename(
            columns={"unadjusted_price": "price"}
        )
        # --- END Correct column return ---

    logging.info(
        f"--- Finished Price Unadjustment ({unadjusted_count} symbols processed with splits) ---"
    )
    return unadjusted_prices_yf


# --- Helper Functions for Point-in-Time Historical Calculation ---
def get_historical_price(
    symbol_key: str, target_date: date, prices_dict: Dict[str, pd.DataFrame]
) -> Optional[float]:
    """
    Gets the historical price for a symbol on a specific date, using forward fill for missing dates.

    Looks up the symbol (which can be a YF ticker like 'AAPL' or an FX pair like 'EUR=X')
    in the provided dictionary of price DataFrames. Finds the price for the `target_date`.
    If the exact date is missing, it uses the price from the most recent previous date available
    (forward fill).

    Args:
        symbol_key (str): The key (YF ticker or FX pair string) to look up in `prices_dict`.
        target_date (date): The specific date for which the price is required.
        prices_dict (Dict[str, pd.DataFrame]): A dictionary where keys are symbol keys (str) and
            values are DataFrames indexed by date objects, containing a 'price' column.

    Returns:
        Optional[float]: The price for the symbol on the target date (or the last available price
                         before it). Returns None if the symbol is not found, the date is before
                         the first available price point, or an error occurs during lookup.
    """
    if symbol_key not in prices_dict or prices_dict[symbol_key].empty:
        return None
    df = prices_dict[symbol_key]
    try:
        # Ensure df index is date objects
        if not all(isinstance(idx, date) for idx in df.index):
            # Attempt conversion if needed
            df.index = pd.to_datetime(df.index).date

        # Ensure target_date is a date object
        if not isinstance(target_date, date):
            return None

        # Efficient lookup using reindex with forward fill
        # Make sure index is sorted before reindexing
        if not df.index.is_monotonic_increasing:
            df.sort_index(inplace=True)
        combined_index = df.index.union([target_date])
        df_reindexed = df.reindex(combined_index, method="ffill")
        price = df_reindexed.loc[target_date]["price"]

        # If target_date is before the first price, ffill gives NaN, which is correct
        if pd.isna(price) and target_date < df.index.min():
            return None
        return float(price) if pd.notna(price) else None
    except KeyError:
        return None
    except Exception as e:
        logging.error(
            f"ERROR getting historical price for {symbol_key} on {target_date}: {e}"
        )  # Reduced verbosity
        return None


# --- Calculates historical rate TO/FROM via USD bridge ---
def get_historical_rate_via_usd_bridge(
    from_curr: str,
    to_curr: str,
    target_date: date,
    historical_fx_data: Dict[
        str, pd.DataFrame
    ],  # Expects {'EUR=X': df, 'THB=X': df, ...}
) -> float:
    """
    Gets the historical FX rate (units of to_curr per 1 unit of from_curr) for a specific date using a USD bridge.

    Calculates cross rates via USD using the provided historical FX data dictionary. It fetches
    the rate of `from_curr` per USD and `to_curr` per USD for the `target_date` (using forward fill
    via `get_historical_price`) and then computes the cross rate.

    Formula: Rate(TO/FROM) = Rate(TO/USD) / Rate(FROM/USD)

    Args:
        from_curr (str): The currency code to convert FROM.
        to_curr (str): The currency code to convert TO.
        target_date (date): The specific date for which the FX rate is required.
        historical_fx_data (Dict[str, pd.DataFrame]): A dictionary where keys are Yahoo Finance
            FX pair tickers against USD (e.g., 'EUR=X', 'JPY=X') and values are DataFrames
            indexed by date objects, containing a 'price' column (representing units of the
            other currency per 1 USD).

    Returns:
        float: The calculated historical conversion rate (units of `to_curr` for 1 unit of `from_curr`).
               Returns np.nan if `from_curr` equals `to_curr` (should return 1.0, fixed), if necessary
               intermediate rates (vs USD) cannot be found for the date, if inputs are invalid,
               or if a calculation error occurs (e.g., division by zero).
               Returns 1.0 if `from_curr` == `to_curr`.
    """
    if (
        not from_curr
        or not isinstance(from_curr, str)
        or not to_curr
        or not isinstance(to_curr, str)
        or not isinstance(target_date, date)
    ):
        logging.warning(
            f"Hist FX Bridge: Invalid input - from={from_curr}, to={to_curr}, date={target_date}"
        )
        return np.nan  # Return NaN for invalid input

    if from_curr == to_curr:
        return 1.0

    if not isinstance(historical_fx_data, dict):
        logging.warning(f"Hist FX Bridge: Invalid historical_fx_data type received.")
        return np.nan  # Return NaN for invalid dict type

    from_curr_upper = from_curr.upper()
    to_curr_upper = to_curr.upper()

    # --- Get intermediate rates: Currency per 1 USD for the target_date ---

    # Rate A (FROM / USD)
    rate_A_per_USD = np.nan
    if from_curr_upper == "USD":
        rate_A_per_USD = 1.0
    else:
        pair_A = f"{from_curr_upper}=X"  # e.g., THB=X
        price_A = get_historical_price(
            pair_A, target_date, historical_fx_data
        )  # Use helper
        if price_A is not None and pd.notna(price_A):
            rate_A_per_USD = price_A
        # else: rate_A_per_USD remains NaN

    # Rate B (TO / USD)
    rate_B_per_USD = np.nan
    if to_curr_upper == "USD":
        rate_B_per_USD = 1.0
    else:
        pair_B = f"{to_curr_upper}=X"  # e.g., EUR=X
        price_B = get_historical_price(
            pair_B, target_date, historical_fx_data
        )  # Use helper
        if price_B is not None and pd.notna(price_B):
            rate_B_per_USD = price_B
        # else: rate_B_per_USD remains NaN

    # --- Log intermediate rates for debugging ---
    # Optional: Add specific debug logging here if needed, similar to previous attempts
    # if from_curr_upper == 'THB' and to_curr_upper == 'USD':
    #    logging.debug(f"Hist FX Bridge Debug ({target_date}): Rate A ({from_curr}/USD)={rate_A_per_USD}, Rate B ({to_curr}/USD)={rate_B_per_USD}")

    # --- Calculate final rate: TO / FROM = (TO / USD) / (FROM / USD) ---
    rate_B_per_A = np.nan  # Initialize rate for B per A (TO / FROM)
    if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
        if (
            abs(rate_A_per_USD) > 1e-12
        ):  # Check denominator (FROM/USD) is not effectively zero
            try:
                rate_B_per_A = rate_B_per_USD / rate_A_per_USD
            except (ZeroDivisionError, TypeError, OverflowError):
                # Keep NaN if calculation fails
                pass
        # else: Denominator is zero/invalid

    # --- Final check and return ---
    if pd.isna(rate_B_per_A):
        # Log warning only if BOTH intermediate rates were initially found but calculation failed
        if pd.notna(rate_A_per_USD) and pd.notna(rate_B_per_USD):
            logging.warning(
                f"Hist FX Bridge: Calculation failed for {from_curr}->{to_curr} on {target_date} despite finding intermediate rates ({rate_A_per_USD}, {rate_B_per_USD})."
            )
        # else: Don't warn if intermediate rates weren't even found
        return np.nan  # Return NaN on any failure
    else:
        return float(rate_B_per_A)


def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame,  # Assumes 'Local Currency' column exists
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str],  # <-- Added
    default_currency: str,  # <-- Added
    processed_warnings: set,
) -> Tuple[float, bool]:  # Returns (net_flow, fx_lookup_failed)
    """
    Gets the historical FX rate (units of to_curr per 1 unit of from_curr) for a specific date using a USD bridge.

    Calculates cross rates via USD using the provided historical FX data dictionary. It fetches
    the rate of `from_curr` per USD and `to_curr` per USD for the `target_date` (using forward fill
    via `get_historical_price`) and then computes the cross rate.

    Formula: Rate(TO/FROM) = Rate(TO/USD) / Rate(FROM/USD)

    Args:
        from_curr (str): The currency code to convert FROM.
        to_curr (str): The currency code to convert TO.
        target_date (date): The specific date for which the FX rate is required.
        historical_fx_data (Dict[str, pd.DataFrame]): A dictionary where keys are Yahoo Finance
            FX pair tickers against USD (e.g., 'EUR=X', 'JPY=X') and values are DataFrames
            indexed by date objects, containing a 'price' column (representing units of the
            other currency per 1 USD).

    Returns:
        float: The calculated historical conversion rate (units of `to_curr` for 1 unit of `from_curr`).
               Returns np.nan if `from_curr` equals `to_curr` (should return 1.0, fixed), if necessary
               intermediate rates (vs USD) cannot be found for the date, if inputs are invalid,
               or if a calculation error occurs (e.g., division by zero).
               Returns 1.0 if `from_curr` == `to_curr`.
    """
    fx_lookup_failed = False
    net_flow_target_curr = 0.0
    # Filter for transactions ON the target date
    daily_tx = transactions_df[transactions_df["Date"].dt.date == target_date].copy()
    if daily_tx.empty:
        return 0.0, False

    # Filter ONLY for explicit $CASH Deposit/Withdrawal transactions
    # These represent money entering or leaving the portfolio boundary.
    external_flow_types = ["deposit", "withdrawal"]  # Standard external flows
    cash_flow_tx = daily_tx[
        (daily_tx["Symbol"] == CASH_SYMBOL_CSV)
        & (daily_tx["Type"].isin(external_flow_types))
    ].copy()

    if cash_flow_tx.empty:
        return 0.0, False  # No external flows on this day

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

        # Determine flow direction based ONLY on deposit/withdrawal
        if tx_type == "deposit":
            flow_local = abs(qty) - commission_local  # Cash IN (+)
        elif tx_type == "withdrawal":
            flow_local = -abs(qty) - commission_local  # Cash OUT (-)
        # Other types ('buy $CASH', 'sell $CASH') are ignored by the filter above

        # Convert flow to target currency
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
            break  # Check if loop was broken

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


# --- Point-in-Time Portfolio Value Calculation (Uses UNADJUSTED for portfolio, needed by worker) ---
def _calculate_portfolio_value_at_date_unadjusted(
    target_date: date,
    transactions_df: pd.DataFrame,  # Full transaction set for the selected accounts scope
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,  # Note: processed_warnings isn't really used here currently
) -> Tuple[float, bool]:
    """
    Calculates the total portfolio market value for a specific date using UNADJUSTED historical prices.

    This function is crucial for Time-Weighted Return (TWR) calculations. It simulates the portfolio
    state on a given `target_date` by processing all transactions up to that date. It recalculates
    holding quantities, applying splits correctly across all relevant accounts as they occurred.
    It then determines the price of each holding on the `target_date` using the provided *unadjusted*
    historical prices (falling back to last transaction price if needed). Cash balances are also included.
    Finally, it converts the local market value of each position to the `target_currency` using
    historical FX rates and sums them up to get the total portfolio value for that day.

    Args:
        target_date (date): The specific date for which to calculate the portfolio value.
        transactions_df (pd.DataFrame): The DataFrame containing all transactions relevant to the
                                        portfolio scope being analyzed (already filtered by account
                                        inclusion/exclusion). Must include standard transaction columns.
        historical_prices_yf_unadjusted (Dict[str, pd.DataFrame]): Dictionary mapping Yahoo Finance tickers
            to DataFrames containing UNADJUSTED historical prices (indexed by date).
        historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping Yahoo Finance FX pair tickers (vs USD)
            to DataFrames containing historical rates (indexed by date).
        target_currency (str): The currency in which to report the total portfolio value.
        internal_to_yf_map (Dict[str, str]): Mapping from internal symbols to Yahoo Finance tickers.
        account_currency_map (Dict[str, str]): Mapping from account name to local currency.
        default_currency (str): Default currency.
        processed_warnings (set): A set for tracking warnings (currently unused).

    Returns:
        Tuple[float, bool]:
            - float: The total market value of the portfolio in the `target_currency` on the `target_date`.
                     Returns np.nan if any critical price or FX lookup fails for any holding with non-zero quantity.
            - bool: any_lookup_nan_on_date - True if any required price or FX lookup failed, False otherwise.
    """
    # --- Add Debug Flag Check ---
    IS_DEBUG_DATE = (
        target_date == HISTORICAL_DEBUG_DATE_VALUE
        if "HISTORICAL_DEBUG_DATE_VALUE" in globals()
        else False
    )
    if IS_DEBUG_DATE:
        logging.debug(f"--- DEBUG VALUE CALC for {target_date} ---")
    # --- End Debug Flag Check ---

    # Filter transactions ONCE up to the target date
    transactions_til_date = transactions_df[
        transactions_df["Date"].dt.date <= target_date
    ].copy()
    if transactions_til_date.empty:
        if IS_DEBUG_DATE:
            logging.debug(f"  No transactions found up to {target_date}.")
        return 0.0, False

    # --- Recalculate Holdings Quantity up to target_date ---
    holdings: Dict[Tuple[str, str], Dict] = {}
    for index, row in transactions_til_date.iterrows():
        symbol = str(row.get("Symbol", "UNKNOWN")).strip()
        account = str(row.get("Account", "Unknown"))  # Account from this row
        local_currency_from_row = str(row.get("Local Currency", default_currency))
        holding_key_from_row = (symbol, account)
        tx_type = str(row.get("Type", "UNKNOWN_TYPE")).lower().strip()
        tx_date_row = row["Date"].date()  # Date of the current transaction row

        # Initialize holding if needed (for the account in this specific row)
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
            # Overwrite currency if it differs (or log warning)
            holdings[holding_key_from_row]["local_currency"] = local_currency_from_row
            if IS_DEBUG_DATE:
                logging.debug(
                    f"  WARN (Value Calc): Currency overwritten for {holding_key_from_row} to {local_currency_from_row}"
                )

        # --- Apply Transactions (with Corrected Split Logic) ---
        if symbol == CASH_SYMBOL_CSV:
            continue  # Skip cash quantity updates here

        try:
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            split_ratio = pd.to_numeric(
                row.get("Split Ratio"), errors="coerce"
            )  # Read ratio if present

            # --- Corrected Split Handling ---
            if tx_type in ["split", "stock split"]:
                if pd.notna(split_ratio) and split_ratio > 0:
                    # Apply split to ALL accounts holding this symbol
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
                                # Adjust short qty if needed (optional here if not tracking short value)
                                # if old_qty < -1e-9 and symbol in SHORTABLE_SYMBOLS: ...
                                if abs(h_data["qty"]) < 1e-9:
                                    h_data["qty"] = 0.0
                else:
                    if IS_DEBUG_DATE:
                        logging.warning(
                            f"  Skipping invalid split ratio ({split_ratio}) for {symbol} on {tx_date_row}"
                        )
                continue  # Don't process split row further

            # --- Standard Buy/Sell/Short ---
            holding_to_update = holdings.get(holding_key_from_row)
            if not holding_to_update:
                continue  # Should not happen if initialized

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
            # Dividend, Fees don't change quantity

        except Exception as e_h:
            if IS_DEBUG_DATE:
                logging.error(
                    f"      ERROR processing holding qty for {holding_key_from_row} on row index {index}: {e_h}"
                )
            pass
    # --- End Quantity Recalculation Loop ---

    # --- Calculate Cash Balances (as before) ---
    cash_summary: Dict[str, Dict] = {}
    cash_transactions = transactions_til_date[
        transactions_til_date["Symbol"] == CASH_SYMBOL_CSV
    ].copy()
    if not cash_transactions.empty:

        def get_signed_quantity_cash(row):
            type_lower = str(row.get("Type", "")).lower()
            qty = pd.to_numeric(row.get("Quantity"), errors="coerce")
            return (
                0.0
                if pd.isna(qty)
                else (
                    abs(qty)
                    if type_lower in ["buy", "deposit"]
                    else (-abs(qty) if type_lower in ["sell", "withdrawal"] else 0.0)
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

    # --- Combine stock and cash positions ---
    all_positions: Dict[Tuple[str, str], Dict] = {
        **holdings,
        **{(CASH_SYMBOL_CSV, acc): data for acc, data in cash_summary.items()},
    }

    # --- Calculate Total Market Value (uses recalculated quantities) ---
    total_market_value_display_curr_agg = 0.0
    any_lookup_nan_on_date = False
    if IS_DEBUG_DATE:
        logging.debug(
            f"  Value Aggregation Start - Combined Positions ({len(all_positions)}): {list(all_positions.keys())}"
        )

    for (internal_symbol, account), data in all_positions.items():
        current_qty = data.get("qty", 0.0)  # Use the QTY calculated in the loop above
        local_currency = data.get("local_currency", default_currency)
        is_stock = data.get("is_stock", internal_symbol != CASH_SYMBOL_CSV)

        DO_DETAILED_LOG = IS_DEBUG_DATE  # Simplified debug log condition for value part
        if DO_DETAILED_LOG:
            logging.debug(
                f"    Value Agg: Processing {internal_symbol}/{account}, Qty: {current_qty:.4f}"
            )

        if abs(current_qty) < 1e-9:
            continue

        # --- Step 1: Get FX Rate ---
        fx_rate = get_historical_rate_via_usd_bridge(
            local_currency, target_currency, target_date, historical_fx_yf
        )  # Use the corrected FX function
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

        # --- Step 2: Determine Local Price (with fallback) ---
        current_price_local = np.nan
        force_fallback = internal_symbol in YFINANCE_EXCLUDED_SYMBOLS
        price_val = None
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
        # Fallback Logic
        if (
            pd.isna(current_price_local) or force_fallback
        ) and is_stock:  # Only fallback for stocks
            try:
                # ... (Fallback TX lookup logic - unchanged) ...
                fallback_tx = transactions_df[
                    (transactions_df["Symbol"] == internal_symbol)
                    & (transactions_df["Account"] == account)
                    & (transactions_df["Price/Share"].notna())
                    & (transactions_df["Price/Share"] > 1e-9)
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

        # --- Step 3: Check if Price Determination Failed ---
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

        # --- Step 4/5: Calculate and Aggregate Market Value ---
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


# --- Worker function for multiprocessing ---
def _calculate_daily_metrics_worker(
    eval_date: date,
    # --- Passed via functools.partial ---
    transactions_df: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],
    target_currency: str,
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    benchmark_symbols_yf: List[str],
) -> Optional[Dict]:
    """
    Worker function (for multiprocessing) to calculate key metrics for a single date.

    Designed to be used with `multiprocessing.Pool`. For a given evaluation date (`eval_date`),
    it calculates:
    1. Total portfolio market value using unadjusted prices (`_calculate_portfolio_value_at_date_unadjusted`).
    2. Net external cash flow for the day (`_calculate_daily_net_cash_flow`).
    3. Prices for specified benchmark symbols using adjusted prices (`get_historical_price`).

    Args:
        eval_date (date): The specific date for which to calculate metrics.
        transactions_df (pd.DataFrame): Passed via `functools.partial`. Filtered transactions.
        historical_prices_yf_unadjusted (Dict): Passed via `functools.partial`. Unadjusted prices.
        historical_prices_yf_adjusted (Dict): Passed via `functools.partial`. Adjusted prices (for benchmarks).
        historical_fx_yf (Dict): Passed via `functools.partial`. Historical FX rates vs USD.
        target_currency (str): Passed via `functools.partial`. Target currency for value/flow.
        internal_to_yf_map (Dict): Passed via `functools.partial`. Symbol mapping.
        account_currency_map (Dict): Passed via `functools.partial`. Account currency mapping.
        default_currency (str): Passed via `functools.partial`. Default currency.
        benchmark_symbols_yf (List[str]): Passed via `functools.partial`. List of benchmark YF tickers.

    Returns:
        Optional[Dict]: A dictionary containing the calculated metrics for the `eval_date`:
            {'Date': date, 'value': float|nan, 'net_flow': float|nan,
             'value_lookup_failed': bool, 'flow_lookup_failed': bool, 'bench_lookup_failed': bool,
             '{Bench1} Price': float|nan, ...}.
            Returns a dictionary with NaNs and error flags set if a critical error occurs within the worker.
            May return None in catastrophic failure scenarios (though designed to return dict with NaNs).
    """
    # --- Add initial log ---
    # logging.debug(f"Worker Start: Processing date {eval_date}") # Can be very verbose
    # --- CONFIGURE LOGGING WITHIN WORKER (for debugging) ---
    # This ensures messages from this worker process are output.
    # Use force=True (Python 3.8+) to override potential root logger setup issues.
    # Note: This simple setup might cause issues if you have complex file logging
    #       in the main process that you expect workers to use directly.
    #       For production, consider using a QueueHandler.
    logging.basicConfig(
        level=LOGGING_LEVEL,  # Match the desired level
        format="%(asctime)s [%(levelname)-8s] PID:%(process)d {%(module)s:%(lineno)d} %(message)s",  # Add PID
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,  # Override existing config if any
    )
    # --- END WORKER LOGGING CONFIG ---

    try:
        dummy_warnings_set = (
            set()
        )  # Local scope for warnings within this worker instance

        # --- Log before calling value calculation ---
        # logging.debug(f"Worker {eval_date}: Calling _calculate_portfolio_value_at_date_unadjusted...")

        # 1. Calculate Portfolio Value (Pass map/default)
        portfolio_value, val_lookup_failed = (
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
            )
        )

        # --- Log after value calculation ---
        # logging.debug(f"Worker {eval_date}: Value calculated: {portfolio_value}, LookupFailed: {val_lookup_failed}")

        # 2. Calculate Net External Cash Flow (Pass map/default)
        # Handle case where value failed - no point calculating flow if value is NaN
        if pd.isna(portfolio_value):
            # logging.debug(f"Worker {eval_date}: Portfolio value is NaN, skipping cash flow calc.")
            net_cash_flow = np.nan
            flow_lookup_failed = val_lookup_failed  # Inherit failure status
        else:
            # logging.debug(f"Worker {eval_date}: Calling _calculate_daily_net_cash_flow...")
            net_cash_flow, flow_lookup_failed = _calculate_daily_net_cash_flow(
                eval_date,
                transactions_df,
                target_currency,
                historical_fx_yf,
                account_currency_map,
                default_currency,
                dummy_warnings_set,
            )
            # logging.debug(f"Worker {eval_date}: Flow calculated: {net_cash_flow}, LookupFailed: {flow_lookup_failed}")

        # 3. Get benchmark prices (unchanged)
        # logging.debug(f"Worker {eval_date}: Getting benchmark prices...")
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
        # logging.debug(f"Worker {eval_date}: Benchmarks done. LookupFailed: {bench_lookup_failed}")

        # 4. Assemble result (unchanged)
        result_row = {
            "Date": eval_date,
            "value": portfolio_value,
            "net_flow": net_cash_flow,
            "value_lookup_failed": val_lookup_failed,
            "flow_lookup_failed": flow_lookup_failed,  # Combine value/flow failure?
            "bench_lookup_failed": bench_lookup_failed,
        }
        result_row.update(benchmark_prices)
        # logging.debug(f"Worker {eval_date}: Result assembled.")
        return result_row

    except Exception as e:  # Error handling for the *entire* worker function
        # --- Make this error logging more prominent ---
        logging.critical(
            f"!!! CRITICAL ERROR in worker process for date {eval_date}: {e}"
        )
        logging.exception("Worker Traceback:")  # Log the full traceback
        # --- End prominent logging ---

        failed_row = {"Date": eval_date, "value": np.nan, "net_flow": np.nan}
        for bm_symbol in benchmark_symbols_yf:
            failed_row[f"{bm_symbol} Price"] = np.nan
        failed_row["value_lookup_failed"] = True
        failed_row["flow_lookup_failed"] = True
        failed_row["bench_lookup_failed"] = True
        # Add a specific error flag
        failed_row["worker_error"] = True
        failed_row["worker_error_msg"] = str(e)  # Store error message if needed later
        return failed_row


def _prepare_historical_inputs(
    transactions_csv_file: str,
    account_currency_map: Dict,
    default_currency: str,
    include_accounts: Optional[List[str]],
    exclude_accounts: Optional[List[str]],
    start_date: date,
    end_date: date,
    benchmark_symbols_yf: List[str],  # Cleaned list
    display_currency: str,
    current_hist_version: str = "v10",  # Pass version for cache keys
    raw_cache_prefix: str = HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,  # Pass prefixes
    daily_cache_prefix: str = DAILY_RESULTS_CACHE_PATH_PREFIX,
) -> Tuple[
    Optional[pd.DataFrame],  # transactions_df_effective
    Optional[pd.DataFrame],  # original_transactions_df
    Set[int],
    Dict[int, str],  # ignored_indices, ignored_reasons
    List[str],  # all_available_accounts_list
    List[str],  # included_accounts_list_sorted
    List[str],  # excluded_accounts_list_sorted
    List[str],  # symbols_for_stocks_and_benchmarks_yf
    List[str],  # fx_pairs_for_api_yf
    Dict[str, str],  # internal_to_yf_map
    Dict[str, str],  # yf_to_internal_map_hist
    Dict[str, List[Dict]],  # splits_by_internal_symbol
    str,  # raw_data_cache_file
    str,  # raw_data_cache_key
    Optional[str],  # daily_results_cache_file
    Optional[str],  # daily_results_cache_key
    str,  # filter_desc
]:
    """
    Prepares all necessary inputs for the historical performance calculation.

    This function centralizes the initial setup steps:
    1. Loads and cleans all transactions using `load_and_clean_transactions`.
    2. Filters transactions based on `include_accounts` and `exclude_accounts`.
    3. Extracts split information from the transactions.
    4. Determines the unique set of stock/ETF symbols and currencies involved in the filtered transactions.
    5. Generates the lists of Yahoo Finance tickers required for fetching stock/benchmark prices and FX rates.
    6. Creates mappings between internal symbols and Yahoo Finance tickers.
    7. Generates cache keys and filenames for both raw historical data and calculated daily results,
       incorporating file hashes and configuration parameters to ensure cache validity.

    Args:
        transactions_csv_file (str): Path to the transactions CSV.
        account_currency_map (Dict): Account to currency mapping.
        default_currency (str): Default currency.
        include_accounts (Optional[List[str]]): Accounts to include (None=all).
        exclude_accounts (Optional[List[str]]): Accounts to exclude.
        start_date (date): Analysis start date.
        end_date (date): Analysis end date.
        benchmark_symbols_yf (List[str]): Cleaned list of benchmark YF tickers.
        display_currency (str): Target currency for results.
        current_hist_version (str): Version string for daily results cache key.
        raw_cache_prefix (str): Prefix for the raw data cache filename.
        daily_cache_prefix (str): Prefix for the daily results cache filename.

    Returns:
        Tuple containing various prepared inputs required by subsequent historical calculation steps.
        Returns a tuple of Nones/empty collections if loading/cleaning fails critically.
        Specific elements include: filtered transactions DataFrame, original transactions DataFrame,
        ignored transaction info, account lists, symbol/FX lists for fetching, symbol maps,
        split data, cache filenames, cache keys, and a filter description string.
    """
    logging.info("Preparing inputs for historical calculation...")

    # Initialize return values for failure cases
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

    # --- 1. Load & Clean ALL Transactions ---
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
            "RROR in _prepare_historical_inputs: Failed to load/clean transactions."
        )
        return empty_tuple_return  # Return empties/None on failure

    # --- Get available accounts ---
    all_available_accounts_list = []
    if "Account" in all_transactions_df.columns:
        all_available_accounts_list = sorted(
            all_transactions_df["Account"].unique().tolist()
        )

    # --- 1b. Filter Transactions ---
    transactions_df_effective = pd.DataFrame()
    available_accounts_set = set(all_available_accounts_list)
    included_accounts_list_sorted = []
    excluded_accounts_list_sorted = []
    filter_desc = "All Accounts"  # Default description

    # Apply inclusion filter
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
            return empty_tuple_return  # Or maybe return partial data? For now, return empty.
        transactions_df_included = all_transactions_df[
            all_transactions_df["Account"].isin(valid_include_accounts)
        ].copy()
        included_accounts_list_sorted = sorted(valid_include_accounts)
        filter_desc = f"Included: {', '.join(included_accounts_list_sorted)}"

    # Apply exclusion filter
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
            # Update description
            if include_accounts:
                filter_desc += (
                    f" (Excluding: {', '.join(excluded_accounts_list_sorted)})"
                )
            else:
                filter_desc = f"All Accounts (Excluding: {', '.join(excluded_accounts_list_sorted)})"
        else:  # No valid exclusions
            transactions_df_effective = transactions_df_included.copy()

    if transactions_df_effective.empty:
        logging.warning(
            "WARN in _prepare_historical_inputs: No transactions remain after filtering."
        )
        # Return effective df as empty, but provide other info if possible
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

    # --- Extract Split Information (use original full df for splits) ---
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

    # --- Determine Required Symbols & FX Pairs (use EFFECTIVE df) ---
    all_symbols_internal = list(set(transactions_df_effective["Symbol"].unique()))
    symbols_to_fetch_yf_portfolio = []
    internal_to_yf_map = {}
    yf_to_internal_map_hist = {}
    for internal_sym in all_symbols_internal:
        if internal_sym == CASH_SYMBOL_CSV:
            continue
        yf_sym = map_to_yf_symbol(internal_sym)
        # Assumes map_to_yf_symbol helper exists
        if yf_sym:
            symbols_to_fetch_yf_portfolio.append(yf_sym)
            internal_to_yf_map[internal_sym] = yf_sym
            yf_to_internal_map_hist[yf_sym] = internal_sym
    symbols_to_fetch_yf_portfolio = sorted(list(set(symbols_to_fetch_yf_portfolio)))
    symbols_for_stocks_and_benchmarks_yf = sorted(
        list(set(symbols_to_fetch_yf_portfolio + benchmark_symbols_yf))
    )  # Use cleaned benchmark list passed in

    all_currencies_in_tx = set(transactions_df_effective["Local Currency"].unique())
    all_currencies_needed = all_currencies_in_tx.union(
        {display_currency, default_currency}
    )  # Add display and default
    all_currencies_needed.discard(None)  # Remove None if present
    all_currencies_needed.discard("N/A")  # Remove N/A

    # Generate necessary YF FX tickers (CURRENCY=X format)
    fx_pairs_for_api_yf = set()
    for curr in all_currencies_needed:
        if curr != "USD":
            fx_pairs_for_api_yf.add(f"{curr}=X")
    fx_pairs_for_api_yf = sorted(list(fx_pairs_for_api_yf))

    # --- Generate Cache Keys & Filenames ---
    raw_data_cache_file = (
        f"{raw_cache_prefix}_{start_date.isoformat()}_{end_date.isoformat()}.json"
    )
    raw_data_cache_key = f"ADJUSTED_v7::{start_date.isoformat()}::{end_date.isoformat()}::{'_'.join(sorted(symbols_for_stocks_and_benchmarks_yf))}::{'_'.join(fx_pairs_for_api_yf)}"  # Use correct FX list

    daily_results_cache_file = None
    daily_results_cache_key = None
    try:
        tx_file_hash = _get_file_hash(transactions_csv_file)  # Assumes helper exists
        acc_map_str = json.dumps(account_currency_map, sort_keys=True)
        included_accounts_str = json.dumps(included_accounts_list_sorted)
        excluded_accounts_str = json.dumps(excluded_accounts_list_sorted)
        daily_results_cache_key = f"DAILY_RES_{current_hist_version}::{start_date.isoformat()}::{end_date.isoformat()}::{tx_file_hash}::{'_'.join(sorted(benchmark_symbols_yf))}::{display_currency}::{acc_map_str}::{default_currency}::{included_accounts_str}::{excluded_accounts_str}"
        cache_key_hash = hashlib.sha256(daily_results_cache_key.encode()).hexdigest()[
            :16
        ]  # Use first 16 chars of hash
        daily_results_cache_file = f"{daily_cache_prefix}_{cache_key_hash}.json"
    except Exception as e_key:
        logging.warning(
            f"Hist Prep WARN: Could not generate daily results cache key/filename: {e_key}."
        )
        # Keep them as None

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


def _load_or_fetch_raw_historical_data(
    symbols_to_fetch_yf: List[str],  # Combined list of stock, benchmark YF tickers
    fx_pairs_to_fetch_yf: List[str],  # List of YF FX tickers (e.g., ['JPY=X', 'EUR=X'])
    start_date: date,
    end_date: date,
    use_raw_data_cache: bool,
    raw_data_cache_file: str,
    raw_data_cache_key: str,  # Key to validate cache content
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], bool]:
    """
    Loads adjusted historical price/FX data from cache if valid, otherwise fetches fresh data.

    Checks for a cache file containing raw historical data (adjusted prices and FX rates vs USD).
    Validates the cache using a generated `raw_data_cache_key`. If the cache is valid and complete,
    it loads the data. Otherwise, it identifies missing symbols/FX pairs and fetches them using
    `fetch_yf_historical`. If data was fetched, the cache file is updated.

    Args:
        symbols_to_fetch_yf (List[str]): List of YF stock/benchmark tickers required.
        fx_pairs_to_fetch_yf (List[str]): List of YF FX tickers (e.g., 'JPY=X') required.
        start_date (date): Start date for historical data.
        end_date (date): End date for historical data.
        use_raw_data_cache (bool): Flag to enable reading/writing the raw data cache.
        raw_data_cache_file (str): Path to the raw historical data cache file.
        raw_data_cache_key (str): Cache validation key generated by `_prepare_historical_inputs`.

    Returns:
        Tuple containing:
        - historical_prices_yf_adjusted (Dict[str, pd.DataFrame]): Dictionary mapping YF stock/benchmark
            tickers to DataFrames containing adjusted historical prices (indexed by date).
        - historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers (e.g., 'JPY=X')
            to DataFrames containing historical rates vs USD (indexed by date).
        - fetch_failed (bool): True if fetching/loading critical data (especially FX rates) failed,
                               False otherwise.
    """
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
    historical_fx_yf: Dict[str, pd.DataFrame] = {}
    cache_valid_raw = False
    fetch_failed = False  # Track if essential fetching fails

    # --- 1. Try Loading Cache ---
    if (
        use_raw_data_cache
        and raw_data_cache_file
        and os.path.exists(raw_data_cache_file)
    ):
        logging.info(
            f"Hist Raw: Attempting to load raw data cache: {raw_data_cache_file}"
        )
        try:
            with open(raw_data_cache_file, "r") as f:
                cached_data = json.load(f)
            if cached_data.get("cache_key") == raw_data_cache_key:
                deserialization_errors = 0
                # Load Prices
                cached_prices = cached_data.get("historical_prices", {})
                for symbol in symbols_to_fetch_yf:
                    data_json = cached_prices.get(symbol)
                    df = None
                    if data_json:
                        try:
                            df = pd.read_json(
                                StringIO(data_json),
                                orient="split",
                                dtype={"price": float},
                            )
                            df.index = pd.to_datetime(
                                df.index, errors="coerce"
                            ).date  # Convert index to date objects
                            df = df.dropna(subset=["price"])  # Ensure price is valid
                            df = df[pd.notnull(df.index)]  # Drop NaT indices
                            if not df.empty:
                                historical_prices_yf_adjusted[symbol] = df.sort_index()
                            else:
                                historical_prices_yf_adjusted[symbol] = (
                                    pd.DataFrame()
                                )  # Store empty if no valid data
                        except Exception as e_deser:
                            logging.debug(
                                f"DEBUG: Error deserializing cached price for {symbol}: {e_deser}"
                            )  # Optional
                            deserialization_errors += 1
                            historical_prices_yf_adjusted[symbol] = pd.DataFrame()
                    else:
                        historical_prices_yf_adjusted[symbol] = (
                            pd.DataFrame()
                        )  # Symbol missing from cache

                # Load FX Rates (keyed by JPY=X, etc.)
                cached_fx = cached_data.get("historical_fx_rates", {})
                for pair in fx_pairs_to_fetch_yf:
                    data_json = cached_fx.get(pair)
                    df = None
                    if data_json:
                        try:
                            df = pd.read_json(
                                StringIO(data_json),
                                orient="split",
                                dtype={"price": float},
                            )
                            df.index = pd.to_datetime(
                                df.index, errors="coerce"
                            ).date  # Convert index to date objects
                            df = df.dropna(subset=["price"])
                            df = df[pd.notnull(df.index)]
                            if not df.empty:
                                historical_fx_yf[pair] = df.sort_index()
                            else:
                                historical_fx_yf[pair] = pd.DataFrame()
                        except Exception as e_deser_fx:
                            logging.debug(
                                f"DEBUG: Error deserializing cached FX for {pair}: {e_deser_fx}"
                            )  # Optional
                            deserialization_errors += 1
                            historical_fx_yf[pair] = pd.DataFrame()
                    else:
                        historical_fx_yf[pair] = pd.DataFrame()  # Pair missing

                # Validate Cache Completeness and Types
                all_symbols_loaded = all(
                    s in historical_prices_yf_adjusted for s in symbols_to_fetch_yf
                )
                all_fx_loaded = all(p in historical_fx_yf for p in fx_pairs_to_fetch_yf)
                prices_are_dict = isinstance(historical_prices_yf_adjusted, dict)
                fx_are_dict = isinstance(historical_fx_yf, dict)

                if (
                    all_symbols_loaded
                    and all_fx_loaded
                    and prices_are_dict
                    and fx_are_dict
                    and deserialization_errors == 0
                ):
                    logging.info("Hist Raw: Cache valid and complete.")
                    cache_valid_raw = True
                else:  # If anything failed or is wrong type
                    logging.warning(
                        f"Hist WARN: RAW cache load failed validation (Symbols loaded: {all_symbols_loaded}, FX loaded: {all_fx_loaded}, Price type OK: {prices_are_dict}, FX type OK: {fx_are_dict}, Errors: {deserialization_errors}). Refetching if needed."
                    )
                    cache_valid_raw = False  # Force refetch if validation fails
                    # Reset dictionaries if types were wrong
                    if not prices_are_dict:
                        historical_prices_yf_adjusted = {}
                    if not fx_are_dict:
                        historical_fx_yf = {}
            else:
                logging.warning(f"Hist Raw: Cache key mismatch. Ignoring cache.")
        except Exception as e:
            logging.error(
                f"Error reading hist RAW cache {raw_data_cache_file}: {e}. Ignoring cache."
            )
            historical_prices_yf_adjusted = {}  # Clear potentially corrupted data
            historical_fx_yf = {}

    # --- 2. Fetch Missing Data if Cache Invalid/Incomplete ---
    if not cache_valid_raw:
        logging.info("Hist Raw: Fetching data from Yahoo Finance...")
        # Determine which symbols *still* need fetching (if cache was partial)
        symbols_needing_fetch = [
            s
            for s in symbols_to_fetch_yf
            if s not in historical_prices_yf_adjusted
            or historical_prices_yf_adjusted[s].empty
        ]
        fx_needing_fetch = [
            p
            for p in fx_pairs_to_fetch_yf
            if p not in historical_fx_yf or historical_fx_yf[p].empty
        ]

        if symbols_needing_fetch:
            logging.info(
                f"Hist Raw: Fetching {len(symbols_needing_fetch)} stock/benchmark symbols..."
            )
            fetched_stock_data = fetch_yf_historical(
                symbols_needing_fetch, start_date, end_date
            )  # Assumes helper exists
            historical_prices_yf_adjusted.update(fetched_stock_data)  # Add fetched data
        else:
            logging.info(
                "Hist Raw: All stock/benchmark symbols found in cache or not needed."
            )

        if fx_needing_fetch:
            logging.info(f"Hist Raw: Fetching {len(fx_needing_fetch)} FX pairs...")
            # Use the same fetcher, assuming it handles CURRENCY=X tickers
            fetched_fx_data = fetch_yf_historical(
                fx_needing_fetch, start_date, end_date
            )
            historical_fx_yf.update(fetched_fx_data)  # Add fetched data
        else:
            logging.info("Hist Raw: All FX pairs found in cache or not needed.")

        # --- Validation after fetch ---
        final_symbols_missing = [
            s
            for s in symbols_to_fetch_yf
            if s not in historical_prices_yf_adjusted
            or historical_prices_yf_adjusted[s].empty
        ]
        final_fx_missing = [
            p
            for p in fx_pairs_to_fetch_yf
            if p not in historical_fx_yf or historical_fx_yf[p].empty
        ]

        if final_symbols_missing:
            logging.warning(
                f"Hist WARN: Failed to fetch/load adjusted prices for: {', '.join(final_symbols_missing)}"
            )
            # Decide if this is critical - depends if any portfolio holdings use these
            # For now, let's flag fetch_failed only if essential FX is missing
        if final_fx_missing:
            logging.warning(
                f"Hist WARN: Failed to fetch/load FX rates for: {', '.join(final_fx_missing)}"
            )
            # If ANY required FX pair is missing, mark as failed for safety
            fetch_failed = True

        # --- 3. Update Cache if Fetch Occurred ---
        if use_raw_data_cache and (
            symbols_needing_fetch or fx_needing_fetch
        ):  # Only save if we fetched something
            logging.info(
                f"Hist Raw: Saving updated raw data to cache: {raw_data_cache_file}"
            )
            # Prepare JSON-serializable data
            prices_to_cache = {
                symbol: df.to_json(orient="split", date_format="iso")
                for symbol, df in historical_prices_yf_adjusted.items()
                if not df.empty  # Only cache non-empty
            }
            fx_to_cache = {
                pair: df.to_json(orient="split", date_format="iso")
                for pair, df in historical_fx_yf.items()
                if not df.empty  # Only cache non-empty
            }
            cache_content = {
                "cache_key": raw_data_cache_key,
                "timestamp": datetime.now().isoformat(),
                "historical_prices": prices_to_cache,
                "historical_fx_rates": fx_to_cache,
            }
            try:
                cache_dir_raw = os.path.dirname(raw_data_cache_file)
                if cache_dir_raw:
                    os.makedirs(cache_dir_raw, exist_ok=True)
                with open(raw_data_cache_file, "w") as f:
                    json.dump(cache_content, f, indent=2)
            except Exception as e:
                logging.error(f"Error writing hist RAW cache: {e}")
        elif not use_raw_data_cache:
            logging.info("Hist Raw: Caching disabled, skipping save.")

    # --- 4. Final Check and Return ---
    # Check again if critical data is missing after cache/fetch attempts
    if (
        not fetch_failed and fx_pairs_to_fetch_yf
    ):  # Re-check FX only if pairs were needed
        if any(
            p not in historical_fx_yf or historical_fx_yf[p].empty
            for p in fx_pairs_to_fetch_yf
        ):
            logging.error("Hist ERROR: Critical FX data missing after final check.")
            fetch_failed = True

    if not fetch_failed and symbols_to_fetch_yf:  # Re-check stocks
        if any(
            s not in historical_prices_yf_adjusted
            or historical_prices_yf_adjusted[s].empty
            for s in symbols_to_fetch_yf
        ):
            # Allow proceeding but maybe warn later if specific symbols are needed and missing
            logging.warning(
                "Hist WARN: Some stock/benchmark data missing after final check."
            )

    return historical_prices_yf_adjusted, historical_fx_yf, fetch_failed


def _load_or_calculate_daily_results(
    use_daily_results_cache: bool,
    daily_results_cache_file: Optional[str],
    daily_results_cache_key: Optional[str],
    start_date: date,
    end_date: date,
    # Arguments needed by the worker function & prep:
    transactions_df_effective: pd.DataFrame,
    historical_prices_yf_unadjusted: Dict[str, pd.DataFrame],
    historical_prices_yf_adjusted: Dict[str, pd.DataFrame],
    historical_fx_yf: Dict[str, pd.DataFrame],  # Keyed by 'JPY=X' etc.
    display_currency: str,  # Target currency for calculations
    internal_to_yf_map: Dict[str, str],
    account_currency_map: Dict[str, str],
    default_currency: str,
    clean_benchmark_symbols_yf: List[str],
    num_processes: Optional[int] = None,
    current_hist_version: str = "v10",  # For logging
    filter_desc: str = "All Accounts",  # For logging
) -> Tuple[pd.DataFrame, bool, str]:  # Returns daily_df, cache_was_valid, status_update
    """
    Loads calculated daily results from cache or calculates them using parallel processing.

    Checks for a cache file containing pre-calculated daily metrics (portfolio value, net flow,
    benchmark prices, daily return, daily gain). Validates the cache using `daily_results_cache_key`.
    If the cache is valid, loads the data.
    Otherwise, it determines the relevant market days within the date range, sets up a
    multiprocessing pool, and calls the `_calculate_daily_metrics_worker` for each date in parallel.
    After calculation, it computes daily returns and gains based on the daily values and flows.
    If data was calculated, it saves the results to the cache file.

    Args:
        use_daily_results_cache (bool): Flag to enable reading/writing the daily results cache.
        daily_results_cache_file (Optional[str]): Path to the daily results cache file.
        daily_results_cache_key (Optional[str]): Cache validation key.
        start_date (date): Analysis start date.
        end_date (date): Analysis end date.
        transactions_df_effective (pd.DataFrame): Filtered transactions.
        historical_prices_yf_unadjusted (Dict): Unadjusted prices.
        historical_prices_yf_adjusted (Dict): Adjusted prices (for benchmarks).
        historical_fx_yf (Dict): Historical FX rates vs USD.
        display_currency (str): Target currency for calculations.
        internal_to_yf_map (Dict): Symbol mapping.
        account_currency_map (Dict): Account currency mapping.
        default_currency (str): Default currency.
        clean_benchmark_symbols_yf (List[str]): List of benchmark YF tickers.
        num_processes (Optional[int]): Number of worker processes for parallel calculation.
        current_hist_version (str): Version string for logging/cache key.
        filter_desc (str): Description of the account filter scope for logging.

    Returns:
        Tuple containing:
        - daily_df (pd.DataFrame): DataFrame indexed by date, containing calculated daily metrics:
            'value', 'net_flow', 'daily_gain', 'daily_return', and benchmark prices ('{Bench} Price').
            Returns an empty DataFrame if calculation fails critically.
        - cache_was_valid_daily_results (bool): True if the data was successfully loaded from cache, False otherwise.
        - status_update (str): A string snippet summarizing whether data was loaded or calculated,
                               and reporting any issues during calculation (e.g., worker failures).
    """
    daily_df = pd.DataFrame()
    cache_valid_daily_results = False
    status_update = ""

    # --- 1. Check Daily Results Cache ---
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
                            # --- Added Cache Validation ---
                            if not isinstance(daily_df, pd.DataFrame):
                                raise ValueError(
                                    "Loaded daily cache is not a DataFrame."
                                )
                            if not isinstance(daily_df.index, pd.DatetimeIndex):
                                daily_df.index = pd.to_datetime(
                                    daily_df.index, errors="coerce"
                                )
                                daily_df = daily_df[
                                    pd.notnull(daily_df.index)
                                ]  # Drop NaT if conversion failed

                            daily_df.sort_index(inplace=True)
                            required_cols = [
                                "value",
                                "net_flow",
                                "daily_return",
                                "daily_gain",
                            ]  # Check core calculated columns
                            # No need to check benchmark prices strictly here, worker might have failed for those
                            missing_cols = [
                                c for c in required_cols if c not in daily_df.columns
                            ]

                            if not missing_cols and not daily_df.empty:
                                # --- End Added Cache Validation ---
                                logging.info(
                                    f"Hist Daily (Scope: {filter_desc}): Loaded {len(daily_df)} rows from cache."
                                )
                                cache_valid_daily_results = True
                                status_update = " Daily results loaded from cache."
                            else:
                                logging.warning(
                                    f"Hist WARN (Scope: {filter_desc}): Daily cache missing columns ({missing_cols}), empty, or failed validation. Recalculating."
                                )
                                daily_df = pd.DataFrame()  # Reset df
                        except Exception as e_load_df:
                            logging.warning(
                                f"Hist WARN (Scope: {filter_desc}): Error deserializing/validating daily cache DF: {e_load_df}. Recalculating."
                            )
                            daily_df = pd.DataFrame()  # Reset df
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
    else:  # Caching enabled but file/key invalid from prep stage
        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Daily cache file/key invalid. Calculating."
        )

    # --- 2. Calculate Daily Metrics if Cache Invalid/Disabled ---
    if not cache_valid_daily_results:
        status_update = " Calculating daily values..."

        # --- Determine Calculation Dates ---
        first_tx_date = (
            transactions_df_effective["Date"].min().date()
            if not transactions_df_effective.empty
            else start_date
        )
        calc_start_date = max(start_date, first_tx_date)
        calc_end_date = end_date

        # Use benchmark data to find likely market days
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
            ):  # Check index type
                try:
                    # Convert index to date objects robustly
                    market_days_index = pd.Index(
                        pd.to_datetime(bench_df.index, errors="coerce").date
                    )
                    market_days_index = market_days_index.dropna()
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
            # Filter market days index to be within the required calculation range
            all_dates_to_process = market_days_index[
                (market_days_index >= calc_start_date)
                & (market_days_index <= calc_end_date)
            ].tolist()

        # <<< ADD DEBUG LOGGING HERE >>>
        if not all_dates_to_process:
            logging.error(
                f"Hist ERROR (Scope: {filter_desc}): No calculation dates found in range {calc_start_date} to {calc_end_date}."
            )
            # Return empty df immediately if no dates to process
            return pd.DataFrame(), False, status_update + " No calculation dates found."
        else:
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Determined {len(all_dates_to_process)} calculation dates from {min(all_dates_to_process)} to {max(all_dates_to_process)}"
            )
        # <<< END DEBUG LOGGING >>>

        if not all_dates_to_process:
            logging.error(
                f"Hist ERROR (Scope: {filter_desc}): No calculation dates found in range."
            )
            return (
                pd.DataFrame(),
                False,
                status_update + " No calculation dates found.",
            )  # Return empty df

        logging.info(
            f"Hist Daily (Scope: {filter_desc}): Calculating {len(all_dates_to_process)} daily metrics parallel..."
        )

        # --- Setup and Run Pool ---
        # Use functools.partial to pass fixed arguments to the worker
        partial_worker = partial(
            _calculate_daily_metrics_worker,  # Assumes this worker function is defined
            transactions_df=transactions_df_effective,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            target_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            benchmark_symbols_yf=clean_benchmark_symbols_yf,
        )

        daily_results_list = []
        pool_start_time = time.time()
        if num_processes is None:
            try:
                num_processes = max(1, os.cpu_count() - 1)  # Leave one core free
            except NotImplementedError:
                num_processes = 1
        # Ensure num_processes is at least 1
        num_processes = max(1, num_processes)

        try:
            # Use freeze_support() if needed, especially for packaged apps (see previous discussion)
            # multiprocessing.freeze_support() # Typically called only in main script guard

            # Consider using imap_unordered for potentially better performance with uneven task times
            # Chunksize calculation aims for roughly 4 chunks per worker initially
            chunksize = max(1, len(all_dates_to_process) // (num_processes * 4))
            logging.info(
                f"Hist Daily: Starting pool with {num_processes} processes, chunksize={chunksize}"
            )

            # Use context manager for the pool
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Use imap_unordered for potentially faster processing as results come in
                results_iterator = pool.imap_unordered(
                    partial_worker, all_dates_to_process, chunksize=chunksize
                )
                # Process results as they complete
                for i, result in enumerate(results_iterator):
                    if i % 100 == 0 and i > 0:  # Print progress occasionally
                        logging.info(
                            f"  Processed {i}/{len(all_dates_to_process)} days..."
                        )
                    if result:  # Append if worker returned data (not None on error)
                        daily_results_list.append(result)
                logging.info(
                    f"  Finished processing all {len(all_dates_to_process)} days."
                )

        except Exception as e_pool:
            logging.error(
                f"Hist CRITICAL (Scope: {filter_desc}): Pool failed: {e_pool}"
            )
            import traceback

            traceback.print_exc()
            # Cannot proceed without daily results
            return pd.DataFrame(), False, status_update + " Multiprocessing failed."
        finally:  # Ensure pool timing is printed even if errors occur later
            pool_end_time = time.time()
            logging.info(
                f"Hist Daily (Scope: {filter_desc}): Pool finished in {pool_end_time - pool_start_time:.2f}s."
            )

        # --- Process Pool Results ---
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

            # Convert columns to numeric, coercing errors
            cols_to_numeric = ["value", "net_flow"] + [
                f"{bm} Price"
                for bm in clean_benchmark_symbols_yf
                if f"{bm} Price" in daily_df.columns
            ]
            for col in cols_to_numeric:
                daily_df[col] = pd.to_numeric(daily_df[col], errors="coerce")

            # Drop rows where essential 'value' calculation failed
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

            # Calculate Daily Gain and Return
            previous_value = daily_df["value"].shift(1)
            # Ensure net_flow is numeric and fill NaNs with 0 for calculation
            net_flow_filled = daily_df["net_flow"].fillna(0.0)

            daily_df["daily_gain"] = (
                daily_df["value"] - previous_value - net_flow_filled
            )
            daily_df["daily_return"] = np.nan  # Initialize

            # --- ADD DETAILED LOGGING FOR SPECIFIC DATE ---
            DEBUG_TWR_DATE = date(
                2020, 1, 7
            )  # <<< CONFIRM this is the correct deposit date you want to check
            logging.debug(
                f"Checking for DEBUG_TWR_DATE: {DEBUG_TWR_DATE} (Type: {type(DEBUG_TWR_DATE)}) in daily_df index."
            )

            try:  # Add try-except around the lookup
                debug_timestamp = pd.Timestamp(DEBUG_TWR_DATE)  # Ensure Timestamp type
                if debug_timestamp in daily_df.index:
                    debug_row = daily_df.loc[debug_timestamp]  # Use Timestamp for .loc
                    dbg_start = debug_row["previous_value"]
                    dbg_end = debug_row["value"]
                    dbg_flow = debug_row["net_flow_filled"]
                    dbg_gain = debug_row["daily_gain"]
                    dbg_return_calc = np.nan
                    if pd.notna(dbg_start) and abs(dbg_start) > 1e-9:
                        dbg_return_calc = dbg_gain / dbg_start

                    logging.debug(
                        f"--- TWR DEBUG for Deposit Date {DEBUG_TWR_DATE} ---"
                    )
                    logging.debug(f"  Start Value (Prev End): {dbg_start:,.2f}")
                    logging.debug(f"  End Value (Current):    {dbg_end:,.2f}")
                    logging.debug(
                        f"  Net External Flow:      {dbg_flow:,.2f}"
                    )  # Should be + deposit amount
                    logging.debug(
                        f"  Calculated Daily Gain:  {dbg_gain:,.2f} (End - Start - Flow)"
                    )
                    logging.debug(
                        f"  Calculated Daily Return:{dbg_return_calc:.6f} (Gain / Start)"
                    )
                    # Also log the final assigned return
                    dbg_final_return = debug_row["daily_return"]
                    logging.debug(f"  Final Assigned Return:  {dbg_final_return:.6f}")
                    logging.debug(f"--- End TWR DEBUG ---")
                else:
                    logging.warning(
                        f"DEBUG_TWR_DATE {DEBUG_TWR_DATE} (as {debug_timestamp}) NOT FOUND in daily_df index after dropna."
                    )
            except KeyError:
                logging.error(
                    f"KeyError looking up debug date {DEBUG_TWR_DATE} (as {debug_timestamp}). Might be missing from index."
                )
            except Exception as e_dbg:
                logging.error(
                    f"Error during TWR debug logging for {DEBUG_TWR_DATE}: {e_dbg}"
                )
            # --- END DETAILED LOGGING ---

            # Calculate return where previous value is valid and non-zero
            valid_prev_value_mask = previous_value.notna() & (
                abs(previous_value) > 1e-9
            )
            daily_df.loc[valid_prev_value_mask, "daily_return"] = (
                daily_df.loc[valid_prev_value_mask, "daily_gain"]
                / previous_value.loc[valid_prev_value_mask]
            )

            # Handle cases where previous value was zero
            zero_gain_mask = daily_df["daily_gain"].notna() & (
                abs(daily_df["daily_gain"]) < 1e-9
            )
            zero_prev_value_mask = previous_value.notna() & (
                abs(previous_value) <= 1e-9
            )
            # Set return to 0% if both gain and previous value were zero
            daily_df.loc[zero_gain_mask & zero_prev_value_mask, "daily_return"] = 0.0
            # Handle infinite return if gain exists but previous value was zero (unlikely for portfolio value)
            # Note: This might produce inf/-inf, which is mathematically correct but maybe not desired for display

            # Set first day's gain/return to NaN
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
            import traceback

            traceback.print_exc()
            return pd.DataFrame(), False, status_update + " Error processing results."

        # --- 3. Save Daily Results Cache ---
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
                ),  # Save df with calculated returns
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


def _calculate_accumulated_gains_and_resample(
    daily_df: pd.DataFrame,  # Input DataFrame with daily value, gain, return, bench prices
    benchmark_symbols_yf: List[str],  # Cleaned list of benchmark tickers
    interval: str,  # 'D', 'W', 'M'
) -> Tuple[pd.DataFrame, float, str]:  # Returns final_df, twr_factor, status_update
    """
    Calculates accumulated gains for portfolio and benchmarks, and optionally resamples the data.

    1. Calculates the cumulative portfolio gain factor (Time-Weighted Return factor) based on the
       *daily* returns present in `daily_df`. Stores the final TWR factor.
    2. Calculates the cumulative gain factor for each benchmark based on the *daily* percentage
       change of their adjusted prices.
    3. If `interval` is 'W' (Weekly) or 'M' (Monthly), it resamples the DataFrame to the specified
       frequency, taking the last value/price and summing the daily gains within each period.
    4. **Crucially**, after resampling, it *recalculates* the 'Portfolio Accumulated Gain' and
       '{Benchmark} Accumulated Gain' columns based on the percentage changes of the *resampled*
       values/prices. This ensures the accumulated gain shown reflects the chosen interval's perspective.
    5. Selects and renames columns for the final output DataFrame.

    Args:
        daily_df (pd.DataFrame): DataFrame indexed by Date, requires 'value', 'daily_gain',
                                 'daily_return', and '{Bench} Price' columns. Result from
                                 `_load_or_calculate_daily_results`.
        benchmark_symbols_yf (List[str]): Cleaned list of benchmark YF tickers.
        interval (str): Resampling interval ('D', 'W', 'M').

    Returns:
        Tuple containing:
        - final_df_resampled (pd.DataFrame): DataFrame with results at the specified interval.
            Columns include 'Portfolio Value', 'Portfolio Daily Gain' (summed if resampled),
            'Portfolio Accumulated Gain' (recalculated based on interval), and corresponding
            benchmark price and accumulated gain columns. Index is DatetimeIndex.
        - final_twr_factor (float): The final portfolio TWR factor calculated from the *daily* returns
                                    over the entire period (1 + TWR). np.nan if calculation fails.
        - status_update (str): Status message snippet indicating completion or resampling failure.
    """
    if daily_df.empty:
        return pd.DataFrame(), np.nan, " (No daily data for final calcs)"
    if "daily_return" not in daily_df.columns:
        return pd.DataFrame(), np.nan, " (Daily return column missing)"

    status_update = ""
    final_twr_factor = (
        np.nan
    )  # TWR factor should reflect the *entire period* based on daily returns
    results_df = daily_df.copy()  # Work on a copy

    try:
        # --- Portfolio Accumulated Gain (Based on DAILY returns) ---
        # Calculate this first, before any resampling, to get the true daily TWR factor
        gain_factors_portfolio = 1 + results_df["daily_return"].fillna(0.0)
        results_df["Portfolio Accumulated Gain Daily"] = (
            gain_factors_portfolio.cumprod()
        )  # Keep daily version temporarily
        # Ensure first value is NaN if daily_return was NaN
        if not results_df.empty and pd.isna(results_df["daily_return"].iloc[0]):
            first_idx_daily = results_df.index[0]
            results_df.loc[first_idx_daily, "Portfolio Accumulated Gain Daily"] = np.nan

        # Extract the FINAL TWR factor from the daily calculation
        if (
            not results_df.empty
            and "Portfolio Accumulated Gain Daily" in results_df.columns
        ):
            last_valid_twr = (
                results_df["Portfolio Accumulated Gain Daily"].dropna().iloc[-1:]
            )
            if not last_valid_twr.empty:
                final_twr_factor = last_valid_twr.iloc[0]

        # --- Benchmark Accumulated Gain (Based on DAILY returns) ---
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_daily = f"{bm_symbol} Accumulated Gain Daily"  # Temp daily column
            if price_col in results_df.columns:
                bench_prices_no_na = results_df[price_col].dropna()
                if not bench_prices_no_na.empty:
                    # Calculate daily returns based on the adjusted prices
                    bench_daily_returns = bench_prices_no_na.pct_change()
                    # Reindex to match results_df, forward fill gaps, fill initial NaN with 0 for cumprod start
                    bench_daily_returns = (
                        bench_daily_returns.reindex(results_df.index)
                        .ffill()
                        .fillna(0.0)
                    )

                    gain_factors_bench = 1 + bench_daily_returns
                    accum_gains_bench = gain_factors_bench.cumprod()
                    results_df[accum_col_daily] = accum_gains_bench
                    # Set first day's accumulated gain to NaN (return relative to start)
                    if not results_df.empty:
                        first_idx_daily = results_df.index[0]
                        results_df.loc[first_idx_daily, accum_col_daily] = np.nan
                else:
                    results_df[accum_col_daily] = np.nan  # No valid prices
            else:
                results_df[accum_col_daily] = np.nan  # Price column missing
        status_update += " Daily accum gain calc complete."

        # --- Apply Resampling ---
        if interval == "D":
            final_df_resampled = results_df  # No resampling needed
            # Rename daily accum gain columns for consistency in output
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

        elif interval in ["W", "M"] and not results_df.empty:
            logging.info(f"Hist Final: Resampling to interval '{interval}'...")
            try:
                # --- Define aggregation methods ---
                resampling_agg = {
                    "value": "last",  # Take last value in period
                    "daily_gain": "sum",  # Sum gains over period
                }
                # Benchmark prices: take last
                for bm_symbol in benchmark_symbols_yf:
                    price_col = f"{bm_symbol} Price"
                    if price_col in results_df.columns:
                        resampling_agg[price_col] = "last"

                # Create the resampled DataFrame
                final_df_resampled = results_df.resample(interval).agg(resampling_agg)

                # --- Recalculate Accumulated Gains on the RESAMPLED data ---
                # Portfolio
                if (
                    "value" in final_df_resampled.columns
                    and not final_df_resampled["value"].dropna().empty
                ):
                    # Calculate returns based on resampled (e.g., weekly/monthly) values
                    resampled_returns = final_df_resampled["value"].pct_change()
                    # Need to fill initial NaN for cumprod, but keep it NaN in the final result
                    resampled_gain_factors = 1 + resampled_returns.fillna(0.0)
                    final_df_resampled["Portfolio Accumulated Gain"] = (
                        resampled_gain_factors.cumprod()
                    )
                    # Set first resampled gain back to NaN if original return was NaN
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
                    final_df_resampled["Portfolio Accumulated Gain"] = (
                        np.nan
                    )  # Handle missing/empty value column

                # Benchmarks
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
                # Fallback: Return the daily results if resampling failed
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

        else:  # Should not happen if interval validation is done earlier
            final_df_resampled = results_df  # Fallback
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

        # --- Select Final Columns ---
        columns_to_keep = [
            "value",
            "daily_gain",
            "Portfolio Accumulated Gain",
        ]  # Start with core portfolio columns
        # Add benchmark prices and accumulated gains if they exist
        for bm_symbol in benchmark_symbols_yf:
            price_col = f"{bm_symbol} Price"
            accum_col_final = f"{bm_symbol} Accumulated Gain"  # Final resampled name
            if price_col in final_df_resampled.columns:
                columns_to_keep.append(price_col)
            if accum_col_final in final_df_resampled.columns:
                columns_to_keep.append(accum_col_final)

        # Ensure columns exist before selecting and renaming
        columns_to_keep = [
            col for col in columns_to_keep if col in final_df_resampled.columns
        ]
        final_df_output = final_df_resampled[columns_to_keep].copy()

        # Rename columns for final output
        final_df_output.rename(
            columns={"value": "Portfolio Value", "daily_gain": "Portfolio Daily Gain"},
            inplace=True,
        )

        # Drop the intermediate daily return column if it exists and interval is not D
        if interval != "D" and "daily_return" in final_df_output.columns:
            final_df_output.drop(columns=["daily_return"], inplace=True)

    except Exception as e_accum:
        logging.exception(
            f"Hist CRITICAL: Accum gain/resample calc error"
        )  # Use logging.exception
        status_update += " Accum gain/resample calc failed."
        return pd.DataFrame(), np.nan, status_update  # Return empty on critical error

    return final_df_output, final_twr_factor, status_update


# --- Historical Performance Calculation Wrapper Function ---
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
) -> Tuple[pd.DataFrame, str]:
    """
    Calculates historical portfolio performance (TWR, value over time) and benchmarks.

    This is the main wrapper function for historical analysis. It orchestrates the entire process:
    1. Prepares inputs (loads/cleans/filters transactions, identifies symbols/FX, gets splits, cache keys).
    2. Loads/fetches raw adjusted historical price/FX data (using cache).
    3. Derives unadjusted prices needed for portfolio valuation by reversing splits.
    4. Loads/calculates daily portfolio value, net flow, and benchmark prices using parallel processing (using cache).
    5. Calculates daily returns/gains from the daily values/flows.
    6. Calculates accumulated gains for portfolio (TWR factor) and benchmarks based on daily data.
    7. Resamples the results to the desired interval ('D', 'W', 'M') and recalculates accumulated gains for the interval.
    8. Returns the final performance DataFrame and a status message.

    Args:
        transactions_csv_file (str): Path to the transactions CSV file.
        start_date (date): Start date for the analysis period.
        end_date (date): End date for the analysis period.
        interval (str): Resampling interval ('D' for Daily, 'W' for Weekly, 'M' for Monthly).
        benchmark_symbols_yf (List[str]): List of Yahoo Finance tickers for benchmarks (e.g., ['SPY', 'QQQ']).
        display_currency (str): The target currency for results (value, gain).
        account_currency_map (Dict): Mapping from account name (str) to its local currency (str).
        default_currency (str): Default currency for accounts not in the map.
        use_raw_data_cache (bool, optional): Whether to use cache for raw historical price/FX data. Defaults to True.
        use_daily_results_cache (bool, optional): Whether to use cache for calculated daily values/returns. Defaults to True.
        num_processes (Optional[int], optional): Number of processes for parallel calculation. Defaults to os.cpu_count() - 1.
        include_accounts (Optional[List[str]], optional): Accounts to specifically include (None or empty list means all). Defaults to None.
        exclude_accounts (Optional[List[str]], optional): Accounts to explicitly exclude. Defaults to None.

    Returns:
        Tuple[pd.DataFrame, str]:
            - pd.DataFrame: DataFrame indexed by date (resampled to `interval`), containing historical
                            performance results ('Portfolio Value', 'Portfolio Accumulated Gain', benchmark prices
                            and accumulated gains). Returns an empty DataFrame on critical failure.
            - str: Status string summarizing the calculation outcome (e.g., "Success", "Finished with Warnings",
                   "Finished with Errors"), filter scope, cache usage, and any issues encountered. Includes the
                   final TWR factor appended (e.g., "|||TWR_FACTOR:1.234567").
    """
    CURRENT_HIST_VERSION = "v1.0"
    start_time_hist = time.time()
    has_errors = False
    has_warnings = False
    status_parts = []  # Collect status parts

    # --- Initial Checks & Cleaning ---
    if not YFINANCE_AVAILABLE:
        return pd.DataFrame(), "Error: yfinance library not installed."
    if start_date >= end_date:
        return pd.DataFrame(), "Error: Start date must be before end date."
    if interval not in ["D", "W", "M"]:
        return pd.DataFrame(), f"Error: Invalid interval '{interval}'."
    clean_benchmark_symbols_yf = []
    if benchmark_symbols_yf and isinstance(benchmark_symbols_yf, list):
        clean_benchmark_symbols_yf = [
            b.upper().strip()
            for b in benchmark_symbols_yf
            if isinstance(b, str) and b.strip()
        ]
    elif not benchmark_symbols_yf:
        logging.info(
            f"Hist INFO ({CURRENT_HIST_VERSION}): No benchmark symbols provided."
        )
    else:
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

    if transactions_df_effective is None:  # Critical failure during prep
        status_msg = f"Error: Failed to prepare inputs (load/clean/filter failed)."
        return pd.DataFrame(), status_msg
    status_parts.append(f"Inputs prepared ({filter_desc})")
    if ignored_reasons:
        status_parts.append(f"{len(ignored_reasons)} tx ignored")

    processed_warnings = set()  # Keep this for unadjust warnings
    final_twr_factor = np.nan
    daily_df = pd.DataFrame()

    # --- 3. Load or Fetch ADJUSTED Historical Raw Data ---
    historical_prices_yf_adjusted, historical_fx_yf, fetch_failed = (
        _load_or_fetch_raw_historical_data(
            symbols_to_fetch_yf=symbols_for_stocks_and_benchmarks_yf,
            fx_pairs_to_fetch_yf=fx_pairs_for_api_yf,
            start_date=start_date,
            end_date=end_date,
            use_raw_data_cache=use_raw_data_cache,
            raw_data_cache_file=raw_data_cache_file,
            raw_data_cache_key=raw_data_cache_key,
        )
    )
    if fetch_failed:
        has_errors = True
    # Note: _load_or_fetch... logs its own warnings/errors

    if has_errors:
        status_msg = f"Error: Failed fetching critical historical FX/Price data."
        status_parts.append("Fetch Error")
        # Construct final status
        final_status_prefix = "Finished with Errors"
        final_status = f"{final_status_prefix} ({filter_desc})"
        if status_parts:
            final_status += f" [{'; '.join(status_parts)}]"
        return pd.DataFrame(), final_status
    status_parts.append("Raw adjusted data loaded/fetched")

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
        has_warnings = True  # If _unadjust logged warnings

    # --- 5 & 6. Load or Calculate Daily Results ---
    daily_df, cache_was_valid_daily, status_update_daily = (
        _load_or_calculate_daily_results(
            use_daily_results_cache=use_daily_results_cache,
            daily_results_cache_file=daily_results_cache_file,
            daily_results_cache_key=daily_results_cache_key,
            start_date=start_date,
            end_date=end_date,
            transactions_df_effective=transactions_df_effective,
            historical_prices_yf_unadjusted=historical_prices_yf_unadjusted,
            historical_prices_yf_adjusted=historical_prices_yf_adjusted,
            historical_fx_yf=historical_fx_yf,
            display_currency=display_currency,
            internal_to_yf_map=internal_to_yf_map,
            account_currency_map=account_currency_map,
            default_currency=default_currency,
            clean_benchmark_symbols_yf=clean_benchmark_symbols_yf,
            num_processes=num_processes,
            current_hist_version=CURRENT_HIST_VERSION,
            filter_desc=filter_desc,
        )
    )
    if status_update_daily:
        status_parts.append(status_update_daily.strip())
    # Check if daily calc failed critically
    if daily_df is None or daily_df.empty:
        if "Error" in status_update_daily or "failed" in status_update_daily.lower():
            has_errors = True
        else:
            has_warnings = True  # Empty DF usually indicates a warning or no data
        status_parts.append("Daily calc failed/empty")
    if "Error" in status_update_daily or "failed" in status_update_daily.lower():
        has_errors = True
    elif "WARN" in status_update_daily.upper():
        has_warnings = True

    if has_errors or daily_df.empty:  # Cannot proceed without daily data
        final_status_prefix = (
            "Finished with Errors"
            if has_errors
            else ("Finished with Warnings" if has_warnings else "Success")
        )
        final_status = f"{final_status_prefix} ({filter_desc})"
        if status_parts:
            final_status += f" [{'; '.join(status_parts)}]"
        return pd.DataFrame(), final_status

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
        # ... (column ordering logic remains the same) ...
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

    return final_df_filtered, final_status


# --- Example Usage (Main block for testing this file directly) ---
if __name__ == "__main__":
    # Configure logging for the test run
    logging.basicConfig(
        level=LOGGING_LEVEL,  # Show INFO, WARNING, ERROR, CRITICAL messages
        format="%(asctime)s [%(levelname)-8s] %(module)s:%(lineno)d - %(message)s",  # Simplified module info
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    multiprocessing.freeze_support()  # Add for safety if creating executables later
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

    logging.info(
        "\n--- Testing Historical Performance Calculation (v10 - Account/Currency) ---"
    )
    test_start = date(2023, 1, 1)
    test_end = date(2024, 6, 30)
    test_interval = "M"
    test_benchmarks = ["SPY", "QQQ"]
    test_use_raw_cache_flag = True
    test_use_daily_results_cache_flag = True
    test_num_processes = None
    test_accounts_subset1 = ["IBKR", "E*TRADE"]
    test_exclude_set = ["SET"]
    test_accounts_subset2 = ["E*TRADE"]

    test_scenarios = {
        "E*TRADE": {"include": test_accounts_subset2, "exclude": None},
        # "IBKR_E*TRADE": {"include": test_accounts_subset1, "exclude": None},
        # "All_Exclude_SET": {"include": None, "exclude": test_exclude_set}, # Test exclude
    }

    if not os.path.exists(test_csv_file):
        logging.error(
            f"ERROR: Test transactions file '{test_csv_file}' not found. Cannot run tests."
        )
    else:
        for name, scenario in test_scenarios.items():
            logging.info(
                f"\n--- Running Historical Test: Scenario='{name}', Include={scenario['include']}, Exclude={scenario['exclude']} ---"
            )
            start_time_run = time.time()
            hist_df, hist_status = calculate_historical_performance(
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
                include_accounts=scenario["include"],
                exclude_accounts=scenario["exclude"],
            )
            end_time_run = time.time()
            logging.info(f"Test '{name}' Status: {hist_status}")
            logging.info(
                f"Test '{name}' Execution Time: {end_time_run - start_time_run:.2f} seconds"
            )
            if not hist_df.empty:
                logging.info(f"Test '{name}' DF tail:\n{hist_df.tail().to_string()}")
            else:
                logging.info(f"Test '{name}' Result: Empty DataFrame")

        logging.info(
            "\n--- Testing Current Portfolio Summary (Account Inclusion/Currency) ---"
        )
        summary_metrics, holdings_df, ignored_df_final, account_metrics, status = (
            calculate_portfolio_summary(
                transactions_csv_file=test_csv_file,
                display_currency=test_display_currency,
                account_currency_map=test_account_currency_map,
                default_currency=test_default_currency,
                include_accounts=test_accounts_subset1,
            )
        )
        logging.info(
            f"Current Summary Status (Subset {test_accounts_subset1}): {status}"
        )
        if summary_metrics:
            logging.info(f"Overall Metrics (Subset): {summary_metrics}")
        if holdings_df is not None and not holdings_df.empty:
            logging.info(
                f"Holdings DF (Subset) Head:\n{holdings_df.head().to_string()}"
            )
        if account_metrics:
            logging.info(f"Account Metrics (Subset): {account_metrics}")
        if ignored_df_final is not None and not ignored_df_final.empty:
            logging.warning(
                f"Ignored transactions found:\n{ignored_df_final.to_string()}"
            )

        # ---> ADD THIS CHECK <---
        if ignored_df_final is not None and not ignored_df_final.empty:
            logging.warning(f"--- Ignored Transactions ({len(ignored_df_final)}) ---")
            # Filter specifically for AAPL/ETRADE Buys if needed for clarity
            ignored_aapl_buys = ignored_df_final[
                (ignored_df_final["Stock / ETF Symbol"] == "AAPL")
                & (ignored_df_final["Investment Account"] == "E*TRADE")
                & (ignored_df_final["Transaction Type"] == "Buy")
            ]
            if not ignored_aapl_buys.empty:
                logging.warning("!!! Found Ignored AAPL/E*TRADE Buy Transactions:")
                logging.warning(f"\n{ignored_aapl_buys.to_string()}")
            else:
                logging.warning(
                    "No ignored AAPL/E*TRADE Buy transactions found, showing all ignored:"
                )
                logging.warning(
                    f"\n{ignored_df_final.to_string()}"
                )  # Show all if no specific ones found
        else:
            logging.info("No transactions were ignored.")

    logging.info("Finished portfolio_logic.py tests.")
# --- END OF REVISED portfolio_logic.py ---
