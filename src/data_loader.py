# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          data_loader.py
 Purpose:       Handles loading and cleaning of transaction data from CSV or SQLite.
                *** MODIFIED: Now primarily loads from SQLite, with CSV as a fallback/migration source. ***

 Author:        Kit Matan and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Created:       28/04/2025
 Modified:      02/05/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import pandas as pd
import numpy as np
import os
import logging
from typing import List, Tuple, Dict, Optional, Any, Set
import re  # For robust date parsing
from datetime import datetime

# --- Import Configuration ---
try:
    from config import CASH_SYMBOL_CSV, DEFAULT_CURRENCY
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import from config.py in data_loader.py."
    )
    CASH_SYMBOL_CSV = "$CASH"
    DEFAULT_CURRENCY = "USD"

# --- ADDED: Import DB Utilities ---
try:
    from db_utils import (
        initialize_database,
        load_all_transactions_from_db,
        migrate_csv_to_db,
        check_if_db_empty_and_csv_exists,
    )

    DB_UTILS_AVAILABLE = True
except ImportError:
    logging.error(
        "CRITICAL ERROR: Could not import from db_utils.py in data_loader.py. Database functionality will be unavailable."
    )
    DB_UTILS_AVAILABLE = False

    # Define dummy functions if needed for structure, though this indicates a problem
    def initialize_database(*args, **kwargs):
        return None

    def load_all_transactions_from_db(*args, **kwargs):
        return None, False

    def migrate_csv_to_db(*args, **kwargs):
        return 0, 1

    def check_if_db_empty_and_csv_exists(*args, **kwargs):
        return False


# --- END ADDED ---


# Standard expected column names after cleaning/mapping (used internally)
# These are the names portfolio_logic.py and other modules expect.
EXPECTED_CLEANED_COLUMNS = [
    "Date",
    "Type",
    "Symbol",
    "Quantity",
    "Price/Share",
    "Total Amount",
    "Commission",
    "Account",
    "Split Ratio",
    "Note",
    "Local Currency",
    "original_index",  # original_index is crucial
    "To Account",
]

# --- Column Mapping (for CSV loading if still used) ---
# This map defines how various original CSV headers map to our internal standard names.
# Keys are potential original CSV headers (case-insensitive matching will be applied).
# Values are the standardized internal column names from EXPECTED_CLEANED_COLUMNS.
COLUMN_MAPPING_CSV_TO_INTERNAL: Dict[str, str] = {
    # Standardized Headers (already clean)
    "Date": "Date",
    "Type": "Type",
    "Symbol": "Symbol",
    "Quantity": "Quantity",
    "Price/Share": "Price/Share",  # Keep for Price/Share
    "Price per Share": "Price/Share",  # Alias
    "Price / Share": "Price/Share",  # Alias
    "Price": "Price/Share",  # Common alias
    "Total Amount": "Total Amount",
    "Commission": "Commission",
    "Fees": "Commission",  # Alias
    "Account": "Account",
    "Split Ratio": "Split Ratio",
    "Note": "Note",
    "Local Currency": "Local Currency",
    "Currency": "Local Currency",  # Alias
    # Original Verbose Headers (from previous projects or common formats)
    "Date (MMM DD, YYYY)": "Date",
    "Transaction Type": "Type",
    "Stock / ETF Symbol": "Symbol",
    "Quantity of Units": "Quantity",
    "Amount per unit": "Price/Share",
    # "Total Amount": "Total Amount", # Already covered
    # "Fees": "Commission", # Already covered
    "Investment Account": "Account",
    "Split Ratio (new shares per old share)": "Split Ratio",
    # "Note": "Note", # Already covered
    # Add other common variations as needed
    "Transaction Date": "Date",
    "Ticker": "Symbol",
    "Shares": "Quantity",
    "Cost Basis": "Total Amount",  # Can sometimes mean total cost
    "Broker": "Account",
}


# --- Helper function to normalize CSV headers ---
def _normalize_header(header: str) -> str:
    """Normalizes a header string by stripping whitespace and converting to title case for matching."""
    if not isinstance(header, str):
        return ""
    return (
        header.strip()
    )  # Keep original case for matching against COLUMN_MAPPING_CSV_TO_INTERNAL keys


def _map_headers_and_drop_unwanted(
    df: pd.DataFrame, col_mapping: Dict[str, str], expected_cols: List[str]
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """
    Maps DataFrame column headers to standard internal names using col_mapping,
    and drops columns not in expected_cols (after mapping).

    Args:
        df (pd.DataFrame): The input DataFrame with original headers.
        col_mapping (Dict[str, str]): Mapping from original (normalized) headers to internal standard names.
        expected_cols (List[str]): List of internal standard column names that should be kept.

    Returns:
        Tuple[pd.DataFrame, Dict[str, str]]:
            - The DataFrame with standardized and filtered columns.
            - A dictionary mapping the original CSV headers found in the input df
              to the cleaned internal names that were applied.
    """
    original_to_cleaned_applied_map: Dict[str, str] = {}
    rename_dict: Dict[str, str] = {}
    df_columns_normalized = {col: _normalize_header(col) for col in df.columns}

    for original_col_raw, normalized_original_col in df_columns_normalized.items():
        # Try direct match in mapping first (case-sensitive, as mapping keys are specific)
        if normalized_original_col in col_mapping:
            cleaned_name = col_mapping[normalized_original_col]
            rename_dict[original_col_raw] = cleaned_name
            original_to_cleaned_applied_map[original_col_raw] = cleaned_name
        # Fallback: if the normalized original col IS ALREADY a standard expected name
        elif (
            normalized_original_col in expected_cols
            and original_col_raw not in rename_dict
        ):
            # This means the CSV already used a cleaned name for this column.
            # We don't need to rename it, but we record the "mapping" (identity).
            original_to_cleaned_applied_map[original_col_raw] = normalized_original_col
        # No explicit mapping and not already a cleaned name - it will be dropped later if not in expected_cols

    df_renamed = df.rename(columns=rename_dict)

    # Keep only the expected columns (now that they are renamed)
    # and any other columns that were already standard and not in rename_dict
    final_columns_to_keep = [col for col in expected_cols if col in df_renamed.columns]

    # Add any columns that were ALREADY standard but not explicitly in expected_cols if needed,
    # though typically expected_cols should be comprehensive.
    # For now, strict filtering to expected_cols.
    df_final = df_renamed[final_columns_to_keep].copy()

    return df_final, original_to_cleaned_applied_map


def _parse_date_robustly(date_series: pd.Series) -> pd.Series:
    """Attempts to parse a Series of date strings using multiple common formats."""
    # Define a list of common date formats to try
    common_formats = [
        "%Y-%m-%d",  # Standard ISO
        "%m/%d/%Y",  # US common
        "%d/%m/%Y",  # EU common
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%b %d, %Y",  # E.g., Jan 01, 2023 (this is Investa's standard CSV output)
        "%d-%b-%Y",  # E.g., 25-Apr-2023
        "%B %d, %Y",  # E.g., January 01, 2023
        "%Y%m%d",  # E.g., 20230101
        # Add more formats if needed
    ]
    # Attempt direct conversion first (pandas is good at inferring)
    # errors='coerce' will turn unparseable dates into NaT (Not a Time)
    parsed_dates = pd.to_datetime(date_series, errors="coerce")

    # For any dates that failed direct conversion (are NaT), try specific formats
    failed_indices = parsed_dates[parsed_dates.isna()].index
    if not failed_indices.empty:
        logging.debug(
            f"Robust date parsing: {len(failed_indices)} dates initially failed direct parse. Trying specific formats."
        )
        for idx in failed_indices:
            original_date_str = date_series.loc[idx]
            if (
                pd.isna(original_date_str)
                or not isinstance(original_date_str, str)
                or not original_date_str.strip()
            ):
                continue  # Skip NaN or empty strings

            # Clean common ordinal suffixes (st, nd, rd, th)
            cleaned_date_str = re.sub(
                r"(\d+)(st|nd|rd|th)",
                r"\1",
                original_date_str.strip(),
                flags=re.IGNORECASE,
            )

            for fmt in common_formats:
                try:
                    parsed_dates.loc[idx] = datetime.strptime(cleaned_date_str, fmt)
                    break  # Stop if successfully parsed
                except (ValueError, TypeError):
                    continue  # Try next format
            # If still NaT after trying all formats, it will remain NaT
    return parsed_dates


def load_and_clean_transactions(
    source_path: str,  # Can be CSV path or DB path (though DB path not directly used here yet)
    account_currency_map: Dict[str, str],
    default_currency: str,
    is_db_source: bool = False,  # ADDED: Flag to indicate if source is DB
) -> Tuple[
    Optional[pd.DataFrame],
    Optional[pd.DataFrame],
    Set[int],
    Dict[int, str],
    bool,
    bool,
    Dict[str, str],
]:
    """
    Loads transaction data, cleans it, and standardizes column names and data types.
    Primarily loads from SQLite DB if available and populated.
    Falls back to CSV loading/migration if DB is empty and CSV path is valid.

    Args:
        source_path (str): Path to the data source. If `is_db_source` is True, this is the DB path.
                           Otherwise, it's the CSV path.
        account_currency_map (Dict[str, str]): Mapping of account names to their local currencies.
        default_currency (str): Default currency to use if not found.
        is_db_source (bool): True if `source_path` refers to an SQLite database.

    Returns:
        Tuple containing:
        - all_transactions_df (Optional[pd.DataFrame]): Cleaned DataFrame of all transactions.
        - original_transactions_df (Optional[pd.DataFrame]): DataFrame representing the raw data loaded
          (either from DB or CSV before some cleaning steps, primarily for ignored rows context).
        - ignored_indices (Set[int]): Set of 'original_index' values from rows skipped.
        - ignored_reasons (Dict[int, str]): Maps 'original_index' to reason for skipping.
        - has_errors (bool): True if critical errors occurred.
        - has_warnings (bool): True if non-critical warnings occurred.
        - original_to_cleaned_header_map (Dict[str,str]): Map of original CSV headers to cleaned names.
                                                          Empty if loaded from DB.
    """
    logging.info(
        f"Data Loader: Starting load from {'DB' if is_db_source else 'CSV'}: {source_path}"
    )
    has_errors = False
    has_warnings = False
    ignored_indices: Set[int] = set()
    ignored_reasons: Dict[int, str] = {}
    original_to_cleaned_header_map: Dict[str, str] = {}  # For CSV context
    raw_df_for_ignored_context: Optional[pd.DataFrame] = None

    db_conn = None
    loaded_from_db = False

    if is_db_source and DB_UTILS_AVAILABLE:
        db_conn = initialize_database(source_path)  # source_path is db_path
        if db_conn:
            df, success = load_all_transactions_from_db(
                db_conn, account_currency_map, default_currency
            )  # MODIFIED: Pass map and default
            if success and df is not None and not df.empty:
                raw_df_for_ignored_context = (
                    df.copy()
                )  # DB data is already fairly clean
                # `original_index` is already the DB `id`
                # `Date` is already parsed to datetime by load_all_transactions_from_db
                # Numeric types are also handled by load_all_transactions_from_db
                loaded_from_db = True
                logging.info(
                    f"Successfully loaded {len(df)} transactions from database."
                )
            elif success and (df is None or df.empty):
                logging.info(
                    "Database is empty or load_all_transactions_from_db returned no data. Will check for CSV."
                )
                raw_df_for_ignored_context = pd.DataFrame()  # Ensure it's an empty DF
            else:  # Load failed
                logging.error(
                    "Failed to load transactions from database. Will check for CSV."
                )
                has_errors = True  # Mark as error if DB load explicitly fails
                raw_df_for_ignored_context = pd.DataFrame()
        else:  # DB connection failed
            logging.error(
                f"Failed to connect to database at {source_path}. Will check for CSV."
            )
            has_errors = True  # Mark as error if DB connection fails
            raw_df_for_ignored_context = pd.DataFrame()

    if (
        not loaded_from_db
    ):  # If DB load failed, or DB was empty, or not a DB source initially
        if is_db_source:  # This means DB was specified but load failed or was empty
            logging.info(
                f"DB source specified but no data loaded. Checking for CSV at '{source_path}' (assuming it might be a CSV for migration)."
            )
            # The source_path here is problematic if it was a DB path. We need the CSV path.
            # This function's design needs to be clearer about which path is which.
            # For now, let's assume if is_db_source=True and loaded_from_db=False,
            # we expect a CSV to be at a *configured default CSV path* for migration,
            # NOT at the `source_path` which was the DB path.
            # This part of the logic needs refinement based on how the GUI calls it.
            # TEMPORARY: If DB source fails, we don't automatically load CSV here.
            # The GUI should handle the "DB empty, CSV exists, want to migrate?" flow.
            # So, if is_db_source and not loaded_from_db, we consider it an error or empty state.
            if not has_errors:  # If DB was just empty, not an error yet
                logging.info(
                    "Database is empty. No CSV migration attempted by data_loader directly."
                )
            # raw_df_for_ignored_context is already pd.DataFrame() or None
            # Fall through to the CSV loading section only if is_db_source was False.
        # else: # This means is_db_source was False, so source_path IS the CSV path

        if (
            not is_db_source
        ):  # Only proceed with CSV logic if it was explicitly a CSV source
            logging.info(f"Loading from CSV: {source_path}")
            if not os.path.exists(source_path):
                logging.error(f"Transactions CSV file not found: {source_path}")
                return (
                    None,
                    None,
                    ignored_indices,
                    ignored_reasons,
                    True,
                    has_warnings,
                    original_to_cleaned_header_map,
                )

            try:
                # Read raw CSV for ignored rows context, keeping original headers
                # Keep all columns initially to allow user to see full ignored row
                raw_df_for_ignored_context = pd.read_csv(
                    source_path, dtype=str, keep_default_na=False, skipinitialspace=True
                )
                if raw_df_for_ignored_context.empty and not list(
                    raw_df_for_ignored_context.columns
                ):
                    logging.warning(
                        f"CSV file '{source_path}' is empty or has no headers."
                    )
                    # Return empty but valid structure
                    return (
                        pd.DataFrame(columns=EXPECTED_CLEANED_COLUMNS),
                        pd.DataFrame(),
                        set(),
                        {},
                        False,
                        True,
                        {},
                    )

                raw_df_for_ignored_context["original_index"] = (
                    raw_df_for_ignored_context.index
                )  # Add original_index based on CSV row order

                # Now, process for the main df (map headers, clean data)
                # Reload or use a copy for processing to avoid altering raw_df_for_ignored_context too much
                df_processing = raw_df_for_ignored_context.copy()

                # 1. Map headers and select expected columns
                df_processed, original_to_cleaned_applied_map_csv = (
                    _map_headers_and_drop_unwanted(
                        df_processing,
                        COLUMN_MAPPING_CSV_TO_INTERNAL,
                        EXPECTED_CLEANED_COLUMNS,
                    )
                )
                original_to_cleaned_header_map = (
                    original_to_cleaned_applied_map_csv  # Store this for GUI
                )

                if df_processed.empty and not raw_df_for_ignored_context.empty:
                    logging.warning(
                        f"No columns mapped to expected standard names from CSV: {source_path}. Check headers."
                    )
                    # Keep raw_df_for_ignored_context for potential display of all rows as "ignored"
                    for idx in raw_df_for_ignored_context.index:
                        ignored_indices.add(idx)
                        ignored_reasons[idx] = "CSV Header Mapping Failed"
                    has_warnings = True
                    # Return the raw_df as 'all_transactions_df' so GUI can show it in ignored log
                    return (
                        raw_df_for_ignored_context,
                        raw_df_for_ignored_context,
                        ignored_indices,
                        ignored_reasons,
                        has_errors,
                        has_warnings,
                        original_to_cleaned_header_map,
                    )

                df = df_processed  # Use the processed DataFrame from here
                logging.info(
                    f"Loaded {len(df)} rows from CSV '{source_path}'. Mapped headers."
                )

            except pd.errors.EmptyDataError:
                logging.warning(f"Transactions CSV file is empty: {source_path}")
                return (
                    pd.DataFrame(columns=EXPECTED_CLEANED_COLUMNS),
                    pd.DataFrame(),
                    ignored_indices,
                    ignored_reasons,
                    has_errors,
                    True,
                    original_to_cleaned_header_map,
                )
            except Exception as e:
                logging.error(
                    f"Error reading or initially processing CSV file {source_path}: {e}"
                )
                return (
                    None,
                    None,
                    ignored_indices,
                    ignored_reasons,
                    True,
                    has_warnings,
                    original_to_cleaned_header_map,
                )
        else:  # is_db_source was true but no data loaded
            df = pd.DataFrame()  # Start with an empty DF if DB load failed or was empty
            if not raw_df_for_ignored_context:  # Ensure it's an empty DF if not set
                raw_df_for_ignored_context = pd.DataFrame()
            # No original_to_cleaned_header_map from DB source

    # --- Common Cleaning Steps (apply to df loaded from either DB or CSV) ---
    # Ensure df is a DataFrame, even if empty from failed load attempts above
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()  # Default to empty if something went very wrong
        if not has_errors:  # If not already marked as error, this is a new problem
            logging.error(
                "DataFrame `df` is not a pandas DataFrame after load attempts."
            )
            has_errors = True

    # If df is empty at this point, can return early
    if df.empty:
        logging.info("No transactions to process after loading stage.")
        # raw_df_for_ignored_context should hold original CSV data if loaded, or empty if DB was empty/failed
        return (
            df,
            raw_df_for_ignored_context,
            ignored_indices,
            ignored_reasons,
            has_errors,
            has_warnings,
            original_to_cleaned_header_map,
        )

    # --- Add 'original_index' if it's missing (should be present from DB or CSV raw load) ---
    if "original_index" not in df.columns:
        if not df.empty:  # Only add if df is not empty
            df["original_index"] = df.index  # Fallback if missing
            logging.warning(
                "Added 'original_index' from DataFrame index as it was missing."
            )
            has_warnings = True
        elif (
            raw_df_for_ignored_context is not None
            and "original_index" in raw_df_for_ignored_context.columns
        ):
            # If df is empty but raw_df has original_index, this is fine for ignored context.
            pass
        else:  # df is empty and raw_df doesn't have it either
            if (
                raw_df_for_ignored_context is not None
                and not raw_df_for_ignored_context.empty
            ):
                raw_df_for_ignored_context["original_index"] = (
                    raw_df_for_ignored_context.index
                )

    # --- Ensure all EXPECTED_CLEANED_COLUMNS exist, add if missing (with NaN/None) ---
    for col in EXPECTED_CLEANED_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA  # Use pd.NA for broader NA compatibility
            logging.debug(f"Added missing expected column '{col}' to DataFrame.")

    # --- Data Type Conversions and Cleaning ---
    # Date Parsing (critical)
    if "Date" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(
            df["Date"]
        ):  # If not already datetime (e.g. from DB)
            df["Date"] = _parse_date_robustly(df["Date"])
        # Store rows that failed date parsing
        failed_date_parse_mask = df["Date"].isna()
        if failed_date_parse_mask.any():
            for idx in df[failed_date_parse_mask].index:
                original_idx_val = df.loc[idx, "original_index"]
                ignored_indices.add(original_idx_val)
                ignored_reasons[original_idx_val] = "Invalid or Unparseable Date"
            df = df[~failed_date_parse_mask].copy()  # Keep only rows with valid dates
            logging.warning(
                f"Removed {failed_date_parse_mask.sum()} rows due to unparseable dates."
            )
            has_warnings = True
    else:  # Date column is fundamental
        logging.error(
            "CRITICAL: 'Date' column is missing. Cannot proceed with processing."
        )
        # Mark all rows as ignored if Date is missing
        if "original_index" in df.columns:
            for original_idx_val in df["original_index"].tolist():
                ignored_indices.add(original_idx_val)
                ignored_reasons[original_idx_val] = "Missing Date Column"
        return (
            df.iloc[0:0],
            raw_df_for_ignored_context,
            ignored_indices,
            ignored_reasons,
            True,
            has_warnings,
            original_to_cleaned_header_map,
        )

    # Numeric Conversions (Quantity, Price/Share, Total Amount, Commission, Split Ratio)
    numeric_cols = [
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Split Ratio",
    ]
    for col in numeric_cols:
        if col in df.columns:
            # Store original values before conversion for more precise error reporting
            original_values_for_error = df[col].copy()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            # Log rows where numeric conversion failed (became NaN) but original was not NaN/empty
            # errors='coerce' already turns unconvertible to NaN
            conversion_failed_mask = (
                df[col].isna()
                & original_values_for_error.notna()
                & (original_values_for_error != "")
            )
            if conversion_failed_mask.any():
                for idx in df[conversion_failed_mask].index:
                    original_idx_val = df.loc[idx, "original_index"]
                    # Don't add to ignored_indices here if the field is optional (e.g. Split Ratio for Buy)
                    # Portfolio_logic will handle missing essential numerics per transaction type.
                    # Just log a warning.
                    logging.warning(
                        f"Row {original_idx_val}: Could not convert '{col}' value '{original_values_for_error.loc[idx]}' to numeric. Set to NaN."
                    )
                    # ignored_reasons[original_idx_val] = f"Invalid numeric value for {col}: {original_values_for_error.loc[idx]}" # Optional: add to reasons
                has_warnings = True

    # Text Column Cleaning (strip whitespace, ensure string type)
    text_cols = ["Type", "Symbol", "Account", "Note", "Local Currency"]
    for col in text_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace("nan", "", regex=False)
                .replace("None", "", regex=False)
            )

    # --- Local Currency Assignment (crucial) ---
    if (
        "Local Currency" not in df.columns
        or df["Local Currency"].isin(["", None, np.nan, pd.NA]).all()
    ):
        logging.info(
            "'Local Currency' column missing or empty. Assigning based on account_currency_map."
        )
        df["Local Currency"] = (
            df["Account"].map(account_currency_map).fillna(default_currency)
        )
        has_warnings = True  # Adding this column based on map is a "fix"
    else:  # Local Currency column exists, fill blanks
        # --- MODIFIED: More robust cleaning for Local Currency ---
        # Create a boolean mask for rows with invalid or missing currency codes.
        # This logic mirrors the robust check in _process_transactions_to_holdings.
        # 1. Strip whitespace and convert to uppercase for consistent comparison.
        # 2. Check for various null-like strings, non-3-char codes, or actual NaNs.
        normalized_currency = df["Local Currency"].astype(str).str.strip().str.upper()
        invalid_currency_mask = (
            normalized_currency.isin(["", "<NA>", "NAN", "NONE", "N/A"])
            | (normalized_currency.str.len() != 3)
            | (df["Local Currency"].isna())
        )

        if invalid_currency_mask.any():
            # For rows with invalid currency, map the account to its currency.
            # Use .loc with the mask to update only the necessary rows.
            df.loc[invalid_currency_mask, "Local Currency"] = (
                df.loc[invalid_currency_mask, "Account"]
                .map(account_currency_map)
                .fillna(default_currency)
            )
            logging.info(
                f"Filled or corrected {invalid_currency_mask.sum()} missing/invalid Local Currency values."
            )
            has_warnings = True
    df["Local Currency"] = df["Local Currency"].str.upper()  # Standardize to uppercase

    # --- Convert legacy $CASH symbol to currency-specific cash symbols ---
    # This logic is removed. The symbol will remain '$CASH' and currency is handled by the 'Local Currency' column.
    # Any currency-specific cash symbols like '$CASH_USD' will be treated as distinct symbols if they exist in old data.
    # Ensure `original_transactions_df` for context of ignored rows.
    # If loaded from DB, `raw_df_for_ignored_context` is already set.
    # If from CSV, `raw_df_for_ignored_context` holds the raw CSV read.
    # This df should have 'original_index'.
    final_original_df_for_context = (
        raw_df_for_ignored_context
        if raw_df_for_ignored_context is not None
        else pd.DataFrame()
    )

    logging.info(
        f"Data loading and cleaning finished. Errors: {has_errors}, Warnings: {has_warnings}, Ignored: {len(ignored_indices)}"
    )
    return (
        df,
        final_original_df_for_context,
        ignored_indices,
        ignored_reasons,
        has_errors,
        has_warnings,
        original_to_cleaned_header_map,
    )
