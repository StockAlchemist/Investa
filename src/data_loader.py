# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          data_loader.py
 Purpose:       Handles loading and cleaning of transaction data from CSV or SQLite.


 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import investa_core
import pandas as pd
import logging
from typing import List, Tuple, Dict, Optional, Set

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
    "Tags",
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
    # Transfer / To Account mappings
    "Transfer To": "To Account",
    "To Account": "To Account",
    "Destination Account": "To Account",
    "Destination": "To Account",
    # Add other common variations as needed
    "Transaction Date": "Date",
    "Ticker": "Symbol",
    "Shares": "Quantity",
    "Cost Basis": "Total Amount",  # Can sometimes mean total cost
    "Broker": "Account",
    "Tags": "Tags",
    "Labels": "Tags",
    "Groups": "Tags",
}


# --- Helper function to normalize CSV headers ---
def _normalize_header(header: str) -> str:
    """Normalizes a header string by stripping whitespace and converting to title case for matching."""
    # Ported to Rust: rust/investa_core/src/dl.rs
    return investa_core.dl_normalize_header(header)


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
    # Ported to Rust: rust/investa_core/src/dl.rs
    return investa_core.dl_map_headers(df, col_mapping, expected_cols)


def _parse_date_robustly(date_series: pd.Series) -> pd.Series:
    """Attempts to parse a Series of date strings using multiple common formats."""
    # Ported to Rust: rust/investa_core/src/dl.rs
    return investa_core.dl_parse_date_robustly(date_series)


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
    # Ported to Rust: rust/investa_core/src/dl.rs
    return investa_core.dl_load_and_clean(
        source_path, account_currency_map, default_currency, is_db_source
    )
