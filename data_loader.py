# data_loader.py
import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Set, Dict
from io import StringIO  # Keep if used in tests or function

# Import constants from config.py (assuming it's in the same directory)
try:
    from config import CASH_SYMBOL_CSV, LOGGING_LEVEL  # Add others if needed
except ImportError:
    logging.error(
        "CRITICAL: Could not import constants from config.py in data_loader.py"
    )
    # Define fallbacks if absolutely necessary, but fixing the import path is better
    CASH_SYMBOL_CSV = "$CASH"
    LOGGING_LEVEL = logging.INFO

# Add a basic module docstring (optional but good practice)
"""
-------------------------------------------------------------------------------
 Name:          data_loader.py
 Purpose:       Handles loading and cleaning of transaction data from CSV.

 Author:        Kit Matan (Derived from portfolio_logic.py) and Google Gemini 2.5
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
# Constants
SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}


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
    except pd.errors.EmptyDataError:
        logging.warning(f"Warning: Input CSV file is empty: {transactions_csv_file}")
        has_warnings = True  # It's a warning, not a critical error
        return (
            None,
            None,  # No original DF either
            ignored_row_indices,
            ignored_reasons,
            has_errors,  # Should still be False here
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

    # If loading failed critically earlier, transactions_df will be None.
    # An empty DataFrame (e.g., header-only file) should proceed to cleaning.
    if transactions_df is None:
        logging.warning(
            "Warning: Transactions DataFrame is None after loading attempt."
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

    # Check if the DataFrame is empty *after* successful loading (e.g., header only)
    # This is different from EmptyDataError which means the file itself was empty.
    if transactions_df is not None and transactions_df.empty:
        logging.warning(
            "Warning: CSV loaded successfully but contains no data rows (header only?)."
        )
        # Set warning flag, but allow the empty DataFrame to be returned
        # as it's technically a valid load result.
        has_warnings = True
        # We don't return here, let the empty DF proceed through the (no-op) cleaning
        # and be returned at the end.

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
        # transactions_df["Date"] = pd.to_datetime(
        #     original_date_strings, format="%b %d, %Y", errors="coerce"
        # )  # <-- MODIFIED LINE
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

        # If the dataframe is empty at this point (either started empty or all rows dropped),
        # it's still a valid result (an empty DataFrame), so proceed to sort and return.

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
