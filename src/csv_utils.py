# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          csv_utils.py
 Purpose:       Utility functions for CSV file manipulation, such as header
                standardization.


 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import investa_core
import pandas as pd
import csv
import logging
import os
from typing import Dict, List, Tuple

# Standard mapping from verbose original CSV headers to cleaned internal headers
# This map is crucial for interpreting various user-provided CSVs.
STANDARD_ORIGINAL_TO_CLEANED_MAP: Dict[str, str] = {
    "Date (MMM DD, YYYY)": "Date",
    "Transaction Date": "Date",  # Alias
    "Transaction Type": "Type",
    "Stock / ETF Symbol": "Symbol",
    "Ticker": "Symbol",  # Alias
    "Quantity of Units": "Quantity",
    "Shares": "Quantity",  # Alias
    "Amount per unit": "Price/Share",
    "Price per Share": "Price/Share",  # Alias
    "Price / Share": "Price/Share",  # Alias
    "Price": "Price/Share",  # Alias for price per unit
    "Total Amount": "Total Amount",
    "Cost Basis": "Total Amount",  # Often used this way
    "Fees": "Commission",
    "Commission": "Commission",  # Already clean
    "Investment Account": "Account",
    "Broker": "Account",  # Alias
    "Account Name": "Account",  # Alias
    "Split Ratio (new shares per old share)": "Split Ratio",
    "Split Ratio": "Split Ratio",  # Already clean
    "Note": "Note",
    "Description": "Note",  # Alias
    "Local Currency": "Local Currency",  # Already clean
    "Currency": "Local Currency",  # Alias
}

# Desired order of columns in the standardized CSV (these are the "cleaned" names)
# This order is used when writing a standardized CSV.
# data_loader.py will ensure these columns exist in the DataFrame it passes to logic.
DESIRED_CLEANED_COLUMN_ORDER: List[str] = [
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
    "Local Currency",  # Added Local Currency here for completeness
]


def convert_csv_headers_to_cleaned_format(
    input_csv_path: str, output_csv_path: str
) -> Tuple[bool, str]:
    """
    Reads a CSV file, renames its headers to a cleaned/standardized format
    based on STANDARD_ORIGINAL_TO_CLEANED_MAP, reorders columns according to
    DESIRED_CLEANED_COLUMN_ORDER, and saves it.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the modified CSV file.
                               Can be the same as input_csv_path to overwrite.

    Returns:
        Tuple[bool, str]: (success_status, message)
    """
    # Ported to Rust: rust/investa_core/src/dl.rs
    return investa_core.dl_convert_csv_headers(input_csv_path, output_csv_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s"
    )
    test_input_dir = "test_csv_data_headers"  # Use a different dir to avoid conflict
    os.makedirs(test_input_dir, exist_ok=True)
    test_input_file = os.path.join(
        test_input_dir, "sample_transactions_various_headers.csv"
    )
    test_output_file = os.path.join(
        test_input_dir, "sample_transactions_standardized_headers.csv"
    )

    # Sample data with various possible headers that map to our standard
    sample_data_various = {
        "Date (MMM DD, YYYY)": ["Jan 01, 2023", "Feb 02, 2023"],
        "Transaction Type": ["Buy", "Sell"],
        "Ticker": ["GOOGL", "MSFT"],  # Uses "Ticker" instead of "Stock / ETF Symbol"
        "Quantity of Units": ["10", "5"],
        "Price": ["100.00", "250.00"],  # Uses "Price" instead of "Amount per unit"
        "Total Amount": ["1000.00", "1250.00"],
        "Commission": ["5.00", "4.95"],  # Uses "Commission" (already clean)
        "Investment Account": ["BrokerX", "BrokerY"],
        "Split Ratio": ["", ""],  # Uses "Split Ratio" (already clean)
        "Description": [
            "Bought Google",
            "Sold Microsoft",
        ],  # Uses "Description" for "Note"
        "Currency": ["USD", "USD"],  # Uses "Currency" for "Local Currency"
    }
    pd.DataFrame(sample_data_various).to_csv(
        test_input_file, index=False, quoting=csv.QUOTE_MINIMAL
    )

    print("--- Testing CSV Header Standardization ---")
    print(f"Input file: {test_input_file}")
    print(f"Output file: {test_output_file}")

    success, message = convert_csv_headers_to_cleaned_format(
        test_input_file, test_output_file
    )
    print(f"Conversion Result: {success} - {message}")

    if success and os.path.exists(test_output_file):
        print(f"\nContents of standardized output '{test_output_file}':")
        df_out = pd.read_csv(test_output_file)
        print(df_out.to_string())
        print(f"\nColumns in output: {df_out.columns.tolist()}")

    # Test with an empty file
    empty_input_file = os.path.join(test_input_dir, "empty_input_for_std.csv")
    empty_output_file = os.path.join(test_input_dir, "empty_output_std_cleaned.csv")
    with open(empty_input_file, "w") as f:
        f.write("")
    success_empty, msg_empty = convert_csv_headers_to_cleaned_format(
        empty_input_file, empty_output_file
    )
    print(f"\nEmpty File Conversion Result: {success_empty} - {msg_empty}")
    if success_empty and os.path.exists(empty_output_file):
        df_empty_out = pd.read_csv(empty_output_file)
        print(f"Contents of '{empty_output_file}':\n{df_empty_out.to_string()}")
        print(f"Columns in empty output: {df_empty_out.columns.tolist()}")
