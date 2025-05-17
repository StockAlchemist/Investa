# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          csv_utils.py
 Purpose:       Utility functions for CSV file manipulation, such as header
                standardization.

 Author:        Kit Matan and Google Gemini
 Author Email:  kittiwit@gmail.com

 Created:       17/05/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import pandas as pd
import csv
import logging
import os
from typing import Dict, List, Tuple

# Standard mapping from verbose original CSV headers to cleaned internal headers
STANDARD_ORIGINAL_TO_CLEANED_MAP: Dict[str, str] = {
    "Date (MMM DD, YYYY)": "Date",
    "Transaction Type": "Type",
    "Stock / ETF Symbol": "Symbol",
    "Quantity of Units": "Quantity",
    "Amount per unit": "Price/Share",
    "Total Amount": "Total Amount",
    "Fees": "Commission",
    "Investment Account": "Account",
    "Split Ratio (new shares per old share)": "Split Ratio",
    "Note": "Note",
}

# Desired order of columns in the standardized CSV
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
]


def convert_csv_headers_to_cleaned_format(
    input_csv_path: str, output_csv_path: str
) -> Tuple[bool, str]:
    """
    Reads a CSV file, renames its headers to a cleaned/standardized format,
    reorders columns, and saves it.

    Args:
        input_csv_path (str): Path to the input CSV file.
        output_csv_path (str): Path to save the modified CSV file.
                               Can be the same as input_csv_path to overwrite.

    Returns:
        Tuple[bool, str]: (success_status, message)
    """
    logging.info(
        f"Attempting to standardize CSV headers for '{input_csv_path}' -> '{output_csv_path}'"
    )
    try:
        # Read the CSV, treating all data as strings to preserve original values
        # keep_default_na=False ensures empty strings are read as such, not NaN
        df = pd.read_csv(
            input_csv_path, dtype=str, keep_default_na=False, skipinitialspace=True
        )
        if df.empty and not list(
            df.columns
        ):  # Handles completely empty file or file with no headers
            logging.warning(
                f"Input CSV '{input_csv_path}' is empty or has no headers. No conversion performed."
            )
            # Create an empty CSV with desired headers if output is different or if overwriting an empty file
            pd.DataFrame(columns=DESIRED_CLEANED_COLUMN_ORDER).to_csv(
                output_csv_path,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,
            )
            return (
                True,
                "Input CSV was empty; output CSV (if different) created with standard headers.",
            )

        original_columns_stripped = [col.strip() for col in df.columns]
        df.columns = original_columns_stripped  # Apply stripped column names to df for consistent processing

        rename_map: Dict[str, str] = {}
        current_cleaned_names_in_df = set()

        for current_col in original_columns_stripped:
            if current_col in STANDARD_ORIGINAL_TO_CLEANED_MAP:
                cleaned_name = STANDARD_ORIGINAL_TO_CLEANED_MAP[current_col]
                rename_map[current_col] = cleaned_name
                current_cleaned_names_in_df.add(cleaned_name)
            elif (
                current_col in STANDARD_ORIGINAL_TO_CLEANED_MAP.values()
            ):  # Already a cleaned name
                current_cleaned_names_in_df.add(current_col)
            else:  # Unknown column, will be kept as is
                current_cleaned_names_in_df.add(current_col)

        df.rename(columns=rename_map, inplace=True)
        logging.info(f"Applied column renames: {rename_map}")

        # Reorder columns: desired ones first, then any others
        final_column_order = [
            col for col in DESIRED_CLEANED_COLUMN_ORDER if col in df.columns
        ]
        other_columns = [col for col in df.columns if col not in final_column_order]
        df = df[final_column_order + other_columns]

        # Save the modified DataFrame
        df.to_csv(
            output_csv_path, index=False, encoding="utf-8", quoting=csv.QUOTE_MINIMAL
        )
        msg = f"Successfully standardized CSV headers and saved to '{output_csv_path}'"
        logging.info(msg)
        return True, msg

    except FileNotFoundError:
        msg = f"Error: Input CSV file not found at '{input_csv_path}'"
        logging.error(msg)
        return False, msg
    except Exception as e:
        msg = f"An unexpected error occurred during CSV header conversion: {e}"
        logging.exception(msg)
        return False, msg


if __name__ == "__main__":
    # Example Usage (for testing this script directly)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(message)s"
    )
    # Create a dummy input CSV for testing
    test_input_dir = "test_csv_data"
    os.makedirs(test_input_dir, exist_ok=True)
    test_input_file = os.path.join(
        test_input_dir, "sample_transactions_original_headers.csv"
    )
    test_output_file = os.path.join(
        test_input_dir, "sample_transactions_cleaned_headers.csv"
    )

    sample_data = {
        "Date (MMM DD, YYYY)": ["Jan 01, 2023", "Feb 02, 2023"],
        "Transaction Type": ["Buy", "Sell"],
        "Stock / ETF Symbol": ["AAPL", "MSFT"],
        "Quantity of Units": ["10", "5"],
        "Amount per unit": ["150.00", "250.00"],
        "Total Amount": ["1500.00", "1250.00"],
        "Fees": ["5.00", "5.00"],
        "Investment Account": ["BrokerageA", "BrokerageB"],
        "Split Ratio (new shares per old share)": ["", ""],
        "Note": ["Bought Apple", "Sold Microsoft"],
        "Custom User Column": ["Val1", "Val2"],  # An extra column
    }
    pd.DataFrame(sample_data).to_csv(
        test_input_file, index=False, quoting=csv.QUOTE_MINIMAL
    )

    success, message = convert_csv_headers_to_cleaned_format(
        test_input_file, test_output_file
    )
    print(f"Conversion Result: {success} - {message}")

    if success:
        print(f"\nContents of '{test_output_file}':")
        df_out = pd.read_csv(test_output_file)
        print(df_out.to_string())

    # Test with an empty file
    empty_input_file = os.path.join(test_input_dir, "empty_input.csv")
    empty_output_file = os.path.join(test_input_dir, "empty_output_cleaned.csv")
    with open(empty_input_file, "w") as f:
        f.write("")  # Create an empty file
    success_empty, msg_empty = convert_csv_headers_to_cleaned_format(
        empty_input_file, empty_output_file
    )
    print(f"\nEmpty File Conversion Result: {success_empty} - {msg_empty}")
    if success_empty and os.path.exists(empty_output_file):
        df_empty_out = pd.read_csv(empty_output_file)
        print(f"Contents of '{empty_output_file}':\n{df_empty_out.to_string()}")
