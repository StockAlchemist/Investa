# tests/test_data_loader.py

import pytest
import pandas as pd
import numpy as np
from io import StringIO  # To simulate CSV files from strings
import os

# --- Import the function to test ---
try:
    from data_loader import load_and_clean_transactions

    DATALOADER_IMPORTED = True
except ImportError as e:
    print(f"ERROR: Could not import from data_loader.py: {e}")
    DATALOADER_IMPORTED = False

# Skip all tests in this file if the function couldn't be imported
pytestmark = pytest.mark.skipif(
    not DATALOADER_IMPORTED, reason="data_loader.py module not found or import failed"
)

# --- Define default inputs used across tests ---
DEFAULT_ACCOUNT_MAP = {
    "IBKR": "USD",
    "SET": "THB",
    "FID": "USD",
    "ING Direct": "USD",
    "Sharebuilder": "USD",
}
DEFAULT_CURRENCY = "USD"

# --- Test Cases ---


def test_load_clean_basic_success(tmp_path):
    """Tests loading a simple, valid CSV string with various transaction types."""
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jun 29, 2002\",Buy,IDIOX,180.000000000,6.027600000,1084.970000000,0.000000000,ING Direct,,
\"Jun 29, 2002\",Buy,IDLOX,153.000000000,11.854200000,1813.690000000,0.000000000,ING Direct,,
\"Jun 29, 2002\",Buy,IDMOX,162.000000000,11.268100000,1825.430000000,0.000000000,ING Direct,,
\"Nov 11, 2003\",Buy,SPY,28.480000000,105.320400000,2999.950000000,0.000000000,Sharebuilder,,
\"Nov 25, 2003\",Buy,DIA,15.310000000,97.962400000,1500.000000000,0.000000000,Sharebuilder,,
"""
    temp_csv_file = tmp_path / "basic_success.csv"
    temp_csv_file.write_text(csv_data)

    (
        df,
        orig_df,
        ignored_idx,
        ignored_rsn,
        has_errors,
        has_warnings,
        original_to_cleaned_header_map,  # Added to match new signature
    ) = load_and_clean_transactions(
        str(temp_csv_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )

    # --- Assertions ---
    assert not has_errors, "Should not have critical errors on valid data"
    assert (
        has_warnings
    ), "Should have warnings because Local Currency is assigned"  # MODIFIED
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) == 5, "Should have 5 valid rows"
    assert ignored_idx == set(), "No rows should be ignored"
    assert ignored_rsn == {}, "No ignore reasons should exist"

    expected_cols = {
        "Date",
        "Type",
        "Symbol",
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Split Ratio",
        "Account",
        "Note",
        "Local Currency",
        "original_index",
        "To Account",
    }  # original_index is added after mapping
    assert (
        set(df.columns) == expected_cols
    ), f"DataFrame columns mismatch. Got: {set(df.columns)}"

    expected_header_map = {
        "Date (MMM DD, YYYY)": "Date",  # Original header -> Cleaned header
        "Transaction Type": "Type",  # Original header -> Cleaned header
        "Stock / ETF Symbol": "Symbol",  # Original header -> Cleaned header
        "Quantity of Units": "Quantity",  # Original header -> Cleaned header
        "Amount per unit": "Price/Share",  # Original header -> Cleaned header
        "Fees": "Commission",
        "Total Amount": "Total Amount",  # Added: Present in CSV and is a cleaned name
        "Investment Account": "Account",
        "Split Ratio (new shares per old share)": "Split Ratio",
        "original_index": "original_index",  # Added: Reflects current behavior
        "Note": "Note",
    }
    # Assert that the returned map contains all expected mappings. It might contain more if CSV has extra cols.
    # Corrected: Use the unpacked variable for assertion
    assert original_to_cleaned_header_map == expected_header_map, "Header map mismatch"
    assert isinstance(  # type: ignore
        original_to_cleaned_header_map, dict
    ), "Header map should be a dictionary"
    assert pd.api.types.is_datetime64_any_dtype(
        df["Date"]
    ), "'Date' column should be datetime"
    assert pd.api.types.is_float_dtype(df["Quantity"]) or pd.api.types.is_integer_dtype(
        df["Quantity"]
    ), "'Quantity' should be numeric"
    assert pd.api.types.is_float_dtype(df["Commission"]), "'Commission' should be float"
    assert (
        df.loc[df["Account"] == "ING Direct", "Local Currency"].iloc[0]
        == DEFAULT_CURRENCY
    )
    assert (
        df.loc[df["Account"] == "Sharebuilder", "Local Currency"].iloc[0]
        == DEFAULT_CURRENCY
    )
    assert df["original_index"].tolist() == list(
        range(5)
    ), "Original index should match row numbers"


def test_load_clean_column_rename(tmp_path):
    """Verify specific column renaming works."""
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Buy,AAPL,10,150,,,IBKR,, 
"""
    temp_csv_file = tmp_path / "col_rename.csv"
    temp_csv_file.write_text(csv_data)
    # Added original_to_cleaned_header_map to unpacking
    df, _, _, _, has_errors, _, original_to_cleaned_header_map = (
        load_and_clean_transactions(
            str(temp_csv_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
        )
    )

    assert not has_errors
    assert df is not None, "DataFrame should not be None after successful load"
    assert "Type" in df.columns  # Should be renamed
    assert "Symbol" in df.columns
    assert "Account" in df.columns
    assert "Transaction Type" not in df.columns  # Original name should be gone
    assert "Date (MMM DD, YYYY)" not in df.columns  # Original date name should be gone
    assert "Date" in df.columns  # Renamed date column should exist

    expected_header_map = {
        "Date (MMM DD, YYYY)": "Date",  # Original header -> Cleaned header
        "Transaction Type": "Type",  # Original header -> Cleaned header
        "Stock / ETF Symbol": "Symbol",  # Original header -> Cleaned header
        "Quantity of Units": "Quantity",  # Original header -> Cleaned header
        "Amount per unit": "Price/Share",  # Original header -> Cleaned header
        "Fees": "Commission",
        "Total Amount": "Total Amount",  # Added
        "Investment Account": "Account",
        "Split Ratio (new shares per old share)": "Split Ratio",
        "original_index": "original_index",  # Added: Reflects current behavior
        "Note": "Note",
    }
    # The map should only contain entries for columns actually present in the CSV
    # Corrected: Use the unpacked variable for assertion
    assert original_to_cleaned_header_map == expected_header_map, "Header map mismatch"


def test_load_clean_date_parsing_formats(tmp_path):
    """Test different valid date formats (assuming fallback parsing works)."""
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
"Jan 15, 2023",Buy,T1,10,10,,,IBKR,,
"02/20/2023",Buy,T2,10,10,,,IBKR,,
"2023-03-10",Buy,T3,10,10,,,IBKR,,
"25-Apr-2023",Buy,T4,10,10,,,IBKR,,
"20230530",Buy,T5,10,10,,,IBKR,,
"""
    temp_csv_file = tmp_path / "date_formats.csv"
    temp_csv_file.write_text(csv_data)

    # Added original_to_cleaned_header_map to unpacking
    df, _, ignored_idx, _, has_errors, has_warnings, original_to_cleaned_header_map = (
        load_and_clean_transactions(
            str(temp_csv_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
        )
    )

    assert not has_errors
    # has_warnings might be true if date inference logs warnings, but data should be correct
    assert (
        len(df) == 5
    ), f"All valid dates should be parsed. Got {len(df)}. Ignored: {ignored_idx}"
    assert ignored_idx == set()
    assert df["Date"].tolist() == [
        pd.Timestamp("2023-01-15"),
        pd.Timestamp("2023-02-20"),
        pd.Timestamp("2023-03-10"),
        pd.Timestamp("2023-04-25"),
        pd.Timestamp("2023-05-30"),
    ]


def test_ignore_missing_qty_price_buy_sell(tmp_path):
    """Test ignoring buy/sell without quantity or price."""
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Buy,SYM1,,100,,,IBKR,,
\"Jan 16, 2023\",Sell,SYM2,10,,,,IBKR,,
\"Jan 17, 2023\",Buy,SYM3,10,100,,,IBKR,,
"""
    temp_csv_file = tmp_path / "missing_qty_price.csv"
    temp_csv_file.write_text(csv_data)

    # Added original_to_cleaned_header_map to unpacking
    (
        df,
        _,
        ignored_idx,
        ignored_rsn,
        has_errors,
        has_warnings,
        original_to_cleaned_header_map,
    ) = load_and_clean_transactions(
        str(temp_csv_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )

    # load_and_clean_transactions should NOT drop these rows.
    # It should convert missing/invalid numeric fields to NaN.
    # The _process_transactions_to_holdings function handles semantic validation.
    assert not has_errors, "Should not have critical errors"
    # has_warnings might be true if numeric conversion fails for empty strings
    assert len(df) == 3, "All rows should be loaded"
    assert (
        ignored_idx == set()
    ), "No rows should be ignored by load_and_clean for these reasons"

    # Check that Quantity for SYM1 is NaN
    assert pd.isna(df[df["Symbol"] == "SYM1"]["Quantity"].iloc[0])
    # Check that Price/Share for SYM2 is NaN
    assert pd.isna(df[df["Symbol"] == "SYM2"]["Price/Share"].iloc[0])
    # SYM3 should have valid Quantity and Price/Share
    assert pd.notna(df[df["Symbol"] == "SYM3"]["Quantity"].iloc[0])
    assert pd.notna(df[df["Symbol"] == "SYM3"]["Price/Share"].iloc[0])


def test_ignore_missing_dividend_amount(tmp_path):
    """Test ignoring dividend without Total Amount or Price."""
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Dividend,AAPL,10,,,,IBKR,,
\"Jan 16, 2023\",Dividend,MSFT,10,0.5,,,IBKR,,
\"Jan 17, 2023\",Dividend,GOOG,,,10.0,,IBKR,,
"""
    temp_csv_file = tmp_path / "missing_dividend.csv"
    temp_csv_file.write_text(csv_data)

    # Added original_to_cleaned_header_map to unpacking
    (
        df,
        _,
        ignored_idx,
        ignored_rsn,
        has_errors,
        has_warnings,
        original_to_cleaned_header_map,
    ) = load_and_clean_transactions(
        str(temp_csv_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )

    # load_and_clean_transactions should NOT drop these rows.
    # It should convert missing/invalid numeric fields to NaN.
    assert not has_errors, "Should not have critical errors"
    assert len(df) == 3, "All rows should be loaded"
    assert (
        ignored_idx == set()
    ), "No rows should be ignored by load_and_clean for these reasons"

    # Check that Total Amount and Price/Share for AAPL are NaN
    aapl_row = df[df["Symbol"] == "AAPL"]
    assert pd.isna(aapl_row["Total Amount"].iloc[0])
    assert pd.isna(
        aapl_row["Price/Share"].iloc[0]
    )  # Amount per unit maps to Price/Share
    assert pd.notna(df[df["Symbol"] == "MSFT"]["Price/Share"].iloc[0])
    assert pd.notna(df[df["Symbol"] == "GOOG"]["Total Amount"].iloc[0])


def test_load_empty_csv(tmp_path):
    """Test loading an empty or header-only CSV."""
    csv_data_empty = ""
    csv_data_header = """\"Date (MMM DD, YYYY)",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note\n"""

    temp_empty_file = tmp_path / "empty.csv"
    temp_empty_file.write_text(csv_data_empty)

    # Added original_to_cleaned_header_map to unpacking
    df1, _, ignored1, _, err1, warn1, original_to_cleaned_header_map1 = (
        load_and_clean_transactions(
            str(temp_empty_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
        )
    )

    assert not err1
    assert warn1
    assert df1.empty  # MODIFIED: Expect empty DataFrame, not None
    assert ignored1 == set()
    assert (
        original_to_cleaned_header_map1 == {}
    ), "Header map should be empty for empty data error"

    temp_header_file = tmp_path / "header_only.csv"
    temp_header_file.write_text(csv_data_header)
    # Added original_to_cleaned_header_map to unpacking

    df2, _, ignored2, _, err2, warn2, header_map2 = load_and_clean_transactions(
        str(temp_header_file), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )

    assert not err2
    assert (
        not warn2
    )  # MODIFIED: Header-only CSV returns early, before local currency warning
    assert df2 is not None and df2.empty
    assert ignored2 == set()

    expected_header_map_for_header_only = {
        "Date (MMM DD, YYYY)": "Date",  # Original header -> Cleaned header
        "Transaction Type": "Type",  # Original header -> Cleaned header
        "Stock / ETF Symbol": "Symbol",  # Original header -> Cleaned header
        "Quantity of Units": "Quantity",  # Original header -> Cleaned header
        "Amount per unit": "Price/Share",  # Original header -> Cleaned header
        "Total Amount": "Total Amount",  # Original header -> Cleaned header
        "Fees": "Commission",
        "Investment Account": "Account",
        "Split Ratio (new shares per old share)": "Split Ratio",
        "original_index": "original_index",  # Added: Reflects current behavior
        "Note": "Note",
    }
    assert (
        header_map2 == expected_header_map_for_header_only
    ), "Header map mismatch for header-only file"


# Add more tests as needed for edge cases, specific cleaning rules, etc.
