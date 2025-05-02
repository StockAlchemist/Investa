# tests/test_data_loader.py

import pytest
import pandas as pd
import numpy as np
from datetime import date
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


def test_load_clean_basic_success():
    """Tests loading a simple, valid CSV string with various transaction types."""
    # --- CORRECTED csv_data format (Dates Quoted) ---
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jun 29, 2002\",Buy,IDIOX,180.000000000,6.027600000,1084.970000000,0.000000000,ING Direct,,
\"Jun 29, 2002\",Buy,IDLOX,153.000000000,11.854200000,1813.690000000,0.000000000,ING Direct,,
\"Jun 29, 2002\",Buy,IDMOX,162.000000000,11.268100000,1825.430000000,0.000000000,ING Direct,,
\"Nov 11, 2003\",Buy,SPY,28.480000000,105.320400000,2999.950000000,0.000000000,Sharebuilder,,
\"Nov 25, 2003\",Buy,DIA,15.310000000,97.962400000,1500.000000000,0.000000000,Sharebuilder,,
"""  # Note: This data has 5 rows
    # --- END CORRECTION ---
    csv_file_like = StringIO(csv_data)

    df, orig_df, ignored_idx, ignored_rsn, has_errors, has_warnings = (
        load_and_clean_transactions(
            csv_file_like, DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
        )
    )

    # --- Assertions (Should remain mostly the same) ---
    assert not has_errors, "Should not have critical errors on valid data"
    assert isinstance(df, pd.DataFrame), "Result should be a DataFrame"
    assert not df.empty, "DataFrame should not be empty"
    assert len(df) == 5, "Should have 5 valid rows from the string"
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
    }
    assert (
        set(df.columns) == expected_cols
    ), f"DataFrame columns mismatch. Got: {set(df.columns)}"
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


def test_load_clean_column_rename():
    """Verify specific column renaming works."""
    # --- CORRECTED csv_data format (Removed quotes from data row date) ---
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Buy,AAPL,10,150,,,IBKR,, 
"""
    # --- END CORRECTION ---
    csv_file_like = StringIO(csv_data)
    df, _, _, _, has_errors, _ = load_and_clean_transactions(
        csv_file_like, DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    # --- Assertions should now pass ---
    assert not has_errors
    assert (
        df is not None
    ), "DataFrame should not be None after successful load"  # Add check
    assert "Type" in df.columns
    assert "Symbol" in df.columns
    assert "Account" in df.columns
    assert "Transaction Type" not in df.columns  # Original name should be gone
    assert "Date (MMM DD, YYYY)" not in df.columns  # Original date name should be gone
    assert "Date" in df.columns  # Renamed date column should exist


def test_load_clean_date_parsing_formats():
    """Test different valid date formats (assuming fallback parsing works)."""
    # --- CORRECTED csv_data format (Dates Quoted) ---
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
"Jan 15, 2023",Buy,T1,10,10,,,IBKR,,
"02/20/2023",Buy,T2,10,10,,,IBKR,,
"2023-03-10",Buy,T3,10,10,,,IBKR,,
"25-Apr-2023",Buy,T4,10,10,,,IBKR,,
"20230530",Buy,T5,10,10,,,IBKR,,
"""
    # --- END CORRECTION ---
    csv_file_like = StringIO(csv_data)
    df, _, ignored_idx, _, has_errors, _ = load_and_clean_transactions(
        csv_file_like, DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    assert not has_errors
    assert len(df) == 5, "All valid dates should be parsed"
    assert ignored_idx == set()
    assert df["Date"].tolist() == [
        pd.Timestamp("2023-01-15"),
        pd.Timestamp("2023-02-20"),
        pd.Timestamp("2023-03-10"),
        pd.Timestamp("2023-04-25"),
        pd.Timestamp("2023-05-30"),
    ]


def test_ignore_missing_qty_price_buy_sell():
    """Test ignoring buy/sell without quantity or price."""
    # --- CORRECTED csv_data format (Dates Quoted) ---
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Buy,SYM1,,100,,,IBKR,,
\"Jan 16, 2023\",Sell,SYM2,10,,,,IBKR,,
\"Jan 17, 2023\",Buy,SYM3,10,100,,,IBKR,,
"""
    # --- END CORRECTION ---
    csv_file_like = StringIO(csv_data)
    df, _, ignored_idx, ignored_rsn, _, _ = load_and_clean_transactions(
        csv_file_like, DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    assert len(df) == 1
    assert df.iloc[0]["Symbol"] == "SYM3"
    assert ignored_idx == {0, 1}
    assert "Missing Qty/Price" in ignored_rsn[0]
    assert "Missing Qty/Price" in ignored_rsn[1]


def test_ignore_missing_dividend_amount():
    """Test ignoring dividend without Total Amount or Price."""
    # --- CORRECTED csv_data format (Dates Quoted) ---
    csv_data = """\"Date (MMM DD, YYYY)\",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
\"Jan 15, 2023\",Dividend,AAPL,10,,,,IBKR,,
\"Jan 16, 2023\",Dividend,MSFT,10,0.5,,,IBKR,,
\"Jan 17, 2023\",Dividend,GOOG,,,10.0,,IBKR,,
"""
    # --- END CORRECTION ---
    csv_file_like = StringIO(csv_data)
    df, _, ignored_idx, ignored_rsn, _, _ = load_and_clean_transactions(
        csv_file_like, DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    assert len(df) == 2
    assert df["Symbol"].tolist() == ["MSFT", "GOOG"]
    assert ignored_idx == {0}
    assert "Missing Dividend Amt/Price" in ignored_rsn[0]


def test_load_empty_csv():
    """Test loading an empty or header-only CSV."""
    csv_data_empty = ""
    # --- CORRECTED csv_data format (Header Only) ---
    csv_data_header = """\"Date (MMM DD, YYYY)",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note\n"""
    # --- END CORRECTION ---

    df1, _, ignored1, _, err1, warn1 = load_and_clean_transactions(
        StringIO(csv_data_empty), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    assert not err1
    assert warn1
    assert df1 is None

    df2, _, ignored2, _, err2, warn2 = load_and_clean_transactions(
        StringIO(csv_data_header), DEFAULT_ACCOUNT_MAP, DEFAULT_CURRENCY
    )
    assert not err2
    assert warn2
    assert df2 is not None and df2.empty
    assert ignored2 == set()


# Add more tests as needed for edge cases, specific cleaning rules, etc.
