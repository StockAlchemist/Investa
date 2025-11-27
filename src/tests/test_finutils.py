# tests/test_finutils.py

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
import tempfile
import os
import hashlib
from unittest.mock import patch, mock_open  # For mocking file operations
import sys

# --- Add src directory to sys.path for module import ---
# This ensures that the test runner can find the 'finutils' module.
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

# --- Import functions from the NEW finutils module ---
try:
    from finutils import (
        _get_file_hash,
        calculate_npv,
        calculate_irr,
        get_cash_flows_for_symbol_account,
        get_cash_flows_for_mwr,
        get_conversion_rate,
        get_historical_price,
        get_historical_rate_via_usd_bridge,
        # Add other functions you moved if needed
    )

except ImportError as e:
    pytest.fail(f"Failed to import from finutils module: {e}")

# --- Tests for _get_file_hash ---


def test_get_file_hash_success():
    """Tests successful hash calculation for an existing file."""
    content = b"This is test content."
    expected_hash = hashlib.sha256(content).hexdigest()

    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(content)
        filepath = tmp_file.name

    try:
        actual_hash = _get_file_hash(filepath)
        assert actual_hash == expected_hash
    finally:
        os.remove(filepath)  # Clean up the temporary file


def test_get_file_hash_file_not_found():
    """Tests the behavior when the file does not exist."""
    non_existent_file = "non_existent_file_for_hash_test.txt"
    assert _get_file_hash(non_existent_file) == "FILE_NOT_FOUND"


@patch("builtins.open", new_callable=mock_open)  # Mock the open function
def test_get_file_hash_permission_error(mock_file_open):
    """Tests handling of PermissionError during file open."""
    mock_file_open.side_effect = PermissionError("Permission denied")
    assert _get_file_hash("dummy_path.txt") == "HASHING_ERROR_PERMISSION"


@patch("builtins.open", new_callable=mock_open)
def test_get_file_hash_io_error(mock_file_open):
    """Tests handling of IOError during file read."""
    # Simulate IOError on read, not open
    mock_file_handle = mock_file_open.return_value.__enter__.return_value
    mock_file_handle.read.side_effect = IOError("Disk read error")
    assert _get_file_hash("dummy_path.txt") == "HASHING_ERROR_IO"


# --- Tests for calculate_npv ---


def test_calculate_npv_basic():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]
    rate = 0.10
    # Expected: -100 + 110 / (1.10)^1 = 0.0
    assert calculate_npv(rate, dates, flows) == pytest.approx(0.0)


def test_calculate_npv_multiple_flows():
    dates = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]
    flows = [-100, 5, 105]
    rate = 0.05
    days1 = (dates[1] - dates[0]).days
    days2 = (dates[2] - dates[0]).days
    expected = (
        -100 + 5 / (1 + rate) ** (days1 / 365.0) + 105 / (1 + rate) ** (days2 / 365.0)
    )
    assert calculate_npv(rate, dates, flows) == pytest.approx(expected)


def test_calculate_npv_zero_rate():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]
    rate = 0.0
    assert calculate_npv(rate, dates, flows) == pytest.approx(10.0)  # Sum of flows


def test_calculate_npv_empty():
    assert calculate_npv(0.1, [], []) == 0.0


def test_calculate_npv_mismatched_lengths():
    with pytest.raises(ValueError):
        calculate_npv(0.1, [date(2023, 1, 1)], [-100, 100])


def test_calculate_npv_invalid_rate():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]
    assert np.isnan(calculate_npv(np.nan, dates, flows))
    assert np.isnan(calculate_npv(None, dates, flows))  # Should handle None gracefully


def test_calculate_npv_rate_minus_one():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]
    # NPV is undefined or infinite when rate = -1 and time > 0
    assert np.isnan(calculate_npv(-1.0, dates, flows))


def test_calculate_npv_unsorted_dates():
    dates = [date(2024, 1, 1), date(2023, 1, 1)]  # Unsorted
    flows = [110, -100]
    rate = 0.10
    # Expect NaN because the time delta calculation relies on the first date being the earliest
    assert np.isnan(calculate_npv(rate, dates, flows))


# --- Tests for calculate_irr ---


def test_calculate_irr_basic_positive():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 110]  # 10% return
    assert calculate_irr(dates, flows) == pytest.approx(0.10)


def test_calculate_irr_basic_negative():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, 90]  # -10% return
    assert calculate_irr(dates, flows) == pytest.approx(-0.10)


def test_calculate_irr_multiple_flows():
    dates = [date(2023, 1, 1), date(2023, 7, 1), date(2024, 1, 1)]
    flows = [-100, 5, 105]  # IRR should be approx 5%
    # Find the rate where NPV is zero
    assert calculate_irr(dates, flows) == pytest.approx(0.10252, abs=1e-5)


def test_calculate_irr_less_than_two_flows():
    assert np.isnan(calculate_irr([date(2023, 1, 1)], [-100]))
    assert np.isnan(calculate_irr([], []))


def test_calculate_irr_all_positive():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [100, 110]
    assert np.isnan(calculate_irr(dates, flows))


def test_calculate_irr_all_negative():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, -10]
    assert np.isnan(calculate_irr(dates, flows))


def test_calculate_irr_first_flow_positive():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [10, -100]  # First non-zero is positive
    assert np.isnan(calculate_irr(dates, flows))


def test_calculate_irr_zero_flows():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [0, 0]
    assert np.isnan(calculate_irr(dates, flows))


def test_calculate_irr_non_finite_flows():
    dates = [date(2023, 1, 1), date(2024, 1, 1)]
    flows = [-100, np.nan]
    assert np.isnan(calculate_irr(dates, flows))


def test_calculate_irr_unsorted_dates():
    dates = [date(2024, 1, 1), date(2023, 1, 1)]  # Unsorted
    flows = [110, -100]
    assert np.isnan(calculate_irr(dates, flows))


# --- Tests for get_cash_flows_for_symbol_account ---


@pytest.fixture
def sample_transactions_df():
    """Provides a sample transactions DataFrame for cash flow tests."""
    data = {
        "Date": pd.to_datetime(
            [
                "2023-01-15",
                "2023-06-01",
                "2023-09-10",
                "2023-11-01",
                "2024-01-15",
                "2024-02-20",
            ]
        ),
        "Symbol": ["AAPL", "AAPL", "AAPL", "MSFT", "AAPL", "AAPL"],
        "Account": ["IBKR", "IBKR", "IBKR", "IBKR", "IBKR", "IBKR"],
        "Type": ["buy", "dividend", "sell", "buy", "fees", "split"],
        "Quantity": [10.0, 10.0, 5.0, 20.0, np.nan, np.nan],
        "Price/Share": [150.0, 0.5, 180.0, 300.0, np.nan, np.nan],
        "Commission": [5.0, 0.0, 5.0, 10.0, 2.0, 1.0],  # Fee on split
        "Total Amount": [
            np.nan,
            5.0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],  # Dividend total amount
        "Split Ratio": [np.nan, np.nan, np.nan, np.nan, np.nan, 2.0],
        "Local Currency": ["USD", "USD", "USD", "USD", "USD", "USD"],
        "original_index": [0, 1, 2, 3, 4, 5],
    }
    return pd.DataFrame(data)


def test_get_cash_flows_irr_basic(sample_transactions_df):
    symbol = "AAPL"
    account = "IBKR"
    final_mv_local = 5.0 * 190.0  # Assume 5 shares left @ $190
    end_date = date(2024, 3, 1)

    # Expected flows for AAPL/IBKR:
    # 2023-01-15: Buy 10 @ 150 + 5 comm = -1505.0
    # 2023-06-01: Dividend 5.0 (Total Amount used) = +5.0
    # 2023-09-10: Sell 5 @ 180 - 5 comm = +895.0
    # 2024-01-15: Fees 2.0 = -2.0
    # 2024-02-20: Split Fee 1.0 = -1.0
    # 2024-03-01: Final MV = +950.0
    expected_dates = [
        date(2023, 1, 15),
        date(2023, 6, 1),
        date(2023, 9, 10),
        date(2024, 1, 15),
        date(2024, 2, 20),
        date(2024, 3, 1),
    ]
    expected_flows = [-1505.0, 5.0, 895.0, -2.0, -1.0, 950.0]

    dates, flows = get_cash_flows_for_symbol_account(
        symbol, account, sample_transactions_df, final_mv_local, is_transfer_a_flow=False, report_date=end_date
    )

    assert dates == expected_dates
    assert flows == pytest.approx(expected_flows)


def test_get_cash_flows_irr_no_transactions():
    empty_df = pd.DataFrame(
        columns=[
            "Date",
            "Symbol",
            "Account",
            "Type",
            "Quantity",
            "Price/Share",
            "Commission",
            "Total Amount",
            "Split Ratio",
            "Local Currency",
            "original_index",
        ]
    )
    dates, flows = get_cash_flows_for_symbol_account(
        "XYZ", "IBKR", empty_df, 0.0, date(2024, 1, 1)
    )
    assert dates == []
    assert flows == []


def test_get_cash_flows_irr_only_final_value(sample_transactions_df):
    # Test case where only final value exists (e.g., start date after all transactions)
    # Based on current logic, this might return empty or handle differently. Let's test its behavior.
    symbol = "AAPL"
    account = "IBKR"
    final_mv_local = 950.0
    end_date = date(2025, 1, 1)
    # Filter transactions to be empty (as if start_date is after last tx)
    tx_filtered = sample_transactions_df[
        sample_transactions_df["Date"] > pd.Timestamp(end_date)
    ]

    # The function *should* probably return empty if there are no cash flows before the end date.
    # Let's assert that. If the implementation changes, this test needs adjustment.
    dates, flows = get_cash_flows_for_symbol_account(
        symbol, account, tx_filtered, final_mv_local, is_transfer_a_flow=False, report_date=end_date
    )
    assert dates == []
    assert flows == []

def test_get_cash_flows_missing_to_account_column():
    """
    Test that get_cash_flows_for_symbol_account handles a DataFrame missing the 'To Account' column
    without raising a KeyError.
    """
    symbol = "AAPL"
    account = "MyAccount"
    
    # Create DataFrame WITHOUT 'To Account'
    data = {
        "Date": [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-15")],
        "Symbol": ["AAPL", "AAPL"],
        "Account": ["MyAccount", "MyAccount"],
        "Type": ["Buy", "Sell"],
        "Quantity": [10.0, 5.0],
        "Price/Share": [100.0, 110.0],
        "Commission": [1.0, 1.0],
        "Total Amount": [1001.0, 549.0],
        "Local Currency": ["USD", "USD"],
        "original_index": [1, 2]
    }
    df = pd.DataFrame(data)
    
    final_mv_local = 550.0 # Remaining 5 shares * 110
    
    # Should not raise KeyError
    dates, flows = get_cash_flows_for_symbol_account(
        symbol, account, df, final_mv_local, is_transfer_a_flow=False
    )
    
    assert len(dates) > 0
    assert len(flows) > 0


# --- Tests for get_conversion_rate ---


@pytest.fixture
def sample_fx_rates():
    """Provides sample current FX rates vs USD."""
    return {"EUR": 0.9, "JPY": 150.0, "THB": 35.0, "USD": 1.0}


def test_get_conversion_rate_direct(sample_fx_rates):
    # USD -> EUR: Need EUR per 1 USD
    assert get_conversion_rate("USD", "EUR", sample_fx_rates) == pytest.approx(0.9)
    # EUR -> USD: Need USD per 1 EUR = 1 / (EUR per 1 USD)
    assert get_conversion_rate("EUR", "USD", sample_fx_rates) == pytest.approx(1 / 0.9)


def test_get_conversion_rate_bridge(sample_fx_rates):
    # EUR -> JPY: (JPY/USD) / (EUR/USD)
    expected = 150.0 / 0.9
    assert get_conversion_rate("EUR", "JPY", sample_fx_rates) == pytest.approx(expected)
    # JPY -> THB: (THB/USD) / (JPY/USD)
    expected = 35.0 / 150.0
    assert get_conversion_rate("JPY", "THB", sample_fx_rates) == pytest.approx(expected)


def test_get_conversion_rate_same_currency(sample_fx_rates):
    assert get_conversion_rate("USD", "USD", sample_fx_rates) == 1.0
    assert get_conversion_rate("EUR", "EUR", sample_fx_rates) == 1.0


def test_get_conversion_rate_missing(sample_fx_rates):
    # CAD is missing from sample_fx_rates
    assert np.isnan(
        get_conversion_rate("USD", "CAD", sample_fx_rates)
    )  # MODIFIED: Expect NaN
    assert np.isnan(
        get_conversion_rate("CAD", "EUR", sample_fx_rates)
    )  # MODIFIED: Expect NaN


def test_get_conversion_rate_invalid_input(sample_fx_rates):
    assert np.isnan(get_conversion_rate(None, "USD", sample_fx_rates))
    assert np.isnan(get_conversion_rate("USD", "", sample_fx_rates))
    assert np.isnan(get_conversion_rate("USD", "EUR", None))
    assert np.isnan(get_conversion_rate("USD", "EUR", "not_a_dict"))


# --- Tests for get_historical_price ---


@pytest.fixture
def sample_prices_dict():
    """Provides sample historical price data."""
    dates1 = pd.to_datetime(["2023-01-10", "2023-01-11", "2023-01-13"]).date
    prices1 = [100.0, 101.0, 102.0]
    df1 = pd.DataFrame({"price": prices1}, index=dates1)

    dates2 = pd.to_datetime(["2023-01-10", "2023-01-12"]).date
    prices2 = [50.0, 51.0]
    df2 = pd.DataFrame({"price": prices2}, index=dates2)

    return {"AAPL": df1, "MSFT": df2, "EMPTY": pd.DataFrame(columns=["price"])}


def test_get_historical_price_exact_match(sample_prices_dict):
    assert get_historical_price(
        "AAPL", date(2023, 1, 11), sample_prices_dict
    ) == pytest.approx(101.0)


def test_get_historical_price_forward_fill(sample_prices_dict):
    # Date 2023-01-12 is missing for AAPL, should use 2023-01-11 price
    assert get_historical_price(
        "AAPL", date(2023, 1, 12), sample_prices_dict
    ) == pytest.approx(101.0)
    # Date 2023-01-14 is missing for AAPL, should use 2023-01-13 price
    assert get_historical_price(
        "AAPL", date(2023, 1, 14), sample_prices_dict
    ) == pytest.approx(102.0)


def test_get_historical_price_before_start(sample_prices_dict):
    assert get_historical_price("AAPL", date(2023, 1, 9), sample_prices_dict) is None


def test_get_historical_price_symbol_missing(sample_prices_dict):
    assert get_historical_price("GOOG", date(2023, 1, 11), sample_prices_dict) is None


def test_get_historical_price_empty_df(sample_prices_dict):
    assert get_historical_price("EMPTY", date(2023, 1, 11), sample_prices_dict) is None


def test_get_historical_price_invalid_date(sample_prices_dict):
    assert get_historical_price("AAPL", None, sample_prices_dict) is None
    assert (
        get_historical_price("AAPL", "2023-01-11", sample_prices_dict) is None
    )  # Must be date object


# --- Tests for get_historical_rate_via_usd_bridge ---


# Mock function to simulate get_historical_price behavior
def mock_get_historical_price(symbol_key, target_date, prices_dict):
    # Simple mock: return predefined values for specific symbol/date combinations
    if symbol_key == "EUR=X" and target_date == date(2023, 5, 15):
        return 0.92
    if symbol_key == "JPY=X" and target_date == date(2023, 5, 15):
        return 145.0
    if symbol_key == "EUR=X" and target_date == date(
        2023, 5, 16
    ):  # Simulate missing JPY rate
        return 0.93
    # Add more cases as needed
    return None  # Default to None if not matched


@patch("finutils.get_historical_price", side_effect=mock_get_historical_price)
def test_get_historical_rate_bridge_success(mock_get_price):
    target_date = date(2023, 5, 15)
    dummy_hist_fx_data = {}  # The mock doesn't actually use this dict content

    # EUR -> JPY: (JPY/USD) / (EUR/USD) = 145.0 / 0.92
    expected = 145.0 / 0.92
    assert get_historical_rate_via_usd_bridge(
        "EUR", "JPY", target_date, dummy_hist_fx_data
    ) == pytest.approx(expected)


@patch("finutils.get_historical_price", side_effect=mock_get_historical_price)
def test_get_historical_rate_bridge_to_usd(mock_get_price):
    target_date = date(2023, 5, 15)
    dummy_hist_fx_data = {}
    # EUR -> USD: 1 / (EUR/USD) = 1 / 0.92
    expected = 1 / 0.92
    assert get_historical_rate_via_usd_bridge(
        "EUR", "USD", target_date, dummy_hist_fx_data
    ) == pytest.approx(expected)


@patch("finutils.get_historical_price", side_effect=mock_get_historical_price)
def test_get_historical_rate_bridge_from_usd(mock_get_price):
    target_date = date(2023, 5, 15)
    dummy_hist_fx_data = {}
    # USD -> JPY: (JPY/USD) / (USD/USD) = 145.0 / 1.0
    expected = 145.0
    assert get_historical_rate_via_usd_bridge(
        "USD", "JPY", target_date, dummy_hist_fx_data
    ) == pytest.approx(expected)


@patch("finutils.get_historical_price", side_effect=mock_get_historical_price)
def test_get_historical_rate_bridge_missing_rate(mock_get_price):
    target_date = date(2023, 5, 16)  # JPY=X rate is missing on this date in mock
    dummy_hist_fx_data = {}
    # EUR -> JPY: Should fail because JPY/USD is None
    assert np.isnan(
        get_historical_rate_via_usd_bridge(
            "EUR", "JPY", target_date, dummy_hist_fx_data
        )
    )


def test_get_historical_rate_bridge_same_currency():
    # No mocking needed
    assert (
        get_historical_rate_via_usd_bridge("USD", "USD", date(2023, 5, 15), {}) == 1.0
    )


def test_get_historical_rate_bridge_invalid_input():
    # No mocking needed
    assert np.isnan(
        get_historical_rate_via_usd_bridge(None, "USD", date(2023, 5, 15), {})
    )
    assert np.isnan(get_historical_rate_via_usd_bridge("EUR", "USD", "2023-05-15", {}))
    assert np.isnan(
        get_historical_rate_via_usd_bridge(
            "EUR", "USD", date(2023, 5, 15), "not_a_dict"
        )
    )


# --- Tests for get_cash_flows_for_mwr ---
# These are lower priority but follow a similar pattern to IRR tests,
# requiring mocking of get_conversion_rate.


@pytest.fixture
def sample_transactions_mwr_df():
    """Sample transactions across different accounts/currencies."""
    data = {
        "Date": pd.to_datetime(
            ["2023-01-15", "2023-02-10", "2023-03-05", "2023-04-20"]
        ),
        "Symbol": ["MSFT", "$CASH", "VTI", "$CASH"],
        "Account": ["IBKR", "SET", "IBKR", "SET"],
        "Type": ["buy", "deposit", "sell", "withdrawal"],
        "Quantity": [10.0, 50000.0, 5.0, 10000.0],
        "Price/Share": [300.0, np.nan, 210.0, np.nan],
        "Commission": [5.0, 100.0, 5.0, 50.0],  # Note: Comms on cash tx
        "Total Amount": [np.nan, np.nan, np.nan, np.nan],
        "Split Ratio": [np.nan, np.nan, np.nan, np.nan],
        "Local Currency": ["USD", "THB", "USD", "THB"],  # Different local currencies
        "original_index": [10, 11, 12, 13],
    }
    return pd.DataFrame(data)


# Mock function for get_conversion_rate
def mock_get_conversion_rate(from_curr, to_curr, fx_rates):
    if from_curr == to_curr:
        return 1.0
    if fx_rates is None:
        return 1.0  # Fallback
    rate_from_usd = fx_rates.get(from_curr, np.nan)
    rate_to_usd = fx_rates.get(to_curr, np.nan)
    if from_curr == "USD":
        rate_from_usd = 1.0
    if to_curr == "USD":
        rate_to_usd = 1.0

    if pd.isna(rate_from_usd) or pd.isna(rate_to_usd) or abs(rate_from_usd) < 1e-9:
        return 1.0  # Fallback if rates missing or invalid denominator
    return rate_to_usd / rate_from_usd


@patch("finutils.get_conversion_rate", side_effect=mock_get_conversion_rate)
def test_get_cash_flows_mwr_basic(mock_conv_rate, sample_transactions_mwr_df):
    account = "IBKR"
    target_currency = "USD"
    final_mv_target = 5 * 310.0  # Assume 5 MSFT left @ $310
    end_date = date(2023, 5, 1)
    # Dummy fx_rates - mock doesn't use it, but pass something
    fx_rates_dummy = {"USD": 1.0}

    account_tx = sample_transactions_mwr_df[
        sample_transactions_mwr_df["Account"] == account
    ]

    # Expected flows for IBKR (USD -> USD, MWR sign flip):
    # 2023-01-15: Buy 10 MSFT @ 300 + 5 comm = -3005 -> +3005 (MWR)
    # 2023-03-05: Sell 5 VTI @ 210 - 5 comm = +1045 -> -1045 (MWR)
    # 2023-05-01: Final MV = +1550
    expected_dates = [date(2023, 1, 15), date(2023, 3, 5), date(2023, 5, 1)]
    expected_flows = [3005.0, -1045.0, 1550.0]

    dates, flows = get_cash_flows_for_mwr(
        account_tx,
        final_mv_target,
        end_date,
        target_currency,
        fx_rates_dummy,
        target_currency,
    )

    assert dates == expected_dates
    assert flows == pytest.approx(expected_flows)


@patch("finutils.get_conversion_rate", side_effect=mock_get_conversion_rate)
def test_get_cash_flows_mwr_conversion(mock_conv_rate, sample_transactions_mwr_df):
    account = "SET"
    target_currency = "USD"
    final_mv_target = 40000.0 / 35.0  # Assume 40k THB left, convert to USD @ 35 THB/USD
    end_date = date(2023, 5, 1)
    # Provide rates for the mock to use (even though mock logic is simplified)
    fx_rates_for_mock = {"THB": 35.0, "USD": 1.0}

    account_tx = sample_transactions_mwr_df[
        sample_transactions_mwr_df["Account"] == account
    ]

    # Expected flows for SET (THB -> USD @ 35, MWR sign flip):
    # 2023-02-10: Deposit 50000 THB - 100 comm = +49900 THB -> +49900 / 35 USD (MWR)
    # 2023-04-20: Withdraw 10000 THB - 50 comm = -10050 THB -> -(-10050 / 35) USD (MWR) = +10050 / 35 USD
    # 2023-05-01: Final MV = +40000 / 35 USD
    expected_dates = [date(2023, 2, 10), date(2023, 4, 20), date(2023, 5, 1)]
    expected_flows = [-49900.0 / 35.0, 10050.0 / 35.0, 40000.0 / 35.0]

    dates, flows = get_cash_flows_for_mwr(
        account_tx,
        final_mv_target,
        end_date,
        target_currency,
        fx_rates_for_mock,
        target_currency,
    )

    assert dates == expected_dates
    assert flows == pytest.approx(expected_flows)
