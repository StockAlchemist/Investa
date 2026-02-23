import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from market_fallback import fetch_data_fallback, fetch_info_fallback

@patch('market_fallback.requests.get')
def test_fetch_data_fallback_success(mock_get):
    """
    Test that fetch_data_fallback correctly parses an Alpha Vantage daily response
    and returns a properly formatted DataFrame.
    """
    # Mock Alpha Vantage JSON response shape
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Meta Data": {
            "1. Information": "Daily Prices (open, high, low, close) and Volumes",
            "2. Symbol": "AAPL",
            "3. Last Refreshed": "2026-02-20",
        },
        "Time Series (Daily)": {
            "2026-02-20": {
                "1. open": "150.00",
                "2. high": "152.00",
                "3. low": "149.00",
                "4. close": "151.00",
                "5. volume": "1000000"
            },
            "2026-02-19": {
                "1. open": "148.00",
                "2. high": "151.00",
                "3. low": "147.00",
                "4. close": "150.00",
                "5. volume": "900000"
            }
        }
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    # Call the fallback function
    df = fetch_data_fallback(["AAPL"], "2026-02-19", "2026-02-20", "1d")

    # Assertions
    assert df is not None
    assert not df.empty
    assert "Open" in df.columns
    assert "Close" in df.columns
    assert "Volume" in df.columns

    # Verify data types and values
    assert df.loc["2026-02-20", "Close"] == 151.0
    assert df.loc["2026-02-20", "Volume"] == 1000000.0

@patch('market_fallback.requests.get')
def test_fetch_data_fallback_api_limit(mock_get):
    """
    Test that fetch_data_fallback detects Alpha Vantage rate limits and handles it gracefully.
    """
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "Information": "Thank you for using Alpha Vantage! Our standard API call frequency is 5 calls per minute and 500 calls per day."
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    df = fetch_data_fallback(["AAPL"], None, None, "1d")

    # Assuming we change the condition slightly or "Information" isn't a "Time Series" match
    assert df is None

@patch('market_fallback.requests.get')
def test_fetch_info_fallback_success(mock_get, monkeypatch):
    """
    Test that fetch_info_fallback correctly parses a Finnhub quote response.
    """
    # Ensure FINNHUB_API_KEY is conditionally active for the mock test
    monkeypatch.setattr('market_fallback.FINNHUB_API_KEY', "test_key")

    mock_response = MagicMock()
    mock_response.json.return_value = {
        "c": 261.74, # Current
        "d": 1.88,   # Change
        "dp": 0.723, # Percent change
        "h": 263.31, # High
        "l": 260.68, # Low
        "o": 261.07, # Open
        "pc": 259.86,# Previous Close
        "t": 1582641000
    }
    mock_response.raise_for_status.return_value = None
    mock_get.return_value = mock_response

    info = fetch_info_fallback("MSFT")

    assert info is not None
    assert info["symbol"] == "MSFT"
    assert info["currentPrice"] == 261.74
    assert info["previousClose"] == 259.86
    assert info["_fallback_used"] is True
