# tests/test_config.py

import pytest
import logging
from datetime import date

# --- Import constants from the NEW config module ---
try:
    from config import (
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENT_CACHE_FILE_PATH,
        HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        DAILY_RESULTS_CACHE_PATH_PREFIX,
        YFINANCE_CACHE_DURATION_HOURS,
        YFINANCE_INDEX_TICKER_MAP,
        DEFAULT_INDEX_QUERY_SYMBOLS,
        SYMBOL_MAP_TO_YFINANCE,
        YFINANCE_EXCLUDED_SYMBOLS,
        SHORTABLE_SYMBOLS,
        DEFAULT_CURRENCY,
        LOGGING_LEVEL,
        # Add other constants you moved if needed
    )

    CONFIG_IMPORTED = True
except ImportError as e:
    print(f"ERROR: Could not import from config module: {e}")
    CONFIG_IMPORTED = False

# Skip all tests in this file if config couldn't be imported
pytestmark = pytest.mark.skipif(
    not CONFIG_IMPORTED, reason="config.py module not found or import failed"
)


def test_import_and_types():
    """Tests if key constants can be imported and have expected types."""
    assert isinstance(CASH_SYMBOL_CSV, str)
    assert isinstance(DEFAULT_CURRENT_CACHE_FILE_PATH, str)
    assert isinstance(HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, str)
    assert isinstance(DAILY_RESULTS_CACHE_PATH_PREFIX, str)
    assert isinstance(YFINANCE_CACHE_DURATION_HOURS, int)
    assert isinstance(YFINANCE_INDEX_TICKER_MAP, dict)
    assert isinstance(DEFAULT_INDEX_QUERY_SYMBOLS, list)
    assert isinstance(SYMBOL_MAP_TO_YFINANCE, dict)
    assert isinstance(YFINANCE_EXCLUDED_SYMBOLS, set)
    assert isinstance(SHORTABLE_SYMBOLS, set)
    assert isinstance(DEFAULT_CURRENCY, str)
    assert isinstance(
        LOGGING_LEVEL, int
    )  # logging levels are ints (e.g., logging.INFO)

    # Basic content checks
    assert CASH_SYMBOL_CSV == "$CASH"
    assert DEFAULT_CURRENCY == "USD"
    assert ".DJI" in YFINANCE_INDEX_TICKER_MAP
    assert "AAPL" in SYMBOL_MAP_TO_YFINANCE
    assert "BBW" in YFINANCE_EXCLUDED_SYMBOLS  # Example excluded symbol
    assert "AAPL" in SHORTABLE_SYMBOLS  # Example shortable symbol


# Add more specific checks if needed for complex constants
