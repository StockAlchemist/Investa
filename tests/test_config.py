# tests/test_config.py

import pytest
import sys
import os

# --- Add src directory to sys.path for module import ---
# This ensures that the test runner can find the 'config' module.
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
# --- End Path Addition ---

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
        # --- Constants moved from main_gui.py ---
        DEBOUNCE_INTERVAL_MS,
        MANUAL_OVERRIDES_FILENAME,
        # APP_NAME is already tested, APP_NAME_FOR_QT was merged
        DEFAULT_API_KEY,
        CHART_MAX_SLICES,
        PIE_CHART_FIG_SIZE,
        PERF_CHART_FIG_SIZE,
        CHART_DPI,
        INDICES_FOR_HEADER,
        CSV_DATE_FORMAT,
        COMMON_CURRENCIES,
        DEFAULT_GRAPH_DAYS_AGO,
        DEFAULT_GRAPH_INTERVAL,
        DEFAULT_GRAPH_BENCHMARKS,
        BENCHMARK_MAPPING,
        BENCHMARK_OPTIONS_DISPLAY,
        COLOR_BG_DARK,
        COLOR_TEXT_DARK,
        COLOR_GAIN,
        COLOR_LOSS,  # Sample of colors
        DEFAULT_CSV,  # Ensure DEFAULT_CSV is tested
        # --- End moved constants ---
    )

except ImportError as e:
    pytest.fail(f"Failed to import from config module: {e}")


def test_import_and_types():
    """Tests if key constants can be imported and have expected types."""
    assert isinstance(CASH_SYMBOL_CSV, str)
    assert isinstance(DEFAULT_CURRENT_CACHE_FILE_PATH, str)
    assert isinstance(HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX, str)
    assert isinstance(DAILY_RESULTS_CACHE_PATH_PREFIX, str)
    assert isinstance(YFINANCE_CACHE_DURATION_HOURS, int)
    assert isinstance(YFINANCE_INDEX_TICKER_MAP, dict)
    assert isinstance(DEFAULT_INDEX_QUERY_SYMBOLS, list)
    assert isinstance(SYMBOL_MAP_TO_YFINANCE, dict)  # Remains dict, starts empty
    assert isinstance(YFINANCE_EXCLUDED_SYMBOLS, dict)  # Changed from set to dict
    assert isinstance(SHORTABLE_SYMBOLS, set)
    assert isinstance(DEFAULT_CURRENCY, str)
    assert isinstance(
        LOGGING_LEVEL, int
    )  # logging levels are ints (e.g., logging.INFO)
    assert isinstance(DEFAULT_CSV, str)  # Test for DEFAULT_CSV

    # --- Test types for constants moved from main_gui.py ---
    assert isinstance(DEBOUNCE_INTERVAL_MS, int)
    assert isinstance(MANUAL_OVERRIDES_FILENAME, str)
    assert isinstance(DEFAULT_API_KEY, str)
    assert isinstance(CHART_MAX_SLICES, int)
    assert isinstance(PIE_CHART_FIG_SIZE, tuple)
    assert isinstance(PERF_CHART_FIG_SIZE, tuple)
    assert isinstance(CHART_DPI, int)
    assert isinstance(INDICES_FOR_HEADER, list)
    assert isinstance(CSV_DATE_FORMAT, str)
    assert isinstance(COMMON_CURRENCIES, list)
    assert isinstance(DEFAULT_GRAPH_DAYS_AGO, int)
    assert isinstance(DEFAULT_GRAPH_INTERVAL, str)
    assert isinstance(DEFAULT_GRAPH_BENCHMARKS, list)
    assert isinstance(BENCHMARK_MAPPING, dict)
    assert isinstance(BENCHMARK_OPTIONS_DISPLAY, list)
    assert isinstance(COLOR_BG_DARK, str)
    assert isinstance(COLOR_TEXT_DARK, str)
    assert isinstance(COLOR_GAIN, str)
    assert isinstance(COLOR_LOSS, str)

    # Basic content checks
    assert CASH_SYMBOL_CSV == "$CASH"
    assert DEFAULT_CURRENCY == "USD"
    assert DEFAULT_CSV == "my_transactions.csv"
    assert ".DJI" in YFINANCE_INDEX_TICKER_MAP
    # assert "AAPL" in SYMBOL_MAP_TO_YFINANCE # Removed: SYMBOL_MAP_TO_YFINANCE starts empty
    # assert "BBW" in YFINANCE_EXCLUDED_SYMBOLS  # Removed: YFINANCE_EXCLUDED_SYMBOLS starts empty
    # assert "AAPL" in SHORTABLE_SYMBOLS  # Example shortable symbol

    # Basic content checks for moved constants
    assert DEBOUNCE_INTERVAL_MS == 400
    assert MANUAL_OVERRIDES_FILENAME == "manual_overrides.json"
    assert "USD" in COMMON_CURRENCIES
    assert "S&P 500" in BENCHMARK_MAPPING
    assert COLOR_GAIN == "#198754"


# Add more specific checks if needed for complex constants
