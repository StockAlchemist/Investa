# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          config.py
 Purpose:       Configuration constants for the Investa Portfolio Dashboard.
                *** MODIFIED: Reviewed for DB migration context. ***

 Author:        Kit Matan (Derived from portfolio_logic.py) and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import logging
from datetime import date

# --- Cache Configuration ---
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = "yf_portfolio_hist_raw_adjusted"
DAILY_RESULTS_CACHE_PATH_PREFIX = (
    "yf_portfolio_daily_results"  # For daily calculated values
)
YFINANCE_CACHE_DURATION_HOURS = 24
FUNDAMENTALS_CACHE_DURATION_HOURS = 24 * 30 * 3  # Cache fundamentals for 3 months
CURRENT_QUOTE_CACHE_DURATION_MINUTES = 1  # For current stock/index quotes (prices, fx)

# --- Logging Configuration ---
# This level is a default; main_gui.py might override it for its own logging.
LOGGING_LEVEL = logging.WARNING

# --- Application Name (used by QStandardPaths if it doesn't infer from bundle) ---
APP_NAME = "Investa"  # Used by db_utils.py for fallback folder if QStandardPaths fails

# --- Debugging Flags ---
HISTORICAL_DEBUG_USD_CONVERSION = False
HISTORICAL_DEBUG_SET_VALUE = False
DEBUG_DATE_VALUE = date(2024, 2, 5)  # General debug date
HISTORICAL_DEBUG_DATE_VALUE = date(2025, 4, 29)  # Specific for historical calcs
HISTORICAL_DEBUG_SYMBOL = None

# --- Core Symbols ---
CASH_SYMBOL_CSV = "$CASH"  # Standardized cash symbol

# --- Define a constant for the aggregated cash account name ---
_AGGREGATE_CASH_ACCOUNT_NAME_ = "Cash"

# --- Caching Configuration ---
DEFAULT_CURRENT_CACHE_FILE_PATH = (
    "portfolio_cache_yf.json"  # Used by MarketDataProvider default for current quotes
)

# --- Yahoo Finance Mappings & Configuration ---
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())

SYMBOL_MAP_TO_YFINANCE = {}
# {
# "BRK.B": "BRK-B",  # Example: Map internal "BRK.B" to YF "BRK-B"
# "SET:ADVANC": "ADVANC.BK" # Example for Thai stock
# }

YFINANCE_EXCLUDED_SYMBOLS = {}
# {"VTSAX", "VTIAX"} # Example: Mutual funds often don't have good YF data
SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}

# DEFAULT_CSV constant:
# In the context of a DB-first application, this is less about the *primary data source*
# and more about a *default filename for CSV operations* like export or import examples.
# main_gui.py uses DEFAULT_CSV_FOR_IMPORT_FALLBACK for prompting migration.
# DEFAULT_CSV is also used in PortfolioApp.load_config as a fallback.
# Note: main_gui.py previously redefined this; ensure it now imports from here.
DEFAULT_CSV = "my_transactions.csv"  # Default name for CSV related operations (e.g. export suggestion)

# --- Calculation Method Configuration ---
HISTORICAL_CALC_METHOD = "numba"  # Options: 'python', 'numba'
HISTORICAL_COMPARE_METHODS = (
    False  # If true, historical calc will run both methods and log diffs
)

# --- Position Closing Tolerance ---
STOCK_QUANTITY_CLOSE_TOLERANCE = (
    1e-6  # Quantities smaller than this are considered closed
)

# --- Bar Chart Configuration ---
BAR_CHART_MAX_PERIODS_ANNUAL = 50
BAR_CHART_MAX_PERIODS_QUARTERLY = 60  # 15 years
BAR_CHART_MAX_PERIODS_MONTHLY = 120  # 10 years
BAR_CHART_MAX_PERIODS_WEEKLY = 260  # 5 years
BAR_CHART_MAX_PERIODS_DAILY = 365  # 1 year


# --- Dividend Chart Configuration ---
DIVIDEND_CHART_DEFAULT_PERIODS_ANNUAL = 10
DIVIDEND_CHART_DEFAULT_PERIODS_QUARTERLY = 12  # (3 years)
DIVIDEND_CHART_DEFAULT_PERIODS_MONTHLY = 24  # (2 years)

# --- Constants moved from main_gui.py ---
DEBOUNCE_INTERVAL_MS = (
    400  # Debounce interval for live table filtering (e.g., holdings table)
)
MANUAL_OVERRIDES_FILENAME = (
    "manual_overrides.json"  # Filename for manual overrides (prices, sectors, etc.)
)

DEFAULT_API_KEY = (
    ""  # Default API key (e.g., for FMP, though currently unused by yfinance logic)
)

CHART_MAX_SLICES = 10  # Max slices before grouping into 'Other' in pie charts
PIE_CHART_FIG_SIZE = (6.5, 4.25)  # Figure size for pie charts
PERF_CHART_FIG_SIZE = (7.5, 3.0)  # Figure size for performance graphs
CHART_DPI = 95  # Dots per inch for charts

INDICES_FOR_HEADER = [".DJI", "IXIC", ".INX"]  # Indices to display in the header bar
CSV_DATE_FORMAT = "%b %d, %Y"  # Date format used in CSV files (e.g., "Jan 01, 2023")

# --- Default Settings ---
DEFAULT_CURRENCY = "USD"

COMMON_CURRENCIES = [  # List of commonly used currency codes
    "USD",
    "EUR",
    "GBP",
    "JPY",
    "CAD",
    "AUD",
    "CHF",
    "CNY",
    "HKD",
    "SGD",
    "THB",
]

# --- Graph Defaults ---
DEFAULT_GRAPH_DAYS_AGO = 365 * 2  # Default start for graphs, days relative to today
DEFAULT_GRAPH_INTERVAL = "W"  # Default graph interval (D, W, M)

# --- Benchmark Definitions ---
# DEFAULT_GRAPH_BENCHMARKS uses display names.
DEFAULT_GRAPH_BENCHMARKS = [
    "S&P 500"
]  # Default benchmarks selected in the UI (user-friendly names)
# Maps user-friendly benchmark names to Yahoo Finance tickers
BENCHMARK_MAPPING = {  # Maps user-friendly benchmark names to Yahoo Finance tickers
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",
    "SPY (S&P 500 ETF)": "SPY",
    "QQQ (Nasdaq 100 ETF)": "QQQ",
    "DIA (Dow Jones ETF)": "DIA",
    "S&P 500 Total Return": "^SP500TR",  # Note: Total return indices might need specific handling or data source
}

BENCHMARK_OPTIONS_DISPLAY = [  # Order for UI display of benchmark options in menu
    "S&P 500",
    "Dow Jones",
    "NASDAQ",
    "Russell 2000",
    "SPY (S&P 500 ETF)",
    "QQQ (Nasdaq 100 ETF)",
    "DIA (Dow Jones ETF)",
    "S&P 500 Total Return",
]

# --- Theme Colors (Hex Strings) ---
COLOR_BG_DARK = "#FFFFFF"
COLOR_BG_HEADER_LIGHT = "#F8F9FA"
COLOR_BG_HEADER_ORIGINAL = "#495057"
COLOR_TEXT_DARK = "#212529"
COLOR_TEXT_SECONDARY = "#6C757D"
COLOR_ACCENT_TEAL = "#6C757D"  # Main accent color, often used for portfolio lines
COLOR_BORDER_LIGHT = "#DEE2E6"
COLOR_BORDER_DARK = "#ADB5BD"
COLOR_GAIN = "#198754"
COLOR_LOSS = "#DC3545"

# --- Dark Theme Color Palette (Hex Strings) ---
# These constants define the color scheme for the dark theme.
DARK_COLOR_BG_DARK = "#1e1e1e"  # Main background color
DARK_COLOR_BG_LIGHT = "#2c2c2c"  # Lighter background for elements like cards, headers
DARK_COLOR_TEXT_DARK = "#e0e0e0"  # Primary text color (on dark backgrounds)
DARK_COLOR_TEXT_LIGHT = "#b0b0b0"  # Secondary/dimmer text color
DARK_COLOR_HEADER_TEXT = "#f0f0f0"  # Text color specifically for headers
DARK_COLOR_BORDER = "#444444"  # Border color for tables, widgets, etc.
DARK_COLOR_ACCENT_PRIMARY = (
    "#007bff"  # Primary accent color (e.g., buttons, highlights)
)
DARK_COLOR_ACCENT_SECONDARY = "#6c757d"  # Secondary accent color
DARK_COLOR_GAIN = "#28a745"  # Color for indicating financial gains
DARK_COLOR_LOSS = "#dc3545"  # Color for indicating financial losses
DARK_COLOR_TABLE_ALT_ROW = "#252525"  # Background color for alternating rows in tables
DARK_COLOR_INPUT_BG = "#2a2a2a"  # Background for input fields
DARK_COLOR_INPUT_TEXT = "#e0e0e0"  # Text color for input fields
DARK_COLOR_INPUT_BORDER = "#555555"  # Border color for input fields
