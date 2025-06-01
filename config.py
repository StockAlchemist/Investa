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
DAILY_RESULTS_CACHE_PATH_PREFIX = "yf_portfolio_daily_results"
YFINANCE_CACHE_DURATION_HOURS = 24
FUNDAMENTALS_CACHE_DURATION_HOURS = 24 * 30 * 3
CURRENT_QUOTE_CACHE_DURATION_MINUTES = 24 * 60  # For current stock/index quotes

# --- Logging Configuration ---
# This level is a default; main_gui.py might override it for its own logging.
LOGGING_LEVEL = logging.DEBUG

# --- Application Name (used by QStandardPaths if it doesn't infer from bundle) ---
APP_NAME = "Investa"  # Used by db_utils.py for fallback folder if QStandardPaths fails

# --- Debugging Flags ---
HISTORICAL_DEBUG_USD_CONVERSION = False
HISTORICAL_DEBUG_SET_VALUE = False
DEBUG_DATE_VALUE = date(2024, 2, 5)
HISTORICAL_DEBUG_DATE_VALUE = date(2025, 4, 29)
HISTORICAL_DEBUG_SYMBOL = None

# --- Core Symbols ---
CASH_SYMBOL_CSV = "$CASH"  # Standardized cash symbol

# --- Caching Configuration ---
DEFAULT_CURRENT_CACHE_FILE_PATH = (
    "portfolio_cache_yf.json"  # Used by MarketDataProvider default
)

# --- Yahoo Finance Mappings & Configuration ---
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())

SYMBOL_MAP_TO_YFINANCE = {}

YFINANCE_EXCLUDED_SYMBOLS = {}

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
HISTORICAL_COMPARE_METHODS = False

# --- Position Closing Tolerance ---
STOCK_QUANTITY_CLOSE_TOLERANCE = 1e-6

# --- Dividend Chart Configuration ---
DIVIDEND_CHART_DEFAULT_PERIODS_ANNUAL = 10
DIVIDEND_CHART_DEFAULT_PERIODS_QUARTERLY = 12  # (3 years)
DIVIDEND_CHART_DEFAULT_PERIODS_MONTHLY = 24  # (2 years)

# --- Constants moved from main_gui.py ---
DEBOUNCE_INTERVAL_MS = 400  # Debounce interval for live table filtering
MANUAL_OVERRIDES_FILENAME = (
    "manual_overrides.json"  # Filename for manual overrides (prices, sectors, etc.)
)

DEFAULT_API_KEY = ""  # Default API key (e.g., for FMP, though currently unused directly by yfinance logic)

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

BENCHMARK_OPTIONS_DISPLAY = [  # Order for UI display of benchmark options
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
