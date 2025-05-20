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

# --- Logging Configuration ---
# This level is a default; main_gui.py might override it for its own logging.
LOGGING_LEVEL = logging.INFO

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
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = "yf_portfolio_hist_raw_adjusted"
DAILY_RESULTS_CACHE_PATH_PREFIX = "yf_portfolio_daily_results"
YFINANCE_CACHE_DURATION_HOURS = 4
FUNDAMENTALS_CACHE_DURATION_HOURS = 24
CURRENT_QUOTE_CACHE_DURATION_MINUTES = 15  # For current stock/index quotes

# --- Yahoo Finance Mappings & Configuration ---
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())

SYMBOL_MAP_TO_YFINANCE = {
    "BRK.B": "BRK-B",
    "AAPL": "AAPL",
    "GOOG": "GOOG",
    "GOOGL": "GOOGL",
    "MSFT": "MSFT",
    "AMZN": "AMZN",
    "LQD": "LQD",
    "SPY": "SPY",
    "VTI": "VTI",
    "KHC": "KHC",
    "DIA": "DIA",
    "AXP": "AXP",
    "BLV": "BLV",
    "NVDA": "NVDA",
    "PLTR": "PLTR",
    "JNJ": "JNJ",
    "XLE": "XLE",
    "VDE": "VDE",
    "BND": "BND",
    "VWO": "VWO",
    "DPZ": "DPZ",
    "QQQ": "QQQ",
    "BHP": "BHP",
    "DAL": "DAL",
    "QSR": "QSR",
    "ASML": "ASML",
    "NLY": "NLY",
    "ADRE": "ADRE",
    "GS": "GS",
    "EPP": "EPP",
    "EFA": "EFA",
    "IBM": "IBM",
    "VZ": "VZ",
    "BBW": "BBW",
    "CVX": "CVX",
    "NKE": "NKE",
    "KO": "KO",
    "BAC": "BAC",
    "VGK": "VGK",
    "C": "C",
    "TLT": "TLT",
    "AGG": "AGG",
    "^GSPC": "^GSPC",
    "VT": "VT",
    "IWM": "IWM",
    "TCAP:BKK": "TCAP.BK",
    "BEM:BKK": "BEM.BK",
    "GENCO:BKK": "GENCO.BK",
    "NOK:BKK": "NOK.BK",
    "AOT:BKK": "AOT.BK",
    "THAI:BKK": "THAI.BK",
    "TRUE:BKK": "TRUE.BK",
    "BECL:BKK": "BECL.BK",
    "AMARIN:BKK": "AMARIN.BK",
    "PTT:BKK": "PTT.BK",
    "CPF:BKK": "CPF.BK",
    "BANPU:BKK": "BANPU.BK",
    "AAV:BKK": "AAV.BK",
    "BBL:BKK": "BBL.BK",
    "MBK:BKK": "MBK.BK",
    "ZEN:BKK": "ZEN.BK",
    "CPALL:BKK": "CPALL.BK",
    "SCC:BKK": "SCC.BK",
}

YFINANCE_EXCLUDED_SYMBOLS = {  # Symbols to generally ignore for YFinance fetching
    "BBW",
    "IDBOX",
    "IDIOX",
    "ES-Fixed_Income",
    "GENCO:BKK",
    "UOBBC",
    "ES-JUMBO25",
    "SCBCHA-SSF",
    "ES-SET50",
    "ES-Tresury",
    "UOBCG",
    "ES-GQG",
    "SCBRM1",
    "SCBRMS50",
    "AMARIN:BKK",
    "RIMM",
    "SCBSFF",
    "BANPU:BKK",
    "AAV:BKK",
    "CPF:BKK",
    "EMV",
    "IDMOX",
    "BML:BKK",
    "ZEN:BKK",
    "SCBRCTECH",
    "MBK:BKK",
    "DSV",
    "THAI:BKK",
    "IDLOX",
    "SCBRMS&P500",
    "AOT:BKK",
    "BECL:BKK",
    "TCAP:BKK",
    "KRFT",
    "AAUKY",
    "NOK:BKK",
    "ADRE",
    "SCC:BKK",
    "CPALL:BKK",
    "TRUE:BKK",
    "PTT:BKK",
    "ES-FIXED_INCOME",
    "ES-TRESURY",
    "BEM:BKK",
    # Note: DIA, SPY, QQQ, LQD are often used as benchmarks and might be fetched anyway if requested.
    # This list is more for portfolio holdings that shouldn't be looked up on YF.
}

SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}

# --- Default Settings ---
DEFAULT_CURRENCY = "USD"

# DEFAULT_CSV constant:
# In the context of a DB-first application, this is less about the *primary data source*
# and more about a *default filename for CSV operations* like export or import examples.
# main_gui.py uses DEFAULT_CSV_FOR_IMPORT_FALLBACK for prompting migration.
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
