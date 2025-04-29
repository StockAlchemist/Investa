# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          config.py
 Purpose:       Configuration constants for the Investa Portfolio Dashboard.

 Author:        Kit Matan (Derived from portfolio_logic.py)
 Author Email:  kittiwit@gmail.com

 Created:       [Date you create this file]
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import logging
from datetime import date

# --- Logging Configuration ---
# Set the desired global level here (e.g., logging.INFO, logging.DEBUG)
LOGGING_LEVEL = logging.INFO  # Or logging.DEBUG for more detail

# --- Debugging Flags (Consider removing or managing differently in production) ---
HISTORICAL_DEBUG_USD_CONVERSION = (
    False  # Set to True only when debugging USD conversion issues
)
HISTORICAL_DEBUG_SET_VALUE = (
    False  # Set to True only when debugging SET account value issues
)
DEBUG_DATE_VALUE = date(
    2024, 2, 5
)  # Example date for specific debugging, adjust as needed
HISTORICAL_DEBUG_DATE_VALUE = date(
    2025, 4, 29
)  # Example date for specific debugging, adjust as needed
HISTORICAL_DEBUG_SYMBOL = (
    None  # Optional: Focus debug logs on a specific symbol/account
)

# --- Core Symbols ---
CASH_SYMBOL_CSV = "$CASH"  # Standardized cash symbol used in transactions CSV

# --- Caching Configuration ---
DEFAULT_CURRENT_CACHE_FILE_PATH = (
    "portfolio_cache_yf.json"  # Cache for current stock/FX prices
)
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = (
    "yf_portfolio_hist_raw_adjusted"  # Prefix for raw historical data cache files
)
DAILY_RESULTS_CACHE_PATH_PREFIX = (
    "yf_portfolio_daily_results"  # Prefix for calculated daily results cache files
)
YFINANCE_CACHE_DURATION_HOURS = (
    4  # Max age in hours for the CURRENT data cache to be considered valid
)

# --- Yahoo Finance Mappings & Configuration ---
# Mapping for specific index symbols used in the header/logic to YF tickers
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(
    YFINANCE_INDEX_TICKER_MAP.keys()
)  # Default indices to query

# Mapping for specific internal stock symbols to YF tickers (e.g., handling BRK.B)
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
}

# Set of internal symbols explicitly excluded from Yahoo Finance fetching/processing
YFINANCE_EXCLUDED_SYMBOLS = set(
    [
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
    ]
)

# Set of symbols allowed for short selling logic
SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}  # Used RIMM instead of BB

# --- Default Settings ---
DEFAULT_CURRENCY = "USD"  # Default currency for calculations if not specified
