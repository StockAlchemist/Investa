# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          config.py
 Purpose:       Configuration constants for the Investa Portfolio Dashboard.


 Author:        Google Gemini (Derived from portfolio_logic.py)

 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import logging
import os
import platform
from datetime import date
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- API Keys ---
# FMP API Key is managed here but can also be overridden by env
FMP_API_KEY = os.getenv("FMP_API_KEY") # Prioritize ENV

# Gemini API Key for AI Stock Analysis
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# --- Cache Configuration ---
HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = "yf_portfolio_hist_raw_adjusted_v2"
DAILY_RESULTS_CACHE_PATH_PREFIX = (
    "yf_portfolio_daily_results"  # For daily calculated values
)
YFINANCE_CACHE_DURATION_HOURS = 24
FUNDAMENTALS_CACHE_DURATION_HOURS = 24 * 30 * 3  # Cache fundamentals for 3 months
CURRENT_QUOTE_CACHE_DURATION_MINUTES = 1  # For current stock/index quotes (prices, fx)
METADATA_CACHE_FILE_NAME = "yf_metadata_cache.json"  # Long-lived cache for static data (Name, Currency)
METADATA_CACHE_DURATION_DAYS = 30
AI_REVIEW_CACHE_TTL = 604800  # 1 week for AI stock analysis

# --- Logging Configuration ---
# This level is a default; main_gui.py might override it for its own logging.
LOGGING_LEVEL = logging.WARNING  # Default logging level for the application

# --- Application Name (used by QStandardPaths if it doesn't infer from bundle) ---
APP_NAME = "Investa"  # Used by db_utils.py for fallback folder if QStandardPaths fails
APP_VERSION = "1.0.1"
ORG_NAME = "StockAlchemist"  # Used for cache path consistency


def get_app_data_dir() -> str:
    """
    Returns the centralized application data directory.
    - macOS: Standard AppDataLocation (~/Library/Application Support/StockAlchemist/Investa)
    - Linux: Consolidated hidden folder in home (~/.investa)
    - Windows: Standard AppDataLocation (~/AppData/Roaming/StockAlchemist/Investa)
    """
    try:
        from PySide6.QtCore import QStandardPaths
        pyside_available = True
    except ImportError:
        pyside_available = False

    system = platform.system()

    # Priority 1: Consolidated Linux behavior
    if system == "Linux":
        home = os.path.expanduser("~")
        linux_path = os.path.join(home, f".{APP_NAME.lower()}")
        try:
            os.makedirs(linux_path, exist_ok=True)
            return linux_path
        except Exception:
            pass  # Fallthrough to standard logic if mkdir fails

    # Priority 2: Standard OS-specific logic via PySide if available
    if pyside_available:
        path = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not path:
            path = QStandardPaths.writableLocation(QStandardPaths.AppConfigLocation)

        if path:
            # If QStandardPaths didn't include the app name (common in scripts), append it manually
            expected_suffix = os.path.join(ORG_NAME, APP_NAME)
            if APP_NAME not in path:
                path = os.path.join(path, expected_suffix)
            
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except Exception:
                pass

    # Priority 3: Cross-platform fallback based on HOME
    home = os.path.expanduser("~")
    fallback_path = os.path.join(home, f".{APP_NAME.lower()}")
    try:
        os.makedirs(fallback_path, exist_ok=True)
        return fallback_path
    except Exception:
        # Final fallback: current working directory
        return os.getcwd()


def get_app_cache_dir() -> Optional[str]:
    """
    Returns the centralized application cache directory (e.g. ~/Library/Caches/StockAlchemist/Investa).
    """
    try:
        from PySide6.QtCore import QStandardPaths
        pyside_available = True
    except ImportError:
        pyside_available = False

    if pyside_available:
        path = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        if path:
            # If QStandardPaths didn't include the app name, append it manually
            # On macOS, it's usually already correct if the app is bundled, 
            # but for scripts we might need to be sure.
            if APP_NAME not in path:
                path = os.path.join(path, ORG_NAME, APP_NAME)
            
            try:
                os.makedirs(path, exist_ok=True)
                return path
            except Exception:
                pass

    # Fallback: Check standard locations manually
    system = platform.system()
    home = os.path.expanduser("~")
    if system == "Darwin":
        path = os.path.join(home, "Library", "Caches", ORG_NAME, APP_NAME)
    elif system == "Windows":
        path = os.path.join(os.environ.get("LOCALAPPDATA", home), ORG_NAME, APP_NAME, "Cache")
    else: # Linux/Other
        path = os.path.join(home, ".cache", APP_NAME.lower())

    try:
        os.makedirs(path, exist_ok=True)
        return path
    except Exception:
        return None

# --- Debugging Flags ---
HISTORICAL_DEBUG_USD_CONVERSION = False
HISTORICAL_DEBUG_SET_VALUE = False
DEBUG_DATE_VALUE = date(2024, 2, 5)  # General debug date
HISTORICAL_DEBUG_DATE_VALUE = date(2025, 4, 29)  # Specific for historical calcs
HISTORICAL_DEBUG_SYMBOL = None

# --- Core Symbols ---
CASH_SYMBOL_CSV = "$CASH"  # Standardized cash symbol

# --- Define a constant for the aggregated cash account name in the holdings view ---
_AGGREGATE_CASH_ACCOUNT_NAME_ = "Cash"

# --- Caching Configuration ---
DEFAULT_CURRENT_CACHE_FILE_PATH = (
    "portfolio_cache_yf.json"  # Used by MarketDataProvider default for current quotes
)

# --- Yahoo Finance Mappings & Configuration ---
YFINANCE_INDEX_TICKER_MAP = {".DJI": "^DJI", "IXIC": "^IXIC", ".INX": "^GSPC"}
DEFAULT_INDEX_QUERY_SYMBOLS = list(YFINANCE_INDEX_TICKER_MAP.keys())

SYMBOL_MAP_TO_YFINANCE = {
    "BRK.B": "BRK-B",
    "BECL.BK": "BEM.BK",  # BECL merged into BEM
    "NOK.BK": "NOK.BK",   # Verify if valid, or maybe delisted
}

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
HISTORICAL_CALC_METHOD = "numba_chrono"  # Options: 'python', 'numba', 'numba_chrono'
HISTORICAL_COMPARE_METHODS = False  # If true, historical calc will run both methods and log diffs

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

# --- Asset Allocation & Symbol Settings ---
STANDARD_ASSET_TYPES = [
    "",  # Blank for 'not set'
    "STOCK",
    "ETF",
    "MUTUALFUND",
    "CURRENCY",
    "INDEX",
    "FUTURE",
    "OPTION",
    "CRYPTOCURRENCY",
    "Other",
]

STANDARD_SECTORS = sorted(
    list(
        set(
            [
                "",  # Blank for 'not set'
                "Other",
                "Basic Materials",
                "Communication Services",
                "Consumer Cyclical",
                "Consumer Defensive",
                "Energy",
                "Financial Services",
                "Healthcare",
                "Industrials",
                "Real Estate",
                "Technology",
                "Utilities",
                "Exchange-Traded Fund",
            ]
        )
    )
)

# Mapping of sectors to industries for dependent dropdowns
SECTOR_INDUSTRY_MAP = {
    "": [""],  # For when no sector is selected
    "Other": ["Other"],
    "Basic Materials": sorted(
        [
            "Agricultural Inputs",
            "Aluminum",
            "Building Materials",
            "Chemicals",
            "Coking Coal",
            "Copper",
            "Gold",
            "Lumber & Wood Production",
            "Other Industrial Metals & Mining",
            "Other Precious Metals & Mining",
            "Paper & Paper Products",
            "Silver",
            "Specialty Chemicals",
            "Steel",
            "Thematic",
        ]
    ),
    "Communication Services": sorted(
        [
            "Advertising Agencies",
            "Broadcasting",
            "Communication Equipment",
            "Electronic Gaming & Multimedia",
            "Entertainment",
            "Internet Content & Information",
            "Publishing",
            "Telecom Services",
            "Thematic",
        ]
    ),
    "Consumer Cyclical": sorted(
        [
            "Apparel Manufacturing",
            "Apparel Retail",
            "Auto & Truck Dealerships",
            "Auto Manufacturers",
            "Auto Parts",
            "Department Stores",
            "Footwear & Accessories",
            "Furnishings, Fixtures & Appliances",
            "Gambling",
            "Home Improvement Retail",
            "Internet Retail",
            "Leisure",
            "Lodging",
            "Luxury Goods",
            "Packaging & Containers",
            "Personal Services",
            "Recreational Vehicles",
            "Residential Construction",
            "Resorts & Casinos",
            "Restaurants",
            "Specialty Retail",
            "Textile Manufacturing",
            "Travel Services",
            "Thematic",
        ]
    ),
    "Consumer Defensive": sorted(
        [
            "Beverages - Brewers",
            "Beverages - Non-Alcoholic",
            "Beverages - Wineries & Distilleries",
            "Confectioners",
            "Discount Stores",
            "Education & Training Services",
            "Farm Products",
            "Food Distribution",
            "Grocery Stores",
            "Household & Personal Products",
            "Packaged Foods",
            "Pharmaceutical Retailers",
            "Tobacco",
            "Thematic",
        ]
    ),
    "Energy": sorted(
        [
            "Oil & Gas Drilling",
            "Oil & Gas E&P",
            "Oil & Gas Equipment & Services",
            "Oil & Gas Integrated",
            "Oil & Gas Midstream",
            "Oil & Gas Refining & Marketing",
            "Thermal Coal",
            "Uranium",
            "Thematic",
        ]
    ),
    "Exchange-Traded Fund": sorted(
        [
            "Broad Market - Emerging Markets",
            "Broad Market - International Developed",
            "Broad Market - US",
            "Commodity - Gold",
            "Commodity - Oil",
            "Commodity - Silver",
            "Dividend - Growth",
            "Dividend - High Yield",
            "Factor - Growth",
            "Factor - Low Volatility",
            "Factor - Momentum",
            "Factor - Quality",
            "Factor - Value",
            "Fixed Income - Corporate Bond",
            "Fixed Income - Government Bond (Long-Term)",
            "Fixed Income - Government Bond (Short-Term)",
            "Fixed Income - High-Yield Bond",
            "Fixed Income - Municipal Bond",
            "Sector - Communication Services",
            "Sector - Consumer Cyclical",
            "Sector - Consumer Defensive",
            "Sector - Energy",
            "Sector - Financials",
            "Sector - Healthcare",
            "Sector - Industrials",
            "Sector - Materials",
            "Sector - Real Estate",
            "Sector - Technology",
            "Sector - Utilities",
        ]
    ),
    "Financial Services": sorted(
        [
            "Asset Management",
            "Banks - Diversified",
            "Banks - Regional",
            "Capital Markets",
            "Credit Services",
            "Financial Conglomerates",
            "Financial Data & Stock Exchanges",
            "Insurance - Diversified",
            "Insurance - Life",
            "Insurance - Property & Casualty",
            "Insurance - Reinsurance",
            "Insurance Brokers",
            "Mortgage Finance",
            "Shell Companies",
            "Thematic",
        ]
    ),
    "Healthcare": sorted(
        [
            "Biotechnology",
            "Diagnostics & Research",
            "Drug Manufacturers - General",
            "Drug Manufacturers - Specialty & Generic",
            "Health Information Services",
            "Healthcare Plans",
            "Medical Care Facilities",
            "Medical Devices",
            "Medical Distribution",
            "Medical Instruments & Supplies",
            "Thematic",
        ]
    ),
    "Industrials": sorted(
        [
            "Aerospace & Defense",
            "Airlines",
            "Airport Services",
            "Building Products & Equipment",
            "Business Equipment & Supplies",
            "Conglomerates",
            "Construction & Engineering",
            "Consulting Services",
            "Electrical Equipment & Parts",
            "Farm & Heavy Construction Machinery",
            "Industrial Distribution",
            "Infrastructure Operations",
            "Integrated Freight & Logistics",
            "Marine Shipping",
            "Metal Fabrication",
            "Pollution & Treatment Controls",
            "Railroads",
            "Rental & Leasing Services",
            "Security & Protection Services",
            "Specialty Business Services",
            "Specialty Industrial Machinery",
            "Staffing & Employment Services",
            "Tools & Accessories",
            "Trucking",
            "Waste Management",
            "Thematic",
        ]
    ),
    "Real Estate": sorted(
        [
            "Real Estate - Development",
            "Real Estate - Diversified",
            "Real Estate Services",
            "REIT - Diversified",
            "REIT - Healthcare Facilities",
            "REIT - Hotel & Motel",
            "REIT - Industrial",
            "REIT - Mortgage",
            "REIT - Office",
            "REIT - Residential",
            "REIT - Retail",
            "REIT - Specialty",
            "Thematic",
        ]
    ),
    "Technology": sorted(
        [
            "Computer Hardware",
            "Consumer Electronics",
            "Electronic Components",
            "Electronics & Computer Distribution",
            "Information Technology Services",
            "Scientific & Technical Instruments",
            "Semiconductor Equipment & Materials",
            "Semiconductors",
            "Software - Application",
            "Software - Infrastructure",
            "Solar",
            "Thematic",
        ]
    ),
    "Utilities": sorted(
        [
            "Utilities - Diversified",
            "Utilities - Independent Power Producers",
            "Utilities - Regulated Electric",
            "Utilities - Regulated Gas",
            "Utilities - Regulated Water",
            "Utilities - Renewable",
            "Thematic",
        ]
    ),
}

CHART_MAX_SLICES = 10  # Max slices before grouping into 'Other' in pie charts
PIE_CHART_FIG_SIZE = (6.5, 4.25)  # Figure size for pie charts
PERF_CHART_FIG_SIZE = (7.5, 3.0)  # Figure size for performance graphs
CHART_DPI = 95  # Dots per inch for charts

INDICES_FOR_HEADER = [".DJI", "IXIC", ".INX"]  # Indices to display in the header bar
CSV_DATE_FORMAT = "%b %d, %Y"  # Date format used in CSV files (e.g., "Jan 01, 2023")

INDEX_DISPLAY_NAMES = {
    ".DJI": "Dow",
    "IXIC": "Nasdaq",
    ".INX": "S&P 500",
    "^DJI": "Dow",
    "^IXIC": "Nasdaq",
    "^GSPC": "S&P 500"
}

# --- Default Settings ---
DEFAULT_CURRENCY = "USD"
DISPLAY_TIMEZONE = "America/New_York"  # Universal display timezone for graphs and metrics

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

# --- ADDED: Currency Symbol Mapping ---
CURRENCY_SYMBOLS = {
    "USD": "$",
    "THB": "฿",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "CAD": "$",
    "AUD": "$",
    "CHF": "Fr",
    "CNY": "¥",
    "HKD": "$",
    "SGD": "$",
}
# --- END ADDED ---

# --- Graph Defaults ---
DEFAULT_GRAPH_DAYS_AGO = 365 * 2  # Default start for graphs, days relative to today
DEFAULT_GRAPH_INTERVAL = "D"  # Default graph interval (D, W, M)

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

# Trading hours for various exchanges (in local time of the exchange)
# Used by the intraday chart to filter for market hours.
# The key is the exchange short name provided by yfinance (ticker.info['exchange']).
# The value is a tuple of (open_time, close_time) in "HH:MM" format.
EXCHANGE_TRADING_HOURS = {
    # --- US Exchanges ---
    "NMS": ("09:30", "16:00"),  # NASDAQ
    "NYQ": ("09:30", "16:00"),  # NYSE
    "PCX": ("09:30", "16:00"),  # NYSE Arca
    "ASE": ("09:30", "16:00"),  # NYSE American
    # --- Asian Exchanges ---
    "SET": (
        "10:00",
        "16:30",
    ),  # Stock Exchange of Thailand (Note: Ignores lunch break for a continuous filter)
    "HKG": ("09:30", "16:00"),  # Hong Kong Stock Exchange
    "JPX": ("09:00", "15:00"),  # Japan Exchange Group (Tokyo)
    # --- European Exchanges ---
    "LSE": ("08:00", "16:30"),  # London Stock Exchange
    "FRA": ("08:00", "20:00"),  # Frankfurt Stock Exchange (XETRA) - Note: long hours
    "PAR": ("09:00", "17:30"),  # Euronext Paris
}
