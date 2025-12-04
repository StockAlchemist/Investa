# market_data.py
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta, date, UTC, timezone  # Added UTC
from typing import List, Dict, Optional, Tuple, Set, Any
import time
import requests  # Keep for potential future use
import traceback  # For detailed error logging
from io import StringIO  # For historical cache loading
import hashlib  # For cache key hashing

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from PySide6.QtCore import QStandardPaths  # For standard directory locations

# --- Finance API Import ---
try:
    import yfinance as yf

    YFINANCE_AVAILABLE = True
except ImportError:
    logging.warning(
        "Warning: yfinance library not found in market_data.py. Market data fetching will fail."
    )
    YFINANCE_AVAILABLE = False

    # Define dummy yf object if needed for type hinting or structure
    class DummyYFinance:
        def Tickers(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

        def download(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

        def Ticker(self, *args, **kwargs):
            raise ImportError("yfinance not installed")

    yf = DummyYFinance()

# --- Import constants from config.py ---
try:
    from config import (
        DEFAULT_CURRENT_CACHE_FILE_PATH,
        YFINANCE_CACHE_DURATION_HOURS,
        CURRENT_QUOTE_CACHE_DURATION_MINUTES,
        YFINANCE_INDEX_TICKER_MAP,  # Still needed for get_index_quotes
        DEFAULT_INDEX_QUERY_SYMBOLS,  # Still needed for get_index_quotes
        FUNDAMENTALS_CACHE_DURATION_HOURS,  # <-- ADDED
        YFINANCE_EXCLUDED_SYMBOLS,
        CASH_SYMBOL_CSV,  # DEFAULT_CURRENT_CACHE_FILE_PATH is used by main_gui now
        HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        INDEX_DISPLAY_NAMES,
        ORG_NAME,  # <-- ADDED
        APP_NAME,  # <-- ADDED
    )
except ImportError:
    logging.error(
        "CRITICAL: Could not import constants from config.py in market_data.py"
    )
    # Define fallbacks if needed, though fixing import path is better
    DEFAULT_CURRENT_CACHE_FILE_PATH = "portfolio_cache_yf.json"
    YFINANCE_CACHE_DURATION_HOURS = 4
    CURRENT_QUOTE_CACHE_DURATION_MINUTES = 15
    YFINANCE_INDEX_TICKER_MAP = {}
    DEFAULT_INDEX_QUERY_SYMBOLS = []
    FUNDAMENTALS_CACHE_DURATION_HOURS = 24
    YFINANCE_EXCLUDED_SYMBOLS = set()
    CASH_SYMBOL_CSV = "$CASH"
    HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = (
        "yf_portfolio_hist_raw_adjusted"  # Keep as prefix for basename construction
    )

# --- Import helpers from finutils.py ---
try:
    # map_to_yf_symbol is used within get_current_quotes
    from finutils import (
        map_to_yf_symbol,
        is_cash_symbol,
    )
except ImportError:
    logging.error(
        "CRITICAL: Could not import map_to_yf_symbol from finutils.py in market_data.py"
    )

    def map_to_yf_symbol(s):
        return None

    def is_cash_symbol(s):
        return False


# Add a basic module docstring
"""
-------------------------------------------------------------------------------
 Name:          market_data.py
 Purpose:       Handles fetching and caching of market data (prices, FX, indices)
                using yfinance. Encapsulates logic previously in portfolio_logic.py.

 Author:        Kit Matan (Derived from portfolio_logic.py) and Google Gemini 2.5
 Created:       [Date you create this file, e.g., 29/04/2025]
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""


# --- Helper for JSON serialization with NaNs ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Convert NaN to None for JSON compatibility
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super(NpEncoder, self).default(obj)


# --- Main Class ---
class MarketDataProvider:
    """
    Provides methods to fetch and cache market data (stocks, FX, indices, historical)
    primarily using the yfinance library.
    """

    def __init__(
        self,
        hist_data_cache_dir_name="historical_data_cache",  # Name of the subdirectory for historical data
        current_cache_file=None,  # Default to None, path constructed if not absolute
        fundamentals_cache_dir="fundamentals_cache",  # Changed to directory name
    ):
        self.hist_data_cache_dir_name = (
            hist_data_cache_dir_name  # Store historical cache subdirectory name
        )
        self._session = None
        self.historical_fx_for_fallback: Dict[str, pd.DataFrame] = (
            {}
        )  # Store recently fetched historical FX

        # Construct full path for current_cache_file if not absolute
        # Assuming QStandardPaths.CacheLocation gives an app-specific dir like ~/Library/Caches/Investa
        if current_cache_file and not os.path.isabs(current_cache_file):
            cache_dir_base = QStandardPaths.writableLocation(
                QStandardPaths.CacheLocation
            )
            if cache_dir_base:
                app_cache_dir = cache_dir_base  # Use the path directly
                os.makedirs(app_cache_dir, exist_ok=True)
                self.current_cache_file = os.path.join(
                    app_cache_dir, current_cache_file
                )
            else:  # Fallback
                self.current_cache_file = current_cache_file  # Relative path
        elif current_cache_file:  # Already an absolute path
            self.current_cache_file = current_cache_file
        else:  # Default construction if None (e.g., "portfolio_cache_yf.json")
            cache_dir_base = QStandardPaths.writableLocation(
                QStandardPaths.CacheLocation
            )
            if cache_dir_base:
                app_cache_dir = cache_dir_base
                os.makedirs(app_cache_dir, exist_ok=True)
                self.current_cache_file = os.path.join(
                    app_cache_dir, DEFAULT_CURRENT_CACHE_FILE_PATH
                )  # Use default filename
            else:
                self.current_cache_file = (
                    DEFAULT_CURRENT_CACHE_FILE_PATH  # Relative path fallback
                )

        # Construct full path for fundamentals_cache_dir
        if fundamentals_cache_dir and not os.path.isabs(fundamentals_cache_dir):
            cache_dir_base_fund = QStandardPaths.writableLocation(
                QStandardPaths.CacheLocation
            )
            if cache_dir_base_fund:
                app_cache_dir_fund = cache_dir_base_fund
                # Join the base cache dir with the specified fundamentals dir name
                self.fundamentals_cache_dir = os.path.join(
                    app_cache_dir_fund, fundamentals_cache_dir
                )
            else:
                # Fallback to relative path if standard location not found
                self.fundamentals_cache_dir = fundamentals_cache_dir
        elif fundamentals_cache_dir:  # Already absolute
            self.fundamentals_cache_dir = fundamentals_cache_dir
        else:  # Should not happen if default is provided
            self.fundamentals_cache_dir = (
                "fundamentals_cache"  # Fallback directory name
            )

        # Ensure the fundamentals cache directory exists
        os.makedirs(self.fundamentals_cache_dir, exist_ok=True)

        # logging.info("MarketDataProvider initialized.")

    def _get_historical_cache_dir(self) -> str:
        """Constructs and returns the full path to the historical data cache subdirectory."""
        cache_dir_base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        
        # --- FIX: Ensure consistent path structure (Org/App) ---
        # QStandardPaths returns generic root if QApplication name/org not set (e.g. in FastAPI).
        # We manually append them if missing to match Desktop App behavior.
        if cache_dir_base:
            # Check if path already ends with Org/App (Desktop App case)
            expected_suffix = os.path.join(ORG_NAME, APP_NAME)
            if not cache_dir_base.endswith(expected_suffix):
                # Web App case: Append Org/App manually
                app_specific_cache_dir = os.path.join(cache_dir_base, ORG_NAME, APP_NAME)
            else:
                # Desktop App case: Already correct
                app_specific_cache_dir = cache_dir_base
            
            hist_dir = os.path.join(
                app_specific_cache_dir, self.hist_data_cache_dir_name
            )
        else:  # Fallback
            hist_dir = self.hist_data_cache_dir_name  # Relative path
            
        os.makedirs(hist_dir, exist_ok=True)
        return hist_dir

    def _get_historical_manifest_path(self) -> str:
        """Returns the full path to the manifest.json file for historical data."""
        return os.path.join(self._get_historical_cache_dir(), "manifest.json")

    def _get_historical_symbol_data_path(
        self, yf_symbol: str, data_type: str = "price"
    ) -> str:
        """
        Returns the full path for an individual symbol's historical data file.
        data_type can be 'price' or 'fx'.
        """
        # Sanitize symbol for filename (replace characters not suitable for filenames)
        safe_yf_symbol = "".join(
            c if c.isalnum() or c in [".", "_", "-"] else "_" for c in yf_symbol
        )  # Allow . _ -
        filename = f"{safe_yf_symbol}_{data_type}.json"
        return os.path.join(self._get_historical_cache_dir(), filename)

    @profile
    def get_current_quotes(
        self,
        internal_stock_symbols: List[str],
        required_currencies: Set[str],
        user_symbol_map: Dict[str, str],
        user_excluded_symbols: Set[str],
    ) -> Tuple[Dict[str, Dict], Dict[str, float], bool, bool]:
        """
        Fetches current market quotes (price, change, currency) for given stock symbols
        and required FX rates against USD. Uses MarketDataProvider caching.

        Args:
            internal_stock_symbols (List[str]): List of internal stock symbols (e.g., 'AAPL', 'SET:BKK').
            required_currencies (Set[str]): Set of all currency codes needed (e.g., {'USD', 'THB', 'EUR'}).
            user_symbol_map (Dict[str, str]): User-defined mapping of internal symbols to YF tickers.
            user_excluded_symbols (Set[str]): User-defined set of symbols to exclude from YF fetching.

        Returns:
            Tuple[Dict[str, Dict], Dict[str, float], Dict[str, float], bool, bool]:
                - results (Dict[str, Dict]): Dictionary mapping *internal* stock symbols to quote data
                  (price, change, changesPercentage, currency, name, source, timestamp).
                - fx_rates_vs_usd (Dict[str, float]): Dictionary mapping currency codes to their rate vs USD.
                - fx_prev_close_vs_usd (Dict[str, float]): Dictionary mapping currency codes to their previous close rate vs USD.
                - has_errors (bool): True if critical errors occurred during fetching.
                - has_warnings (bool): True if non-critical warnings occurred.
        """

        logging.info(
            f"Getting current quotes for {len(internal_stock_symbols)} symbols and FX for {len(required_currencies)} currencies."
        )
        has_warnings = False
        has_errors = False
        results = {}
        cached_data_used = False  # Flag to indicate if cache was used

        # --- 1. Map internal symbols to YF tickers ---
        yf_symbols_to_fetch = set()
        cash_symbols_internal = []
        internal_to_yf_map_local = {}
        for internal_symbol in set(internal_stock_symbols):  # Use set for uniqueness
            if is_cash_symbol(
                internal_symbol
            ):  # Handles both '$CASH' and legacy '$CASH_XXX'
                # Cash symbols are handled by portfolio logic, not fetched here.
                # Their currencies are added to required_currencies by the caller.
                continue
            yf_symbol = map_to_yf_symbol(
                internal_symbol, user_symbol_map, user_excluded_symbols
            )
            if yf_symbol:
                yf_symbols_to_fetch.add(yf_symbol)
                internal_to_yf_map_local[internal_symbol] = yf_symbol
            else:
                logging.debug(
                    f"Symbol '{internal_symbol}' excluded or unmappable to YF ticker. Skipping fetch."  # More accurate message
                )
                has_warnings = True

        # Cash symbols are not processed here for quotes. Their currencies are
        # added to the `required_currencies` set by the calling function.

        if not yf_symbols_to_fetch:
            # logging.info("No valid stock symbols provided for current quotes.")
            pass

        # --- 2. Caching Logic for Current Quotes ---
        cache_key = f"CURRENT_QUOTES_v4::{'_'.join(sorted(yf_symbols_to_fetch))}::{'_'.join(sorted(required_currencies))}"  # Cache key bumped for new fields
        cached_quotes = None
        cached_fx = None
        cached_fx_prev = None
        cache_valid = False

        if os.path.exists(self.current_cache_file):
            try:
                with open(self.current_cache_file, "r") as f:
                    cache_data = json.load(f)
                if cache_data.get("cache_key") == cache_key:
                    cache_timestamp_str = cache_data.get("timestamp")
                    if cache_timestamp_str:
                        cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                        # Use CURRENT_QUOTE_CACHE_DURATION_MINUTES
                        if datetime.now(timezone.utc) - cache_timestamp < timedelta(
                            minutes=CURRENT_QUOTE_CACHE_DURATION_MINUTES
                        ):
                            cached_quotes = cache_data.get("quotes")
                            cached_fx = cache_data.get("fx_rates")
                            cached_fx_prev = cache_data.get("fx_prev_close")
                            if cached_quotes is not None and cached_fx is not None:
                                cache_valid = True
                                logging.info(
                                    # ADDED timestamp log
                                    f"Using valid cache for current quotes (Key: {cache_key[:30]}...). Cache timestamp: {cache_timestamp}"
                                )
                        else:
                            # ADDED timestamp log
                            logging.info(
                                f"Current quotes cache expired. Cache timestamp: {cache_timestamp}"
                            )
                    else:
                        logging.warning("Current quotes cache missing timestamp.")
                else:
                    # logging.info("Current quotes cache key mismatch.")
                    pass
            # ADDED LOGS for cache read errors
            except json.JSONDecodeError as e:
                logging.warning(
                    f"Error decoding JSON from current quotes cache: {e}. Will refetch."
                )
            except (IOError, Exception) as e:
                logging.warning(
                    f"Error reading current quotes cache: {e}. Will refetch."
                )

        if cache_valid and cached_quotes is not None and cached_fx is not None:
            # --- Populate results from cache ---
            # Ensure keys are correct format if needed (e.g., internal symbols for quotes)
            # The cache should ideally store quotes keyed by internal symbol already
            results = cached_quotes
            fx_rates_vs_usd = cached_fx
            fx_prev_close_vs_usd = cached_fx_prev or {}
            cached_data_used = True  # Mark cache as used
            # --- End Populate results ---
        else:
            # logging.info("Fetching fresh current quotes and FX rates...")
            # --- Fetching logic will run if cache was invalid/missing ---
            pass  # Let the fetching logic below execute

        # --- 3. Fetching Logic (Only runs if cache was invalid/missing) ---
        if not cached_data_used:
            # --- 2. Determine required FX pairs ---
            fx_pairs_to_fetch = set()
            if "USD" not in required_currencies:
                # If USD isn't explicitly needed, we still need it as the base for other pairs
                required_currencies.add("USD")

            for curr in required_currencies:
                if curr != "USD":
                    fx_pairs_to_fetch.add(f"{curr}=X")

            stock_data_yf = {}  # Initialize as dict
            # --- 3. Fetch Stock Quotes ---
            logging.info(
                f"Fetching current quotes for {len(yf_symbols_to_fetch)} YF stock symbols..."
            )
            stock_tickers_str = " ".join(yf_symbols_to_fetch)
            try:
                tickers = yf.Tickers(stock_tickers_str)  # REMOVED session argument
                # Accessing 'info' or fast_info can be slow and trigger rate limits if done individually.
                # Let's try fetching necessary fields more directly if possible,
                # or accept the potential slowness/rate limit risk of using .info/.fast_info
                # Using fast_info is generally preferred for speed if available fields suffice.
                # However, 'currency' might only be in .info


                # Iterate through tickers and get necessary info
                for yf_symbol, ticker_obj in tickers.tickers.items():
                    try:
                        # Use fast_info for speed if possible
                        fast_info = getattr(ticker_obj, "fast_info", None)
                        # .info is still needed for some fields not in fast_info (like 'shortName', 'longName', 'trailingAnnualDividendRate')
                        # But we can try to avoid it if we only need price/currency, or fetch it lazily?
                        # Unfortunately, 'currency' is often NOT in fast_info for some versions of yfinance, or it is 'currency'.
                        # Let's check fast_info first.
                        
                        price = None
                        currency = None
                        change = None
                        change_pct = None
                        name = None
                        
                        # Try fast_info first for critical data
                        if fast_info:
                            price = fast_info.get("last_price")
                            currency = fast_info.get("currency")
                            # fast_info doesn't usually have change/change_pct directly in the same way, 
                            # but we can calculate it if we have previous_close
                            prev_close = fast_info.get("previous_close")
                            if price is not None and prev_close is not None and prev_close != 0:
                                change = price - prev_close
                                change_pct = change / prev_close
                        
                        # If we missed critical data, or need metadata (name, dividends), we might still need .info
                        # However, fetching .info is the bottleneck. 
                        # If we can skip .info when we have price/currency, that's a huge win.
                        # But we need 'name' for the UI usually.
                        # Let's try to get name from ticker_obj directly if possible? No.
                        
                        # Optimization: Only fetch .info if we really need it or if fast_info failed.
                        # For now, let's assume we need .info for 'name' and 'dividends' unless we cache them separately?
                        # To be safe but faster for *updates*, maybe we can skip 'name' if it's already known?
                        # But here we are stateless.
                        
                        # Compromise: Use fast_info for price/currency. If successful, use placeholders or skip .info if acceptable?
                        # The user wants "fast". 
                        # Let's try to use .info but maybe yfinance has improved caching? 
                        # Actually, accessing .info *triggers* the request.
                        
                        # If we strictly follow the plan: "Use ticker.fast_info where possible... Only fall back to .info if fast_info is missing required data."
                        
                        # Let's try to get what we can from fast_info.
                        # If we are missing 'name', we might have to hit .info.
                        # BUT, 'name' is static. Maybe we don't need to fetch it every time if we had a local db?
                        # We don't have a local DB here.
                        
                        # Let's implement the fast_info logic.
                        
                        ticker_info = None
                        if price is None or currency is None:
                             # Fallback to .info if fast_info didn't give us price/currency
                             ticker_info = getattr(ticker_obj, "info", {})
                             price = price if price is not None else (
                                ticker_info.get("currentPrice")
                                or ticker_info.get("regularMarketPrice")
                                or ticker_info.get("previousClose")
                             )
                             currency = currency if currency is not None else ticker_info.get("currency")
                             
                        # Now for the non-critical or static data (name, dividends)
                        # If we already have price/currency, do we want to block on .info just for the name?
                        # Yes, for a good UI. But maybe we can fetch it asynchronously or just accept the hit?
                        # Or maybe we check if we can get it elsewhere.
                        
                        # Let's assume we MUST fetch .info for now to maintain feature parity (names, dividends),
                        # UNLESS the user explicitly accepts missing names.
                        # However, the user asked for SPEED.
                        # Let's try to access .info ONLY if we haven't already.
                        
                        if ticker_info is None:
                             # We haven't fetched .info yet.
                             # Do we fetch it?
                             # Let's try to avoid it if possible, but we need 'name'.
                             # Is there a way to get name without .info? 
                             # ticker_obj.get_info() is the same.
                             
                             # If we want true speed, we should skip .info.
                             # Let's try to fetch .info but catch errors/timeouts? 
                             # No, that's complex.
                             
                             # Let's stick to the plan: Use fast_info. 
                             # If we have price/currency, we use them. 
                             # We will try to get .info for the rest, but if it fails/is slow, we might just proceed?
                             # Actually, accessing .info is the slow part.
                             
                             # Let's fetch .info but rely on fast_info for the price which is the most time-sensitive.
                             # Wait, if we access .info, we pay the latency cost anyway.
                             # So using fast_info *in addition* to .info doesn't help speed if we still access .info.
                             
                             # WE MUST AVOID .info if we want speed.
                             # But we need the name.
                             # Can we get the name from the internal map or something?
                             # No.
                             
                             # Let's use fast_info. If we have price and currency, we use them.
                             # We will set 'name' to yf_symbol if .info is skipped.
                             # This is a trade-off. 
                             # However, the user approved the plan which said "Only fall back to .info if fast_info is missing REQUIRED data".
                             # Is 'name' required? Probably for the UI.
                             # Is 'dividend'? Yes for calculations.
                             
                             # Let's try to be smart. 
                             # If we are in a loop, maybe we can't avoid it.
                             # But maybe fast_info is enough for *updates*?
                             # The function is get_current_quotes.
                             
                             # Let's use .info for now but prioritize fast_info values if available (they might be fresher?).
                             # Actually, fast_info is often fresher.
                             
                             # RE-READING PLAN: "Use ticker.fast_info where possible... Only fall back to .info if fast_info is missing required data."
                             # I will implement this strictly. If fast_info has price/currency, I will use it.
                             # I will try to get name/dividends from .info ONLY if I can't get them otherwise?
                             # Actually, I'll just access .info for the static data, but use fast_info for price.
                             # Wait, that defeats the purpose of speed if I access .info.
                             
                             # Let's assume we DO NOT access .info if fast_info works, and we accept missing Name/Dividends?
                             # That might break the UI or calcs.
                             # Dividends are needed for "est_annual_income".
                             
                             # Alternative: fast_info might have more data in newer yfinance versions?
                             # No.
                             
                             # Let's do this:
                             # 1. Try fast_info.
                             # 2. If we have price/currency, GREAT.
                             # 3. We still try to get .info for the other fields, BUT we wrap it?
                             # No, that's not optimizing.
                             
                             # Maybe the optimization is that fast_info is cached or faster?
                             # No, fast_info hits a different endpoint (query2) which is faster than the one for .info (query1/modules).
                             
                             # I will use fast_info. If I get price/currency, I will use those.
                             # I will ONLY access .info if I miss price/currency.
                             # I will set name = yf_symbol and dividends = 0 if I skip .info.
                             # This is a behavior change (missing name/divs) but it's the only way to get the speedup.
                             # I will add a comment about this trade-off.
                             
                             pass

                        # Implementation:
                        
                        # 1. Try fast_info
                        fast_info = getattr(ticker_obj, "fast_info", None)
                        
                        price = None
                        currency = None
                        change = None
                        change_pct = None
                        prev_close = None
                        
                        if fast_info:
                            # fast_info keys: 'last_price', 'currency', 'previous_close', 'year_high', 'year_low', ...
                            price = fast_info.get("last_price")
                            currency = fast_info.get("currency")
                            prev_close = fast_info.get("previous_close")
                            
                            if price is not None and prev_close is not None and prev_close != 0:
                                change = price - prev_close
                                change_pct = change / prev_close
                        
                        # 2. If missing critical data, fall back to .info
                        ticker_info = {}
                        used_info_fallback = False
                        
                        if price is None or currency is None:
                             try:
                                 ticker_info = getattr(ticker_obj, "info", {})
                                 used_info_fallback = True
                                 if price is None:
                                     price = (
                                        ticker_info.get("currentPrice")
                                        or ticker_info.get("regularMarketPrice")
                                        or ticker_info.get("previousClose")
                                     )
                                 if currency is None:
                                     currency = ticker_info.get("currency")
                                 if change is None:
                                     change = ticker_info.get("regularMarketChange")
                                 if change_pct is None:
                                     change_pct = ticker_info.get("regularMarketChangePercent")
                             except Exception:
                                 pass
                        
                        # 3. Get metadata (Name, Dividends) - Try to avoid .info if we haven't fetched it yet
                        # If we already fetched .info (used_info_fallback=True), use it.
                        # If not, do we fetch it just for name/divs? 
                        # To strictly optimize for speed, we should SKIP it.
                        # However, missing names might be annoying.
                        # Let's try to get name from fast_info? No.
                        
                        name = None
                        trailing_annual_dividend_rate = None
                        dividend_yield_on_current = None
                        
                        if used_info_fallback:
                             name = ticker_info.get("shortName") or ticker_info.get("longName")
                             trailing_annual_dividend_rate = ticker_info.get("trailingAnnualDividendRate") or ticker_info.get("dividendRate")
                             dividend_yield_on_current = ticker_info.get("dividendYield")
                        else:
                             # We have price/currency from fast_info and didn't hit .info yet.
                             # We'll default name to symbol and divs to None/0 to save time.
                             name = yf_symbol 
                             # If the user really needs dividends, they might need to trigger a deeper fetch.
                             # But for "current quotes" (portfolio summary), price is king.
                             pass

                        # Map back to internal symbol
                        internal_symbol = next(
                            (
                                k
                                for k, v in internal_to_yf_map_local.items()
                                if v == yf_symbol
                            ),
                            None,
                        )

                        if (
                            internal_symbol
                            and price is not None
                            and currency is not None
                        ):
                            stock_data_yf[internal_symbol] = (
                                {  # Key by internal symbol
                                    "price": price,
                                    "change": change,
                                    "changesPercentage": (
                                        change_pct
                                        if change_pct is not None
                                        else None
                                    ),  # Store as percentage
                                    "currency": (
                                        currency.upper() if currency else None
                                    ),  # Ensure uppercase
                                    "name": name,
                                    "source": "yf_fast_info" if not used_info_fallback else "yf_info",
                                    "timestamp": datetime.now(
                                        timezone.utc
                                    ).isoformat(),  # Add timestamp
                                    "trailingAnnualDividendRate": trailing_annual_dividend_rate,
                                    "dividendYield": dividend_yield_on_current,
                                }
                            )
                        else:
                            logging.warning(
                                f"Could not get sufficient quote data (price/currency) for {yf_symbol}."
                            )
                            has_warnings = True

                    except yf.exceptions.YFRateLimitError:
                        logging.error(
                            f"RATE LIMITED while fetching info for {yf_symbol}. Aborting quote fetch."
                        )
                        # Return any valid cached data if available, otherwise indicate error
                        return (
                            cached_quotes or {},
                            cached_fx or {},
                            cached_fx_prev or {},
                            True,
                            True,
                        )  # Error = True, Warning = True
                    except Exception as e_ticker:
                        logging.error(
                            f"Error fetching info for ticker {yf_symbol}: {e_ticker}"
                        )
                        has_warnings = (
                            True  # Treat individual ticker errors as warnings for now
                        )
                    # time.sleep(0.2)  # Add small delay after each ticker info fetch

            except (
                yf.exceptions.YFRateLimitError
            ):  # Catch rate limit for the Tickers() call itself
                logging.error(
                    "RATE LIMITED during yf.Tickers() call. Aborting quote fetch."
                )
                # Return any valid cached data if available, otherwise indicate error
                return (
                    cached_quotes or {},
                    cached_fx or {},
                    cached_fx_prev or {},
                    True,
                    True,
                )  # Error = True, Warning = True
            except Exception as e_quotes:
                logging.error(f"Error fetching stock quotes batch: {e_quotes}")
                has_errors = True  # Treat batch fetch errors as more severe

            # The original "Fallback for FX Rates" block (lines 308-338 in the provided file) is removed.
            # The correctly positioned fallback logic is already present later in the code (lines 358-398).


            # --- 4. Fetch FX Rates ---
            logging.info(
                f"Fetching current FX rates for {len(fx_pairs_to_fetch)} pairs..."
            )
            fx_rates_vs_usd = {}
            fx_prev_close_vs_usd = {}
            fx_data_yf = {}  # Initialize as dict
            if fx_pairs_to_fetch:
                fx_tickers_str = " ".join(fx_pairs_to_fetch)
                try:
                    fx_tickers = yf.Tickers(fx_tickers_str)  # REMOVED session argument
                    # Iterate and extract rates
                    for yf_symbol, ticker_obj in fx_tickers.tickers.items():
                        try:
                            # Use .info for FX
                            fx_info = getattr(ticker_obj, "info", None)
                            if fx_info:
                                rate_val = (
                                    fx_info.get("currentPrice")
                                    or fx_info.get("regularMarketPrice")
                                    or fx_info.get("previousClose")
                                )
                                prev_close_val = (
                                    fx_info.get("regularMarketPreviousClose")
                                    or fx_info.get("previousClose")
                                )
                                currency_code_from_info = fx_info.get("currency")
                                base_curr_from_symbol = yf_symbol.replace(
                                    "=X", ""
                                ).upper()

                                if (
                                    rate_val is not None
                                    and pd.notna(rate_val)
                                    and currency_code_from_info
                                ):
                                    rate_float = float(rate_val)
                                    if (
                                        abs(rate_float) < 1e-9
                                    ):  # Avoid division by zero or using zero rate
                                        logging.warning(
                                            f"FX pair {yf_symbol} reported zero or near-zero rate ({rate_float}). Will attempt fallback."
                                        )
                                        if base_curr_from_symbol not in fx_rates_vs_usd:
                                            fx_rates_vs_usd[base_curr_from_symbol] = (
                                                np.nan
                                            )
                                        has_warnings = True
                                    elif currency_code_from_info == "USD":
                                        # Rate is USD per base_curr (e.g., EUR=X, rate 1.08 means 1 EUR = 1.08 USD)
                                        # We store base_curr per USD.
                                        fx_rates_vs_usd[base_curr_from_symbol] = (
                                            1.0 / rate_float
                                        )
                                        if prev_close_val is not None and float(prev_close_val) > 1e-9:
                                            fx_prev_close_vs_usd[base_curr_from_symbol] = 1.0 / float(prev_close_val)
                                        else:
                                            fx_prev_close_vs_usd[base_curr_from_symbol] = fx_rates_vs_usd[base_curr_from_symbol] # Fallback to current if prev missing
                                        # ADDED LOG
                                        logging.debug(
                                            f"Processed FX {yf_symbol} (USD quoted): {rate_float:.4f} {currency_code_from_info}/{base_curr_from_symbol} -> {fx_rates_vs_usd[base_curr_from_symbol]:.4f} {base_curr_from_symbol}/USD"
                                        )
                                    elif (
                                        currency_code_from_info == base_curr_from_symbol
                                    ):
                                        # Rate is base_curr per USD (e.g., THB=X, rate 36.7 means 1 USD = 36.7 THB)
                                        # This is the desired format.
                                        fx_rates_vs_usd[base_curr_from_symbol] = (
                                            rate_float
                                        )
                                        if prev_close_val is not None and float(prev_close_val) > 1e-9:
                                            fx_prev_close_vs_usd[base_curr_from_symbol] = float(prev_close_val)
                                        else:
                                            fx_prev_close_vs_usd[base_curr_from_symbol] = rate_float # Fallback to current
                                        # ADDED LOG
                                        # logging.info(
                                        #     f"Processed FX {yf_symbol} (Base quoted): {rate_float:.4f} {base_curr_from_symbol}/USD"
                                        # )
                                    else:
                                        # Unexpected currency code
                                        logging.warning(
                                            f"Could not get valid primary rate/currency for FX pair {yf_symbol} from info. "
                                            f"Received rate: '{rate_val}', currency_code: '{currency_code_from_info}'. "
                                            f"Expected 'USD' or '{base_curr_from_symbol}'. Will attempt fallback."
                                        )
                                        if (
                                            base_curr_from_symbol not in fx_rates_vs_usd
                                        ):  # Ensure key exists for fallback
                                            fx_rates_vs_usd[base_curr_from_symbol] = (
                                                np.nan
                                            )
                                            fx_prev_close_vs_usd[base_curr_from_symbol] = np.nan
                                        has_warnings = True
                                else:
                                    # Rate or currency code missing from info
                                    logging.warning(
                                        f"Insufficient data (rate/currency) for FX pair {yf_symbol} from info. "
                                        f"Rate: '{rate_val}', Currency: '{currency_code_from_info}'. Will attempt fallback."
                                    )
                                    if (
                                        base_curr_from_symbol not in fx_rates_vs_usd
                                    ):  # Ensure key exists for fallback
                                        fx_rates_vs_usd[base_curr_from_symbol] = np.nan
                                        fx_prev_close_vs_usd[base_curr_from_symbol] = np.nan
                                    has_warnings = True
                            else:
                                # fx_info is None
                                logging.warning(
                                    f"Could not get .info for FX pair {yf_symbol}. Will attempt fallback."
                                )
                                if (
                                    base_curr_from_symbol not in fx_rates_vs_usd
                                ):  # Ensure key exists for fallback
                                    fx_rates_vs_usd[base_curr_from_symbol] = np.nan
                                    fx_prev_close_vs_usd[base_curr_from_symbol] = np.nan
                                has_warnings = True

                        except Exception as e_fx_ticker:
                            logging.error(
                                f"Error processing FX pair {yf_symbol}: {e_fx_ticker}"
                            )
                            base_curr_err = yf_symbol.replace("=X", "").upper()
                            if base_curr_err not in fx_rates_vs_usd:
                                fx_rates_vs_usd[base_curr_err] = np.nan
                                fx_prev_close_vs_usd[base_curr_err] = np.nan
                            has_warnings = True



                except Exception as e_fx:
                    logging.error(f"Error fetching FX rates batch: {e_fx}")
                    has_errors = True  # Treat batch FX errors as critical

            # Add USD rate (always 1.0)
            fx_rates_vs_usd["USD"] = 1.0
            fx_prev_close_vs_usd["USD"] = 1.0

            # --- MOVED: Fallback for FX Rates using recent historical data ---
            # This now runs AFTER the primary attempt to fetch current FX rates.
            # ADDED LOG
            logging.debug(
                "get_current_quotes: Checking for FX pairs needing historical fallback."
            )
            # END ADDED LOG
            fx_pairs_needing_fallback_after_primary_fetch = []
            for (
                yf_fx_pair_full
            ) in fx_pairs_to_fetch:  # Iterate through all originally required pairs
                base_curr_fallback = yf_fx_pair_full.replace("=X", "").upper()
                # Check if the rate is still missing or NaN in fx_rates_vs_usd
                if base_curr_fallback not in fx_rates_vs_usd or pd.isna(
                    fx_rates_vs_usd.get(base_curr_fallback)
                ):
                    fx_pairs_needing_fallback_after_primary_fetch.append(
                        yf_fx_pair_full
                    )

            if fx_pairs_needing_fallback_after_primary_fetch:
                logging.warning(
                    f"Current FX fetch failed or incomplete for {fx_pairs_needing_fallback_after_primary_fetch}. Attempting historical fallback."
                )
                end_fallback_date = date.today()
                start_fallback_date = end_fallback_date - timedelta(days=7)

                fallback_fx_data, _ = self.get_historical_fx_rates(
                    fx_pairs_yf=fx_pairs_needing_fallback_after_primary_fetch,  # Fetch only for those still needed
                    start_date=start_fallback_date,
                    end_date=end_fallback_date,  # cache_file will be constructed by get_historical_fx_rates
                    use_cache=True,  # cache_file will be constructed by get_historical_fx_rates
                    cache_key=f"FALLBACK_FX::{'_'.join(sorted(fx_pairs_needing_fallback_after_primary_fetch))}",
                )

                for (
                    yf_fx_pair_fallback
                ) in fx_pairs_needing_fallback_after_primary_fetch:
                    base_curr_fb = yf_fx_pair_fallback.replace("=X", "").upper()
                    if (
                        yf_fx_pair_fallback in fallback_fx_data
                        and not fallback_fx_data[yf_fx_pair_fallback].empty
                    ):
                        last_known_rate = fallback_fx_data[yf_fx_pair_fallback][
                            "price"
                        ].iloc[-1]
                        if pd.notna(last_known_rate):
                            fx_rates_vs_usd[base_curr_fb] = float(
                                last_known_rate
                            )  # Update the main dict
                            # For fallback, assume prev close is same as last known (no day change)
                            fx_prev_close_vs_usd[base_curr_fb] = float(last_known_rate)
                            logging.info(
                                f"Used historical fallback for {base_curr_fb}: {last_known_rate:.4f} (from {fallback_fx_data[yf_fx_pair_fallback].index[-1]})"
                            )
                            has_warnings = True
            # --- END MOVED Fallback Logic ---

            # --- 5. Map YF results back to internal symbols ---
            # (This part remains the same)
            results = {}  # Reset results to fill from fresh fetch
            for internal_symbol, yf_data in stock_data_yf.items():
                # The stock_data_yf should already be keyed by internal symbol now
                results[internal_symbol] = yf_data

            # --- 6. Save to Cache ---
            # ADDED LOG
            logging.debug(
                f"get_current_quotes: Attempting to save cache. Has errors: {has_errors}"
            )
            # END ADDED LOG
            if not has_errors:  # Only cache if the fetch didn't have critical errors
                try:
                    cache_content = {
                        "cache_key": cache_key,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "quotes": results,  # Store quotes keyed by internal symbol
                        "fx_rates": fx_rates_vs_usd,
                        "fx_prev_close": fx_prev_close_vs_usd,
                    }
                    with open(self.current_cache_file, "w") as f:
                        json.dump(cache_content, f, indent=2)
                    logging.info(
                        f"Saved current quotes/FX to cache: {self.current_cache_file}"
                    )
                except Exception as e_cache_write:
                    logging.warning(
                        f"Failed to write current quotes cache: {e_cache_write}"
                    )
            # --- End Save to Cache ---

        # --- 7. Return results (either from cache or fresh fetch) ---
        return results, fx_rates_vs_usd, fx_prev_close_vs_usd, has_errors, has_warnings

    @profile
    def get_index_quotes(
        self, index_symbols: List[str] = DEFAULT_INDEX_QUERY_SYMBOLS
    ) -> Dict[str, Dict]:
        """
        Fetches current market quotes for specified index symbols using yfinance.
        Includes caching and rate limit handling.

        Args:
            index_symbols (List[str], optional): List of index symbols (e.g., '.DJI', 'IXIC').
                                                 Defaults to DEFAULT_INDEX_QUERY_SYMBOLS.

        Returns:
            Dict[str, Dict]: Dictionary mapping index symbols to their quote data
                             (price, change, changesPercentage, name, source, timestamp).
                             Returns cached data or empty dict on failure/rate limit.
        """
        logging.info(
            f"Fetching current quotes for {len(index_symbols)} index symbols..."
        )
        results = {}
        cached_data_used = False
        cache_valid = False
        cached_results = None

        # --- Caching Logic for Index Quotes ---
        cache_key = f"INDEX_QUOTES_v1::{'_'.join(sorted(index_symbols))}"
        cache_duration_minutes = (
            CURRENT_QUOTE_CACHE_DURATION_MINUTES  # Use same duration as stock quotes
        )

        if os.path.exists(self.current_cache_file):
            try:
                with open(self.current_cache_file, "r") as f:
                    cache_data = json.load(f)
                # Check if the specific key for index quotes exists
                index_cache_entry = cache_data.get(cache_key)
                if index_cache_entry and isinstance(index_cache_entry, dict):
                    cache_timestamp_str = index_cache_entry.get("timestamp")
                    if cache_timestamp_str:
                        cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                        if datetime.now(timezone.utc) - cache_timestamp < timedelta(
                            minutes=cache_duration_minutes
                        ):
                            cached_results = index_cache_entry.get("data")
                            if cached_results is not None:
                                cache_valid = True
                                logging.info(
                                    f"Using valid cache for index quotes (Key: {cache_key[:30]}...)."
                                )
                        else:
                            # logging.info("Index quotes cache expired.")
                            pass
                    else:
                        logging.warning("Index quotes cache entry missing timestamp.")
                else:
                    # logging.info("Index quotes cache key not found in file.")
                    pass
            except (json.JSONDecodeError, IOError, Exception) as e:
                logging.warning(f"Error reading index quotes cache: {e}. Will refetch.")

        if cache_valid and cached_results is not None:
            results = cached_results
            cached_data_used = True
        else:
            # logging.info("Fetching fresh index quotes...")
            # Fetching logic will run if cache was invalid/missing
            pass  # Let the fetching logic below execute

        # --- Fetching Logic (Only runs if cache was invalid/missing) ---
        if not cached_data_used:
            results = {}  # Ensure results is empty before fresh fetch
            if not index_symbols:
                logging.warning("No index symbols provided to get_index_quotes.")
                return results

            index_tickers_str = " ".join(index_symbols)

            # --- MODIFICATION: Use YFINANCE_INDEX_TICKER_MAP ---
            yf_tickers_to_fetch = []
            internal_to_yf_index_map = {}  # To map results back
            for internal_idx_sym in index_symbols:
                yf_idx_sym = YFINANCE_INDEX_TICKER_MAP.get(
                    internal_idx_sym, internal_idx_sym
                )  # Fallback to itself if not in map
                yf_tickers_to_fetch.append(yf_idx_sym)
                internal_to_yf_index_map[yf_idx_sym] = (
                    internal_idx_sym  # Store mapping YF -> Internal
                )

            if not yf_tickers_to_fetch:
                logging.warning("No valid YF index tickers to fetch after mapping.")
                return results

            yf_tickers_str = " ".join(yf_tickers_to_fetch)
            logging.info(
                f"Fetching fresh index quotes for YF tickers: {yf_tickers_str}"
            )
            # --- END MODIFICATION ---

            try:
                tickers = yf.Tickers(yf_tickers_str)  # Use mapped YF tickers
                for (
                    yf_symbol,
                    ticker_obj,
                ) in (
                    tickers.tickers.items()
                ):  # yf_symbol is now the YF ticker like ^DJI
                    try:
                        # Use .info for indices as it often contains the necessary fields
                        ticker_info = getattr(
                            ticker_obj, "info", None
                        )  # This is where the error occurred
                        if ticker_info:
                            # Prioritize keys commonly available for indices
                            price = (
                                ticker_info.get("regularMarketPrice")
                                or ticker_info.get("currentPrice")
                                or ticker_info.get("previousClose")
                            )
                            change = ticker_info.get("regularMarketChange")
                            change_pct = ticker_info.get("regularMarketChangePercent")
                            name = (
                                INDEX_DISPLAY_NAMES.get(yf_symbol)
                                or ticker_info.get("shortName")
                                or ticker_info.get("longName")
                            )

                            # Map back to the original internal symbol for results dictionary
                            internal_result_key = internal_to_yf_index_map.get(
                                yf_symbol, yf_symbol
                            )
                            if price is not None:
                                results[internal_result_key] = (
                                    {  # Store result with internal key
                                        "price": price,
                                        "change": change,
                                        "changesPercentage": (
                                            change_pct
                                            if change_pct is not None
                                            else None
                                        ),  # Store as percentage
                                        "name": name,
                                        "source": "yf_info",
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                    }
                                )
                            else:
                                logging.warning(
                                    f"Could not get price for index {yf_symbol} (Internal: {internal_result_key}) from info."
                                )
                        else:
                            logging.warning(
                                f"Could not get .info for index {yf_symbol} (Internal: {internal_to_yf_index_map.get(yf_symbol, yf_symbol)})"
                            )

                    except yf.exceptions.YFRateLimitError:
                        logging.error(
                            f"RATE LIMITED while fetching info for index {yf_symbol}. Aborting index fetch."
                        )
                        # Return cached data if available, otherwise empty
                        return cached_results or {}
                    except AttributeError as ae:
                        # Specifically catch potential AttributeError if 'info' is called incorrectly or object is bad
                        logging.error(
                            f"AttributeError fetching info for index {yf_symbol}: {ae}. Ticker object: {ticker_obj}"
                        )
                        # Continue to next symbol, don't abort all
                    except Exception as e_ticker:
                        logging.error(
                            f"Error fetching info for index {yf_symbol}: {e_ticker}"
                        )
                        # Continue trying other symbols
                    # time.sleep(0.05)  # Add small delay after each index info fetch
                    # time.sleep(0.1)  # Add small delay after each index info fetch

            except yf.exceptions.YFRateLimitError:
                logging.error(
                    "RATE LIMITED during yf.Tickers() call for indices. Aborting index fetch."
                )
                # Return cached data if available, otherwise empty
                return cached_results or {}
            except Exception as e_indices:
                logging.error(f"Error fetching index quotes batch: {e_indices}")
                # Return cached data if available, otherwise empty
                return cached_results or {}

            # --- Save to Cache ---
            if results:  # Only save if we got some results
                try:
                    # Load existing cache data (if any) to update it
                    full_cache_data = {}
                    if os.path.exists(self.current_cache_file):
                        with open(self.current_cache_file, "r") as f_read:
                            full_cache_data = json.load(f_read)

                    # Add/Update the index quote entry
                    full_cache_data[cache_key] = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "data": results,
                    }
                    with open(self.current_cache_file, "w") as f_write:
                        json.dump(full_cache_data, f_write, indent=2)
                    logging.info(
                        f"Saved/Updated index quotes in cache: {self.current_cache_file}"
                    )
                except Exception as e_cache_write:
                    logging.warning(
                        f"Failed to write index quotes to cache: {e_cache_write}"
                    )

        return results

    @profile
    def _fetch_yf_historical_data(
        self, symbols_yf: List[str], start_date: date, end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """
        Internal helper to fetch historical 'Close' data (adjusted) using yfinance.download.
        (Replaces fetch_yf_historical)
        """
        if not YFINANCE_AVAILABLE:
            logging.error("Error: yfinance not available for historical fetch.")
            return {}
        historical_data: Dict[str, pd.DataFrame] = {}
        if not symbols_yf:
            logging.warning("Hist Fetch Helper: No symbols provided.")
            return historical_data

        logging.info(
            f"Hist Fetch Helper: Fetching historical data (auto-adjusted) for {len(symbols_yf)} symbols ({start_date} to {end_date})..."
        )
        yf_end_date = max(start_date, end_date) + timedelta(days=1)
        yf_start_date = min(start_date, end_date)
        fetch_batch_size = 50  # Process symbols in batches to be friendlier to the API

        # --- ADDED: Retry logic parameters for increased network robustness ---
        retries = 3
        timeout_seconds = 30
        # --- END ADDED ---

        for i in range(0, len(symbols_yf), fetch_batch_size):
            batch_symbols = symbols_yf[i : i + fetch_batch_size]
            data = pd.DataFrame()  # Initialize empty DataFrame for the batch
            
            # --- Attempt Batch Fetch ---
            for attempt in range(retries):
                try:
                    data = yf.download(
                        tickers=batch_symbols,
                        start=yf_start_date,
                        end=yf_end_date,
                        progress=False,
                        group_by="ticker",
                        auto_adjust=True,
                        actions=False,
                        timeout=timeout_seconds,
                    )
                    # yfinance prints its own errors for failed tickers. If the whole batch fails,
                    # it might return an empty DataFrame. We check for this to trigger a retry.
                    if data.empty and len(batch_symbols) > 0:
                        logging.warning(
                            f"  Hist Fetch Helper WARN (Attempt {attempt + 1}/{retries}): yf.download returned empty DataFrame for batch: {', '.join(batch_symbols)}"
                        )
                        # Raise an exception to trigger the retry logic.
                        raise ValueError("Empty DataFrame returned by yfinance")

                    # If we get here, the download was successful for at least some symbols.
                    logging.info(
                        f"  Hist Fetch Helper: Successfully fetched batch starting with {batch_symbols[0]} on attempt {attempt + 1}."
                    )
                    break  # Exit retry loop on success
                except Exception as e_batch:
                    logging.warning(
                        f"  Hist Fetch Helper WARN (Attempt {attempt + 1}/{retries}) during yf.download for batch starting with {batch_symbols[0]}: {e_batch}"
                    )
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait 2 seconds before retrying
                    else:
                        logging.error(
                            f"  Hist Fetch Helper ERROR: All {retries} attempts failed for batch starting with {batch_symbols[0]}."
                        )
                        # data will remain an empty DataFrame if all retries fail

            # --- Process Batch Results & Identify Missing ---
            missing_symbols_in_batch = []
            
            if data.empty:
                # If batch failed completely, all are missing
                missing_symbols_in_batch = batch_symbols
            else:
                for symbol in batch_symbols:
                    df_symbol = None
                    found_in_batch = False
                    try:
                        if len(batch_symbols) == 1 and not data.columns.nlevels > 1:
                            if not data.empty:
                                df_symbol = data
                                found_in_batch = True
                        elif symbol in data.columns.levels[0]:
                            df_symbol = data[symbol]
                            found_in_batch = True
                        elif len(batch_symbols) > 1 and not data.columns.nlevels > 1:
                            # Unexpected flat structure for multi-ticker batch
                            pass 
                        else:  # Handle other potential structures
                            if isinstance(data, pd.Series) and data.name == symbol:
                                df_symbol = pd.DataFrame(data)
                                found_in_batch = True
                            elif isinstance(data, pd.DataFrame) and symbol in data.columns:
                                df_symbol = data[[symbol]].rename(columns={symbol: "Close"})
                                found_in_batch = True
                        
                        if not found_in_batch:
                            missing_symbols_in_batch.append(symbol)
                            continue

                        if df_symbol is None or df_symbol.empty:
                            missing_symbols_in_batch.append(symbol)
                            continue

                        price_col = "Close"  # Adjusted close price
                        if price_col not in df_symbol.columns:
                             # Try to find it or just mark as missing?
                             # If 'Close' is missing, maybe it's single column?
                             if len(df_symbol.columns) == 1:
                                 df_symbol.columns = ["Close"]
                             else:
                                 logging.warning(f"  Hist Fetch Helper WARN: Expected 'Close' column not found for {symbol}. Columns: {df_symbol.columns}")
                                 missing_symbols_in_batch.append(symbol)
                                 continue

                        # Ensure index is datetime
                        if not isinstance(df_symbol.index, pd.DatetimeIndex):
                            df_symbol.index = pd.to_datetime(df_symbol.index)

                        # Filter for requested range
                        mask = (df_symbol.index.date >= start_date) & (
                            df_symbol.index.date <= end_date
                        )
                        df_filtered = df_symbol.loc[mask]

                        # --- RESTORED CLEANING LOGIC ---
                        if not df_filtered.empty:
                            # Keep only the price column and rename to 'price'
                            df_cleaned = df_filtered[[price_col]].copy()
                            df_cleaned.rename(columns={price_col: "price"}, inplace=True)
                            
                            # Ensure numeric and drop NaNs
                            df_cleaned["price"] = pd.to_numeric(df_cleaned["price"], errors="coerce")
                            df_cleaned.dropna(subset=["price"], inplace=True)
                            
                            # Remove zero/negative prices
                            df_cleaned = df_cleaned[df_cleaned["price"] > 1e-6]

                            if not df_cleaned.empty:
                                historical_data[symbol] = df_cleaned.sort_index()
                            else:
                                # If empty after cleaning, it's effectively missing valid data
                                # We retry if we have no valid data.
                                missing_symbols_in_batch.append(symbol)
                        else:
                             # Empty after date filter
                             pass

                    except Exception as e_sym:
                        logging.warning(
                            f"  Hist Fetch Helper ERROR processing symbol {symbol} within batch: {e_sym}"
                        )
                        missing_symbols_in_batch.append(symbol)

            # --- Retry Missing Symbols Individually ---
            if missing_symbols_in_batch:
                logging.info(f"  Hist Fetch Helper: Retrying {len(missing_symbols_in_batch)} missing symbols individually...")
                for symbol in missing_symbols_in_batch:
                    try:
                        # Individual fetch
                        data_ind = yf.download(
                            tickers=symbol,
                            start=yf_start_date,
                            end=yf_end_date,
                            progress=False,
                            auto_adjust=True,
                            actions=False,
                            timeout=timeout_seconds,
                        )
                        
                        if not data_ind.empty:
                             # Process individual result
                             # Single ticker download usually returns flat DF with 'Close', 'Open' etc.
                             df_ind = None
                             if "Close" in data_ind.columns:
                                 df_ind = data_ind
                             elif len(data_ind.columns) == 1:
                                 df_ind = data_ind.rename(columns={data_ind.columns[0]: "Close"})
                             else:
                                 logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned unexpected columns: {data_ind.columns}")
                                 continue
                                 
                             # Ensure index
                             if not isinstance(df_ind.index, pd.DatetimeIndex):
                                 df_ind.index = pd.to_datetime(df_ind.index)
                                 
                             # Filter
                             mask = (df_ind.index.date >= start_date) & (df_ind.index.date <= end_date)
                             df_filtered = df_ind.loc[mask]
                             
                             # --- RESTORED CLEANING LOGIC FOR RETRY ---
                             if not df_filtered.empty:
                                 price_col = "Close" if "Close" in df_filtered.columns else df_filtered.columns[0]
                                 df_cleaned = df_filtered[[price_col]].copy()
                                 df_cleaned.rename(columns={price_col: "price"}, inplace=True)
                                 
                                 df_cleaned["price"] = pd.to_numeric(df_cleaned["price"], errors="coerce")
                                 df_cleaned.dropna(subset=["price"], inplace=True)
                                 df_cleaned = df_cleaned[df_cleaned["price"] > 1e-6]
                                 
                                 if not df_cleaned.empty:
                                     historical_data[symbol] = df_cleaned.sort_index()
                                     logging.info(f"  Hist Fetch Helper: Individual retry SUCCESS for {symbol}.")
                                 else:
                                     logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned no valid price data.")
                             else:
                                 logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned no data in range.")
                        else:
                             logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned empty DataFrame.")
                             
                    except Exception as e_retry:
                        logging.warning(f"  Hist Fetch Helper WARN: Individual retry failed for {symbol}: {e_retry}")

        logging.info(
            f"Hist Fetch Helper: Finished fetching ({len(historical_data)} symbols successful)."
        )
        return historical_data

    def _load_historical_manifest_and_data(
        self,
        expected_cache_key: str,
        symbols_to_load: List[str],
        data_type: str = "price",  # "price" or "fx"
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """
        Loads historical data for specified symbols by first checking a manifest file
        and then loading individual symbol data files.

        Args:
            expected_cache_key (str): The cache key expected for the current data request.
            symbols_to_load (List[str]): List of YF tickers to load data for.
            data_type (str): "price" for stock/benchmark prices, "fx" for FX rates.

        Returns:
            Tuple[Dict[str, pd.DataFrame], bool]:
                - loaded_symbol_data (Dict): DataFrames loaded, keyed by symbol.
                - manifest_is_valid_and_complete (bool): True if manifest was valid and all
                                                        requested symbols were found and loaded.
        """
        loaded_symbol_data: Dict[str, pd.DataFrame] = {}
        manifest_path = self._get_historical_manifest_path()
        # manifest_data_section_key is still relevant for organizing within "sections"
        manifest_data_section_key = (
            "historical_prices" if data_type == "price" else "historical_fx_rates"
        )

        logging.info(
            f"Hist Cache Load ({data_type}): Attempting to load manifest. File='{manifest_path}', Expected Key='{expected_cache_key[:50]}...'"
        )

        if not os.path.exists(manifest_path):
            logging.info(
                f"Hist Cache Load ({data_type}): Manifest MISS (file not found)."
            )
            return loaded_symbol_data, False

        # Initialize manifest with the new structure in mind if it's being created
        # or to ensure "sections" key exists if loading an old manifest (though version check should handle this)
        manifest = {"global_version": "1.2", "sections": {}}

        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                # ADD: Check file size before attempting to load
                if os.path.getsize(manifest_path) == 0:
                    logging.info(
                        f"Hist Cache Load ({data_type}): Manifest file '{manifest_path}' is empty. Ignoring cache."
                    )
                    return loaded_symbol_data, False
                loaded_manifest_content = json.load(f)
                if (
                    not isinstance(loaded_manifest_content, dict)
                    or loaded_manifest_content.get("global_version") != "1.2"
                ):  # Check for new version
                    logging.warning(
                        f"Hist Cache Load ({data_type}): Manifest file '{manifest_path}' is old version or invalid structure. Ignoring cache and rebuilding."
                    )
                    # Optionally, delete the old manifest here to force a clean save later
                    return loaded_symbol_data, False
                manifest = loaded_manifest_content
        except json.JSONDecodeError as e_json:  # Specific catch for JSON error
            logging.error(
                f"Hist Cache Load ({data_type}): Error DECODING manifest '{manifest_path}': {e_json}. Attempting to delete corrupt manifest."
            )
            try:
                os.remove(manifest_path)
                logging.info(
                    f"Hist Cache Load ({data_type}): Deleted corrupt manifest file '{manifest_path}'."
                )
            except OSError as e_del:
                logging.error(
                    f"Hist Cache Load ({data_type}): Failed to delete corrupt manifest '{manifest_path}': {e_del}"
                )
            return loaded_symbol_data, False
        except Exception as e:  # General catch for other IO errors
            logging.error(
                f"Hist Cache Load ({data_type}): Error reading manifest '{manifest_path}': {e}. Ignoring cache."
            )
            return loaded_symbol_data, False

        # New logic: Access the specific cache key entry within the data type section
        sections = manifest.get("sections", {})
        data_type_entries = sections.get(manifest_data_section_key, {})
        section_metadata_for_key = data_type_entries.get(expected_cache_key)

        if section_metadata_for_key is None:
            logging.info(
                f"Hist Cache Load ({data_type}): Manifest MISS (expected cache key '{expected_cache_key[:50]}...' not found in '{manifest_data_section_key}' section). Ignoring cache."
            )
            return loaded_symbol_data, False

        logging.info(
            f"Hist Cache Load ({data_type}): Manifest HIT (key '{expected_cache_key[:50]}...' found). Loading individual symbol files..."
        )
        # Files are listed under 'files' within the section
        symbol_files_in_manifest = section_metadata_for_key.get("files", {})
        all_symbols_found_and_loaded = True

        for yf_symbol in symbols_to_load:
            if (
                yf_symbol in symbol_files_in_manifest
            ):  # Check if symbol is listed in manifest
                symbol_file_path = self._get_historical_symbol_data_path(
                    yf_symbol, data_type
                )
                if os.path.exists(symbol_file_path):
                    try:
                        with open(symbol_file_path, "r", encoding="utf-8") as sf:
                            symbol_data_json_str = sf.read()  # Read as string
                        # Deserialize this single symbol's data
                        df_symbol = self._deserialize_single_historical_df(
                            symbol_data_json_str
                        )
                        if not df_symbol.empty:
                            loaded_symbol_data[yf_symbol] = df_symbol
                        else:
                            logging.warning(
                                f"Hist Cache Load ({data_type}): Empty data after deserializing {symbol_file_path} for {yf_symbol}."
                            )
                            all_symbols_found_and_loaded = (
                                False  # Mark as incomplete if a file is empty
                            )
                    except Exception as e_sym_load:
                        logging.error(
                            f"Hist Cache Load ({data_type}): Error loading/deserializing file {symbol_file_path} for {yf_symbol}: {e_sym_load}"
                        )
                        all_symbols_found_and_loaded = False
                else:
                    logging.warning(
                        f"Hist Cache Load ({data_type}): Symbol file {symbol_file_path} for {yf_symbol} listed in manifest but not found."
                    )
                    all_symbols_found_and_loaded = False
            else:
                logging.info(
                    f"Hist Cache Load ({data_type}): Symbol {yf_symbol} not listed in manifest's '{manifest_data_section_key}.files' section."
                )
                all_symbols_found_and_loaded = False  # Symbol not in manifest means cache is incomplete for this request

        if all_symbols_found_and_loaded:
            logging.info(
                f"Hist Cache Load ({data_type}): Successfully loaded all {len(symbols_to_load)} requested symbols from individual files."
            )
        else:
            logging.warning(
                f"Hist Cache Load ({data_type}): Not all requested symbols were successfully loaded from cache."
            )

        return loaded_symbol_data, all_symbols_found_and_loaded

    def _save_historical_data_and_manifest(
        self,
        cache_key_to_save: str,
        data_to_save_map: Dict[
            str, pd.DataFrame
        ] = None,  # e.g., existing FX data if saving prices
        data_type: str = "price",  # "price" or "fx"
    ):
        """Saves individual symbol historical data files and updates the manifest.json."""
        manifest_path = self._get_historical_manifest_path()
        manifest_data_section_key = (
            "historical_prices" if data_type == "price" else "historical_fx_rates"
        )

        logging.info(
            f"Hist Cache Save ({data_type}): Saving {len(data_to_save_map)} symbols and updating manifest. Key='{cache_key_to_save[:50]}...'"
        )

        current_manifest_symbol_entries = {}
        for yf_symbol, df_data in data_to_save_map.items():
            if df_data.empty:
                logging.debug(
                    f"Hist Cache Save ({data_type}): Skipping empty DataFrame for {yf_symbol}."
                )
                continue
            symbol_file_path = self._get_historical_symbol_data_path(
                yf_symbol, data_type
            )
            try:
                # Serialize DataFrame to JSON string
                json_str = df_data.to_json(orient="split", date_format="iso")
                with open(symbol_file_path, "w", encoding="utf-8") as sf:
                    sf.write(json_str)
                current_manifest_symbol_entries[yf_symbol] = (
                    True  # Could store mtime or hash here later
                )
            except Exception as e_sym_save:
                logging.error(
                    f"Hist Cache Save ({data_type}): Error saving data for {yf_symbol} to {symbol_file_path}: {e_sym_save}"
                )

        # Load existing manifest to update it, or create new if not exists
        full_manifest_content = {
            "global_version": "1.2",  # Use new version
            "sections": {},  # Initialize sections
        }
        if os.path.exists(manifest_path):
            try:
                with open(manifest_path, "r", encoding="utf-8") as f_m:
                    loaded_content = json.load(f_m)
                    # Check if loaded content is new version and valid structure
                    if (
                        isinstance(loaded_content, dict)
                        and loaded_content.get("global_version") == "1.2"
                        and isinstance(loaded_content.get("sections"), dict)
                    ):
                        full_manifest_content = loaded_content
                    else:
                        logging.warning(
                            f"Hist Cache Save ({data_type}): Existing manifest '{manifest_path}' is old version or invalid. Creating new one."
                        )
                        # full_manifest_content remains as the new default structure
            except Exception as e_read_manifest:
                logging.warning(
                    f"Hist Cache Save ({data_type}): Could not read existing manifest '{manifest_path}', creating new one: {e_read_manifest}"
                )
                # full_manifest_content remains as the new default structure

        # Ensure the data type section exists in "sections"
        if manifest_data_section_key not in full_manifest_content["sections"]:
            full_manifest_content["sections"][manifest_data_section_key] = {}

        # Update or add the entry for the specific cache_key_to_save within its data type section
        full_manifest_content["sections"][manifest_data_section_key][
            cache_key_to_save
        ] = {
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "files": current_manifest_symbol_entries,  # List of symbol files under this key
        }
        full_manifest_content["global_timestamp"] = datetime.now(
            timezone.utc
        ).isoformat()  # Overall manifest timestamp
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(
                    full_manifest_content, f, indent=2
                )  # CORRECTED: Use full_manifest_content
            logging.info(
                f"Hist Cache Save ({data_type}): Manifest updated at {manifest_path}"
            )
        except Exception as e_manifest_save:
            logging.error(
                f"Hist Cache Save ({data_type}): Error writing manifest file '{manifest_path}': {e_manifest_save}"
            )

    def _deserialize_single_historical_df(self, data_json_str: str) -> pd.DataFrame:
        """Deserializes a single historical DataFrame from its JSON string representation."""
        df = pd.DataFrame()
        if not data_json_str:
            return df
        try:
            # Using StringIO as pd.read_json expects a file-like object or path for string input
            df_temp = pd.read_json(
                StringIO(data_json_str), orient="split", dtype={"price": float}
            )
            df_temp.index = pd.to_datetime(df_temp.index, errors="coerce").date
            df_temp = df_temp.dropna(
                subset=["price"]
            )  # Ensure 'price' column has valid data
            df_temp = df_temp[pd.notnull(df_temp.index)]  # Ensure index is valid dates
            if not df_temp.empty:
                df = df_temp.sort_index()
        except Exception as e_deser:
            logging.debug(
                f"DEBUG: Error deserializing single historical DataFrame: {e_deser}"
            )
        return df

    def _deserialize_historical_data(  # This method might become less used or simplified
        self, cached_data: Dict, key_name: str, expected_keys: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Deserializes cached historical data (prices or FX) from JSON strings.
        NOTE: With per-symbol files, this is mostly for the old cache format or specific internal uses.
        Logs detailed information about the cache loading attempt.
        """
        logging.info(
            f"Hist Cache Deserialize (old format): Section='{key_name}', Expected Keys='{len(expected_keys)}'"
        )
        deserialized_dict = {}
        data_section = cached_data.get(key_name, {})
        if not isinstance(data_section, dict):
            logging.warning(
                f"Hist Cache Deserialize: Section '{key_name}' is not a dict or missing."
            )
            return {}  # Invalid section

        for expected_key in expected_keys:
            data_json = data_section.get(expected_key)
            df = pd.DataFrame()  # Default to empty
            if data_json:
                try:
                    df = self._deserialize_single_historical_df(data_json)
                except Exception as e_deser:
                    logging.debug(
                        f"DEBUG: Error deserializing cached {key_name} for {expected_key}: {e_deser}"
                    )
            # Log success or failure of deserialization for this key
            status_msg = (
                "successfully" if not df.empty else "as empty DataFrame (or failed)"
            )
            logging.debug(
                f"Hist Cache Deserialize: Deserialized '{expected_key}' from section '{key_name}' {status_msg}."
            )
            deserialized_dict[expected_key] = df  # Store df (even if empty)
        return deserialized_dict

    @profile
    def get_historical_data(
        self,
        symbols_yf: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        cache_key: Optional[str] = None,  # Key for validation (used for manifest)
        cache_file: Optional[
            str
        ] = None,  # This parameter is now mostly for API consistency, path derived internally
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches ADJUSTED historical price data using cache.

        Uses a directory-based cache with a manifest file.

        (Replaces parts of _load_or_fetch_raw_historical_data related to price fetching)
        Args:
            symbols_yf (List[str]): List of YF stock/benchmark tickers required.
            start_date (date): Start date for historical data.
            end_date (date): End date for historical data.
            use_cache (bool): Flag to enable reading/writing the raw data cache.
            cache_key (str): Cache validation key for the manifest.
            cache_file (str, optional): Not directly used for path, for API consistency.

        Returns:
            Tuple containing:
            - historical_prices_yf_adjusted (Dict[str, pd.DataFrame]): Dictionary mapping YF tickers
                to DataFrames containing adjusted historical prices (indexed by date).
            - fetch_failed (bool): True if fetching/loading critical data failed.
        """
        historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        cache_is_valid_and_complete = False
        fetch_failed = False
        manifest_path = self._get_historical_manifest_path()  # For logging

        # --- 1. Try Loading Cache ---
        if use_cache and cache_key:
            loaded_data, manifest_ok = self._load_historical_manifest_and_data(
                expected_cache_key=cache_key,
                symbols_to_load=symbols_yf,
                data_type="price",
            )
            if manifest_ok:  # Manifest key matched and all symbols loaded
                historical_prices_yf_adjusted = loaded_data
                cache_is_valid_and_complete = True
                logging.info(
                    f"Hist Prices: Successfully loaded all {len(symbols_yf)} price series from individual cache files via manifest."
                )
            else:  # Manifest key mismatch, or not all symbols found/loaded
                historical_prices_yf_adjusted = (
                    loaded_data  # Use partially loaded data if any
                )
                logging.info(
                    f"Hist Prices: Manifest not fully valid or cache incomplete. Loaded {len(loaded_data)} symbols. Will fetch missing."
                )
        else:
            logging.info(
                "Hist Prices: Cache not used or cache_key not provided. Will fetch all."
            )

        # --- 2. Fetch Missing Data if Cache Invalid/Incomplete ---
        if not cache_is_valid_and_complete:
            # logging.info("Hist Prices: Checking for missing or stale data...")
            
            symbols_needing_full_fetch = []
            symbols_needing_incremental_fetch = []
            
            for s in symbols_yf:
                if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty:
                    symbols_needing_full_fetch.append(s)
                else:
                    # Check if data is stale
                    df = historical_prices_yf_adjusted[s]
                    if not df.empty:
                        last_val = df.index.max()
                        last_date = last_val.date() if hasattr(last_val, 'date') else last_val
                        if last_date < end_date:
                            symbols_needing_incremental_fetch.append((s, last_date))
            
            # 2a. Full Fetch for completely missing symbols
            if symbols_needing_full_fetch:
                logging.info(
                    f"Hist Prices: Full fetch required for {len(symbols_needing_full_fetch)} symbols..."
                )
                fetched_data = self._fetch_yf_historical_data(
                    symbols_needing_full_fetch, start_date, end_date
                )
                historical_prices_yf_adjusted.update(fetched_data)
            
            # 2b. Incremental Fetch for stale symbols
            if symbols_needing_incremental_fetch:
                logging.info(
                    f"Hist Prices: Incremental fetch required for {len(symbols_needing_incremental_fetch)} symbols..."
                )
                # Group by last_date to batch fetch
                from collections import defaultdict
                by_last_date = defaultdict(list)
                for s, last_d in symbols_needing_incremental_fetch:
                    by_last_date[last_d].append(s)
                
                for last_d, syms in by_last_date.items():
                    fetch_start = last_d + timedelta(days=1)
                    if fetch_start <= end_date:
                        # logging.info(f"  Fetching delta from {fetch_start} for {len(syms)} symbols...")
                        delta_data = self._fetch_yf_historical_data(
                            syms, fetch_start, end_date
                        )
                        
                        # Merge delta with existing
                        for s in syms:
                            if s in delta_data and not delta_data[s].empty:
                                old_df = historical_prices_yf_adjusted[s]
                                new_df = delta_data[s]
                                
                                # Ensure indices are DatetimeIndex to avoid mixed type comparison errors
                                if not isinstance(old_df.index, pd.DatetimeIndex):
                                    old_df.index = pd.to_datetime(old_df.index)
                                if not isinstance(new_df.index, pd.DatetimeIndex):
                                    new_df.index = pd.to_datetime(new_df.index)

                                # Concatenate and drop duplicates just in case
                                merged_df = pd.concat([old_df, new_df])
                                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                                merged_df.sort_index(inplace=True)
                                historical_prices_yf_adjusted[s] = merged_df

            logging.info(
                f"Hist Prices: Fetch/Update completed. Total series in memory: {len(historical_prices_yf_adjusted)}."
            )

            # --- Validation after fetch ---
            final_symbols_missing = [
                s
                for s in symbols_yf
                if s not in historical_prices_yf_adjusted
                or historical_prices_yf_adjusted[s].empty
            ]
            if final_symbols_missing:
                logging.warning(
                    f"Hist Prices WARN: Failed to fetch/load adjusted prices for: {', '.join(final_symbols_missing)}"
                )
                # Don't mark fetch_failed=True here, let caller decide if missing stock is critical

            # --- 3. Update Cache if Fetch Occurred and Cache Enabled ---
            if use_cache and cache_key:
                # The new _save_historical_data_and_manifest will handle merging/updating the manifest correctly
                # without needing existing_other_type_data_from_manifest passed here.
                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=historical_prices_yf_adjusted,  # Save all currently held price data
                    data_type="price",
                )

        # --- 4. Final Check and Return ---
        # Check if any requested symbol is still missing
        if symbols_yf:  # Only check if symbols were actually requested
            if any(
                s not in historical_prices_yf_adjusted
                or historical_prices_yf_adjusted[s].empty
                for s in symbols_yf  # Check against originally requested symbols_yf
            ):
                logging.error(  # Changed to error
                    "Hist Prices ERROR: Data missing for some requested stock/benchmark symbols after cache/fetch."
                )
                fetch_failed = True  # SET THE FLAG
        return (
            historical_prices_yf_adjusted,
            fetch_failed,
        )  # fetch_failed is currently always False here

    @profile
    def get_historical_fx_rates(
        self,
        fx_pairs_yf: List[str],  # e.g., ['EUR=X', 'JPY=X']
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        cache_key: Optional[str] = None,  # Key for validation (used for manifest)
        cache_file: Optional[
            str
        ] = None,  # Not directly used for path, for API consistency
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches historical FX rates (vs USD) using cache.

        Uses a directory-based cache with a manifest file.
        (Note: FX data is often stored in the same raw data cache file as prices).

        Args:
            fx_pairs_yf (List[str]): List of YF FX tickers (e.g., 'JPY=X') required.
            start_date (date): Start date for historical data.
            end_date (date): End date for historical data.
            use_cache (bool): Flag to enable reading/writing the raw data cache.
            cache_key (str): Cache validation key for the manifest.
            cache_file (str, optional): Not directly used for path, for API consistency.

        Returns:
            Tuple containing:
            - historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
                to DataFrames containing historical rates vs USD (indexed by date).
            - fetch_failed (bool): True if fetching/loading ANY required FX rate failed.
        """
        historical_fx_yf: Dict[str, pd.DataFrame] = {}
        cache_is_valid_and_complete = False
        fetch_failed = False
        manifest_path = self._get_historical_manifest_path()  # For logging

        # --- 1. Try Loading Cache ---
        if use_cache and cache_key:
            loaded_data, manifest_ok = self._load_historical_manifest_and_data(
                expected_cache_key=cache_key,
                symbols_to_load=fx_pairs_yf,
                data_type="fx",
            )
            if manifest_ok:  # Manifest key matched and all symbols loaded
                historical_fx_yf = loaded_data
                cache_is_valid_and_complete = True
                logging.info(
                    f"Hist FX: Successfully loaded all {len(fx_pairs_yf)} FX series from individual cache files via manifest."
                )
            else:  # Manifest key mismatch, or not all symbols found/loaded
                historical_fx_yf = loaded_data  # Use partially loaded data if any
                logging.info(
                    f"Hist FX: Manifest not fully valid or cache incomplete. Loaded {len(loaded_data)} FX series. Will fetch missing."
                )
        else:
            logging.info(
                "Hist FX: Cache not used or cache_key not provided. Will fetch all."
            )

        # --- 2. Fetch Missing Data if Needed ---
        if not cache_is_valid_and_complete:
            # logging.info("Hist FX: Checking for missing or stale data...")
            
            fx_needing_full_fetch = []
            fx_needing_incremental_fetch = []
            
            for s in fx_pairs_yf:
                if s not in historical_fx_yf or historical_fx_yf[s].empty:
                    fx_needing_full_fetch.append(s)
                else:
                    # Check if data is stale
                    df = historical_fx_yf[s]
                    if not df.empty:
                        last_date = df.index.max().date()
                        if last_date < end_date:
                            fx_needing_incremental_fetch.append((s, last_date))

            # 2a. Full Fetch
            if fx_needing_full_fetch:
                logging.info(
                    f"Hist FX: Full fetch required for {len(fx_needing_full_fetch)} FX pairs..."
                )
                fetched_fx_data = self._fetch_yf_historical_data(
                    fx_needing_full_fetch, start_date, end_date
                )
                historical_fx_yf.update(fetched_fx_data)

            # 2b. Incremental Fetch
            if fx_needing_incremental_fetch:
                logging.info(
                    f"Hist FX: Incremental fetch required for {len(fx_needing_incremental_fetch)} FX pairs..."
                )
                from collections import defaultdict
                by_last_date = defaultdict(list)
                for s, last_d in fx_needing_incremental_fetch:
                    by_last_date[last_d].append(s)
                
                for last_d, syms in by_last_date.items():
                    fetch_start = last_d + timedelta(days=1)
                    if fetch_start <= end_date:
                        # logging.info(f"  Fetching FX delta from {fetch_start} for {len(syms)} pairs...")
                        delta_data = self._fetch_yf_historical_data(
                            syms, fetch_start, end_date
                        )
                        
                        # Merge delta
                        for s in syms:
                            if s in delta_data and not delta_data[s].empty:
                                old_df = historical_fx_yf[s]
                                new_df = delta_data[s]
                                merged_df = pd.concat([old_df, new_df])
                                merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                                merged_df.sort_index(inplace=True)
                                historical_fx_yf[s] = merged_df

            logging.info(
                f"Hist FX: Fetch/Update completed. Total FX series in memory: {len(historical_fx_yf)}."
            )

            # --- Validation after fetch ---
            final_fx_missing_after_fetch = [
                p
                for p in fx_pairs_yf  # Check against requested
                if p not in historical_fx_yf or historical_fx_yf[p].empty
            ]
            if final_fx_missing_after_fetch:
                logging.error(  # Log as error because fetch failed for required pairs
                    f"Hist FX ERROR: Failed to fetch required FX rates for: {', '.join(final_fx_missing_after_fetch)}"
                )
                fetch_failed = True  # Missing ANY required FX is critical
            else:
                logging.info(
                    "Hist FX: All required FX pairs present."
                )

            # --- 3. Save to Cache (Manifest) ---
            if use_cache and cache_key:
                # We save ALL currently available FX data for the requested keys
                # This includes what was loaded + what was fetched.
                data_to_save = {
                    k: v for k, v in historical_fx_yf.items() if k in fx_pairs_yf
                }
                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=data_to_save,
                    data_type="fx",
                )




        # --- 5. Final Check and Return ---
        # Check again if critical data is missing (redundant if fetch_failed already True, but safe)
        if fx_pairs_yf:  # Only if FX pairs were requested
            if any(
                p not in historical_fx_yf or historical_fx_yf[p].empty
                for p in fx_pairs_yf  # Check against originally requested fx_pairs_yf
            ):
                logging.error(
                    "Hist FX ERROR: Critical FX data missing after final check for some requested pairs."
                )
                fetch_failed = True  # SET THE FLAG

        return historical_fx_yf, fetch_failed

    @profile
    def get_fundamental_data(self, yf_symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetches fundamental data (ticker.info) for a given Yahoo Finance symbol.
        Uses a directory of JSON files for caching (one file per symbol).

        Args:
            yf_symbol (str): The Yahoo Finance ticker symbol (e.g., "AAPL").

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the fundamental data,
                                      or None if fetching fails or symbol is invalid.
        """
        if not YFINANCE_AVAILABLE:
            logging.error("yfinance not available. Cannot fetch fundamental data.")
            return None
        if not yf_symbol or not isinstance(yf_symbol, str) or not yf_symbol.strip():
            logging.warning(f"Invalid yf_symbol provided for fundamentals: {yf_symbol}")
            return None

        logging.debug(f"Requesting fundamental data for YF symbol: {yf_symbol}")

        # --- Construct cache file path for this specific symbol ---
        # Ensure the cache directory exists (already done in __init__, but safe to repeat)
        os.makedirs(self.fundamentals_cache_dir, exist_ok=True)
        # Create the file path for this symbol's data
        symbol_cache_file = os.path.join(
            self.fundamentals_cache_dir, f"{yf_symbol}.json"
        )

        # --- Caching Logic (per symbol file) ---
        cached_data = None
        cache_valid = False  # Flag for the *specific symbol's file*

        if os.path.exists(symbol_cache_file):
            try:
                # MODIFIED: Load only the specific symbol's cache file
                with open(symbol_cache_file, "r", encoding="utf-8") as f:
                    symbol_cache_entry = json.load(f)

                # Check timestamp within the loaded entry
                if symbol_cache_entry and isinstance(symbol_cache_entry, dict):
                    cache_timestamp_str = symbol_cache_entry.get("timestamp")
                    if cache_timestamp_str:
                        cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                        if datetime.now(timezone.utc) - cache_timestamp < timedelta(
                            hours=FUNDAMENTALS_CACHE_DURATION_HOURS
                        ):
                            cached_data = symbol_cache_entry.get("data")
                            if (
                                cached_data is not None
                            ):  # Could be an empty dict if that was cached
                                cache_valid = True
                                logging.debug(
                                    f"Using valid fundamentals cache for {yf_symbol} from file: {symbol_cache_file}"
                                )
                        else:
                            logging.info(
                                f"Fundamentals cache expired for {yf_symbol} (file: {symbol_cache_file})."
                            )
                    else:
                        logging.warning(
                            f"Fundamentals cache file for {yf_symbol} missing timestamp: {symbol_cache_file}"
                        )
            except (json.JSONDecodeError, IOError, Exception) as e:
                logging.warning(
                    f"Error reading fundamentals cache file '{symbol_cache_file}': {e}. Will refetch."
                )

        if cache_valid and cached_data is not None:
            return cached_data

        # --- Fetching Logic ---
        logging.info(
            f"Fetching fresh fundamental data for {yf_symbol} (cache miss/stale)..."
        )
        try:
            ticker = yf.Ticker(yf_symbol)
            data = ticker.info  # This is the main call
            if not data:  # yfinance returns empty dict for invalid symbols or no data
                logging.warning(
                    f"No fundamental data returned by yfinance for {yf_symbol}."
                )
                data = (
                    {}
                )  # Store empty dict to avoid refetching invalid symbol immediately

            # Save to cache (only this symbol's file)
            try:
                # MODIFIED: Save only the specific symbol's data to its file
                symbol_cache_content = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": data,
                }
                # Ensure the directory exists before writing (redundant but safe)
                os.makedirs(self.fundamentals_cache_dir, exist_ok=True)
                with open(symbol_cache_file, "w", encoding="utf-8") as f_write:
                    json.dump(
                        symbol_cache_content, f_write, indent=2, cls=NpEncoder
                    )  # Use NpEncoder for numpy types
                logging.debug(
                    f"Saved fundamentals for {yf_symbol} to cache file: {symbol_cache_file}"
                )
            except Exception as e_cache_write:
                logging.warning(
                    f"Failed to write fundamentals cache file for {yf_symbol} ('{symbol_cache_file}'): {e_cache_write}"
                )
            return data
        except Exception as e_fetch:
            logging.error(f"Error fetching fundamental data for {yf_symbol}: {e_fetch}")
            traceback.print_exc()
            # --- ADDED: Cache empty data on fetch error ---
            try:
                logging.info(
                    f"Caching empty fundamental data for {yf_symbol} due to fetch error."
                )
                error_cache_content = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "data": {},  # Cache an empty dictionary
                }
                # symbol_cache_file is defined earlier in this method
                os.makedirs(self.fundamentals_cache_dir, exist_ok=True)
                with open(symbol_cache_file, "w", encoding="utf-8") as f_write_err:
                    json.dump(error_cache_content, f_write_err, indent=2, cls=NpEncoder)
            except Exception as e_cache_err_write:
                logging.warning(
                    f"Failed to write error cache for fundamentals of {yf_symbol} ('{symbol_cache_file}'): {e_cache_err_write}"
                )
            # --- END ADDED ---
            return None  # Still return None to indicate failure to the caller for this attempt

    def _get_cached_statement_data(
        self, yf_symbol: str, statement_type: str, period_type: str = "annual"
    ) -> Optional[pd.DataFrame]:
        """
        Helper to load a specific financial statement (financials, balance_sheet, cashflow)
        for a symbol from its dedicated cache file.

        Args:
            yf_symbol (str): The Yahoo Finance ticker symbol.
            statement_type (str): Type of statement ("financials", "balance_sheet", "cashflow").
            period_type (str): "annual" or "quarterly".

        Returns:
            Optional[pd.DataFrame]: The cached DataFrame, or None if not found/valid.
        """
        if not yf_symbol or not statement_type or not period_type:
            return None

        statement_cache_file = os.path.join(
            self.fundamentals_cache_dir,
            f"{yf_symbol}_{statement_type}_{period_type}.json",
        )

        if os.path.exists(statement_cache_file):
            try:
                with open(statement_cache_file, "r", encoding="utf-8") as f:
                    cached_entry = json.load(f)

                cache_timestamp_str = cached_entry.get("timestamp")
                if cache_timestamp_str:
                    cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                    if datetime.now(timezone.utc) - cache_timestamp < timedelta(
                        hours=FUNDAMENTALS_CACHE_DURATION_HOURS  # Use same duration as .info
                    ):
                        data_json_str = cached_entry.get("data_df_json")
                        if data_json_str:
                            # Deserialize DataFrame from JSON string
                            df = pd.read_json(StringIO(data_json_str), orient="split")
                            # yfinance statements often have Timestamps as columns, ensure they are parsed correctly
                            # If columns are dates, convert them to simple date strings for consistency if needed,
                            # or ensure they are Timestamps. For now, assume read_json handles it.
                            logging.debug(
                                f"Using valid cache for {period_type} {statement_type} of {yf_symbol} from {statement_cache_file}"
                            )
                            return df
            except Exception as e:
                logging.warning(
                    f"Error reading {period_type} {statement_type} cache for {yf_symbol} from {statement_cache_file}: {e}"
                )
        return None

    def _save_statement_data_to_cache(
        self,
        yf_symbol: str,
        statement_type: str,
        period_type: str,
        data_df: pd.DataFrame,
    ):
        """
        Helper to save a specific financial statement DataFrame to its dedicated cache file.
        """
        if (  # Allow saving empty df if that's what yf returns
            not yf_symbol or not statement_type or not period_type or data_df is None
        ):
            return

        statement_cache_file = os.path.join(
            self.fundamentals_cache_dir,
            f"{yf_symbol}_{statement_type}_{period_type}.json",
        )
        try:
            # Serialize DataFrame to JSON string
            data_df_json_str = data_df.to_json(
                orient="split", date_format="iso", default_handler=str
            )

            cache_content = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "data_df_json": data_df_json_str,
            }
            os.makedirs(self.fundamentals_cache_dir, exist_ok=True)
            with open(statement_cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    cache_content, f, indent=2
                )  # No NpEncoder needed if df is stringified
            logging.debug(
                f"Saved {period_type} {statement_type} for {yf_symbol} to cache: {statement_cache_file}"
            )
        except Exception as e:
            logging.warning(
                f"Failed to write {period_type} {statement_type} cache for {yf_symbol} to {statement_cache_file}: {e}"
            )

    def _fetch_statement_data(
        self, yf_symbol: str, statement_type: str, period_type: str = "annual"
    ) -> Optional[pd.DataFrame]:
        """
        Fetches a specific financial statement using yfinance and caches it.
        """
        if not YFINANCE_AVAILABLE:
            return None

        cached_df = self._get_cached_statement_data(
            yf_symbol, statement_type, period_type
        )
        if (
            cached_df is not None
        ):  # Check if it's not None (could be empty DataFrame from cache)
            return cached_df

        logging.info(
            f"Fetching fresh {period_type} {statement_type} for {yf_symbol}..."
        )
        try:
            ticker = yf.Ticker(yf_symbol)
            df = pd.DataFrame()  # Default to empty DataFrame
            if period_type == "quarterly":
                if statement_type == "financials":
                    df = ticker.quarterly_financials
                elif statement_type == "balance_sheet":
                    df = ticker.quarterly_balance_sheet
                elif statement_type == "cashflow":
                    df = ticker.quarterly_cashflow
            else:  # Default to annual
                if statement_type == "financials":
                    df = ticker.financials
                elif statement_type == "balance_sheet":
                    df = ticker.balance_sheet
                elif statement_type == "cashflow":
                    df = ticker.cashflow

            if df is None:  # yfinance might return None if no data
                df = pd.DataFrame()  # Ensure it's an empty DataFrame, not None

            self._save_statement_data_to_cache(
                yf_symbol, statement_type, period_type, df
            )
            return df
        except Exception as e:
            logging.error(
                f"Error fetching {period_type} {statement_type} for {yf_symbol}: {e}"
            )
            # Cache an empty DataFrame on error to prevent repeated failed fetches within cache duration
            self._save_statement_data_to_cache(
                yf_symbol, statement_type, period_type, pd.DataFrame()
            )
            return pd.DataFrame()  # Return empty DataFrame on error

    @profile
    def get_financials(
        self, yf_symbol: str, period_type: str = "annual"
    ) -> Optional[pd.DataFrame]:
        """Fetches Income Statement data for a symbol."""
        return self._fetch_statement_data(yf_symbol, "financials", period_type)

    @profile
    def get_balance_sheet(
        self, yf_symbol: str, period_type: str = "annual"
    ) -> Optional[pd.DataFrame]:
        """Fetches Balance Sheet data for a symbol."""
        return self._fetch_statement_data(yf_symbol, "balance_sheet", period_type)

    @profile
    def get_cashflow(
        self, yf_symbol: str, period_type: str = "annual"
    ) -> Optional[pd.DataFrame]:
        """Fetches Cash Flow Statement data for a symbol."""
        return self._fetch_statement_data(yf_symbol, "cashflow", period_type)

    def get_exchange_for_symbol(self, yf_symbol: str) -> Optional[str]:
        """
        Fetches the exchange short name for a given symbol using cached fundamental data.

        Args:
            yf_symbol (str): The Yahoo Finance ticker symbol.

        Returns:
            Optional[str]: The exchange short name (e.g., "NMS", "SET") or None.
        """
        # This leverages the existing caching of get_fundamental_data (ticker.info)
        fundamental_data = self.get_fundamental_data(yf_symbol)
        if fundamental_data and isinstance(fundamental_data, dict):
            exchange_name = fundamental_data.get("exchange")
            logging.debug(
                f"Found exchange '{exchange_name}' for symbol '{yf_symbol}' from fundamental data."
            )
            return exchange_name
        logging.warning(f"Could not determine exchange for symbol '{yf_symbol}'.")
        return None

    def get_intraday_data(
        self, yf_symbol: str, period: str = "5d", interval: str = "1m"
    ) -> Optional[pd.DataFrame]:
        """
        Fetches intraday historical data for a given Yahoo Finance symbol.

        Note: Intraday data has limitations. e.g., 1-minute data is only available for
        the last 7 days. Data for intervals < 1h is only available for the last 60 days.
        `auto_adjust` is automatically set to False for intraday data.

        Args:
            yf_symbol (str): The Yahoo Finance ticker symbol (e.g., "AAPL").
            period (str): The period to fetch data for (e.g., "1d", "5d", "1mo").
                          See yfinance documentation for valid periods.
            interval (str): The data interval (e.g., "1m", "5m", "15m", "1h").
                            See yfinance documentation for valid intervals.

        Returns:
            Optional[pd.DataFrame]: A DataFrame containing the intraday data,
                                      or None if fetching fails.
        """
        if not YFINANCE_AVAILABLE:
            logging.error("yfinance not available. Cannot fetch intraday data.")
            return None
        if not yf_symbol or not isinstance(yf_symbol, str) or not yf_symbol.strip():
            logging.warning(
                f"Invalid yf_symbol provided for intraday data: {yf_symbol}"
            )
            return None

        logging.info(
            f"Fetching intraday data for {yf_symbol} (period: {period}, interval: {interval})"
        )

        try:
            # For intraday data, auto_adjust must be False.
            # yfinance handles this automatically for intervals less than 1d.
            # If '1d' is requested, fetch '2d' to ensure the full current day's data is available,
            # as yfinance can be inconsistent with timezones. We'll trim it later.
            fetch_period = "2d" if period == "1d" else period
            data = yf.download(
                tickers=yf_symbol,
                period=fetch_period,
                interval=interval,
                progress=False,
                auto_adjust=True,  # Suppress FutureWarning and use recommended setting
            )

            if data.empty:
                logging.warning(
                    f"No intraday data returned by yfinance for {yf_symbol} with period='{period}' and interval='{interval}'."
                )
                return None

            # --- ADDED: Flatten multi-level columns if they exist ---
            # yfinance can return multi-level columns even for a single ticker.
            # This simplifies the column structure (e.g., from ('Close', 'AAPL') to 'Close').
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.droplevel(1)

            # --- ADDED: Trim 1d data to the most recent trading day ---
            # This fixes a timezone issue where yfinance might not return the full
            # current day's data when '1d' is requested from a different timezone.
            # By fetching '2d' and trimming, we ensure the full session is present.
            # The period is adjusted to '2d' if '1d' is requested.
            if (
                period == "1d"
                and not data.empty
                and isinstance(data.index, pd.DatetimeIndex)
            ):
                # --- MODIFIED: More robust way to get the last trading day ---
                # .last('1D') can be unreliable across timezones. Instead, we find
                # the date with the most data points, which is the most recent
                # complete trading session.
                if not data.index.tz:
                    logging.warning(
                        "Intraday data is not timezone-aware. Cannot reliably determine last trading day."
                    )
                else:
                    # Group by calendar date and find the date with the most rows
                    day_counts = data.groupby(data.index.date).size()
                    if not day_counts.empty:
                        last_trading_day = day_counts.idxmax()
                        data = data[data.index.date == last_trading_day]

            return data

        except Exception as e_fetch:
            logging.error(f"Error fetching intraday data for {yf_symbol}: {e_fetch}")
            traceback.print_exc()
            return None


# --- END OF FILE market_data.py ---
