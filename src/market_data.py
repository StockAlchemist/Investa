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
# Lazy load yfinance to improve startup time
yf = None
YFINANCE_AVAILABLE = None

def _ensure_yfinance():
    global yf, YFINANCE_AVAILABLE
    if YFINANCE_AVAILABLE is not None:
        return YFINANCE_AVAILABLE

    try:
        import yfinance as _yf
        yf = _yf
        YFINANCE_AVAILABLE = True
    except ImportError:
        logging.warning(
            "Warning: yfinance library not found. Market data fetching will fail."
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
    
    return YFINANCE_AVAILABLE

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
        METADATA_CACHE_FILE_NAME,  # <-- ADDED
        METADATA_CACHE_DURATION_DAYS,  # <-- ADDED
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

INVALID_SYMBOLS_CACHE_FILE = "invalid_symbols_cache.json"
INVALID_SYMBOLS_DURATION = 24 * 60 * 60  # 24 hours in seconds

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
                # --- FIX: Ensure consistent path structure (Org/App) ---
                expected_suffix = os.path.join(ORG_NAME, APP_NAME)
                if not cache_dir_base.endswith(expected_suffix):
                     app_cache_dir = os.path.join(cache_dir_base, ORG_NAME, APP_NAME)
                else:
                     app_cache_dir = cache_dir_base
                
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
                # --- FIX: Ensure consistent path structure (Org/App) ---
                expected_suffix = os.path.join(ORG_NAME, APP_NAME)
                if not cache_dir_base.endswith(expected_suffix):
                     app_cache_dir = os.path.join(cache_dir_base, ORG_NAME, APP_NAME)
                else:
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

    def _get_metadata_cache_path(self) -> str:
        """Returns the full path to the metadata cache file."""
        return os.path.join(self._get_historical_cache_dir(), METADATA_CACHE_FILE_NAME)

    def _load_metadata_cache(self) -> Dict[str, Dict]:
        """Loads the metadata cache (Name, Currency) from disk."""
        path = self._get_metadata_cache_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading metadata cache: {e}")
            return {}

    def _save_metadata_cache(self, cache: Dict[str, Dict]):
        """Saves the metadata cache to disk."""
        path = self._get_metadata_cache_path()
        try:
            with open(path, "w") as f:
                json.dump(cache, f, indent=2)
        except Exception as e:
            logging.warning(f"Error saving metadata cache: {e}")

    def _ensure_metadata_batch(self, yf_symbols: Set[str]) -> Dict[str, Dict]:
        """
        Ensures metadata (Name, Currency) exists and is fresh for given symbols.
        Fetches missing data and updates cache.
        Returns a dict of metadata keyed by YF symbol.
        """
        cache = self._load_metadata_cache()
        now_ts = datetime.now(timezone.utc)
        missing_symbols = []
        
        # Ensure yfinance is loaded
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
            return cache
        
        # Check cache validity
        for sym in yf_symbols:
            entry = cache.get(sym)
            if not entry:
                missing_symbols.append(sym)
                continue
                
            ts_str = entry.get("timestamp")
            if not ts_str:
                missing_symbols.append(sym)
                continue
                
            try:
                entry_ts = datetime.fromisoformat(ts_str)
                # Check expiration
                if (now_ts - entry_ts).days > METADATA_CACHE_DURATION_DAYS:
                    missing_symbols.append(sym)
            except ValueError:
                missing_symbols.append(sym)

        if missing_symbols:
            logging.info(f"Metadata Cache: Fetching missing metadata for {len(missing_symbols)} symbols...")
            try:
                 # Helper to fetch info for a single ticker to avoid deep nesting usage
                 def fetch_single_meta(t_obj, symbol_name):
                     try:
                         # Try fast_info for currency first? No, just use .info because we need name anyway.
                         # Unless we want to skip name? No, user wants aesthetics.
                         info = t_obj.info
                         name = info.get("shortName") or info.get("longName") or symbol_name
                         currency = info.get("currency")
                         return {"name": name, "currency": currency}
                     except Exception:
                         return None

                 # Use yf.Tickers for cleaner API usage
                 chunk_size = 50
                 for i in range(0, len(missing_symbols), chunk_size):
                     chunk = missing_symbols[i:i+chunk_size]
                     tickers = yf.Tickers(" ".join(chunk))
                     
                     for sym, ticker in tickers.tickers.items():
                         meta = fetch_single_meta(ticker, sym)
                         if meta:
                             cache[sym] = {
                                 "name": meta["name"],
                                 "currency": meta["currency"],
                                 "timestamp": now_ts.isoformat()
                             }
                         else:
                             logging.warning(f"Failed to fetch metadata for {sym}. Using placehoders.")
                             # Cache placeholder to avoid retry loop
                             if sym not in cache:
                                 cache[sym] = {
                                     "name": sym,
                                     "currency": None, 
                                     "timestamp": now_ts.isoformat()
                                 }
            except Exception as e_batch:
                 logging.error(f"Error in metadata batch fetch: {e_batch}")
            
            self._save_metadata_cache(cache)
            
        return cache

    def _get_fundamentals_cache_path(self) -> str:
        """Returns the full path to the aggregate fundamentals cache file."""
        return os.path.join(self.fundamentals_cache_dir, "fundamentals_aggregate.json")

    def _load_fundamentals_cache(self) -> Dict[str, Dict]:
        """Loads the aggregate fundamentals cache from disk."""
        path = self._get_fundamentals_cache_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            logging.warning(f"Error loading fundamentals cache: {e}")
            return {}

    def _save_fundamentals_cache(self, cache: Dict[str, Dict]):
        """Saves the aggregate fundamentals cache to disk."""
        path = self._get_fundamentals_cache_path()
        try:
            with open(path, "w") as f:
                json.dump(cache, f, indent=2, cls=NpEncoder)
        except Exception as e:
            logging.warning(f"Error saving fundamentals cache: {e}")

    def get_fundamental_data_batch(self, yf_symbols: Set[str]) -> Dict[str, Dict]:
        """
        Fetches fundamental data (specifically dividend info) for the given symbols.
        Uses caching to avoid slow fetching.
        """
        cache = self._load_fundamentals_cache()
        now_ts = datetime.now(timezone.utc)
        missing_symbols = []
        
        # Ensure yfinance is loaded
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
            return cache  # Return whatever is in cache if yf unavailable

        # Check cache validity
        for sym in yf_symbols:
            entry = cache.get(sym)
            if not entry:
                missing_symbols.append(sym)
                continue
            
            ts_str = entry.get("timestamp")
            if not ts_str:
                missing_symbols.append(sym)
                continue
            
            try:
                entry_ts = datetime.fromisoformat(ts_str)
                # Cache valid for 24 hours (fundamentals don't change often)
                if (now_ts - entry_ts).days >= 1: # 1 day expiration
                     missing_symbols.append(sym)
            except ValueError:
                missing_symbols.append(sym)
        
        if missing_symbols:
            logging.info(f"Fundamentals: Fetching missing data for {len(missing_symbols)} symbols...")
            try:
                chunk_size = 50
                for i in range(0, len(missing_symbols), chunk_size):
                    chunk = missing_symbols[i:i+chunk_size]
                    
                    # Using yf.Tickers to get info
                    # Note: yf.Tickers(list).tickers returns a dict of Ticker objects
                    # Accessing .info properties usually triggers the fetch
                    tickers = yf.Tickers(" ".join(chunk))
                    
                    # Force fetch by accessing info
                    for sym, ticker in tickers.tickers.items():
                        try:
                            info = ticker.info
                            
                            # Extract dividend data
                            # dividendRate is annual dividend in currency
                            # dividendYield is percentage (e.g. 0.05 for 5%)
                            div_rate = info.get("dividendRate", 0.0)
                            div_yield = info.get("dividendYield", 0.0)
                            
                            cache[sym] = {
                                "trailingAnnualDividendRate": div_rate if div_rate else 0.0,
                                "dividendYield": div_yield if div_yield else 0.0,
                                "timestamp": now_ts.isoformat()
                            }
                        except Exception as e_tick:
                            logging.warning(f"Error fetching fundamentals for {sym}: {e_tick}")
                            # Cache failure to avoid retry loop for this run? 
                            # Maybe not, transient errors exist. 
                            # But if persistent, it slows down startup.
                            # Let's verify if we should cache 'zero' or nothing.
                            # If we don't cache, we retry next time.
                            pass
                            
            except Exception as e_batch:
                logging.error(f"Error in fundamentals batch fetch: {e_batch}")
            
            self._save_fundamentals_cache(cache)
            
        return cache
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
        OPTIMIZED: Uses batch fetching (yf.download) and metadata caching.
        """
        logging.info(
            f"Getting current quotes for {len(internal_stock_symbols)} symbols and FX for {len(required_currencies)} symbols."
        )
        
        # Ensure yfinance is available
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
            return {}, {}, {}, True, True # Treat as error

        has_warnings = False
        has_errors = False
        results = {}

        # --- 1. Map internal symbols to YF tickers ---
        yf_symbols_to_fetch = set()
        internal_to_yf_map_local = {}
        for internal_symbol in set(internal_stock_symbols):
            if is_cash_symbol(internal_symbol):
                continue
            yf_symbol = map_to_yf_symbol(
                internal_symbol, user_symbol_map, user_excluded_symbols
            )
            if yf_symbol:
                yf_symbols_to_fetch.add(yf_symbol)
                internal_to_yf_map_local[internal_symbol] = yf_symbol
            else:
                logging.debug(f"Symbol '{internal_symbol}' excluded or unmappable. Skipping.")
                has_warnings = True

        if not yf_symbols_to_fetch:
            # Still process FX if needed, so don't return early
            pass

        # --- 2. Caching Logic for Current Quotes (Short-term Price Cache) ---
        cache_key = f"CURRENT_QUOTES_v7_FINAL::{'_'.join(sorted(yf_symbols_to_fetch))}::{'_'.join(sorted(required_currencies))}"
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
                        if datetime.now(timezone.utc) - cache_timestamp < timedelta(
                            minutes=CURRENT_QUOTE_CACHE_DURATION_MINUTES
                        ):
                            cached_quotes = cache_data.get("quotes")
                            cached_fx = cache_data.get("fx_rates")
                            cached_fx_prev = cache_data.get("fx_prev_close")
                            if cached_quotes is not None and cached_fx is not None:
                                cache_valid = True
                                logging.info(
                                    f"Using valid cache for current quotes (Key: {cache_key[:30]}...). Age: {datetime.now(timezone.utc) - cache_timestamp}"
                                )
            except Exception as e:
                logging.warning(f"Error reading current quotes cache: {e}")

        if cache_valid:
            return cached_quotes, cached_fx, cached_fx_prev, False, has_warnings

        # --- 3. FETCHING FRESH DATA (Optimized) ---
        
        # 3a. Ensure Metadata (Name, Currency) - Long-lived cache
        metadata_map = self._ensure_metadata_batch(yf_symbols_to_fetch)
        
        # 3c. Fetch Fundamentals (Dividends) - Cached
        fundamentals_map = self.get_fundamental_data_batch(yf_symbols_to_fetch)

        # 3b. Batch Fetch Prices using yf.download (Existing Logic)
        stock_data_yf = {}
        if yf_symbols_to_fetch:
            logging.info(f"Batch fetching prices for {len(yf_symbols_to_fetch)} symbols...")
            try:
                # Use download for speed
                # threads=True is default but explicit is good.
                # progress=False suppresses stdout.
                # timeout=30 (seconds) explicitly to avoid default 10s failures on slow nets
                df = yf.download(
                    list(yf_symbols_to_fetch),
                    period="5d", 
                    group_by="ticker",
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                    timeout=30 
                )
                
                if df.empty:
                     logging.warning("Batch price fetch returned empty DataFrame.")
                     has_errors = True
                else:
                    # Process results
                    for internal_sym, yf_sym in internal_to_yf_map_local.items():
                        try:
                            price = None
                            prev_close = None
                            
                            # Handle yf.download structure
                            sym_df = pd.DataFrame()
                            if len(yf_symbols_to_fetch) > 1:
                                if isinstance(df.columns, pd.MultiIndex) and yf_sym in df.columns.levels[0]:
                                     sym_df = df[yf_sym]
                                elif isinstance(df.columns, pd.MultiIndex) and yf_sym not in df.columns.levels[0]:
                                     # Ticker missing from result
                                     pass
                                elif yf_sym in df.columns:
                                     # Flat structure?
                                     sym_df = df[[yf_sym]].rename(columns={yf_sym: "Close"}) # Rough guess
                            else:
                                # Single ticker download
                                sym_df = df
                            
                            if not sym_df.empty:
                                # Check for Close or Price column
                                col_name = "Close" if "Close" in sym_df.columns else sym_df.columns[0]
                                
                                valid_days = sym_df.dropna(subset=[col_name])
                                if not valid_days.empty:
                                    price = float(valid_days[col_name].iloc[-1])
                                    if len(valid_days) >= 2:
                                        prev_close = float(valid_days[col_name].iloc[-2])
                            
                            # Retrieve metadata
                            meta = metadata_map.get(yf_sym, {})
                            currency = meta.get("currency")
                            name = meta.get("name")
                            
                            # Retrieve fundamentals
                            fund = fundamentals_map.get(yf_sym, {})
                            
                            if price is not None and currency:
                                change = (price - prev_close) if (price and prev_close) else 0.0
                                change_pct = ((change / prev_close) * 100.0) if (change and prev_close) else 0.0
                                
                                stock_data_yf[internal_sym] = {
                                    "price": price,
                                    "change": change,
                                    "changesPercentage": change_pct,
                                    "currency": currency.upper(),
                                    "name": name,
                                    "source": "yf_batch_download",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "trailingAnnualDividendRate": fund.get("trailingAnnualDividendRate", 0),
                                    "dividendYield": fund.get("dividendYield", 0)
                                }
                            else:
                                # logging.warning(f"Missing price or currency for {yf_sym}.")
                                has_warnings = True

                        except Exception as e_sym:
                            logging.warning(f"Error processing batch result for {yf_sym}: {e_sym}")
                            has_warnings = True
                            
            except Exception as e_down:
                logging.error(f"Batch download failed: {e_down}")
                has_errors = True

        # --- 4. Fetch FX Rates ---
        fx_rates_vs_usd = {"USD": 1.0}
        fx_prev_close_vs_usd = {"USD": 1.0}
        
        if "USD" not in required_currencies:
            required_currencies.add("USD")
            
        fx_pairs = [f"{c}=X" for c in required_currencies if c != "USD"]
        
        if fx_pairs:
            logging.info(f"Fetching FX for {len(fx_pairs)} pairs...")
            try:
                # Use sequential Tickers check for FX to ensure we get directionality correctly
                # (Same logic as original roughly, but simplified)
                tickers = yf.Tickers(" ".join(fx_pairs))
                for yf_symbol, ticker_obj in tickers.tickers.items():
                    try:
                        # Use fast_info
                        fi = getattr(ticker_obj, "fast_info", None)
                        if fi:
                            price = fi.last_price
                            prev = fi.previous_close
                            quote_currency = fi.currency
                            
                            base_curr_from_symbol = yf_symbol.replace("=X", "").upper()
                            
                            # For symbols like "EUR=X", "THB=X", "JPY=X":
                            # The price returned by yfinance is consistently "Units of Currency per 1 USD".
                            # E.g. THB=X -> 34.0 (34 THB = 1 USD).
                            # E.g. EUR=X -> 0.85 (0.85 EUR = 1 USD).
                            # finutils.get_conversion_rate expects "Units per USD".
                            # Therefore, we store the price directly.
                            
                            if price and price > 0:
                                fx_rates_vs_usd[base_curr_from_symbol] = price
                                if prev and prev > 0:
                                    fx_prev_close_vs_usd[base_curr_from_symbol] = prev
                            else:
                                logging.warning(f"Invalid price {price} for FX {yf_symbol}")
                                has_warnings = True

                        else:
                             logging.warning(f"No fast_info for FX {yf_symbol}")
                             has_warnings = True
                    except Exception as e_fx_sym:
                         logging.warning(f"Error for FX {yf_symbol}: {e_fx_sym}")
                         
            except Exception as e_fx:
                logging.error(f"FX fetch error: {e_fx}")
        
        # --- 5. Save Cache ---
        if not has_errors:
            try:
                # Populate results from stock_data_yf
                for internal_sym, data in stock_data_yf.items():
                    results[internal_sym] = data
                    
                cache_content = {
                    "cache_key": cache_key,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "quotes": results, 
                    "fx_rates": fx_rates_vs_usd,
                    "fx_prev_close": fx_prev_close_vs_usd,
                }
                
                with open(self.current_cache_file, "w") as f:
                    json.dump(cache_content, f, indent=2)
            except Exception as e:
                logging.warning(f"Error saving current quotes cache: {e}")

        # Return (fresh results)
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
        
        # Ensure yfinance is available
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
             return {}

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

    def _get_invalid_symbols_cache_path(self) -> str:
        """Returns the full path to the invalid symbols cache file."""
        return os.path.join(self._get_historical_cache_dir(), INVALID_SYMBOLS_CACHE_FILE)

    def _load_invalid_symbols_cache(self) -> Dict[str, float]:
        """Loads the map of invalid symbols and their discovery timestamp."""
        path = self._get_invalid_symbols_cache_path()
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logging.warning(f"Error loading invalid symbols cache: {e}")
            return {}

    def _save_invalid_symbols_cache(self, invalid_map: Dict[str, float]):
        """Saves the map of invalid symbols."""
        path = self._get_invalid_symbols_cache_path()
        try:
            with open(path, "w") as f:
                json.dump(invalid_map, f)
        except OSError as e:
            logging.warning(f"Error saving invalid symbols cache: {e}")

    @profile
    def _fetch_yf_historical_data(
        self, symbols_yf: List[str], start_date: date, end_date: date
    ) -> Dict[str, pd.DataFrame]:
        """
        Internal helper to fetch historical 'Close' data (adjusted) using yfinance.download.
        (Replaces fetch_yf_historical)
        """
        _ensure_yfinance()
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

        # Ensure start_date and end_date are date objects for downstream comparisons
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()

        # --- OPTIMIZATION: Filter out known invalid symbols ---
        invalid_cache = self._load_invalid_symbols_cache()
        now_ts = time.time()
        symbols_to_fetch = []
        filtered_count = 0
        cache_needs_update = False
        
        for s in symbols_yf:
            if s in invalid_cache:
                timestamp = invalid_cache[s]
                if now_ts - timestamp < INVALID_SYMBOLS_DURATION:
                    filtered_count += 1
                    continue  # Skip this symbol
                else:
                    # Expired, retry fetching
                    del invalid_cache[s]
                    cache_needs_update = True
            symbols_to_fetch.append(s)

        if cache_needs_update:
            self._save_invalid_symbols_cache(invalid_cache)

        if filtered_count > 0:
            logging.info(f"Hist Fetch Helper: Skipped {filtered_count} known invalid/delisted symbols (cache active).")
        
        symbols_yf = symbols_to_fetch # Update the list to process
        # --- END OPTIMIZATION ---

        yf_end_date = max(start_date, end_date) + timedelta(days=1)
        yf_start_date = min(start_date, end_date)

        # Ensure we are working with date objects for comparison
        if isinstance(yf_start_date, pd.Timestamp):
            yf_start_date = yf_start_date.date()
        if isinstance(yf_end_date, pd.Timestamp):
            yf_end_date = yf_end_date.date()

        # --- FIX: Prevent fetching future dates which causes YFPricesMissingError ---
        # If the requested start date is today or in the future, yfinance might error 
        # because the "day" hasn't happened yet in the market's timezone.
        # Historical data implies "past" data. Real-time is handled elsewhere.
        today = date.today()
        if yf_start_date >= today:
             logging.info(f"Hist Fetch Helper: Request start date {yf_start_date} is >= today ({today}). Skipping historical fetch to avoid yfinance errors.")
             return {}
        
        # Clamp end date to today (exclusive in yfinance means up to yesterday) 
        # if we are not asking for today's data (which we shouldn't be here).
        # Actually, if we ask for end=tomorrow, yfinance tries to get today.
        # If today isn't effectively over or started, it might be an issue.
        # Safer to clamp end to 'today' (fetching UP TO yesterday) for strict history.
        if yf_end_date > today:
             yf_end_date = today
        # --- END FIX ---
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
                                     invalid_cache[symbol] = now_ts
                                     cache_needs_update = True
                             else:
                                 logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned no data in range.")
                                 invalid_cache[symbol] = now_ts
                                 cache_needs_update = True
                        else:
                             logging.warning(f"  Hist Fetch Helper WARN: Individual retry for {symbol} returned empty DataFrame.")
                             invalid_cache[symbol] = now_ts
                             cache_needs_update = True
                             
                    except Exception as e_retry:
                        logging.warning(f"  Hist Fetch Helper WARN: Individual retry failed for {symbol}: {e_retry}")
                        invalid_cache[symbol] = now_ts
                        cache_needs_update = True

        if cache_needs_update:
            self._save_invalid_symbols_cache(invalid_cache)

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
        cache_key: Optional[str] = None,  # Kept for API compatibility, but logic relies on individual files
        cache_file: Optional[str] = None,  # Not directly used
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches ADJUSTED historical price data using cache.

        Refactored to check individual symbol files regardless of a global cache key.
        This maximizes cache reuse (e.g. if adding 1 new symbol to a list of 10 already cached).

        Args:
            symbols_yf (List[str]): List of YF stock/benchmark tickers required.
            start_date (date): Start date for data (inclusive).
            end_date (date): End date for data (inclusive/exclusive depending on usage, but typically inclusive).
            use_cache (bool, optional): Whether to attempt loading from cache. Defaults to True.
            cache_key (Optional[str], optional): Legacy cache key. Defaults to None.
            cache_file (Optional[str], optional): Legacy cache file. Defaults to None.

        Returns:
            Tuple[Dict[str, pd.DataFrame], bool]:
                - historical_data (Dict): Map of yf_symbol -> DataFrame.
                - all_fresh (bool): True if all requested data was retrieved successfully.
        """
        # Ensure start_date and end_date are date objects
        if isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()
            
        logging.info(
            f"Hist Prices: Fetching historical data for {len(symbols_yf)} symbols ({start_date} to {end_date})..."
        )


        historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        fetch_failed = False
        
        symbols_needing_full_fetch = []
        symbols_needing_incremental_fetch = []

        # --- 1. Try Loading Individual Cache Files ---
        if use_cache:
            logging.info(f"Hist Prices: Checking local cache for {len(symbols_yf)} symbols...")
            for yf_symbol in symbols_yf:
                symbol_file_path = self._get_historical_symbol_data_path(yf_symbol, data_type="price")
                
                # Check if file exists
                if os.path.exists(symbol_file_path):
                    try:
                        # Load data
                        with open(symbol_file_path, "r", encoding="utf-8") as sf:
                            symbol_data_json_str = sf.read()
                        
                        df_symbol = self._deserialize_single_historical_df(symbol_data_json_str)
                        
                        if not df_symbol.empty:
                            # Check data coverage
                            first_avail = df_symbol.index.min()
                            last_avail = df_symbol.index.max()
                            if hasattr(first_avail, 'date'): first_avail = first_avail.date()
                            if hasattr(last_avail, 'date'): last_avail = last_avail.date()
                            
                            # Determine if we have enough coverage
                            # We generally care more about the END date coverage (up to now).
                            # If start date is earlier than cache, we technically need to backfill, 
                            # but YF doesn't support easy "prepend" without potentially refetching everything or complex merging.
                            # For simplified logic: if we have data up to (end_date - 1 day) or later, we consider it "fresh enough" for that end.
                            # Or strictly: fetch incremental if last_avail < end_date.

                            # Check for staleness at the HEAD (new data needed)
                            if last_avail < end_date:
                                # We have data, but it's old. Use what we have, but mark for update.
                                historical_prices_yf_adjusted[yf_symbol] = df_symbol
                                symbols_needing_incremental_fetch.append((yf_symbol, last_avail))
                            else:
                                # Cache is good for the recent side.
                                # Optional: Check start date. If we need 2020 but cache starts 2021, we might need full fetch/prepend.
                                # For simplicity in this optimization phase, we assume historical start dates rarely retreat significantly 
                                # or that the user accepts the cached start. 
                                # Use full fetch if gap is huge? Let's assume re-fetch if start_date is significantly earlier (< 30 days gap ok?)
                                # Actually, safest is: if start_date < first_avail, we basically need the older history. 
                                # yfinance is best at "fetch X to Y". Merging 'older' data is tricky with splits. 
                                # Strategy: If we need older data, just re-fetch full to be safe on splits/adjustments.
                                if start_date < first_avail:
                                    # logging.info(f"  Gap at start for {yf_symbol}: Need {start_date}, have {first_avail}. Marking for full fetch.")
                                    # Don't add to historical_prices_yf_adjusted, so it goes to full fetch
                                    symbols_needing_full_fetch.append(yf_symbol)
                                else:
                                    historical_prices_yf_adjusted[yf_symbol] = df_symbol
                        else:
                             # File empty?
                             symbols_needing_full_fetch.append(yf_symbol)
                    except Exception as e_load:
                        logging.warning(f"  Error loading cache for {yf_symbol}: {e_load}. Will re-fetch.")
                        symbols_needing_full_fetch.append(yf_symbol)
                else:
                    # File missing
                    symbols_needing_full_fetch.append(yf_symbol)
        else:
            symbols_needing_full_fetch = list(symbols_yf)

        # --- 2. Perform Fetches ---
        
        # 2a. Incremental Fetch (Append new days)
        if symbols_needing_incremental_fetch:
            logging.info(f"Hist Prices: Incremental fetch needed for {len(symbols_needing_incremental_fetch)} symbols...")
            from collections import defaultdict
            by_last_date = defaultdict(list)
            for s, last_d in symbols_needing_incremental_fetch:
                by_last_date[last_d].append(s)
            
            for last_d, syms in by_last_date.items():
                fetch_start = last_d + timedelta(days=1)
                if fetch_start <= end_date:
                    delta_data = self._fetch_yf_historical_data(syms, fetch_start, end_date)
                    
                    # Merge
                    for s in syms:
                        if s in delta_data and not delta_data[s].empty:
                            old_df = historical_prices_yf_adjusted[s]
                            new_df = delta_data[s]
                            # Ensure index types match
                            if not isinstance(old_df.index, pd.DatetimeIndex): old_df.index = pd.to_datetime(old_df.index)
                            if not isinstance(new_df.index, pd.DatetimeIndex): new_df.index = pd.to_datetime(new_df.index)
                                
                            merged_df = pd.concat([old_df, new_df])
                            merged_df = merged_df[~merged_df.index.duplicated(keep='last')]
                            merged_df.sort_index(inplace=True)
                            historical_prices_yf_adjusted[s] = merged_df
        
        # 2b. Full Fetch (Missing or needing earlier start)
        # Also clean up list to assume any symbol not already in 'historical_prices_yf_adjusted' needs full fetch
        # (Though our logic above populated 'symbols_needing_full_fetch' explicitly)
        if symbols_needing_full_fetch:
             logging.info(f"Hist Prices: Full/Gap fetch needed for {len(symbols_needing_full_fetch)} symbols...")
             fetched_data = self._fetch_yf_historical_data(symbols_needing_full_fetch, start_date, end_date)
             historical_prices_yf_adjusted.update(fetched_data)
        
        logging.info(f"Hist Prices: Data available for {len(historical_prices_yf_adjusted)}/{len(symbols_yf)} symbols.")

        # --- 3. Save Updates to Cache ---
        if use_cache:
            # We save everything we have in memory for the requested symbols.
            # This refreshes the timestamp on the manifest and updates files.
            # Only save if we actually fetched new data or merged data - 
            # ideally we check dirty flags, but for now safe to save.
            # actually, writing to disk every time might be slow? 
            # _save_historical_data_and_manifest writes ALL passed data to disk.
            # Optimization: Only save modified data?
            # Current structure expects us to pass the map.
            # Let's pass it all, OS file caching helps. Or we could optimize _save... later.
            
            # Map for manifest update (legacy)
            if cache_key:
                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=historical_prices_yf_adjusted,
                    data_type="price"
                )
            else:
                 # Even without a key, we should save the individual files!
                 # The _save_... method updates manifest. If no key, maybe skip manifest but save files?
                 # Depending on implementation. But simpler: reuse logic with a dummy key or skip manifest if None.
                 # Let's iterate and save files directly if strictly no key, but usually cache_key is provided by caller.
                 if not cache_key:
                     # Just save files manually to ensure persistence
                     self._save_files_only(historical_prices_yf_adjusted, "price")

        # --- 4. Final Validation ---
        invalid_cache = self._load_invalid_symbols_cache()
        missing_valid_symbols = []
        for s in symbols_yf:
            if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty:
                if s not in invalid_cache:
                    missing_valid_symbols.append(s)
        
        if missing_valid_symbols:
            logging.error(f"Hist Prices ERROR: Missing data for: {', '.join(missing_valid_symbols)}")
            fetch_failed = True

        return historical_prices_yf_adjusted, fetch_failed

    def _save_files_only(self, data_map, data_type):
        """Helper to save just file content without manifest update (if needed)."""
        for sym, df in data_map.items():
             if df.empty: continue
             path = self._get_historical_symbol_data_path(sym, data_type)
             try:
                 df.to_json(path, orient="split", date_format="iso")
             except Exception as e:
                 logging.error(f"Error saving file for {sym}: {e}")

    @profile
    def get_historical_fx_rates(
        self,
        fx_pairs_yf: List[str],  # e.g., ['EUR=X', 'JPY=X']
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        cache_key: Optional[str] = None,  # Key for validation (used for manifest)
        cache_file: Optional[str] = None,  # Not directly used
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches historical FX rates (vs USD) using cache.
        
        Refactored to check individual symbol files regardless of a global cache key.
        """
        historical_fx_yf: Dict[str, pd.DataFrame] = {}
        fetch_failed = False
        
        fx_needing_full_fetch = []
        fx_needing_incremental_fetch = []

        # --- 1. Load from cache files ---
        if use_cache:
            logging.info(f"Hist FX: Checking local cache for {len(fx_pairs_yf)} pairs...")
            for pair in fx_pairs_yf:
                symbol_file_path = self._get_historical_symbol_data_path(pair, data_type="fx")
                
                if os.path.exists(symbol_file_path):
                    try:
                        with open(symbol_file_path, "r", encoding="utf-8") as sf:
                            symbol_data_json_str = sf.read()
                        
                        df_idx = self._deserialize_single_historical_df(symbol_data_json_str)
                        
                        if not df_idx.empty:
                            first_avail = df_idx.index.min()
                            last_avail = df_idx.index.max()
                            
                            # Ensure validation uses date vs date
                            if isinstance(first_avail, pd.Timestamp):
                                first_avail = first_avail.date()
                            if isinstance(last_avail, pd.Timestamp):
                                last_avail = last_avail.date()
                            
                            
                            if last_avail < end_date:
                                historical_fx_yf[pair] = df_idx
                                fx_needing_incremental_fetch.append((pair, last_avail))
                            else:
                                if start_date < first_avail:
                                    # Need older data -> full fetch
                                    # MODIFIED: Keep the partial cache as fallback in case fetch fails
                                    historical_fx_yf[pair] = df_idx
                                    logging.info(f"  Hist FX: Found partial cache for {pair} (starts {first_avail}, need {start_date}). Will try full fetch, but keeping cache as fallback.")
                                    fx_needing_full_fetch.append(pair)
                                else:
                                    historical_fx_yf[pair] = df_idx
                        else:
                             fx_needing_full_fetch.append(pair)
                    except Exception as e:
                        logging.warning(f"Error loading FX cache for {pair}: {e}")
                        fx_needing_full_fetch.append(pair)
                else:
                    fx_needing_full_fetch.append(pair)
        else:
            fx_needing_full_fetch = list(fx_pairs_yf)

        # --- 2. Fetches ---
        
        # 2a. Incremental
        if fx_needing_incremental_fetch:
            # Group by date
            from collections import defaultdict
            by_last_date = defaultdict(list)
            for s, last_d in fx_needing_incremental_fetch:
                by_last_date[last_d].append(s)
            
            for last_d, syms in by_last_date.items():
                fetch_start = last_d + timedelta(days=1)
                if fetch_start <= end_date:
                    delta_data = self._fetch_yf_historical_data(syms, fetch_start, end_date)
                    for s in syms:
                        if s in delta_data and not delta_data[s].empty:
                            old_df = historical_fx_yf[s]
                            new_df = delta_data[s]
                            if not isinstance(old_df.index, pd.DatetimeIndex): old_df.index = pd.to_datetime(old_df.index)
                            if not isinstance(new_df.index, pd.DatetimeIndex): new_df.index = pd.to_datetime(new_df.index)
                            merged = pd.concat([old_df, new_df])
                            merged = merged[~merged.index.duplicated(keep='last')]
                            merged.sort_index(inplace=True)
                            historical_fx_yf[s] = merged

        # 2b. Full
        if fx_needing_full_fetch:
             logging.info(f"Hist FX: Full fetch needed for {len(fx_needing_full_fetch)} pairs...")
             fetched_fx = self._fetch_yf_historical_data(fx_needing_full_fetch, start_date, end_date)
             historical_fx_yf.update(fetched_fx)
             
        # --- 3. Save ---
        if use_cache:
            if cache_key:
                data_to_save = {k: v for k, v in historical_fx_yf.items() if k in fx_pairs_yf}
                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=data_to_save,
                    data_type="fx"
                )
            else:
                 # Check if we should save manually without key
                 data_to_save = {k: v for k, v in historical_fx_yf.items() if k in fx_pairs_yf}
                 self._save_files_only(data_to_save, "fx")

        # --- 4. Validation ---
        missing_fx = [p for p in fx_pairs_yf if p not in historical_fx_yf or historical_fx_yf[p].empty]
        if missing_fx:
             logging.error(f"Hist FX ERROR: Critical FX data missing for: {', '.join(missing_fx)}")
             fetch_failed = True

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
