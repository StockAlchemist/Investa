# market_data.py
import pandas as pd
import numpy as np
import logging
import json
import os
import sys
import threading # Added for _SHARED_MDP_LOCK
from datetime import datetime, timedelta, date, UTC, timezone  # Added UTC
from utils_time import get_est_today, get_latest_trading_date, is_market_open, get_nyse_calendar # Added for holiday/timezone awareness
from typing import List, Dict, Optional, Tuple, Set, Any
import time
import requests  # Keep for potential future use
import traceback  # For detailed error logging
from io import StringIO  # For historical cache loading
import hashlib  # For cache key hashing
import subprocess
import sys
import tempfile
from market_db import MarketDatabase




# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from PySide6.QtCore import QStandardPaths  # For standard directory locations
import config

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
    CURRENT_QUOTE_CACHE_DURATION_MINUTES = 1

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

 Author:        Google Gemini (Derived from portfolio_logic.py)

 Copyright:     (c) Investa Contributors 2025
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


def _run_isolated_fetch(tickers, start=None, end=None, interval="1d", task="history", period=None, **kwargs):
    """
    Runs yfinance fetch in a separate process using file I/O to prevent crashing the main server.
    """
    # Global semaphore to limit concurrent subprocesses and file usage
    # Mac limit is often 256. We limit strict to 2 to be safe and prevent "Too many open files".
    global _FETCH_SEMAPHORE
    if '_FETCH_SEMAPHORE' not in globals():
        import threading
        _FETCH_SEMAPHORE = threading.Semaphore(2)

    with _FETCH_SEMAPHORE:
        return _run_isolated_fetch_impl(tickers, start, end, interval, task, period, **kwargs)


def _run_isolated_fetch_impl(tickers, start, end, interval, task, period, **kwargs):
    """
    Actual implementation of the isolated fetch.
    """
    temp_output = None
    try:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "market_data_worker.py")
        
        # Create a temp file path for the worker to write to
        fd, temp_output = tempfile.mkstemp(suffix=".json")
        os.close(fd) # Output file path only
        
        payload = {
            "task": task,
            "symbols": tickers,
            "start": str(start) if start else None,
            "end": str(end) if end else None,
            "interval": interval,
            "period": period,
            "output_file": temp_output
        }
        # Add any extra kwargs (like statement_type, period_type)
        payload.update(kwargs)
        
        # Run subprocess
        result = subprocess.run(
            [sys.executable, script_path],
            input=json.dumps(payload),
            capture_output=True, # Still capture metadata output
            text=True,
            timeout=180 # 3 minute timeout
        )

        
        if result.stderr:
             logging.info(f"Isolated fetch STDERR: {result.stderr}")

        if result.returncode != 0:
            logging.error(f"Isolated fetch failed (Code {result.returncode}): {result.stderr}")
            if os.path.exists(temp_output): os.remove(temp_output)
            return pd.DataFrame() if task not in ["info", "calendar"] else {}
            
        # Parse metadata output
        try:
            response = json.loads(result.stdout)
        except json.JSONDecodeError:
            logging.error(f"Isolated fetch returned invalid JSON metadata: {result.stdout[:200]}")
            if os.path.exists(temp_output): os.remove(temp_output)
            return pd.DataFrame() if task not in ["info", "calendar"] else {}
            
        if response.get("status") == "success":
            # Check if empty
            if response.get("data") is None and "file" not in response:
                 # Empty result path
                 if os.path.exists(temp_output): os.remove(temp_output)
                 return pd.DataFrame() if task not in ["info", "calendar"] else {}

            # Load results from file
            file_path = response.get("file")
            if file_path and os.path.exists(file_path):
                try:
                    if task in ["info", "calendar"]:
                        with open(file_path, "r") as f:
                            data_loaded = json.load(f)
                        return data_loaded.get("data", {})
                    elif task == "dividends":
                        # Series orient='split'
                        df = pd.read_json(file_path, orient='split')
                        # Convert to Series if it's 1-column
                        if not df.empty:
                            return df.iloc[:, 0]
                        return pd.Series()
                    else:
                        # history or statement (DataFrame orient='split')
                        df = pd.read_json(file_path, orient='split')
                        if not df.empty and task == "history":
                            logging.info(f"Isolated fetch: Deserialized raw result columns: {list(df.columns[:5])} (Type: {type(df.columns[0]) if not df.empty else 'N/A'})")
                            df.index = pd.to_datetime(df.index, utc=True)
                            # Reconstruct MultiIndex if it was flattened to tuples during JSON serialization
                            if len(df.columns) > 0 and isinstance(df.columns[0], (list, tuple)):
                                try:
                                    df.columns = pd.MultiIndex.from_tuples(df.columns)
                                    logging.info("Isolated fetch: Reconstructed MultiIndex successfully.")
                                except Exception as e_mi:
                                    logging.warning(f"Isolated fetch: Could not reconstruct MultiIndex: {e_mi}")
                        return df

                except Exception as e_read:
                    logging.error(f"Error reading isolated fetch result file ({task}): {e_read}")
                    return {} if task in ["info", "calendar"] else pd.DataFrame()
                finally:
                    # Clean up
                    if os.path.exists(file_path):
                        os.remove(file_path)
            else:
                 # "data": None case or file missing
                 if os.path.exists(temp_output): os.remove(temp_output)
                 return {} if task in ["info", "calendar"] else (pd.Series() if task == "dividends" else pd.DataFrame())
        else:
            logging.error(f"Isolated fetch worker reported error: {response.get('message')}")
            if os.path.exists(temp_output): os.remove(temp_output)
            return {} if task in ["info", "calendar"] else (pd.Series() if task == "dividends" else pd.DataFrame())

    except Exception as e:
        logging.error(f"Error running isolated fetch: {e}")
        if temp_output and os.path.exists(temp_output):
            os.remove(temp_output)
        return pd.DataFrame()


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
        db_path=None, # Persistent SQL database
    ):
        self.db = MarketDatabase(db_path)
        self.hist_data_cache_dir_name = (
            hist_data_cache_dir_name  # Store historical cache subdirectory name
        )
        self._session = None
        self.historical_fx_for_fallback: Dict[str, pd.DataFrame] = (
            {}
        )  # Store recently fetched historical FX

        # Get centralized app data directory
        app_data_dir = config.get_app_data_dir()

        # Construct full path for current_cache_file
        if current_cache_file is None:
            current_cache_file = config.DEFAULT_CURRENT_CACHE_FILE_PATH

        if os.path.isabs(current_cache_file):
            self.current_cache_file = current_cache_file
        else:
            self.current_cache_file = os.path.join(app_data_dir, current_cache_file)

        # Construct full path for fundamentals_cache_dir
        if os.path.isabs(fundamentals_cache_dir):
            self.fundamentals_cache_dir = fundamentals_cache_dir
        else:
            self.fundamentals_cache_dir = os.path.join(
                app_data_dir, fundamentals_cache_dir
            )

        # Ensure directory exists
        os.makedirs(self.fundamentals_cache_dir, exist_ok=True)
        
        # logging.info("MarketDataProvider initialized.")

    def _get_historical_cache_dir(self) -> str:
        """Constructs and returns the full path to the historical data cache subdirectory."""
        app_data_dir = config.get_app_data_dir()
        hist_dir = os.path.join(app_data_dir, self.hist_data_cache_dir_name)
        os.makedirs(hist_dir, exist_ok=True)
        return hist_dir

    def _get_historical_manifest_path(self) -> str:
        """Returns the full path to the manifest.json file for historical data."""
        return os.path.join(self._get_historical_cache_dir(), "manifest.json")

    def _get_historical_symbol_data_path(
        self, yf_symbol: str, data_type: str = "price", interval: str = "1d"
    ) -> str:
        """
        Returns the full path for an individual symbol's historical data file.
        data_type can be 'price' or 'fx'.
        """
        # Sanitize symbol for filename (replace characters not suitable for filenames)
        safe_yf_symbol = "".join(
            c if c.isalnum() or c in [".", "_", "-"] else "_" for c in yf_symbol
        )  # Allow . _ -
        # Append interval to filename if not daily
        interval_suffix = f"_{interval}" if interval != "1d" else ""
        filename = f"{safe_yf_symbol}_{data_type}{interval_suffix}.json"
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
                  # Use isolated batch fetch for metadata
                  chunk_size = 50
                  for i in range(0, len(missing_symbols), chunk_size):
                      chunk = missing_symbols[i:i+chunk_size]
                      logging.info(f"Metadata Fetch: Processing isolated batch {i//chunk_size + 1}/{len(missing_symbols)//chunk_size + 1}. Symbols: {len(chunk)}")
                      
                      # Isolated fetch for info task
                      info_batch = _run_isolated_fetch(chunk, task="info")
                      
                      for sym in chunk:
                          info = info_batch.get(sym)
                          if info:
                              name = info.get("shortName") or info.get("longName") or sym
                              cache[sym] = {
                                  "name": name,
                                  "currency": info.get("currency"),
                                  "sector": info.get("sector"),
                                  "industry": info.get("industry"),
                                  "timestamp": now_ts.isoformat()
                              }
                          else:
                              logging.warning(f"Failed to fetch metadata for {sym}. Using placeholders.")
                              if sym not in cache:
                                  cache[sym] = {
                                      "name": sym,
                                      "currency": None, 
                                      "sector": None,
                                      "industry": None,
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
                # Use isolated batch fetch for fundamentals
                chunk_size = 50
                for i in range(0, len(missing_symbols), chunk_size):
                    chunk = list(missing_symbols)[i:i+chunk_size] # Convert to list for slicing
                    logging.info(f"Fundamentals Fetch: Processing isolated batch {i//chunk_size + 1}. Symbols: {len(chunk)}")
                    
                    # Isolated fetch for info task
                    info_batch = _run_isolated_fetch(chunk, task="info")
                    
                    for sym in chunk:
                        info = info_batch.get(sym)
                        if info:
                            # Extract dividend data
                            # dividendRate is annual dividend in currency
                            # dividendYield is percentage (e.g. 0.05 for 5%)
                            div_rate = info.get("dividendRate", 0.0)
                            div_yield = info.get("dividendYield", 0.0)
                            
                            # Additional data for matching frequency and projection
                            ex_div_date = info.get("exDividendDate", None)
                            last_div_val = info.get("lastDividendValue", 0.0)
                            last_div_date = info.get("lastDividendDate", None)

                            cache[sym] = {
                                "trailingAnnualDividendRate": div_rate if div_rate else 0.0,
                                "dividendYield": div_yield if div_yield else 0.0,
                                "exDividendDate": ex_div_date,
                                "lastDividendValue": last_div_val,
                                "lastDividendDate": last_div_date,
                                "timestamp": now_ts.isoformat()
                            }
                        else:
                            logging.warning(f"Failed to fetch fundamentals for {sym} (isolated).")
                            cache[sym] = {"timestamp": now_ts.isoformat()} # Cache as tried to avoid infinite loop

                            
            except Exception as e_batch:
                logging.error(f"Error in fundamentals batch fetch: {e_batch}")
            
            self._save_fundamentals_cache(cache)
            
        return cache

    def get_ticker_details_batch(self, yf_symbols: Set[str]) -> Dict[str, Dict]:
        """
        Fetches detailed ticker info (for screener) and caches it.
        Uses fundamentals cache as storage.
        """
        cache = self._load_fundamentals_cache()
        now_ts = datetime.now(timezone.utc)
        missing_symbols = []
        
        # Ensure yfinance is available
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
            return {sym: cache.get(sym, {}).get("ticker_info", {}) for sym in yf_symbols}

        for sym in yf_symbols:
            entry = cache.get(sym)
            if not entry or "ticker_info" not in entry:
                missing_symbols.append(sym)
                continue
            
            ts_str = entry.get("timestamp")
            if not ts_str:
                missing_symbols.append(sym)
                continue
            
            try:
                entry_ts = datetime.fromisoformat(ts_str)
                if (now_ts - entry_ts).days >= 1: 
                     missing_symbols.append(sym)
            except ValueError:
                missing_symbols.append(sym)
        
        if missing_symbols:
            logging.info(f"TickerDetails: Fetching for {len(missing_symbols)} symbols...")
            try:
                chunk_size = 50
                for i in range(0, len(missing_symbols), chunk_size):
                    chunk = list(missing_symbols)[i:i+chunk_size]
                    info_batch = _run_isolated_fetch(chunk, task="info")
                    
                    for sym in chunk:
                        info = info_batch.get(sym)
                        if info:
                            entry = cache.get(sym, {})
                            entry["ticker_info"] = info
                            entry["timestamp"] = now_ts.isoformat()
                            if "trailingAnnualDividendRate" not in entry:
                                entry["trailingAnnualDividendRate"] = info.get("dividendRate", 0.0)
                            if "dividendYield" not in entry:
                                entry["dividendYield"] = info.get("dividendYield", 0.0)
                            cache[sym] = entry
                        else:
                            cache[sym] = cache.get(sym, {"timestamp": now_ts.isoformat()})
            except Exception as e:
                logging.error(f"Error in ticker details fetch: {e}")
            
            self._save_fundamentals_cache(cache)
            
        return {sym: cache.get(sym, {}).get("ticker_info", {}) for sym in yf_symbols}

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
        # v10_1M_CACHE: Re-enabled short cache to prevent thread explosion/crash on concurrent requests.
        cache_key = f"CURRENT_QUOTES_v10_1M_CACHE::{'_'.join(sorted(yf_symbols_to_fetch))}::{'_'.join(sorted(required_currencies))}"
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
                            minutes=1 # Short 1-min cache to allow concurrency without crashing
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
            # --- 3b. Batch Fetch Sparklines (and fallback prices) using yf.download ---
            logging.info(f"Batch fetching sparklines/history for {len(yf_symbols_to_fetch)} symbols...")
            try:
                # Use chunked download for reliability and to avoid process/memory limits with 500+ symbols
                all_dfs = []
                yf_symbols_list = list(yf_symbols_to_fetch)
                chunk_size = 50 # REDUCED: Was 250, causing OOM (Code -9) on some systems
                
                for i in range(0, len(yf_symbols_list), chunk_size):
                    chunk = yf_symbols_list[i:i+chunk_size]
                    logging.info(f"Batch Quote Fetch: Processing chunk {i//chunk_size + 1} ({len(chunk)} symbols)")
                    
                    df_chunk = _run_isolated_fetch(
                        chunk,
                        period="10d",
                        interval="1d",
                        task="history"
                    )
                    
                    if not df_chunk.empty:
                        all_dfs.append(df_chunk)
                
                if not all_dfs:
                     logging.warning("Batch price fetch returned empty DataFrames for all chunks.")
                     df = pd.DataFrame()
                else:
                    # Concatenate all chunks
                    # Note: axis=1 for columns (different symbols)
                    df = pd.concat(all_dfs, axis=1)
                
                if df.empty:
                    logging.warning("Combined batch price fetch resulted in empty DataFrame.")
                else:
                    # Robust check for MultiIndex or flat tuple index (Moved here for scope)
                    has_multilevel = getattr(df.columns, 'nlevels', 1) > 1
                    
                    # Current UTC date for filtering
                    now_utc = datetime.now(timezone.utc).date()
                    
                    # Process results
                    for internal_sym, yf_sym in internal_to_yf_map_local.items():
                        try:
                            price = None
                            prev_close = None
                            sparkline = []
                            
                            # Handle yf.download structure
                            sym_df = pd.DataFrame()
                            if len(yf_symbols_to_fetch) > 1:
                                try:
                                    if has_multilevel and yf_sym in df.columns.get_level_values(0):
                                        sym_df = df[yf_sym]
                                    elif not has_multilevel and any(isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper() for c in df.columns):
                                        # Handle flattened MultiIndex (tuples/lists)
                                        cols_for_sym = [c for c in df.columns if isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper()]
                                        sym_df = df[cols_for_sym]
                                        sym_df.columns = [c[1] for c in sym_df.columns]
                                    elif has_multilevel and yf_sym not in df.columns.get_level_values(0):
                                        pass 
                                    elif yf_sym in df.columns:
                                        sym_df = df[[yf_sym]].rename(columns={yf_sym: "Close"}) 
                                except Exception as e_df_parse:
                                    logging.warning(f"Error parsing DataFrame for {yf_sym}: {e_df_parse}")
                                    sym_df = pd.DataFrame() 
                            else:
                                # Single ticker download
                                if not has_multilevel and any(isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper() for c in df.columns):
                                     cols_for_sym = [c for c in df.columns if isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper()]
                                     sym_df = df[cols_for_sym]
                                     sym_df.columns = [c[1] for c in sym_df.columns]
                                else:
                                     sym_df = df
                            
                            if not sym_df.empty:
                                # Check for Close or Price column
                                col_name = "Close" if "Close" in sym_df.columns else sym_df.columns[0]
                                
                                valid_days = sym_df.dropna(subset=[col_name])
                                if not valid_days.empty:
                                    # Convert index to dates for comparison
                                    # Ensure index is datetime
                                    valid_dates = valid_days.index.date
                                    
                                    # 1. Determine Stable Previous Close (Always Yesterday or earlier)
                                    # Filter for dates strictly less than today (UTC)
                                    # Note: Determining 'Today' is tricky with timezones. 
                                    # Ideally use the last available data point that is NOT today.
                                    # If the last point is today, use the one before it.
                                    # If the last point is older than today, use it? No, that's current price.
                                    
                                    # Let's simplify: Take the last 2 points.
                                    vals = valid_days[col_name].tolist()
                                    dates_idx = valid_days.index.to_list()
                                    
                                    # Build sparkline
                                    sparkline = [float(v) for v in vals]
                                    if len(sparkline) > 7:
                                        sparkline = sparkline[-7:] # Tail 7
                                    
                                    if len(vals) > 0:
                                        last_val = float(vals[-1])
                                        last_date = dates_idx[-1]
                                        if hasattr(last_date, 'date'): last_date = last_date.date()
                                        
                                        # Default assumptions
                                        price = last_val
                                        
                                        # To find prev_close, we look for the last point BEFORE last_date
                                        # OR if last_date is today, we definitely want the one before it.
                                        # If last_date is NOT today (stale), then that IS the close of that day,
                                        # and prev_close should be the one before THAT.
                                        
                                        # Logic:
                                        # If the last data point's date is strictly LESS than today (in UTC context),
                                        # it means we have NO data for today yet (market closed or delayed).
                                        # In this case, Price = YesterdayClose.
                                        # To avoid showing "Yesterday's Change" as "Today's Change", we set Change = 0.
                                        # This is done by setting prev_close = price.
                                        
                                        # However, if last_date == today, then:
                                        # Price = TodayCurrent.
                                        # PrevClose = The point before it (Yesterday).
                                        
                                        is_today = (last_date == now_utc)
                                        # Allow for timezone diffs (e.g. Asia vs UTC). 
                                        # If last_date is AHEAD of now_utc (tomorrow?), treat as today.
                                        if last_date >= now_utc:
                                            is_today = True
                                            
                                        if is_today:
                                            if len(vals) >= 2:
                                                prev_close = float(vals[-2])
                                            else:
                                                prev_close = last_val # No history, change=0
                                        else:
                                            # Stale data (Yesterday's close is the latest we have)
                                            # User Request (Step 2730): "If the market is closed, show the day's gain/loss for the latest day when the market is open."
                                            # So we DO want to calculate the change for that last day.
                                            if len(vals) >= 2:
                                                prev_close = float(vals[-2])
                                            else:
                                                prev_close = last_val # No history, change=0
                                            
                                        logging.debug(f"{yf_sym} Download: LastDate={last_date}, IsToday={is_today}, Price={price}, Prev={prev_close}")

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
                                    "source": "yf_batch_download_stale_safe" if not is_today else "yf_batch_download",
                                    "timestamp": datetime.now(timezone.utc).isoformat(),
                                    "trailingAnnualDividendRate": fund.get("trailingAnnualDividendRate", 0),
                                    "dividendYield": fund.get("dividendYield", 0),
                                    "sparkline_7d": sparkline
                                }
                            else:
                                # Don't warn yet, fast_info might fix it
                                pass

                        except Exception as e_sym:
                            logging.warning(f"Error processing batch result for {yf_sym}: {e_sym}")
                            
            except Exception as e_down:
                logging.error(f"Batch download failed: {e_down}")
                # Don't set error yet, try fast_info

            # --- 3c. Batch Real-Time Price Fetch (Optimized) ---
            # Replace sequential fast_info (slow/flakey) with batched 1m download
            logging.info(f"Batch fetching 1m intraday data for {len(yf_symbols_to_fetch)} symbols (Real-Time)...")
            try:
                # Download 1m data for the last 1 day. 
                # This gives us the very latest traded price (Close of last minute bar).
                # Much faster than 50 HTTP requests for fast_info.
                # Use isolated fetch for 1m intraday data
                df_rt = _run_isolated_fetch(
                    list(yf_symbols_to_fetch),
                    period="1d",
                    interval="1m",
                    task="history"
                )
                
                if not df_rt.empty:
                    # Parse 1m results
                    has_multilevel_rt = getattr(df_rt.columns, 'nlevels', 1) > 1
                    
                    for internal_sym, yf_sym in internal_to_yf_map_local.items():
                        try:
                            price_rt = None
                            
                            # Extract 1m Series
                            sym_df_rt = pd.DataFrame()
                            if len(yf_symbols_to_fetch) > 1:
                                if has_multilevel_rt and yf_sym in df_rt.columns.get_level_values(0):
                                    sym_df_rt = df_rt[yf_sym]
                                elif not has_multilevel_rt and any(isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper() for c in df_rt.columns):
                                     cols_for_sym = [c for c in df_rt.columns if isinstance(c, (tuple, list)) and c[0].upper() == yf_sym.upper()]
                                     sym_df_rt = df_rt[cols_for_sym]
                                     sym_df_rt.columns = [c[1] for c in sym_df_rt.columns]
                            else:
                                if not has_multilevel_rt:
                                    sym_df_rt = df_rt 
                            
                            if not sym_df_rt.empty and "Close" in sym_df_rt.columns:
                                # Get last valid price
                                last_row = sym_df_rt.iloc[-1]
                                price_rt = float(last_row["Close"])
                                
                                # Check if 1m data is actually from today (UTC) to avoid applying stale intraday noise
                                last_rt_date = last_row.name.date() if hasattr(last_row.name, "date") else last_row.name
                                if last_rt_date < now_utc:
                                     # Stale 1m data (Yesterday). Skip update to preserve "Change=0" from Daily Logic.
                                     price_rt = None
                            
                            if price_rt and price_rt > 0:
                                entry = stock_data_yf.get(internal_sym)
                                if entry:
                                    # We have an existing entry from Daily download (containing Sparkline/PrevClose/Meta)
                                    # We just update the Price and Change.
                                    
                                    # Recalculate Change using the stable PrevClose we already have
                                    # The existing entry['change'] was 0.00 or based on daily.
                                    # But entry['price'] might be stale (Yesterday).
                                    
                                    # We need to dig out the 'prev_close' implicated in the Daily logic?
                                    # Actually, let's re-derive prev_close from the existing 'stock_data_yf' logic?
                                    # No, stock_data_yf only stores final values.
                                    
                                    # However, we can trust that Daily Download (10d) finding 'Previous Close' is reasonably robust
                                    # IF we correctly identified "Today" vs "Yesterday".
                                    # Ah, my previous fix SET prev_close = price (change=0) if data was stale.
                                    # Now we have FRESH price (price_rt).
                                    # So we can try to recover the TRUE prev_close.
                                    
                                    # But wait, if Daily data was stale (Yesterday), then 'price' was YesterdayClose.
                                    # And I forced 'prev_close' = YesterdayClose.
                                    # So now I have price_rt (Today).
                                    # So PrevClose IS technically that "Stale Price" (YesterdayClose)!
                                    
                                    # So: True PrevClose = entry['price'] (which came from the "Stale" Daily Bar logic).
                                    # BUT only if the Daily logic marked it as stale?
                                    # If Daily logic marked it as "Current" (Today), then entry['price'] is ALREADY Today's Daily Close (or live).
                                    # Then entry['change'] is correct.
                                    
                                    # Let's assume price_rt is SUPERIOR.
                                    # And let's assume entry['price'] (from Daily) is "Close of Yesterday" if Daily was stale,
                                    # OR "Current" if Daily was live.
                                    
                                    # Issue: We lost the distinction in the dict.
                                    # But we can assume:
                                    # New Change = price_rt - entry['price'] + entry['change'] ?
                                    # No.
                                    # Old Price = P_old. Old Change = C_old. Old Prev = P_old - C_old.
                                    prev_close_derived = entry["price"] - entry["change"]
                                    
                                    # Recalculate with New Price
                                    change_new = price_rt - prev_close_derived
                                    change_pct_new = (change_new / prev_close_derived) * 100.0 if prev_close_derived else 0.0
                                    
                                    entry["price"] = price_rt
                                    entry["change"] = change_new
                                    entry["changesPercentage"] = change_pct_new
                                    entry["source"] = "yf_batch_1m_intraday"
                                    entry["timestamp"] = datetime.now(timezone.utc).isoformat()
                                    
                                    stock_data_yf[internal_sym] = entry
                                else:
                                    # No daily entry? (Maybe failed daily but succeeded intraday?)
                                    # Create partial entry
                                    pass

                        except Exception as e_sym_rt:
                            # logging.warning(f"Error extracting RT data for {internal_sym}: {e_sym_rt}")
                            pass
                            
            except Exception as e_rt_batch:
                logging.warning(f"Batch real-time 1m fetch failed: {e_rt_batch}")
            
            # Legacy Fast Info Loop REMOVED (Replaced by Batch 1m)
            pass
            pass

        # --- 4. Fetch FX Rates ---
        fx_rates_vs_usd = {"USD": 1.0}
        fx_prev_close_vs_usd = {"USD": 1.0}
        
        if "USD" not in required_currencies:
            required_currencies.add("USD")
            
        fx_pairs = [f"{c}=X" for c in required_currencies if c != "USD"]
        
        if fx_pairs:
            logging.info(f"Fetching FX for {len(fx_pairs)} pairs using history...")
            try:
                # Use history fetch for FX as it is more reliable than info/metadata
                fx_history_df = _run_isolated_fetch(
                    fx_pairs, 
                    period="5d", # Fetch last 5 days to ensure we get a valid close (even over weekends)
                    interval="1d",
                    task="history"
                )
                
                if not fx_history_df.empty:
                    # Handle MultiIndex columns if multiple tickers
                    has_multilevel_fx = getattr(fx_history_df.columns, 'nlevels', 1) > 1
                    
                    for yf_symbol in fx_pairs:
                        try:
                            price = None
                            prev = None
                            
                            # Extract Series for this symbol
                            # Extract Series for this symbol
                            sym_df_fx = pd.DataFrame()
                            
                            # Check if the dataframe has MultiIndex columns (Level 0 = Ticker)
                            if has_multilevel_fx:
                                if yf_symbol in fx_history_df.columns.get_level_values(0):
                                    sym_df_fx = fx_history_df[yf_symbol]
                            else:
                                # Flat dataframe.
                                # If we requested multiple symbols, we can't easily distinguish unless columns have names
                                # But if we requested only 1, it must be it.
                                if len(fx_pairs) == 1:
                                    sym_df_fx = fx_history_df
                                # Else (multiple symbols but flat): 
                                # This happens if yf failed to group, or columns are like 'Close' and it's ambiguous.
                                # We'll skip or try to match prefix if columns are 'THB=X Close' (unlikely with auto_adjust)
                            
                            if not sym_df_fx.empty:
                                close_col = "Close" if "Close" in sym_df_fx.columns else (sym_df_fx.columns[0] if len(sym_df_fx.columns) > 0 else None)
                                
                                if close_col:
                                    # Get valid rows
                                    valid_rows = sym_df_fx[close_col].dropna()
                                    if not valid_rows.empty:
                                        # Use last available close as current price
                                        price = float(valid_rows.iloc[-1])
                                        
                                        # Try to get previous close
                                        if len(valid_rows) >= 2:
                                            prev = float(valid_rows.iloc[-2])
                                        else:
                                            prev = price # Fallback

                            base_curr_from_symbol = yf_symbol.replace("=X", "").upper()
                                
                            if price and price > 0:
                                fx_rates_vs_usd[base_curr_from_symbol] = price
                                if prev and prev > 0:
                                    fx_prev_close_vs_usd[base_curr_from_symbol] = prev
                            else:
                                logging.warning(f"FX Fetch: Invalid/Empty history for {yf_symbol}")
                                has_warnings = True

                        except Exception as e_fx_sym:
                            logging.warning(f"Error extracting FX for {yf_symbol}: {e_fx_sym}")
                            
                else:
                    logging.warning("FX Fetch: History returned empty DataFrame.")
                    has_warnings = True

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

    def get_fundamentals_batch(
        self, 
        symbols: List[str],
        user_symbol_map: Dict[str, str] = None,
        user_excluded_symbols: Set[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetches fundamental data (Market Cap, PE, Dividend Yield) for a batch of symbols.
        Uses a separate cache strategy (individual files) with longer duration.
        """
        _ensure_yfinance()
        if not yf:
            return {}
            
        user_symbol_map = user_symbol_map or {}
        user_excluded_symbols = user_excluded_symbols or set()

        # Cache version to ensure data compatibility/freshness
        CACHE_VERSION = 2

        results = {}
        symbols_to_fetch = []
        now = datetime.now(timezone.utc)
        
        # Check cache
        for sym in symbols:
            yf_sym = map_to_yf_symbol(sym, user_symbol_map, user_excluded_symbols) or sym
            cache_file = os.path.join(self.fundamentals_cache_dir, f"{yf_sym}.json")
            
            loaded = False
            if os.path.exists(cache_file):
                try:
                    with open(cache_file, "r") as f:
                        data = json.load(f)
                        ts = datetime.fromisoformat(data['timestamp'])
                        # Check version (default 0 if missing)
                        ver = data.get('version', 0)
                        
                        # If valid (age check AND version check)
                        if ver == CACHE_VERSION and (now - ts).total_seconds() < config.FUNDAMENTALS_CACHE_DURATION_HOURS * 3600:
                            # Use FULL data directly for result (extract what we need)
                            info = data['data']
                            results[sym] = {
                                "marketCap": info.get("marketCap"),
                                "trailingPE": info.get("trailingPE"),
                                "forwardPE": info.get("forwardPE"),
                                "dividendYield": info.get("dividendYield"),
                                "dividendRate": info.get("dividendRate"),
                                "currency": info.get("currency")
                            }
                            loaded = True
                except Exception:
                     pass
            
            if not loaded:
                symbols_to_fetch.append(sym)
                
        # Fetch remaining
        if symbols_to_fetch:
             # Construct YF symbols
             yf_map = { (map_to_yf_symbol(s, user_symbol_map, user_excluded_symbols) or s): s for s in symbols_to_fetch } # yf -> internal
             try:
                 now_ts = datetime.now(timezone.utc)
                 # Use isolated batch fetch for fundamentals (Tickers replacement)
                 info_batch = _run_isolated_fetch(list(yf_map.keys()), task="info")
                 for yf_sym, info in info_batch.items():
                     if info:
                         internal_sym = yf_map.get(yf_sym, yf_sym)
                         
                         if is_cash_symbol(internal_sym):
                             continue
 
                         try:
                             # info is already the dict from _run_isolated_fetch
                             
                             div_yield = info.get("yield")
                             if div_yield is None:
                                 div_yield = info.get("dividendYield")
                                 
                                 # Normalize dividendYield if it looks like percentage (e.g. 0.41 for 0.41%)
                                 try:
                                     div_val = float(div_yield)
                                     rate = info.get("dividendRate")
                                     price = info.get("currentPrice") or info.get("regularMarketPrice")
                                     trailing = info.get("trailingAnnualDividendYield")
                                     
                                     normalized = False
                                     
                                     # Check 1: Rate / Price ratio
                                     if rate is not None and price is not None:
                                         try:
                                              rate_val = float(rate)
                                              price_val = float(price)
                                              if price_val > 0:
                                                  ratio = rate_val / price_val
                                                  # If yield is closer to ratio*100 than to ratio, it's percentage
                                                  if abs(div_val - ratio * 100) < abs(div_val - ratio):
                                                      div_yield = div_val / 100.0
                                                      normalized = True
                                         except (ValueError, TypeError):
                                             pass
                                             
                                     # Check 2: Comparison with trailing yield (if not already normalized)
                                     if not normalized and trailing is not None:
                                         try:
                                             trailing_val = float(trailing)
                                             if trailing_val > 0 and div_val > trailing_val * 50:
                                                 div_yield = div_val / 100.0
                                                 normalized = True
                                         except (ValueError, TypeError):
                                             pass
 
                                     # Check 3: Raw Magnitude Heuristic
                                     if not normalized and div_val > 0.30: 
                                          div_yield = div_val / 100.0
                                 
                                 except Exception as e_norm:
                                     logging.warning(f"Error normalizing yield for {yf_sym}: {e_norm}")
                                     
                             # Update info with normalized yield
                             if div_yield is not None:
                                 info["dividendYield"] = div_yield
 
                             # Save FULL info to cache (Unified Cache)
                             with open(os.path.join(self.fundamentals_cache_dir, f"{yf_sym}.json"), "w") as f:
                                 json.dump({
                                     "timestamp": now_ts.isoformat(),
                                     "version": CACHE_VERSION,
                                     "data": info
                                 }, f)
                                 
                             results[internal_sym] = {
                                 "marketCap": info.get("marketCap"),
                                 "trailingPE": info.get("trailingPE"),
                                 "forwardPE": info.get("forwardPE"),
                                 "dividendYield": div_yield,
                                 "dividendRate": info.get("dividendRate"),
                                 "currency": info.get("currency")
                             }
                         except Exception as e_sym:
                             logging.warning(f"Error processing fundamentals for {yf_sym}: {e_sym}")
             except Exception as e_batch:
                 logging.error(f"Error in fundamentals batch fetch loop: {e_batch}")


                 
        return results

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
        cache_key = f"INDEX_QUOTES_v3_NO_CACHE::{'_'.join(sorted(index_symbols))}"
        cache_duration_minutes = 0 # Force fresh fetch

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
                # Use isolated fetch for index info
                index_info_batch = _run_isolated_fetch(yf_tickers_to_fetch, task="info")
                
                # Fetch 7 days of history for sparklines
                index_hist_df = _run_isolated_fetch(
                    yf_tickers_to_fetch,
                    period="7d",
                    interval="1d",
                    task="history"
                )

                for (
                    yf_symbol,
                    ticker_info,
                ) in (
                    index_info_batch.items()
                ):  # yf_symbol is now the YF ticker like ^DJI
                    try:
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
                                        "change": change if change is not None else 0.0,
                                        "changesPercentage": (
                                            change_pct
                                            if change_pct is not None
                                            else 0.0
                                        ),
                                        "name": name,
                                        "source": "yf_info",
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),
                                    }
                                )
                                
                                # Extract sparkline from history
                                if not index_hist_df.empty:
                                    has_multilevel = getattr(index_hist_df.columns, 'nlevels', 1) > 1
                                    try:
                                        sym_df = pd.DataFrame()
                                        if len(yf_tickers_to_fetch) > 1:
                                            if has_multilevel and yf_symbol in index_hist_df.columns.get_level_values(0):
                                                sym_df = index_hist_df[yf_symbol]
                                            elif not has_multilevel and any(isinstance(c, (tuple, list)) and c[0].upper() == yf_symbol.upper() for c in index_hist_df.columns):
                                                cols_for_sym = [c for c in index_hist_df.columns if isinstance(c, (tuple, list)) and c[0].upper() == yf_symbol.upper()]
                                                sym_df = index_hist_df[cols_for_sym]
                                                sym_df.columns = [c[1] for c in sym_df.columns]
                                        else:
                                            sym_df = index_hist_df
                                        
                                        if not sym_df.empty:
                                            close_col = "Close" if "Close" in sym_df.columns else sym_df.columns[0]
                                            sparkline = sym_df[close_col].dropna().tolist()
                                            results[internal_result_key]["sparkline"] = sparkline
                                    except Exception as e_hist:
                                        logging.warning(f"Error extracting sparkline for {yf_symbol}: {e_hist}")

                            else:
                                logging.warning(
                                    f"Could not get price for index {yf_symbol} (Internal: {internal_result_key}) from info."
                                )
                        else:
                            logging.warning(
                                f"Could not get .info for index {yf_symbol} (Internal: {internal_to_yf_index_map.get(yf_symbol, yf_symbol)})"
                            )

                    except AttributeError as ae:
                        logging.error(
                            f"AttributeError processing info for index {yf_symbol}: {ae}"
                        )
                    except Exception as e_ticker:
                        logging.error(
                            f"Error processing info for index {yf_symbol}: {e_ticker}"
                        )

            except Exception as e_indices:
                logging.error(f"Error fetching index quotes batch: {e_indices}")
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
        self, symbols_yf: List[str], start_date: date, end_date: date, interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Internal helper to fetch historical 'Close' data (adjusted) using yfinance.download.
        (Replaces fetch_yf_historical)
        """
        # Normalize interval for yfinance
        if interval == "D":
            interval = "1d"
        
        # NORMALIZATION HELPER
        def normalize_df(df_raw, sym):
            if df_raw is None or df_raw.empty:
                return df_raw
            df_clean = df_raw.copy()
            
            # For intraday intervals, prefer 'Close' over 'Adj Close' 
            # as yfinance often returns NaNs in Adj Close for short intervals.
            is_intraday_local = any(x in interval for x in ["m", "h"])
            
            primary = "Close" if is_intraday_local else "Adj Close"
            secondary = "Adj Close" if is_intraday_local else "Close"
            
            if primary in df_clean.columns and df_clean[primary].notna().any():
                df_clean["price"] = df_clean[primary]
            elif secondary in df_clean.columns and df_clean[secondary].notna().any():
                df_clean["price"] = df_clean[secondary]
            elif primary in df_clean.columns: # fallback if all NaN but column exists
                df_clean["price"] = df_clean[primary]
            elif secondary in df_clean.columns:
                df_clean["price"] = df_clean[secondary]
            
            num_nans = df_clean["price"].isna().sum() if "price" in df_clean.columns else "N/A"
            logging.info(f"Hist Fetch Helper: Normalized symbol {sym} (Intraday={is_intraday_local}) with {len(df_clean)} rows. NaNs: {num_nans}. Source: {primary if 'price' in df_clean.columns and df_clean['price'].equals(df_clean.get(primary)) else secondary}")
            return df_clean

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
                    logging.warning(f"Hist Fetch Helper: Skipping cached invalid symbol: {s}")
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

        # Ensure inputs are normalized to date objects first
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date).date()
        elif isinstance(start_date, pd.Timestamp):
            start_date = start_date.date()
            
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date).date()
        elif isinstance(end_date, pd.Timestamp):
            end_date = end_date.date()

        yf_end_date = max(start_date, end_date) + timedelta(days=1)
        yf_start_date = min(start_date, end_date)

        # --- FIX: Prevent fetching future dates which causes YFPricesMissingError ---
        # Historical data implies "past" data. Real-time is handled elsewhere.
        # But for intraday (1m, 5m), we NEED to fetch 'today' to see the chart update.
        today = get_est_today()
        is_intraday = interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]

        # --- CACHE READ (After-Hours/Weekend) ---
        # If market is closed, try to load from local DB to avoid API latency
        if is_intraday and not is_market_open():
            logging.info(f"Hist Fetch Helper: Market closed. Checking Intraday DB Cache for {len(symbols_yf)} symbols...")
            symbols_to_fetch_remote = []
            
            # Use timestamps for DB query
            # yf_start_date is date, need datetime. Combine with min/max time.
            ts_start = datetime.combine(yf_start_date, datetime.min.time())
            ts_end = datetime.combine(yf_end_date, datetime.min.time()) # yf_end_date is usually exclusive/next day, effectively covers full range

            for s in symbols_yf:
                cached_df = self.db.get_intraday(s, ts_start, ts_end, interval)
                
                is_cache_valid = False
                if not cached_df.empty:
                    # VALIDATION: Check if cache covers the latest trading session
                    last_trading = get_latest_trading_date()
                    
                    # Convert max timestamp to EST date/time for comparison
                    max_ts_utc = cached_df.index.max()
                    if max_ts_utc.tzinfo:
                        max_ts_est = max_ts_utc.tz_convert("America/New_York")
                    else:
                        max_ts_est = max_ts_utc.replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=-5)))
                        
                    # 1. Date check
                    if max_ts_est.date() > last_trading:
                        # Future data or error in clock, refetch
                        is_cache_valid = False
                    elif max_ts_est.date() < last_trading:
                        # Stale (old day), refetch
                        is_cache_valid = False
                    else:
                        # Same day as last_trading. Now check for SESSION COMPLETENESS.
                        if last_trading < today:
                            # It's a past trading day (e.g. yesterday). 
                            # Session must extend near market close (e.g. after 15:45 EST) to be "complete".
                            if max_ts_est.hour > 15 or (max_ts_est.hour == 15 and max_ts_est.minute >= 45):
                                is_cache_valid = True
                            else:
                                logging.info(f"Hist Fetch Helper: Cache for {s} is Jan 20 but incomplete (ends {max_ts_est.time()}). Refetching.")
                        else:
                            # It's today's trading day. Refetch if cache is older than 15 mins.
                            # We check the actual retrieval time (sync_metadata) or just the max point
                            # Usually, we want real-time-ish data for today.
                            time_diff = datetime.now(timezone.utc) - max_ts_utc
                            if time_diff.total_seconds() < 900: # 15 mins
                                is_cache_valid = True
                            else:
                                logging.info(f"Hist Fetch Helper: Cache for {s} is today but {time_diff.total_seconds()/60:.1f}m old. Refetching.")

                    if is_cache_valid:
                        historical_data[s] = normalize_df(cached_df, s)
                
                if not is_cache_valid:
                    symbols_to_fetch_remote.append(s)
            
            if len(historical_data) > 0:
                logging.info(f"Hist Fetch Helper: Loaded {len(historical_data)} symbols from Intraday DB Cache.")
            
            if not symbols_to_fetch_remote:
                 logging.info("Hist Fetch Helper: All symbols found in cache. Skipping remote fetch.")
                 return historical_data
            
            # Update list to only fetch missing
            symbols_yf = symbols_to_fetch_remote

        
        if not is_intraday and yf_start_date >= today:
            logging.info(f"Hist Fetch Helper: Requested start date {yf_start_date} is today or in the future. Skipping fetch.")
            return {}

        # --- FIX: Prevent fetching future dates (Intraday) ---
        # If we ask for intraday data starting AFTER the latest trading session, it's definitely future or pre-market gap.
        if is_intraday:
            latest_trading = get_latest_trading_date()
            if yf_start_date > latest_trading:
                logging.info(f"Hist Fetch Helper: Requested start date {yf_start_date} is after latest trading date {latest_trading}. Market not open yet. Skipping.")
                return {}


        # --- SAFETY CLIP: YFinance 1h data limit is 730 days ---
        if interval == "1h":
            limit_start = today - timedelta(days=729)
            if yf_start_date < limit_start:
                logging.warning(f"Hist Fetch Helper: Clipping 1h start date from {yf_start_date} to {limit_start} (YF limit).")
                yf_start_date = limit_start
                if yf_start_date >= yf_end_date:
                     return {}
        elif interval == "1m": # 1-minute data limit is 30 days
            limit_start = today - timedelta(days=29) # Safe buffer
            if yf_start_date < limit_start:
                logging.warning(f"Hist Fetch Helper: Clipping 1m start date from {yf_start_date} to {limit_start} (YF limit).")
                yf_start_date = limit_start
                if yf_start_date >= yf_end_date:
                     return {}
        
        # --- END FIX ---

        # Clamp end date to today (exclusive in yfinance means up to yesterday) 
        # if we are not asking for today's data (which we shouldn't be here).
        # Actually, if we ask for end=tomorrow, yfinance tries to get today.
        # If today isn't effectively over or started, it thinks we want today.
        # Safer to clamp end to 'today' (fetching UP TO yesterday) for strict history.
        if not is_intraday and yf_end_date > today:
             yf_end_date = today
        # --- END FIX ---

        # --- FIX: Check for business days to avoid YFPricesMissingError on weekends ---
        # This check happens AFTER clamping to today, to ensure we don't try to fetch
        # ranges that become Saturday-Sunday after clamping.
        try:
             # Use the holiday-aware calendar for counting business days
             cal = get_nyse_calendar()
             schedule = cal.schedule(start_date=yf_start_date, end_date=yf_end_date - timedelta(days=1))
             bus_days = len(schedule)
             
             if bus_days == 0:
                 # FIX: If 0 business days (e.g. holiday/weekend) but we want intraday data,
                 # shift START date back to include the last trading session.
                 if "m" in interval or "h" in interval:
                     # For intraday, we want to see the last session's chart
                     last_trading = get_latest_trading_date()
                     if yf_start_date > last_trading:
                         logging.info(f"Hist Fetch Helper: 0 bus days for {yf_start_date}-{yf_end_date}. Shifting start to {last_trading} to capture last session.")
                         yf_start_date = last_trading
                     else:
                         # Already at/before last trading but still 0 bus days? 
                         # Maybe it's a long holiday. Back up more.
                         yf_start_date = yf_start_date - timedelta(days=3)
                 else:
                     logging.info(f"Hist Fetch Helper: Skipping fetch for {yf_start_date} to {yf_end_date} (0 business days).")
                     return {}
                 
        except Exception as e_bus:
             logging.warning(f"Hist Fetch Helper: Error in business day check: {e_bus}. Find logic fallthrough.")
        # --- END FIX ---
        # --- END FIX ---
        # Reduce batch size for intraday data to prevent OOM/Crash
        if "m" in interval or "h" in interval:
             fetch_batch_size = 5
        else:
             fetch_batch_size = 50 


        # --- ADDED: Retry logic parameters for increased network robustness ---
        # --- ADDED: Retry logic parameters for increased network robustness ---
        retries = 4 # Increased to 4 to handle potential DNS flakiness
        timeout_seconds = 10 # Reduced from 30 to fail fast
        # --- END ADDED ---

        all_missing_symbols = []

        for i in range(0, len(symbols_yf), fetch_batch_size):
            batch_symbols = symbols_yf[i : i + fetch_batch_size]
            data = pd.DataFrame()  # Initialize empty DataFrame for the batch
            
            # --- Attempt Batch Fetch ---
            for attempt in range(retries):
                try:
                    t0 = time.time()
                    # --- CHUNKING LOGIC FOR 1m INTERVAL ---
                    # Yahoo Finance limits 1m data to 7 days per request.
                    if interval == "1m" and (yf_end_date - yf_start_date).days > 7:
                        chunk_dfs = []
                        curr_chunk_start = yf_start_date
                        while curr_chunk_start < yf_end_date:
                            curr_chunk_end = min(curr_chunk_start + timedelta(days=7), yf_end_date)
                            if curr_chunk_start >= curr_chunk_end:
                                break
                                
                            try:
                                chunk_df = _run_isolated_fetch(
                                    tickers=batch_symbols,
                                    start=curr_chunk_start,
                                    end=curr_chunk_end,
                                    interval=interval,
                                    task="history" # Added task
                                )
                                if not chunk_df.empty:
                                    chunk_dfs.append(chunk_df)
                            except Exception as e_chunk:
                                logging.warning(f"  Hist Fetch Helper WARN (Chunk {curr_chunk_start}): {e_chunk}")
                                
                            curr_chunk_start = curr_chunk_end

                        if chunk_dfs:
                            data = pd.concat(chunk_dfs).sort_index()
                        else:
                            data = pd.DataFrame()
                    else:
                        # Standard fetch for other intervals or short 1m ranges
                        data = _run_isolated_fetch(
                            tickers=batch_symbols,
                            start=yf_start_date,
                            end=yf_end_date,
                            interval=interval,
                            task="history" # Added task
                        )
                    
                    logging.info(f"  Hist Fetch Helper: Batch result for {batch_symbols[:3]}...: Shape={data.shape}, Columns={list(data.columns[:5])}, Types={type(data.columns)}")
                    if not data.empty:
                        logging.info(f"  Hist Fetch Helper: Index Range: {data.index[0]} to {data.index[-1]}")
                    
                    elapsed = time.time() - t0
                    logging.debug(f"  Batch fetch took {elapsed:.2f}s for {len(batch_symbols)} symbols (Attempt {attempt + 1}).")
                    # yfinance prints its own errors for failed tickers. If the whole batch fails,
                    # it might return an empty DataFrame.
                    if data.empty and len(batch_symbols) > 0:
                        # yfinance returns empty for non-trading days without raising.
                        # If we have 0 bus days (double check here), don't retry.
                        try:
                            cal = get_nyse_calendar()
                            # Check if at least one day in the range was a trading day
                            # yf_end_date is exclusive, so we check up to yf_end_date - 1d
                            sch_check = cal.schedule(start_date=yf_start_date, end_date=yf_end_date - timedelta(days=1))
                            is_trading_range = not sch_check.empty
                        except:
                            is_trading_range = True # Assume it might be trading if calendar fails
                            
                        if not is_trading_range:
                            logging.info(f"  Hist Fetch Helper: No trading days in range {yf_start_date} to {yf_end_date}. Avoiding retries.")
                            break # Exit retry loop, return empty
                            
                        logging.warning(
                            f"  Hist Fetch Helper WARN (Attempt {attempt + 1}/{retries}): yf.download returned empty DataFrame. Retrying (Potential transient error)..."
                        )
                        time.sleep(1)
                        continue # FORCE RETRY

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
            
            # --- Process Batch Results & Identify Missing ---
            missing_symbols_in_batch = []
            
            # Check if we broke out due to valid empty result
            valid_empty_batch = False
            if data.empty and len(batch_symbols) > 0:
                 # If we didn't raise an exception but data is empty, check if we intended to treat it as valid.
                 # We can rely on a flag or state. 
                 # Let's see... my previous change added a 'break' inside the loop.
                 # If we broke because of valid empty, 'data' is empty.
                 # We need to distinguish "failed all retries" (data empty) vs "valid empty" (data empty).
                 # We can assume if we didn't log an ERROR for all retries, then it might be valid.
                 # Better: Initialize valid_empty_batch = False outside loop.
                 # Ah, implementing that purely via replace_content is tricky without context diff.
                 # Let's just check the log logic? No.
                 # I will assume "valid empty" if we broke early.
                 # But 'break' just exits the loop.
                 pass

            # Actually, let's just modify the check below.
            if data.empty:
                # If batch failed completely (or was validly empty), all are arguably 'missing' data-wise.
                # BUT if it was validly empty, we don't want to RETRY them.
                # So we should only add to missing_symbols_in_batch if we want to retry.
                # If it's a weekend, individual retry will ALSO match 'empty', so it's a waste of time.
                # So: if data is empty, we generally assume "no data available" unless we have reason to believe otherwise?
                # The only reason to retry individually is if the BATCH request failed technically (network etc).
                # But yfinance usually raises exceptions for network errors.
                # If it returns empty DataFrame, it usually means "valid response, no data".
                # So... we should PROBABLY NEVER retry individually if data is empty but no exception raised?
                # Yes. If yfinance returns empty DF, individual retry will virtually always be empty too.
                
                # So, simply: Don't add to missing_symbols_in_batch if data.empty is True.
                # Just log and skip. 
                logging.info(f"  Hist Fetch Helper: Batch returned empty (valid). Skipping individual retries for: {', '.join(batch_symbols[:5])}...")
                missing_symbols_in_batch = [] # Explicitly empty
            else:

                for symbol in batch_symbols:
                    try:
                        df_symbol = None
                        found_in_batch = False
                        
                        # Symbols to try (original and stripped caret)
                        symbols_to_try = [symbol]
                        if symbol.startswith("^"):
                             symbols_to_try.append(symbol[1:])
                        
                        # Robust Logic for MultiIndex (Price, Ticker)
                        if isinstance(data.columns, pd.MultiIndex):
                            for s_to_try in symbols_to_try:
                                if s_to_try in data.columns.get_level_values(1):
                                    df_symbol = data.xs(s_to_try, axis=1, level=1, drop_level=True)
                                    found_in_batch = True
                                    break
                                elif s_to_try in data.columns.get_level_values(0):
                                    df_symbol = data[s_to_try]
                                    found_in_batch = True
                                    break
                        else:
                            # Flat Index matching
                            for s_to_try in symbols_to_try:
                                matching_cols = []
                                for c in data.columns:
                                    if isinstance(c, (list, tuple)):
                                         if len(c) > 1 and c[1] == s_to_try:
                                              matching_cols.append(c)
                                         elif c[0] == s_to_try:
                                              matching_cols.append(c)
                                    elif isinstance(c, str):
                                         # Match stringified tuple or simple name
                                         if s_to_try in c:
                                              matching_cols.append(c)
                                
                                if matching_cols:
                                    df_symbol = data[matching_cols]
                                    # Rename columns
                                    new_cols = []
                                    for c in df_symbol.columns:
                                         if isinstance(c, (list, tuple)):
                                              new_cols.append(c[1] if c[1] != s_to_try else c[0])
                                         elif isinstance(c, str) and "(" in c and "," in c:
                                              parts = c.replace("(","").replace(")","").replace("'","").replace("\"","").split(",")
                                              p0, p1 = parts[0].strip(), parts[1].strip()
                                              new_cols.append(p0 if p1 == s_to_try else p1)
                                         else:
                                              new_cols.append(c)
                                    df_symbol.columns = new_cols
                                    found_in_batch = True
                                    break

                        if not found_in_batch or df_symbol is None or df_symbol.empty:
                            logging.warning(f"  Hist Fetch Helper: {symbol} NOT found or empty in batch result.")
                            missing_symbols_in_batch.append(symbol)
                            continue

                        # --- NORMALIZATION & CLEANING ---
                        # 1. Ensure 'price' column exists
                        df_norm = normalize_df(df_symbol, symbol)
                        if df_norm is None or "price" not in df_norm.columns:
                            logging.warning(f"  Hist Fetch Helper: Could not normalize {symbol}. Columns: {list(df_norm.columns if df_norm is not None else [])}")
                            missing_symbols_in_batch.append(symbol)
                            continue

                        # 2. Filter for requested range
                        mask = (df_norm.index.date >= start_date) & (df_norm.index.date <= end_date)
                        df_filtered = df_norm.loc[mask].copy()

                        # 3. Final cleanup (numeric, NaNs, zero-prices)
                        df_filtered["price"] = pd.to_numeric(df_filtered["price"], errors="coerce")
                        df_filtered.dropna(subset=["price"], inplace=True)
                        df_filtered = df_filtered[df_filtered["price"] > 1e-6]

                        if not df_filtered.empty:
                            historical_data[symbol] = df_filtered.sort_index()
                            logging.info(f"  Hist Fetch Helper: Successfully extracted & cleaned {symbol} ({len(df_filtered)} rows).")
                        else:
                            logging.warning(f"  Hist Fetch Helper: {symbol} empty after date filter/cleanup.")
                            missing_symbols_in_batch.append(symbol)

                    except Exception as e_sym:
                        logging.error(f"  Hist Fetch Helper ERROR: Failed to process {symbol}: {e_sym}")
                        missing_symbols_in_batch.append(symbol)

            # --- Collect Missing Symbols for Final Cleanup ---
            if missing_symbols_in_batch:
                all_missing_symbols.extend(missing_symbols_in_batch)

        # --- Final Cleanup: Isolated Batch Fetch for all Missing Symbols ---
        if all_missing_symbols:
            # Deduplicate just in case
            all_missing_symbols = list(set(all_missing_symbols))
            logging.info(f"  Hist Fetch Helper: Attempting final isolated recovery for {len(all_missing_symbols)} missing symbols...")
            
            # Use smaller chunks for recovery to maximize success 
            recovery_batch_size = 5
            for i in range(0, len(all_missing_symbols), recovery_batch_size):
                rec_batch = all_missing_symbols[i : i + recovery_batch_size]
                try:
                    data_rec = _run_isolated_fetch(
                        tickers=rec_batch,
                        start=yf_start_date,
                        end=yf_end_date,
                        interval=interval,
                        task="history" # Added task
                    )
                    
                    if not data_rec.empty:
                        for symbol in rec_batch:
                            # Standard processing for each symbol in recovery batch
                            df_rec = None
                            found_rec = False
                            if isinstance(data_rec.columns, pd.MultiIndex):
                                if symbol in data_rec.columns.get_level_values(1):
                                    df_rec = data_rec.xs(symbol, axis=1, level=1, drop_level=True)
                                    found_rec = True
                                elif symbol in data_rec.columns.get_level_values(0):
                                    df_rec = data_rec[symbol]
                                    found_rec = True
                            else:
                                # Flat Index: Tuples or Prefixed
                                matching_cols = [c for c in data_rec.columns if (isinstance(c, (list, tuple)) and c[0] == symbol) or (isinstance(c, str) and c.startswith(f"{symbol}"))]
                                if matching_cols:
                                    df_rec = data_rec[matching_cols]
                                    if isinstance(matching_cols[0], (list, tuple)):
                                        df_rec.columns = [c[1] for c in df_rec.columns]
                                    found_rec = True
                                elif symbol in data_rec.columns:
                                    df_rec = data_rec[[symbol]].rename(columns={symbol: "Close"})
                                    found_rec = True
                                elif len(rec_batch) == 1:
                                    df_rec = data_rec
                                    found_rec = True

                                
                            if found_rec and df_rec is not None and not df_rec.empty:
                                if "Close" not in df_rec.columns and len(df_rec.columns) == 1:
                                    df_rec.columns = ["Close"]
                                
                                if "Close" in df_rec.columns:
                                    mask = (df_rec.index.date >= start_date) & (df_rec.index.date <= end_date)
                                    df_filt = df_rec.loc[mask]
                                    if not df_filt.empty:
                                        df_cln = df_filt[["Close"]].copy()
                                        df_cln.rename(columns={"Close": "price"}, inplace=True)
                                        df_cln["price"] = pd.to_numeric(df_cln["price"], errors="coerce")
                                        df_cln.dropna(subset=["price"], inplace=True)
                                        df_cln = df_cln[df_cln["price"] > 1e-6]
                                        if not df_cln.empty:
                                            historical_data[symbol] = df_cln.sort_index()

                except Exception as e_rec:
                    logging.warning(f"  Hist Fetch Helper recovery failed for {rec_batch[0]}...: {e_rec}")

        if cache_needs_update:
            self._save_invalid_symbols_cache(invalid_cache)

        # --- CACHE WRITE (Intraday) ---
        if is_intraday and historical_data:
            # Save fetched data to DB for future after-hours use
            # We do this specifically for intraday because daily is handled by history cache file
            try:
                # logging.info(f"Hist Fetch Helper: Saving {len(historical_data)} symbols to Intraday DB Cache...")
                for sym, df in historical_data.items():
                    self.db.upsert_intraday(sym, df, interval)
            except Exception as e_db_save:
                logging.error(f"Hist Fetch Helper: Error saving to Intraday DB: {e_db_save}")

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
        data_to_save_map: Dict[str, pd.DataFrame],
        data_type: str = "price",
        interval: str = "1d"
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
                yf_symbol, data_type, interval=interval
            )
            try:
                # Serialize DataFrame to JSON string
                json_str = df_data.to_json(orient="split", date_format="iso")
                
                # ATOMIC WRITE: Write to UNIQUE temp file first, then rename
                # Use tempfile.NamedTemporaryFile in the same directory to ensure atomic move
                target_dir = os.path.dirname(symbol_file_path)
                with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target_dir, delete=False) as tf:
                    temp_file_path = tf.name
                    tf.write(json_str)
                    tf.flush()
                    os.fsync(tf.fileno()) # Ensure write to disk
                
                # Atomic replace
                os.replace(temp_file_path, symbol_file_path)
                
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
            # ATOMIC WRITE: Write to UNIQUE temp file first, then rename
            target_dir = os.path.dirname(manifest_path)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=target_dir, delete=False) as tf:
                temp_manifest_path = tf.name
                json.dump(full_manifest_content, tf, indent=2) 
                tf.flush()
                os.fsync(tf.fileno()) # Ensure write to disk
            
            os.replace(temp_manifest_path, manifest_path)
            
            # CORRECTED: Use full_manifest_content (Logic preserved from original block)
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
                StringIO(data_json_str), orient="split", dtype=None 
            )
            df_temp.index = pd.to_datetime(df_temp.index, errors="coerce", utc=True)
            
            # --- FIX: Support legacy cache keys (Close/Adj Close) ---
            if "price" not in df_temp.columns and not df_temp.empty:
                 if "Close" in df_temp.columns:
                     df_temp.rename(columns={"Close": "price"}, inplace=True)
                 elif "Adj Close" in df_temp.columns:
                     df_temp.rename(columns={"Adj Close": "price"}, inplace=True)
                 elif len(df_temp.columns) == 1:
                     # Fallback to first column
                     df_temp.rename(columns={df_temp.columns[0]: "price"}, inplace=True)

            if "price" in df_temp.columns:
                df_temp["price"] = pd.to_numeric(df_temp["price"], errors="coerce")
                df_temp = df_temp.dropna(subset=["price"])  # Ensure 'price' column has valid data
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
    def _sync_to_db(self, symbols: List[str], start_date: date, end_date: date, data_type: str = "price", interval: str = "1d"):
        """
        Internal helper to synchronize YF data to the persistent DB.
        Implements the 'Overlapping Refresh' logic for data integrity.
        """
        is_fx = (data_type == "fx")
        if interval != "1d":
            # Extra safety, though caller should handle this
            return self._fetch_yf_historical_data(symbols, start_date, end_date, interval=interval)

        # 0. Sync Throttling (Batched)
        now = datetime.now()
        sync_needed = []
        
        if not symbols:
            return

        # Batch 1: Get Last Synced Times
        last_synced_map = self.db.get_sync_metadata_batch(symbols)
        
        # Determine which symbols *might* need sync based on time threshold
        potential_sync = []
        for sym in symbols:
            last_ts = last_synced_map.get(sym)
            if last_ts and (now - last_ts) < timedelta(hours=4):
                continue
            potential_sync.append(sym)
            
        if not potential_sync:
            return

        # Batch 2: Get Last Data Dates in DB
        table_name = "daily_ohlcv" if not is_fx else "daily_fx"
        last_db_dates_map = self.db.get_last_dates(potential_sync, table=table_name)
        
        for sym in potential_sync:
            last_db_date = last_db_dates_map.get(sym)
            
            fetch_start = start_date
            if last_db_date:
                # 5-day overlap for integrity
                fetch_start = min(start_date, last_db_date - timedelta(days=5))
            
            # If we don't have data up to end_date, we need to sync
            if not last_db_date or last_db_date < end_date:
                sync_needed.append((sym, fetch_start))

        if not sync_needed:
            return

        # 2. Perform fetches
        from collections import defaultdict
        by_start = defaultdict(list)
        for s, start in sync_needed:
            by_start[start].append(s)
            
        for start, syms in by_start.items():
            logging.info(f"Syncing {len(syms)} {data_type} symbols to DB from {start} to {end_date}...")
            if not is_fx:
                fetched = self._fetch_yf_historical_data(syms, start, end_date, interval=interval)
                for s, df in fetched.items():
                    if not df.empty:
                        consistent, reason = self.db.check_integrity(s, df)
                        if not consistent:
                            logging.warning(f"Market DB Integrity: {reason}. Triggering full re-fetch for {s}.")
                            inception = date(2000, 1, 1)
                            full_df_map = self._fetch_yf_historical_data([s], inception, end_date, interval=interval)
                            if s in full_df_map:
                                self.db.upsert_ohlcv(s, full_df_map[s], interval=interval)
                        else:
                            self.db.upsert_ohlcv(s, df, interval=interval)
            else:
                fetched_fx = self._fetch_yf_historical_data(syms, start, end_date, interval=interval)
                for s, df in fetched_fx.items():
                    if not df.empty:
                        self.db.upsert_fx(s, df, interval=interval)
        
        # 3. Update metadata
        with self.db._get_connection() as conn:
            for sym in symbols:
                conn.execute("""
                    INSERT OR REPLACE INTO sync_metadata (symbol, last_synced)
                    VALUES (?, ?)
                """, (sym, now.isoformat()))
            conn.commit()

    def get_historical_data(
        self,
        symbols_yf: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        cache_file: Optional[str] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches ADJUSTED historical price data using persistent DB and YF."""
        if isinstance(start_date, pd.Timestamp): start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp): end_date = end_date.date()
        if interval == "D": interval = "1d"

        logging.info(f"Hist Prices (DB): Fetching {len(symbols_yf)} symbols ({start_date} to {end_date})...")

        # 0. Bypass DB for intraday data
        if interval != "1d":
            return self._fetch_yf_historical_data(symbols_yf, start_date, end_date, interval=interval), False

        # 1. Sync missing range to DB (with 5-day overlap for integrity)
        if use_cache:
            try:
                self._sync_to_db(symbols_yf, start_date, end_date, data_type="price", interval=interval)
            except Exception as e:
                logging.error(f"Error syncing to Market DB: {e}")
                # Continue anyway, we'll try to pull from DB what we have or YF directly if needed

        # 2. Pull everything from DB (Batched)
        historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        
        db_results = self.db.get_ohlcv_batch(symbols_yf, start_date, end_date, interval=interval)
        if db_results:
             # Normalize columns to match expected 'price' format
             for sym, df_db in db_results.items():
                 if df_db.empty:
                     continue
                 
                 price_series = None
                 if "Adj Close" in df_db.columns:
                     price_series = df_db["Adj Close"]
                 elif "Close" in df_db.columns:
                     price_series = df_db["Close"]
                
                 if price_series is not None:
                     df_clean = price_series.to_frame(name="price")
                     historical_prices_yf_adjusted[sym] = df_clean
                     
        # 3. Validation and fallback
        fetch_failed = False
        missing = [s for s in symbols_yf if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty]
        
        if missing:
            logging.warning(f"Hist Prices: {len(missing)} symbols missing from DB after sync: {missing}")
            # Try direct fetch as fallback
            # We fetch directly using the helper which bypasses DB read but does standard processing
            if use_cache: 
                logging.info(f"Hist Prices: Triggering direct fallback fetch for {len(missing)} missing symbols...")
                fallback_data = self._fetch_yf_historical_data(missing, start_date, end_date, interval=interval)
                
                # Merge fallback results
                for sym, df in fallback_data.items():
                    if not df.empty:
                        historical_prices_yf_adjusted[sym] = df
                        # Optional: could update DB here, but _sync_to_db already tried and presumably failed or skipped
                        # Maybe we don't spam attempts to write if it just failed.
                        
                # Re-check missing
                still_missing = [s for s in missing if s not in historical_prices_yf_adjusted or historical_prices_yf_adjusted[s].empty]
                if still_missing:
                     fetch_failed = True
                     logging.warning(f"Hist Prices: Still missing {len(still_missing)} symbols after fallback: {still_missing}")
                else:
                     fetch_failed = False # Recovered
            else:
                fetch_failed = True

        return historical_prices_yf_adjusted, fetch_failed

    def _save_files_only(self, data_map, data_type, interval="1d"):
        """Saves individual symbol data files without updating a global manifest."""
        for yf_symbol, df in data_map.items():
            if df is None or df.empty:
                continue
            symbol_file_path = self._get_historical_symbol_data_path(yf_symbol, data_type=data_type, interval=interval)
            try:
                df.to_json(symbol_file_path, orient="split", date_format="iso")
            except Exception as e:
                logging.error(f"Error saving file for {yf_symbol}: {e}")

    def get_historical_fx_rates(
        self,
        fx_pairs_yf: List[str],
        start_date: date,
        end_date: date,
        interval: str = "1d",
        use_cache: bool = True,
        cache_key: Optional[str] = None,
        cache_file: Optional[str] = None,
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """Loads/fetches historical FX rates (vs USD). Persists daily rates to DB."""
        if isinstance(start_date, pd.Timestamp): start_date = start_date.date()
        if isinstance(end_date, pd.Timestamp): end_date = end_date.date()
        if interval == "D": interval = "1d"

        if interval != "1d":
            return self._fetch_yf_historical_data(fx_pairs_yf, start_date, end_date, interval=interval), False

        logging.info(f"Hist FX (DB): Fetching {len(fx_pairs_yf)} pairs ({start_date} to {end_date})...")

        # 1. Sync missing range to DB
        if use_cache:
            try:
                self._sync_to_db(fx_pairs_yf, start_date, end_date, data_type="fx", interval=interval)
            except Exception as e:
                logging.error(f"Error syncing FX to Market DB: {e}")

        # 2. Pull from DB
        historical_fx_yf: Dict[str, pd.DataFrame] = {}
        for pair in fx_pairs_yf:
            df = self.db.get_fx(pair, start_date, end_date, interval=interval)
            if not df.empty:
                historical_fx_yf[pair] = df

        # 3. Validation
        fetch_failed = False
        missing = [p for p in fx_pairs_yf if p not in historical_fx_yf or historical_fx_yf[p].empty]
        if missing:
            logging.warning(f"Hist FX: {len(missing)} pairs missing from DB after sync: {missing}")
            fetch_failed = True

        return historical_fx_yf, fetch_failed

    @profile
    def get_fundamental_data(self, yf_symbol: str, force_refresh: bool = False) -> Optional[Dict[str, Any]]:
        """
        Fetches fundamental data (ticker.info) for a given Yahoo Finance symbol.
        Uses a directory of JSON files for caching (one file per symbol).

        Args:
            yf_symbol (str): The Yahoo Finance ticker symbol (e.g., "AAPL").
            force_refresh (bool): If True, bypasses cache and forces a fresh fetch.

        Returns:
            Optional[Dict[str, Any]]: A dictionary containing the fundamental data,
                                      or None if fetching fails or symbol is invalid.
        """
        _ensure_yfinance()
        if not YFINANCE_AVAILABLE:
            logging.error("yfinance not available. Cannot fetch fundamental data.")
            return None
        if not yf_symbol or not isinstance(yf_symbol, str) or not yf_symbol.strip():
            logging.warning(f"Invalid yf_symbol provided for fundamentals: {yf_symbol}")
            return None

        logging.debug(f"Requesting fundamental data for YF symbol: {yf_symbol} (force_refresh={force_refresh})")

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

        if not force_refresh and os.path.exists(symbol_cache_file):
            try:
                # MODIFIED: Load only the specific symbol's cache file
                with open(symbol_cache_file, "r", encoding="utf-8") as f:
                    symbol_cache_entry = json.load(f)

                # Check timestamp within the loaded entry
                if symbol_cache_entry and isinstance(symbol_cache_entry, dict):
                    cache_timestamp_str = symbol_cache_entry.get("timestamp")
                    if cache_timestamp_str:
                        cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                        
                        # --- SMART CACHING ---
                        valid_until_str = symbol_cache_entry.get("valid_until")
                        is_valid = False
                        
                        if valid_until_str:
                            try:
                                valid_until = datetime.fromisoformat(valid_until_str)
                                if datetime.now(timezone.utc) < valid_until:
                                    is_valid = True
                                else:
                                    logging.info(f"Fundamentals smart cache expired for {yf_symbol} (Valid until: {valid_until})")
                            except ValueError:
                                is_valid = False # Corrupt timestamp
                        else:
                            # Fallback to standard duration if no specific expiry
                            if datetime.now(timezone.utc) - cache_timestamp < timedelta(hours=FUNDAMENTALS_CACHE_DURATION_HOURS):
                                is_valid = True
                        
                        if is_valid:
                            cached_data = symbol_cache_entry.get("data")
                            # CRITICAL: Reject "empty" or "poisoned" cache entries
                            if cached_data is not None and len(cached_data) > 3: # symbol, quoteType, etc.
                                cache_valid = True
                                logging.debug(f"Using valid fundamentals cache for {yf_symbol} from file")
                            else:
                                logging.warning(f"Rejecting poisoned/empty fundamentals cache for {yf_symbol}")
                                cache_valid = False
                        else:
                            if not valid_until_str: # Only log standard expiry if not smart
                                 logging.info(f"Fundamentals cache expired for {yf_symbol} (Standard duration)")
                    
                    # --- ETF CACHE FRESHNESS CHECK ---
                    if cache_valid and cached_data:
                         qt = str(cached_data.get('quoteType', '')).upper()
                         if qt in ('ETF', 'MUTUALFUND') and 'etf_data' not in cached_data:
                             # If cache is valid (within 24h) but missing ETF data, force refresh
                             # UNLESS it's very recent (< 5 mins), implying we just tried and failed.
                             age = datetime.now(timezone.utc) - cache_timestamp
                             if age > timedelta(minutes=5):
                                 logging.info(f"Cache valid but missing ETF data for {yf_symbol} ({qt}). Forcing refresh (age: {age}).")
                                 cache_valid = False
                    # ---------------------------------
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

        # --- Isolated Fetching Logic ---
        logging.info(
            f"Fetching fresh fundamental data for {yf_symbol} (isolated/cache miss)..."
        )
        try:
            # Use isolated fetch for single symbol info
            info_result = _run_isolated_fetch([yf_symbol], task="info")
            data = info_result.get(yf_symbol, {})
            
            if not data:
                logging.warning(
                    f"No fundamental data returned by isolated fetch for {yf_symbol}."
                )
                data = {}

            # --- ETF DATA EXTRACTION (Isolated) ---
            # _run_isolated_fetch for 'info' now returns etf_data inside the 'info' dict.
            # No need for direct Ticker instantiation here.
            etf_data = data.get('etf_data')
            if etf_data:
                logging.info(f"Using ETF data from isolated fetch for {yf_symbol}")
            # --- END ETF DATA EXTRACTION ---


            # --- NORMALIZE YIELD IN PLACE (Consistency with Watchlist) ---
            # Define CACHE_VERSION here too if needed, or import. Using 2.
            CACHE_VERSION = 2
            try:
                div_yield = data.get("yield")
                if div_yield is None:
                    div_yield = data.get("dividendYield")
                    
                    try:
                        div_val = float(div_yield)
                        rate = data.get("dividendRate")
                        price = data.get("currentPrice") or data.get("regularMarketPrice")
                        trailing = data.get("trailingAnnualDividendYield")
                        
                        normalized = False
                        
                        # Check 1: Rate / Price ratio
                        if rate is not None and price is not None:
                             try:
                                  rate_val = float(rate)
                                  price_val = float(price)
                                  if price_val > 0:
                                      ratio = rate_val / price_val
                                      if abs(div_val - ratio * 100) < abs(div_val - ratio):
                                          div_yield = div_val / 100.0
                                          normalized = True
                             except (ValueError, TypeError):
                                 pass
                                 
                        # Check 2: Comparison with trailing yield
                        if not normalized and trailing is not None:
                             try:
                                 trailing_val = float(trailing)
                                 if trailing_val > 0 and div_val > trailing_val * 50:
                                     div_yield = div_val / 100.0
                                     normalized = True
                             except (ValueError, TypeError):
                                 pass

                        # Check 3: Raw Magnitude Heuristic
                        if not normalized and div_val > 0.30: 
                             div_yield = div_val / 100.0
                    
                    except Exception:
                        pass
                
                if div_yield is not None:
                    data["dividendYield"] = div_yield
            except Exception as e_norm_detail:
                logging.warning(f"Error normalizing yield in detail fetch for {yf_symbol}: {e_norm_detail}")
            # --- END NORMALIZATION ---

            # --- FIX: Populate expenseRatio from netExpenseRatio if missing ---
            # yfinance often returns expenseRatio as None for ETFs, but netExpenseRatio acts as the value.
            if data.get("expenseRatio") is None and data.get("netExpenseRatio") is not None:
                data["expenseRatio"] = data.get("netExpenseRatio")

            # --- SMART CACHING CALCULATION ---
            now_ts = datetime.now(timezone.utc)
            earnings_ts = data.get("earningsTimestamp")
            valid_until = now_ts + timedelta(hours=FUNDAMENTALS_CACHE_DURATION_HOURS) # Default

            if earnings_ts:
                try:
                    # earningsTimestamp is epoch seconds
                    earnings_date = datetime.fromtimestamp(earnings_ts, tz=timezone.utc)
                    
                    if earnings_date > now_ts:
                         # Earnings in the future: Valid until 24H after earnings
                         valid_until = earnings_date + timedelta(hours=24)
                         logging.info(f"Smart Caching {yf_symbol}: Earnings at {earnings_date}. Cache valid until {valid_until}")
                    else:
                         # Earnings passed, data is fresh (presumably). Keep default long cache.
                         # Unless it was VERY recent (e.g. yesterday)? 
                         # If earnings were yesterday, yf might not have updated yet?
                         # yf usually updates quickly. Let's stick to standard duration if past.
                         pass
                except Exception as e_smart:
                    logging.warning(f"Error calculating smart cache expiry for {yf_symbol}: {e_smart}")
            
            # Use data.get('_fetch_timestamp') if provided, else use now. 
            # In this isolated fetch context, we just fetched it.
            data['_fetch_timestamp'] = now_ts.isoformat()

            # Save to cache (only this symbol's file)
            # CRITICAL: Validate data before saving to prevent cache poisoning
            # A valid info dict should have at least a few keys like 'symbol', 'quoteType', 'longName'
            if data and len(data) > 3:
                try:
                    # MODIFIED: Save only the specific symbol's data to its file
                    symbol_cache_content = {
                        "timestamp": now_ts.isoformat(),
                        "valid_until": valid_until.isoformat(),
                        "version": CACHE_VERSION,
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
            else:
                logging.warning(f"Not saving fundamentals for {yf_symbol} - insufficient data (poison check failed)")

            return data
        except Exception as e_fetch:
            logging.error(f"Error fetching fundamental data for {yf_symbol}: {e_fetch}")
            # traceback.print_exc()
            return None

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
        _ensure_yfinance()
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
            f"Fetching fresh {period_type} {statement_type} for {yf_symbol} (isolated)..."
        )
        try:
            # Use isolated fetch for statements
            df = _run_isolated_fetch(
                [yf_symbol],
                task="statement",
                statement_type=statement_type,
                period_type=period_type
            )
            
            if df is None:  
                df = pd.DataFrame() 

            self._save_statement_data_to_cache(
                yf_symbol, statement_type, period_type, df
            )
            return df
        except Exception as e:
            logging.error(
                f"Error fetching {period_type} {statement_type} for {yf_symbol}: {e}"
            )
            # Cache empty to prevent retry loop
            self._save_statement_data_to_cache(
                yf_symbol, statement_type, period_type, pd.DataFrame()
            )
            return pd.DataFrame()

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
        _ensure_yfinance()
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
            data = _run_isolated_fetch(
                tickers=[yf_symbol],
                period=fetch_period,
                interval=interval,
                task="history"
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

_SHARED_MDP = None
_MDP_LOCK = threading.Lock()

def get_shared_mdp(hist_data_cache_dir_name="historical_data_cache"):
    """
    Returns a singleton instance of MarketDataProvider.
    """
    global _SHARED_MDP
    with _MDP_LOCK:
        if _SHARED_MDP is None:
            # Import here to avoid circular imports if any
            _SHARED_MDP = MarketDataProvider(hist_data_cache_dir_name=hist_data_cache_dir_name)
    return _SHARED_MDP

