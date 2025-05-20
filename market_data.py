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
    from finutils import map_to_yf_symbol
except ImportError:
    logging.error(
        "CRITICAL: Could not import map_to_yf_symbol from finutils.py in market_data.py"
    )

    # Define a dummy function if needed
    def map_to_yf_symbol(s):
        return None


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

        logging.info("MarketDataProvider initialized.")

    def _get_historical_cache_dir(self) -> str:
        """Constructs and returns the full path to the historical data cache subdirectory."""
        cache_dir_base = QStandardPaths.writableLocation(QStandardPaths.CacheLocation)
        if cache_dir_base:
            app_specific_cache_dir = (
                cache_dir_base  # QStandardPaths.CacheLocation is already app-specific
            )
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
            Tuple[Dict[str, Dict], Dict[str, float], bool, bool]:
                - results (Dict[str, Dict]): Dictionary mapping *internal* stock symbols to quote data
                  (price, change, changesPercentage, currency, name, source, timestamp).
                - fx_rates_vs_usd (Dict[str, float]): Dictionary mapping currency codes to their rate vs USD.
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
        internal_to_yf_map_local = {}
        for internal_symbol in internal_stock_symbols:
            if internal_symbol == CASH_SYMBOL_CSV:
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

        if not yf_symbols_to_fetch:
            logging.info(f"No valid stock symbols provided for current quotes.")
            return {}, {}, has_errors, has_warnings  # Return empty if no symbols

        # --- 2. Caching Logic for Current Quotes ---
        cache_key = f"CURRENT_QUOTES_v3::{'_'.join(sorted(yf_symbols_to_fetch))}::{'_'.join(sorted(required_currencies))}"  # Cache key bumped for new fields
        cached_quotes = None
        cached_fx = None
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
                            if cached_quotes is not None and cached_fx is not None:
                                cache_valid = True
                                logging.info(
                                    f"Using valid cache for current quotes (Key: {cache_key[:30]}...)."
                                )
                        else:
                            logging.info("Current quotes cache expired.")
                    else:
                        logging.warning("Current quotes cache missing timestamp.")
                else:
                    logging.info("Current quotes cache key mismatch.")
            except (json.JSONDecodeError, IOError, Exception) as e:
                logging.warning(
                    f"Error reading current quotes cache: {e}. Will refetch."
                )

        if cache_valid and cached_quotes is not None and cached_fx is not None:
            # --- Populate results from cache ---
            # Ensure keys are correct format if needed (e.g., internal symbols for quotes)
            # The cache should ideally store quotes keyed by internal symbol already
            results = cached_quotes
            fx_rates_vs_usd = cached_fx
            cached_data_used = True  # Mark cache as used
            # --- End Populate results ---
        else:
            logging.info("Fetching fresh current quotes and FX rates...")
            # --- Fetching logic will run if cache is not valid ---
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
                        # Use .info as fast_info might miss crucial fields like currency
                        ticker_info = getattr(ticker_obj, "info", None)
                        if ticker_info:
                            price = (
                                ticker_info.get("currentPrice")
                                or ticker_info.get("regularMarketPrice")
                                or ticker_info.get("previousClose")
                            )
                            change = ticker_info.get(
                                "regularMarketChange"
                            )  # Absolute change
                            change_pct = ticker_info.get("regularMarketChangePercent")
                            currency = ticker_info.get("currency")
                            name = ticker_info.get("shortName") or ticker_info.get(
                                "longName"
                            )  # Parenthesis moved here
                            # --- ADDED: Fetch dividend data ---
                            trailing_annual_dividend_rate = ticker_info.get(
                                "trailingAnnualDividendRate"
                            ) or ticker_info.get(
                                "dividendRate"
                            )  # Latter is fallback
                            dividend_yield_on_current = ticker_info.get(
                                "dividendYield"
                            )  # This is a fraction, e.g. 0.02 for 2%

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
                                        "source": "yf_info",
                                        "timestamp": datetime.now(
                                            timezone.utc
                                        ).isoformat(),  # Add timestamp
                                        "trailingAnnualDividendRate": trailing_annual_dividend_rate,
                                        "dividendYield": dividend_yield_on_current,
                                    }
                                )
                            else:
                                logging.warning(
                                    f"Could not get sufficient quote data (price/currency) for {yf_symbol} from info."
                                )
                                has_warnings = True
                        else:
                            logging.warning(
                                f"Could not get .info for ticker {yf_symbol}"
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
                                    or fx_info.get(
                                        "regularMarketPreviousClose"
                                    )  # Added one more fallback
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
                                        logging.info(
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
                                        logging.info(
                                            f"Processed FX {yf_symbol} (Base quoted): {rate_float:.4f} {base_curr_from_symbol}/USD"
                                        )
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
                                    has_warnings = True
                            else:
                                # fx_info is None
                                logging.warning(
                                    f"Could not get .info for FX pair {yf_symbol}. Will attempt fallback."
                                )
                                base_curr_no_info = yf_symbol.replace("=X", "").upper()
                                if base_curr_no_info not in fx_rates_vs_usd:
                                    fx_rates_vs_usd[base_curr_no_info] = np.nan
                                has_warnings = True

                        except yf.exceptions.YFRateLimitError:
                            logging.error(
                                f"RATE LIMITED while fetching info for FX {yf_symbol}. Aborting FX fetch."
                            )
                            # Return any valid cached data if available, otherwise indicate error
                            return (
                                cached_quotes or {},
                                cached_fx or {},
                                True,
                                True,
                            )  # Error = True, Warning = True
                        except Exception as e_fx_ticker:
                            logging.error(
                                f"Error fetching info for FX pair {yf_symbol}: {e_fx_ticker}"
                            )
                        # time.sleep(0.05)  # Add small delay after each FX info fetch
                        # time.sleep(0.2)  # Add small delay after each FX info fetch

                except yf.exceptions.YFRateLimitError:
                    logging.error(
                        "RATE LIMITED during yf.Tickers() call for FX. Aborting FX fetch."
                    )
                    # Return any valid cached data if available, otherwise indicate error
                    return (
                        cached_quotes or {},
                        cached_fx or {},
                        True,
                        True,
                    )  # Error = True, Warning = True
                except Exception as e_fx:
                    logging.error(f"Error fetching FX rates batch: {e_fx}")
                    has_errors = True  # Treat batch FX errors as critical

            # Add USD rate (always 1.0)
            fx_rates_vs_usd["USD"] = 1.0

            # --- MOVED: Fallback for FX Rates using recent historical data ---
            # This now runs AFTER the primary attempt to fetch current FX rates.
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
            if not has_errors:  # Only cache if the fetch didn't have critical errors
                try:
                    cache_content = {
                        "cache_key": cache_key,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "quotes": results,  # Store quotes keyed by internal symbol
                        "fx_rates": fx_rates_vs_usd,
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
        return results, fx_rates_vs_usd, has_errors, has_warnings

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
                            logging.info("Index quotes cache expired.")
                    else:
                        logging.warning("Index quotes cache entry missing timestamp.")
                else:
                    logging.info("Index quotes cache key not found in file.")
            except (json.JSONDecodeError, IOError, Exception) as e:
                logging.warning(f"Error reading index quotes cache: {e}. Will refetch.")

        if cache_valid and cached_results is not None:
            results = cached_results
            cached_data_used = True
        else:
            logging.info("Fetching fresh index quotes...")
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
                            name = ticker_info.get("shortName") or ticker_info.get(
                                "longName"
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
        fetch_batch_size = 50
        symbols_processed = 0

        for i in range(0, len(symbols_yf), fetch_batch_size):
            batch_symbols = symbols_yf[i : i + fetch_batch_size]
            try:
                data = yf.download(
                    tickers=batch_symbols,
                    start=yf_start_date,
                    end=yf_end_date,
                    progress=False,
                    group_by="ticker",
                    auto_adjust=True,  # Get adjusted prices
                    actions=False,
                )
                if data.empty:
                    logging.warning(
                        f"  Hist Fetch Helper WARN: No data returned for batch: {', '.join(batch_symbols)}"
                    )
                    continue

                for symbol in batch_symbols:
                    df_symbol = None
                    try:
                        if len(batch_symbols) == 1 and not data.columns.nlevels > 1:
                            df_symbol = data if not data.empty else None
                        elif symbol in data.columns.levels[0]:
                            df_symbol = data[symbol]
                        elif len(batch_symbols) > 1 and not data.columns.nlevels > 1:
                            logging.warning(
                                f"  Hist Fetch Helper WARN: Unexpected flat DataFrame structure for multi-ticker batch. Symbol {symbol} might be missing."
                            )
                            continue
                        else:  # Handle other potential structures (less common with group_by='ticker')
                            if isinstance(data, pd.Series) and data.name == symbol:
                                df_symbol = pd.DataFrame(data)
                            elif (
                                isinstance(data, pd.DataFrame)
                                and symbol in data.columns
                            ):
                                df_symbol = data[[symbol]].rename(
                                    columns={symbol: "Close"}
                                )
                            else:
                                logging.warning(
                                    f"  Hist Fetch Helper WARN: Symbol {symbol} not found in download results for this batch."
                                )
                                continue

                        if df_symbol is None or df_symbol.empty:
                            continue

                        price_col = "Close"  # Adjusted close price
                        if price_col not in df_symbol.columns:
                            logging.warning(
                                f"  Hist Fetch Helper WARN: Expected 'Close' column not found for {symbol}. Columns: {df_symbol.columns}"
                            )
                            continue

                        df_filtered = df_symbol[[price_col]].copy()
                        df_filtered.rename(columns={price_col: "price"}, inplace=True)
                        df_filtered.index = pd.to_datetime(
                            df_filtered.index
                        ).date  # Use date objects
                        df_filtered["price"] = pd.to_numeric(
                            df_filtered["price"], errors="coerce"
                        )
                        df_filtered = df_filtered.dropna(subset=["price"])
                        df_filtered = df_filtered[
                            df_filtered["price"] > 1e-6
                        ]  # Remove zero/neg prices

                        if not df_filtered.empty:
                            historical_data[symbol] = df_filtered.sort_index()

                    except Exception as e_sym:
                        logging.warning(
                            f"  Hist Fetch Helper ERROR processing symbol {symbol} within batch: {e_sym}"
                        )

            except Exception as e_batch:
                logging.warning(
                    f"  Hist Fetch Helper ERROR during yf.download for batch starting with {batch_symbols[0]}: {e_batch}"
                )

            symbols_processed += len(batch_symbols)
            # time.sleep(0.2)  # Small delay between batches

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

        manifest = {}
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception as e:
            logging.error(
                f"Hist Cache Load ({data_type}): Error reading manifest '{manifest_path}': {e}. Ignoring cache."
            )
            return loaded_symbol_data, False

        loaded_manifest_cache_key = manifest.get("cache_key")
        logging.info(
            f"Hist Cache Load ({data_type}): Manifest exists. Found Key='{str(loaded_manifest_cache_key)[:50]}...'"
        )

        if loaded_manifest_cache_key != expected_cache_key:
            logging.info(
                f"Hist Cache Load ({data_type}): Manifest MISS (key MISMATCH). Ignoring cache."
            )
            return loaded_symbol_data, False

        logging.info(
            f"Hist Cache Load ({data_type}): Manifest HIT (key MATCH). Loading individual symbol files..."
        )
        manifest_data_section = manifest.get(manifest_data_section_key, {})
        all_symbols_found_and_loaded = True

        for yf_symbol in symbols_to_load:
            if (
                yf_symbol in manifest_data_section
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
                    f"Hist Cache Load ({data_type}): Symbol {yf_symbol} not listed in manifest's '{manifest_data_section_key}' section."
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
        ],  # Keyed by symbol, value is DataFrame
        data_type: str = "price",  # "price" or "fx"
        existing_other_type_data_from_manifest: Optional[
            Dict
        ] = None,  # e.g., existing FX data if saving prices
    ):
        """Saves individual symbol historical data files and updates the manifest.json."""
        manifest_path = self._get_historical_manifest_path()
        manifest_data_section_key = (
            "historical_prices" if data_type == "price" else "historical_fx_rates"
        )
        other_manifest_data_section_key = (
            "historical_fx_rates" if data_type == "price" else "historical_prices"
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

        # Update manifest
        manifest_content = {
            "cache_key": cache_key_to_save,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            manifest_data_section_key: current_manifest_symbol_entries,
            # Preserve the other data type if it was passed
            other_manifest_data_section_key: existing_other_type_data_from_manifest
            or {},
        }

        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_content, f, indent=2)
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
            logging.info("Hist Prices: Fetching required data...")
            symbols_needing_fetch = [
                s
                for s in symbols_yf
                if s not in historical_prices_yf_adjusted
                or historical_prices_yf_adjusted[s].empty
            ]

            if (
                symbols_needing_fetch
            ):  # Only fetch if there are symbols actually needing it
                logging.info(
                    f"Hist Prices: Fetching {len(symbols_needing_fetch)} stock/benchmark symbols..."
                )
                fetched_stock_data = self._fetch_yf_historical_data(
                    symbols_needing_fetch, start_date, end_date
                )
                historical_prices_yf_adjusted.update(fetched_stock_data)
                logging.info(
                    f"Hist Prices: Fetch completed. Total series in memory now: {len(historical_prices_yf_adjusted)}."
                )
            else:
                logging.info(
                    "Hist Prices: All stock/benchmark symbols found in cache or not needed."
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
            if (
                use_cache and cache_key
            ):  # Always try to save if cache is enabled and key exists
                # We need to load existing FX data from manifest if we are to preserve it
                # For simplicity now, if we fetched prices, we assume FX might also be refetched by its own call.
                # A more robust solution would merge, but let's keep it focused.
                # If this call is ONLY for prices, we can load the FX part of the manifest to preserve it.
                existing_fx_from_manifest = {}
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as f_m:
                            m_data = json.load(f_m)
                            if (
                                m_data.get("cache_key") == cache_key
                            ):  # Only use if key matches
                                existing_fx_from_manifest = m_data.get(
                                    "historical_fx_rates", {}
                                )
                    except:
                        pass

                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=historical_prices_yf_adjusted,  # Save all currently held price data
                    data_type="price",
                    existing_other_type_data_from_manifest=existing_fx_from_manifest,
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
            fx_needing_fetch = [
                s
                for s in fx_pairs_yf
                if s not in historical_fx_yf or historical_fx_yf[s].empty
            ]
            if fx_needing_fetch:  # Only fetch if there are symbols actually needing it
                logging.info(
                    f"Hist FX: Fetching {len(fx_needing_fetch)} required FX pairs..."
                )
                fetched_fx_data = self._fetch_yf_historical_data(
                    fx_needing_fetch, start_date, end_date
                )
                historical_fx_yf.update(
                    fetched_fx_data
                )  # Update dict with fetched data
                logging.info(
                    f"Hist FX: Fetch completed. Total FX series in memory now: {len(historical_fx_yf)}."
                )

                # --- Validation after fetch ---
                final_fx_missing_after_fetch = [
                    p
                    for p in fx_needing_fetch  # Only check pairs we tried to fetch
                    if p not in historical_fx_yf or historical_fx_yf[p].empty
                ]
                if final_fx_missing_after_fetch:
                    logging.error(  # Log as error because fetch failed for required pairs
                        f"Hist FX ERROR: Failed to fetch required FX rates for: {', '.join(final_fx_missing_after_fetch)}"
                    )
                    fetch_failed = True  # Missing ANY required FX is critical
            else:
                logging.info(
                    "Hist FX: No FX pairs needed fetching (all were present or cache was complete)."
                )

            # --- 3. Update Cache if Fetch Occurred (or if cache was incomplete) and Cache Enabled ---
            if (
                use_cache and cache_key
            ):  # Always try to save if cache is enabled and key exists
                existing_prices_from_manifest = {}
                if os.path.exists(manifest_path):
                    try:
                        with open(manifest_path, "r") as f_m:
                            m_data = json.load(f_m)
                            if (
                                m_data.get("cache_key") == cache_key
                            ):  # Only use if key matches
                                existing_prices_from_manifest = m_data.get(
                                    "historical_prices", {}
                                )
                    except:
                        pass

                self._save_historical_data_and_manifest(
                    cache_key_to_save=cache_key,
                    data_to_save_map=historical_fx_yf,  # Save all currently held FX data
                    data_type="fx",
                    existing_other_type_data_from_manifest=existing_prices_from_manifest,
                )

        # --- 4. Final Check and Return ---
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
            return None


# --- END OF FILE market_data.py ---
