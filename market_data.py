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
        YFINANCE_INDEX_TICKER_MAP,
        DEFAULT_INDEX_QUERY_SYMBOLS,
        SYMBOL_MAP_TO_YFINANCE,
        YFINANCE_EXCLUDED_SYMBOLS,
        CASH_SYMBOL_CSV,
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
    SYMBOL_MAP_TO_YFINANCE = {}
    YFINANCE_EXCLUDED_SYMBOLS = set()
    CASH_SYMBOL_CSV = "$CASH"
    HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX = "yf_portfolio_hist_raw_adjusted"

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

 Author:        Kit Matan (Derived from portfolio_logic.py)
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
        hist_raw_cache_prefix=HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
        current_cache_file=DEFAULT_CURRENT_CACHE_FILE_PATH,  # <-- ADDED
    ):
        self.hist_raw_cache_prefix = hist_raw_cache_prefix
        self.current_cache_file = current_cache_file  # <-- ADDED
        self._session = None  # Initialize session attribute
        self.historical_fx_for_fallback: Dict[str, pd.DataFrame] = (
            {}
        )  # Store recently fetched historical FX
        logging.info("MarketDataProvider initialized.")

    def get_current_quotes(
        self,
        internal_stock_symbols: List[str],
        required_currencies: Set[str],
        manual_prices_dict: Optional[Dict[str, float]] = None,
    ) -> Tuple[Dict[str, Dict], Dict[str, float], bool, bool]:
        """
        Fetches current market quotes (price, change, currency) for given stock symbols
        and required FX rates against USD. Uses MarketDataProvider caching.

        Args:
            internal_stock_symbols (List[str]): List of internal stock symbols (e.g., 'AAPL', 'SET:BKK').
            required_currencies (Set[str]): Set of all currency codes needed (e.g., {'USD', 'THB', 'EUR'}).
            manual_prices_dict (Optional[Dict[str, float]]): Dictionary of manual price overrides.

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
            yf_symbol = map_to_yf_symbol(internal_symbol)
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
        cache_key = f"CURRENT_QUOTES_v2::{'_'.join(sorted(yf_symbols_to_fetch))}::{'_'.join(sorted(required_currencies))}"
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
                            )

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
                    time.sleep(0.2)  # Add small delay after each ticker info fetch

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
                                rate = (
                                    fx_info.get("currentPrice")
                                    or fx_info.get("regularMarketPrice")
                                    or fx_info.get(
                                        "regularMarketPreviousClose"
                                    )  # Added one more fallback
                                    or fx_info.get("previousClose")
                                )
                                currency_code = fx_info.get(
                                    "currency"
                                )  # Should be USD for =X pairs, but check

                                base_curr = yf_symbol.replace("=X", "").upper()
                                if (
                                    rate is not None
                                    and pd.notna(rate)
                                    and currency_code == "USD"
                                ):
                                    # Key is the base currency (e.g., EUR from EUR=X)
                                    fx_rates_vs_usd[base_curr] = float(rate)
                                else:
                                    logging.error(
                                        f"Could not get valid rate/currency for FX pair {yf_symbol} from info."
                                    )
                                    # Ensure the key exists, even if NaN, so fallback logic can target it
                                    if base_curr not in fx_rates_vs_usd:
                                        fx_rates_vs_usd[base_curr] = np.nan
                                    has_warnings = True
                            else:
                                logging.error(
                                    f"Could not get .info for FX pair {yf_symbol}"
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
                        time.sleep(0.05)  # Add small delay after each FX info fetch
                        time.sleep(0.2)  # Add small delay after each FX info fetch

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
                    end_date=end_fallback_date,
                    use_cache=True,
                    cache_file=f"{self.hist_raw_cache_prefix}_fallback_fx.json",
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
                    time.sleep(0.05)  # Add small delay after each index info fetch
                    time.sleep(0.1)  # Add small delay after each index info fetch

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

    # --- Historical Data Fetching ---
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
            time.sleep(0.2)  # Small delay between batches

        logging.info(
            f"Hist Fetch Helper: Finished fetching ({len(historical_data)} symbols successful)."
        )
        return historical_data

    # --- Historical Cache Helpers ---
    def _load_historical_cache(
        self, cache_file: str, cache_key: str
    ) -> Tuple[Optional[Dict], bool]:
        """Loads historical data (prices and FX) from a JSON cache file if the key matches."""
        if not cache_file or not os.path.exists(cache_file):
            return None, False  # Cache doesn't exist

        logging.info(f"Hist Cache: Attempting to load cache: {cache_file}")
        try:
            with open(cache_file, "r") as f:
                cached_data = json.load(f)
            if cached_data.get("cache_key") == cache_key:
                logging.info("Hist Cache: Cache key MATCH. Data loaded.")
                return cached_data, True  # Return loaded data and valid flag
            else:
                logging.warning("Hist Cache: Cache key MISMATCH. Ignoring cache.")
                return None, False
        except Exception as e:
            logging.error(
                f"Error reading hist cache {cache_file}: {e}. Ignoring cache."
            )
            return None, False

    def _save_historical_cache(
        self, cache_file: str, cache_key: str, data_to_cache: Dict
    ):
        """Saves historical data (prices and FX) to a JSON cache file."""
        if not cache_file or not cache_key or not data_to_cache:
            logging.error("Hist Cache: Invalid arguments for saving cache.")
            return

        logging.info(f"Hist Cache: Saving updated raw data to cache: {cache_file}")
        cache_content = {
            "cache_key": cache_key,
            "timestamp": datetime.now().isoformat(),
            **data_to_cache,  # Merge the data (e.g., {"historical_prices": ..., "historical_fx_rates": ...})
        }
        try:
            cache_dir = os.path.dirname(cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, "w") as f:
                json.dump(cache_content, f, indent=2)  # Use indent for readability
        except Exception as e:
            logging.error(f"Error writing hist cache: {e}")

    def _deserialize_historical_data(
        self, cached_data: Dict, key_name: str, expected_keys: List[str]
    ) -> Dict[str, pd.DataFrame]:
        """Deserializes cached historical data (prices or FX) from JSON strings."""
        deserialized_dict = {}
        data_section = cached_data.get(key_name, {})
        if not isinstance(data_section, dict):
            return {}  # Invalid section

        for expected_key in expected_keys:
            data_json = data_section.get(expected_key)
            df = pd.DataFrame()  # Default to empty
            if data_json:
                try:
                    df_temp = pd.read_json(
                        StringIO(data_json), orient="split", dtype={"price": float}
                    )
                    df_temp.index = pd.to_datetime(df_temp.index, errors="coerce").date
                    df_temp = df_temp.dropna(subset=["price"])
                    df_temp = df_temp[pd.notnull(df_temp.index)]
                    if not df_temp.empty:
                        df = df_temp.sort_index()
                except Exception as e_deser:
                    logging.debug(
                        f"DEBUG: Error deserializing cached {key_name} for {expected_key}: {e_deser}"
                    )
            deserialized_dict[expected_key] = df  # Store df (even if empty)
        return deserialized_dict

    # --- Public Historical Methods ---
    def get_historical_data(
        self,
        symbols_yf: List[str],
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        cache_key: Optional[str] = None,  # Key for validation
        cache_file: Optional[str] = None,  # Specific file path
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """
        Loads/fetches ADJUSTED historical price data using cache.
        (Replaces parts of _load_or_fetch_raw_historical_data related to price fetching)

        Args:
            symbols_yf (List[str]): List of YF stock/benchmark tickers required.
            start_date (date): Start date for historical data.
            end_date (date): End date for historical data.
            use_cache (bool): Flag to enable reading/writing the raw data cache.
            cache_file (str): Path to the raw historical data cache file.
            cache_key (str): Cache validation key generated by `_prepare_historical_inputs`.

        Returns:
            Tuple containing:
            - historical_prices_yf_adjusted (Dict[str, pd.DataFrame]): Dictionary mapping YF tickers
                to DataFrames containing adjusted historical prices (indexed by date).
            - fetch_failed (bool): True if fetching/loading critical data failed.
        """
        historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        cache_valid_raw = False
        fetch_failed = False
        cache_data_loaded = None

        # --- 1. Try Loading Cache ---
        if use_cache and cache_file and cache_key:
            cache_data_loaded, cache_valid_raw = self._load_historical_cache(
                cache_file, cache_key
            )
            if cache_valid_raw and cache_data_loaded:
                historical_prices_yf_adjusted = self._deserialize_historical_data(
                    cache_data_loaded, "historical_prices", symbols_yf
                )
                # Check if all requested symbols were actually loaded from cache
                if not all(s in historical_prices_yf_adjusted for s in symbols_yf):
                    logging.warning(
                        "Hist Cache: Price cache valid but incomplete. Will fetch missing."
                    )
                    cache_valid_raw = False  # Mark as invalid to trigger fetch
            elif cache_data_loaded is None and cache_valid_raw is False:
                logging.info(
                    "Hist Cache: Cache file not found or key mismatch."
                )  # Already logged by helper
            elif cache_valid_raw is True and cache_data_loaded is None:
                logging.error(
                    "Hist Cache: Logic error - cache marked valid but no data loaded."
                )
                cache_valid_raw = False  # Treat as invalid

        # --- 2. Fetch Missing Data if Cache Invalid/Incomplete ---
        if not cache_valid_raw:
            logging.info("Hist Prices: Fetching required data...")
            symbols_needing_fetch = [
                s
                for s in symbols_yf
                if s not in historical_prices_yf_adjusted
                or historical_prices_yf_adjusted[s].empty
            ]

            if symbols_needing_fetch:
                logging.info(
                    f"Hist Prices: Fetching {len(symbols_needing_fetch)} stock/benchmark symbols..."
                )
                fetched_stock_data = self._fetch_yf_historical_data(
                    symbols_needing_fetch, start_date, end_date
                )
                historical_prices_yf_adjusted.update(fetched_stock_data)
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
            if use_cache and cache_file and cache_key and symbols_needing_fetch:
                # Prepare data for saving (only prices for this method)
                prices_to_cache = {
                    symbol: df.to_json(orient="split", date_format="iso")
                    for symbol, df in historical_prices_yf_adjusted.items()
                    if not df.empty
                }
                # If cache was loaded partially, merge existing FX data
                existing_fx_data = {}
                if cache_data_loaded:
                    existing_fx_data = cache_data_loaded.get("historical_fx_rates", {})

                data_to_save = {
                    "historical_prices": prices_to_cache,
                    "historical_fx_rates": existing_fx_data,  # Preserve existing FX
                }
                self._save_historical_cache(cache_file, cache_key, data_to_save)

        # --- 4. Final Check and Return ---
        # Check if any requested symbol is still missing
        if any(
            s not in historical_prices_yf_adjusted
            or historical_prices_yf_adjusted[s].empty
            for s in symbols_yf
        ):
            # Log warning, but don't set fetch_failed=True unless caller deems it critical
            logging.warning(
                "Hist Prices WARN: Some requested symbols have no data after cache/fetch."
            )

        return (
            historical_prices_yf_adjusted,
            fetch_failed,
        )  # fetch_failed is currently always False here

    def get_historical_fx_rates(
        self,
        fx_pairs_yf: List[str],  # e.g., ['EUR=X', 'JPY=X']
        start_date: date,
        end_date: date,
        use_cache: bool = True,
        cache_key: Optional[str] = None,  # Key for validation
        cache_file: Optional[str] = None,  # Specific file path
    ) -> Tuple[Dict[str, pd.DataFrame], bool]:
        """
        Loads/fetches historical FX rates (vs USD) using cache.
        (Replaces parts of _load_or_fetch_raw_historical_data related to FX fetching)

        Args:
            fx_pairs_yf (List[str]): List of YF FX tickers (e.g., 'JPY=X') required.
            start_date (date): Start date for historical data.
            end_date (date): End date for historical data.
            use_cache (bool): Flag to enable reading/writing the raw data cache.
            cache_file (str): Path to the raw historical data cache file.
            cache_key (str): Cache validation key generated by `_prepare_historical_inputs`.

        Returns:
            Tuple containing:
            - historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
                to DataFrames containing historical rates vs USD (indexed by date).
            - fetch_failed (bool): True if fetching/loading ANY required FX rate failed.
        """
        historical_fx_yf: Dict[str, pd.DataFrame] = {}
        cache_valid_initially = False  # Flag for initial key match
        fetch_occurred = False  # Flag if we actually fetched new data
        fetch_failed = False
        cache_data_loaded = None
        fx_needing_fetch = []  # Initialize list of pairs to fetch

        # --- 1. Try Loading Cache ---
        if use_cache and cache_file and cache_key:
            cache_data_loaded, cache_valid_initially = self._load_historical_cache(
                cache_file, cache_key
            )
            if cache_valid_initially and cache_data_loaded:
                logging.info("Hist FX: Cache key matched. Deserializing FX data...")
                historical_fx_yf = self._deserialize_historical_data(
                    cache_data_loaded, "historical_fx_rates", fx_pairs_yf
                )
                # --- MODIFICATION START: Check completeness immediately after load ---
                fx_needing_fetch = [
                    p
                    for p in fx_pairs_yf
                    if p not in historical_fx_yf or historical_fx_yf[p].empty
                ]
                if fx_needing_fetch:
                    logging.warning(
                        f"Hist FX Cache: Cache valid but incomplete. Missing/empty FX pairs: {fx_needing_fetch}. Will fetch missing."
                    )
                    # Keep cache_valid_initially=True, but fx_needing_fetch will trigger fetch below
                else:
                    logging.info(
                        "Hist FX Cache: All required FX pairs found in valid cache."
                    )
                # --- MODIFICATION END ---
            elif cache_data_loaded is None and not cache_valid_initially:
                logging.info("Hist FX Cache: Cache file not found or key mismatch.")
                fx_needing_fetch = list(fx_pairs_yf)  # Need to fetch all
            elif cache_valid_initially and cache_data_loaded is None:
                logging.error(
                    "Hist FX Cache: Logic error - cache marked valid but no data loaded."
                )
                fx_needing_fetch = list(fx_pairs_yf)  # Need to fetch all
            else:  # Cache disabled or file/key invalid
                fx_needing_fetch = list(fx_pairs_yf)  # Need to fetch all
        else:  # Cache disabled or file/key invalid
            logging.info(
                "Hist FX Cache: Cache disabled or file/key not provided. Will fetch all."
            )
            fx_needing_fetch = list(fx_pairs_yf)  # Need to fetch all

        # --- 2. Fetch Missing Data if Needed ---
        # --- MODIFICATION: Trigger fetch if cache was initially invalid OR if pairs are missing ---
        if fx_needing_fetch:
            logging.info(
                f"Hist FX: Fetching {len(fx_needing_fetch)} required FX pairs..."
            )
            fetch_occurred = True  # Mark that we are attempting a fetch
            fetched_fx_data = self._fetch_yf_historical_data(
                fx_needing_fetch, start_date, end_date
            )
            historical_fx_yf.update(fetched_fx_data)  # Update dict with fetched data

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
            logging.info("Hist FX: No FX pairs needed fetching.")
        # --- END MODIFICATION ---

        # --- 3. Update Cache if Fetch Occurred and Cache Enabled ---
        if fetch_occurred and use_cache and cache_file and cache_key:
            # Prepare data for saving (only FX for this method)
            fx_to_cache = {
                pair: df.to_json(orient="split", date_format="iso")
                for pair, df in historical_fx_yf.items()
                if not df.empty
            }
            # If cache was loaded partially, merge existing price data
            existing_price_data = {}
            if cache_data_loaded:
                existing_price_data = cache_data_loaded.get("historical_prices", {})

            data_to_save = {
                "historical_prices": existing_price_data,  # Preserve existing prices
                "historical_fx_rates": fx_to_cache,
            }
            self._save_historical_cache(cache_file, cache_key, data_to_save)

        # --- 4. Final Check and Return ---
        # Check again if critical data is missing (redundant if fetch_failed already True, but safe)
        if not fetch_failed and fx_pairs_yf:  # Re-check only if not already failed
            if any(
                p not in historical_fx_yf or historical_fx_yf[p].empty
                for p in fx_pairs_yf
            ):
                logging.error(
                    "Hist FX ERROR: Critical FX data missing after final check."
                )
                fetch_failed = True

        return historical_fx_yf, fetch_failed


# --- END OF FILE market_data.py ---
