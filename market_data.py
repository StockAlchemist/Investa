# market_data.py
import pandas as pd
import numpy as np
import logging
import json
import os
from datetime import datetime, timedelta, date, UTC  # Added UTC
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
        current_cache_file: str = DEFAULT_CURRENT_CACHE_FILE_PATH,
        current_cache_duration_hours: int = YFINANCE_CACHE_DURATION_HOURS,
        hist_raw_cache_prefix: str = HISTORICAL_RAW_ADJUSTED_CACHE_PATH_PREFIX,
    ):
        """
        Initializes the provider with cache settings.

        Args:
            current_cache_file (str): Path for the current quotes cache file.
            current_cache_duration_hours (int): Max age for the current quotes cache.
            hist_raw_cache_prefix (str): Prefix for historical raw data cache files.
        """
        self.current_cache_file = current_cache_file
        self.current_cache_duration_hours = current_cache_duration_hours
        self.hist_raw_cache_prefix = hist_raw_cache_prefix
        logging.info(
            f"MarketDataProvider initialized. Current Cache: '{self.current_cache_file}', Hist Prefix: '{self.hist_raw_cache_prefix}'"
        )

    # --- Current Quotes and FX ---
    def get_current_quotes(
        self, internal_stock_symbols: List[str], required_currencies: Set[str]
    ) -> Tuple[
        Optional[Dict[str, Dict[str, Optional[float]]]],
        Optional[Dict[str, float]],
        bool,
        bool,
    ]:
        """
        Gets current stock/ETF price/change data and FX rates relative to USD, using cache.
        (Replaces get_cached_or_fetch_yfinance_data)

        Args:
            internal_stock_symbols (List[str]): List of internal stock/ETF symbols.
            required_currencies (Set[str]): Set of currency codes (e.g., 'USD', 'EUR') needed.

        Returns:
            Tuple containing:
            - Stock Data Dict (Optional[Dict[internal_symbol, Dict[metric, value]]]):
              Inner dict keys: 'price', 'change', 'changesPercentage', 'previousClose'. Values can be float or None.
            - FX Rates vs USD Dict (Optional[Dict[currency_code, rate]]): Rate is units of currency per 1 USD.
            - bool: has_errors - True if critical fetch/load errors occurred.
            - bool: has_warnings - True if recoverable issues occurred.
        """
        stock_data_internal: Optional[Dict[str, Dict[str, Optional[float]]]] = None
        fx_rates_vs_usd: Optional[Dict[str, float]] = {"USD": 1.0}
        cache_needs_update = True
        cached_data = {}
        now = datetime.now()
        has_errors = False
        has_warnings = False

        # --- Cache Loading Logic ---
        if not self.current_cache_file:
            logging.warning(
                "Warning: Invalid current cache file path. Cache read skipped."
            )
            has_warnings = True
            cache_needs_update = True
        elif os.path.exists(self.current_cache_file):
            try:
                with open(self.current_cache_file, "r") as f:
                    cached_data = json.load(f)
                cache_timestamp_str = cached_data.get("timestamp")
                cached_stocks = cached_data.get("stock_data_internal")
                cached_fx_vs_usd = cached_data.get("fx_rates_vs_usd")

                if (
                    cache_timestamp_str
                    and isinstance(cached_stocks, dict)
                    and isinstance(cached_fx_vs_usd, dict)
                ):
                    cache_timestamp = datetime.fromisoformat(cache_timestamp_str)
                    cache_age = now - cache_timestamp
                    if cache_age <= timedelta(hours=self.current_cache_duration_hours):
                        # Check completeness
                        all_stocks_present = all(
                            s in cached_stocks
                            for s in internal_stock_symbols
                            if s not in YFINANCE_EXCLUDED_SYMBOLS
                            and s != CASH_SYMBOL_CSV
                        )
                        stock_format_ok = (
                            all(
                                isinstance(v, dict) and "price" in v
                                for v in cached_stocks.values()
                            )
                            if cached_stocks
                            else True
                        )
                        all_fx_present = required_currencies.issubset(
                            cached_fx_vs_usd.keys()
                        )

                        if all_stocks_present and stock_format_ok and all_fx_present:
                            logging.info(
                                f"Current quotes cache is valid (age: {cache_age}). Using cached data."
                            )
                            stock_data_internal = cached_stocks
                            fx_rates_vs_usd = cached_fx_vs_usd
                            cache_needs_update = False
                        else:
                            logging.warning(
                                "Current quotes cache is recent but incomplete/invalid. Will fetch/update."
                            )
                            has_warnings = True
                            stock_data_internal = (
                                cached_stocks if stock_format_ok else {}
                            )
                            fx_rates_vs_usd = (
                                cached_fx_vs_usd
                                if isinstance(cached_fx_vs_usd, dict)
                                else {"USD": 1.0}
                            )
                            cache_needs_update = True
                    else:
                        logging.info(
                            f"Current quotes cache is outdated (age: {cache_age}). Will fetch fresh data."
                        )
                        cache_needs_update = True
                else:
                    logging.warning(
                        "Current quotes cache is invalid or missing data. Will fetch fresh data."
                    )
                    has_warnings = True
                    cache_needs_update = True
            except (json.JSONDecodeError, FileNotFoundError, Exception) as e:
                logging.warning(
                    f"Error reading current quotes cache file {self.current_cache_file}: {e}. Will fetch fresh data."
                )
                has_warnings = True
                cache_needs_update = True
                stock_data_internal = {}
                fx_rates_vs_usd = {"USD": 1.0}
        else:
            logging.info(
                f"Current quotes cache file {self.current_cache_file} not found. Will fetch fresh data."
            )
            cache_needs_update = True

        # --- Data Fetching Logic ---
        if cache_needs_update:
            if not YFINANCE_AVAILABLE:
                logging.error("Cannot fetch current quotes: yfinance not available.")
                return None, None, True, has_warnings  # Critical error

            logging.info(
                "Fetching/Updating current quotes from Yahoo Finance (Stocks & FX vs USD)..."
            )
            fetched_stocks = (
                stock_data_internal if stock_data_internal is not None else {}
            )
            # fx_rates_vs_usd is already initialized or loaded partially

            # Determine symbols/tickers to fetch
            yf_tickers_to_fetch = set()
            internal_to_yf_map = {}
            yf_to_internal_map = {}
            missing_stock_symbols = []
            explicitly_excluded_count = 0

            # Stocks/ETFs
            for internal_sym in internal_stock_symbols:
                if internal_sym == CASH_SYMBOL_CSV:
                    continue
                if internal_sym in YFINANCE_EXCLUDED_SYMBOLS:
                    if internal_sym not in fetched_stocks:
                        explicitly_excluded_count += 1
                        fetched_stocks[internal_sym] = {
                            "price": None,
                            "change": None,
                            "changesPercentage": None,
                            "previousClose": None,
                        }
                    continue

                yf_ticker = map_to_yf_symbol(internal_sym)  # Use the helper
                if yf_ticker:
                    should_fetch_stock = True
                    if internal_sym in fetched_stocks:
                        cached_entry = fetched_stocks[internal_sym]
                        # Check if price is valid (not None, not NaN)
                        if (
                            isinstance(cached_entry, dict)
                            and cached_entry.get("price") is not None
                            and pd.notna(cached_entry.get("price"))
                        ):
                            should_fetch_stock = False
                    if should_fetch_stock:
                        yf_tickers_to_fetch.add(yf_ticker)
                        internal_to_yf_map[internal_sym] = yf_ticker
                        yf_to_internal_map[yf_ticker] = internal_sym
                else:  # Invalid mapping/format
                    missing_stock_symbols.append(internal_sym)
                    if internal_sym not in fetched_stocks:
                        logging.warning(
                            f"Warning: No valid Yahoo Finance ticker mapping for: {internal_sym}."
                        )
                        has_warnings = True
                        fetched_stocks[internal_sym] = {
                            "price": None,
                            "change": None,
                            "changesPercentage": None,
                            "previousClose": None,
                        }

            if explicitly_excluded_count > 0:
                logging.info(
                    f"Info: Explicitly excluded {explicitly_excluded_count} stock symbols."
                )

            # FX Rates vs USD
            fx_tickers_to_fetch = set()
            currencies_to_fetch_vs_usd = required_currencies - set(
                fx_rates_vs_usd.keys()
            )
            for currency in currencies_to_fetch_vs_usd:
                if currency != "USD" and currency and isinstance(currency, str):
                    fx_ticker = f"{currency.upper()}=X"
                    fx_tickers_to_fetch.add(fx_ticker)
                    yf_tickers_to_fetch.add(fx_ticker)  # Add to the main fetch list

            # Fetching using yfinance
            if not yf_tickers_to_fetch:
                logging.info("No new Stock/FX tickers need fetching.")
            else:
                logging.info(
                    f"Fetching/Updating {len(yf_tickers_to_fetch)} tickers from Yahoo: {list(yf_tickers_to_fetch)}"
                )
                fetch_success = True
                all_fetched_data = {}
                try:
                    yf_ticker_list = list(yf_tickers_to_fetch)
                    fetch_batch_size = 50  # Keep batching
                    for i in range(0, len(yf_ticker_list), fetch_batch_size):
                        batch = yf_ticker_list[i : i + fetch_batch_size]
                        logging.info(
                            f"  Fetching batch {i//fetch_batch_size + 1} ({len(batch)} tickers)..."
                        )
                        try:
                            tickers_data = yf.Tickers(" ".join(batch))
                            for yf_ticker, ticker_obj in tickers_data.tickers.items():
                                ticker_info = getattr(ticker_obj, "info", None)
                                price, change, pct_change, prev_close = (
                                    None,
                                    None,
                                    None,
                                    None,
                                )
                                if ticker_info:
                                    price_raw = ticker_info.get(
                                        "regularMarketPrice",
                                        ticker_info.get("currentPrice"),
                                    )
                                    change_raw = ticker_info.get("regularMarketChange")
                                    pct_change_raw = ticker_info.get(
                                        "regularMarketChangePercent"
                                    )  # Fraction
                                    prev_close_raw = ticker_info.get("previousClose")
                                    try:
                                        price = (
                                            float(price_raw)
                                            if price_raw is not None
                                            else None
                                        )
                                    except (ValueError, TypeError):
                                        price = None
                                    try:
                                        change = (
                                            float(change_raw)
                                            if change_raw is not None
                                            else None
                                        )
                                    except (ValueError, TypeError):
                                        change = None
                                    try:
                                        pct_change = (
                                            float(pct_change_raw)
                                            if pct_change_raw is not None
                                            else None
                                        )  # Keep as fraction
                                    except (ValueError, TypeError):
                                        pct_change = None
                                    try:
                                        prev_close = (
                                            float(prev_close_raw)
                                            if prev_close_raw is not None
                                            else None
                                        )
                                    except (ValueError, TypeError):
                                        prev_close = None

                                    # Validate price
                                    if not (
                                        price is not None
                                        and pd.notna(price)
                                        and price > 1e-9
                                    ):
                                        price = None  # Treat invalid price as None

                                    # Convert pct_change fraction to percentage value for storage/consistency
                                    if pct_change is not None and pd.notna(pct_change):
                                        pct_change *= 100.0

                                all_fetched_data[yf_ticker] = {
                                    "price": price,
                                    "change": change,
                                    "changesPercentage": pct_change,  # Store percentage value
                                    "previousClose": prev_close,
                                }
                        except requests.exceptions.ConnectionError as conn_err:
                            logging.error(f"  NETWORK ERROR fetching batch: {conn_err}")
                            fetch_success = False
                            has_errors = True
                            break
                        except requests.exceptions.Timeout as timeout_err:
                            logging.warning(
                                f"  TIMEOUT ERROR fetching batch: {timeout_err}"
                            )
                            fetch_success = False
                            has_warnings = True
                        except requests.exceptions.HTTPError as http_err:
                            logging.error(f"  HTTP ERROR fetching batch: {http_err}")
                            fetch_success = False
                            has_errors = True
                        except Exception as yf_err:
                            logging.error(
                                f"  YFINANCE ERROR fetching batch info: {yf_err}"
                            )
                            fetch_success = False
                            has_warnings = True
                        time.sleep(0.1)  # Small delay between batches
                except Exception as e:
                    logging.exception("ERROR during Yahoo Finance fetch loop")
                    fetch_success = False
                    has_errors = True

                # Process fetched data
                # Stocks
                for yf_ticker, data_dict in all_fetched_data.items():
                    internal_sym = yf_to_internal_map.get(yf_ticker)
                    if internal_sym:
                        fetched_stocks[internal_sym] = data_dict
                # Ensure entries for all requested symbols
                for sym in internal_stock_symbols:
                    if (
                        sym != CASH_SYMBOL_CSV
                        and sym not in YFINANCE_EXCLUDED_SYMBOLS
                        and sym not in fetched_stocks
                    ):
                        if sym not in missing_stock_symbols:
                            logging.warning(
                                f"Warning: Data for {sym} still not found after fetch."
                            )
                            has_warnings = True
                        fetched_stocks[sym] = {
                            "price": None,
                            "change": None,
                            "changesPercentage": None,
                            "previousClose": None,
                        }
                # FX Rates vs USD
                for currency in currencies_to_fetch_vs_usd:
                    if currency == "USD":
                        continue
                    fx_ticker = f"{currency.upper()}=X"
                    fx_data_dict = all_fetched_data.get(fx_ticker)
                    price = (
                        fx_data_dict.get("price")
                        if isinstance(fx_data_dict, dict)
                        else None
                    )
                    if price is not None and pd.notna(price):
                        fx_rates_vs_usd[currency.upper()] = price
                    else:
                        logging.warning(
                            f"Warning: Failed to fetch/update FX rate for {fx_ticker}."
                        )
                        has_warnings = True
                        if currency.upper() not in fx_rates_vs_usd:
                            fx_rates_vs_usd[currency.upper()] = (
                                None  # Store None if fetch failed
                            )

            # Assign final results
            stock_data_internal = fetched_stocks

            # Save Cache
            if cache_needs_update and self.current_cache_file:
                logging.info(
                    f"Saving updated current quotes data to cache: {self.current_cache_file}"
                )
                if "USD" not in fx_rates_vs_usd:
                    fx_rates_vs_usd["USD"] = 1.0
                content = {
                    "timestamp": now.isoformat(),
                    "stock_data_internal": stock_data_internal,
                    "fx_rates_vs_usd": fx_rates_vs_usd,
                }
                try:
                    cache_dir = os.path.dirname(self.current_cache_file)
                    if cache_dir and not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    with open(self.current_cache_file, "w") as f:
                        json.dump(content, f, indent=4, cls=NpEncoder)  # Use NpEncoder
                except (TypeError, IOError, Exception) as e:
                    logging.error(
                        f"Error writing current quotes cache ('{self.current_cache_file}'): {e}"
                    )
                    # Don't mark as critical error, just failed to save cache

        # Final Checks
        if not isinstance(stock_data_internal, dict):
            stock_data_internal = None
            has_errors = True
        if not isinstance(fx_rates_vs_usd, dict):
            fx_rates_vs_usd = None
            has_errors = True
        if isinstance(fx_rates_vs_usd, dict) and "USD" not in fx_rates_vs_usd:
            fx_rates_vs_usd["USD"] = 1.0
        if stock_data_internal is not None:
            for sym in internal_stock_symbols:
                if (
                    sym != CASH_SYMBOL_CSV
                    and sym not in YFINANCE_EXCLUDED_SYMBOLS
                    and sym not in stock_data_internal
                ):
                    stock_data_internal[sym] = {
                        "price": None,
                        "change": None,
                        "changesPercentage": None,
                        "previousClose": None,
                    }
                    logging.warning(
                        f"Warning: Data for symbol {sym} missing in final check."
                    )
                    has_warnings = True

        # If fetch failed for critical data, mark as error
        if fx_rates_vs_usd is None or stock_data_internal is None:
            has_errors = True

        return stock_data_internal, fx_rates_vs_usd, has_errors, has_warnings

    # --- Index Quotes ---
    def get_index_quotes(
        self, query_symbols: List[str] = DEFAULT_INDEX_QUERY_SYMBOLS
    ) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Fetches near real-time quotes for specified INDEX symbols using yfinance.
        (Replaces fetch_index_quotes_yfinance)

        Args:
            query_symbols (List[str], optional): List of internal index symbols (e.g., '.DJI').

        Returns:
            Dict[str, Optional[Dict[str, Any]]]: Dict keyed by internal symbol. Value is dict
                with 'price', 'change', 'changesPercentage' (fraction), 'name', etc., or None on failure.
        """
        if not YFINANCE_AVAILABLE:
            logging.error("Cannot fetch index quotes: yfinance not available.")
            return {q_sym: None for q_sym in query_symbols}

        if not query_symbols:
            return {}
        yf_tickers_to_fetch = []
        yf_ticker_to_query_symbol_map = {}
        for q_sym in query_symbols:
            yf_ticker = YFINANCE_INDEX_TICKER_MAP.get(q_sym)
            if yf_ticker:
                if yf_ticker not in yf_ticker_to_query_symbol_map:
                    yf_tickers_to_fetch.append(yf_ticker)
                yf_ticker_to_query_symbol_map[yf_ticker] = q_sym
            else:
                logging.warning(f"Warn: No yfinance ticker mapping for index: {q_sym}")

        if not yf_tickers_to_fetch:
            return {q_sym: None for q_sym in query_symbols}

        results: Dict[str, Optional[Dict[str, Any]]] = {}
        logging.info(
            f"Fetching index quotes via yfinance for tickers: {', '.join(yf_tickers_to_fetch)}"
        )
        try:
            tickers_data = yf.Tickers(" ".join(yf_tickers_to_fetch))
            for yf_ticker, ticker_obj in tickers_data.tickers.items():
                original_query_symbol = yf_ticker_to_query_symbol_map.get(yf_ticker)
                if not original_query_symbol:
                    continue
                ticker_info = getattr(ticker_obj, "info", None)
                if ticker_info:
                    try:
                        price_raw = ticker_info.get(
                            "regularMarketPrice", ticker_info.get("currentPrice")
                        )
                        prev_close_raw = ticker_info.get("previousClose")
                        name = ticker_info.get(
                            "shortName",
                            ticker_info.get("longName", original_query_symbol),
                        )
                        change_val_raw = ticker_info.get("regularMarketChange")
                        change_pct_val_raw = ticker_info.get(
                            "regularMarketChangePercent"
                        )  # Fraction

                        price = float(price_raw) if price_raw is not None else None
                        prev_close = (
                            float(prev_close_raw)
                            if prev_close_raw is not None
                            else None
                        )
                        change = (
                            float(change_val_raw)
                            if change_val_raw is not None
                            else None
                        )
                        changesPercentage = (
                            float(change_pct_val_raw)
                            if change_pct_val_raw is not None
                            else None
                        )  # Keep fraction

                        # Calculate change if missing
                        if (
                            (change is None or pd.isna(change))
                            and (price is not None and pd.notna(price))
                            and (prev_close is not None and pd.notna(prev_close))
                        ):
                            change = price - prev_close

                        # Fallback for percentage ONLY if primary fetch failed AND change/prev_close are valid
                        if (
                            (changesPercentage is None or pd.isna(changesPercentage))
                            and (change is not None and pd.notna(change))
                            and (
                                prev_close is not None
                                and pd.notna(prev_close)
                                and prev_close != 0
                            )
                        ):
                            changesPercentage = change / prev_close  # Store as fraction

                        if price is not None and pd.notna(price):
                            results[original_query_symbol] = {
                                "price": price,
                                "change": change if pd.notna(change) else None,
                                "changesPercentage": (
                                    changesPercentage
                                    if pd.notna(changesPercentage)
                                    else None
                                ),  # Store fraction
                                "name": name,
                                "symbol": original_query_symbol,
                                "yf_ticker": yf_ticker,
                            }
                        else:
                            results[original_query_symbol] = None
                            logging.warning(
                                f"Warn: Index {original_query_symbol} ({yf_ticker}) missing price."
                            )
                    except (ValueError, TypeError, KeyError, AttributeError) as e:
                        results[original_query_symbol] = None
                        logging.warning(
                            f"Warn: Error parsing index {original_query_symbol} ({yf_ticker}): {e}"
                        )
                else:
                    results[original_query_symbol] = None
                    logging.warning(f"Warn: No yfinance info for index: {yf_ticker}")
        except Exception as e:
            logging.error(f"Error fetching yfinance index data: {e}")
            traceback.print_exc()
            results = {q_sym: None for q_sym in query_symbols}

        # Ensure all requested symbols have an entry (even if None)
        for q_sym in query_symbols:
            if q_sym not in results:
                results[q_sym] = None
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
