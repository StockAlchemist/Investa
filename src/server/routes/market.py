"""Market data routes: quotes, search, news, history, fundamentals, valuation."""

import asyncio
import json
import logging
import os
import re
import sqlite3
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from config_manager import ConfigManager
from db_utils import get_cached_screener_results, update_intrinsic_value_in_cache
from finutils import is_cash_symbol
from market_data import map_to_yf_symbol
from server.ai_analyzer import generate_stock_review
from server.dependencies import get_config_manager, get_transaction_data, get_user_db_connection
from server.route_utils import SWRCache, _lru_get, _lru_put, clean_nans, get_mdp
from utils_time import get_est_today, is_market_open

try:
    from financial_ratios import (
        calculate_key_ratios_timeseries,
        calculate_current_valuation_ratios,
        get_comprehensive_intrinsic_value,
        get_intrinsic_value_for_symbol
    )
    FINANCIAL_RATIOS_AVAILABLE = True
except ImportError:
    logging.warning("financial_ratios.py not found or import failed. Ratios will be disabled.")
    FINANCIAL_RATIOS_AVAILABLE = False

router = APIRouter()

# Short-TTL LRU cache for /market_history responses.
_MARKET_HISTORY_CACHE: OrderedDict = OrderedDict()


def clear_market_history_cache():
    _MARKET_HISTORY_CACHE.clear()


@router.get("/market_status")
def get_market_status():
    """
    Returns whether the US stock market is currently open.
    """
    return {"is_open": is_market_open()}


@router.get("/indices")
async def get_indices():
    """
    Current quotes for the header indices (Dow / Nasdaq / S&P).

    Served off the /summary critical path so portfolio totals render immediately.
    The underlying yfinance fetch is cached and run in a worker thread so a slow
    upstream call never blocks the event loop (and thus other requests).
    """
    try:
        mdp = get_mdp()
        data = await asyncio.to_thread(
            mdp.get_index_quotes, config.INDICES_FOR_HEADER
        )
        return data or {}
    except Exception as e:
        logging.warning(f"Failed to fetch index quotes: {e}")
        return {}


@router.get("/search")
def search_symbols(q: str = Query("", min_length=1)):
    """Symbol / name autocomplete using yfinance Search."""
    try:
        import yfinance as yf
        results = yf.Search(q, max_results=8).quotes
        out = []
        for r in results:
            symbol = r.get("symbol") or ""
            name = r.get("shortname") or r.get("longname") or ""
            kind = r.get("typeDisp") or r.get("quoteType") or ""
            if symbol:
                out.append({"symbol": symbol, "name": name, "type": kind})
        return out
    except Exception:
        return []


@router.get("/markets/news")
def get_market_news(
    limit: int = Query(20, ge=1, le=50),
    symbols: Optional[str] = Query(None),
):
    """Fetch latest market news.
    When symbols are provided, uses yfinance Search + relatedTickers filtering so
    only articles explicitly tagged to that ticker are returned.
    When no symbols are provided, returns general market news via the SPY RSS feed."""
    import yfinance as yf
    import urllib.request
    import xml.etree.ElementTree as ET
    from email.utils import parsedate_to_datetime
    from datetime import datetime, timezone

    def _search_news(symbol: str, fetch_count: int = 30) -> list:
        """Fetch news via yf.Search and keep only articles tagged to this symbol."""
        try:
            raw = yf.Search(symbol, news_count=fetch_count, max_results=0).news or []
        except Exception:
            return []
        sym_upper = symbol.upper()
        out = []
        for n in raw:
            related = [t.upper() for t in n.get("relatedTickers", [])]
            if sym_upper not in related:
                continue
            title = n.get("title", "").strip()
            if not title:
                continue
            # Thumbnail: pick smallest resolution ≥ 100px wide, or first available
            thumb = None
            resolutions = (n.get("thumbnail") or {}).get("resolutions", [])
            for r in resolutions:
                if r.get("width", 0) >= 100:
                    thumb = r.get("url")
                    break
            if not thumb and resolutions:
                thumb = resolutions[0].get("url")
            # Convert unix timestamp to ISO-8601
            ts = n.get("providerPublishTime", 0)
            pub_date = (
                datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
                if ts else ""
            )
            out.append({
                "title": title,
                "summary": "",
                "url": n.get("link", ""),
                "thumbnail": thumb,
                "provider": n.get("publisher", ""),
                "pub_date": pub_date,
                "symbol": symbol,
            })
        return out

    def _fetch_rss(symbol: str, rss_limit: int = 20) -> list:
        """Fallback: Yahoo Finance RSS for general / SPY market news."""
        url = (
            f"https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={symbol}&region=US&lang=en-US"
        )
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=6) as resp:
                xml_data = resp.read()
            root = ET.fromstring(xml_data)
            out = []
            for item in root.findall(".//item")[:rss_limit]:
                title = item.findtext("title", "").strip()
                if not title:
                    continue
                link = item.findtext("link", "").strip()
                pub_date_str = item.findtext("pubDate", "")
                pub_date = ""
                if pub_date_str:
                    try:
                        pub_date = parsedate_to_datetime(pub_date_str).isoformat()
                    except Exception:
                        pub_date = pub_date_str
                out.append({
                    "title": title,
                    "summary": item.findtext("description", "").strip(),
                    "url": link,
                    "thumbnail": None,
                    "provider": "Yahoo Finance",
                    "pub_date": pub_date,
                    "symbol": symbol,
                })
            return out
        except Exception:
            return []

    if symbols:
        symbol_list = [s.strip().upper() for s in symbols.split(",") if s.strip()][:20]
        seen: set = set()
        all_news: list = []
        for sym in symbol_list:
            for item in _search_news(sym, fetch_count=30):
                if item["title"] not in seen:
                    seen.add(item["title"])
                    all_news.append(item)
        all_news.sort(key=lambda x: x.get("pub_date") or "", reverse=True)
        return all_news[:limit]
    else:
        return _fetch_rss("SPY", limit)


@router.get("/market_history")
def get_market_history(
    benchmarks: List[str] = Query(...),
    period: str = "1y",
    interval: str = "1d",
    currency: str = "USD",
):
    """
    Returns historical return % for given market indices/benchmarks.
    """
    # 0. Global Cache Check
    cache_key = (tuple(sorted(benchmarks)), period, interval, currency)
    now_ts_cache = time.time()
    cached_mh = _lru_get(_MARKET_HISTORY_CACHE, cache_key)
    if cached_mh is not None:
        entry, expiry = cached_mh
        if now_ts_cache < expiry:
            logging.info(f"Market History Cache HIT: {cache_key}")
            return entry

    try:
        from utils_time import get_est_today, get_latest_trading_date
        
        # MAPPING: Convert benchmark display names to YF tickers (reuse logic from history)
        mapped_benchmarks = []
        ticker_to_name = {}
        bm_mapping_lower = {k.lower(): v for k, v in config.BENCHMARK_MAPPING.items()}
        yf_map_lower = {k.lower(): v for k, v in config.YFINANCE_INDEX_TICKER_MAP.items()}
        
        for b in benchmarks:
            b_lower = b.lower()
            if b_lower in bm_mapping_lower:
                ticker = bm_mapping_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            elif b_lower in yf_map_lower:
                ticker = yf_map_lower[b_lower]
                mapped_benchmarks.append(ticker)
                ticker_to_name[ticker] = b
            else:
                mapped_benchmarks.append(b)
        
        # Determine date range (simplified logic from history)
        end_date = get_est_today() + timedelta(days=1)
        if period == "1d":
            interval = "2m" # Force Intraday
            start_date = get_latest_trading_date()
        elif period == "5d" or period == "7d":
            start_date = end_date - timedelta(days=7)
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "3y":
            start_date = end_date - timedelta(days=365 * 3)
        elif period == "5y":
            start_date = end_date - timedelta(days=365 * 5)
        elif period == "10y":
            start_date = end_date - timedelta(days=365 * 10)
        elif period == "ytd":
            start_date = date(end_date.year, 1, 1)
        elif period == "all" or period == "max":
            start_date = date(1980, 1, 1) # Return full history
        else:
            start_date = end_date - timedelta(days=365)

        mdp = get_mdp()
        hist_data, _ = mdp.get_historical_data(
            symbols_yf=mapped_benchmarks,
            start_date=start_date,
            end_date=end_date,
            interval=interval
        )
        
        if not hist_data:
            return []

        # Process and normalize data (Return %) using Vectorized Pandas logic
        # Result should be a list of dicts: [{date: '...', '^GSPC': 0.1, ...}, ...]
        dfs = []
        for ticker in mapped_benchmarks:
            if ticker in hist_data and not hist_data[ticker].empty:
                df = hist_data[ticker][['price']].copy()
                # Normalize returns relative to first point
                first_price = df['price'].iloc[0]
                display_name = ticker_to_name.get(ticker, ticker)
                if first_price != 0:
                    df[display_name] = (df['price'] / first_price - 1) * 100
                else:
                    df[display_name] = 0.0
                
                # Keep the price column but rename it so it doesn't conflict
                df[f"{display_name}_price"] = df['price']
                
                dfs.append(df.drop(columns=['price']))
        
        if not dfs:
            return []
            
        # Combine all indices into one DataFrame
        combined_df = pd.concat(dfs, axis=1)
        combined_df = combined_df.sort_index()
        
        # Reset index to get dates as a column
        combined_df.index.name = 'date'
        combined_df = combined_df.reset_index()
        
        # Convert dates to string (using pd.to_datetime to be safe if types differ)
        combined_df['date'] = pd.to_datetime(combined_df['date'])
        
        is_intraday_local = any(x in interval for x in ["m", "h"])
        if is_intraday_local:
             combined_df['date'] = combined_df['date'].dt.strftime("%Y-%m-%d %H:%M:%S")
        else:
             combined_df['date'] = combined_df['date'].dt.strftime("%Y-%m-%d")

        result = clean_nans(combined_df.to_dict(orient="records"))
        
        _lru_put(_MARKET_HISTORY_CACHE, cache_key, (result, now_ts_cache + 900))
        
        return result

    except Exception as e:
        logging.error(f"Error in get_market_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logging.error(f"Error in get_market_history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock_history/{symbol}")
def get_stock_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    benchmarks: Optional[List[str]] = Query(None),
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns historical price data for a single stock, with optional benchmarks.
    """
    try:
        _, _, user_symbol_map, user_excluded_symbols, _, _, _, _ = data
        mdp = get_mdp()
        
        # 1. Map Symbol
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
        if not yf_symbol:
            # Try as direct ticker if not found in map (e.g. for benchmarks or unheld stocks)
            yf_symbol = symbol 

        # 2. Map Benchmarks
        mapped_benchmarks = []
        if benchmarks:
            for b in benchmarks:
                if b in config.BENCHMARK_MAPPING:
                    mapped_benchmarks.append(config.BENCHMARK_MAPPING[b])
                else:
                    mapped_benchmarks.append(b)
        
        # 3. Determine Date Range
        # Using helper logic similar to _calculate_historical_performance_internal
        from utils_time import get_est_today, get_latest_trading_date
        
        # End Date: Today + 1 (exclusive) to ensure we get today's data
        end_date = get_est_today() + timedelta(days=1)
        
        # Start Date
        if period == "1d":
            interval = "2m" # Force Intraday
            # For 1D, we want the last trading session.
            # If today is trading, get today. If weekend, get Friday.
            latest_trading = get_latest_trading_date()
            start_date = latest_trading
            # For intraday '1d', end_date logic in market_data handles the "up to now"
            # But let's be explicit: start at latest_trading, end at latest_trading + 1
            end_date = latest_trading + timedelta(days=1)
        elif period == "5d":
            start_date = end_date - timedelta(days=7) # Go back a week to cover 5 trading days
            interval = "15m" # Higher res for 5d
        elif period == "1m":
            start_date = end_date - timedelta(days=30)
            interval = "1d" # Daily is fine, or 60m? Daily is standard for 1M.
        elif period == "3m":
            start_date = end_date - timedelta(days=90)
        elif period == "6m":
            start_date = end_date - timedelta(days=180)
        elif period == "1y":
            start_date = end_date - timedelta(days=365)
        elif period == "3y":
            start_date = end_date - timedelta(days=365*3)
        elif period == "5y":
            start_date = end_date - timedelta(days=365*5)
        elif period == "10y":
            start_date = end_date - timedelta(days=365*10)
        elif period == "ytd":
            # start_date = date(end_date.year, 1, 1) 
            # Better YTD: Start of current year
            today = get_est_today()
            start_date = date(today.year, 1, 1)
        elif period == "max" or period == "all":
             # Arbitrary long history
             start_date = date(1980, 1, 1)
        else:
            # Default to 1y if unknown
            start_date = end_date - timedelta(days=365)

        # 4. Fetch Data (Main Symbol + Benchmarks)
        symbols_to_fetch = [yf_symbol] + mapped_benchmarks
        
        # Use get_historical_data (handles DB sync and cache)
        # Note: For intraday (1m, 5m etc), get_historical_data bypasses DB write usually/reads from special table
        # We need to make sure interval is passed correct.
        
        hist_data, _ = mdp.get_historical_data(
            symbols_to_fetch,
            start_date,
            end_date,
            interval=interval
        )
        
        if yf_symbol not in hist_data or hist_data[yf_symbol].empty:
            # Fallback: Maybe it's a crypto or something that failed mapping?
            # Or just no data.
            return []

        # 5. Align and Process
        # We want a single list of dicts: { date, price, volume, bm1, bm2... }
        # Merge on index (Date)
        
        main_df = hist_data[yf_symbol].copy()
        main_df.rename(columns={"price": "value", "Volume": "volume"}, inplace=True)
        
        if "value" not in main_df.columns:
            # Fallback if rename failed or price missing
            if "price" in main_df.columns:
                main_df["value"] = main_df["price"]
            else:
                # Last resort: use first column
                if not main_df.empty:
                    main_df["value"] = main_df.iloc[:, 0]
                else:
                    return [] # Should be caught above

        if "volume" not in main_df.columns:
            main_df["volume"] = 0.0

        # --- NEW: Filter Intraday Data (Market Hours Only: 09:30 - 16:00 EST) ---
        if period in ["1d", "5d"] and not main_df.empty:
            try:
                # Ensure index is timezone-aware. yfinance usually returns tz-aware (America/New_York) for intraday.
                if main_df.index.tz is None:
                    main_df.index = main_df.index.tz_localize("America/New_York", ambiguous='infer')
                else:
                    main_df.index = main_df.index.tz_convert("America/New_York")
                
                # Filter strictly between 09:30 and 16:00
                main_df = main_df.between_time("09:30", "16:00")
            except Exception as e:
                logging.warning(f"Error filtering market hours for {symbol}: {e}")

        # Calculate Return % (Normalized to start)
        if not main_df.empty and "value" in main_df.columns:
            first_val = main_df["value"].iloc[0]
            if first_val and first_val > 0:
                main_df["return_pct"] = (main_df["value"] / first_val - 1) * 100
            else:
                main_df["return_pct"] = 0.0
                
        # Join Benchmarks
        cols_to_keep = ["value", "volume", "return_pct"]
        result_df = main_df[cols_to_keep].copy()
        
        for bm in mapped_benchmarks:
            if bm in hist_data and not hist_data[bm].empty:
                bm_df = hist_data[bm].copy()
                # Normalize benchmark
                if "price" in bm_df.columns:
                    bm_start = bm_df["price"].iloc[0]
                    if bm_start and bm_start > 0:
                         bm_series = (bm_df["price"] / bm_start - 1) * 100
                    else:
                         bm_series = 0.0
                         
                    # Reindex to match main_df (ffill for missing days if mismatched trading cals)
                    aligned_bm = bm_series.reindex(result_df.index, method='ffill')
                    result_df[bm] = aligned_bm

        # 6. Format for JSON
        # Reset index to get Date/timestamp column
        result_df = result_df.reset_index()
        # Rename index col to 'date' usually
        date_col = "date" if "date" in result_df.columns else ("Date" if "Date" in result_df.columns else "index")
        result_df.rename(columns={date_col: "date"}, inplace=True)
        
        # Convert date to isoformat
        result_df["date"] = result_df["date"].apply(lambda x: x.isoformat())
        
        records = result_df.to_dict(orient="records")
        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error serving stock history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/earnings_dates/{symbol}")
def get_earnings_dates(
    symbol: str,
    limit: int = 24,
    data: tuple = Depends(get_transaction_data)
):
    """
    Returns historical (and upcoming) earnings report dates for a single stock,
    used to overlay earnings markers on the price chart.
    """
    from market_data import _run_isolated_fetch

    try:
        _, _, user_symbol_map, user_excluded_symbols, _, _, _, _ = data
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols) or symbol

        df = _run_isolated_fetch([yf_symbol], task="earnings_dates", limit=limit)
        if df is None or getattr(df, "empty", True):
            return []

        df = df.reset_index()
        # The earnings datetime column is named 'date' by the worker; fall back defensively.
        date_col = "date" if "date" in df.columns else df.columns[0]

        # Map yfinance's verbose column names to a stable shape.
        col_map = {
            "EPS Estimate": "eps_estimate",
            "Reported EPS": "eps_actual",
            "Surprise(%)": "surprise_pct",
        }

        records = []
        for _, row in df.iterrows():
            raw_date = row[date_col]
            try:
                date_str = pd.to_datetime(raw_date).date().isoformat()
            except Exception:
                continue
            entry = {"date": date_str}
            for src, dst in col_map.items():
                if src in df.columns:
                    entry[dst] = row[src]
            records.append(entry)

        return clean_nans(records)

    except Exception as e:
        logging.error(f"Error serving earnings dates for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stock-analysis/{symbol}")
def get_stock_analysis(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """
    Returns AI-powered stock analysis for a given symbol.
    """
    try:
        (_, _, user_symbol_map, user_excluded_symbols, _, _, _, _) = data
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols) or symbol
        
        mdp = get_mdp()
        # 1. Fetch Fundamentals
        fund_data = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        if not fund_data:
            fund_data = {}
            
        # 2. Fetch Release Statements and calculate ratios
        financials_df = mdp.get_financials(yf_symbol, "annual", force_refresh=force)
        balance_sheet_df = mdp.get_balance_sheet(yf_symbol, "annual", force_refresh=force)
        cashflow_df = mdp.get_cashflow(yf_symbol, "annual", force_refresh=force)
        
        ratios = {}
        if financials_df is not None and not financials_df.empty and balance_sheet_df is not None and not balance_sheet_df.empty:
            try:
                ratios_df = calculate_key_ratios_timeseries(
                    financials_df,
                    balance_sheet_df,
                    cashflow_df
                )
                if not ratios_df.empty:
                    # Take the most recent period ratios
                    ratios = ratios_df.iloc[0].to_dict()
            except Exception as e_ratio:
                logging.warning(f"Ratio calculation failed for analysis: {e_ratio}")

        # 3. Generate AI Review — reads/writes go through the global screener DB.
        analysis = generate_stock_review(symbol, fund_data, ratios, force_refresh=force)

        # 4. Interactive Calculation of Intrinsic Value & Cache Update
        try:
            # Check cache first if not forced
            iv_results = None
            if not force:
                try:
                    cached_results = get_cached_screener_results([symbol])
                    if symbol in cached_results:
                        cached_entry = cached_results[symbol]
                        
                        # Extract cached metadata
                        cached_fy_end = cached_entry.get("last_fiscal_year_end")
                        cached_mrq = cached_entry.get("most_recent_quarter")
                        cached_val_details_str = cached_entry.get("valuation_details")
                        
                        current_fy_end = fund_data.get("lastFiscalYearEnd")
                        current_mrq = fund_data.get("mostRecentQuarter")
                        
                        # Validate if cache is fresh enough (Timestamps match)
                        is_fresh = True
                        if current_fy_end and cached_fy_end != current_fy_end:
                            is_fresh = False
                        if current_mrq and cached_mrq != current_mrq:
                            is_fresh = False
                            
                        # Also ensure we actually have the detailed JSON stored
                        if is_fresh and cached_val_details_str:
                             logging.info(f"Using cached Intrinsic Value for {symbol} (Freshness verified)")
                             try:
                                 iv_results = json.loads(cached_val_details_str)
                                 # Ensure top-level metrics match cache (consistency check)
                                 # (optional, but good for safety)
                             except json.JSONDecodeError:
                                 logging.warning(f"Failed to decode cached valuation details for {symbol}")
                                 iv_results = None
                except Exception as e_cache_read:
                    logging.warning(f"Error checking IV cache for {symbol}: {e_cache_read}")

            if iv_results is None:
                # We calculate this ON THE FLY to ensure the frontend gets the detailed value
                # immediately when "Analyze" is clicked, updating the screener row live.
                logging.info(f"Recalculating Intrinsic Value for {symbol}...")
                iv_results = get_comprehensive_intrinsic_value(
                    fund_data, financials_df, balance_sheet_df, cashflow_df
                )
                
                # Serialize the entire result for storage
                iv_json = json.dumps(iv_results, default=str)
                
                # Inject into info dict so it gets picked up by update_intrinsic_value_in_cache
                # We copy it to avoid mutating the original fund_data for other uses if any
                info_for_update = fund_data.copy()
                info_for_update["valuation_details"] = iv_json

                # Update cache so screener table reads it next time (or live update listens to it)
                update_intrinsic_value_in_cache(
                    symbol,
                    iv_results.get("average_intrinsic_value"),
                    iv_results.get("margin_of_safety_pct"),
                    fund_data.get("lastFiscalYearEnd"),
                    fund_data.get("mostRecentQuarter"),
                    info=info_for_update
                )
            
            # Inject into response so frontend event can carry it
            analysis["intrinsic_value_data"] = iv_results

        except Exception as iv_e:
            logging.error(f"Failed to calculate IV during stock analysis for {symbol}: {iv_e}")

        return clean_nans(analysis)
    except Exception as e:
        logging.error(f"Error in stock analysis for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fundamentals/{symbol}")
def get_fundamentals_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns fundamental data (ticker.info) for a symbol."""
    (_, _, user_symbol_map, user_excluded_symbols, _, _, _, _) = data
    if is_cash_symbol(symbol):
        return {
            "symbol": symbol,
            "shortName": "Cash Balance",
            "longName": "Cash and Cash Equivalents",
            "regularMarketPrice": 1.0,
            "currentPrice": 1.0,
            "quoteType": "CASH",
            "sector": "Cash",
            "industry": "Cash",
            "marketCap": 0,
            "dividendYield": 0.0,
            "trailingPE": None,
            "forwardPE": None
        }

    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        fundamental_data = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        if fundamental_data is None:
             raise HTTPException(status_code=404, detail=f"No fundamental data found for {yf_symbol}")

        # Best-effort live price piggyback: read the existing current-quotes cache file
        # (populated by the dashboard's batch fetch) without triggering a new subprocess.
        # This keeps the modal open path subprocess-free on cache hits.
        try:
            cache_file = getattr(mdp, "current_cache_file", None)
            if cache_file and os.path.exists(cache_file):
                with open(cache_file, "r") as _f:
                    _cache = json.load(_f)
                _quotes = _cache.get("quotes") or {}
                _ts_str = _cache.get("timestamp")
                _fresh = False
                if _ts_str:
                    try:
                        _ts = datetime.fromisoformat(_ts_str)
                        from utils_time import is_market_open
                        _ttl_min = 5 if is_market_open() else 240
                        if datetime.now(timezone.utc) - _ts < timedelta(minutes=_ttl_min):
                            _fresh = True
                    except Exception:
                        pass
                if _fresh and symbol in _quotes:
                    _live = _quotes[symbol]
                    _price = _live.get("price")
                    if _price:
                        fundamental_data["regularMarketPrice"] = _price
                        fundamental_data["currentPrice"] = _price
                        if "day_change" in _live:
                            fundamental_data["regularMarketChange"] = _live["day_change"]
                        if "day_change_percent" in _live:
                            fundamental_data["regularMarketChangePercent"] = _live["day_change_percent"]
        except Exception as e_live:
            logging.debug(f"Live price piggyback skipped for {symbol}: {e_live}")

        # Next earnings report / just-reported quarter / next dividend for the
        # detail modal's Overview tab. Derived from the blob we already hold, so
        # it costs no extra fetch and agrees with the dashboard's Events panel.
        try:
            from server.calendar_events import upcoming_events

            # Fill in a just-reported quarter's figures if the blob has none, so
            # the Overview tab shows what was printed rather than that something
            # was (same backfill the dashboard's Events panel uses).
            fundamental_data = mdp.with_reported_earnings(yf_symbol, fundamental_data)
            # No `today` argument: the horizon is reckoned on the exchange's own
            # clock, not this server's.
            fundamental_data["upcoming_events"] = upcoming_events(symbol, fundamental_data)
        except Exception as e_events:
            logging.debug(f"Upcoming-events derivation skipped for {symbol}: {e_events}")

        # The valuation / earnings / profitability / market block the detail
        # window shows. Derived from the blob already in hand plus one indexed
        # read of the local EDGAR store, so it costs no fetch; a symbol with no
        # filings gets the same block with the filed-history fields absent.
        try:
            fundamental_data["key_metrics"] = _key_metrics_for_symbol(symbol, fundamental_data)
        except Exception as e_metrics:
            logging.debug(f"Key-metrics derivation skipped for {symbol}: {e_metrics}")

        return clean_nans(fundamental_data)
    except Exception as e:
        logging.error(f"Error fetching fundamentals for {yf_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/financials/{symbol}")
def get_financials_endpoint(
    symbol: str,
    period_type: str = "annual",
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns historical financial statements for a symbol."""
    (_, _, user_symbol_map, user_excluded_symbols, _, _, _, _) = data
    if is_cash_symbol(symbol):
        return {"symbol": symbol, "period": period_type, "income_statement": [], "balance_sheet": [], "cash_flow": []}

    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        financials = mdp.get_financials(yf_symbol, period_type, force_refresh=force)
        balance_sheet = mdp.get_balance_sheet(yf_symbol, period_type, force_refresh=force)
        cashflow = mdp.get_cashflow(yf_symbol, period_type, force_refresh=force)
        
        # Convert DataFrames to dicts for JSON serialization
        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            return json.loads(df.to_json(orient="split", date_format="iso"))

        # Extract Shareholders' Equity from Balance Sheet if possible
        equity_items = [
            "Stockholders Equity", "Total Equity Gross Minority Interest", 
            "Common Stock Equity", "Retained Earnings", "Capital Stock", 
            "Common Stock", "Other Equity Adjustments", 
            "Gains Losses Not Affecting Retained Earnings",
            "Treasury Shares Number", "Ordinary Shares Number", "Share Issued"
        ]
        shareholders_equity = None
        if balance_sheet is not None and not balance_sheet.empty:
            # Filter rows that exist in the balance sheet index
            existing_equity_items = [item for item in equity_items if item in balance_sheet.index]
            if existing_equity_items:
                shareholders_equity = balance_sheet.loc[existing_equity_items]

        return clean_nans({
            "financials": df_to_dict(financials),
            "balance_sheet": df_to_dict(balance_sheet),
            "cashflow": df_to_dict(cashflow),
            "shareholders_equity": df_to_dict(shareholders_equity)
        })
    except Exception as e:
        logging.error(f"Error fetching financials for {yf_symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/ratios/{symbol}")
def get_ratios_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data)
):
    """Returns calculated financial ratios for a symbol."""
    if not FINANCIAL_RATIOS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Financial ratios module not available.")

    (_, _, user_symbol_map, user_excluded_symbols, _, _, _, _) = data
    if is_cash_symbol(symbol):
        return {"symbol": symbol, "historical_ratios": [], "current_valuation": {}}

    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        # Fetch data needed for ratios
        info = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        financials = mdp.get_financials(yf_symbol, "annual", force_refresh=force)
        balance_sheet = mdp.get_balance_sheet(yf_symbol, "annual", force_refresh=force)
        # Cash flow is here for the free-cash-flow margin: whether reported
        # profit turns into cash is the one thing the other ratios cannot say.
        cashflow = mdp.get_cashflow(yf_symbol, "annual", force_refresh=force)

        # Calculate historical ratios
        historical_ratios_df = calculate_key_ratios_timeseries(
            financials, balance_sheet, cashflow
        )
        
        # Calculate current valuation ratios
        current_valuation = calculate_current_valuation_ratios(info, financials, balance_sheet)
        
        # Format historical ratios
        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            # Reset index to include 'Period'
            df_reset = df.reset_index()
            if 'Period' in df_reset.columns:
                df_reset['Period'] = df_reset['Period'].astype(str)
            return df_reset.to_dict(orient="records")

        return clean_nans({
            "historical": df_to_dict(historical_ratios_df),
            "valuation": current_valuation
        })
    except Exception as e:
        logging.error(f"Error calculating ratios for {yf_symbol}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/intrinsic_value/{symbol}")
def get_intrinsic_value_endpoint(
    symbol: str,
    force: bool = Query(False),
    data: tuple = Depends(get_transaction_data),
    config_manager: ConfigManager = Depends(get_config_manager),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """Returns calculated intrinsic value results for a symbol."""
    logging.info(f"CALCULATING INTRINSIC VALUE FOR {symbol} - CODE VERSION: 1.1 (CAPS ENABLED)")
    if not FINANCIAL_RATIOS_AVAILABLE:
        raise HTTPException(status_code=501, detail="Financial ratios module not available.")

    (_, _, user_symbol_map, user_excluded_symbols, _, _, _, _) = data
    if is_cash_symbol(symbol):
        return {"symbol": symbol, "intrinsic_value": 1.0, "current_price": 1.0, "upside_potential": 0.0, "is_cash": True}

    yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
    if not yf_symbol:
        if symbol.upper() in user_excluded_symbols:
             raise HTTPException(status_code=400, detail=f"Symbol {symbol} is currently in the exclusion list.")
        raise HTTPException(status_code=400, detail=f"Could not map {symbol} to Yahoo Finance symbol.")
    
    try:
        mdp = get_mdp()
        results = get_intrinsic_value_for_symbol(symbol, mdp, config_manager, force_refresh=force)
        
        if "error" in results:
             raise HTTPException(status_code=500, detail=results["error"])
        
        # We still need info for the sync function below
        yf_symbol = map_to_yf_symbol(symbol, user_symbol_map, user_excluded_symbols)
        info = mdp.get_fundamental_data(yf_symbol, force_refresh=force)
        
        # Sync to global screener cache
        try:
            if info:
                info["valuation_details"] = results
            update_intrinsic_value_in_cache(
                symbol,
                results.get("average_intrinsic_value"),
                results.get("margin_of_safety_pct"),
                info.get("lastFiscalYearEnd") if info else None,
                info.get("mostRecentQuarter") if info else None,
                info=info
            )
        except Exception as e_sync:
            logging.warning(f"Failed to sync intrinsic value to cache for {symbol}: {e_sync}")

        return clean_nans(results)
    except Exception as e:
        logging.error(f"Error calculating intrinsic value for {yf_symbol}: {e}")
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# FX rate cache: {currency_code: (rate, expiry_timestamp)}
_FX_RATE_CACHE: dict = {}
_FX_RATE_TTL = 300  # 5 minutes

def _fetch_fx_rate_sync(currency_code: str) -> float | None:
    """Fetch FX rate via yfinance (blocking I/O)."""
    import yfinance as yf
    import time
    
    ticker = f"{currency_code}=X"
    if currency_code == "USD":
        return 1.0
        
    now = time.time()
    if currency_code in _FX_RATE_CACHE:
        rate, expiry = _FX_RATE_CACHE[currency_code]
        if now < expiry:
            return rate
            
    try:
        tkr = yf.Ticker(ticker)
        data = tkr.history(period="1d", interval="1d")
        if not data.empty and "Close" in data.columns:
            rate = float(data["Close"].iloc[-1])
            _FX_RATE_CACHE[currency_code] = (rate, now + _FX_RATE_TTL)
            return rate
    except Exception as e:
        print(f"Error fetching fx rate for {currency_code}: {e}")
    return None

@router.get("/fx_rate/{currency}")
async def get_fx_rate(currency: str):
    """
    Returns the current exchange rate for USD to the given currency.
    """
    if currency.upper() == "USD":
        return {"rate": 1.0}
        
    if not re.match(r"^[A-Z]{3}$", currency.upper()):
        raise HTTPException(status_code=400, detail="Invalid currency code format")
        
    rate = await asyncio.to_thread(_fetch_fx_rate_sync, currency.upper())
    if rate is None:
        raise HTTPException(status_code=404, detail=f"Exchange rate not found for {currency}")
    return {"rate": rate}


# ---------------------------------------------------------------------------
# S&P 500 Heatmap
# ---------------------------------------------------------------------------

_SP500_HEATMAP_CACHE = SWRCache(max_size=2)

# The heatmap quotes 500 symbols. `get_current_quotes` keys a *single* cache file
# on the symbol set, so sharing the portfolio's provider would make the two
# permanently evict each other: every heatmap rebuild would overwrite the
# dashboard's 60-symbol entry and vice versa, and neither would ever hit cache.
# A dedicated provider gives the heatmap its own quotes file; the per-symbol
# fundamentals/metadata/history caches are shared as normal (no clobbering
# there — those are keyed per symbol).
_SP500_HEATMAP_MDP = None
_SP500_HEATMAP_MDP_LOCK = threading.Lock()


def _get_heatmap_mdp():
    """Market data provider whose current-quotes cache is private to the heatmap."""
    global _SP500_HEATMAP_MDP
    with _SP500_HEATMAP_MDP_LOCK:
        if _SP500_HEATMAP_MDP is None:
            from market_data import MarketDataProvider

            _SP500_HEATMAP_MDP = MarketDataProvider(
                current_cache_file="sp500_heatmap_quotes.json"
            )
    return _SP500_HEATMAP_MDP


def _dedupe_share_classes(constituents: list) -> list:
    """Keep one ticker per company so market cap is not double-counted.

    Yahoo reports the *company's* market cap against every share class, so
    keeping both GOOGL and GOOG would draw Alphabet at twice its true weight.
    Wikipedia's CIK column identifies the issuer, and it lists the class with
    the broader float first (GOOGL before GOOG, FOXA before FOX), so first-wins
    keeps the primary line without hardcoding a ticker list that goes stale as
    the index changes.
    """
    seen_ciks = set()
    kept = []
    for c in constituents:
        cik = c.get("cik")
        if cik:
            if cik in seen_ciks:
                continue
            seen_ciks.add(cik)
        kept.append(c)
    return kept


def _dividend_yield_fraction(info: dict, price) -> Optional[float]:
    """Yahoo's `dividendYield` as a fraction, or None.

    Yahoo encodes this field as *percent* for some symbols and as a fraction for
    others, and the two ranges overlap (a 0.35 could be 0.35% or 35%), so it
    cannot be settled by magnitude. Resolve it the way the watchlist route does:
    against dividend rate over price, falling back to the trailing yield, which
    Yahoo always reports as a fraction.
    """
    rate = info.get("dividendRate") or info.get("trailingAnnualDividendRate")
    if rate and price and price > 0:
        return float(rate) / float(price)

    trailing = info.get("trailingAnnualDividendYield")
    if trailing is not None:
        return float(trailing)

    return None


# Yahoo drops individual symbols from a large multi-symbol response when it is
# under pressure (yfinance logs "possibly delisted" and omits the column), and
# the wider the chunk the more a single bad response costs.
_HEATMAP_HISTORY_CHUNK = 50

# How long a fetched history frame stays usable. Ten years of monthly bars only
# change when a month closes, and daily bars once a day — re-fetching 500
# symbols of both every five minutes was pure waste, and it was the load that
# got the rebuild throttled in the first place.
_HISTORY_CACHE_TTL = {"1mo": 12 * 3600, "1d": 45 * 60}
# Below this share of symbols having usable history, the build is treated as
# failed rather than cached — see `_fetch_monthly_closes`.
_HEATMAP_MIN_HISTORY_COVERAGE = 0.5
# Empty chunks usually mean Yahoo throttled us, so the retry pass waits first.
_HEATMAP_RETRY_BACKOFF = 15.0


def _naive_index(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop any timezone from the index.

    yfinance returns tz-aware bars for some intervals and naive for others, and
    a cached frame can disagree with a fresh one. Left alone, merging the two
    raises "Cannot join tz-naive with tz-aware DatetimeIndex". These are
    calendar days on an exchange, so the zone carries no information here.
    """
    if not frame.empty and getattr(frame.index, "tz", None) is not None:
        frame = frame.copy()
        frame.index = frame.index.tz_localize(None)
    return frame


def _fetch_closes(symbols: list, start, end, interval: str, min_coverage: float) -> pd.DataFrame:
    """Adjusted closes per symbol, fetched through the isolated worker.

    Deliberately NOT a direct ``yf.download``. yfinance keeps module-level state
    across threads and has no rate-limit memory, so calling it in-process races
    the refresh worker and the portfolio's own fetches: symbols come back empty
    and every period column for them silently reads n/a. Routing through
    ``_run_isolated_fetch`` reuses the retry/backoff, the global 429 cool-down
    and the crash isolation the rest of the codebase already depends on.

    ``min_coverage`` is the share of symbols that must come back with data before
    the result is considered usable; below it the fetch raises.
    """
    from market_data import _extract_ticker_from_df, _run_isolated_fetch

    def _fetch(batch: list) -> list:
        frames = []
        for i in range(0, len(batch), _HEATMAP_HISTORY_CHUNK):
            chunk = batch[i : i + _HEATMAP_HISTORY_CHUNK]
            try:
                df = _run_isolated_fetch(
                    chunk, start=start, end=end, interval=interval, task="history"
                )
                if df is not None and not df.empty:
                    frames.append(df)
            except Exception as e:
                logging.warning(f"Heatmap history chunk {i // _HEATMAP_HISTORY_CHUNK} failed: {e}")
        return frames

    def _to_closes(frames: list) -> dict:
        if not frames:
            return {}
        combined = pd.concat(frames, axis=1) if len(frames) > 1 else frames[0]
        out = {}
        for sym in symbols:
            try:
                sym_df = _extract_ticker_from_df(combined, sym)
            except Exception:
                continue
            if sym_df.empty:
                continue
            # Adjusted where available so the returns are total returns; the
            # worker requests auto_adjust=False, which keeps both columns. The
            # raw close rides along because it is the price a user recognises.
            adj_col = next((c for c in ("Adj Close", "Close") if c in sym_df.columns), None)
            if adj_col is None:
                continue
            series = sym_df[adj_col].dropna()
            if series.empty:
                continue
            raw = sym_df["Close"].dropna() if "Close" in sym_df.columns else series
            out[sym] = (series, raw)
        return out

    closes = _to_closes(_fetch(symbols))

    # One retry for whatever came back empty. Yahoo drops symbols sporadically
    # under load, and a second pass over the stragglers is far cheaper than
    # serving a map where a third of the tiles read n/a.
    stragglers = [s for s in symbols if s not in closes]
    if stragglers and len(stragglers) < len(symbols):
        # Back off first. Empty chunks usually mean Yahoo throttled us, and an
        # immediate retry just collects the same 429.
        logging.info(f"Heatmap: retrying {interval} history for {len(stragglers)} symbols")
        time.sleep(_HEATMAP_RETRY_BACKOFF)
        closes.update(_to_closes(_fetch(stragglers)))

    covered = len(closes) / len(symbols) if symbols else 0
    if covered < min_coverage:
        # Refuse rather than return a mostly-empty frame: the caller caches its
        # result for up to an hour, so a degraded build would pin every period
        # column at n/a long after Yahoo recovered. Raising leaves the previous
        # good payload in place.
        raise RuntimeError(
            f"Heatmap {interval} history covered only {len(closes)}/{len(symbols)} symbols"
        )
    if stragglers:
        logging.warning(
            f"Heatmap: no {interval} history for {len(symbols) - len(closes)}/{len(symbols)} symbols"
        )

    return (
        _naive_index(pd.DataFrame({s: v[0] for s, v in closes.items()})),
        _naive_index(pd.DataFrame({s: v[1] for s, v in closes.items()})),
    )


def _history_cache_path(interval: str) -> str:
    return os.path.join(
        config.get_app_data_dir(), config.CACHE_DIR, f"sp500_heatmap_history_{interval}.pkl"
    )


def _load_cached_history(interval: str):
    """(adjusted, raw, age_seconds) from the on-disk frame, or None."""
    path = _history_cache_path(interval)
    try:
        if not os.path.exists(path):
            return None
        age = time.time() - os.path.getmtime(path)
        adjusted, raw = pd.read_pickle(path)
        if adjusted is None or adjusted.empty:
            return None
        # Frames written before the index was normalised may still be tz-aware.
        return _naive_index(adjusted), _naive_index(raw), age
    except Exception as e:
        logging.warning(f"Heatmap {interval} history cache unreadable ({e}); refetching")
        return None


def _get_history(symbols: list, start, end, interval: str, min_coverage: float):
    """Adjusted and raw closes, fetched at most once per cache TTL.

    A fresh fetch is *merged over* the previous frame rather than replacing it.
    Yahoo silently omits symbols it is rate-limiting, so a replace turns one bad
    minute into a map where a third of the tiles read n/a; merging means a
    partial response can only ever add coverage.
    """
    cached = _load_cached_history(interval)
    if cached is not None:
        adjusted, raw, age = cached
        covered = len([s for s in symbols if s in adjusted.columns]) / max(len(symbols), 1)
        if age < _HISTORY_CACHE_TTL.get(interval, 3600) and covered >= min_coverage:
            logging.info(
                f"Heatmap: reusing {interval} history ({covered:.0%} of symbols, {age / 60:.0f}m old)"
            )
            return adjusted, raw

    fresh_adj, fresh_raw = _fetch_closes(symbols, start, end, interval, 0.0)

    if cached is not None:
        prev_adj, prev_raw, _ = cached
        # New values win; the previous frame fills only what this fetch missed.
        fresh_adj = fresh_adj.combine_first(prev_adj) if not fresh_adj.empty else prev_adj
        fresh_raw = fresh_raw.combine_first(prev_raw) if not fresh_raw.empty else prev_raw

    keep = [s for s in symbols if s in fresh_adj.columns]
    covered = len(keep) / max(len(symbols), 1)
    if covered < min_coverage:
        raise RuntimeError(
            f"Heatmap {interval} history covered only {len(keep)}/{len(symbols)} symbols"
        )

    try:
        pd.to_pickle((fresh_adj, fresh_raw), _history_cache_path(interval))
    except Exception as e:
        logging.warning(f"Could not persist heatmap {interval} history: {e}")

    return fresh_adj, fresh_raw


def _fetch_monthly_closes(symbols: list, start, end) -> pd.DataFrame:
    """Long-horizon monthly closes. Load-bearing: the build fails without them."""
    adjusted, _ = _get_history(symbols, start, end, "1mo", _HEATMAP_MIN_HISTORY_COVERAGE)
    return adjusted


def _fetch_daily_closes(symbols: list, start, end):
    """Recent daily bars: adjusted closes for returns, raw closes for price.

    Also the source of the live price and the 1-day change. Deriving both from
    this frame is what lets the build skip `get_current_quotes`, which chunks
    500 symbols into ~25 separate worker fetches plus a 500-symbol intraday
    pull — enough traffic on its own to get the whole rebuild rate-limited.

    Supplementary: a failure degrades the short-horizon metrics to n/a rather
    than sinking the map, so it carries no coverage floor.
    """
    try:
        return _get_history(symbols, start, end, "1d", 0.0)
    except Exception as e:
        logging.warning(f"Heatmap daily history unavailable: {e}")
        return pd.DataFrame(), pd.DataFrame()


# EPS and revenue come from filed annual figures rather than Yahoo, which only
# carries the trailing twelve months. Revenue moved tags over the years, so the
# fallback chain mirrors edgar_provider's.
_EDGAR_EPS_TAG = "EarningsPerShareDiluted"
_EDGAR_REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
)
_EDGAR_LT_DEBT_TAG = "LongTermDebtNoncurrent"
_EDGAR_EQUITY_TAG = "StockholdersEquity"


def _edgar_annual_facts(constituents: list) -> dict:
    """Filed annual EPS / revenue / long-term debt / equity, keyed by symbol.

    One bulk query over the local EDGAR fact store (~0.5s for the whole index)
    rather than a per-company round trip.
    """
    by_cik = {}
    for c in constituents:
        cik = str(c.get("cik", "")).strip()
        if cik.isdigit():
            by_cik.setdefault(cik.zfill(10), c["symbol"])
    if not by_cik:
        return {}

    tags = (_EDGAR_EPS_TAG, *_EDGAR_REVENUE_TAGS, _EDGAR_LT_DEBT_TAG, _EDGAR_EQUITY_TAG)
    try:
        from edgar_provider import get_store

        store = get_store()
        with store._connect() as conn:
            rows = conn.execute(
                f"SELECT cik, tag, period_end, val FROM facts "
                f"WHERE tag IN ({','.join('?' * len(tags))}) "
                f"AND cik IN ({','.join('?' * len(by_cik))})",
                [*tags, *by_cik],
            ).fetchall()
    except Exception as e:
        logging.warning(f"Heatmap: EDGAR facts unavailable ({e}); filed-history metrics will be n/a")
        return {}

    out: dict = {}
    for cik, tag, period_end, val in rows:
        sym = by_cik.get(cik)
        if sym is None or val is None:
            continue
        out.setdefault(sym, {}).setdefault(tag, {})[period_end] = float(val)
    return out


def _annual_series(facts: dict, tags) -> list:
    """(period_end, value) pairs, newest last, from the first tag that has data."""
    for tag in (tags,) if isinstance(tags, str) else tags:
        series = facts.get(tag)
        if series:
            return sorted(series.items())
    return []


def _cagr_over(series: list, years: int):
    """Annualised growth across ``years`` of filed annuals, or None.

    Returns None when either endpoint is non-positive — a CAGR through zero or
    a sign change is not a real growth rate, and reporting one would be worse
    than admitting the figure does not exist.
    """
    if len(series) < 2:
        return None
    end_date, end_val = series[-1]
    target = pd.Timestamp(end_date) - pd.DateOffset(years=years)
    # Nearest filed annual on or before the target, tolerating a ragged fiscal
    # calendar by allowing a quarter of slack.
    candidates = [(d, v) for d, v in series[:-1] if pd.Timestamp(d) <= target + pd.DateOffset(months=3)]
    if not candidates:
        return None
    start_date, start_val = candidates[-1]
    span = (pd.Timestamp(end_date) - pd.Timestamp(start_date)).days / 365.25
    if span < 1 or start_val <= 0 or end_val <= 0:
        return None
    return (end_val / start_val) ** (1 / span) - 1


def _safe_div(numerator, denominator):
    """Quotient, or None when either side is missing or the divisor is zero."""
    try:
        if numerator is None or not denominator:
            return None
        return float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _latest_eps_surprise(info: dict):
    """Most recent reported quarter's EPS surprise, as a fraction.

    Yahoo stashes `surprise_pct` in percent points; everything else on the wire
    is a fraction, so it is converted here rather than in each client.
    """
    history = info.get("_earnings_history")
    if not isinstance(history, dict):
        return None
    for _, row in sorted(history.items(), reverse=True):
        if not isinstance(row, dict):
            continue
        if row.get("eps_actual") is not None and row.get("surprise_pct") is not None:
            return float(row["surprise_pct"]) / 100.0
    return None


def _days_to_earnings(info: dict, today):
    """Calendar days until the next reported earnings date (negative if past)."""
    ts = info.get("earningsTimestamp")
    if not ts:
        return None
    try:
        # Yahoo dates the event in the market's own timezone, not the server's.
        when = datetime.fromtimestamp(float(ts), tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None
    return (when - today).days


def _fundamental_metrics(info: dict, facts: dict, *, market_cap, pe_ratio, price, today) -> dict:
    """Valuation / earnings / profitability / market readings for one company.

    Shared by the S&P 500 heatmap and the per-symbol detail window so the two can
    never disagree about what "P/FCF" or "ROIC" means. Every field here is a
    fraction, a plain ratio or a count; the two that are percent *points* are
    called out where they are produced.

    `facts` is that company's filed annual series (see `_edgar_annual_facts`);
    pass `{}` when there are none and the filed-history fields report absent
    rather than guessing from the trailing twelve months.
    """
    eps_annual = _annual_series(facts, _EDGAR_EPS_TAG)
    revenue_annual = _annual_series(facts, _EDGAR_REVENUE_TAGS)
    lt_debt = _annual_series(facts, _EDGAR_LT_DEBT_TAG)
    equity = _annual_series(facts, _EDGAR_EQUITY_TAG)

    # Yahoo has no ROIC; derive it from figures it does carry. Net income
    # stands in for NOPAT and book equity comes from the per-share book
    # value, so this is a screening approximation, not a filed figure.
    book_equity = None
    if info.get("bookValue") and info.get("sharesOutstanding"):
        book_equity = float(info["bookValue"]) * float(info["sharesOutstanding"])
    invested_capital = None
    if book_equity is not None and info.get("totalDebt") is not None:
        invested_capital = book_equity + float(info["totalDebt"])

    return {
        # --- Valuation
        "pe_ratio": pe_ratio,
        "forward_pe": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "ps_ratio": info.get("priceToSalesTrailing12Months"),
        "pb_ratio": info.get("priceToBook"),
        "p_fcf": _safe_div(market_cap, info.get("freeCashflow")),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "ev_sales": info.get("enterpriseToRevenue"),
        # Always a fraction, unlike Yahoo's own field — see
        # `_dividend_yield_fraction`.
        "dividend_yield": _dividend_yield_fraction(info, price),
        # --- Earnings & sales
        "eps_ttm": info.get("trailingEps"),
        "eps_qoq": info.get("earningsQuarterlyGrowth"),
        "eps_growth_3y": _cagr_over(eps_annual, 3),
        "eps_growth_5y": _cagr_over(eps_annual, 5),
        "eps_surprise": _latest_eps_surprise(info),
        "sales_ttm": info.get("totalRevenue"),
        "sales_qoq": info.get("revenueGrowth"),
        "sales_growth_3y": _cagr_over(revenue_annual, 3),
        "sales_growth_5y": _cagr_over(revenue_annual, 5),
        # --- Profitability & balance sheet
        "roa": info.get("returnOnAssets"),
        "roe": info.get("returnOnEquity"),
        "roic": _safe_div(info.get("netIncomeToCommon"), invested_capital),
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "net_margin": info.get("profitMargins"),
        "quick_ratio": info.get("quickRatio"),
        "current_ratio": info.get("currentRatio"),
        "debt_equity": info.get("debtToEquity"),
        # Filed figures, so expressed like debt_equity: percent points.
        # Negative book equity makes the ratio meaningless (it flips
        # sign rather than growing), so it is reported as absent.
        "lt_debt_equity": (
            _safe_div(lt_debt[-1][1], equity[-1][1]) * 100
            if lt_debt and equity and equity[-1][1] > 0
            else None
        ),
        # --- Market & sentiment
        "relative_volume": _safe_div(
            info.get("volume") or info.get("regularMarketVolume"),
            info.get("averageVolume"),
        ),
        "float_short": info.get("shortPercentOfFloat"),
        # Yahoo's 1 (strong buy) .. 5 (sell) consensus.
        "analyst_recom": info.get("recommendationMean"),
        "earnings_days": _days_to_earnings(info, today),
    }


def _cik_for_symbol(symbol: str) -> Optional[str]:
    """This symbol's zero-padded CIK, or None — without ever hitting the network.

    The detail window is on a request path, so a miss must cost nothing: the
    ticker→CIK map is read from whatever the universe build already left on
    disk, and a symbol that is not in it simply has no filed history to show.
    """
    try:
        from universe import get_cached_cik_map

        return get_cached_cik_map().get(symbol.upper())
    except Exception as e:
        logging.debug(f"CIK lookup unavailable for {symbol}: {e}")
        return None


def _key_metrics_for_symbol(symbol: str, info: dict) -> dict:
    """The detail window's metric block for one symbol.

    Same computation the heatmap runs over the whole index, so a stock reads the
    same either side of a click.
    """
    cik = _cik_for_symbol(symbol)
    facts = _edgar_annual_facts([{"symbol": symbol, "cik": cik}]).get(symbol, {}) if cik else {}

    price = info.get("currentPrice") or info.get("regularMarketPrice")
    # Days-to-earnings is a "days from now" a user reads, so it is counted on the
    # exchange's own calendar rather than this server's (see utils_time).
    try:
        from server.calendar_events import market_today

        today = market_today(info)
    except Exception:
        today = get_est_today()

    return _fundamental_metrics(
        info,
        facts,
        market_cap=info.get("marketCap"),
        pe_ratio=info.get("trailingPE"),
        price=price,
        today=today,
    )


def _build_sp500_heatmap_sync() -> list:
    """Heavy lifting for the S&P 500 heatmap: fetch constituents + live quotes.

    Called from a thread (via asyncio.to_thread) so it can use the synchronous
    yfinance / Wikipedia helpers without blocking the event loop.
    """
    from server.screener_service import get_sp500_constituents

    constituents = _dedupe_share_classes(get_sp500_constituents())
    if not constituents:
        return []

    symbols = [c["symbol"] for c in constituents]
    meta_by_symbol = {c["symbol"]: c for c in constituents}

    # Price and the 1-day change come out of the daily frame below rather than
    # `get_current_quotes`: that call chunks 500 symbols into ~25 worker fetches
    # and then pulls 500 symbols of 1-minute data, which by itself was enough to
    # get the rebuild rate-limited and leave most of the map blank.
    mdp = _get_heatmap_mdp()

    # OPTIMIZATION: Do not use `mdp.get_ticker_details_batch(set(symbols))` for 500 symbols
    # as it does synchronous scraping when the file cache expires (taking 30s+).
    # Instead, use the screener database which is maintained by the background worker.
    screener_data = get_cached_screener_results(symbols)

    # We opportunistically get fundamental data from the file cache if available.
    # The background worker populates this. If it's missing, we don't want to block the heatmap.
    fundamental_data = {}
    for sym in symbols:
        path = mdp._get_symbol_fundamentals_path(sym)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    entry = json.load(f)
                info = entry.get("ticker_info") or entry.get("data")
                if info:
                    fundamental_data[sym] = info
            except Exception:
                pass

    # Cutoffs are market dates, never the server's (Investa runs on a Bangkok
    # clock that is up to a day ahead of New York — see utils_time).
    today_pd = pd.Timestamp(get_est_today())
    ytd_date = pd.Timestamp(year=today_pd.year - 1, month=12, day=31)
    date_1y = today_pd - pd.DateOffset(years=1)
    date_3y = today_pd - pd.DateOffset(years=3)
    date_5y = today_pd - pd.DateOffset(years=5)
    date_10y = today_pd - pd.DateOffset(years=10)

    # Monthly bars keep the payload small (60K points vs 1.25M daily).
    #
    # Start a quarter before the oldest cutoff rather than passing period="10y":
    # that returns exactly 120 bars beginning *after* the 10-year mark, so the
    # earliest close available is already inside the window and the 10Y column
    # can never resolve.
    hist_start = (date_10y - pd.DateOffset(months=3)).date()
    adj_close = _fetch_monthly_closes(symbols, hist_start, today_pd.date())

    # A monthly bar cannot answer "1 week" or "month to date", so the short
    # horizons take their own daily pull. Eight months covers the 6M lookback
    # and reaches back past the most recent earnings date.
    daily_close, daily_raw = _fetch_daily_closes(
        symbols, (today_pd - pd.DateOffset(months=8)).date(), today_pd.date()
    )
    daily_index = daily_close.index
    if getattr(daily_index, "tz", None) is not None:
        daily_index = daily_index.tz_localize(None)

    daily_masks = {}
    if not daily_close.empty:
        for key, cutoff in (
            ("1w", today_pd - pd.DateOffset(weeks=1)),
            ("1m", today_pd - pd.DateOffset(months=1)),
            # Month to date measures from the last close of the previous month.
            ("mtd", today_pd.replace(day=1) - pd.Timedelta(days=1)),
            ("3m", today_pd - pd.DateOffset(months=3)),
            ("6m", today_pd - pd.DateOffset(months=6)),
            ("now", today_pd),
        ):
            daily_masks[key] = daily_index <= cutoff

    edgar_facts = _edgar_annual_facts(constituents)

    # Monthly bars are labelled with the month's *first* day but carry that
    # month's closing price, so the label understates the observation by up to a
    # month — comparing labels directly makes "1Y change" span only 11 months.
    # Score each bar by the date its close actually belongs to: the month end,
    # or today for the current (still forming) month.
    if not adj_close.empty:
        raw_index = adj_close.index
        if getattr(raw_index, "tz", None) is not None:
            raw_index = raw_index.tz_localize(None)
        effective = raw_index + pd.offsets.MonthEnd(0)
        effective = pd.DatetimeIndex(
            np.minimum(effective.values, np.datetime64(today_pd.to_pydatetime()))
        )
    else:
        effective = pd.DatetimeIndex([])

    def _mask_upto(target):
        return effective <= target

    masks = {
        "ytd": _mask_upto(ytd_date),
        "1y": _mask_upto(date_1y),
        "3y": _mask_upto(date_3y),
        "5y": _mask_upto(date_5y),
        "10y": _mask_upto(date_10y),
        "now": _mask_upto(today_pd),
    }

    def _price_at(column, mask):
        if column is None:
            return None
        past = column[mask].dropna()
        if past.empty:
            return None
        value = float(past.iloc[-1])
        return value if value > 0 else None

    def _pct_change(current, past):
        if current is None or not past:
            return None
        return (current - past) / past

    has_history = not adj_close.empty
    result = []
    for sym in symbols:
        meta = meta_by_symbol.get(sym, {})
        screen = screener_data.get(sym, {})
        info = fundamental_data.get(sym, {})

        # The last daily bar is today's once the session opens, so this tracks
        # the live price the same way the batch quote path did.
        raw_bars = daily_raw[sym].dropna() if sym in daily_raw.columns else None
        price = float(raw_bars.iloc[-1]) if raw_bars is not None and not raw_bars.empty else None
        if not price:
            price = screen.get("price") or info.get("currentPrice") or info.get("regularMarketPrice")
        if not price:
            continue

        # Percent points, unlike every other return on the payload — this is the
        # one field the clients do not rescale.
        change_pct = None
        if raw_bars is not None and len(raw_bars) >= 2:
            prev = float(raw_bars.iloc[-2])
            if prev > 0:
                change_pct = (price - prev) / prev * 100.0
        if change_pct is None:
            change_pct = 0.0

        # The screener DB is filled by the background sweep; fall back to the
        # per-symbol fundamentals blob so the tile still gets a size (and so
        # `cap` mode does not silently drop the stock) before that sweep runs.
        market_cap = screen.get("market_cap") or info.get("marketCap")
        pe_ratio = screen.get("pe_ratio") or info.get("trailingPE")

        column = adj_close[sym] if (has_history and sym in adj_close.columns) else None
        # Prices are dividend/split adjusted on both ends, so these are total returns.
        current_adj = _price_at(column, masks["now"])

        d_col = daily_close[sym] if (sym in daily_close.columns) else None
        d_now = _price_at(d_col, daily_masks["now"]) if daily_masks else None

        def _daily_change(key, _col=d_col, _now=d_now):
            if not daily_masks:
                return None
            return _pct_change(_now, _price_at(_col, daily_masks[key]))

        high_52w = info.get("fiftyTwoWeekHigh")
        low_52w = info.get("fiftyTwoWeekLow")

        result.append(
            {
                "symbol": sym,
                "name": meta.get("name", sym),
                "sector": meta.get("sector", "Unknown"),
                "sub_industry": meta.get("sub_industry", "Unknown"),
                "price": price,
                "market_cap": market_cap,
                # --- Performance. `change_pct` is percent points (it comes
                # straight off the quote); every other return here is a
                # fraction, as are the growth/margin/ratio-of-quantities fields.
                "change_pct": change_pct,
                "week_change_pct": _daily_change("1w"),
                "month_change_pct": _daily_change("1m"),
                "mtd_change_pct": _daily_change("mtd"),
                "3m_change_pct": _daily_change("3m"),
                "6m_change_pct": _daily_change("6m"),
                "ytd_change_pct": _pct_change(current_adj, _price_at(column, masks["ytd"])),
                "1y_change_pct": _pct_change(current_adj, _price_at(column, masks["1y"])),
                "3y_change_pct": _pct_change(current_adj, _price_at(column, masks["3y"])),
                "5y_change_pct": _pct_change(current_adj, _price_at(column, masks["5y"])),
                "10y_change_pct": _pct_change(current_adj, _price_at(column, masks["10y"])),
                # Zero or below by construction; the high is an upper bound.
                "drawdown_52w": _pct_change(price, high_52w) if high_52w else None,
                "gain_from_52w_low": _pct_change(price, low_52w) if low_52w else None,
                # --- Valuation, earnings, profitability and market readings.
                # Shared with the per-symbol detail window so a stock reads the
                # same either side of a click.
                **_fundamental_metrics(
                    info,
                    edgar_facts.get(sym, {}),
                    market_cap=market_cap,
                    pe_ratio=pe_ratio,
                    price=price,
                    today=today_pd.date(),
                ),
            }
        )

    return result


@router.get("/sp500/heatmap")
async def get_sp500_heatmap():
    """Return S&P 500 constituent data for the heatmap visualisation.

    Each item includes symbol, name, sector, sub_industry, price,
    change_pct, market_cap, pe_ratio, and dividend_yield.
    Results are cached with adaptive TTL (5 min during market hours,
    60 min off-hours) using stale-while-revalidate.
    """
    ttl = 300 if is_market_open() else 3600  # 5 min / 60 min

    async def _compute():
        return await asyncio.to_thread(_build_sp500_heatmap_sync)

    try:
        data = await _SP500_HEATMAP_CACHE.get_or_compute(
            "sp500_heatmap", ttl=ttl, compute=_compute
        )
        return clean_nans(data)
    except Exception as e:
        # Logged in full; the client gets a generic message rather than the
        # internal exception text.
        logging.error(f"SP500 heatmap error: {e}", exc_info=True)
        # A rebuild can fail transiently (Yahoo throttling mid-fetch). Yesterday's
        # map is far more useful than an error page, so fall back to whatever was
        # last built rather than making the failure the user's problem.
        stale = _SP500_HEATMAP_CACHE.peek("sp500_heatmap")
        if stale:
            logging.warning("SP500 heatmap: serving the last good payload after a failed rebuild")
            return clean_nans(stale)
        raise HTTPException(
            status_code=503, detail="S&P 500 heatmap is temporarily unavailable"
        )
