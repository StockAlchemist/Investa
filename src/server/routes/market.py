"""Market data routes: quotes, search, news, history, fundamentals, valuation."""

import asyncio
import json
import logging
import os
import sqlite3
import time
import traceback
from collections import OrderedDict
from datetime import datetime, date, timedelta, timezone
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query

import config
from config_manager import ConfigManager
from db_utils import get_cached_screener_results, update_intrinsic_value_in_cache
from finutils import is_cash_symbol
from market_data import map_to_yf_symbol
from server.ai_analyzer import generate_stock_review
from server.dependencies import get_config_manager, get_transaction_data, get_user_db_connection
from server.route_utils import _lru_get, _lru_put, clean_nans, get_mdp
from utils_time import is_market_open

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
                    balance_sheet_df
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
        
        # Calculate historical ratios
        historical_ratios_df = calculate_key_ratios_timeseries(financials, balance_sheet)
        
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
