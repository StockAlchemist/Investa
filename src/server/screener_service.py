import pandas as pd
import logging
import os
import io
import json
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests
from concurrent.futures import ThreadPoolExecutor

from market_data import get_shared_mdp
from financial_ratios import get_intrinsic_value_for_symbol
from db_utils import (
    get_watchlist,
    get_db_connection,
    upsert_screener_results,
    refresh_screener_rows_by_symbol,
    get_cached_screener_results,
    get_screener_results_by_universe,
    get_all_distinct_screener_results,
)
import config

# Cache for S&P 500 tickers
SP500_CACHE_FILE = "sp500_tickers_cache.json"
SP500_CACHE_TTL = 86400  # 24 hours

def get_etf_holdings(product_id: str, ticker: str, filename: str) -> List[str]:
    """
    Fetches ETF holdings from iShares (BlackRock) CSV format.
    Commonly used for:
      - Russell 2000 (IWM): 239710
      - S&P MidCap 400 (IJH): 239763
    """
    cache_path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, filename)
    # 7-day TTL for static lists
    CACHE_TTL = 86400 * 7
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", 0)
                if time.time() - timestamp < CACHE_TTL:
                    logging.info(f"Using cached {ticker} list")
                    return data.get("tickers", [])
        except Exception as e:
            logging.warning(f"Error reading {ticker} cache: {e}")

    logging.info(f"Fetching {ticker} list from iShares...")
    try:
        # iShares Ajax URL format
        base_url = f"https://www.ishares.com/us/products/{product_id}/fund/1467271812596.ajax"
        params = {
            "fileType": "csv",
            "fileName": f"{ticker}_holdings",
            "dataType": "fund"
        }
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(base_url, params=params, headers=headers)
        response.raise_for_status()
        
        # Parse CSV - skips metadata rows automatically by finding header
        # iShares CSVs usually have ~10 lines of metadata before the header "Ticker,Name,..."
        # pandas read_csv with 'header' argument usually works if we skip bad lines, or we can inspect.
        # Safe way: read string, find "Ticker", start there.
        
        content = response.text
        
        if "<html" in content.lower():
            raise ValueError(f"Received HTML instead of CSV (likely bot protection) for {ticker}")
            
        lines = content.split('\n')
        start_row = 0
        for i, line in enumerate(lines[:30]):
            if "Ticker" in line and "Name" in line:
                start_row = i
                break
        
        if start_row > 0:
            df = pd.read_csv(io.StringIO(content), header=start_row)
        else:
            # Fallback
            df = pd.read_csv(io.StringIO(content))

        # Filter for valid tickers
        if "Ticker" in df.columns:
            # Drop NaN and non-string
            tickers = df["Ticker"].dropna().astype(str).tolist()
            
            # Clean up tickers (e.g. BRK.B -> BRK-B if needed, though iShares often uses dot)
            # Remove purely numeric or cash placeholders, or disclaimer text
            valid_tickers = []
            for t in tickers:
                t = str(t).strip()
                
                # Basic validity checks
                if not t or t == "-" or t.lower() == "nan": 
                    continue
                
                # Exclude obvious non-tickers (Disclaimers are long sentences)
                if len(t) > 12: 
                    continue
                
                # Tickers shouldn't have spaces usually (unless it's a special class like "BRK B" but we handle that)
                # But legal text has many spaces.
                if " " in t:
                    continue
                
                # Must start with a letter (some specialized tickers might not, but standard stocks do)
                # Allowing numbers just in case of weird ETFs/Futures, but for Russell 2000 stocks they are alpha.
                # Actually, let's just protect against the massive text block. 
                # The length check > 12 is the most effective against the disclaimer.
                
                # Yahoo Finance conversion
                t = t.replace('.', '-')
                valid_tickers.append(t)
            
            tickers = valid_tickers
            
            # Save to cache
            try:
                with open(cache_path, "w") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "tickers": tickers
                    }, f)
            except Exception as e:
                logging.warning(f"Error saving {ticker} cache: {e}")
                
            return tickers
        else:
            logging.error(f"Column 'Ticker' not found in {ticker} CSV")
            return []
            
    except Exception as e:
        logging.error(f"Failed to fetch {ticker} list: {e}")
        # Fallback to stale cache
        if os.path.exists(cache_path):
            logging.info(f"Falling back to stale cache for {ticker}")
            try:
                with open(cache_path, "r") as f:
                    data = json.load(f)
                    return data.get("tickers", [])
            except Exception as cache_err:
                logging.error(f"Failed to read stale cache for {ticker}: {cache_err}")
        return []

def get_sp500_tickers() -> List[str]:
    """
    Fetches the list of S&P 500 companies from Wikipedia.
    Uses caching to avoid repeated web requests.
    """
    cache_path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, SP500_CACHE_FILE)
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", 0)
                if time.time() - timestamp < SP500_CACHE_TTL:
                    logging.info("Using cached S&P 500 list")
                    return data.get("tickers", [])
        except Exception as e:
            logging.warning(f"Error reading S&P 500 cache: {e}")

    logging.info("Fetching S&P 500 list from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        # The first table usually contains the constituents
        df = tables[0]
        tickers = df["Symbol"].tolist()
        
        # Clean tickers (Wikipedia uses dots for classes like BRK.B, YF uses hyphens BRK-B)
        tickers = [s.replace('.', '-') for s in tickers]
        
        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "tickers": tickers
                }, f)
        except Exception as e:
            logging.warning(f"Error saving S&P 500 cache: {e}")
            
        return tickers
    except Exception as e:
        logging.error(f"Failed to fetch S&P 500 list: {e}")
        return []

def get_russell2000_tickers() -> List[str]:
    """
    Fetches Russell 2000 tickers from a stable GitHub repository.
    Uses caching to avoid repeated web requests.
    """
    cache_path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "russell2000_tickers_cache.json")
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", 0)
                if time.time() - timestamp < SP500_CACHE_TTL:
                    logging.info("Using cached Russell 2000 list")
                    return data.get("tickers", [])
        except Exception as e:
            logging.warning(f"Error reading Russell 2000 cache: {e}")

    logging.info("Fetching Russell 2000 list from GitHub...")
    try:
        url = "https://raw.githubusercontent.com/ikoniaris/Russell2000/master/russell_2000_components.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        df = pd.read_csv(io.StringIO(response.text))
        
        if "Ticker" not in df.columns:
            logging.error(f"Russell 2000 GitHub CSV missing ticker column. Found: {df.columns}")
            return []
            
        tickers = df["Ticker"].dropna().astype(str).tolist()
        # Clean tickers
        tickers = [t.strip().replace('.', '-') for t in tickers if t.strip()]
        
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "tickers": tickers
                }, f)
        except Exception as e:
            logging.warning(f"Error saving Russell 2000 cache: {e}")
            
        return tickers
    except Exception as e:
        logging.error(f"Failed to fetch Russell 2000 list: {e}")
        # Fallback to stale cache
        if os.path.exists(cache_path):
            logging.info("Falling back to stale Russell 2000 cache")
            try:
                with open(cache_path, "r") as f:
                    return json.load(f).get("tickers", [])
            except Exception:
                pass
        return []

def get_sp400_tickers() -> List[str]:
    """
    Fetches the list of S&P MidCap 400 companies from Wikipedia.
    Uses caching to avoid repeated web requests.
    """
    cache_path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "sp400_tickers_cache.json")
    
    # Check cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                data = json.load(f)
                timestamp = data.get("timestamp", 0)
                if time.time() - timestamp < SP500_CACHE_TTL:
                    logging.info("Using cached S&P 400 list")
                    return data.get("tickers", [])
        except Exception as e:
            logging.warning(f"Error reading S&P 400 cache: {e}")

    logging.info("Fetching S&P 400 list from Wikipedia...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        tables = pd.read_html(io.StringIO(response.text))
        df = tables[0]
        
        col = "Symbol" if "Symbol" in df.columns else "Ticker symbol"
        if col not in df.columns:
            logging.error(f"S&P 400 Wikipedia table missing ticker column. Found: {df.columns}")
            return []
            
        tickers = df[col].tolist()
        tickers = [str(s).replace('.', '-') for s in tickers]
        
        try:
            with open(cache_path, "w") as f:
                json.dump({
                    "timestamp": time.time(),
                    "tickers": tickers
                }, f)
        except Exception as e:
            logging.warning(f"Error saving S&P 400 cache: {e}")
            
        return tickers
    except Exception as e:
        logging.error(f"Failed to fetch S&P 400 list: {e}")
        # Fallback to stale cache
        if os.path.exists(cache_path):
            logging.info("Falling back to stale S&P 400 cache")
            try:
                with open(cache_path, "r") as f:
                    return json.load(f).get("tickers", [])
            except Exception:
                pass
        return []

def screen_stocks(universe_type: str, universe_id: Optional[str] = None, manual_symbols: Optional[List[str]] = None, db_conn: Optional[Any] = None, fast_mode: bool = False) -> List[Dict[str, Any]]:
    """
    Screens a list of stocks based on the specified universe.
    Calculates Intrinsic Value and Margin of Safety.
    """
    # Determine universe_tag FIRST to check cache before resolving symbols (which might be slow e.g. scraping Wikipedia)
    universe_tag = universe_type
    if universe_type == "watchlist" and universe_id:
        universe_tag = f"watchlist_{universe_id}"

    # --- SMART HYBRID FAST-PATH ---
    # Goal: If we have a fresh cache (< 10 mins), return instantly.
    # If we have a stale-today cache (> 10 mins), only refresh prices.
    cached_results = []
    use_pure_fast_path = False
    
    # If fast_mode is explicitly requested, we return whatever is in DB immediately (stale or not)
    if fast_mode:
        try:
            if universe_type == "all":
                # 'all' has no dedicated row tag anymore — dedupe by-symbol across universes.
                cached_results = get_all_distinct_screener_results()
            else:
                cached_results = get_screener_results_by_universe(universe_tag)
            if cached_results:
                logging.info(f"Screener: FAST MODE requested. Returning {len(cached_results)} cached items immediately.")
                cached_results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
                return cached_results
        except Exception as e:
            logging.error(f"Error in fast_mode retrieval: {e}")
        # If fast mode returns nothing, we return empty list immediately so UI can show loading state for FRESH load
        # The UI explicitly calls fast_mode=True then fast_mode=False.
        return []

    # --- SLOW PATH: Resolve Symbols ---
    # Only done if we are NOT in fast_mode (or fast_mode failed early, but here we are in the fresh loading phase)
    symbols = []
    if universe_type == "manual":
        symbols = manual_symbols or []
    elif universe_type == "watchlist" or universe_type == "holdings":
        if universe_type == "holdings":
            symbols = manual_symbols or []
        elif universe_id:
             # Use provided conn or create new one
             conn = db_conn or get_db_connection()
             try:
                wl_data = get_watchlist(conn, int(universe_id)) # This returns list of Dict with 'symbol'
                if wl_data:
                    symbols = [item['Symbol'] for item in wl_data]
             except Exception as e:
                 logging.error(f"Error fetching watchlist {universe_id}: {e}")
                 return []
             finally:
                 if not db_conn and conn:
                     conn.close()
    elif universe_type == "sp500":
        symbols = get_sp500_tickers()
    elif universe_type == "russell2000":
        symbols = get_russell2000_tickers()
    elif universe_type == "sp400":
        symbols = get_sp400_tickers()
    elif universe_type == "all":
        # For 'all', we fetch the distinct-by-symbol view of the global DB.
        # fast_mode is handled by the early-return block above.
        try:
            cached_all = get_all_distinct_screener_results()
            symbols = [r['symbol'] for r in cached_all]
        except Exception as e:
            logging.error(f"Error fetching 'all' universe symbols: {e}")
            return []
    
    if not symbols:
        return []
        
    # Recalculate tag just in case logic diverged? No, it's consistent.
    # universe_tag already set above.

    if universe_type in ["sp500", "russell2000", "sp400", "watchlist", "holdings"]:
        try:
            raw_cached = get_screener_results_by_universe(universe_tag)
            if raw_cached:
                # Find the MOST RECENT update in this entire batch
                all_timestamps = []
                for r in raw_cached:
                    ts_str = r.get("updated_at")
                    if ts_str:
                        try:
                            all_timestamps.append(datetime.fromisoformat(ts_str))
                        except ValueError:
                            pass

                if all_timestamps:
                    latest_update_ts = max(all_timestamps)
                    now_ts = datetime.now()

                    if latest_update_ts.date() == now_ts.date():
                        cached_results = raw_cached
                        age_seconds = (now_ts - latest_update_ts).total_seconds()
                        if age_seconds < 3600:
                            logging.info(f"Screener: Pure Fast-path load for '{universe_tag}' ({len(cached_results)} items, {age_seconds:.0f}s old)")
                            use_pure_fast_path = True
                        else:
                            logging.info(f"Screener: Stale Fast-path found {len(cached_results)} items. Age: {age_seconds:.0f}s. Refreshing.")
        except Exception as e:
            logging.error(f"Error in hybrid fast-path: {e}")

    if use_pure_fast_path and fast_mode:
        # Sort just in case (though DB should be sorted)
        cached_results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
        return cached_results

    # Determine what's missing from today's cache (Gaps)
    cached_symbols = {r['symbol'] for r in cached_results}
    missing_symbols = [s for s in symbols if s not in cached_symbols]
    
    # --- FETCH LIVE METADATA ---
    # If we have cached_results from today, we only NEED quotes for them.
    # For missing symbols, we need BOTH quotes and details.
    
    mdp = get_shared_mdp()
    dummy_map = {} 
    dummy_excluded = set()
    
    # If we have a lot of cached results, we might want to prioritize quotes for everyone
    # but only details for the gaps.
    logging.info(f"Screener: Fetching quotes for {len(symbols)} symbols...")
    quotes, _, _, _, _ = mdp.get_current_quotes(
        internal_stock_symbols=symbols,
        required_currencies={"USD"},
        user_symbol_map=dummy_map,
        user_excluded_symbols=dummy_excluded
    )
    
    # Fundamentals are cached for 24h inside market_data.py
    # ONLY FETCH DETAILS FOR MISSING SYMBOLS OR IF CACHE IS STALE
    # Actually, we need details for process_screener_results to work correctly
    # but we can try to minimize it. 
    # For now, let's fetch for all if missing > 0, otherwise it's fast anyway from MD cache.
    details_map = mdp.get_ticker_details_batch(set(symbols))
    
    # Check DB for individual caches (Smart Invalidation lookup)
    cached_map = {}
    try:
        cached_map = get_cached_screener_results(symbols)
    except Exception as e:
        logging.error(f"Error loading screener cache map: {e}")
            
    # Batch Fetch Financial Statements for Missing Symbols
    batch_statements = {}
    if missing_symbols:
         needed_statements_syms = []
         logging.info(f"Screener: Checking cache validity for {len(missing_symbols)} missing symbols before batch fetch...")
         
         for sym in missing_symbols:
             # Check if we can likely use the cache
             cached = cached_map.get(sym)
             info = details_map.get(sym, {})
             
             # If we don't have basic info, we probably can't calculate anyway, but let's try fetching to be safe if no cache
             if not info:
                 needed_statements_syms.append(sym)
                 continue

             live_fy_end = info.get("lastFiscalYearEnd")
             live_quarter = info.get("mostRecentQuarter")
             
             can_use_cache = False
             if cached:
                # Same logic as in process_symbol to determine if we need to recalculate
                # IMPORTANT: Also check if intrinsic_value is actually present in cache
                if (cached.get("last_fiscal_year_end") == live_fy_end and 
                    cached.get("most_recent_quarter") == live_quarter and
                    cached.get("intrinsic_value") is not None):
                    can_use_cache = True
             
             if not can_use_cache:
                 needed_statements_syms.append(sym)
         
         if needed_statements_syms:
             logging.info(f"Screener: Batch fetching financials for {len(needed_statements_syms)} symbols...")
             # Use the new batch method
             batch_statements = mdp.get_financial_statements_batch(needed_statements_syms)
         else:
             logging.info("Screener: All missing symbols have valid individual cache. Skipping batch fetch.")

    # Process everything to merge live quotes
    final_results = process_screener_results(
        symbols, 
        quotes, 
        details_map, 
        cached_map, 
        universe_tag, 
        db_conn=db_conn,
        prefetched_statements=batch_statements
    )
    
    # Final Sort
    final_results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
    
    return final_results

def process_screener_results(
    symbols: List[str], 
    quotes: Dict[str, Any], 
    details_map: Dict[str, Any],
    cached_map: Dict[str, Dict[str, Any]] = None,
    universe_tag: str = None,
    db_conn: Optional[Any] = None,
    prefetched_statements: Dict[str, Dict] = None
) -> List[Dict[str, Any]]:
    results = []
    cached_map = cached_map or {}
    prefetched_statements = prefetched_statements or {}
    
    # Pre-scan AI review directory to avoid 500+ repeated disk lookups (slow on OneDrive)
    ai_cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")
    existing_reviews = set()
    if os.path.exists(ai_cache_dir):
        try:
            # Get list of symbols that have a .json review
            files = os.listdir(ai_cache_dir)
            existing_reviews = {f.split("_")[0].upper() for f in files if f.endswith("_analysis.json")}
        except Exception:
            pass

    # --- PARALLEL PROCESSING ---
    # We use a helper function for the loop body to work with ThreadPoolExecutor
    def process_symbol(sym):
        try:
            # Get Price (Live)
            quote = quotes.get(sym, {})
            price = quote.get("price")
            
            # Get Info (Fundementals - may be from 24h cache)
            info = details_map.get(sym, {}).copy()
            
            if price:
                info["currentPrice"] = price
                info["regularMarketPrice"] = price
            elif info.get("currentPrice"):
                price = info.get("currentPrice")
            
            if not price:
                 price = info.get("previousClose")
            
            if not price:
                return None
                
            # --- SMART INVALIDATION LOGIC ---
            cached = cached_map.get(sym)
            
            # Current report identifiers
            live_fy_end = info.get("lastFiscalYearEnd")
            live_quarter = info.get("mostRecentQuarter")
            
            can_use_cache = False
            if cached:
                # If the financial report identifiers haven't changed, 
                # we can reuse the cached intrinsic value and AI scores.
                # FIX: Only trust the match if identifiers are present and IV is actually cached.
                # (None == None match is too weak and often indicates poisoned data).
                if (live_fy_end and live_quarter and 
                    cached.get("last_fiscal_year_end") == live_fy_end and 
                    cached.get("most_recent_quarter") == live_quarter and 
                    cached.get("intrinsic_value") is not None):
                    can_use_cache = True
            
            valuation_details_json = None
            if can_use_cache:
                avg_iv = cached.get("intrinsic_value")
                valuation_details_str = cached.get("valuation_details")
                valuation_details_json = valuation_details_str
                _valuation_details = None
                if valuation_details_str and isinstance(valuation_details_str, str):
                    try:
                        _valuation_details = json.loads(valuation_details_str)
                    except Exception:
                        pass
                
                # Recalculate MOS since price changes daily
                if avg_iv and price:
                    mos = ((avg_iv - price) / price) * 100 if price > 0 else 0
                else:
                    mos = None
                
                # --- AI REVIEW DATA LOADING ---
                # Check timestamps to decide between Cache vs Disk
                ai_cache_path = os.path.join(ai_cache_dir, f"{sym.upper()}_analysis.json")
                use_ai_from_disk = False
                
                # Default to cache values
                has_ai = cached.get("has_ai_review") if "has_ai_review" in cached else (cached.get("ai_summary") is not None)
                ai_score = cached.get("ai_score")
                ai_summary = cached.get("ai_summary")
                ai_moat = cached.get("ai_moat")
                ai_financial_strength = cached.get("ai_financial_strength")
                ai_predictability = cached.get("ai_predictability")
                ai_growth = cached.get("ai_growth")
                
                # FIX: Initialize sentiment and catalysts from cache as well to prevent "Ticker Poisoning"
                ai_sentiment = cached.get("ai_sentiment")
                ai_catalysts = cached.get("ai_catalysts")
                
                # Parse catalysts if they are stored as JSON string in cache
                if isinstance(ai_catalysts, str) and ai_catalysts:
                    try:
                        ai_catalysts = json.loads(ai_catalysts)
                    except Exception:
                        pass

                if os.path.exists(ai_cache_path):
                    try:
                        file_mtime = os.path.getmtime(ai_cache_path)
                        cache_ts_str = cached.get("updated_at")
                        cache_ts = 0
                        if cache_ts_str:
                            try:
                                cache_ts = datetime.fromisoformat(cache_ts_str).timestamp()
                            except ValueError:
                                pass
                        
                        # If file is newer than cache (plus small buffer), or cache has no AI data
                        # Buffer of 1s to avoid race conditions where they are created "simultaneously"
                        if file_mtime > (cache_ts + 1.0) or not has_ai:
                            use_ai_from_disk = True
                    except OSError:
                        pass

                if use_ai_from_disk:
                    try:
                        with open(ai_cache_path, 'r') as f:
                            ai_data = json.load(f)
                            # Support both old and new cache structures
                            analysis_obj = ai_data.get("analysis") if "analysis" in ai_data else ai_data
                            scorecard = analysis_obj.get("scorecard", {})
                            
                            ai_moat = scorecard.get("moat")
                            ai_financial_strength = scorecard.get("financial_strength")
                            ai_predictability = scorecard.get("predictability")
                            ai_growth = scorecard.get("growth")
                            
                            ai_sentiment = analysis_obj.get("sentiment")
                            ai_catalysts = analysis_obj.get("catalysts")
                            
                            vals = [v for v in [ai_moat, ai_financial_strength, ai_predictability, ai_growth] if isinstance(v, (int, float))]
                            if vals:
                                ai_score = sum(vals) / len(vals)
                                has_ai = True
                                ai_summary = analysis_obj.get("summary")
                                logging.info(f"Screener: Detected new AI review for {sym}. Reloading from disk.")
                            
                    except Exception as e_ai:
                        logging.warning(f"Failed to load AI cache for {sym}: {e_ai}")

                return {
                    "symbol": sym,
                    "name": info.get("shortName", sym),
                    "price": price,
                    "intrinsic_value": avg_iv,
                    "margin_of_safety": mos,
                    "pe_ratio": info.get("trailingPE"),
                    "market_cap": info.get("marketCap"),
                    "sector": info.get("sector"),
                    "has_ai_review": has_ai,
                    "ai_score": ai_score,
                    "ai_moat": ai_moat,
                    "ai_financial_strength": ai_financial_strength,
                    "ai_predictability": ai_predictability,
                    "ai_growth": ai_growth,
                    "ai_summary": ai_summary,
                    "ai_sentiment": ai_sentiment,
                    "ai_catalysts": ai_catalysts,
                    "valuation_details": valuation_details_json,
                    "last_fiscal_year_end": live_fy_end,
                    "most_recent_quarter": live_quarter
                }
            else:
                # Re-calculation needed
                mdp = get_shared_mdp()
                
                # PREFETCHED DATA CHECK
                avg_iv = None
                mos = None
                valuation_details_json = None
                _valuation_details = {}
                p_fin = None
                p_bs = None
                p_cf = None
                if sym in prefetched_statements:
                    p_data = prefetched_statements[sym]
                    if p_data:
                        p_fin = p_data.get('financials')
                        p_bs = p_data.get('balance_sheet')
                        p_cf = p_data.get('cashflow')

                # SPEED OPTIMIZATION: Reduced MC iterations for screener bulk view
                iv_res = get_intrinsic_value_for_symbol(
                    sym, 
                    mdp, 
                    iterations=1000, 
                    prefetched_financials=p_fin,
                    prefetched_balance_sheet=p_bs,
                    prefetched_cashflow=p_cf
                )
                
                avg_iv = iv_res.get("average_intrinsic_value")
                mos = iv_res.get("margin_of_safety_pct")
                valuation_details_json = json.dumps(iv_res, default=str)
                
                # AI Review Check
                has_ai = sym.upper() in existing_reviews
                
                ai_sentiment = None
                ai_catalysts = None
                ai_summary = None
                ai_moat = None
                ai_fin = None
                ai_pred = None
                ai_growth = None
                ai_score = None

                if has_ai:
                    try:
                        with open(ai_cache_path, 'r') as f:
                            ai_data = json.load(f)
                            # Support both old and new cache structures
                            analysis_obj = ai_data.get("analysis") if "analysis" in ai_data else ai_data
                            scorecard = analysis_obj.get("scorecard", {})
                            ai_summary = analysis_obj.get("summary")
                            ai_moat = scorecard.get("moat")
                            ai_fin = scorecard.get("financial_strength")
                            ai_pred = scorecard.get("predictability")
                            ai_growth = scorecard.get("growth")
                            ai_sentiment = analysis_obj.get("sentiment")
                            ai_catalysts = analysis_obj.get("catalysts")
                            vals = [v for v in [ai_moat, ai_fin, ai_pred, ai_growth] if isinstance(v, (int, float))]
                            if vals:
                                ai_score = sum(vals) / len(vals)
                    except Exception:
                        pass

                return {
                    "symbol": sym,
                    "name": info.get("shortName", sym),
                    "price": price,
                    "intrinsic_value": avg_iv,
                    "margin_of_safety": mos,
                    "pe_ratio": info.get("trailingPE"),
                    "market_cap": info.get("marketCap"),
                    "sector": info.get("sector"),
                    "has_ai_review": has_ai,
                    "ai_score": ai_score,
                    "ai_moat": ai_moat,
                    "ai_financial_strength": ai_fin,
                    "ai_predictability": ai_pred,
                    "ai_growth": ai_growth,
                    "ai_summary": ai_summary,
                    "ai_sentiment": ai_sentiment,
                    "ai_catalysts": ai_catalysts,
                    "valuation_details": valuation_details_json, 
                    "last_fiscal_year_end": live_fy_end,
                    "most_recent_quarter": live_quarter
                }
        except Exception as e:
            logging.error(f"Error processing {sym} in screener: {e}")
            return None

    logging.info(f"Screener: Processing results for {len(symbols)} symbols in parallel...")
    # WORKER LIMIT: Set to 2 (Ultra-Safe Mode). 3 was borderline for OOM on constrained systems.
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(process_symbol, sym) for sym in symbols]
        for future in futures:
            res = future.result()
            if res:
                results.append(res)
        
    # Batch Update DB Cache
    if results:
        try:
            if universe_tag == "all":
                # 'all' is a read view — refresh canonical rows by symbol, never insert an 'all' duplicate.
                refresh_screener_rows_by_symbol(results)
            else:
                upsert_screener_results(results, universe_tag)
        except Exception as e:
            logging.error(f"Error upserting screener cache: {e}")

    # Sort by Margin of Safety desc
    results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
    
    return results

def run_narrative_search(prompt: str) -> List[Dict[str, Any]]:
    """
    Experimental: Converts a natural language prompt into a SQL query for the screener_cache.
    """
    logging.info(f"Screener: Narrative Search for '{prompt}'")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logging.error("Screener: GEMINI_API_KEY not found in environment.")
        return []

    # Describe the schema to the AI
    # ai_score is a Virtual column for the AI to understand (it's calc'd from components in DB)
    schema_info = """
    Table: screener_cache
    Columns (available for filtering/ordering):
    - symbol (TEXT): Stock ticker (e.g., 'AAPL', 'TSLA')
    - name (TEXT): Company name
    - price (REAL): Current price
    - intrinsic_value (REAL): Calculated fair value
    - margin_of_safety (REAL): Upside/downside percentage from fair value
    - pe_ratio (REAL): P/E ratio
    - market_cap (REAL): Total market capitalization
    - sector (TEXT): e.g., 'Technology', 'Healthcare', 'Financial Services'
    - ai_moat (REAL): Quality of economic moat (1-10)
    - ai_financial_strength (REAL): Health of balance sheet (1-10)
    - ai_predictability (REAL): Reliability of business (1-10)
    - ai_growth (REAL): Growth outlook (1-10)
    - ai_sentiment (REAL): Market sentiment score (0-100)
    - ai_summary (TEXT): AI generated business summary
    """
    
    # We provide a hint about ai_score = (ai_moat + ai_financial_strength + ai_predictability + ai_growth) / 4.0
    
    ai_prompt = f"""
    You are an expert SQL analyst for a stock screening application. 
    Translate the user's natural language request into a single SQLite SELECT statement.
    
    DATABASE SCHEMA:
    {schema_info}
    
    VIRTUAL CALCULATIONS:
    - If user asks for "AI Score", use: ((IFNULL(ai_moat,0) + IFNULL(ai_financial_strength,0) + IFNULL(ai_predictability,0) + IFNULL(ai_growth,0)) / 4.0)
    
    USER REQUEST:
    "{prompt}"
    
    RULES:
    1. ONLY return the SQL. Do not include markdown blocks (```sql) unless necessary.
    2. The query MUST start with "SELECT * FROM screener_cache WHERE".
    3. Use standard SQLite syntax. 
    4. For sector names or company names, use 'LIKE' with '%' wildcards for fuzzy matching.
    5. Filter out rows where intrinsic_value is NULL unless explicitly asked (e.g. WHERE intrinsic_value IS NOT NULL).
    6. Sort by margin_of_safety DESC by default unless specified otherwise.
    7. LIMIT results to 100 maximum.
    
    SQL:
    """

    payload = {
        "contents": [{"parts": [{"text": ai_prompt}]}],
        "generationConfig": {
            "temperature": 0.1, # Low temperature for consistent SQL
            "topP": 0.95,
            "maxOutputTokens": 256
        }
    }

    try:
        # --- MODEL FALLBACK CHAIN FOR NL2SQL ---
        # Note: Some models (like Gemini 3 Preview) use tokens for "thoughts" (Chain of Thought), 
        # which can exceed small maxOutputTokens limits.
        SCREENER_MODELS = ["gemini-flash-latest", "gemini-2.0-flash", "gemini-3-flash-preview"]
        sql = None
        raw_response = None
        
        # Increase token limit significantly to accommodate model reasoning/thoughts
        gen_config = {
            "temperature": 0.1,
            "maxOutputTokens": 2048,
            "topP": 0.95
        }
        payload["generationConfig"] = gen_config

        # Add instruction to minimize reasoning if it helps with token usage
        if "contents" in payload and payload["contents"]:
            payload["contents"][0]["parts"][0]["text"] += "\n(Skip verbose reasoning, return ONLY the final SQL string.)"

        for model in SCREENER_MODELS:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            try:
                response = requests.post(url, json=payload, timeout=30)
                if response.status_code in [404, 429, 503]:
                    continue
                response.raise_for_status()
                
                data = response.json()
                if 'candidates' in data and data['candidates']:
                    candidate = data['candidates'][0]
                    # Check if model finished normally
                    if candidate.get('finishReason') == 'SAFETY':
                        logging.warning(f"Screener: Model '{model}' triggered safety filter.")
                        continue
                        
                    if 'content' in candidate and 'parts' in candidate['content']:
                        raw_response = candidate['content']['parts'][0]['text']
                        sql = raw_response.strip()
                        break 
            except Exception as e:
                logging.warning(f"Screener: Model '{model}' error: {e}")
                continue

        if not sql:
            logging.error(f"Screener: No SQL generated. Last response keys: {data.keys() if 'data' in locals() else 'None'}")
            return []
        
        logging.warning(f"Screener: AI Raw Output (Length {len(sql)}): {repr(sql)}")
        
        # --- ROBUST SQL EXTRACTION ---
        if "```" in sql:
            import re
            match = re.search(r'```(?:sql)?\s*(.*?)\s*```', sql, re.DOTALL | re.IGNORECASE)
            if match:
                sql = match.group(1).strip()
            else:
                sql = re.sub(r'```(?:sql)?', '', sql, flags=re.IGNORECASE).replace('```', '').strip()
        
        # AUTO-CORRECT common truncation
        if "screener_" in sql.lower() and "screener_cache" not in sql.lower():
            sql = sql.replace("screener_", "screener_cache")
            
        sql = sql.rstrip(';').strip()
        
        # Safety Check
        sql_lower = sql.lower()
        forbidden = ["delete", "drop", "update", "insert", "alter", "vacuum", "attach", "detach", "union", "join"]
        
        if not sql_lower.startswith("select") or "screener_cache" not in sql_lower or any(x in sql_lower for x in forbidden):
             logging.warning(f"Screener: Invalid AI SQL: {repr(sql)}")
             return []
             
        logging.info(f"Screener: Executing Code: {sql}")

        # Narrative search runs market-wide, so point at the shared screener
        # DB instead of the per-user portfolio DB returned by the default
        # get_db_connection() — that file is missing AI/IV rows for most
        # symbols.
        from db_utils import get_global_screener_db_path
        conn = get_db_connection(get_global_screener_db_path(), use_cache=False)
        if not conn:
             logging.error("Screener: DB Conn failed.")
             return []
        
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        
        results = []
        for row in rows:
            item = dict(row)
            # Re-calculate AI Score for frontend consistency
            vals = [v for v in [item.get("ai_moat"), item.get("ai_financial_strength"), item.get("ai_predictability"), item.get("ai_growth")] if isinstance(v, (int, float))]
            item["ai_score"] = sum(vals) / len(vals) if vals else None
            
            # Ensure catalysts is parsed
            cats = item.get("ai_catalysts")
            if cats and isinstance(cats, str):
                try:
                    item["ai_catalysts"] = json.loads(cats)
                except Exception:
                    pass
            
            results.append(item)
            
        conn.close()
        return results
        
    except Exception as e:
        logging.error(f"Screener: Narrative Search failed: {e}")
        return []
