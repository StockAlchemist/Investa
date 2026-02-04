import pandas as pd
import logging
import os
import io
import json
import time
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
    get_cached_screener_results,
    get_screener_results_by_universe
)
from server.ai_analyzer import generate_stock_review
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
        
        tables = pd.read_html(response.text)
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
    """Fetches Russell 2000 tickers from iShares IWM ETF."""
    return get_etf_holdings("239710", "IWM", "russell2000_tickers_cache.json")

def get_sp400_tickers() -> List[str]:
    """Fetches S&P MidCap 400 tickers from iShares IJH ETF."""
    return get_etf_holdings("239763", "IJH", "sp400_tickers_cache.json")

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
         conn = db_conn or get_db_connection()
         if conn:
            try:
                cached_results = get_screener_results_by_universe(conn, universe_tag)
                if cached_results:
                     logging.info(f"Screener: FAST MODE requested. Returning {len(cached_results)} cached items immediately.")
                     # Sort just in case
                     cached_results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
                     return cached_results
            except Exception as e:
                logging.error(f"Error in fast_mode retrieval: {e}")
            finally:
                if not db_conn and conn: conn.close()
         # If fast mode returns nothing, we return empty list immediately so UI can show loading state for FRESH load
         # Or should we fallback to slow load? 
         # The UI explicitly calls fast_mode=True then fast_mode=False. 
         # So we return [] here.
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
                 if not db_conn and conn: conn.close()
    elif universe_type == "sp500":
        symbols = get_sp500_tickers()
    elif universe_type == "russell2000":
        symbols = get_russell2000_tickers()
    elif universe_type == "sp400":
        symbols = get_sp400_tickers()
    
    if not symbols:
        return []
        
    # Recalculate tag just in case logic diverged? No, it's consistent.
    # universe_tag already set above.

    if universe_type in ["sp500", "russell2000", "sp400", "watchlist", "holdings"]:
        conn = db_conn or get_db_connection()
        if conn:
            try:
                raw_cached = get_screener_results_by_universe(conn, universe_tag)
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
                            # Increase TTL to 1 hour (3600s) for better "reload" experience
                            age_seconds = (now_ts - latest_update_ts).total_seconds()
                            if age_seconds < 3600: # Increased from 30s to 1 hour
                                logging.info(f"Screener: Pure Fast-path load for '{universe_tag}' ({len(cached_results)} items, {age_seconds:.0f}s old)")
                                use_pure_fast_path = True
                            else:
                                logging.info(f"Screener: Stale Fast-path found {len(cached_results)} items. Age: {age_seconds:.0f}s. Refreshing.")
            except Exception as e:
                logging.error(f"Error in hybrid fast-path: {e}")
            finally:
                if not db_conn and conn: conn.close()

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
    conn = db_conn if db_conn else get_db_connection()
    cached_map = {}
    if conn:
        try:
            cached_map = get_cached_screener_results(conn, symbols)
        except Exception as e:
            logging.error(f"Error loading screener cache map: {e}")
        finally:
            if not db_conn and conn: conn.close()
            
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
                valuation_details = None
                if valuation_details_str and isinstance(valuation_details_str, str):
                    try:
                        valuation_details = json.loads(valuation_details_str)
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
                            analysis_obj = ai_data.get("analysis", {})
                            scorecard = analysis_obj.get("scorecard", {})
                            
                            ai_moat = scorecard.get("moat")
                            ai_financial_strength = scorecard.get("financial_strength")
                            ai_predictability = scorecard.get("predictability")
                            ai_growth = scorecard.get("growth")
                            
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
                valuation_details = {}
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
                ai_cache_path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache", f"{sym.upper()}_analysis.json")
                has_ai = os.path.exists(ai_cache_path)
                
                ai_score = None
                ai_moat = None
                ai_fin = None
                ai_pred = None
                ai_growth = None
                ai_summary = None
                
                if has_ai:
                    try:
                        with open(ai_cache_path, 'r') as f:
                            ai_data = json.load(f)
                            analysis_obj = ai_data.get("analysis", {})
                            scorecard = analysis_obj.get("scorecard", {})
                            ai_summary = analysis_obj.get("summary")
                            ai_moat = scorecard.get("moat")
                            ai_fin = scorecard.get("financial_strength")
                            ai_pred = scorecard.get("predictability")
                            ai_growth = scorecard.get("growth")
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
        conn = db_conn or get_db_connection()
        if conn:
            try:
                upsert_screener_results(conn, results, universe_tag)
            except Exception as e:
                logging.error(f"Error upserting screener cache: {e}")
            finally:
                if not db_conn and conn: conn.close()

    # Sort by Margin of Safety desc
    results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
    
    return results
