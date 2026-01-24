import pandas as pd
import logging
import os
import json
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
import requests

from market_data import get_shared_mdp
from financial_ratios import get_comprehensive_intrinsic_value
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

def get_sp500_tickers() -> List[str]:
    """
    Fetches the list of S&P 500 companies from Wikipedia.
    Uses caching to avoid repeated web requests.
    """
    cache_path = os.path.join(config.get_app_data_dir(), SP500_CACHE_FILE)
    
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

def screen_stocks(universe_type: str, universe_id: Optional[str] = None, manual_symbols: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Screens a list of stocks based on the specified universe.
    Calculates Intrinsic Value and Margin of Safety.
    """
    symbols = []
    
    if universe_type == "manual":
        symbols = manual_symbols or []
    elif universe_type == "watchlist":
        if universe_id:
             conn = get_db_connection()
             try:
                wl_data = get_watchlist(conn, int(universe_id)) # This returns list of Dict with 'symbol'
                if wl_data:
                    symbols = [item['Symbol'] for item in wl_data]
             except Exception as e:
                 logging.error(f"Error fetching watchlist {universe_id}: {e}")
                 return []
             finally:
                 if conn: conn.close()
    elif universe_type == "sp500":
        symbols = get_sp500_tickers()
    
    if not symbols:
        return []

    universe_tag = universe_type
    if universe_type == "watchlist" and universe_id:
        universe_tag = f"watchlist_{universe_id}"
        
    # --- SMART HYBRID FAST-PATH ---
    # Goal: If we have a fresh cache (< 10 mins), return instantly.
    # If we have a stale-today cache (> 10 mins), only refresh prices.
    cached_results = []
    use_pure_fast_path = False
    
    if universe_type in ["sp500", "watchlist"]:
        conn = get_db_connection()
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
                            if age_seconds < 30: # Reduced from 1h for transition
                                logging.info(f"Screener: Pure Fast-path load for '{universe_tag}' ({len(cached_results)} items, {age_seconds:.0f}s old)")
                                use_pure_fast_path = True
                            else:
                                logging.info(f"Screener: Stale Fast-path found {len(cached_results)} items. Age: {age_seconds:.0f}s. Refreshing.")
            except Exception as e:
                logging.error(f"Error in hybrid fast-path: {e}")
            finally:
                if conn: conn.close()

    if use_pure_fast_path:
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
    conn = get_db_connection()
    cached_map = {}
    if conn:
        try:
            cached_map = get_cached_screener_results(conn, symbols)
        except Exception as e:
            logging.error(f"Error loading screener cache map: {e}")
        finally:
            if conn: conn.close()
            
    # Process everything to merge live quotes
    final_results = process_screener_results(symbols, quotes, details_map, cached_map, universe_tag)
    
    # Final Sort
    final_results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
    
    return final_results

def process_screener_results(
    symbols: List[str], 
    quotes: Dict[str, Any], 
    details_map: Dict[str, Any],
    cached_map: Dict[str, Dict[str, Any]] = None,
    universe_tag: str = None
) -> List[Dict[str, Any]]:
    results = []
    cached_map = cached_map or {}
    
    # Pre-scan AI review directory to avoid 500+ repeated disk lookups (slow on OneDrive)
    ai_cache_dir = os.path.join(config.get_app_data_dir(), "ai_analysis_cache")
    existing_reviews = set()
    if os.path.exists(ai_cache_dir):
        try:
            # Get list of symbols that have a .json review
            files = os.listdir(ai_cache_dir)
            existing_reviews = {f.split("_")[0].upper() for f in files if f.endswith("_analysis.json")}
        except Exception:
            pass

    for sym in symbols:
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
            continue
            
        # --- SMART INVALIDATION LOGIC ---
        cached = cached_map.get(sym)
        
        # Current report identifiers
        live_fy_end = info.get("lastFiscalYearEnd")
        live_quarter = info.get("mostRecentQuarter")
        
        can_use_cache = False
        if cached:
            # If the financial report identifiers haven't changed, 
            # we can reuse the cached intrinsic value and AI scores.
            if (cached.get("last_fiscal_year_end") == live_fy_end and 
                cached.get("most_recent_quarter") == live_quarter):
                can_use_cache = True
        
        if can_use_cache:
            avg_iv = cached.get("intrinsic_value")
            valuation_details_str = cached.get("valuation_details")
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
            
            # Use cached AI markers
            has_ai = cached.get("ai_summary") is not None
            ai_score = cached.get("ai_score")
            
            # --- FALLBACK: If DB has no AI info, check pre-scanned list ---
            if not has_ai or ai_score is None:
                if sym.upper() in existing_reviews:
                    ai_cache_path = os.path.join(ai_cache_dir, f"{sym.upper()}_analysis.json")
                    if os.path.exists(ai_cache_path):
                        try:
                            with open(ai_cache_path, 'r') as f:
                                ai_data = json.load(f)
                                analysis_obj = ai_data.get("analysis", {})
                                scorecard = analysis_obj.get("scorecard", {})
                                
                                moat = scorecard.get("moat")
                                fin = scorecard.get("financial_strength")
                                pred = scorecard.get("predictability")
                                growth = scorecard.get("growth")
                                
                                vals = [v for v in [moat, fin, pred, growth] if isinstance(v, (int, float))]
                                if vals:
                                    ai_score = sum(vals) / len(vals)
                                    has_ai = True
                        except Exception:
                            pass

            results.append({
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
                "valuation_details": valuation_details, # Pass through loaded details
                "last_fiscal_year_end": live_fy_end,
                "most_recent_quarter": live_quarter
            })
        else:
            # Report changed or no cache -> Full Re-calculation
            # Now with improved growth estimation by passing 'info' (ticker_info)
            iv_res = get_comprehensive_intrinsic_value(info, None, None, None)
            
            avg_iv = iv_res.get("average_intrinsic_value")
            mos = iv_res.get("margin_of_safety_pct")
            
            # Serialize breakdown for storage
            valuation_details_json = json.dumps(iv_res, default=str)
            
            # AI Review Check (File based for compatibility with existing analyzer)
            ai_cache_path = os.path.join(config.get_app_data_dir(), "ai_analysis_cache", f"{sym.upper()}_analysis.json")
            has_ai = os.path.exists(ai_cache_path)
            
            # Try to get AI score from JSON if it exists
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
                        # Scorecard is nested under "analysis" in the stored JSON
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

            results.append({
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
                "valuation_details": valuation_details_json, # Save strict JSON string
                "last_fiscal_year_end": live_fy_end,
                "most_recent_quarter": live_quarter
            })
        
    # Batch Update DB Cache
    if results:
        conn = get_db_connection()
        if conn:
            try:
                upsert_screener_results(conn, results, universe_tag)
            except Exception as e:
                logging.error(f"Error upserting screener cache: {e}")
            finally:
                if conn: conn.close()

    # Sort by Margin of Safety desc
    results.sort(key=lambda x: (x.get("margin_of_safety") is not None, x.get("margin_of_safety")), reverse=True)
    
    return results
