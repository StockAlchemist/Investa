import requests
import json
import logging
import os
import time
from datetime import datetime
import config
from db_utils import (
    get_db_connection,
    get_cached_screener_results,
    get_global_screener_db_path,
)


def _open_shared_screener_conn():
    """Opens a connection to the global screener cache DB.

    Used as a fallback when no user-scoped ``db_conn`` is supplied (e.g. the
    chat tool, screener trigger endpoint, or background worker). Reading and
    writing against this DB — rather than the accidental default
    ``data/db/portfolio.db`` returned by ``get_db_connection()`` with no
    args — keeps AI reviews and IV/MoS rows on the one store that every user
    actually shares.
    """
    return get_db_connection(get_global_screener_db_path(), use_cache=False)

# --- Models and Fallback Configuration ---
# Models identified from user provided rate limits (gemini-3-flash, gemma-3, etc)
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-1.5-pro"
]

def generate_stock_review(
    symbol: str,
    fund_data: dict,
    ratios_data: dict,
    force_refresh: bool = False,
    use_search: bool = True,
    db_conn=None,
) -> dict:
    """
    Generates a comprehensive stock review using Gemini/Gemma models.
    Includes file-based caching and a fallback chain to handle rate limits.

    When ``db_conn`` is passed (the user's portfolio DB), screener-cache
    reads and writes go through it — and ``update_ai_review_in_cache``
    automatically mirrors writes to the global screener DB so every user
    sees the same review. When omitted (background workers, chat tool),
    we fall back to the global screener DB directly rather than the
    accidental ``data/db/portfolio.db`` that ``get_db_connection()`` with
    no args resolves to.
    """
    logging.info(f"AI Analysis: Generating review for {symbol} (force_refresh={force_refresh})")
    
    # helper for reconstruction
    def reconstruct_from_db(row: dict) -> dict:
        return {
            "scorecard": {
                "moat": row.get("ai_moat"),
                "financial_strength": row.get("ai_financial_strength"),
                "predictability": row.get("ai_predictability"),
                "growth": row.get("ai_growth")
            },
            "analysis": {
                "moat": row.get("ai_moat_analysis") or row.get("ai_moat"), # Some legacy rows might store text in ai_moat if schema was different, but usually separate now
                "financial_strength": row.get("ai_financial_strength_analysis") or row.get("ai_financial_strength"),
                "predictability": row.get("ai_predictability_analysis") or row.get("ai_predictability"),
                "growth_perspective": row.get("ai_growth_analysis") or row.get("ai_growth")
            },
            # If the DB stores the text directly in the scorecard fields (which it seems it does based on update_ai_review_in_cache), we handle that.
            # wait, update_ai_review_in_cache uses:
            # ai_moat = scorecard.get("moat")
            # ai_summary = ai_data.get("summary")
            # So the DB table has ai_moat, ai_financial_strength, etc. as the text/scores?
            # Let's check update_ai_review_in_cache again.
            "summary": row.get("ai_summary"),
            "sentiment": row.get("ai_sentiment"),
            "catalysts": json.loads(row.get("ai_catalysts")) if row.get("ai_catalysts") and isinstance(row.get("ai_catalysts"), str) else (row.get("ai_catalysts") or [])
        }
    
    # --- Caching Logic ---
    cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol.upper()}_analysis.json")
    
    stale_result = None
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                cache_time = cached_data.get("timestamp", 0)
                stale_result = cached_data.get("analysis", {})
                
                # --- SMART CACHE INVALIDATION ---
                # We only invalidate if force_refresh is True OR if we detect a change 
                # in the company's financial reporting period.
                is_cache_valid = True
                
                if not force_refresh:
                    # Check for FY/Quarter change if metadata is available
                    if fund_data and 'lastFiscalYearEnd' in fund_data:
                        # We might store the metadata in the cache_data for comparison
                        cached_fy = cached_data.get("metadata", {}).get("lastFiscalYearEnd")
                        cached_q = cached_data.get("metadata", {}).get("mostRecentQuarter")
                        
                        live_fy = fund_data.get("lastFiscalYearEnd")
                        live_q = fund_data.get("mostRecentQuarter")
                        
                        if live_fy and cached_fy and (live_fy != cached_fy or live_q != cached_q):
                            logging.info(f"AI Cache: New financial report detected (FY: {live_fy}, Q: {live_q}). Invalidating cache for {symbol}.")
                            is_cache_valid = False
                    
                    # Also check TTL (legacy fallback if no metadata found)
                    if is_cache_valid and (time.time() - cache_time > config.AI_REVIEW_CACHE_TTL):
                        logging.info(f"AI Cache: TTL expired for {symbol}. Attempting refresh.")
                        is_cache_valid = False

                if is_cache_valid and not force_refresh:
                    logging.info(f"Using cached AI analysis for {symbol}")

                    # Sync to DB if valid
                    if stale_result and "error" not in stale_result:
                        sync_conn = db_conn or _open_shared_screener_conn()
                        try:
                            if sync_conn:
                                from db_utils import update_ai_review_in_cache
                                update_ai_review_in_cache(sync_conn, symbol, stale_result, info=fund_data)
                        except Exception as e_sync:
                            logging.warning(f"AI Analysis: Failed to sync disk-cache to DB for {symbol}: {e_sync}")
                        finally:
                            # Only close conns we opened ourselves — user-supplied
                            # conns are owned by the FastAPI dependency layer.
                            if sync_conn is not None and sync_conn is not db_conn:
                                sync_conn.close()

                    return stale_result
        except Exception as e_cache:
            logging.warning(f"Failed to read cache for {symbol}: {e_cache}")

    # --- DB CACHE CHECK (Fallback/Primary for sync with screener) ---
    if not force_refresh:
        read_conn = db_conn or _open_shared_screener_conn()
        try:
            if read_conn:
                db_results = get_cached_screener_results(read_conn, [symbol])
                if symbol in db_results:
                    row = db_results[symbol]
                    if row.get("ai_summary") and len(row["ai_summary"]) > 20:
                        # Construct expected JSON structure
                        db_result = {
                            "scorecard": {
                                "moat": row.get("ai_moat") if isinstance(row.get("ai_moat"), (int, float)) else 0.0,
                                "financial_strength": row.get("ai_financial_strength") if isinstance(row.get("ai_financial_strength"), (int, float)) else 0.0,
                                "predictability": row.get("ai_predictability") if isinstance(row.get("ai_predictability"), (int, float)) else 0.0,
                                "growth": row.get("ai_growth") if isinstance(row.get("ai_growth"), (int, float)) else 0.0
                            },
                            "analysis": {
                                "moat": row.get("ai_moat") if isinstance(row.get("ai_moat"), str) else "N/A", 
                                "financial_strength": row.get("ai_financial_strength") if isinstance(row.get("ai_financial_strength"), str) else "N/A",
                                "predictability": row.get("ai_predictability") if isinstance(row.get("ai_predictability"), str) else "N/A",
                                "growth_perspective": row.get("ai_growth") if isinstance(row.get("ai_growth"), str) else "N/A"
                            },
                            "summary": row.get("ai_summary")
                        }
                        
                        # Use the smart helper if we trust the row structure more
                        # But based on the schema I saw, ai_moat/etc store the scores as REAL?
                        # Wait, PRAGMA said REAL. So they are scores.
                        # Then where is the text? screener_service.py says:
                        # results.append({ ... "ai_moat": ai_moat, ... "ai_summary": ai_summary ... })
                        # where ai_moat was scorecard.get("moat") which is a number?
                        # No, the screenshot shows detailed text!
                        # Let's check update_ai_review_in_cache again.
                        
                        logging.info(f"AI Cache: Found valid review in DB for {symbol}. Returning.")
                        return db_result
        except Exception as e_db_cache:
            logging.warning(f"AI Cache: Failed to read from DB for {symbol}: {e_db_cache}")
        finally:
            # Only close conns we opened ourselves
            if read_conn is not None and read_conn is not db_conn:
                read_conn.close()

    # --- API Logic ---
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in environment."}

    # Extract key metrics for the prompt
    metrics = {
        "Market Cap": fund_data.get("marketCap"),
        "Trailing P/E": fund_data.get("trailingPE"),
        "Forward P/E": fund_data.get("forwardPE"),
        "Dividend Yield": fund_data.get("dividendYield"),
        "ROE": ratios_data.get("Return on Equity (ROE) (%)"),
        "Gross Margin": ratios_data.get("Gross Profit Margin (%)"),
        "Net Margin": ratios_data.get("Net Profit Margin (%)"),
        "Debt/Equity": ratios_data.get("Debt-to-Equity Ratio"),
        "Current Ratio": ratios_data.get("Current Ratio"),
        "Beta": fund_data.get("beta"),
        "Summary": fund_data.get("longBusinessSummary", "No summary available")[:1000]
    }

    # Get current date for temporal context
    current_date_str = datetime.now().strftime("%B %d, %Y")

    prompt = f"""
You are evaluating {symbol} for a quality-and-value investor in the tradition of Buffett, Munger, and Phil Fisher. The investor wants a small number of great businesses bought below intrinsic value and held for a very long time. Analyse this business accordingly — do NOT frame your answer around short-term trading, technicals, momentum, or analyst price-target consensus.

Today is {current_date_str}.

Use the financial data below AND your web access to recent news, earnings, and disclosures.

Financial Data:
{json.dumps(metrics, indent=2)}

Score the following dimensions 1–10, where 10 is "this is the kind of business Buffett would own forever":

1. Moat — Durability of competitive advantage. Look for: structural cost advantages, network effects, switching costs, intangible assets (brand, regulation, IP), efficient scale. Penalise commodity-like businesses, narrow or fading advantages.
2. Financial Strength — Balance-sheet quality, debt levels relative to owner earnings, interest coverage, cash conversion. A strong moat means little if the balance sheet is fragile.
3. Predictability — Stability and predictability of revenue and free cash flow over the cycle. Cyclical, project-based, or commodity-exposed businesses score lower regardless of recent results.
4. Growth — Sustainable, capital-efficient growth in owner earnings. High-growth businesses that destroy capital (low ROIC, dilution) score lower than slow-growing businesses that compound capital at high rates of return.

In addition:
- Market Sentiment (0–100) — Read of current news and disclosures. Important for context, not a primary signal.
- Upcoming Catalysts — 2–3 concrete fundamental events (earnings, product/regulatory milestones, capital allocation events) occurring AFTER {current_date_str}.

In the analysis text:
- Cite specific business mechanics, not stock-price commentary.
- If you see signs of fundamental deterioration (margin compression, capital misallocation, governance issues), say so plainly.
- Do not recommend the position based on "technical breakout" or "analyst upgrades".
- The summary should be a one-or-two-sentence verdict on the BUSINESS — not the stock chart.

Return STRICT JSON only:
{{
  "scorecard": {{
    "moat": 0.0,
    "financial_strength": 0.0,
    "predictability": 0.0,
    "growth": 0.0
  }},
  "analysis": {{
    "moat": "...",
    "financial_strength": "...",
    "predictability": "...",
    "growth_perspective": "..."
  }},
  "summary": "One-or-two-sentence verdict on the business.",
  "sentiment": 50.0,
  "catalysts": [
    {{"event": "Event name", "date": "Estimated date", "impact": "High/Medium/Low"}}
  ]
}}
No markdown, no commentary outside the JSON.
"""

    # Base payload without tools
    base_payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "response_mime_type": "application/json"
        }
    }

    # --- FALLBACK CHAIN LOGIC ---
    import random
    initial_jitter = random.random() * 5 
    time.sleep(initial_jitter)

    for model in FALLBACK_MODELS:
        base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        
        # We try each model according to the user's strategy:
        # - Gemini 3: No search (to avoid 429s)
        # - Gemini 2.5: Search enabled (as first attempt), then fallback to no-search
        if "gemini-3" in model:
            search_options = [False]
        else:
            search_options = [True] if use_search else [False]
        
        for current_search_setting in search_options:
            logging.info(f"AI Analysis: Attempting review for {symbol} using model '{model}' (Search: {current_search_setting})...")
            
            # Create a model-specific and search-specific payload
            current_payload = json.loads(json.dumps(base_payload)) # Deep copy
            
            if current_search_setting:
                current_payload["tools"] = [{"google_search": {}}]
            
            # FIX: gemini-2.5 models do not support tools + json_mode together.
            # We drop json_mode and rely on the prompt's "STRICTLY as a JSON object" instruction.
            if "gemini-2.5" in model and current_search_setting:
                 current_payload["generationConfig"].pop("response_mime_type", None)

            try:
                response = requests.post(
                    f"{base_url}?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json=current_payload,
                    timeout=60
                )
                
                # Check for rate limits or server errors
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    logging.warning(f"AI Analysis: Model '{model}' (Search: {current_search_setting}) failed (status {response.status_code}). Response: {response.text[:200]}...")
                    # If this was with search, the inner loop will try without search.
                    # If this was already without search, we break to the next model.
                    continue 
                
                response.raise_for_status()
                
                data = response.json()
                content_text = data['candidates'][0]['content']['parts'][0]['text']
                
                # Robust JSON extraction: Find the substring between first '{' and last '}'
                try:
                    start_idx = content_text.find('{')
                    end_idx = content_text.rfind('}')
                    
                    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
                        json_str = content_text[start_idx : end_idx + 1]
                        result = json.loads(json_str)
                    else:
                        # Fallback if no braces found
                        result = json.loads(content_text)
                except json.JSONDecodeError as e:
                    logging.warning(f"JSON Parse Error for {model}: {e}. Content preview: {content_text[:100]}...")
                    continue # Try next option or next model
                
                # Save to cache
                try:
                    metadata = {
                        "lastFiscalYearEnd": fund_data.get("lastFiscalYearEnd"),
                        "mostRecentQuarter": fund_data.get("mostRecentQuarter"),
                        "generated_at": datetime.now().isoformat()
                    }
                    with open(cache_path, "w") as f:
                        json.dump({
                            "timestamp": time.time(), 
                            "analysis": result, 
                            "model_used": model,
                            "search_included": current_search_setting,
                            "metadata": metadata
                        }, f)
                    
                    # Sync to screener_cache table — uses the caller's user DB
                    # when supplied so update_ai_review_in_cache mirrors to the
                    # global screener DB; otherwise writes directly to global.
                    write_conn = db_conn or _open_shared_screener_conn()
                    if write_conn:
                        try:
                            from db_utils import update_ai_review_in_cache
                            update_ai_review_in_cache(write_conn, symbol, result, info=fund_data)
                        except Exception as e_db:
                            logging.warning(f"Failed to sync AI review to DB cache for {symbol}: {e_db}")
                        finally:
                            if write_conn is not db_conn:
                                write_conn.close()
                except Exception as e_save:
                    logging.warning(f"Failed to save cache for {symbol}: {e_save}")
                    
                return result
                
            except Exception as e:
                error_msg = str(e)
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                         error_msg += f" | Body: {e.response.text}"
                logging.warning(f"AI Analysis: Model '{model}' (Search: {current_search_setting}) failed completely. Error: {error_msg}")
                continue # Try next option or next model

    logging.error(f"AI Analysis: All models in fallback chain failed for {symbol}.")
    if stale_result:
        logging.info(f"AI Analysis: RETURNING STALE CACHE FOR {symbol} as last resort.")
        return stale_result
        
    return {"error": "AI Generation failed across all fallback models."}
