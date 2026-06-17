import requests
import json
import logging
import os
import time
from datetime import datetime
import config
from db_utils import get_cached_screener_results

# --- Models and Fallback Configuration ---
# Models identified from user provided rate limits (gemini-3-flash, gemma-3, etc)
FALLBACK_MODELS = [
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash"
]

def generate_stock_review(
    symbol: str,
    fund_data: dict,
    ratios_data: dict,
    force_refresh: bool = False,
    use_search: bool = True,
) -> dict:
    """
    Generates a comprehensive stock review using Gemini/Gemma models.
    Includes file-based caching and a fallback chain to handle rate limits.

    All screener-cache reads/writes go through the global screener DB.
    """
    logging.info(f"AI Analysis: Generating review for {symbol} (force_refresh={force_refresh})")
    
    # helper for reconstruction
    def reconstruct_from_db(row: dict) -> dict:
        catalysts_raw = row.get("ai_catalysts")
        if catalysts_raw and isinstance(catalysts_raw, str):
            try:
                catalysts = json.loads(catalysts_raw)
            except json.JSONDecodeError:
                catalysts = []
        else:
            catalysts = catalysts_raw or []

        def _score(v):
            return v if isinstance(v, (int, float)) else 0.0

        def _text(v):
            return v if isinstance(v, str) and v.strip() else "N/A"

        return {
            "scorecard": {
                "moat": _score(row.get("ai_moat")),
                "financial_strength": _score(row.get("ai_financial_strength")),
                "predictability": _score(row.get("ai_predictability")),
                "growth": _score(row.get("ai_growth")),
            },
            "analysis": {
                "moat": _text(row.get("ai_moat_analysis")),
                "financial_strength": _text(row.get("ai_financial_strength_analysis")),
                "predictability": _text(row.get("ai_predictability_analysis")),
                "growth_perspective": _text(row.get("ai_growth_perspective_analysis")),
            },
            "summary": row.get("ai_summary"),
            "sentiment": row.get("ai_sentiment"),
            "catalysts": catalysts,
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

                    if stale_result and "error" not in stale_result:
                        try:
                            from db_utils import update_ai_review_in_cache
                            update_ai_review_in_cache(symbol, stale_result, info=fund_data)
                        except Exception as e_sync:
                            logging.warning(f"AI Analysis: Failed to sync disk-cache to DB for {symbol}: {e_sync}")

                    return stale_result
        except Exception as e_cache:
            logging.warning(f"Failed to read cache for {symbol}: {e_cache}")

    # --- DB CACHE CHECK (Fallback/Primary for sync with screener) ---
    if not force_refresh:
        try:
            db_results = get_cached_screener_results([symbol])
            if symbol in db_results:
                row = db_results[symbol]
                if row.get("ai_summary") and len(row["ai_summary"]) > 20:
                    db_result = reconstruct_from_db(row)
                    # If the row pre-dates the per-dimension text columns, all
                    # four narratives will be "N/A". Treat that as a cache miss
                    # so we regenerate and backfill the DB with real text.
                    analysis = db_result.get("analysis", {})
                    if any(v and v != "N/A" for v in analysis.values()):
                        logging.info(f"AI Cache: Found valid review in DB for {symbol}. Returning.")
                        return db_result
                    logging.info(
                        f"AI Cache: DB row for {symbol} lacks per-dimension narratives; regenerating to backfill."
                    )
        except Exception as e_db_cache:
            logging.warning(f"AI Cache: Failed to read from DB for {symbol}: {e_db_cache}")

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

    quota_exhausted = False

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
                    if "exceeded your current quota" in response.text:
                        logging.warning(f"Hard quota limit reached for {model}. Stopping fallback to prevent charges.")
                        quota_exhausted = True
                        break
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
                    
                    # Sync to global screener cache
                    try:
                        from db_utils import update_ai_review_in_cache
                        update_ai_review_in_cache(symbol, result, info=fund_data)
                    except Exception as e_db:
                        logging.warning(f"Failed to sync AI review to DB cache for {symbol}: {e_db}")
                except Exception as e_save:
                    logging.warning(f"Failed to save cache for {symbol}: {e_save}")
                    
                return result
                
            except Exception as e:
                error_msg = str(e)
                if isinstance(e, requests.exceptions.HTTPError) and e.response is not None:
                         error_msg += f" | Body: {e.response.text}"
                logging.warning(f"AI Analysis: Model '{model}' (Search: {current_search_setting}) failed completely. Error: {error_msg}")
                continue # Try next option or next model

        if quota_exhausted:
            break

    logging.error(f"AI Analysis: All models in fallback chain failed for {symbol}.")
    if stale_result:
        logging.info(f"AI Analysis: RETURNING STALE CACHE FOR {symbol} as last resort.")
        return stale_result
        
    if quota_exhausted:
        return {"error": "QUOTA_EXHAUSTED"}
        
    return {"error": "AI Generation failed across all fallback models."}
