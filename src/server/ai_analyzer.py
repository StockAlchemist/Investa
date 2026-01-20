import requests
import json
import logging
import os
import time
import re
from datetime import datetime
import config
from db_utils import get_db_connection, update_ai_review_in_cache

# --- Models and Fallback Configuration ---
# Models identified from user provided rate limits (gemini-3-flash, gemma-3, etc)
FALLBACK_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemma-3-27b",
    "gemma-3-12b"
]

def generate_stock_review(symbol: str, fund_data: dict, ratios_data: dict, force_refresh: bool = False) -> dict:
    """
    Generates a comprehensive stock review using Gemini/Gemma models.
    Includes file-based caching and a fallback chain to handle rate limits.
    """
    logging.info(f"AI Analysis: Generating review for {symbol} (force_refresh={force_refresh})")
    
    # --- Caching Logic ---
    cache_dir = os.path.join(config.get_app_data_dir(), "ai_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol.upper()}_analysis.json")
    
    if os.path.exists(cache_path) and not force_refresh:
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                cache_time = cached_data.get("timestamp", 0)
                
                # --- SMART CACHE INVALIDATION ---
                is_cache_valid = True
                if fund_data and '_fetch_timestamp' in fund_data:
                    try:
                        fund_ts_str = fund_data['_fetch_timestamp']
                        fund_ts = datetime.fromisoformat(fund_ts_str).timestamp()
                        if fund_ts > cache_time:
                            logging.info(f"AI Cache: Fundamental data (TS: {fund_ts}) is newer than AI analysis (TS: {cache_time}). Invalidating AI cache.")
                            is_cache_valid = False
                    except Exception as e_ts:
                        logging.warning(f"Error comparing timestamps for AI cache: {e_ts}")

                if is_cache_valid and (time.time() - cache_time < config.AI_REVIEW_CACHE_TTL):
                    logging.info(f"Using cached AI analysis for {symbol}")
                    analysis_result = cached_data.get("analysis", {})
                    
                    # Sync to DB if valid
                    if analysis_result and "error" not in analysis_result:
                        try:
                            conn = get_db_connection()
                            if conn:
                                from db_utils import update_ai_review_in_cache
                                update_ai_review_in_cache(conn, symbol, analysis_result, info=fund_data)
                                conn.close()
                        except Exception as e_sync:
                            logging.warning(f"AI Analysis: Failed to sync disk-cache to DB for {symbol}: {e_sync}")
                            
                    return analysis_result
        except Exception as e_cache:
            logging.warning(f"Failed to read cache for {symbol}: {e_cache}")

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

    prompt = f"""
Analyze the stock {symbol} based on the following financial data and business summary:
{json.dumps(metrics, indent=2)}

Provide a professional investment review covering these specific topics:
1. Moat: Competitive advantages, pricing power, and market position.
2. Financial Strength: Balance sheet quality, debt levels, and solvency.
3. Predictability: Reliability of earnings, historical consistency, and business cyclicality.
4. Growth Perspective: Future growth drivers, market opportunities, and potential risks.

For each topic, provide a score from 1 to 10.

Return the response STRICTLY as a JSON object with the following structure:
{{
  "scorecard": {{
    "moat": 0.0,
    "financial_strength": 0.0,
    "predictability": 0.0,
    "growth": 0.0
  }},
  "analysis": {{
    "moat": "Analysis text here...",
    "financial_strength": "Analysis text here...",
    "predictability": "Analysis text here...",
    "growth_perspective": "Analysis text here..."
  }},
  "summary": "Overall conclusion in one or two sentences."
}}
Do not include any other markdown formatting or explanations outside the JSON block.
"""

    payload = {
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
        max_retries_per_model = 3
        base_delay = 5 # Reduced base delay since we have fallbacks
        
        logging.info(f"AI Analysis: Attempting review for {symbol} using model '{model}'...")
        
        for attempt in range(max_retries_per_model):
            try:
                response = requests.post(
                    f"{base_url}?key={api_key}",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=60
                )
                
                # Check for rate limits or server errors
                if response.status_code == 429 or 500 <= response.status_code < 600:
                    if attempt < max_retries_per_model - 1:
                        delay = (base_delay * (2 ** attempt)) + (random.random() * 2)
                        logging.warning(f"AI Analysis: Model '{model}' rate limited/error ({response.status_code}). Retrying in {delay:.1f}s...")
                        time.sleep(delay)
                        continue
                    else:
                        logging.warning(f"AI Analysis: Model '{model}' exhausted after {max_retries_per_model} attempts. Falling back...")
                        break # Try next model
                
                response.raise_for_status()
                
                data = response.json()
                content_text = data['candidates'][0]['content']['parts'][0]['text']
                
                if content_text.startswith("```json"):
                    content_text = content_text.replace("```json", "").replace("```", "").strip()
                
                result = json.loads(content_text)
                
                # Save to cache
                try:
                    with open(cache_path, "w") as f:
                        json.dump({"timestamp": time.time(), "analysis": result, "model_used": model}, f)
                    
                    # Sync to screener_cache table in DB
                    conn = get_db_connection()
                    if conn:
                        try:
                            from db_utils import update_ai_review_in_cache
                            update_ai_review_in_cache(conn, symbol, result, info=fund_data)
                        except Exception as e_db:
                            logging.warning(f"Failed to sync AI review to DB cache for {symbol}: {e_db}")
                        finally:
                            if conn: conn.close()
                except Exception as e_save:
                    logging.warning(f"Failed to save cache for {symbol}: {e_save}")
                    
                return result
                
            except Exception as e:
                if attempt < max_retries_per_model - 1:
                    delay = (base_delay * (2 ** attempt)) + (random.random() * 2)
                    logging.warning(f"AI Analysis: transient error for '{model}': {e}. Retrying in {delay:.1f}s...")
                    time.sleep(delay)
                    continue
                else:
                    logging.warning(f"AI Analysis: Model '{model}' failed completely. Falling back...")
                    break # Try next model

    logging.error(f"AI Analysis: All models in fallback chain failed for {symbol}.")
    return {"error": "AI Generation failed across all fallback models."}
