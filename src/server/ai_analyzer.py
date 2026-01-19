import requests
import json
import logging
import os
import time
import re
from datetime import datetime
import config

GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-3-flash-preview:generateContent"

def generate_stock_review(symbol: str, fund_data: dict, ratios_data: dict) -> dict:
    """
    Generates a comprehensive stock review using Gemini 3 Pro.
    Includes file-based caching to minimize API hits.
    """
    # --- Caching Logic ---
    cache_dir = os.path.join(config.get_app_data_dir(), "ai_analysis_cache")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{symbol.upper()}_analysis.json")
    
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)
                cache_time = cached_data.get("timestamp", 0)
                
                # --- SMART CACHE INVALIDATION ---
                # Check if we have fresher fundamental data (e.g. from earnings release)
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
                    return cached_data.get("analysis", {})
        except Exception as e_cache:
            logging.warning(f"Failed to read cache for {symbol}: {e_cache}")

    # --- API Logic ---
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return {"error": "GEMINI_API_KEY not found in environment."}

    # Extract key metrics for the prompt to allow data-driven analysis
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
        "Summary": fund_data.get("longBusinessSummary", "No summary available")[:1000] # Cap summary length
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
    "moat": 8.5,
    "financial_strength": 9.0,
    "predictability": 7.5,
    "growth": 8.0
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

    try:
        response = requests.post(
            f"{GEMINI_API_URL}?key={api_key}",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        data = response.json()
        # Parse the JSON string from the response
        content_text = data['candidates'][0]['content']['parts'][0]['text']
        
        # Strip potential markdown backticks if any (redundant if response_mime_type is set but good for safety)
        if content_text.startswith("```json"):
            content_text = content_text.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content_text)
        
        # Save to cache
        try:
            with open(cache_path, "w") as f:
                json.dump({"timestamp": time.time(), "analysis": result}, f)
        except Exception as e_save:
            logging.warning(f"Failed to save cache for {symbol}: {e_save}")
            
        return result
        
    except Exception as e:
        error_msg = str(e)
        
        # 1. Redact key=... parameters in URLs
        error_msg = re.sub(r'key=[^& \n\'\"]+', 'key=REDACTED', error_msg)
        
        # 2. Redact the exact key if it's leaked as a standalone string
        if api_key and len(api_key) > 10: # Minimum length to avoid redacting single chars
            error_msg = error_msg.replace(api_key, "REDACTED")
            
        logging.error(f"Gemini API Error for {symbol}: {error_msg}")
        return {"error": f"AI Generation failed: {error_msg}"}
