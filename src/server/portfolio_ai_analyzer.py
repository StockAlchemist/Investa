import logging
import json
import os
import hashlib
from datetime import datetime
import config
from server.ai_analyzer import FALLBACK_MODELS
import requests

# --- Helpers ---

def _compute_portfolio_hash(portfolio_data: dict) -> str:
    """Computes a stable hash of the portfolio state for caching."""
    # We use holdings and total value to detect changes.
    # We sort holdings to ensure order doesn't matter.
    
    # Simple approach: Create a string rep of key components
    key_components = []
    
    # 1. Total Value (rounded to nearest 100 to avoid noise? Or just exact. Let's use int Cast)
    metrics = portfolio_data.get("metrics", {})
    key_components.append(str(int(metrics.get("market_value", 0))))
    
    # 2. Holdings (Symbol, Qty)
    holdings_dict = portfolio_data.get("holdings_dict", {})
    # holdings_dict keys are strings "Symbol|Account" if coming from API serialisation, 
    # but inside API it is (Symbol, Account) tuple.
    # We expect this function to be called from API with the raw dictionary.
    
    sorted_holdings = sorted(holdings_dict.items()) # Sort by key (Sym, Acc)
    for k, v in sorted_holdings:
        # k is (Symbol, Account)
        # v is dict with 'qty', etc.
        qty = v.get('qty', 0)
        if abs(qty) > 0.01: # Ignore tiny dust
            key_components.append(f"{k[0]}:{k[1]}:{qty:.2f}")
            
    # 3. Date (Portfolio state is valid for a day essentially)
    key_components.append(datetime.now().strftime("%Y-%m-%d"))
    
    combined_str = "|".join(key_components)
    return hashlib.md5(combined_str.encode()).hexdigest()

def generate_portfolio_review(portfolio_data: dict, risk_metrics: dict, force_refresh: bool = False) -> dict:
    """
    Generates an AI review for the entire portfolio.
    
    Args:
        portfolio_data (dict): The result from calculate_portfolio_summary
        risk_metrics (dict): The result from risk_metrics.calculate_all_risk_metrics
        force_refresh (bool): Whether to bypass cache
        
    Returns:
        dict: The AI review JSON (scorecard + analysis)
    """
    logging.info("AI Analysis: Generating PORTFOLIO review...")
    
    # 1. Check Cache
    cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "portfolio_ai_cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    pf_hash = _compute_portfolio_hash(portfolio_data)
    cache_path = os.path.join(cache_dir, f"pf_{pf_hash}.json")
    
    if not force_refresh and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
                # Check TTL (e.g. 24 hours) - though hash includes date so it auto-expires daily
                logging.info("AI Analysis: Using cached portfolio review.")
                return cached
        except Exception as e:
            logging.warning(f"Failed to read portfolio cache: {e}")

    # 2. Prepare Data for Prompt
    metrics = portfolio_data.get("metrics", {})
    
    # Top Holdings Calculation
    holdings_raw = portfolio_data.get("holdings_dict", {})
    
    # Removed unused holdings iteration that did not compute aggregated positions.
    
    # We will pass the TOP 10 positions by weight if possible.
    # Since we might not have prices easily here without reprocessing, 
    # we can rely on what's available or ask the caller to pass a simplified view.
    # For now, let's assume we can dump the 'metrics' and 'risk_metrics' which are high level.
    
    # 2. Extract key metrics
    # Map from API portfolio summary keys to analyzer keys
    metrics = portfolio_data.get('metrics', {})
    total_value = metrics.get('market_value', 0)
    total_change = metrics.get('total_gain', 0)
    total_change_percent = metrics.get('total_return_pct', 0)

    # Risk Metrics - Map keys from risk_metrics.py
    # Keys seen in log: 'Max Drawdown', 'Volatility (Ann.)', 'Sharpe Ratio', 'Sortino Ratio'
    sharpe = risk_metrics.get('Sharpe Ratio', risk_metrics.get('sharpe_ratio', 'N/A'))
    sortino = risk_metrics.get('Sortino Ratio', risk_metrics.get('sortino_ratio', 'N/A'))
    volatility = risk_metrics.get('Volatility (Ann.)', risk_metrics.get('volatility', 'N/A'))
    max_drawdown = risk_metrics.get('Max Drawdown', risk_metrics.get('max_drawdown', 'N/A'))
    beta = risk_metrics.get('Beta', risk_metrics.get('beta', 'N/A')) # Beta might be missing in pure portfolio stats
    alpha = risk_metrics.get('Alpha', risk_metrics.get('alpha', 'N/A')) 

    # Asset Allocation (if available)
    holdings = portfolio_data.get('holdings', [])
    # Group by sector if available, otherwise just list top holdings
    
    holdings_summary = ""
    if holdings:
        # Sort by value desc (using 'market_value' or 'current_value')
        # Check potential keys: 'market_value', 'value', 'total_value'
        # Also 'allocation_percent' might be 'percent_portfolio' or similar
        sorted_holdings = sorted(holdings, key=lambda x: x.get('market_value', x.get('value', 0)), reverse=True)
        top_5 = sorted_holdings[:5]
        holdings_summary = "Top 5 Holdings:\n"
        for h in top_5:
            symbol = h.get('symbol', 'Unknown')
            # Calculate percent if missing: value / total_portfolio_value
            val = h.get('market_value', h.get('value', 0))
            pct = h.get('allocation_percent', h.get('percent', 0))
            if pct == 0 and total_value > 0:
                pct = (val / total_value) * 100
            
            sector = h.get('sector', 'Unknown Sector')
            holdings_summary += f"- {symbol}: {pct:.2f}% ({sector})\n"
    else:
        holdings_summary = "Holdings data not explicit."

    # 3. Construct Prompt
    prompt = f"""
    You are an expert financial advisor. Review the following investment portfolio and provide a comprehensive analysis.
    
    PORTFOLIO METRICS:
    - Total Value: {total_value:,.2f}
    - Total Change: {total_change:,.2f} ({total_change_percent:.2f}%)
    - Timeframe: Past 1 Year (for risk metrics)
    
    RISK METRICS:
    - Sharpe Ratio: {sharpe}
    - Sortino Ratio: {sortino}
    - Annualized Volatility: {volatility}
    - Max Drawdown: {max_drawdown}
    - Beta: {beta}
    - Alpha: {alpha}
    
    ASSET ALLOCATION:
    {holdings_summary}
    
    INSTRUCTIONS:
    1. Score the portfolio (1-10) on Diversification, Risk Profile, and Performance.
    2. Provide an Executive Summary.
    3. Analyze Diversification, Risk, and Performance separately in detail. 
       - For each category, explicitly explain WHY you assigned the specific score (e.g., "Score: 7/10 because..."). This explanation will be shown in a tooltip, so be clear and concise.
    4. Provide 3-5 concrete, actionable recommendations.
       - CRITICAL: Include at least one rebalancing idea that maintains the current risk and return profile of the portfolio (e.g., swapping a high-beta stock for another high-growth stock in a different sector).
    
    OUTPUT FORMAT (JSON):
    {{
        "scorecard": {{
            "diversification": <score>,
            "risk_profile": <score>,
            "performance": <score>
        }},
        "summary": "<executive_summary>",
        "analysis": {{
            "diversification": "<text>",
            "risk_profile": "<text>",
            "performance": "<text>",
            "actionable_recommendations": "<text>"
        }},
        "recommendations": ["<rec1>", "<rec2>", "..."]
    }}
    """
    
    # 3. Call LLM (using Fallback Chain from ai_analyzer)
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return {
            "scorecard": {
                "diversification": 0,
                "risk_profile": 0,
                "performance": 0
            },
            "summary": "Unable to generate analysis. Error: API Key is missing.",
            "analysis": {
                "diversification": "Analysis unavailable.",
                "risk_profile": "Analysis unavailable.",
                "performance": "Analysis unavailable.",
                "actionable_recommendations": "No recommendations available."
            },
            "recommendations": []
        }
        
    base_payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"response_mime_type": "application/json"}
    }
    
    # Re-use the fallback logic? 
    # To avoid code duplication, we could import a helper, but ai_analyzer's logic is embedded in the function.
    # I'll implement a simplified version here or refactor. 
    # For speed, I'll copy the robust loop structure.
    
    rate_limit_count = 0
    other_error_count = 0
            
    for model in FALLBACK_MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        try:
            # Portfolio review doesn't need Google Search usually (internal data), 
            # so we skip 'tools' to save latency/quota, unless user explicitly wants market context?
            # Let's stick to base_payload (no search) for now to be fast.
            
            resp = requests.post(url, json=base_payload, timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                text = data['candidates'][0]['content']['parts'][0]['text']
                
                # Try to parse JSON
                # Clean markdown
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]
                    
                result = json.loads(text)
                
                # Ensure keys exist
                default_scorecard = {"diversification": 0, "risk_profile": 0, "performance": 0}
                default_analysis = {
                    "diversification": "Analysis unavailable.",
                    "risk_profile": "Analysis unavailable.",
                    "performance": "Analysis unavailable.",
                    "actionable_recommendations": "No recommendations available."
                }
                
                if "scorecard" not in result: result["scorecard"] = default_scorecard
                if "analysis" not in result: result["analysis"] = default_analysis
                if "summary" not in result: result["summary"] = "AI analysis generated, but summary missing."
                if "recommendations" not in result: result["recommendations"] = []
                
                # Cache it
                with open(cache_path, "w") as f:
                    json.dump(result, f)
                    
                return result
            elif resp.status_code == 429:
                logging.warning(f"Portfolio AI: Model {model} rate limited.")
                rate_limit_count += 1
                continue
            else:
                other_error_count += 1
                logging.warning(f"Portfolio AI: Model {model} error {resp.status_code}: {resp.text}")
                
        except Exception as e:
            logging.error(f"Portfolio AI: Error with {model}: {e}")
            other_error_count += 1
            continue
            
    if rate_limit_count > 0 and other_error_count == 0:
         # Try to fallback to cache if available
         if os.path.exists(cache_path):
             try:
                 with open(cache_path, "r") as f:
                     cached = json.load(f)
                     cached["warning"] = "RateLimit"
                     cached["message"] = "AI service is busy. Showing cached analysis from earlier."
                     logging.info("AI Analysis: Rate limited, falling back to cache.")
                     return cached
             except Exception as e:
                 logging.warning(f"Failed to read portfolio cache for fallback: {e}")

         return {
             "error": "RateLimit",
             "message": "AI service usage limit reached. Please try again in a few minutes or check your quota."
         }

    return {"error": "Failed to generate portfolio review."}

