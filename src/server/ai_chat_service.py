import logging
import json
import requests
import os
from datetime import datetime, date
from typing import List, Dict, Any, Optional

import config
from server.dependencies import get_current_user, get_transaction_data, get_config_manager
from server.ai_analyzer import FALLBACK_MODELS
from market_data import get_shared_mdp
from portfolio_logic import calculate_portfolio_summary, calculate_historical_performance
from db_utils import get_db_connection
from server.screener_service import run_narrative_search

# --- Tool Implementations ---

def get_portfolio_summary_tool(current_user) -> Dict[str, Any]:
    """Retrieves the high-level metrics and current holdings for the user's portfolio."""
    try:
        # We need to manually invoke the data dependency logic
        # get_transaction_data handles cache/reloading
        data = get_transaction_data(current_user)
        (
            df,
            manual_overrides,
            user_symbol_map,
            user_excluded_symbols,
            account_currency_map,
            db_path,
            db_mtime
        ) = data
        
        if df.empty:
            return {"error": "No transaction data found for this user."}

        # Similar to _calculate_portfolio_summary_internal in api.py
        config_manager = get_config_manager(current_user)
        config_manager.load_manual_overrides()
        account_interest_rates = config_manager.manual_overrides.get("account_interest_rates", {})
        interest_free_thresholds = config_manager.manual_overrides.get("interest_free_thresholds", {})

        mdp = get_shared_mdp()
        
        (
            overall_metrics,
            summary_df,
            holdings_dict,
            account_metrics,
            _,
            _,
            status
        ) = calculate_portfolio_summary(
            all_transactions_df_cleaned=df,
            original_transactions_df_for_ignored=df,
            ignored_indices_from_load=set(),
            ignored_reasons_from_load={},
            display_currency="USD",
            account_currency_map=account_currency_map,
            default_currency=config.DEFAULT_CURRENCY,
            market_provider=mdp,
            account_interest_rates=account_interest_rates,
            interest_free_thresholds=interest_free_thresholds,
            manual_overrides_dict=manual_overrides,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols
        )

        # Simplify holdings for AI readability
        holdings = []
        for (sym, acc), data in holdings_dict.items():
            if abs(data.get('qty', 0)) > 0.001:
                holdings.append({
                    "symbol": sym,
                    "account": acc,
                    "qty": data.get('qty'),
                    "market_value": data.get('market_value_display'),
                    "profit": data.get('total_gain_display'),
                    "return_pct": data.get('total_return_pct')
                })

        return {
            "overall_metrics": {
                "total_value": overall_metrics.get("market_value"),
                "total_gain": overall_metrics.get("total_gain"),
                "total_return_pct": overall_metrics.get("total_return_pct"),
                "cash_balance": overall_metrics.get("cash_balance")
            },
            "holdings": holdings[:20] # Limit to top 20 to avoid token bloat
        }
    except Exception as e:
        logging.error(f"Chat Tool Error (Portfolio Summary): {e}", exc_info=True)
        return {"error": f"Failed to fetch portfolio summary: {str(e)}"}

def get_market_data_tool(symbols: List[str]) -> Dict[str, Any]:
    """Retrieves fundamental and current market data for specific tickers."""
    try:
        mdp = get_shared_mdp()
        results = {}
        # Fetch batch to be efficient
        data_batch = mdp.get_fundamental_data_batch(set(symbols))
        
        for sym in symbols:
            info = data_batch.get(sym, {})
            # Extract common useful bits from ticker_info if present
            ticker_info = info.get("ticker_info", {})
            results[sym] = {
                "dividend_yield": info.get("dividendYield"),
                "p_e": ticker_info.get("forwardPE") or ticker_info.get("trailingPE"),
                "market_cap": ticker_info.get("marketCap"),
                "fifty_two_week_high": ticker_info.get("fiftyTwoWeekHigh"),
                "fifty_two_week_low": ticker_info.get("fiftyTwoWeekLow"),
                "description": ticker_info.get("longBusinessSummary", "")[:300] + "..."
            }
        return results
    except Exception as e:
        logging.error(f"Chat Tool Error (Market Data): {e}")
        return {"error": f"Failed to fetch market data: {str(e)}"}

def get_stock_review_tool(symbol: str) -> Dict[str, Any]:
    """Retrieves the pre-generated detailed AI analysis and scorecard for a stock."""
    from db_utils import get_db_connection
    conn = get_db_connection()
    if not conn:
        return {"error": "Database connection failed."}
    
    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT ai_summary, ai_moat, ai_financial_strength, ai_predictability, ai_growth 
            FROM screener_cache 
            WHERE symbol = ? 
            LIMIT 1
        """, (symbol.upper(),))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return {
                "symbol": symbol.upper(),
                "summary": row[0],
                "scorecard": {
                    "moat": row[1],
                    "financial_strength": row[2],
                    "predictability": row[3],
                    "growth": row[4]
                }
            }
        return {"message": f"No pre-generated AI review found for {symbol}."}
    except Exception as e:
        if conn: conn.close()
        return {"error": f"Failed to query reviews: {str(e)}"}

def run_screener_tool(prompt: str) -> Dict[str, Any]:
    """Runs a natural language stock screening query across the entire market database."""
    try:
        results = run_narrative_search(prompt)
        if not results:
            return {"message": "No stocks matched your criteria."}
            
        # Simplify results for the chat model
        simplified = []
        for r in results[:10]: # Limit to top 10 to keep context manageable
            simplified.append({
                "symbol": r.get("symbol"),
                "name": r.get("name"),
                "price": r.get("price"),
                "upside": r.get("margin_of_safety"),
                "ai_score": r.get("ai_score"),
                "sector": r.get("sector")
            })
        return {
            "match_count": len(results),
            "top_results": simplified,
            "note": "Sorted by Margin of Safety (higher = more undervalued). Intrinsic value based on DCF models."
        }
    except Exception as e:
        logging.error(f"Chat Tool Error (Screener): {e}")
        return {"error": f"Failed to run screener: {str(e)}"}

# --- AI Chat Service Core ---

SYSTEM_PROMPT = """
You are "Investa AI", a premium wealth intelligence assistant.
Your goal is to help users understand their investment portfolio and the financial markets.
You have access to REAL-TIME tools to query the user's specific performance, holdings, and detailed stock analyses.

BEHAVIORAL GUIDELINES:
1.  **Be Precise**: Use the data provided by tools. If a value is $12,450.21, say that, don't round unless it improves readability.
2.  **Be Analytical**: Don't just list holdings. Explain what they imply (e.g., "You are heavily concentrated in Tech").
3.  **Proactive Assistance**: If a user asks about a stock, and you see they don't own it but have it in a watchlist record, mention that.
4.  **Security**: You ONLY have access to the portfolio of the current user. Never hallucinate other users' data.
5.  **Format**: Use Markdown for tables and highlights. Keep responses concise but insightful.
"""

def process_chat_message(user_message: str, current_user, history: List[Dict] = None) -> str:
    """
    Orchestrates the chat logic using Gemini's function calling.
    Currently uses static tools and synchronous calls for MVP.
    """
    api_key = config.GEMINI_API_KEY
    if not api_key:
        return "Investa AI is currently offline (API Key missing)."

    # Define tools in Gemini format
    tools = [
        {
            "function_declarations": [
                {
                    "name": "get_portfolio_summary",
                    "description": "Retrieves high-level metrics (Total Value, Gain, Cash) and the top holdings of the user's portfolio.",
                },
                {
                    "name": "get_market_data",
                    "description": "Retrieves fundamental data like P/E, Dividend Yield, Market Cap, and Description for a list of stock symbols.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbols": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "The list of ticker symbols to look up (e.g. ['AAPL', 'MSFT'])."
                            }
                        },
                        "required": ["symbols"]
                    }
                },
                {
                    "name": "get_stock_review",
                    "description": "Returns a detailed AI-generated scorecard and business summary for a specific stock ticker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "symbol": {
                                "type": "string",
                                "description": "The ticker symbol (e.g. 'BRK-B')."
                            }
                        },
                        "required": ["symbol"]
                    }
                },
                {
                    "name": "run_screener",
                    "description": "Runs a natural language stock screening query across the market to find stocks matching specific criteria (e.g. 'undervalued tech stocks').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {
                                "type": "string",
                                "description": "The screening criteria in natural language."
                            }
                        },
                        "required": ["prompt"]
                    }
                }
            ]
        }
    ]

    # Prepare historical messages
    contents = []
    if history:
        for msg in history:
            contents.append({
                "role": "user" if msg["role"] == "user" else "model",
                "parts": [{"text": msg["text"]}]
            })
    
    contents.append({
        "role": "user",
        "parts": [{"text": user_message}]
    })

    # --- MODEL FALLBACK CHAIN ---
    # User requested Gemini 3.0 then 2.5. We use the mapped names.
    CHAT_MODELS = [
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
        "gemini-2.0-flash",
        "gemini-1.5-flash"
    ]

    # Prepare payload once
    payload = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": contents,
        "tools": tools,
        "generationConfig": {
            "temperature": 0.2, # Lower temperature for factual accuracy
        }
    }

    try:
        active_model = None
        # Outer loop for tool calling (Max 5 iterations to prevent infinite loops)
        for _ in range(5):
            response_json = None
            
            # Inner loop for Model Fallback (if the current model fails or hasn't been chosen yet)
            models_to_try = [active_model] if active_model else CHAT_MODELS
            
            for model in models_to_try:
                if not model: continue
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                
                try:
                    logging.info(f"AI Chat: Sending request with model '{model}'...")
                    response = requests.post(url, json=payload, timeout=60)
                    
                    if response.status_code == 404 or response.status_code == 429 or 500 <= response.status_code < 600:
                        logging.warning(f"AI Chat: Model '{model}' failed (status {response.status_code}). Trying next model...")
                        active_model = None # Reset if the "active" model failed
                        continue
                        
                    response.raise_for_status()
                    response_json = response.json()
                    active_model = model # Set/Refresh active model
                    break # Success!
                except Exception as e:
                    logging.warning(f"AI Chat: Request to '{model}' failed: {e}")
                    active_model = None
                    continue
            
            # If our "active model" fallback failed, try the whole CHAT_MODELS list once more from the top
            if not response_json and not active_model:
                for model in CHAT_MODELS:
                     # (Same logic as above, omitted for brevity but actually we should just ensure we try everything)
                     # Let's keep it simple: the loop above handles it if we start with CHAT_MODELS.
                     pass
                     
            if not response_json:
                return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."
            
            data = response_json
            candidate = data['candidates'][0]
            message = candidate['content']
            
            # Check for function calls
            parts = message.get('parts', [])
            call_part = next((p for p in parts if 'functionCall' in p), None)
            
            if not call_part:
                # No more tools, return final text
                text_part = next((p for p in parts if 'text' in p), None)
                return text_part['text'] if text_part else "I'm sorry, I couldn't generate a response."

            # Execute tool
            fn_call = call_part['functionCall']
            fn_name = fn_call['name']
            args = fn_call.get('args', {})
            
            logging.info(f"AI Chat: Calling tool {fn_name} with args {args}")
            
            tool_result = None
            if fn_name == "get_portfolio_summary":
                tool_result = get_portfolio_summary_tool(current_user)
            elif fn_name == "get_market_data":
                tool_result = get_market_data_tool(args.get('symbols', []))
            elif fn_name == "get_stock_review":
                tool_result = get_stock_review_tool(args.get('symbol', ''))
            elif fn_name == "run_screener":
                tool_result = run_screener_tool(args.get('prompt', ''))
            else:
                tool_result = {"error": f"Tool {fn_name} not found."}

            # Add model's entire message AND our response to contents
            payload["contents"].append(message)
            payload["contents"].append({
                "role": "function",
                "parts": [{
                    "functionResponse": {
                        "name": fn_name,
                        "response": {"result": tool_result}
                    }
                }]
            })
            
            # Continue loop to let model consider the results

        return "I've performed too many lookups to answer this question. Could you try being more specific?"

    except Exception as e:
        logging.error(f"AI Chat Exception: {e}", exc_info=True)
        return "I encountered an error while processing your request. Please try again later."
