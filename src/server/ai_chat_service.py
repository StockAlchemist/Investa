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

def get_stock_review_tool(symbol: str, db_conn=None) -> Dict[str, Any]:
    """Retrieves the pre-generated detailed AI analysis and scorecard for a stock.

    Reads through ``get_cached_screener_results`` so that — when the user's
    portfolio DB has no row for ``symbol`` — the lookup falls back to the
    global screener DB. The previous raw SELECT against the default DB
    returned by ``get_db_connection()`` (no args) silently missed reviews
    that lived only in the shared global store.
    """
    from db_utils import get_cached_screener_results, get_db_connection, get_global_screener_db_path

    own_conn = False
    conn = db_conn
    if conn is None:
        conn = get_db_connection(get_global_screener_db_path(), use_cache=False)
        own_conn = True
    if not conn:
        return {"error": "Database connection failed."}

    try:
        rows = get_cached_screener_results(conn, [symbol.upper()])
        row = rows.get(symbol.upper())

        if row and row.get("ai_summary"):
            return {
                "symbol": symbol.upper(),
                "summary": row.get("ai_summary"),
                "scorecard": {
                    "moat": row.get("ai_moat"),
                    "financial_strength": row.get("ai_financial_strength"),
                    "predictability": row.get("ai_predictability"),
                    "growth": row.get("ai_growth"),
                },
            }
        return {"message": f"No pre-generated AI review found for {symbol}."}
    except Exception as e:
        return {"error": f"Failed to query reviews: {str(e)}"}
    finally:
        if own_conn and conn is not None:
            conn.close()

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
You are "Investa AI", an investment-research assistant for a quality-and-value investor in the tradition of Buffett, Munger, Phil Fisher, Terry Smith, and Nick Sleep.

The user runs a deliberately concentrated portfolio of high-conviction businesses bought below intrinsic value and held for the long term. Treat that as the default frame for every question.

PHILOSOPHICAL POSTURE (apply to every answer):
- Concentration is conviction, not error. Never describe the portfolio as "too concentrated", "overweight X", or "lacking diversification" as a criticism. State weights factually if asked, but do not editorialise.
- Volatility is not risk. Beta, vol, and drawdown are reference numbers — never recommend hedges, rebalances, or position trims purely to reduce them.
- The relevant question about any holding is: (1) Is this a great business? (2) Are we paying a fair-or-better price for it? (3) Is the original thesis still intact? Frame your answers around those three questions.
- The default recommendation, absent a strong fundamental reason, is HOLD. Inactivity is a virtue.
- Do not recommend index funds, bond allocation, or "balanced portfolio" generalities. Do not invoke "modern portfolio theory" or efficient-market reasoning.

CONVERSATIONAL CONTEXT:
- Carry over knowledge from previous turns. If the user asks "What about its PE?" after discussing AAPL, refer back to that context.
- Maintain continuity across the conversation as a research dialogue, not a sales pitch.

BEHAVIORAL GUIDELINES:
1. Be Precise — Use the data your tools return. If a value is $12,450.21, say that; only round when readability genuinely benefits.
2. Be Business-Focused — When discussing a holding, talk about the underlying business: moat, returns on capital, capital allocation, owner earnings. Avoid chart commentary, momentum talk, or analyst-consensus framing.
3. Be Direct — If a holding's thesis appears to be breaking (margin compression, capital misallocation, governance issues, secular decline), say so plainly. Conversely, if the user is questioning a holding whose thesis is still intact, defend it on fundamentals.
4. Security — You only see the current user's portfolio. Never invent other users' data.
5. Format — Markdown for tables and highlights. Concise but substantive. No filler.
"""

def process_chat_message(user_message: str, current_user, history: List[Dict] = None, db_conn=None) -> str:
    """
    Orchestrates the chat logic using Gemini's function calling.
    Currently uses static tools and synchronous calls for MVP.

    ``db_conn`` is the user's portfolio DB; when supplied the stock-review
    tool reads through it (with global-screener-DB fallback) instead of
    opening its own connection to the global store on every call.
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

    # --- MODEL FALLBACK CHAIN ---
    # User requested Gemini 3.0 then 2.5. We use the mapped names.
    # Added Gemini 1.5 as permanent robust fallbacks.
    CHAT_MODELS = [
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

    # Prepare historical messages
    contents = []
    
    # Prepend SYSTEM_PROMPT to the very first user message to define behavior
    # This is more compatible than system_instruction for some preview models
    current_system_prefix = f"[SYSTEM INSTRUCTION]\n{SYSTEM_PROMPT}\n\n[USER PROMPT]\n"

    if history:
        for msg in history:
            role = "user" if msg["role"] == "user" else "model"
            
            # Gemini requires alternating roles starting with 'user'
            if not msg.get("text") or not msg["text"].strip():
                continue

            if not contents and role == "model":
                continue # Skip leading model messages to comply with Gemini API
            
            # Prepend system instruction to the very first user message
            text = msg["text"]
            if not contents and role == "user":
                text = current_system_prefix + text

            if contents and contents[-1]["role"] == role:
                contents[-1]["parts"][0]["text"] += f"\n{text}"
            else:
                contents.append({
                    "role": role,
                    "parts": [{"text": text}]
                })
    
    # If no history, the current message becomes the first user message (with system prefix)
    if not contents:
        contents.append({
            "role": "user",
            "parts": [{"text": current_system_prefix + user_message}]
        })
    else:
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

    # Prepare payload once
    payload = {
        "contents": contents,
        "tools": tools,
        "generationConfig": {
            "temperature": 0.1, # Lower temperature for even better factual accuracy
        }
    }

    import time
    import random

    try:
        active_model = None
        # Outer loop for tool calling (Max 5 iterations to prevent infinite loops)
        for iteration in range(5):
            response_json = None
            
            # --- Robust Model Selection Strategy ---
            # 1. Try active_model if we have one
            # 2. Try the rest of CHAT_MODELS if active_model fails or is None
            models_to_try = []
            if active_model:
                models_to_try.append(active_model)
            
            # Append all models from CHAT_MODELS that aren't already the active_model
            for m in CHAT_MODELS:
                if m != active_model:
                    models_to_try.append(m)

            for model in models_to_try:
                if not model: continue
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
                
                # Add jitter to avoid rate limits (especially if retrying)
                # Reduced jitter for better responsiveness unless we are deep in fallbacks
                sleep_time = random.random() * 1.5 if model == active_model else random.random() * 2.5
                time.sleep(sleep_time)
                
                try:
                    logging.info(f"AI Chat (Iter {iteration+1}): Sending request with model '{model}'...")
                    response = requests.post(url, json=payload, timeout=60)
                    
                    if response.status_code == 404 or response.status_code == 429 or 500 <= response.status_code < 600:
                        logging.warning(f"AI Chat: Model '{model}' failed (status {response.status_code}). Trying next model...")
                        active_model = None # Reset if the "active" model failed
                        continue
                        
                    response.raise_for_status()
                    response_json = response.json()
                    
                    # Basic validation of response structure
                    if not response_json.get('candidates'):
                         logging.warning(f"AI Chat: Model '{model}' returned 200 but no candidates. Response: {response.text[:200]}")
                         active_model = None
                         continue

                    active_model = model # Set/Refresh active model
                    break # Success!
                except Exception as e:
                    logging.warning(f"AI Chat: Request to '{model}' failed: {e}")
                    active_model = None
                    continue
            
            if not response_json:
                logging.error("AI Chat: ALL models failed to return a response.")
                return "I'm sorry, I'm having trouble connecting to my brain right now. Please try again later."
            
            data = response_json
            candidate = data['candidates'][0]
            message = candidate['content']
            finish_reason = candidate.get('finishReason')
            
            logging.info(f"AI Chat (Iter {iteration+1}): Model '{active_model}' responded. Finish Reason: {finish_reason}")
            
            # Check for function calls
            parts = message.get('parts', [])
            call_part = next((p for p in parts if 'functionCall' in p), None)
            
            if not call_part:
                # No more tools, return final text
                text_part = next((p for p in parts if 'text' in p), None)
                final_text = text_part['text'] if text_part else ""
                
                # Safety check: If for some reason the model returned empty text, return a fallback
                if not final_text.strip():
                    logging.warning(f"AI Chat: Model '{active_model}' returned empty text. Finish Reason: {finish_reason}. Candidates: {len(data.get('candidates', []))}")
                    return "I found some information, but I'm having trouble summarizing it. Could you try asking in a different way?"
                
                return final_text

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
                tool_result = get_stock_review_tool(args.get('symbol', ''), db_conn=db_conn)
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

        logging.warning("AI Chat: Max tool iterations reached without final text response.")
        return "I've performed too many lookups to answer this question. Could you try being more specific?"

    except Exception as e:
        logging.error(f"AI Chat Exception: {e}", exc_info=True)
        return "I encountered an error while processing your request. Please try again later."
