"""Screener and AI chat routes."""

# ruff: noqa: E402
import logging
import sqlite3
from typing import Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from server.ai_analyzer import generate_stock_review
from server.auth import User
from server.dependencies import get_current_user, get_transaction_data, get_user_db_connection
from server.portfolio_service import _calculate_portfolio_summary_internal
from server.route_utils import clean_nans, get_mdp
from server.screener_service import run_narrative_search, screen_stocks

router = APIRouter()


class ScreenerRequest(BaseModel):
    universe_type: str = Field(..., description="watchlist, manual, sp500, russell2000, sp400, holdings, or all")
    universe_id: Optional[str] = None
    manual_symbols: Optional[List[str]] = None
    fast_mode: Optional[bool] = False


@router.post("/screener/run")
async def run_screener(
    request: ScreenerRequest,
    current_user: User = Depends(get_current_user),
    data: tuple = Depends(get_transaction_data),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection)
):
    """
    Runs the stock screener based on the selected universe.
    """
    try:
        universe_type = request.universe_type
        universe_id = request.universe_id
        manual_symbols = request.manual_symbols
        
        # Handle "Holdings" as a dynamic universe if requested
        if universe_type == "holdings":
            summary_data = await _calculate_portfolio_summary_internal(
                data=data, 
                current_user=current_user, 
                show_closed_positions=False
            )
            summary_df = summary_data.get("summary_df")
            if summary_df is not None and not summary_df.empty:
                # Filter out "Total" row and get unique symbols
                manual_symbols = summary_df[~summary_df.get("is_total", False) & (summary_df["Symbol"] != "Total")]["Symbol"].tolist()
                # We treat it as a manual list once resolved
                universe_type = "manual" 

        results = await run_in_threadpool(screen_stocks, universe_type, universe_id, manual_symbols, db_conn=db_conn, fast_mode=request.fast_mode)
        return clean_nans(results)
    except Exception as e:
        logging.error(f"Screener error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screener/narrative")
def narrative_screener(
    item: Dict[str, str],
    current_user: User = Depends(get_current_user)
):
    """Executes a narrative search using natural language via Gemini."""
    prompt = item.get("prompt")
    if not prompt:
        raise HTTPException(status_code=400, detail="Missing prompt")
        
    try:
        results = run_narrative_search(prompt)
        return results
    except Exception as e:
        logging.error(f"Error in narrative screener API: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/screener/review/{symbol}")
def trigger_ai_review(
    symbol: str,
    force: bool = Query(False),
    current_user: User = Depends(get_current_user),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """
    Triggers (or retrieves cached) AI review for a specific stock.
    """
    try:
        # We need to fetch data first to pass to AI
        mdp = get_mdp()
        # Ensure data exists
        quotes, _, _, _, _ = mdp.get_current_quotes([symbol], {"USD"}, {}, set())
        details = mdp.get_ticker_details_batch({symbol})
        
        fund_data = details.get(symbol, {})
        # Merge price
        if symbol in quotes:
            fund_data["currentPrice"] = quotes[symbol].get("price")
            
        if not fund_data:
             raise HTTPException(status_code=404, detail=f"Data not found for {symbol}")
        
        # Ratios data - we can pass empty or minimal if we don't have full ratios calculated.
        # AI Analyzer expects 'ratios_data' dict with keys like 'Return on Equity (ROE) (%)'
        # We can extract some from fund_data if available (trailingPE etc are in fund_data)
        
        # Map info keys to ratio keys - simplified for Screen context
        ratios_data = {
            "Return on Equity (ROE) (%)": fund_data.get("returnOnEquity", 0) * 100 if fund_data.get("returnOnEquity") else None,
            "Gross Profit Margin (%)": fund_data.get("grossMargins", 0) * 100 if fund_data.get("grossMargins") else None,
            "Net Profit Margin (%)": fund_data.get("profitMargins", 0) * 100 if fund_data.get("profitMargins") else None,
            "Debt-to-Equity Ratio": fund_data.get("debtToEquity", 0) / 100 if fund_data.get("debtToEquity") else None,
            "Current Ratio": fund_data.get("currentRatio"),
        }
        
        review = generate_stock_review(symbol, fund_data, ratios_data, force_refresh=force)
        return clean_nans(review)
        
    except Exception as e:
        logging.error(f"AI Review error for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ChatMessage(BaseModel):
    role: str # 'user' or 'ai'
    text: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[ChatMessage]] = None


@router.post("/chat/message")
async def chat_message_endpoint(
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db_conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """
    Conversational AI interface for portfolio and market insights.
    """
    try:
        from server.ai_chat_service import process_chat_message

        # Convert history objects to simple dicts
        history_dicts = []
        if request.history:
            history_dicts = [{"role": m.role, "text": m.text} for m in request.history]

        # Run in threadpool as many AI calls are synchronous requests
        response_text = await run_in_threadpool(
            process_chat_message, request.message, current_user, history_dicts, db_conn
        )
        return {"response": response_text}
        
    except Exception as e:
        logging.error(f"AI Chat error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to generate AI response.")
