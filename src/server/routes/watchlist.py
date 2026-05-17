# -*- coding: utf-8 -*-
"""
Watchlist routes — extracted from server/api.py as the first slice of the
sub-router split (item #4 part 2).

Exposes the same paths as before so the OpenAPI surface is unchanged:
    GET  /watchlists                — list user's watchlists
    POST /watchlists                — create a watchlist
    PUT  /watchlists/{id}           — rename
    DEL  /watchlists/{id}           — delete
    GET  /watchlist?id=             — read items in a watchlist (enriched)
    POST /watchlist                 — add a symbol to a watchlist
    DEL  /watchlist/{symbol}?id=    — remove a symbol from a watchlist

Pattern to follow for further extractions (transactions, holdings,
fundamentals, etc.): one APIRouter per cohesive concern, included into
the main router in api.py via `router.include_router(...)`.
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

import config
from config_manager import ConfigManager
from db_utils import (
    add_to_watchlist,
    create_watchlist,
    delete_watchlist,
    get_all_watchlists,
    get_cached_screener_results,
    get_watchlist,
    remove_from_watchlist,
    rename_watchlist,
)
from server.dependencies import (
    get_config_manager,
    get_current_user,
    get_user_db_connection,
)

# get_mdp / clean_nans / User live in api.py; importing them here would
# create a cycle. They're re-injected via the dependency layer:
#   * get_mdp: imported lazily inside the one route that needs it
#   * clean_nans: ditto
#   * User: passed via Depends(get_current_user) — we only need the type
#       for annotation, so we accept it as Any to avoid the cycle.
from typing import Any as _User  # type alias to keep route signatures readable


router = APIRouter(tags=["watchlist"])


class WatchlistAdd(BaseModel):
    symbol: str
    note: Optional[str] = ""
    watchlist_id: Optional[int] = 1


class WatchlistCreate(BaseModel):
    name: str


class WatchlistRename(BaseModel):
    name: str


@router.get("/watchlists")
async def get_watchlists_endpoint(
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """List all watchlists for the current user."""
    return get_all_watchlists(conn)


@router.post("/watchlists")
async def create_watchlist_endpoint(
    item: WatchlistCreate,
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Create a new watchlist."""
    new_id = create_watchlist(conn, item.name, user_id=current_user.id)
    if not new_id:
        raise HTTPException(status_code=500, detail="Failed to create watchlist")
    return {"id": new_id, "name": item.name}


@router.put("/watchlists/{watchlist_id}")
async def rename_watchlist_endpoint(
    watchlist_id: int,
    item: WatchlistRename,
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Rename a watchlist."""
    success = rename_watchlist(conn, watchlist_id, item.name)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or failed to rename")
    return {"status": "success"}


@router.delete("/watchlists/{watchlist_id}")
async def delete_watchlist_endpoint(
    watchlist_id: int,
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Delete a watchlist."""
    success = delete_watchlist(conn, watchlist_id)
    if not success:
        raise HTTPException(status_code=404, detail="Watchlist not found or failed to delete")
    return {"status": "success"}


@router.get("/watchlist")
async def get_watchlist_endpoint(
    watchlist_id: int = Query(1, alias="id"),
    currency: str = "USD",
    config_manager: ConfigManager = Depends(get_config_manager),
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Fetch a watchlist enriched with current market prices, AI ratings, and fundamentals."""
    # Lazy imports to avoid circular dependency with api.py
    from server.api import get_mdp, clean_nans

    config_manager.load_manual_overrides()
    overrides = config_manager.manual_overrides
    user_symbol_map = overrides.get("user_symbol_map", {})
    user_excluded_symbols = set(overrides.get("user_excluded_symbols", []))

    # Verify ownership
    allowed = [w["id"] for w in get_all_watchlists(conn)]
    if watchlist_id not in allowed:
        if not allowed and watchlist_id == 1:
            pass  # default first-use: empty watchlist 1
        else:
            return []

    db_items = get_watchlist(conn, watchlist_id=watchlist_id)
    if not db_items:
        return []

    mdp = get_mdp()
    symbols = [item["Symbol"] for item in db_items]

    # Fetch current quotes
    try:
        quotes, _fx_rates, _fx_prev, _q_errors, _q_warnings = mdp.get_current_quotes(
            internal_stock_symbols=symbols,
            required_currencies=set(),
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
        )
    except Exception as e_quotes:
        logging.error(f"Failed to fetch quotes for watchlist: {e_quotes}")
        quotes = {}

    # AI ratings
    try:
        ai_results = get_cached_screener_results(conn, symbols)
    except Exception as e_ai:
        logging.error(f"Failed to fetch AI results for watchlist: {e_ai}")
        ai_results = {}

    # Fundamentals (Market Cap, PE, Yield)
    try:
        fundamentals = mdp.get_fundamentals_batch(
            symbols,
            user_symbol_map=user_symbol_map,
            user_excluded_symbols=user_excluded_symbols,
        )
    except Exception as e_fund:
        logging.error(f"Failed to fetch fundamentals for watchlist: {e_fund}")
        fundamentals = {}

    enriched_items = []
    ai_cache_dir = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "ai_analysis_cache")

    for item in db_items:
        symbol = item["Symbol"]
        u_symbol = symbol.upper() if symbol else ""
        quote = quotes.get(symbol, {})
        fund = fundamentals.get(symbol, {})
        ai_res = ai_results.get(u_symbol, {}).copy()

        # Lazy sync from disk cache when DB row is missing sentiment/catalysts
        if not ai_res.get("ai_sentiment") or not ai_res.get("ai_catalysts"):
            ai_cache_path = os.path.join(ai_cache_dir, f"{u_symbol}_analysis.json")
            if os.path.exists(ai_cache_path):
                try:
                    with open(ai_cache_path, "r") as f:
                        ai_disk_data = json.load(f)
                        analysis_obj = ai_disk_data.get("analysis") if "analysis" in ai_disk_data else ai_disk_data
                        if not ai_res.get("ai_sentiment"):
                            ai_res["ai_sentiment"] = analysis_obj.get("sentiment")
                        if not ai_res.get("ai_catalysts"):
                            ai_res["ai_catalysts"] = analysis_obj.get("catalysts")
                        if not ai_res.get("has_ai_review"):
                            ai_res["has_ai_review"] = True
                except Exception:
                    pass

        sparkline = quote.get("sparkline_7d", [])

        enriched_items.append({
            **item,
            "Price": quote.get("price"),
            "Day Change": quote.get("change"),
            "Day Change %": quote.get("changesPercentage"),
            "Name": quote.get("name"),
            "Currency": quote.get("currency"),
            "Sparkline": sparkline,
            "Market Cap": fund.get("marketCap"),
            "PE Ratio": fund.get("trailingPE") or fund.get("forwardPE"),
            "Dividend Yield": fund.get("dividendYield"),
            "ai_score": ai_res.get("ai_score"),
            "intrinsic_value": ai_res.get("intrinsic_value"),
            "margin_of_safety": ai_res.get("margin_of_safety"),
            "has_ai_review": ai_res.get("has_ai_review"),
            "ai_sentiment": ai_res.get("ai_sentiment"),
            "ai_catalysts": ai_res.get("ai_catalysts"),
        })

    return clean_nans(enriched_items)


@router.post("/watchlist")
async def add_to_watchlist_api(
    item: WatchlistAdd,
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Add a symbol to a watchlist."""
    symbol_upper = item.symbol.strip().upper()
    if not symbol_upper:
        raise HTTPException(status_code=400, detail="Symbol is required")

    wl_id = item.watchlist_id or 1
    allowed = [w["id"] for w in get_all_watchlists(conn)]
    if wl_id not in allowed:
        raise HTTPException(
            status_code=403,
            detail="Not authorized to modify this watchlist. Please create a watchlist first.",
        )

    success = add_to_watchlist(conn, symbol_upper, item.note, watchlist_id=wl_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to add to watchlist")
    return {"status": "success"}


@router.delete("/watchlist/{symbol}")
async def remove_from_watchlist_api(
    symbol: str,
    watchlist_id: int = Query(1, alias="id"),
    current_user: _User = Depends(get_current_user),
    conn: sqlite3.Connection = Depends(get_user_db_connection),
):
    """Remove a symbol from a watchlist."""
    allowed = [w["id"] for w in get_all_watchlists(conn)]
    if watchlist_id not in allowed:
        raise HTTPException(status_code=403, detail="Not authorized to modify this watchlist")

    success = remove_from_watchlist(conn, symbol.strip().upper(), watchlist_id=watchlist_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to remove from watchlist")
    return {"success": True}
