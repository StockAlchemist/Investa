"""Settings routes: user settings, manual overrides."""

# ruff: noqa: E402
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

import config
from config_manager import ConfigManager
from server.auth import User
from server.dependencies import clear_settings_cache, get_config_manager, get_current_user
from server.portfolio_service import (
    clear_portfolio_caches,
    reload_data_and_clear_cache,
    trigger_background_precalculation,
)

# Project root (…/Investa) — manual_overrides.json is mirrored here if present
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

router = APIRouter()


def _get_symbol_currency_map_sql(db_path: str) -> Dict[str, str]:
    """Helper to get Symbol -> Local Currency mapping directly from DB without loading all transactions."""
    if not os.path.exists(db_path):
        return {}
    try:
        # Use a separate connection to avoid interfering with pool if any
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        # Query unique Symbol/Currency pairs
        cursor.execute("SELECT DISTINCT Symbol, \"Local Currency\" FROM transactions WHERE Symbol IS NOT NULL AND Symbol != ''")
        rows = cursor.fetchall()
        conn.close()
        return {row[0]: row[1] for row in rows}
    except Exception as e:
        logging.warning(f"Error fetching symbol currency map via SQL: {e}")
        return {}


@router.get("/settings")
def get_settings(
    current_user: User = Depends(get_current_user),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """
    Returns the current application configuration settings.
    Fast version that avoids loading the full transaction dataframe.
    """
    try:
        # Load fresh from disk
        config_manager.load_all()
        
        manual_overrides = config_manager.manual_overrides.get("manual_price_overrides", {})
        user_symbol_map = config_manager.manual_overrides.get("user_symbol_map", {})
        user_excluded_symbols = config_manager.manual_overrides.get("user_excluded_symbols", [])
        account_currency_map = config_manager.gui_config.get("account_currency_map", {"SET": "THB"})

        # Build a mapping of Symbol -> Local Currency from transactions using LIGHTWEIGHT SQL
        user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
        db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
        symbol_to_currency = _get_symbol_currency_map_sql(db_path)

        # Enrich manual_overrides with currency info
        enriched_overrides = {}
        for symbol, override_data in manual_overrides.items():
            symbol_upper = symbol.upper()
            currency = symbol_to_currency.get(symbol_upper)
            
            # If not found in transactions, try to derive from symbol suffix or use default
            if not currency:
                if symbol_upper.endswith(".BK") or ":BKK" in symbol_upper:
                    currency = "THB"
                else:
                    currency = "USD" # Default to USD
            
            # Create a copy and add currency
            enriched_data = override_data.copy() if isinstance(override_data, dict) else {"price": override_data}
            enriched_data["currency"] = currency
            enriched_overrides[symbol_upper] = enriched_data

        return {
            "manual_overrides": enriched_overrides,
            "user_symbol_map": user_symbol_map,
            "user_excluded_symbols": list(user_excluded_symbols),
            "account_currency_map": account_currency_map,
            "account_groups": config_manager.gui_config.get("account_groups", {}),
            "account_group_order": config_manager.gui_config.get("account_group_order", []),
            "available_currencies": config_manager.gui_config.get("available_currencies", ['USD', 'THB', 'EUR', 'GBP', 'JPY', 'CNY']),
            "account_interest_rates": config_manager.manual_overrides.get("account_interest_rates", {}),
            "interest_free_thresholds": config_manager.manual_overrides.get("interest_free_thresholds", {}),
            "valuation_overrides": config_manager.manual_overrides.get("valuation_overrides", {}),
            "visible_items": config_manager.gui_config.get("visible_items", []),
            "benchmarks": config_manager.gui_config.get("benchmarks", ['S&P 500', 'Dow Jones', 'NASDAQ']),
            "show_closed": config_manager.gui_config.get("show_closed", False),
            "display_currency": config_manager.gui_config.get("display_currency", "USD"),
            "selected_accounts": config_manager.gui_config.get("selected_accounts", []),
            "active_tab": config_manager.gui_config.get("active_tab", "performance"),
            "account_cash_mode_map": config_manager.gui_config.get("account_cash_mode_map", {}),
            "account_closure_dates": config_manager.gui_config.get("account_closure_dates", {}),
            "ibkr_token": config_manager.manual_overrides.get("ibkr_token") or getattr(config, "IBKR_TOKEN", None),
            "ibkr_query_id": config_manager.manual_overrides.get("ibkr_query_id") or getattr(config, "IBKR_QUERY_ID", None),
            "target_allocation": config_manager.gui_config.get("target_allocation", {}),
        }
    except Exception as e:
        logging.error(f"Error getting settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class SettingsUpdate(BaseModel):
    manual_price_overrides: Optional[Dict[str, Any]] = None
    user_symbol_map: Optional[Dict[str, str]] = None
    user_excluded_symbols: Optional[List[str]] = None
    account_groups: Optional[Dict[str, List[str]]] = None
    account_group_order: Optional[List[str]] = None
    account_currency_map: Optional[Dict[str, str]] = None
    account_cash_mode_map: Optional[Dict[str, str]] = None
    # Per-account closure dates (ISO YYYY-MM-DD). When a slice consists entirely
    # of closed accounts (date <= today), rate-of-return metrics are gated to
    # avoid the residual-dividend TWR inflation bug.
    account_closure_dates: Optional[Dict[str, str]] = None
    available_currencies: Optional[List[str]] = None
    account_interest_rates: Optional[Dict[str, float]] = None
    interest_free_thresholds: Optional[Dict[str, float]] = None
    valuation_overrides: Optional[Dict[str, Any]] = None
    visible_items: Optional[List[str]] = None
    benchmarks: Optional[List[str]] = None
    show_closed: Optional[bool] = None
    display_currency: Optional[str] = None
    selected_accounts: Optional[List[str]] = None
    active_tab: Optional[str] = None
    ibkr_token: Optional[str] = None
    ibkr_query_id: Optional[str] = None
    # Per-bucket target allocation. Outer key is bucket type (e.g. "quoteType",
    # "sector"); inner maps bucket name → target % of portfolio.
    target_allocation: Optional[Dict[str, Dict[str, float]]] = None


@router.post("/settings/update")
def update_settings(
    settings: SettingsUpdate,
    config_manager = Depends(get_config_manager),
    current_user: User = Depends(get_current_user)
):
    """
    Updates the application settings (manual overrides, symbol map, exclude list).
    """
    try:
        # RELOAD DATA FROM DISK to ensure we have the latest state before merging updates
        config_manager.load_manual_overrides()
        
        current_overrides = config_manager.manual_overrides
        
        # Update Manual Price Overrides
        if settings.manual_price_overrides is not None:
             current_overrides["manual_price_overrides"] = settings.manual_price_overrides

        if settings.user_symbol_map is not None:
            current_overrides["user_symbol_map"] = settings.user_symbol_map
            
        if settings.user_excluded_symbols is not None:
            current_overrides["user_excluded_symbols"] = sorted(list(set(settings.user_excluded_symbols)))
            
        if settings.account_interest_rates is not None:
            current_overrides["account_interest_rates"] = settings.account_interest_rates
            
        if settings.interest_free_thresholds is not None:
            current_overrides["interest_free_thresholds"] = settings.interest_free_thresholds

        if settings.valuation_overrides is not None:
            current_overrides["valuation_overrides"] = settings.valuation_overrides

        if settings.ibkr_token is not None:
            current_overrides["ibkr_token"] = settings.ibkr_token
            
        if settings.ibkr_query_id is not None:
            current_overrides["ibkr_query_id"] = settings.ibkr_query_id

        # Update GUI Config (Dashboard persistence)
        gui_config_changed = False
        if settings.visible_items is not None:
            config_manager.gui_config["visible_items"] = settings.visible_items
            gui_config_changed = True
        
        if settings.benchmarks is not None:
            config_manager.gui_config["benchmarks"] = settings.benchmarks
            gui_config_changed = True

        if settings.target_allocation is not None:
            config_manager.gui_config["target_allocation"] = settings.target_allocation
            gui_config_changed = True
            
        if settings.show_closed is not None:
            config_manager.gui_config["show_closed"] = settings.show_closed
            gui_config_changed = True

        if settings.account_groups is not None:
            config_manager.gui_config["account_groups"] = settings.account_groups
            gui_config_changed = True

        if settings.account_group_order is not None:
            config_manager.gui_config["account_group_order"] = settings.account_group_order
            gui_config_changed = True
            
        if settings.account_currency_map is not None:
             config_manager.gui_config["account_currency_map"] = settings.account_currency_map
             gui_config_changed = True
             
        if settings.account_cash_mode_map is not None:
             config_manager.gui_config["account_cash_mode_map"] = settings.account_cash_mode_map
             gui_config_changed = True

        if settings.account_closure_dates is not None:
             config_manager.gui_config["account_closure_dates"] = settings.account_closure_dates
             gui_config_changed = True

        if settings.available_currencies is not None:
             config_manager.gui_config["available_currencies"] = settings.available_currencies
             gui_config_changed = True

        if settings.display_currency is not None:
             config_manager.gui_config["display_currency"] = settings.display_currency
             gui_config_changed = True

        if settings.selected_accounts is not None:
             config_manager.gui_config["selected_accounts"] = settings.selected_accounts
             gui_config_changed = True

        if settings.active_tab is not None:
             config_manager.gui_config["active_tab"] = settings.active_tab
             gui_config_changed = True

        if gui_config_changed:
            config_manager.save_gui_config()

        # Save to AppData
        if config_manager.save_manual_overrides(current_overrides):
            
            # --- Added: Mirror save to Project Root if file exists ---
            # This ensures the user's "source of truth" in their workspace stays in sync
            project_overrides_file = os.path.join(project_root, "manual_overrides.json")
            if os.path.exists(project_overrides_file):
                try:
                    with open(project_overrides_file, "w", encoding="utf-8") as f:
                        json.dump(current_overrides, f, indent=4, ensure_ascii=False)
                    logging.info(f"Mirrored settings update to {project_overrides_file}")
                except Exception as e:
                    logging.warning(f"Failed to mirror settings to project file: {e}")
            # ---------------------------------------------------------

            # OPTIMIZATION: Only reload settings and clear summary/history caches.
            # Avoid a full 'reload_data()' which wipes the transaction cache and forces a heavy DB reload.
            clear_settings_cache(current_user.username)
            clear_portfolio_caches()
            trigger_background_precalculation(current_user)
            return {"status": "success", "message": "Settings updated and data reloaded"}
        else:
             raise HTTPException(status_code=500, detail="Failed to save settings to file")

    except Exception as e:
        logging.error(f"Error updating settings: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


class ManualOverrideRequest(BaseModel):
    symbol: str
    price: Optional[float] = None


@router.post("/settings/manual_overrides")
def update_manual_override(
    override: ManualOverrideRequest,
    current_user: User = Depends(get_current_user),
    config_manager: ConfigManager = Depends(get_config_manager)
):
    """
    Updates or removes a manual price override for a symbol.
    If price is None, removes the override.
    """
    try:
        # Load latest overrides
        config_manager.load_manual_overrides()
        
        symbol_upper = override.symbol.strip().upper()
        current_overrides = config_manager.manual_overrides.get("manual_price_overrides", {})
        
        if override.price is not None:
            # Add/Update
            # We preserve existing extra fields if any (like date), or create new dict
            existing_entry = current_overrides.get(symbol_upper, {})
            # If strictly setting price, update it.
            # Use 'manual' as source hint
            existing_entry["price"] = override.price
            existing_entry["source"] = "User Manual Override"
            existing_entry["updated_at"] = datetime.now().isoformat()
            
            # Ensure price is valid positive number or 0
            if override.price < 0:
                 raise HTTPException(status_code=400, detail="Price must be non-negative.")

            current_overrides[symbol_upper] = existing_entry
        else:
            # Remove override if it exists
            if symbol_upper in current_overrides:
                del current_overrides[symbol_upper]
        
        config_manager.manual_overrides["manual_price_overrides"] = current_overrides
        success = config_manager.save_manual_overrides()
        
        if not success:
             raise HTTPException(status_code=500, detail="Failed to save manual overrides file.")

        # --- Legacy Clean-up: Also remove from gui_config if present to avoid conflicts ---
        # Because dependencies.py merges them, we must ensure it's gone from legacy config too.
        gui_config_changed = False
        gc = config_manager.gui_config
        
        # Check both possible legacy keys
        for key in ["manual_price_overrides", "manual_overrides_dict"]:
            if key in gc and isinstance(gc[key], dict):
                if symbol_upper in gc[key]:
                    del gc[key][symbol_upper]
                    gui_config_changed = True
        
        # Also if we are Adding, we might want to Add to legacy? 
        # No, let's strictly migrate to JSON. But we must ensure Legacy doesn't overwrite.
        # If we Add to JSON, dependencies.py (with my previous fix) will Update Legacy with JSON.
        # So JSON wins for Add/Update.
        # But for Delete, JSON just lacks the key, so Legacy wins.
        # So we MUST delete from Legacy.
        
        if gui_config_changed:
             config_manager.save_gui_config()
             logging.info(f"Removed {symbol_upper} from legacy gui_config.")
        # -------------------------------------------------------------------------------

        # Force reload of data so next request uses new override
        reload_data_and_clear_cache(current_user)
        
        return {"status": "success", "message": f"Override for {symbol_upper} updated."}

    except Exception as e:
        logging.error(f"Error updating manual override: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
