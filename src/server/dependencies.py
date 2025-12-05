import os
import sys
import pandas as pd
import logging
import json
from typing import Optional, Tuple, Set, Dict, Any

# Ensure src is in path (redundant if imported from main, but good for standalone testing)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from db_utils import get_database_path, load_all_transactions_from_db, DB_FILENAME
from data_loader import load_and_clean_transactions
import config

logging.info("DEPENDENCIES MODULE LOADED - VERSION 2")

# Global cache for transactions to avoid reloading on every request
# In a real production app, this might be handled differently, but for a personal app this is fine.
_TRANSACTIONS_CACHE: Optional[pd.DataFrame] = None
_IGNORED_INDICES: Set[int] = set()
_IGNORED_REASONS: Dict[int, str] = {}
_MANUAL_OVERRIDES: Dict[str, Any] = {}
_USER_SYMBOL_MAP: Dict[str, str] = {}
_USER_EXCLUDED_SYMBOLS: Set[str] = set()
_ACCOUNT_CURRENCY_MAP: Dict[str, str] = {}
_DB_MTIME: float = 0.0

def get_transaction_data() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str], Set[str], Dict[str, str], str]:
    """
    Loads transaction data from the database.
    Checks for file modification to auto-reload.
    """
    global _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATH, _DB_MTIME
    
    # Determine DB path first to check mtime
    db_path = os.path.join(project_root, "my_transactions.db")
    if not os.path.exists(db_path):
        # Fallback
        db_path = get_database_path(DB_FILENAME)
    
    # Check if file exists and get mtime
    current_mtime = 0.0
    if os.path.exists(db_path):
        current_mtime = os.path.getmtime(db_path)
    
    # Reload if cache is empty OR file has changed
    if _TRANSACTIONS_CACHE is None or current_mtime != _DB_MTIME:
        if _TRANSACTIONS_CACHE is not None:
            logging.info(f"Database file changed (mtime {_DB_MTIME} -> {current_mtime}). Reloading...")
        
        logging.info(f"Loading transactions from: {db_path}")
        
        # Try to load gui_config.json for account_currency_map and other settings
        account_currency_map = {"SET": "THB"} # Default
        default_currency = config.DEFAULT_CURRENCY
        
        # Initialize with defaults from config.py
        user_symbol_map = config.SYMBOL_MAP_TO_YFINANCE.copy()
        user_excluded_symbols = set(config.YFINANCE_EXCLUDED_SYMBOLS.copy())

        config_path = os.path.join(project_root, "gui_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    gui_config = json.load(f)
                    if "account_currency_map" in gui_config:
                        account_currency_map.update(gui_config["account_currency_map"])
                    if "user_symbol_map" in gui_config:
                        user_symbol_map.update(gui_config["user_symbol_map"])
                    if "user_excluded_symbols" in gui_config:
                        user_excluded_symbols.update(set(gui_config["user_excluded_symbols"]))
            except Exception as e:
                logging.warning(f"Could not load gui_config.json: {e}")

        # Load manual_overrides.json if it exists
        manual_overrides = {} # The specific dict for portfolio_logic (Symbol -> Data)
        full_overrides_json = {} # The full JSON for config extraction
        
        overrides_path = os.path.join(project_root, "manual_overrides.json")
        if os.path.exists(overrides_path):
            try:
                with open(overrides_path, "r") as f:
                    full_overrides_json = json.load(f)
                    
                    # portfolio_logic expects the dict to be Symbol -> OverrideData
                    # So we extract 'manual_price_overrides' if it exists, or use the whole dict if it's the old format
                    if "manual_price_overrides" in full_overrides_json:
                        manual_overrides = full_overrides_json["manual_price_overrides"]
                    else:
                        manual_overrides = full_overrides_json
                        
                logging.info(f"Loaded manual overrides from {overrides_path}")
            except Exception as e:
                logging.warning(f"Could not load manual_overrides.json: {e}")

        # Merge user_excluded_symbols from FULL JSON if present
        if "user_excluded_symbols" in full_overrides_json:
            loaded_excluded = full_overrides_json.get("user_excluded_symbols", [])
            if isinstance(loaded_excluded, list):
                clean_excluded = {s.upper().strip() for s in loaded_excluded if isinstance(s, str)}
                user_excluded_symbols.update(clean_excluded)
                logging.info(f"Loaded {len(clean_excluded)} excluded symbols from manual_overrides.json")

        # Merge user_symbol_map from FULL JSON if present
        if "user_symbol_map" in full_overrides_json:
            loaded_map = full_overrides_json.get("user_symbol_map", {})
            if isinstance(loaded_map, dict):
                user_symbol_map.update(loaded_map)
                logging.info(f"Loaded {len(loaded_map)} symbol mappings from manual_overrides.json")

        # We use the existing data_loader logic which expects a file path (CSV or DB)
        # load_and_clean_transactions handles DB loading if the path ends in .db
        try:
            # Check if it's a DB path
            is_db = db_path.lower().endswith((".db", ".sqlite", ".sqlite3"))
            
            df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
                source_path=db_path,
                account_currency_map=account_currency_map,
                default_currency=default_currency,
                is_db_source=is_db
            )
            _TRANSACTIONS_CACHE = df
            _IGNORED_INDICES = ignored_indices
            _IGNORED_REASONS = ignored_reasons
            _MANUAL_OVERRIDES = manual_overrides
            _USER_SYMBOL_MAP = user_symbol_map
            _USER_EXCLUDED_SYMBOLS = user_excluded_symbols
            _ACCOUNT_CURRENCY_MAP = account_currency_map
            _DB_PATH = db_path
            _DB_MTIME = current_mtime
            
            logging.info(f"Loaded {len(df)} transactions.")
        except Exception as e:
            logging.error(f"Error loading transactions: {e}", exc_info=True)
            # Return empty/default values on error, but don't crash
            return pd.DataFrame(), {}, {}, set(), {}, ""

    # Return cached data
    return _TRANSACTIONS_CACHE, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATH

def reload_data():
    """Forces a reload of the transaction data."""
    global _TRANSACTIONS_CACHE, _DB_MTIME, _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP
    _TRANSACTIONS_CACHE = None
    _DB_MTIME = 0.0
    _IGNORED_INDICES = set()
    _IGNORED_REASONS = {}
    _MANUAL_OVERRIDES = {}
    _USER_SYMBOL_MAP = {}
    _USER_EXCLUDED_SYMBOLS = set()
    _ACCOUNT_CURRENCY_MAP = {}
    logging.info("Transaction data cache cleared.")
    get_transaction_data()
