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
_OVERRIDES_PATH: Optional[str] = None
_OVERRIDES_MTIME: float = 0.0

# Added caches for config files to avoid redundant I/O
_GUI_CONFIG_CACHE: Optional[Dict[str, Any]] = None
_GUI_CONFIG_LOADED_PATH: Optional[str] = None
_GUI_CONFIG_MTIME: float = 0.0
_MANUAL_OVERRIDES_FILE_CACHE: Optional[Dict[str, Any]] = None
_MANUAL_OVERRIDES_MTIME: float = 0.0


def get_transaction_data() -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str], Set[str], Dict[str, str], str, float]:
    """
    Loads transaction data from the database.
    Checks for file modification to auto-reload.
    """
    global _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATH, _DB_MTIME, _OVERRIDES_PATH, _OVERRIDES_MTIME
    
    # --- 1. Load Configuration First ---
    # Cached loading of gui_config.json
    global _GUI_CONFIG_CACHE, _GUI_CONFIG_LOADED_PATH, _GUI_CONFIG_MTIME
    
    # Define system config path using centralized logic from config.py
    app_data_dir = config.get_app_data_dir()
    system_config_path = os.path.join(app_data_dir, "gui_config.json")
    
    config_paths_to_try = [
        system_config_path,
        os.path.join(project_root, "gui_config.json"),
        os.path.join(src_dir, "gui_config.json"),
        os.path.join(src_dir, "Investa", "gui_config.json"),
    ]
    
    current_gui_config_path = None
    for cp in config_paths_to_try:
        if os.path.exists(cp):
            current_gui_config_path = cp
            break
            
    if current_gui_config_path:
        st = os.stat(current_gui_config_path)
        if _GUI_CONFIG_CACHE is None or current_gui_config_path != _GUI_CONFIG_LOADED_PATH or st.st_mtime != _GUI_CONFIG_MTIME:
            try:
                with open(current_gui_config_path, "r") as f:
                    _GUI_CONFIG_CACHE = json.load(f)
                _GUI_CONFIG_LOADED_PATH = current_gui_config_path
                _GUI_CONFIG_MTIME = st.st_mtime
                logging.info(f"Loaded/Reloaded gui_config.json from: {current_gui_config_path}")
            except Exception as e:
                logging.warning(f"Found gui_config.json at {current_gui_config_path} but failed to load: {e}")
                if _GUI_CONFIG_CACHE is None: _GUI_CONFIG_CACHE = {}
    else:
        _GUI_CONFIG_CACHE = {}
        _GUI_CONFIG_LOADED_PATH = None
        _GUI_CONFIG_MTIME = 0.0

    gui_config = _GUI_CONFIG_CACHE
    config_loaded_path = _GUI_CONFIG_LOADED_PATH


    # --- 1b. Defaults from config.py ---
    account_currency_map = {"SET": "THB"}
    default_currency = config.DEFAULT_CURRENCY
    user_symbol_map = config.SYMBOL_MAP_TO_YFINANCE.copy()
    user_excluded_symbols = set(config.YFINANCE_EXCLUDED_SYMBOLS.copy())
    
    # Merge account_currency_map from gui_config (legacy support, or keep it there if desired)
    # The request was specifically about symbol mapping and excluded symbols.
    if "account_currency_map" in gui_config:
         account_currency_map.update(gui_config["account_currency_map"])

    # --- CHANGED: Do NOT load symbol map or exclusions from gui_config ---
    # These are now consolidated in manual_overrides.json.
    # We deliberately skip reading them from gui_config to avoid confusion/duplication.
    
    # ----------------------------------------------------

    # --- 2. Determine DB Path ---
    db_path = None
    
    # Priority 1: Path from gui_config.json
    if "transactions_file" in gui_config:
        cfg_db_path = gui_config["transactions_file"]
        if cfg_db_path and os.path.exists(cfg_db_path):
            db_path = cfg_db_path
            logging.info(f"Using DB path from gui_config: {db_path}")
        else:
             logging.warning(f"DB path in gui_config not found: {cfg_db_path}")

    # Priority 2: my_transactions.db in project root (Legacy/Default)
    if not db_path:
        root_db_path = os.path.join(project_root, "my_transactions.db")
        if os.path.exists(root_db_path):
            db_path = root_db_path
            logging.info(f"Using default root DB: {db_path}")

    # Priority 3: db_utils fallback
    if not db_path:
        db_path = get_database_path(DB_FILENAME)
        logging.info(f"Using db_utils fallback DB: {db_path}")

    # Check modification time for DB (and WAL/SHM if present)
    current_mtime = 0.0
    if os.path.exists(db_path):
        current_mtime = os.path.getmtime(db_path)
        # Check WAL/SHM
        wal_path = db_path + "-wal"
        shm_path = db_path + "-shm"
        if os.path.exists(wal_path):
            wal_mtime = os.path.getmtime(wal_path)
            if wal_mtime > current_mtime: 
                current_mtime = wal_mtime
        if os.path.exists(shm_path):
            shm_mtime = os.path.getmtime(shm_path)
            if shm_mtime > current_mtime:
                current_mtime = shm_mtime
    
    # Check modification time for Overrides
    overrides_changed = False
    if _OVERRIDES_PATH and os.path.exists(_OVERRIDES_PATH):
        current_overrides_mtime = os.path.getmtime(_OVERRIDES_PATH)
        if current_overrides_mtime != _OVERRIDES_MTIME:
            overrides_changed = True
            logging.info(f"Overrides file changed. Reloading. (Old MTime: {_OVERRIDES_MTIME}, New: {current_overrides_mtime})")

    # Reload if cache is empty OR file has changed OR DB path changed
    # (Note: _DB_PATH tracking handles path changes)
    if _TRANSACTIONS_CACHE is None or current_mtime != _DB_MTIME or db_path != _DB_PATH or overrides_changed:
        if _TRANSACTIONS_CACHE is not None:
            reason = "mtime changed" if current_mtime != _DB_MTIME else "db_path changed"
            if overrides_changed: reason = "overrides changed"
            logging.info(f"Reloading transactions ({reason}). New path: {db_path}")
        
        logging.info(f"Loading transactions from: {db_path}")
        
        # --- 2b. Load manual_overrides.json ---
        global _MANUAL_OVERRIDES_FILE_CACHE, _MANUAL_OVERRIDES_MTIME
        
        manual_overrides = {} 
        full_overrides_json = {} 
        
        overrides_paths_to_try = []
        if config_loaded_path:
            config_dir = os.path.dirname(config_loaded_path)
            overrides_paths_to_try.append(os.path.join(config_dir, "manual_overrides.json"))

        overrides_paths_to_try.append(os.path.join(config.get_app_data_dir(), "manual_overrides.json"))
        overrides_paths_to_try.append(os.path.join(project_root, "manual_overrides.json"))
        
        current_overrides_path = None
        for op in overrides_paths_to_try:
            if os.path.exists(op):
                current_overrides_path = op
                break
        
        if current_overrides_path:
            st_ov = os.stat(current_overrides_path)
            if _MANUAL_OVERRIDES_FILE_CACHE is None or current_overrides_path != _OVERRIDES_PATH or st_ov.st_mtime != _MANUAL_OVERRIDES_MTIME:
                try:
                    with open(current_overrides_path, "r") as f:
                        _MANUAL_OVERRIDES_FILE_CACHE = json.load(f)
                    _OVERRIDES_PATH = current_overrides_path
                    _MANUAL_OVERRIDES_MTIME = st_ov.st_mtime
                    logging.info(f"Loaded/Reloaded manual overrides from {current_overrides_path}")
                except Exception as e:
                    logging.warning(f"Found overrides at {current_overrides_path} but failed to load: {e}")
                    if _MANUAL_OVERRIDES_FILE_CACHE is None: _MANUAL_OVERRIDES_FILE_CACHE = {}
        else:
            _MANUAL_OVERRIDES_FILE_CACHE = {}
            _OVERRIDES_PATH = None
            _MANUAL_OVERRIDES_MTIME = 0.0

        full_overrides_json = _MANUAL_OVERRIDES_FILE_CACHE
        if "manual_price_overrides" in full_overrides_json:
            manual_overrides = full_overrides_json["manual_price_overrides"]
        
        loaded_overrides_path = _OVERRIDES_PATH

        
        # --- CHANGED: JSON overrides are the ONLY authority ---
        # We no longer merge from config.
        # Initialize defaults from explicit JSON load or empty dict
        final_manual_overrides = manual_overrides
        
        if loaded_overrides_path:
             logging.info(f"Using overrides from {loaded_overrides_path}.")
             _OVERRIDES_PATH = loaded_overrides_path
             _OVERRIDES_MTIME = os.path.getmtime(loaded_overrides_path)
        
        # ------------------------------------------------
        
        # Merge other collections from JSON if present
        if "user_excluded_symbols" in full_overrides_json:
            loaded_excluded = full_overrides_json.get("user_excluded_symbols", [])
            if isinstance(loaded_excluded, list):
                clean_excluded = {s.upper().strip() for s in loaded_excluded if isinstance(s, str)}
                user_excluded_symbols.update(clean_excluded)
        if "user_symbol_map" in full_overrides_json:
             user_symbol_map.update(full_overrides_json.get("user_symbol_map", {}))

        try:
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
            _MANUAL_OVERRIDES = final_manual_overrides # FIX: Use final_manual_overrides here
            _USER_SYMBOL_MAP = user_symbol_map
            _USER_EXCLUDED_SYMBOLS = user_excluded_symbols
            _ACCOUNT_CURRENCY_MAP = account_currency_map
            _DB_PATH = db_path
            _DB_MTIME = current_mtime
            
            logging.info(f"Loaded {len(df)} transactions.")
        except Exception as e:
            logging.error(f"Error loading transactions: {e}", exc_info=True)
            return pd.DataFrame(), {}, {}, set(), {}, "", 0.0

    # Return cached data
    return _TRANSACTIONS_CACHE, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATH, _DB_MTIME

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

from config_manager import ConfigManager

def get_config_manager() -> ConfigManager:
    """Dependency that provides a ConfigManager instance."""
    # Use centralized path logic from config.py via get_app_data_dir()
    app_data_dir = config.get_app_data_dir()
    return ConfigManager(app_data_dir)

def reload_config():
    """Forces a reload of global configuration cache."""
    # This is a placeholder if we need to reload global vars derived from config
    # Currently get_transaction_data handles its own reloading of config files on each call if needed/logic allows
    # But for immediate effect of overrides, we might need to clear _TRANSACTIONS_CACHE
    reload_data()
