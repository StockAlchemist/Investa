import os
import sys
import pandas as pd
import logging
import json
import sqlite3
import threading
from typing import Optional, Tuple, Set, Dict, Any
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from server.auth import User, decode_access_token
from db_utils import get_db_connection
from data_loader import load_and_clean_transactions
import config
from config import GLOBAL_DB_FILENAME

# Ensure src is in path (redundant if imported from main, but good for standalone testing)
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)


logging.info("DEPENDENCIES MODULE LOADED - VERSION 3 (Auth Enabled)")

# --- Auth Dependency ---
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = decode_access_token(token)
    if token_data is None:
        raise credentials_exception
    
    # In a real app we might fetch the user from DB to ensure they still exist/active
    # For now, we trust the token's claim if signature is valid, 
    # but let's do a quick DB check to fail if user deleted.
    # Connect to GLOBAL DB for Auth
    global_db_path = os.path.join(config.get_app_data_dir(), config.DB_DIR, GLOBAL_DB_FILENAME)
    conn = get_db_connection(global_db_path, check_same_thread=False)
    if conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, username, is_active, created_at, alias FROM users WHERE username = ?", (token_data.username,))
        row = cursor.fetchone()
        if row is None:
            raise credentials_exception
        user = User(
            id=row[0], 
            username=row[1], 
            is_active=bool(row[2]), 
            created_at=row[3],
            alias=row[4]
        )
        if not user.is_active:
             raise HTTPException(status_code=400, detail="Inactive user")
        return user
    
    # Fallback if DB fails (shouldn't happen)
    raise credentials_exception


# Global cache for transactions to avoid reloading on every request
# Keyed by user_id to ensure isolation
_TRANSACTIONS_CACHE: Dict[int, pd.DataFrame] = {} 
_DATA_LOADING_LOCK = threading.Lock() # Lock to prevent concurrent reloads (OOM protection)
_IGNORED_INDICES: Dict[int, Set[int]] = {}
_IGNORED_REASONS: Dict[int, Dict[int, str]] = {}

# These might remain global if they are system-wide, OR per user if overrides are per user. 
# Plan says: "Assign all existing rows to user_id=1". 
# For now, let's assume overrides are system-wide (legacy) or we need to scope them. 
# Let's scope them by user_id in memory.
_MANUAL_OVERRIDES: Dict[int, Dict[str, Any]] = {}
_USER_SYMBOL_MAP: Dict[int, Dict[str, str]] = {}
_USER_EXCLUDED_SYMBOLS: Dict[int, Set[str]] = {}
_ACCOUNT_CURRENCY_MAP: Dict[int, Dict[str, str]] = {}
_ACCOUNT_CASH_MODE_MAP: Dict[int, Dict[str, str]] = {}

_DB_PATHS: Dict[int, str] = {}
_DB_MTIMES: Dict[int, float] = {}
_OVERRIDES_PATHS: Dict[int, str] = {}
_OVERRIDES_MTIMES: Dict[int, float] = {}

# Added caches for config files to avoid redundant I/O
_GUI_CONFIG_CACHES: Dict[int, Dict[str, Any]] = {}
_GUI_CONFIG_LOADED_PATHS: Dict[int, str] = {}
_GUI_CONFIG_MTIMES: Dict[int, float] = {}
_MANUAL_OVERRIDES_FILE_CACHES: Dict[int, Dict[str, Any]] = {}
_MANUAL_OVERRIDES_FILE_MTIMES: Dict[int, float] = {}


def get_transaction_data(current_user: User = Depends(get_current_user)) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str], Set[str], Dict[str, str], str, float]:
    """
    Loads transaction data from the database for the current user.
    Checks for file modification to auto-reload.
    """
    global _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATHS, _DB_MTIMES, _OVERRIDES_PATHS, _OVERRIDES_MTIMES
    
    user_id = current_user.id
    username = current_user.username
    
    # Define User Data Directory
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, username)
    
    # ---------------------------
    # --- 1. Load User Configuration ---
    # ---------------------------
    # We load gui_config.json from the user's directory fresh or check mtime
    
    # Detect config location (Root vs Config Subdir)
    root_config_path = os.path.join(user_data_dir, config.GUI_CONFIG_FILENAME)
    config_subdir_path = os.path.join(user_data_dir, config.CONFIG_DIR, config.GUI_CONFIG_FILENAME)
    
    current_gui_config_path = root_config_path
    if not os.path.exists(root_config_path) and os.path.exists(config_subdir_path):
        current_gui_config_path = config_subdir_path
    
    gui_config = {}
    if os.path.exists(current_gui_config_path):
         try:
             with open(current_gui_config_path, "r") as f:
                 gui_config = json.load(f)
         except Exception: pass

    config_loaded_path = current_gui_config_path

    # --- 1b. Defaults from config.py ---
    account_currency_map = {"SET": "THB"}
    default_currency = config.DEFAULT_CURRENCY
    user_symbol_map = {} 
    user_excluded_symbols = set(config.YFINANCE_EXCLUDED_SYMBOLS.copy())
    
    if "account_currency_map" in gui_config:
         account_currency_map.update(gui_config["account_currency_map"])
    
    account_cash_mode_map = {}
    if "account_cash_mode_map" in gui_config:
         account_cash_mode_map.update(gui_config["account_cash_mode_map"])

    # --- 2. Determine DB Path for User ---
    db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
    if not os.path.exists(user_data_dir):
        os.makedirs(user_data_dir, exist_ok=True)
    
    # Check modification time for DB (and WAL/SHM if present)
    current_mtime = 0.0
    if os.path.exists(db_path):
        current_mtime = os.path.getmtime(db_path)
        wal_path = db_path + "-wal"
        shm_path = db_path + "-shm"
        if os.path.exists(wal_path):
            wal_mtime = os.path.getmtime(wal_path)
            if wal_mtime > current_mtime: current_mtime = wal_mtime
        if os.path.exists(shm_path):
            shm_mtime = os.path.getmtime(shm_path)
            if shm_mtime > current_mtime: current_mtime = shm_mtime
    
    # Check modification time for Overrides
    overrides_changed = False
    user_overrides_path = _OVERRIDES_PATHS.get(user_id)
    if user_overrides_path and os.path.exists(user_overrides_path):
        current_overrides_mtime = os.path.getmtime(user_overrides_path)
        if current_overrides_mtime != _OVERRIDES_MTIMES.get(user_id, 0.0):
            overrides_changed = True

    user_cache_exists = user_id in _TRANSACTIONS_CACHE
    db_mtime_changed = current_mtime != _DB_MTIMES.get(user_id, 0.0)
    db_path_changed = db_path != _DB_PATHS.get(user_id)
    db_needs_reload = not user_cache_exists or db_mtime_changed or db_path_changed
    
    if db_needs_reload or overrides_changed:
        # CRITICAL FIX: Prevent concurrent reloads which cause OOM
        # Multiple requests (dashboard parts) hit this simultaneously after cache clear.
        with _DATA_LOADING_LOCK:
            # Double-checked locking: Re-evaluate condition inside lock
            user_cache_exists = user_id in _TRANSACTIONS_CACHE
            # Note: Checking _DB_MTIME vs current_mtime again is tricky if another thread updated it.
            # Ideally we check if the cache is now valid.
            
            # Simple check: If cache was populated by another thread while we waited, skip load
            # UNLESS the mtime mismatch was the reason we entered (which means we still need to load if it's old)
            
            # Let's verify if the cache state is "fresh enough"
            is_cache_fresh = (
                user_id in _TRANSACTIONS_CACHE and 
                _DB_MTIMES.get(user_id) == current_mtime and 
                _DB_PATHS.get(user_id) == db_path and 
                not overrides_changed
            )
            
            if is_cache_fresh:
                logging.info(f"Skipping reload for user {user_id}, duplicate request handled by another thread.")
            else:
                if db_needs_reload:
                    logging.info(f"Loading/Reloading transactions for user {user_id} from: {db_path}")
                else:
                    logging.info(f"Reloading only overrides/settings for user {user_id} (Database is fresh).")
                
                # --- 2b. Load manual_overrides.json ---
                global _MANUAL_OVERRIDES_FILE_CACHES, _MANUAL_OVERRIDES_FILE_MTIMES
                
                manual_overrides = {} 
                full_overrides_json = {} 
                overrides_paths_to_try = [] 
                
                if config_loaded_path:
                    config_dir = os.path.dirname(config_loaded_path)
                    overrides_paths_to_try.append(os.path.join(config_dir, config.MANUAL_OVERRIDES_FILENAME))
                
                overrides_paths_to_try.append(os.path.join(user_data_dir, config.MANUAL_OVERRIDES_FILENAME))
                overrides_paths_to_try.append(os.path.join(user_data_dir, config.CONFIG_DIR, config.MANUAL_OVERRIDES_FILENAME))
                
                current_overrides_path = None
                for op in overrides_paths_to_try:
                    if os.path.exists(op):
                        current_overrides_path = op
                        break
                
                if current_overrides_path:
                    st_ov = os.stat(current_overrides_path)
                    old_ov_mtime = _MANUAL_OVERRIDES_FILE_MTIMES.get(user_id, 0.0)
                    old_ov_path = _OVERRIDES_PATHS.get(user_id)
                    
                    if user_id not in _MANUAL_OVERRIDES_FILE_CACHES or current_overrides_path != old_ov_path or st_ov.st_mtime != old_ov_mtime:
                        try:
                            with open(current_overrides_path, "r") as f:
                                _MANUAL_OVERRIDES_FILE_CACHES[user_id] = json.load(f)
                            _OVERRIDES_PATHS[user_id] = current_overrides_path
                            _MANUAL_OVERRIDES_FILE_MTIMES[user_id] = st_ov.st_mtime
                        except Exception as e:
                            logging.warning(f"Failed to load overrides at {current_overrides_path}: {e}")
                            _MANUAL_OVERRIDES_FILE_CACHES[user_id] = _MANUAL_OVERRIDES_FILE_CACHES.get(user_id, {})
                else:
                    _MANUAL_OVERRIDES_FILE_CACHES[user_id] = {}
                    _OVERRIDES_PATHS[user_id] = ""
                    _MANUAL_OVERRIDES_FILE_MTIMES[user_id] = 0.0

                full_overrides_json = _MANUAL_OVERRIDES_FILE_CACHES.get(user_id, {})
                manual_overrides = full_overrides_json.get("manual_price_overrides", {})
                
                final_manual_overrides = manual_overrides
                
                if current_overrides_path:
                     _OVERRIDES_PATHS[user_id] = current_overrides_path
                     _OVERRIDES_MTIMES[user_id] = os.path.getmtime(current_overrides_path)
                
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
                    
                    # Load ALL data first
                    # TODO: Optimize to load only USER data at SQL level if possible
                    # But underlying logic 'load_and_clean_transactions' reads everything.
                    # We will filter dataframe after load.
                    
                    # Only reload from DB if needed
                    if db_needs_reload:
                        df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
                            source_path=db_path,
                            account_currency_map=account_currency_map,
                            default_currency=default_currency,
                            is_db_source=is_db
                        )
                    else:
                        # Use existing cache
                        df = _TRANSACTIONS_CACHE.get(user_id, pd.DataFrame())
                        ignored_indices = _IGNORED_INDICES.get(user_id, set())
                        ignored_reasons = _IGNORED_REASONS.get(user_id, {})
                    
                    # --- FILTER BY USER ID ---
                    # In Isolated Mode, the DB *only* contains this user's data (migrated).
                    # So df should be all theirs.
                    # However, for robustness, if we kept user_id column, we can filter or update it.
                    # If user_id is missing or updated to 1 during copy, we strictly don't care about the column filtering 
                    # as long as the file is isolated.
                    
                    # BUT: If we copied the DB, it has rows for user 1 (old testuser) or 3 (kitmatan).
                    # If we are kitmatan (id 3) and rows are id 3, fine.
                    # If migration DID NOT clean other users' data, we might see others?
                    # Architecture Plan says: "Clean up other users' data... pass" (Step 344)
                    # So migration copied EVERYTHING.
                    # So multiple users' data might exist in this file until we clean it.
                    # So we SHOULD filter by user_id to be safe, assuming the ID matches.
                    
                    # Wait, global ID might differ from local ID if we re-gen?
                    # No, we kept IDs stable in global DB migration.
                    
                    if not df.empty and "user_id" in df.columns:
                         df['user_id'] = pd.to_numeric(df['user_id'], errors='coerce')
                         # Filter only if user_id matches
                         df = df[df['user_id'] == user_id].copy()
                    
                    _TRANSACTIONS_CACHE[user_id] = df
                    _IGNORED_INDICES[user_id] = ignored_indices
                    _IGNORED_REASONS[user_id] = ignored_reasons
                    
                    # These are system-wide for now, but scoped in memory
                    _MANUAL_OVERRIDES[user_id] = final_manual_overrides
                    _USER_SYMBOL_MAP[user_id] = user_symbol_map
                    _USER_EXCLUDED_SYMBOLS[user_id] = user_excluded_symbols
                    _ACCOUNT_CURRENCY_MAP[user_id] = account_currency_map
                    _ACCOUNT_CASH_MODE_MAP[user_id] = account_cash_mode_map
                    
                    _DB_PATHS[user_id] = db_path
                    _DB_MTIMES[user_id] = current_mtime
                    
                    logging.info(f"Loaded {len(df)} transactions for user {user_id}.")
                except Exception as e:
                    logging.error(f"Error loading transactions for user {user_id}: {e}", exc_info=True)
                    return pd.DataFrame(), {}, {}, set(), {}, "", 0.0

    # Return cached data for specific user
    return (
        _TRANSACTIONS_CACHE.get(user_id, pd.DataFrame()),
        _MANUAL_OVERRIDES.get(user_id, {}),
        _USER_SYMBOL_MAP.get(user_id, {}),
        _USER_EXCLUDED_SYMBOLS.get(user_id, set()),
        _ACCOUNT_CURRENCY_MAP.get(user_id, {}),
        _ACCOUNT_CASH_MODE_MAP.get(user_id, {}),
        _DB_PATHS.get(user_id, ""),
        _DB_MTIMES.get(user_id, 0.0)
    )

def reload_data():
    """Forces a full reload of all transaction data and settings for all users."""
    global _TRANSACTIONS_CACHE, _DB_MTIMES, _DB_PATHS, _OVERRIDES_MTIMES, _OVERRIDES_PATHS
    global _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP
    global _GUI_CONFIG_CACHES, _MANUAL_OVERRIDES_FILE_CACHES
    _TRANSACTIONS_CACHE.clear()
    _DB_MTIMES.clear()
    _DB_PATHS.clear()
    _OVERRIDES_MTIMES.clear()
    _OVERRIDES_PATHS.clear()
    _IGNORED_INDICES.clear()
    _IGNORED_REASONS.clear()
    _MANUAL_OVERRIDES.clear()
    _USER_SYMBOL_MAP.clear()
    _USER_EXCLUDED_SYMBOLS.clear()
    _ACCOUNT_CURRENCY_MAP.clear()
    _GUI_CONFIG_CACHES.clear()
    _MANUAL_OVERRIDES_FILE_CACHES.clear()
    logging.info("ALL data caches cleared for all users.")

def clear_settings_cache(user_id: Optional[int] = None):
    """Clears settings and overrides cache, forcing a reload on next access without necessarily reloading the DB."""
    global _MANUAL_OVERRIDES_FILE_MTIMES, _OVERRIDES_MTIMES
    if user_id is not None:
        if user_id in _MANUAL_OVERRIDES_FILE_MTIMES: _MANUAL_OVERRIDES_FILE_MTIMES[user_id] = 0.0
        if user_id in _OVERRIDES_MTIMES: _OVERRIDES_MTIMES[user_id] = 0.0
    else:
        _MANUAL_OVERRIDES_FILE_MTIMES.clear()
        _OVERRIDES_MTIMES.clear()
    logging.info(f"Settings cache cleared for {'user ' + str(user_id) if user_id else 'all users'}.")

from config_manager import ConfigManager

def get_config_manager(current_user: User = Depends(get_current_user)) -> ConfigManager:
    """Dependency that provides a User-specific ConfigManager."""
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
    # Ensure dir exists
    os.makedirs(user_data_dir, exist_ok=True)
    return ConfigManager(user_data_dir)

def reload_config():
    """Forces a reload of global configuration cache."""
    # This is a placeholder if we need to reload global vars derived from config
    # Currently get_transaction_data handles its own reloading of config files on each call if needed/logic allows
    # But for immediate effect of overrides, we might need to clear _TRANSACTIONS_CACHE
    reload_data()

def get_global_db_connection() -> sqlite3.Connection:
    """Dependency for Global DB Connection (Auth)."""
    global_db_path = os.path.join(config.get_app_data_dir(), config.DB_DIR, config.GLOBAL_DB_FILENAME)
    conn = get_db_connection(global_db_path, check_same_thread=False, use_cache=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Global Database unavailable")
    try:
        yield conn
    finally:
        conn.close()

def get_user_db_connection(current_user: User = Depends(get_current_user)) -> sqlite3.Connection:
    """Dependency for User Portfolio DB Connection."""
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
    db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
    conn = get_db_connection(db_path, check_same_thread=False, use_cache=False)
    if not conn:
        raise HTTPException(status_code=500, detail="User Database unavailable")
    try:
        yield conn
    finally:
        conn.close()

