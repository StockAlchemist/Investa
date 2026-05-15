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
import sys

# Ensure src is in path for imports
current_file_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(current_file_path))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

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

# Moved down to support get_current_user dependency

def get_current_user(
    token: str = Depends(oauth2_scheme),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = decode_access_token(token)
    if token_data is None:
        raise credentials_exception
    
    # Connect to GLOBAL DB for Auth - Now uses the 'conn' dependency which handles lifecycle and retries
    try:
        cursor = conn.cursor()
        # Transient retry for cloud drives
        for i in range(3):
            try:
                cursor.execute("SELECT id, username, is_active, created_at, alias FROM users WHERE username = ?", (token_data.username,))
                row = cursor.fetchone()
                break
            except sqlite3.OperationalError as e:
                if "disk i/o error" in str(e).lower() and i < 2:
                    logging.warning(f"Transient disk I/O error during auth query. Retry {i+1}/3...")
                    import time
                    time.sleep(0.1 * (i + 1))
                    continue
                raise
                
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
    except sqlite3.Error as e:
        logging.error(f"Database error during authentication: {e}")
        raise HTTPException(status_code=503, detail="Authentication database temporarily unavailable")

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


# Global cache for transactions to avoid reloading on every request.
# --- KEY: username (str) --- Stable, filesystem-aligned, immune to user_id integer drift.
_TRANSACTIONS_CACHE: Dict[str, pd.DataFrame] = {}
_DATA_LOADING_LOCK = threading.Lock()  # Lock to prevent concurrent reloads (OOM protection)
_IGNORED_INDICES: Dict[str, Set[int]] = {}
_IGNORED_REASONS: Dict[str, Dict[int, str]] = {}

_MANUAL_OVERRIDES: Dict[str, Dict[str, Any]] = {}
_USER_SYMBOL_MAP: Dict[str, Dict[str, str]] = {}
_USER_EXCLUDED_SYMBOLS: Dict[str, Set[str]] = {}
_ACCOUNT_CURRENCY_MAP: Dict[str, Dict[str, str]] = {}
_ACCOUNT_CASH_MODE_MAP: Dict[str, Dict[str, str]] = {}

_DB_PATHS: Dict[str, str] = {}
_DB_MTIMES: Dict[str, float] = {}
_OVERRIDES_PATHS: Dict[str, str] = {}
_OVERRIDES_MTIMES: Dict[str, float] = {}

_GUI_CONFIG_CACHES: Dict[str, Dict[str, Any]] = {}
_GUI_CONFIG_LOADED_PATHS: Dict[str, str] = {}
_GUI_CONFIG_MTIMES: Dict[str, float] = {}
_MANUAL_OVERRIDES_FILE_CACHES: Dict[str, Dict[str, Any]] = {}
_MANUAL_OVERRIDES_FILE_MTIMES: Dict[str, float] = {}


def get_transaction_data(current_user: User = Depends(get_current_user)) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str], Set[str], Dict[str, str], Dict[str, str], str, float]:
    """
    Loads transaction data from the database for the current user.
    Uses username (str) as the cache key — stable and filesystem-aligned.
    Auto-reloads when the DB file modification time changes.
    """
    global _TRANSACTIONS_CACHE, _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _DB_PATHS, _DB_MTIMES, _OVERRIDES_PATHS, _OVERRIDES_MTIMES

    # --- Use username (str) as the sole, stable cache key ---
    username = current_user.username

    # --- User Data Directory ---
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, username)
    os.makedirs(user_data_dir, exist_ok=True)

    # ---------------------------
    # --- 1. Load User Configuration (canonical path: <user_dir>/config/) ---
    # ---------------------------
    config_dir = os.path.join(user_data_dir, config.CONFIG_DIR)
    gui_config_path = os.path.join(config_dir, config.GUI_CONFIG_FILENAME)

    gui_config = {}
    if os.path.exists(gui_config_path):
        try:
            with open(gui_config_path, "r") as f:
                gui_config = json.load(f)
        except Exception:
            pass

    # --- 1b. Defaults ---
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
    if "transactions_file" in gui_config and os.path.exists(gui_config["transactions_file"]):
        db_path = gui_config["transactions_file"]

    # --- 3. Check DB modification time ---
    current_mtime = 0.0
    if os.path.exists(db_path):
        current_mtime = os.path.getmtime(db_path)
        for ext in ("-wal", "-shm"):
            side_file = db_path + ext
            if os.path.exists(side_file):
                side_mtime = os.path.getmtime(side_file)
                if side_mtime > current_mtime:
                    current_mtime = side_mtime

    # --- 4. Check overrides modification time ---
    overrides_changed = False
    user_overrides_path = _OVERRIDES_PATHS.get(username)
    if user_overrides_path and os.path.exists(user_overrides_path):
        current_overrides_mtime = os.path.getmtime(user_overrides_path)
        if current_overrides_mtime != _OVERRIDES_MTIMES.get(username, 0.0):
            overrides_changed = True

    user_cache_exists = username in _TRANSACTIONS_CACHE
    db_mtime_changed = current_mtime != _DB_MTIMES.get(username, 0.0)
    db_path_changed = db_path != _DB_PATHS.get(username)
    db_needs_reload = not user_cache_exists or db_mtime_changed or db_path_changed

    if db_needs_reload or overrides_changed:
        with _DATA_LOADING_LOCK:
            # Double-checked locking
            is_cache_fresh = (
                username in _TRANSACTIONS_CACHE
                and _DB_MTIMES.get(username) == current_mtime
                and _DB_PATHS.get(username) == db_path
                and not overrides_changed
            )

            if is_cache_fresh:
                logging.info(f"Skipping reload for user '{username}', handled by another thread.")
            else:
                if db_needs_reload:
                    logging.info(f"Loading/Reloading transactions for '{username}' from: {db_path}")
                else:
                    logging.info(f"Reloading only overrides for '{username}' (DB is fresh).")

                # --- 4b. Load manual_overrides.json (canonical: <user_dir>/config/) ---
                global _MANUAL_OVERRIDES_FILE_CACHES, _MANUAL_OVERRIDES_FILE_MTIMES

                overrides_path = os.path.join(config_dir, config.MANUAL_OVERRIDES_FILENAME)
                full_overrides_json = {}

                if os.path.exists(overrides_path):
                    st_ov = os.stat(overrides_path)
                    old_mtime = _MANUAL_OVERRIDES_FILE_MTIMES.get(username, 0.0)
                    if username not in _MANUAL_OVERRIDES_FILE_CACHES or st_ov.st_mtime != old_mtime:
                        try:
                            with open(overrides_path, "r") as f:
                                _MANUAL_OVERRIDES_FILE_CACHES[username] = json.load(f)
                            _MANUAL_OVERRIDES_FILE_MTIMES[username] = st_ov.st_mtime
                        except Exception as e:
                            logging.warning(f"Failed to load overrides at {overrides_path}: {e}")
                    full_overrides_json = _MANUAL_OVERRIDES_FILE_CACHES.get(username, {})
                    _OVERRIDES_PATHS[username] = overrides_path
                    _OVERRIDES_MTIMES[username] = os.path.getmtime(overrides_path)
                else:
                    _MANUAL_OVERRIDES_FILE_CACHES[username] = {}
                    _OVERRIDES_PATHS[username] = ""
                    _OVERRIDES_MTIMES[username] = 0.0

                final_manual_overrides = full_overrides_json.get("manual_price_overrides", {})

                if "user_excluded_symbols" in full_overrides_json:
                    loaded_excluded = full_overrides_json.get("user_excluded_symbols", [])
                    if isinstance(loaded_excluded, list):
                        user_excluded_symbols.update({s.upper().strip() for s in loaded_excluded if isinstance(s, str)})
                if "user_symbol_map" in full_overrides_json:
                    user_symbol_map.update(full_overrides_json.get("user_symbol_map", {}))

                try:
                    is_db = db_path.lower().endswith((".db", ".sqlite", ".sqlite3"))

                    if db_needs_reload:
                        df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
                            source_path=db_path,
                            account_currency_map=account_currency_map,
                            default_currency=default_currency,
                            is_db_source=is_db
                        )
                    else:
                        df = _TRANSACTIONS_CACHE.get(username, pd.DataFrame())
                        ignored_indices = _IGNORED_INDICES.get(username, set())
                        ignored_reasons = _IGNORED_REASONS.get(username, {})

                    # --- No user_id filtering: file-level isolation IS the isolation mechanism.
                    # Each user has their own portfolio.db. Filtering by user_id causes NULL-row
                    # drops and ID-drift bugs. See DB Organization Plan, Phase 4.

                    _TRANSACTIONS_CACHE[username] = df
                    _IGNORED_INDICES[username] = ignored_indices
                    _IGNORED_REASONS[username] = ignored_reasons
                    _MANUAL_OVERRIDES[username] = final_manual_overrides
                    _USER_SYMBOL_MAP[username] = user_symbol_map
                    _USER_EXCLUDED_SYMBOLS[username] = user_excluded_symbols
                    _ACCOUNT_CURRENCY_MAP[username] = account_currency_map
                    _ACCOUNT_CASH_MODE_MAP[username] = account_cash_mode_map
                    _DB_PATHS[username] = db_path
                    _DB_MTIMES[username] = current_mtime

                    logging.info(f"Loaded {len(df)} transactions for user '{username}'.")
                except Exception as e:
                    logging.error(f"Error loading transactions for '{username}': {e}", exc_info=True)
                    return pd.DataFrame(), {}, {}, set(), {}, {}, "", 0.0

    return (
        _TRANSACTIONS_CACHE.get(username, pd.DataFrame()),
        _MANUAL_OVERRIDES.get(username, {}),
        _USER_SYMBOL_MAP.get(username, {}),
        _USER_EXCLUDED_SYMBOLS.get(username, set()),
        _ACCOUNT_CURRENCY_MAP.get(username, {}),
        _ACCOUNT_CASH_MODE_MAP.get(username, {}),
        _DB_PATHS.get(username, ""),
        _DB_MTIMES.get(username, 0.0)
    )

def reload_data(username: Optional[str] = None):
    """Forces a full reload of transaction data and settings.
    
    If username is provided, clears only that user's cache.
    If None, clears ALL users' caches.
    """
    global _TRANSACTIONS_CACHE, _DB_MTIMES, _DB_PATHS, _OVERRIDES_MTIMES, _OVERRIDES_PATHS
    global _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP, _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP
    global _GUI_CONFIG_CACHES, _MANUAL_OVERRIDES_FILE_CACHES
    if username is not None:
        for d in [_TRANSACTIONS_CACHE, _DB_MTIMES, _DB_PATHS, _OVERRIDES_MTIMES, _OVERRIDES_PATHS,
                  _IGNORED_INDICES, _IGNORED_REASONS, _MANUAL_OVERRIDES, _USER_SYMBOL_MAP,
                  _USER_EXCLUDED_SYMBOLS, _ACCOUNT_CURRENCY_MAP, _GUI_CONFIG_CACHES,
                  _MANUAL_OVERRIDES_FILE_CACHES]:
            d.pop(username, None)
        logging.info(f"Data cache cleared for user '{username}'.")
    else:
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


def clear_settings_cache(username: Optional[str] = None):
    """Clears settings/overrides cache, forcing a reload on next access.
    
    If username is provided, clears only that user. If None, clears all.
    """
    global _MANUAL_OVERRIDES_FILE_MTIMES, _OVERRIDES_MTIMES
    if username is not None:
        _MANUAL_OVERRIDES_FILE_MTIMES.pop(username, None)
        _OVERRIDES_MTIMES.pop(username, None)
    else:
        _MANUAL_OVERRIDES_FILE_MTIMES.clear()
        _OVERRIDES_MTIMES.clear()
    logging.info(f"Settings cache cleared for {'user ' + str(username) if username else 'all users'}.")

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

# Moved up to support get_current_user dependency

