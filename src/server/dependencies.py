import os
import sys
import pandas as pd
import logging
import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Iterator, Optional, Tuple, Set, Dict, Any
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from server.auth import User, decode_access_token
from db_utils import get_db_connection
from data_loader import load_and_clean_transactions

# Ensure src is in path for imports
current_file_path = os.path.abspath(__file__)
src_path = os.path.dirname(os.path.dirname(current_file_path))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

import config  # noqa: E402


# --- Auth Dependency ---
# auto_error=False so a missing Authorization header doesn't 401 outright — we
# fall back to the httpOnly auth cookie (web app) before deciding.
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login", auto_error=False)

def get_global_db_connection() -> Iterator[sqlite3.Connection]:
    """Dependency for Global DB Connection (Auth)."""
    global_db_path = os.path.join(config.get_app_data_dir(), config.DB_DIR, config.GLOBAL_DB_FILENAME)
    conn = get_db_connection(global_db_path, check_same_thread=False, use_cache=False)
    if not conn:
        raise HTTPException(status_code=500, detail="Global Database unavailable")
    try:
        yield conn
    finally:
        conn.close()


def get_current_user(
    request: Request,
    token: Optional[str] = Depends(oauth2_scheme),
    conn: sqlite3.Connection = Depends(get_global_db_connection)
) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    # The Authorization: Bearer header (native apps) takes precedence; fall back
    # to the httpOnly cookie set for the web app at login.
    if not token:
        token = request.cookies.get(config.AUTH_COOKIE_NAME)
    token_data = decode_access_token(token) if token else None
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

def get_user_db_connection(current_user: User = Depends(get_current_user)) -> Iterator[sqlite3.Connection]:
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


# --- Per-user transaction/settings cache ---

TransactionData = Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str], Set[str], Dict[str, str], Dict[str, str], str, float]

_EMPTY_RESULT: TransactionData = (pd.DataFrame(), {}, {}, set(), {}, {}, "", 0.0)


@dataclass
class _UserCacheEntry:
    """Everything cached for one user. Entries are built off to the side and
    swapped into the cache as a unit, so readers never observe a half-updated
    mix of old and new state (the old design spread this across 12 dicts)."""
    transactions: pd.DataFrame = field(default_factory=pd.DataFrame)
    ignored_indices: Set[int] = field(default_factory=set)
    ignored_reasons: Dict[int, str] = field(default_factory=dict)
    manual_overrides: Dict[str, Any] = field(default_factory=dict)
    symbol_map: Dict[str, str] = field(default_factory=dict)
    excluded_symbols: Set[str] = field(default_factory=set)
    account_currency_map: Dict[str, str] = field(default_factory=dict)
    account_cash_mode_map: Dict[str, str] = field(default_factory=dict)
    db_path: str = ""
    db_mtime: float = 0.0
    overrides_path: str = ""
    overrides_mtime: float = 0.0
    overrides_file_cache: Dict[str, Any] = field(default_factory=dict)
    overrides_file_mtime: float = 0.0

    def as_tuple(self) -> TransactionData:
        return (
            self.transactions,
            self.manual_overrides,
            self.symbol_map,
            self.excluded_symbols,
            self.account_currency_map,
            self.account_cash_mode_map,
            self.db_path,
            self.db_mtime,
        )


class UserDataCache:
    """Per-user transaction/settings cache, keyed by username.

    Usernames are stable and filesystem-aligned (immune to user_id integer
    drift). Reloads happen when the user's portfolio DB (or its WAL/SHM side
    files) or manual_overrides.json changes on disk.
    """

    def __init__(self):
        self._entries: Dict[str, _UserCacheEntry] = {}
        # One global load lock on purpose: parsing a large transaction history
        # is memory-heavy, so at most one (re)load runs at a time (OOM protection).
        self._load_lock = threading.Lock()

    # ---- public API ----

    def get_or_load(self, username: str) -> TransactionData:
        user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, username)
        os.makedirs(user_data_dir, exist_ok=True)
        config_dir = os.path.join(user_data_dir, config.CONFIG_DIR)

        # User configuration is read fresh on every call — it determines the
        # DB path and currency mapping, which drive the reload decision.
        gui_config = self._read_gui_config(config_dir)

        account_currency_map = {"SET": "THB"}
        account_currency_map.update(gui_config.get("account_currency_map", {}))
        account_cash_mode_map = dict(gui_config.get("account_cash_mode_map", {}))

        db_path = os.path.join(user_data_dir, config.PORTFOLIO_DB_FILENAME)
        if "transactions_file" in gui_config and os.path.exists(gui_config["transactions_file"]):
            db_path = gui_config["transactions_file"]

        db_mtime = self._effective_db_mtime(db_path)
        entry = self._entries.get(username)
        db_needs_reload = entry is None or db_mtime != entry.db_mtime or db_path != entry.db_path

        if db_needs_reload or self._overrides_changed(entry):
            with self._load_lock:
                # Double-checked: another thread may have reloaded while we waited.
                entry = self._entries.get(username)
                fresh = (
                    entry is not None
                    and entry.db_mtime == db_mtime
                    and entry.db_path == db_path
                    and not self._overrides_changed(entry)
                )
                if fresh:
                    logging.info(f"Skipping reload for user '{username}', handled by another thread.")
                else:
                    db_needs_reload = entry is None or db_mtime != entry.db_mtime or db_path != entry.db_path
                    new_entry = self._load(
                        username, entry, config_dir, db_path, db_mtime,
                        db_needs_reload, account_currency_map, account_cash_mode_map,
                    )
                    if new_entry is None:
                        return _EMPTY_RESULT
                    self._entries[username] = new_entry

        entry = self._entries.get(username)
        return entry.as_tuple() if entry is not None else _EMPTY_RESULT

    def invalidate(self, username: Optional[str] = None) -> None:
        """Drop cached data so the next access reloads from disk."""
        if username is not None:
            self._entries.pop(username, None)
            logging.info(f"Data cache cleared for user '{username}'.")
        else:
            self._entries.clear()
            logging.info("ALL data caches cleared for all users.")

    def clear_settings(self, username: Optional[str] = None) -> None:
        """Force overrides/settings to be re-read on next access (keeps the DB cache)."""
        targets = [self._entries.get(username)] if username is not None else list(self._entries.values())
        for entry in targets:
            if entry is not None:
                entry.overrides_mtime = 0.0
                entry.overrides_file_mtime = 0.0
        logging.info(f"Settings cache cleared for {'user ' + str(username) if username else 'all users'}.")

    # ---- internals ----

    @staticmethod
    def _read_gui_config(config_dir: str) -> Dict[str, Any]:
        gui_config_path = os.path.join(config_dir, config.GUI_CONFIG_FILENAME)
        if os.path.exists(gui_config_path):
            try:
                with open(gui_config_path, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _effective_db_mtime(db_path: str) -> float:
        """DB mtime, also considering the WAL side file (writes may land there first).

        The -shm file is deliberately excluded: it is WAL's transient shared-memory
        index and gets touched by mere *reads*, so including it made every summary
        request invalidate the cache that the previous one had just populated.
        Durable changes always bump the main DB file or -wal.
        """
        if not os.path.exists(db_path):
            return 0.0
        mtime = os.path.getmtime(db_path)
        for ext in ("-wal",):
            side_file = db_path + ext
            if os.path.exists(side_file):
                mtime = max(mtime, os.path.getmtime(side_file))
        return mtime

    @staticmethod
    def _overrides_changed(entry: Optional[_UserCacheEntry]) -> bool:
        if entry is None or not entry.overrides_path or not os.path.exists(entry.overrides_path):
            return False
        return os.path.getmtime(entry.overrides_path) != entry.overrides_mtime

    def _load(
        self,
        username: str,
        prev: Optional[_UserCacheEntry],
        config_dir: str,
        db_path: str,
        db_mtime: float,
        db_needs_reload: bool,
        account_currency_map: Dict[str, str],
        account_cash_mode_map: Dict[str, str],
    ) -> Optional[_UserCacheEntry]:
        """Build a fresh cache entry. Returns None if loading failed."""
        if db_needs_reload:
            logging.info(f"Loading/Reloading transactions for '{username}' from: {db_path}")
        else:
            logging.info(f"Reloading only overrides for '{username}' (DB is fresh).")

        # manual_overrides.json — parsed copy cached by mtime
        overrides_path = os.path.join(config_dir, config.MANUAL_OVERRIDES_FILENAME)
        overrides_file_cache = prev.overrides_file_cache if prev is not None else {}
        overrides_file_mtime = prev.overrides_file_mtime if prev is not None else 0.0
        new_overrides_path = ""
        new_overrides_mtime = 0.0
        if os.path.exists(overrides_path):
            st_mtime = os.stat(overrides_path).st_mtime
            if st_mtime != overrides_file_mtime:
                try:
                    with open(overrides_path, "r") as f:
                        overrides_file_cache = json.load(f)
                    overrides_file_mtime = st_mtime
                except Exception as e:
                    # Keep the previous parsed copy; mtime stays stale so we retry next time.
                    logging.warning(f"Failed to load overrides at {overrides_path}: {e}")
            new_overrides_path = overrides_path
            new_overrides_mtime = os.path.getmtime(overrides_path)
        else:
            overrides_file_cache = {}
            overrides_file_mtime = 0.0

        manual_overrides = overrides_file_cache.get("manual_price_overrides", {})

        excluded_symbols = set(config.YFINANCE_EXCLUDED_SYMBOLS)
        loaded_excluded = overrides_file_cache.get("user_excluded_symbols", [])
        if isinstance(loaded_excluded, list):
            excluded_symbols.update({s.upper().strip() for s in loaded_excluded if isinstance(s, str)})
        symbol_map = dict(overrides_file_cache.get("user_symbol_map", {}))

        try:
            if db_needs_reload:
                is_db = db_path.lower().endswith((".db", ".sqlite", ".sqlite3"))
                df, _, ignored_indices, ignored_reasons, _, _, _ = load_and_clean_transactions(
                    source_path=db_path,
                    account_currency_map=account_currency_map,
                    default_currency=config.DEFAULT_CURRENCY,
                    is_db_source=is_db,
                )
            else:
                df = prev.transactions if prev is not None else pd.DataFrame()
                ignored_indices = prev.ignored_indices if prev is not None else set()
                ignored_reasons = prev.ignored_reasons if prev is not None else {}
        except Exception as e:
            logging.error(f"Error loading transactions for '{username}': {e}", exc_info=True)
            return None

        # No user_id filtering: file-level isolation IS the isolation mechanism.
        # Each user has their own portfolio.db. Filtering by user_id causes NULL-row
        # drops and ID-drift bugs. See DB Organization Plan, Phase 4.

        if df is None:
            df = pd.DataFrame()
        logging.info(f"Loaded {len(df)} transactions for user '{username}'.")
        return _UserCacheEntry(
            transactions=df,
            ignored_indices=ignored_indices,
            ignored_reasons=ignored_reasons,
            manual_overrides=manual_overrides,
            symbol_map=symbol_map,
            excluded_symbols=excluded_symbols,
            account_currency_map=account_currency_map,
            account_cash_mode_map=account_cash_mode_map,
            db_path=db_path,
            db_mtime=db_mtime,
            overrides_path=new_overrides_path,
            overrides_mtime=new_overrides_mtime,
            overrides_file_cache=overrides_file_cache,
            overrides_file_mtime=overrides_file_mtime,
        )


user_data_cache = UserDataCache()


def get_transaction_data(current_user: User = Depends(get_current_user)) -> TransactionData:
    """FastAPI dependency: the current user's transactions + settings (cached).

    Returns (transactions_df, manual_overrides, symbol_map, excluded_symbols,
    account_currency_map, account_cash_mode_map, db_path, db_mtime).
    """
    return user_data_cache.get_or_load(current_user.username)


def reload_data(username: Optional[str] = None):
    """Forces a full reload of transaction data and settings.

    If username is provided, clears only that user's cache.
    If None, clears ALL users' caches.
    """
    user_data_cache.invalidate(username)


def clear_settings_cache(username: Optional[str] = None):
    """Clears settings/overrides cache, forcing a reload on next access.

    If username is provided, clears only that user. If None, clears all.
    """
    user_data_cache.clear_settings(username)


from config_manager import ConfigManager  # noqa: E402

def get_config_manager(current_user: User = Depends(get_current_user)) -> ConfigManager:
    """Dependency that provides a User-specific ConfigManager."""
    user_data_dir = os.path.join(config.get_app_data_dir(), config.USERS_DIR, current_user.username)
    # Ensure dir exists
    os.makedirs(user_data_dir, exist_ok=True)
    return ConfigManager(user_data_dir)

def reload_config():
    """Forces a reload of global configuration cache."""
    reload_data()
