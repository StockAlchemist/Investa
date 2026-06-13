# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          config_manager.py
 Purpose:       Standalone configuration management for Investa.
                Handles GUI settings and manual overrides.

 Author:        Antigravity (Refactored from main_gui.py)
-------------------------------------------------------------------------------
"""

import os
import json
import logging
import shutil
import threading
import uuid
from typing import Dict, Any, Optional

import config
from display_config import (
    get_column_definitions,
    DEFAULT_GRAPH_START_DATE,
    DEFAULT_GRAPH_END_DATE,
)

def _default_documents_dir() -> str:
    """Best-effort default directory for the CSV import file picker.

    Prefers the user's ~/Documents folder when it exists, otherwise falls back
    to the current working directory.
    """
    documents = os.path.join(os.path.expanduser("~"), "Documents")
    return documents if os.path.isdir(documents) else os.getcwd()


# One lock per config file path, shared across all ConfigManager instances.
# The server creates a fresh ConfigManager per request, so instance-level
# locks would not serialize anything: concurrent settings saves used to race
# on a shared .tmp file and corrupt the JSON (interleaved writes -> "Extra
# data" on load). Guard both the read-modify-write and the file write itself.
# RLock: the settings route holds the lock across its load-mutate-save cycle,
# and save_gui_config re-acquires it internally.
_PATH_LOCKS: Dict[str, threading.RLock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: str) -> threading.RLock:
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(path)
        if lock is None:
            lock = _PATH_LOCKS[path] = threading.RLock()
        return lock


class ConfigManager:
    def __init__(self, app_data_path: str):
        self.app_data_path = app_data_path
        # Try to locate config files in root first, then config subdir
        config_dir = os.path.join(app_data_path, config.CONFIG_DIR)
        
        # Paths to check for gui_config.json
        root_config = os.path.join(app_data_path, "gui_config.json")
        subdir_config = os.path.join(config_dir, "gui_config.json")
        
        if os.path.exists(root_config):
            self.CONFIG_FILE = root_config
        else:
            os.makedirs(config_dir, exist_ok=True)
            self.CONFIG_FILE = subdir_config

        # Paths to check for manual_overrides.json
        root_overrides = os.path.join(app_data_path, config.MANUAL_OVERRIDES_FILENAME)
        subdir_overrides = os.path.join(config_dir, config.MANUAL_OVERRIDES_FILENAME)

        if os.path.exists(root_overrides):
            self.MANUAL_OVERRIDES_FILE = root_overrides
        else:
            self.MANUAL_OVERRIDES_FILE = subdir_overrides
        
        # Initialize defaults
        self.gui_config = self._get_default_gui_config()
        self.manual_overrides = self._get_default_manual_overrides()
        
        # Load from disk
        self.load_all()

    def _get_default_gui_config(self) -> Dict[str, Any]:
        from db_utils import get_database_path, DB_FILENAME
        # Default the transactions DB to THIS ConfigManager's own scope. Web
        # users get a per-user directory (data/users/<name>/) whose portfolio.db
        # is created at registration, so the default must point there. Using the
        # global get_database_path() here made every new user's config point at
        # the shared data/db/portfolio.db, so they all read the same (empty) DB
        # and saw none of their own transactions. Fall back to the centralized
        # lookup for the legacy single-user GUI, whose DB lives in a 'db'
        # subfolder rather than directly in app_data_path.
        scoped_db_path = os.path.join(self.app_data_path, DB_FILENAME)
        default_db_path = scoped_db_path if os.path.exists(scoped_db_path) else get_database_path(DB_FILENAME)
        default_display_currency = "USD"
        
        return {
            "transactions_file": default_db_path,
            "transactions_file_csv_fallback": config.DEFAULT_CSV,
            "display_currency": default_display_currency,
            "show_closed": False,
            "selected_accounts": [],
            "load_on_startup": True,
            "fmp_api_key": getattr(config, "DEFAULT_API_KEY", ""),
            "account_currency_map": {"SET": "THB"},
            "default_currency": getattr(config, "DEFAULT_CURRENCY", "USD"),
            "graph_start_date": DEFAULT_GRAPH_START_DATE.strftime("%Y-%m-%d"),
            "graph_end_date": DEFAULT_GRAPH_END_DATE.strftime("%Y-%m-%d"),
            "graph_interval": config.DEFAULT_GRAPH_INTERVAL,
            "graph_benchmarks": config.DEFAULT_GRAPH_BENCHMARKS,
            "column_visibility": {col: True for col in get_column_definitions(default_display_currency).keys()},
            "bar_periods_annual": 10,
            "bar_periods_monthly": 12,
            "bar_periods_weekly": 12,
            "dividend_agg_period": "Annual",
            "dividend_periods_to_show": 10,
            "last_csv_import_path": _default_documents_dir(),
            "theme": "light",
            "account_groups": {},
            "account_cash_mode_map": {},
            "account_closure_dates": {},
            "available_currencies": ["USD", "THB", "EUR", "GBP", "JPY", "CNY"],
            "visible_items": [],
            "benchmarks": ['S&P 500', 'Dow Jones', 'NASDAQ'],
            "active_tab": "performance",
        }

    def _get_default_manual_overrides(self) -> Dict[str, Any]:
        return {
            "manual_price_overrides": {},
            "user_symbol_map": config.SYMBOL_MAP_TO_YFINANCE.copy(),
            "user_excluded_symbols": sorted(list(config.YFINANCE_EXCLUDED_SYMBOLS.copy())),
            "account_interest_rates": {},
            "interest_free_thresholds": {},
            "valuation_overrides": {},
            "ibkr_token": None,
            "ibkr_query_id": None
        }

    def load_all(self):
        self.load_gui_config()
        self.load_manual_overrides()

    def load_gui_config(self):
        if os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    loaded = json.load(f)
                self.gui_config.update(loaded)
                self._validate_gui_config()
            except (json.JSONDecodeError, OSError) as e:
                logging.error(f"Error loading GUI config: {e}")
                # Backup corrupt file so it's not lost when we save defaults later
                try:
                    corrupt_path = self.CONFIG_FILE + ".corrupt"
                    shutil.copy2(self.CONFIG_FILE, corrupt_path)
                    logging.warning(f"Corrupted config backed up to {corrupt_path}")
                except Exception as backup_e:
                    logging.error(f"Failed to backup corrupt config: {backup_e}")

    def _validate_gui_config(self):
        # Basic validation (simplified from main_gui.py)
        if not isinstance(self.gui_config.get("selected_accounts"), list):
            self.gui_config["selected_accounts"] = []

        # Validate account_cash_mode_map
        cash_mode_map = self.gui_config.get("account_cash_mode_map")
        if not isinstance(cash_mode_map, dict):
            self.gui_config["account_cash_mode_map"] = {}
        else:
            # Strip invalid entries — only "Manual" and "Auto" are valid
            self.gui_config["account_cash_mode_map"] = {
                k: v for k, v in cash_mode_map.items()
                if isinstance(k, str) and isinstance(v, str) and v in ("Manual", "Auto")
            }

    def settings_lock(self) -> threading.RLock:
        """Lock guarding this user's gui_config read-modify-write cycles.

        Callers that load, mutate, and save the config (e.g. the settings
        route) should hold this across the whole cycle so concurrent updates
        don't drop each other's fields.
        """
        return _lock_for(self.CONFIG_FILE)

    def save_gui_config(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is not None:
            self.gui_config = config_dict

        # Security: Remove API keys before saving
        save_data = self.gui_config.copy()
        save_data.pop("fmp_api_key", None)

        with _lock_for(self.CONFIG_FILE):
            # 1. Backup existing
            if os.path.exists(self.CONFIG_FILE):
                try:
                    shutil.copy2(self.CONFIG_FILE, self.CONFIG_FILE + ".bak")
                except Exception as e:
                    logging.warning(f"Failed to create config backup: {e}")

            # 2. Atomic write. Unique tmp name: the lock serializes writers in
            # this process, but the desktop app may run its own backend against
            # the same files — a shared tmp name would let cross-process writers
            # interleave into one file and corrupt it.
            tmp_file = f"{self.CONFIG_FILE}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_file, "w") as f:
                    json.dump(save_data, f, indent=4)
                    f.flush()
                    os.fsync(f.fileno()) # Ensure write to disk

                os.replace(tmp_file, self.CONFIG_FILE) # Atomic move
            except Exception as e:
                logging.error(f"Error saving GUI config: {e}")
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass

    def load_manual_overrides(self):
        loaded = {}
        if os.path.exists(self.MANUAL_OVERRIDES_FILE):
            try:
                with open(self.MANUAL_OVERRIDES_FILE, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                if isinstance(loaded, dict):
                    # Handle price overrides
                    price_overrides = loaded.get("manual_price_overrides", {})
                    if isinstance(price_overrides, dict):
                        self.manual_overrides["manual_price_overrides"] = {
                            k.upper().strip(): v for k, v in price_overrides.items()
                            if isinstance(v, (dict, int, float))
                        }
                    
                    # Handle symbol map
                    symbol_map = loaded.get("user_symbol_map", {})
                    if isinstance(symbol_map, dict):
                        self.manual_overrides["user_symbol_map"] = {
                            k.upper().strip(): v.upper().strip() for k, v in symbol_map.items()
                        }
                    
                    # Handle excluded symbols
                    excluded = loaded.get("user_excluded_symbols", [])
                    if isinstance(excluded, list):
                        self.manual_overrides["user_excluded_symbols"] = sorted(list({
                            s.upper().strip() for s in excluded if isinstance(s, str)
                        }))
            except Exception as e:
                logging.error(f"Error loading manual overrides: {e}")
                
            # Load Account Interest Rates & Thresholds (ensure they exist even if file didn't have them)
            # This handles migration for existing files
            acc_rates = loaded.get("account_interest_rates", {})
            if isinstance(acc_rates, dict):
                self.manual_overrides["account_interest_rates"] = {
                    k.strip(): float(v) for k, v in acc_rates.items() if isinstance(v, (int, float))
                }
                
            thresholds = loaded.get("interest_free_thresholds", {})
            if isinstance(thresholds, dict):
                self.manual_overrides["interest_free_thresholds"] = {
                    k.strip(): float(v) for k, v in thresholds.items() if isinstance(v, (int, float))
                }
                
            val_overrides = loaded.get("valuation_overrides", {})
            if isinstance(val_overrides, dict):
                self.manual_overrides["valuation_overrides"] = {
                    k.upper().strip(): v for k, v in val_overrides.items() if isinstance(v, dict)
                }

            # Load IBKR Credentials
            self.manual_overrides["ibkr_token"] = loaded.get("ibkr_token")
            self.manual_overrides["ibkr_query_id"] = str(loaded.get("ibkr_query_id")) if loaded.get("ibkr_query_id") else None

    def save_manual_overrides(self, overrides_data: Optional[Dict[str, Any]] = None):
        if overrides_data is not None:
            self.manual_overrides = overrides_data

        with _lock_for(self.MANUAL_OVERRIDES_FILE):
            # 1. Backup existing
            if os.path.exists(self.MANUAL_OVERRIDES_FILE):
                try:
                    shutil.copy2(self.MANUAL_OVERRIDES_FILE, self.MANUAL_OVERRIDES_FILE + ".bak")
                except Exception as e:
                    logging.warning(f"Failed to create manual overrides backup: {e}")

            # 2. Atomic write (unique tmp name — see save_gui_config)
            tmp_file = f"{self.MANUAL_OVERRIDES_FILE}.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
            try:
                with open(tmp_file, "w", encoding="utf-8") as f:
                    json.dump(self.manual_overrides, f, indent=4, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())

                os.replace(tmp_file, self.MANUAL_OVERRIDES_FILE)
                return True
            except Exception as e:
                logging.error(f"Error saving manual overrides: {e}")
                if os.path.exists(tmp_file):
                    try:
                        os.remove(tmp_file)
                    except OSError:
                        pass
                return False
