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
import pandas as pd
import numpy as np
from datetime import date
from typing import Dict, Any, Optional, List, Set
from PySide6.QtCore import QStandardPaths

import config
from utils import get_column_definitions, DEFAULT_GRAPH_START_DATE, DEFAULT_GRAPH_END_DATE

class ConfigManager:
    def __init__(self, app_data_path: str):
        self.app_data_path = app_data_path
        self.CONFIG_FILE = os.path.join(app_data_path, "gui_config.json")
        self.MANUAL_OVERRIDES_FILE = os.path.join(app_data_path, config.MANUAL_OVERRIDES_FILENAME)
        
        # Initialize defaults
        self.gui_config = self._get_default_gui_config()
        self.manual_overrides = self._get_default_manual_overrides()

    def _get_default_gui_config(self) -> Dict[str, Any]:
        from db_utils import get_database_path, DB_FILENAME
        default_db_path = get_database_path(DB_FILENAME)
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
            "last_csv_import_path": QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or os.getcwd(),
            "theme": "light",
            "account_groups": {},
        }

    def _get_default_manual_overrides(self) -> Dict[str, Any]:
        return {
            "manual_price_overrides": {},
            "user_symbol_map": config.SYMBOL_MAP_TO_YFINANCE.copy(),
            "user_excluded_symbols": sorted(list(config.YFINANCE_EXCLUDED_SYMBOLS.copy()))
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
            except Exception as e:
                logging.error(f"Error loading GUI config: {e}")

    def _validate_gui_config(self):
        # Basic validation (simplified from main_gui.py)
        if not isinstance(self.gui_config.get("selected_accounts"), list):
            self.gui_config["selected_accounts"] = []

    def save_gui_config(self, config_dict: Optional[Dict[str, Any]] = None):
        if config_dict is not None:
            self.gui_config = config_dict
        
        # Security: Remove API keys before saving
        save_data = self.gui_config.copy()
        save_data.pop("fmp_api_key", None)
        
        try:
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(save_data, f, indent=4)
        except Exception as e:
            logging.error(f"Error saving GUI config: {e}")

    def load_manual_overrides(self):
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
                            if isinstance(v, dict)
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

    def save_manual_overrides(self, overrides_data: Optional[Dict[str, Any]] = None):
        if overrides_data is not None:
            self.manual_overrides = overrides_data
            
        try:
            with open(self.MANUAL_OVERRIDES_FILE, "w", encoding="utf-8") as f:
                json.dump(self.manual_overrides, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            logging.error(f"Error saving manual overrides: {e}")
            return False
