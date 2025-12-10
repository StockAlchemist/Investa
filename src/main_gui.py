# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          main_gui.py
 Purpose:       Main application window and GUI logic for Investa Portfolio Dashboard.
                Handles UI elements, user interaction, background tasks, and visualization.

 Author:        Kit Matan and Google Gemini 2.5
 Author Email:  kittiwit@gmail.com

 Created:       26/04/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

# --- START OF FILE main_gui.py ---

# --- Module Docstring ---
"""
Investa Portfolio Dashboard GUI Application.

Provides a graphical user interface built with PySide6 to visualize and analyze
investment portfolio data based on transaction history loaded from a CSV file.
Features include:
- Displaying current holdings with market values, gains/losses, and performance metrics.
- Visualizing portfolio allocation through pie charts (by account and holding).
- Plotting historical portfolio performance (value and TWR) against benchmarks.
- Fetching near real-time stock/index quotes and FX rates using yfinance.
- Caching fetched data to reduce API calls.
- Filtering portfolio view by account.
- Adding new transactions manually.
- Customizable column visibility in the holdings table.
- Saving and loading UI configuration.
"""

import sys
import os
import pandas as pd
import numpy as np
import json
import traceback
import csv
import re
import warnings
import sqlite3  # Added import for sqlite3
import shutil
from io import StringIO, BytesIO  # <-- ADDED: For in-memory log stream and image handling
import logging
import base64

# Ensure this is defined before any use
HISTORICAL_FN_SUPPORTS_EXCLUDE = False

# --- Add project root to sys.path ---
# Ensures modules like config, data_loader etc. can be found reliably
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# --- End Path Addition ---

from datetime import datetime, date, timedelta, time

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from zoneinfo import ZoneInfo

from typing import Dict, Any, Optional, List, Tuple, Set  # Added Set
import logging
from collections import defaultdict

# --- Configure Logging Globally (as early as possible) ---
# The LOGGING_LEVEL is imported from config.py and used here.
# Default in config.py is logging.INFO, but can be changed there.
import config  # <-- MOVED IMPORT HERE

logging.basicConfig(
    level=config.LOGGING_LEVEL,  # Use LOGGING_LEVEL from config module
    format="%(asctime)s [%(levelname)-8s] %(name)-15s %(module)s:%(lineno)d - %(message)s",  # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
    # Use force=True (Python 3.8+) to ensure this config takes precedence
    # Remove if using older Python or facing issues with existing handlers
    force=True,
)

# Quieten overly verbose libraries (optional, but often useful)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)  # Suppress categorical units warnings
logging.getLogger("yfinance").setLevel(
    logging.WARNING
)  # yfinance can be noisy on DEBUG

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QTableView,
    QHeaderView,
    QStatusBar,
    QFileDialog,
    QFrame,
    QGridLayout,
    QMessageBox,
    QPlainTextEdit,
    QSizePolicy,
    QListWidget,
    QStyle,
    QDateEdit,
    QLineEdit,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QTableWidget,
    QGroupBox,
    QTextEdit,
    QSplitter,
    QTabWidget,
    QTableWidgetItem,
    QAbstractItemView,
    QToolBar,
    QScrollArea,
    QButtonGroup,
    QSpinBox,
    QCompleter,
    QListWidgetItem,
    QInputDialog,
    QSpacerItem,
    QProgressBar,
)
from functools import partial
import contextlib

from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QModelIndex,
    QThreadPool,
    QRunnable,
    Signal,
    Slot,
    QObject,
    QDateTime,
    QDate,
    QPoint,
    QStringListModel,
    QStandardPaths,
    QByteArray,
    QTimer,
    QItemSelection,
    QItemSelectionModel,
    QSize,
    QSizeF,
)
from PySide6.QtGui import (
    QDoubleValidator,
    QColor,
    QPalette,
    QFont,
    QIcon,
    QPixmap,
    QAction,
    QActionGroup,
    QValidator,
    QTextDocument,
    QPageSize,
    QPageLayout,
)
from PySide6.QtPrintSupport import QPrinter

# --- Matplotlib Lazy Loading Setup ---
plt = None
Figure = None
FigureCanvas = None
NavigationToolbar = None
mdates = None
mtick = None
mplcursors = None
MPLCURSORS_AVAILABLE = False
_MATPLOTLIB_INITIALIZED = False

def _ensure_matplotlib():
    global plt, Figure, FigureCanvas, NavigationToolbar, mdates, mtick, mplcursors, MPLCURSORS_AVAILABLE, _MATPLOTLIB_INITIALIZED
    
    if _MATPLOTLIB_INITIALIZED:
        return

    try:
        import matplotlib
        matplotlib.use("QtAgg")
        import matplotlib.pyplot as p
        plt = p
        from matplotlib.figure import Figure as F
        Figure = F
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NT,
            FigureCanvasQTAgg as FC,
        )
        NavigationToolbar = NT
        FigureCanvas = FC
        import matplotlib.dates as md
        mdates = md
        import matplotlib.ticker as mt
        mtick = mt
        
        # --- Matplotlib Font Configuration ---
        # --- TRY THESE FONTS ---
        # font_name = "Thonburi"       # Good choice for macOS Thai support
        font_name = "DejaVu Sans"  # Excellent cross-platform choice if installed
        # font_name = "Helvetica Neue"  # Try if available
        # font_name = "Lucida Grande"  # Try if available
        # font_name = "Arial Unicode MS" # Best coverage if installed
        # font_name = "Arial"          # Original (causes warning)

        # Try setting the default font
        plt.rcParams.update({"font.family": font_name})
        # Update other rcParams as before
        plt.rcParams.update(
            {
                "font.size": 8,
                "axes.labelcolor": "#333333",
                "xtick.color": "#666666",
                "ytick.color": "#666666",
                "text.color": "#333333",
            }
        )
        logging.debug(
            f"Matplotlib default font configured to: {plt.rcParams['font.family']}"
        )
        
    except ImportError as e:
        logging.critical(f"CRITICAL: Failed to import matplotlib: {e}")
    except Exception as e:
         logging.warning(f"Warning: Could not configure Matplotlib font: {e}")

    try:
        import mplcursors as mpc
        mplcursors = mpc
        MPLCURSORS_AVAILABLE = True
    except ImportError:
        logging.warning(
            "Warning: mplcursors library not found. Hover tooltips on graphs will be disabled."
        )
        MPLCURSORS_AVAILABLE = False
        
    _MATPLOTLIB_INITIALIZED = True


from config import (
    DEBOUNCE_INTERVAL_MS,
    MANUAL_OVERRIDES_FILENAME,
    DEFAULT_API_KEY,
    CHART_MAX_SLICES,
    PIE_CHART_FIG_SIZE,
    PERF_CHART_FIG_SIZE,
    CHART_DPI,
    INDICES_FOR_HEADER,
    CSV_DATE_FORMAT,
    COMMON_CURRENCIES,
    DEFAULT_GRAPH_DAYS_AGO,
    DEFAULT_GRAPH_INTERVAL,
    DEFAULT_GRAPH_BENCHMARKS,
    BENCHMARK_MAPPING,
    BENCHMARK_OPTIONS_DISPLAY,
    EXCHANGE_TRADING_HOURS,
    COLOR_BG_DARK,
    COLOR_BG_HEADER_LIGHT,
    COLOR_BG_HEADER_ORIGINAL,
    COLOR_TEXT_DARK,
    COLOR_TEXT_SECONDARY,
    COLOR_ACCENT_TEAL,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_DARK,
    COLOR_GAIN,
    COLOR_LOSS,
    CURRENCY_SYMBOLS,
    DEFAULT_CSV,
    SHORTABLE_SYMBOLS,
    BAR_CHART_MAX_PERIODS_ANNUAL,
    BAR_CHART_MAX_PERIODS_QUARTERLY,
    BAR_CHART_MAX_PERIODS_MONTHLY,
    BAR_CHART_MAX_PERIODS_WEEKLY,
    BAR_CHART_MAX_PERIODS_DAILY,
)

from utils import (
    resource_path,
    get_column_definitions,
    DEFAULT_GRAPH_START_DATE,
    DEFAULT_GRAPH_END_DATE,
    QCOLOR_GAIN,
    QCOLOR_LOSS,
    GroupHeaderDelegate,
)

from dialogs import (
    FundamentalDataDialog,
    LogViewerDialog,
    AccountCurrencyDialog,
    SymbolChartDialog,
    ManualPriceDialog,
    AddTransactionDialog,
)
from workers import (
    WorkerSignals,
    PortfolioCalculatorWorker,
    FundamentalDataWorker,
)
from models import PandasModel
from io_handlers import write_dataframe_to_csv, write_dataframe_to_excel

# --- Core Logic Imports ---
from portfolio_logic import (
    calculate_portfolio_summary,
    CASH_SYMBOL_CSV,
    calculate_historical_performance,
)
from risk_metrics import calculate_all_risk_metrics, calculate_drawdown_series, calculate_max_drawdown, calculate_volatility, calculate_sharpe_ratio, calculate_sortino_ratio
from factor_analyzer import run_factor_regression
from market_data import MarketDataProvider
from finutils import (
    map_to_yf_symbol,
    format_currency_value,
    format_percentage_value,
    is_cash_symbol,
    format_large_number_display,
    format_integer_with_commas,
    format_float_with_commas,
    calculate_irr,
    get_historical_rate_via_usd_bridge,
)

from data_loader import load_and_clean_transactions
from data_loader import COLUMN_MAPPING_CSV_TO_INTERNAL
from csv_utils import DESIRED_CLEANED_COLUMN_ORDER

from portfolio_analyzer import (
    calculate_periodic_returns,
    _AGGREGATE_CASH_ACCOUNT_NAME_,
    extract_dividend_history,
    extract_realized_capital_gains_history,
    calculate_rebalancing_trades,
)

LOGIC_AVAILABLE = True
MARKET_PROVIDER_AVAILABLE = True

# Check if the imported function signature actually supports 'exclude_accounts'
import inspect

sig = inspect.signature(calculate_historical_performance)
HISTORICAL_FN_SUPPORTS_EXCLUDE = "exclude_accounts" in sig.parameters
if not HISTORICAL_FN_SUPPORTS_EXCLUDE:
    logging.warning(
        "Warn: Imported 'calculate_historical_performance' does NOT support 'exclude_accounts' argument. Exclusion logic in GUI will be ignored."
    )

# --- ADDED: db_utils import ---
from db_utils import (
    DB_FILENAME,
    get_database_path,
    initialize_database,
    add_transaction_to_db,
    update_transaction_in_db,
    delete_transaction_from_db,
    migrate_csv_to_db,
    check_if_db_empty_and_csv_exists,
    load_all_transactions_from_db,
)

# --- ADDED: Import financial_ratios and set availability flag ---
try:
    from financial_ratios import (
        calculate_key_ratios_timeseries,
        calculate_current_valuation_ratios,
    )

    FINANCIAL_RATIOS_AVAILABLE = True
except ImportError:
    logging.error(
        "ERROR: financial_ratios.py not found. Financial ratio calculations will be disabled."
    )
    FINANCIAL_RATIOS_AVAILABLE = False

    def calculate_key_ratios_timeseries(*args, **kwargs):
        return pd.DataFrame()

    def calculate_current_valuation_ratios(*args, **kwargs):
        return {}


# --- ADDED: Class attribute for frozen columns ---
FROZEN_COLUMNS_UI = ["Account", "Symbol"]


from ui_helpers import UiHelpersMixin


class PortfolioApp(QMainWindow, UiHelpersMixin):
    """Main application window for the Investa Portfolio Dashboard."""

    # Define icon specifications for toolbar actions for light and dark themes
    # Format: action_attribute_name: {"light": (type, spec), "dark": (type, spec, [fallback_sp_for_dark])}
    # type can be "theme" (for QIcon.fromTheme) or "sp" (for QStyle.standardIcon)
    TOOLBAR_ICON_SPECS = {
        "select_db_action": {
            "light": ("theme", "document-open"),
            "dark": ("sp", QStyle.SP_DialogOpenButton),
        },
        "new_database_file_action": {
            "light": ("theme", "document-new"),
            "dark": ("sp", QStyle.SP_FileIcon),  # Changed from SP_FileDialogNewButton
        },
        "import_csv_action": {  # Original uses SP_DialogOpenButton
            "light": (
                "sp",
                QStyle.SP_DialogOpenButton,
                QStyle.SP_ArrowDown,
            ),  # type, primary, fallback
            "dark": (
                "sp",
                QStyle.SP_DialogOpenButton,
                QStyle.SP_ArrowDown,
            ),  # Assuming SP_DialogOpenButton is fine for dark
        },
        # "export_csv_action": {
        #     "light": (
        #         "theme",
        #         "document-export",
        #         QStyle.SP_DialogSaveButton,
        #     ),  # Added SP_DialogSaveButton as fallback
        #     "dark": ("sp", QStyle.SP_DialogSaveButton),  # Dark theme already uses SP
        # },
        "export_excel_action": {
            "light": ("theme", "document-save-as", QStyle.SP_DialogSaveButton),
            "dark": ("sp", QStyle.SP_DialogSaveButton),
        },
        "refresh_action": {
            "light": ("theme", "view-refresh"),
            "dark": ("sp", QStyle.SP_BrowserReload),
        },
        "add_transaction_action": {
            "light": ("theme", "list-add"),
            "dark": ("sp", QStyle.SP_FileIcon),  # Changed from SP_FileDialogNewButton
        },
    }

    # For the QPushButton used for fundamental lookup (not a QAction on toolbar directly from menu)
    LOOKUP_BUTTON_ICON_SPECS = {
        "light": ("theme", "help-about"),
        "dark": ("sp", QStyle.SP_MessageBoxInformation),
    }

    # Menu helper implementations moved to UiHelpersMixin (ui_helpers.py)

    # Centralized status and message helpers are provided by UiHelpersMixin

    # --- Helper Methods (Define BEFORE they are called in __init__) ---
    def _ensure_all_columns_in_visibility(self):
        """
        Ensures self.column_visibility covers all possible UI columns.

        Updates the `self.column_visibility` dictionary to include entries for all
        columns defined by `get_column_definitions` based on the current display
        currency, preserving existing visibility states where possible.
        """
        current_currency = "USD"
        if hasattr(self, "currency_combo") and self.currency_combo:
            current_currency = self.currency_combo.currentText()
        self.all_possible_ui_columns = list(
            get_column_definitions(current_currency).keys()
        )
        current_visibility = self.column_visibility.copy()
        self.column_visibility = {}
        for col_name in self.all_possible_ui_columns:
            is_visible = current_visibility.get(col_name, True)
            if not isinstance(is_visible, bool):
                is_visible = True
            self.column_visibility[col_name] = is_visible

    def _get_currency_symbol(
        self, get_name=False, currency_code=None
    ):  # <-- Add currency_code=None
        """
        Gets the currency symbol (e.g., "$") or 3-letter name (e.g., "USD").

        If currency_code is provided, it returns the symbol/name for that code.
        Otherwise, it uses the currently selected display currency from the UI or config.

        Args:
            get_name (bool, optional): If True, returns the 3-letter currency code.
                                       Defaults to False.
            currency_code (str | None, optional): If provided, get the symbol/name for
                                                  this specific code. Defaults to None.

        Returns:
            str: The currency symbol or name.
        """
        target_currency_code = None
        if currency_code and isinstance(currency_code, str):
            target_currency_code = currency_code.upper()
        else:
            # Fallback to display currency if no code provided
            if (
                hasattr(self, "currency_combo")
                and self.currency_combo
                and self.currency_combo.count() > 0
            ):
                target_currency_code = self.currency_combo.currentText()
            elif hasattr(self, "config"):
                target_currency_code = self.config.get("display_currency", "USD")
            else:
                target_currency_code = "USD"  # Final fallback

        if get_name:
            return target_currency_code  # Return the 3-letter code

        return CURRENCY_SYMBOLS.get(
            target_currency_code, target_currency_code
        )  # Return code itself if no symbol mapped

    def _calculate_annualized_twr(self, total_twr_factor, start_date, end_date):
        """
        Calculates the annualized Time-Weighted Return (TWR) percentage.

        Args:
            total_twr_factor (float | np.nan): The total TWR factor (1 + total TWR)
                                               over the period.
            start_date (date | None): The start date of the period.
            end_date (date | None): The end date of the period.

        Returns:
            float | np.nan: The annualized TWR as a percentage, or np.nan if inputs
                            are invalid or calculation fails.
        """
        if pd.isna(total_twr_factor) or total_twr_factor <= 0:
            return np.nan
        if start_date is None or end_date is None or start_date >= end_date:
            return np.nan
        try:
            num_days = (end_date - start_date).days
            if num_days <= 0:
                return np.nan
            annualized_twr_factor = total_twr_factor ** (365.25 / num_days)
            return (annualized_twr_factor - 1) * 100.0
        except (TypeError, ValueError, OverflowError):
            return np.nan

    def _safe_calculate_irr(self, cashflows):
        """Calculates IRR with overflow protection."""
        with warnings.catch_warnings():
            # Ignore the specific RuntimeWarning that occurs during IRR calculation
            warnings.filterwarnings(
                "ignore", message="overflow encountered in scalar divide"
            )
            try:
                return calculate_irr(cashflows)
            except (OverflowError, ValueError):
                return np.nan  # Or some other appropriate value

    def load_config(self):
        """
        Loads application configuration from a JSON file (gui_config.json).

        Loads settings like the database file path, display currency, selected
        accounts, graph parameters, and column visibility. Uses default values if
        the file doesn't exist or if specific settings are missing or invalid.
        Performs validation on loaded values. `self.DB_FILE_PATH` is updated here.

        Returns:
            dict: The loaded (and potentially validated/defaulted) configuration dictionary.
        """
        # Default DB path determined by db_utils.get_database_path()
        default_db_path = get_database_path(DB_FILENAME)
        default_csv_fallback_path = DEFAULT_CSV  # For migration prompt if DB is empty

        default_display_currency = "USD"
        default_column_visibility = {
            col: True for col in get_column_definitions(default_display_currency).keys()
        }
        default_account_currency_map = {"SET": "THB"}  # Example default

        config_defaults = {
            "transactions_file": default_db_path,  # Path to .db file
            "transactions_file_csv_fallback": default_csv_fallback_path,  # For migration check
            "display_currency": default_display_currency,
            "show_closed": False,
            "selected_accounts": [],
            "load_on_startup": True,
            "fmp_api_key": (
                config.DEFAULT_API_KEY if hasattr(config, "DEFAULT_API_KEY") else ""
            ),  # Use from config if available
            "account_currency_map": default_account_currency_map,
            "default_currency": (
                config.DEFAULT_CURRENCY
                if hasattr(config, "DEFAULT_CURRENCY")
                else "USD"
            ),
            "graph_start_date": DEFAULT_GRAPH_START_DATE.strftime("%Y-%m-%d"),
            "graph_end_date": DEFAULT_GRAPH_END_DATE.strftime("%Y-%m-%d"),
            "graph_interval": DEFAULT_GRAPH_INTERVAL,
            "graph_benchmarks": DEFAULT_GRAPH_BENCHMARKS,
            "column_visibility": default_column_visibility,
            "bar_periods_annual": 10,
            "bar_periods_monthly": 12,
            "bar_periods_weekly": 12,
            "dividend_agg_period": "Annual",
            "dividend_periods_to_show": 10,
            "dividend_chart_default_periods_annual": (
                config.DIVIDEND_CHART_DEFAULT_PERIODS_ANNUAL
                if hasattr(config, "DIVIDEND_CHART_DEFAULT_PERIODS_ANNUAL")
                else 10
            ),
            "dividend_chart_default_periods_quarterly": (
                config.DIVIDEND_CHART_DEFAULT_PERIODS_QUARTERLY
                if hasattr(config, "DIVIDEND_CHART_DEFAULT_PERIODS_QUARTERLY")
                else 12
            ),
            "dividend_chart_default_periods_monthly": (
                config.DIVIDEND_CHART_DEFAULT_PERIODS_MONTHLY
                if hasattr(config, "DIVIDEND_CHART_DEFAULT_PERIODS_MONTHLY")
                else 24
            ),
            # Defaults for Asset Change spinboxes
            "pvc_annual_periods": 10,
            "pvc_monthly_periods": 12,
            "pvc_weekly_periods": 12,
            "pvc_daily_periods": 30,  # For new daily chart
            "last_csv_import_path": QStandardPaths.writableLocation(
                QStandardPaths.DocumentsLocation
            )
            or os.getcwd(),  # For import dialog
            "last_csv_export_path": QStandardPaths.writableLocation(
                QStandardPaths.DocumentsLocation
            )
            or os.getcwd(),  # For export dialog
            "last_excel_export_path": QStandardPaths.writableLocation(
                QStandardPaths.DocumentsLocation
            )
            or os.getcwd(),  # For export dialog
            "last_image_export_path": QStandardPaths.writableLocation(
                QStandardPaths.DocumentsLocation
            )
            or os.getcwd(),  # For export dialog
            "user_currencies": COMMON_CURRENCIES.copy(),  # Default list of user-selectable currencies
            "cg_agg_period": "Annual",  # Default for Capital Gains aggregation
            "cg_periods_to_show": 10,  # Default periods for Capital Gains chart
            "theme": "light",  # Added theme configuration
            "transactions_management_columns": {},  # For column visibility in the main transactions table
            "stock_tx_columns": {},  # For column visibility in the stock transactions table
            "cash_tx_columns": {},  # For column visibility in the cash transactions table
            "holdings_table_header_state": None,  # For column order and sizes
            "account_groups": {},  # For grouping accounts in the menu
        }
        loaded_app_config = config_defaults.copy()  # Start with defaults

        if self.CONFIG_FILE and os.path.exists(self.CONFIG_FILE):
            try:
                with open(self.CONFIG_FILE, "r") as f:
                    loaded_config_from_file = json.load(f)
                loaded_app_config.update(
                    loaded_config_from_file
                )  # Update defaults with loaded values
                # logging.info(f"Config loaded from {self.CONFIG_FILE}")

                # Specific validation for transactions_file (DB path)
                # If the loaded 'transactions_file' is a CSV, it means it's an old config.
                # We should use it as the CSV fallback and set the transactions_file to the default DB path.
                if "transactions_file" in loaded_app_config:
                    path_from_config = loaded_app_config["transactions_file"]
                    if isinstance(
                        path_from_config, str
                    ) and path_from_config.lower().endswith(".csv"):
                        logging.warning(
                            f"Old config detected: 'transactions_file' points to a CSV ('{path_from_config}'). "
                            f"This will be used as a potential migration source. Main data source is now a database."
                        )
                        loaded_app_config["transactions_file_csv_fallback"] = (
                            path_from_config
                        )
                        loaded_app_config["transactions_file"] = (
                            default_db_path  # Use default DB path
                        )
                    elif not isinstance(
                        path_from_config, str
                    ) or not path_from_config.lower().endswith(
                        (".db", ".sqlite", ".sqlite3")
                    ):
                        logging.warning(
                            f"Invalid 'transactions_file' (DB path) in config: '{path_from_config}'. Using default DB path."
                        )
                        loaded_app_config["transactions_file"] = default_db_path
                else:  # Not in config, use default
                    loaded_app_config["transactions_file"] = default_db_path

                # --- Validate other config keys (largely as before) ---
                if "graph_benchmarks" in loaded_app_config:
                    if isinstance(loaded_app_config["graph_benchmarks"], list):
                        valid_benchmarks = [
                            b
                            for b in loaded_app_config["graph_benchmarks"]
                            if isinstance(b, str) and b in BENCHMARK_OPTIONS_DISPLAY
                        ]
                        loaded_app_config["graph_benchmarks"] = valid_benchmarks
                    else:
                        loaded_app_config["graph_benchmarks"] = DEFAULT_GRAPH_BENCHMARKS

                if "selected_accounts" in loaded_app_config and not isinstance(
                    loaded_app_config["selected_accounts"], list
                ):
                    loaded_app_config["selected_accounts"] = []

                if "column_visibility" in loaded_app_config and isinstance(
                    loaded_app_config["column_visibility"], dict
                ):
                    validated_visibility = {}
                    all_cols = get_column_definitions(
                        loaded_app_config.get(
                            "display_currency", default_display_currency
                        )
                    ).keys()
                    for col in all_cols:
                        val = loaded_app_config["column_visibility"].get(col)
                        validated_visibility[col] = (
                            val if isinstance(val, bool) else True
                        )
                    loaded_app_config["column_visibility"] = validated_visibility
                else:
                    loaded_app_config["column_visibility"] = default_column_visibility

                if "account_currency_map" in loaded_app_config:
                    if not isinstance(
                        loaded_app_config["account_currency_map"], dict
                    ) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in loaded_app_config["account_currency_map"].items()
                    ):
                        loaded_app_config["account_currency_map"] = (
                            default_account_currency_map
                        )
                else:  # Not in config, use default
                    loaded_app_config["account_currency_map"] = (
                        default_account_currency_map
                    )

                if "default_currency" in loaded_app_config:
                    if (
                        not isinstance(loaded_app_config["default_currency"], str)
                        or len(loaded_app_config["default_currency"]) != 3
                    ):
                        loaded_app_config["default_currency"] = config_defaults[
                            "default_currency"
                        ]
                else:  # Not in config, use default
                    loaded_app_config["default_currency"] = config_defaults[
                        "default_currency"
                    ]

                # Validate new column visibility settings
                for key in [
                    "transactions_management_columns",
                    "stock_tx_columns",
                    "cash_tx_columns",
                ]:
                    if key in loaded_app_config:
                        if not isinstance(loaded_app_config[key], dict):
                            logging.warning(
                                f"Invalid '{key}' in config. Resetting to default empty dict."
                            )
                            loaded_app_config[key] = {}
                    else:
                        loaded_app_config[key] = (
                            {}
                        )  # Ensure key exists with default empty dict

                # Validate account_groups
                if "account_groups" in loaded_app_config:
                    if not isinstance(loaded_app_config["account_groups"], dict):
                        logging.warning(
                            "Invalid 'account_groups' in config. Resetting to default empty dict."
                        )
                        loaded_app_config["account_groups"] = {}

            except Exception as e:
                logging.warning(
                    f"Warn: Load config failed ({self.CONFIG_FILE}): {e}. Using defaults."
                )
                loaded_app_config = (
                    config_defaults.copy()
                )  # Revert to full defaults on error
                loaded_app_config["transactions_file"] = (
                    default_db_path  # Ensure DB path is default on error
                )
        else:
            # logging.info(f"Config file {self.CONFIG_FILE} not found. Using defaults.")
            loaded_app_config["transactions_file"] = (
                default_db_path  # Ensure DB path is default if no config file
            )

        # Ensure all default keys exist in the final config, and type check
        for key, default_value in config_defaults.items():
            if key not in loaded_app_config:
                loaded_app_config[key] = default_value
            elif not isinstance(loaded_app_config[key], type(default_value)):
                # Allow for specific keys that might have None or different types legitimately
                allowed_flexible_types = [
                    "fmp_api_key",
                    "selected_accounts",
                    "account_currency_map",
                    "graph_benchmarks",
                    "transactions_file_csv_fallback",
                    "last_csv_import_path",
                    "last_csv_export_path",
                    "last_excel_export_path",
                    "last_image_export_path",
                    "holdings_table_header_state",  # Allow str even if default is None
                ]
                if key not in allowed_flexible_types or (
                    loaded_app_config[key] is not None
                    and not isinstance(
                        loaded_app_config[key], (str, list, dict, bool, int, float)
                    )
                ):
                    logging.warning(
                        f"Warn: Config type mismatch for '{key}'. Loaded type: {type(loaded_app_config[key])}, Default type: {type(default_value)}. Using default."
                    )
                    loaded_app_config[key] = default_value

        # Final date format validation
        try:
            QDate.fromString(loaded_app_config["graph_start_date"], "yyyy-MM-dd")
        except:
            loaded_app_config["graph_start_date"] = DEFAULT_GRAPH_START_DATE.strftime(
                "%Y-%m-%d"
            )
        try:
            QDate.fromString(loaded_app_config["graph_end_date"], "yyyy-MM-dd")
        except:
            loaded_app_config["graph_end_date"] = DEFAULT_GRAPH_END_DATE.strftime(
                "%Y-%m-%d"
            )

        # Validate spinbox periods
        numeric_spinbox_keys = [
            "bar_periods_annual",
            "bar_periods_monthly",
            "bar_periods_weekly",
            "dividend_periods_to_show",
            "dividend_chart_default_periods_annual",
            "dividend_chart_default_periods_quarterly",
            "dividend_chart_default_periods_monthly",
        ]
        # Add PVC spinbox keys
        numeric_spinbox_keys.extend(
            [
                "pvc_annual_periods",
                "pvc_monthly_periods",
                "pvc_weekly_periods",
                "pvc_daily_periods",
            ]
        )
        for key in numeric_spinbox_keys:
            if key in loaded_app_config:
                try:
                    val = int(loaded_app_config[key])
                    loaded_app_config[key] = max(1, min(val, 100))
                except (ValueError, TypeError):
                    loaded_app_config[key] = config_defaults[key]
            else:
                loaded_app_config[key] = config_defaults[key]  # Ensure key exists

        # Validate Capital Gains periods
        cg_numeric_spinbox_keys = ["cg_periods_to_show"]
        for key in cg_numeric_spinbox_keys:
            if key in loaded_app_config:
                try:
                    val = int(loaded_app_config[key])
                    loaded_app_config[key] = max(1, min(val, 100))  # Max 100 periods
                except (ValueError, TypeError):
                    loaded_app_config[key] = config_defaults[key]
            else:
                loaded_app_config[key] = config_defaults[key]

        if loaded_app_config.get("cg_agg_period") not in ["Annual", "Quarterly"]:
            loaded_app_config["cg_agg_period"] = config_defaults["cg_agg_period"]

        # Validate rebalancing targets
        if "rebalancing_targets" in loaded_app_config:
            if not isinstance(loaded_app_config["rebalancing_targets"], dict):
                logging.warning("Invalid 'rebalancing_targets' in config. Resetting.")
                loaded_app_config["rebalancing_targets"] = {}
        else:
            loaded_app_config["rebalancing_targets"] = {}

        if loaded_app_config.get("dividend_agg_period") not in [
            "Annual",
            "Quarterly",
            "Monthly",
        ]:
            loaded_app_config["dividend_agg_period"] = config_defaults[
                "dividend_agg_period"
            ]

        # CRITICAL: Set the instance's DB_FILE_PATH based on the final loaded config
        # Validate user_currencies
        if not isinstance(loaded_app_config.get("user_currencies"), list) or not all(
            isinstance(c, str) and len(c) == 3 and c.isupper()
            for c in loaded_app_config.get("user_currencies", [])
        ):
            logging.warning(
                "Invalid 'user_currencies' in config. Resetting to default."
            )
            loaded_app_config["user_currencies"] = config_defaults["user_currencies"]
        elif not loaded_app_config.get("user_currencies"):  # Ensure not empty
            loaded_app_config["user_currencies"] = config_defaults["user_currencies"]

        self.DB_FILE_PATH = loaded_app_config.get("transactions_file", default_db_path)
        # If somehow it's still not a DB path, force default (should be caught earlier)
        if not isinstance(
            self.DB_FILE_PATH, str
        ) or not self.DB_FILE_PATH.lower().endswith((".db", ".sqlite", ".sqlite3")):
            logging.error(
                f"Post-config load, DB_FILE_PATH is invalid: '{self.DB_FILE_PATH}'. Resetting to default."
            )
            self.DB_FILE_PATH = default_db_path
            loaded_app_config["transactions_file"] = self.DB_FILE_PATH

        return loaded_app_config

    # --- UI Update Methods (Define BEFORE __init__ calls them) ---

    def update_header_info(
        self, loading=False
    ):  # Keep loading param for now, though logic relies on self.index_quote_data
        """Updates the header label with index quotes or a loading message.

        Formats and displays data for indices specified in `INDICES_FOR_HEADER`
        using data stored in `self.index_quote_data`. Shows price, change, and
        percentage change with appropriate coloring for gains/losses.

        Args:
            loading (bool, optional): If True, displays a "Loading..." message
                                      instead of attempting to show quotes.
                                      Defaults to False.
        """
        if not hasattr(self, "header_info_label") or not self.header_info_label:
            return  # Label not ready

        if loading or not self.index_quote_data:
            self.header_info_label.setText("<i>Loading indices...</i>")
            return

        header_parts = []
        # logging.debug(f"DEBUG update_header_info: Data = {self.index_quote_data}") # Optional debug print
        for index_symbol in INDICES_FOR_HEADER:
            data = self.index_quote_data.get(index_symbol)
            if data and isinstance(data, dict):
                price = data.get("price")
                change = data.get("change")
                # --- FIX: Use 'changesPercentage' which should be decimal/fraction ---
                change_pct_decimal = data.get("changesPercentage")
                name = data.get("name", index_symbol).split(" ")[
                    0
                ]  # Use short name part

                price_str = f"{price:,.2f}" if pd.notna(price) else "N/A"
                change_str = "N/A"
                change_color = COLOR_TEXT_DARK  # Default color

                if pd.notna(change) and pd.notna(change_pct_decimal):
                    change_val = float(change)
                    change_pct_val = float(
                        change_pct_decimal
                    )  # already in decimal form %
                    sign = (
                        "+" if change_val >= -1e-9 else ""
                    )  # Add '+' for positive/zero change
                    change_str = (
                        f"{sign}{change_val:,.2f} ({sign}{change_pct_val:.2f}%)"
                    )
                    if change_val > 1e-9:
                        change_color = COLOR_GAIN
                    elif change_val < -1e-9:
                        change_color = COLOR_LOSS
                    # else: keep default color for zero change
                # --- END FIX ---

                # Use HTML for coloring
                header_parts.append(
                    f"<b>{name}:</b> {price_str} <font color='{change_color}'>{change_str}</font>"
                )
            else:
                # Handle case where index data is missing
                header_parts.append(
                    f"<b>{index_symbol.split('.')[0] if '.' in index_symbol else index_symbol}:</b> N/A"
                )

        if header_parts:
            self.header_info_label.setText(" | ".join(header_parts))
            # logging.debug(f"DEBUG update_header_info: Set text to: {' | '.join(header_parts)}") # Optional debug print
        else:
            self.header_info_label.setText("<i>Index data unavailable.</i>")

    def _init_menu_bar(self):
        """Creates the main menu bar."""
        menu_bar = self.menuBar()
        menu_bar.setObjectName("MenuBar")  # Apply object name for styling

        # --- File Menu ---
        file_menu = menu_bar.addMenu("&File")

        # Open Database File
        self.select_db_action = QAction(
            QIcon.fromTheme("document-open"), "Open &Database File...", self
        )
        self.select_db_action.setStatusTip("Select the SQLite database file to load")
        self.select_db_action.triggered.connect(self.select_database_file)
        file_menu.addAction(self.select_db_action)

        # New Database File
        self.new_database_file_action = QAction(
            QIcon.fromTheme("document-new"), "&New Database File...", self
        )
        self.new_database_file_action.setStatusTip(
            "Create a new, empty SQLite database file"
        )
        self.new_database_file_action.triggered.connect(self.create_new_database_file)
        file_menu.addAction(self.new_database_file_action)

        file_menu.addSeparator()

        # Import Transactions from CSV
        self.import_csv_action = QAction(
            self.style().standardIcon(QStyle.SP_DialogOpenButton),
            "&Import Transactions from CSV...",
            self,
        )
        self.import_csv_action.setStatusTip(
            "Import transactions from a CSV file into the current database"
        )
        if self.import_csv_action.icon().isNull():
            logging.warning(
                f"Menu Action Icon for '{self.import_csv_action.text()}' is NULL. Check icon theme for 'document-import'."
            )
        else:
            logging.debug(
                f"Menu Action Icon for '{self.import_csv_action.text()}' is VALID."
            )
        self.import_csv_action.triggered.connect(self.import_transactions_from_csv)
        file_menu.addAction(self.import_csv_action)

        # Export Transactions to CSV
        self.export_csv_action = QAction(
            QIcon.fromTheme("document-export"), "&Export Transactions to CSV...", self
        )
        self.export_csv_action.setStatusTip(
            "Export current transaction data from database to a CSV file"
        )
        self.export_csv_action.triggered.connect(self.export_transactions_to_csv)
        file_menu.addAction(self.export_csv_action)

        # NEW: Export Holdings to Excel
        self.export_excel_action = QAction(
            QIcon.fromTheme("x-office-spreadsheet"),
            "Export &Holdings to Excel...",
            self,
        )
        self.export_excel_action.setStatusTip(
            "Export the current holdings view to an Excel file"
        )
        self.export_excel_action.triggered.connect(self.export_holdings_to_excel)
        file_menu.addAction(self.export_excel_action)

        # NEW: Export to PDF/HTML
        self.export_pdf_html_action = QAction(
            QIcon.fromTheme("application-pdf"),
            "Export to &PDF/HTML...",
            self,
        )
        self.export_pdf_html_action.setStatusTip(
            "Export portfolio report to PDF or HTML"
        )
        self.export_pdf_html_action.triggered.connect(self.export_to_pdf_html)
        file_menu.addAction(self.export_pdf_html_action)

        file_menu.addSeparator()

        # Exit
        exit_action = QAction(QIcon.fromTheme("application-exit"), "E&xit", self)
        exit_action.setStatusTip("Exit application")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # --- View Menu ---
        view_menu = menu_bar.addMenu("&View")
        self.refresh_action = QAction(
            QIcon.fromTheme("view-refresh"), "&Refresh Data", self
        )
        self.refresh_action.setStatusTip(
            "Reload data from database and recalculate"
        )  # MODIFIED Tip
        self.refresh_action.setShortcut("F5")
        self.refresh_action.triggered.connect(self.refresh_data)
        view_menu.addAction(self.refresh_action)

        # View Ignored Log (can also be a button)
        self.view_ignored_log_action = QAction(
            QIcon.fromTheme("dialog-warning"), "View &Ignored Log", self
        )
        self.view_ignored_log_action.setStatusTip(
            "View transactions ignored during the last calculation"
        )
        self.view_ignored_log_action.triggered.connect(self.show_ignored_log)
        self.view_ignored_log_action.setEnabled(False)  # Initially disabled
        view_menu.addAction(self.view_ignored_log_action)

        # --- Transactions Menu ---
        tx_menu = menu_bar.addMenu("&Transactions")
        self.add_transaction_action = QAction(
            QIcon.fromTheme("list-add"), "&Add Transaction...", self
        )
        self.add_transaction_action.setStatusTip(
            "Manually add a new transaction to the database"
        )  # MODIFIED Tip
        self.add_transaction_action.triggered.connect(self.open_add_transaction_dialog)
        tx_menu.addAction(self.add_transaction_action)

        # --- Settings Menu ---
        settings_menu = menu_bar.addMenu("&Settings")
        acc_currency_action = QAction("Account &Currencies...", self)
        acc_currency_action.setStatusTip(
            "Configure currency for each investment account"
        )
        acc_currency_action.triggered.connect(self.show_account_currency_dialog)
        settings_menu.addAction(acc_currency_action)

        symbol_settings_action = QAction("&Symbol Settings...", self)
        symbol_settings_action.setStatusTip(
            "Set manual overrides, symbol mappings, and excluded symbols"
        )
        symbol_settings_action.triggered.connect(self.show_symbol_settings_dialog)
        settings_menu.addAction(symbol_settings_action)

        settings_menu.addSeparator()
        clear_cache_action = QAction("Clear &Cache Files...", self)
        clear_cache_action.setStatusTip(
            "Delete all application cache files (market data, etc.)"
        )
        clear_cache_action.triggered.connect(self.clear_cache_files_action_triggered)
        settings_menu.addAction(clear_cache_action)

        choose_currencies_action = QAction("Choose Currencies...", self)
        choose_currencies_action.setStatusTip(
            "Select which currencies to show in the currency combo box"
        )
        choose_currencies_action.triggered.connect(self.show_choose_currencies_dialog)
        settings_menu.addAction(choose_currencies_action)

        # --- Theme Submenu ---
        settings_menu.addSeparator()
        theme_menu = settings_menu.addMenu("&Theme")
        self.theme_action_group = QActionGroup(self)
        self.theme_action_group.setExclusive(True)

        self.light_theme_action = QAction("Light Mode", self)
        self.light_theme_action.setCheckable(True)
        theme_menu.addAction(self.light_theme_action)
        self.theme_action_group.addAction(self.light_theme_action)
        self.light_theme_action.triggered.connect(
            lambda: self.apply_theme("light")
        )  # Connect action

        self.dark_theme_action = QAction("Dark Mode", self)
        self.dark_theme_action.setCheckable(True)
        theme_menu.addAction(self.dark_theme_action)
        self.theme_action_group.addAction(self.dark_theme_action)
        self.dark_theme_action.triggered.connect(
            lambda: self.apply_theme("dark")
        )  # Connect action
        # --- End Theme Submenu ---

        # --- Help Menu ---
        help_menu = menu_bar.addMenu("&Help")  # Define help_menu
        csv_help_action = QAction(
            "CSV Import Format Help...", self
        )  # Help still relevant for import
        csv_help_action.setStatusTip(
            "Show required CSV file format and headers for import"
        )
        csv_help_action.triggered.connect(self.show_csv_format_help)
        help_menu.addAction(csv_help_action)

        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about_dialog)
        help_menu.addAction(about_action)

    # --- ADDED: Slot for CSV Format Help ---
    @Slot()
    def show_csv_format_help(self):
        """Displays a message box with information about the expected CSV format."""
        help_text = """
<b>Required CSV File Format</b>

The CSV file should contain the following columns (header names must match exactly):

<ol>
    <li><b>"Date (MMM DD, YYYY)"</b>: Transaction date (e.g., <i>Jan 01, 2023</i>, <i>Dec 31, 2024</i>)</li>
    <li><b>Transaction Type</b>: Type of transaction (e.g., <i>Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees</i>)</li>
    <li><b>Stock / ETF Symbol</b>: Ticker symbol (e.g., <i>AAPL, GOOG</i>). Use <i>$CASH</i> for cash deposits/withdrawals.</li>
    <li><b>Quantity of Units</b>: Number of shares/units (positive). Required for most types.</li>
    <li><b>Amount per unit</b>: Price per share/unit (positive). Required for Buy/Sell.</li>
    <li><b>Total Amount</b>: Total value of the transaction (optional, can be calculated for Buy/Sell). Required for some Dividends.</li>
    <li><b>Fees</b>: Transaction fees/commissions (positive).</li>
    <li><b>Investment Account</b>: Name of the account (e.g., <i>Brokerage A, IRA</i>).</li>
    <li><b>Split Ratio (new shares per old share)</b>: Required only for 'Split' type (e.g., <i>2</i> for 2-for-1).</li>
    <li><b>Note</b>: Optional text note for the transaction.</li>
</ol>

<b>Example Rows:</b>
<pre>
"Date (MMM DD, YYYY)",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
"Jan 15, 2023",Buy,AAPL,10,150.25,1502.50,5.95,Brokerage A,,Bought Apple shares
"Feb 01, 2023",Dividend,MSFT,,,50.00,,Brokerage A,,Microsoft dividend received
"Mar 10, 2023",Deposit,$CASH,1000,,1000.00,,IRA,,Initial IRA contribution
</pre>
"""
        QMessageBox.information(self, "CSV Format Help", help_text)

    # --- END ADDED ---

    @Slot(str)  # Ensure Slot decorator is imported
    def _chart_holding_history(self, symbol: str):
        """Handles 'Chart History' context menu action by showing a price chart dialog."""
        # logging.info(f"Action triggered: Chart History for {symbol}")

        if symbol == CASH_SYMBOL_CSV:
            QMessageBox.information(self, "Info", "Cannot chart history for Cash.")
            return

        # --- Get required data ---
        yf_symbol = None
        price_data_df = None

        # --- Use the stored map ---
        if hasattr(self, "internal_to_yf_map") and isinstance(
            self.internal_to_yf_map, dict
        ):
            yf_symbol = self.internal_to_yf_map.get(symbol)
            if not yf_symbol:
                # Fallback: Check if the symbol itself is a valid-looking ticker
                if "." not in symbol and ":" not in symbol and " " not in symbol:
                    logging.debug(
                        f"Symbol '{symbol}' not in map, trying as direct YF ticker."
                    )
                    yf_symbol = symbol.upper()
                else:
                    logging.warning(
                        f"No YF symbol mapping found for internal symbol: {symbol}"
                    )
        else:
            logging.warning(
                "internal_to_yf_map attribute not found or invalid. Cannot look up YF ticker."
            )
            # Attempt to use symbol directly as YF symbol as a last resort
            if "." not in symbol and ":" not in symbol and " " not in symbol:
                logging.debug(
                    f"Symbol map missing, trying '{symbol}' as direct YF ticker."
                )
                yf_symbol = symbol.upper()
            else:
                QMessageBox.warning(
                    self,
                    "Mapping Error",
                    f"Could not determine Yahoo Finance ticker for '{symbol}'.",
                )
                return  # Cannot proceed without a YF ticker
        # --- End Use the stored map ---

        # Use ADJUSTED prices for single symbol history chart
        if (
            yf_symbol
            and hasattr(self, "historical_prices_yf_adjusted")
            and isinstance(self.historical_prices_yf_adjusted, dict)
        ):
            price_data_df = self.historical_prices_yf_adjusted.get(yf_symbol)
            if price_data_df is None:
                logging.warning(
                    f"No adjusted historical price data found for YF symbol: {yf_symbol} (Internal: {symbol})"
                )
        else:
            if not yf_symbol:
                logging.warning("No YF symbol determined.")  # Already logged above
            else:
                logging.warning(
                    "historical_prices_yf_adjusted attribute not found or invalid."
                )

        if price_data_df is None or price_data_df.empty:
            QMessageBox.warning(
                self,
                "No Data",
                f"Could not find historical price data for symbol '{symbol}' (Ticker: {yf_symbol or 'N/A'}).",
            )
            return

        # 2. Date Range (from main UI controls)
        try:
            start_date = self.graph_start_date_edit.date().toPython()
            end_date = self.graph_end_date_edit.date().toPython()

            # --- ADDED: Ensure end_date is not in the future ---
            today = date.today()
            if end_date > today:
                end_date = today
                self.graph_end_date_edit.setDate(QDate(end_date))  # Update UI
                logging.info(
                    f"Graph end date was in future, reset to today: {end_date}"
                )
            # --- END ADDED ---

            if start_date >= end_date:
                QMessageBox.warning(
                    self,
                    "Invalid Date Range",
                    "Graph start date must be before end date.",
                )
                return
        except Exception as e:
            logging.error(f"Error getting date range for symbol chart: {e}")
            QMessageBox.critical(
                self, "Error", "Could not get date range from UI controls."
            )
            return

        # 3. Get Local Currency Code for Labeling
        display_currency_code = self._get_currency_for_symbol(symbol)
        if not display_currency_code:
            display_currency_code = self.config.get("default_currency", "USD")

        # --- Create and Show Dialog ---
        try:
            dialog = SymbolChartDialog(
                symbol=symbol,  # Pass internal symbol for title
                start_date=start_date,
                end_date=end_date,
                price_data=price_data_df.copy(),  # Pass a copy of the data
                display_currency=display_currency_code,  # Pass currency CODE
                parent=self,
            )
            dialog.exec()  # Show modal

        except Exception as e_dialog:
            logging.exception(
                f"Error creating or showing SymbolChartDialog for {symbol}"
            )
            QMessageBox.critical(
                self,
                "Chart Error",
                f"Failed to display chart for {symbol}:\n{e_dialog}",
            )

    def _get_currency_for_symbol(self, symbol_to_find: str) -> Optional[str]:
        """
        Finds the local currency code associated with a symbol based on transaction data
        or the account currency map. Returns the default currency if not found.

        Args:
            symbol_to_find (str): The internal symbol (e.g., 'AAPL', 'SET:BKK').

        Returns:
            Optional[str]: The 3-letter currency code (e.g., 'USD', 'THB') or None if lookup fails badly.
                           Returns default currency as fallback.
        """  # 1. Check if it's a cash symbol
        if is_cash_symbol(symbol_to_find):
            # For a generic $CASH symbol, we can't determine currency from symbol alone.
            # The calling context should handle this. As a fallback, return app's default.
            return self.config.get("default_currency", "USD")

        # 2. Check Holdings Data (most reliable if data is loaded)
        # This requires holdings_data to have the 'Local Currency' column correctly populated.
        if (
            hasattr(self, "holdings_data")
            and not self.holdings_data.empty
            and "Symbol" in self.holdings_data.columns
        ):
            symbol_rows = self.holdings_data[
                self.holdings_data["Symbol"] == symbol_to_find
            ]
            if not symbol_rows.empty and "Local Currency" in symbol_rows.columns:
                first_currency = symbol_rows["Local Currency"].iloc[0]
                if (
                    pd.notna(first_currency)
                    and isinstance(first_currency, str)
                    and len(first_currency) == 3
                ):
                    # logging.debug(f"Found currency '{first_currency}' for symbol '{symbol_to_find}' via holdings_data.")
                    return first_currency.upper()

        # 3. Fallback: Check original transactions (less efficient but broader)
        # This requires original_data to have 'Investment Account' and 'Stock / ETF Symbol'
        if hasattr(self, "original_data") and not self.original_data.empty:
            if (
                "Stock / ETF Symbol" in self.original_data.columns
                and "Investment Account" in self.original_data.columns
            ):
                symbol_rows_orig = self.original_data[
                    self.original_data["Stock / ETF Symbol"] == symbol_to_find
                ]
                if not symbol_rows_orig.empty:
                    first_account = symbol_rows_orig["Investment Account"].iloc[0]
                    if pd.notna(first_account) and isinstance(first_account, str):
                        # Get currency from the account map
                        acc_map = self.config.get("account_currency_map", {})
                        currency_from_map = acc_map.get(first_account)
                        if (
                            currency_from_map
                            and isinstance(currency_from_map, str)
                            and len(currency_from_map) == 3
                        ):
                            # logging.debug(f"Found currency '{currency_from_map}' for symbol '{symbol_to_find}' via original_data/account_map (Account: {first_account}).")
                            return currency_from_map.upper()

        # 4. Final Fallback: Return application's default currency
        default_curr = self.config.get("default_currency", "USD")
        # logging.debug(f"Could not find specific currency for symbol '{symbol_to_find}'. Using default: {default_curr}")
        return default_curr

    # Add this NEW slot method to PortfolioApp
    @Slot()
    def show_account_currency_dialog(self):
        """Shows the dialog to edit account currencies."""
        current_map = self.config.get("account_currency_map", {})
        current_default = self.config.get("default_currency", "USD")
        # Use all accounts detected during the last load/refresh
        accounts_to_show = (
            self.available_accounts
            if self.available_accounts
            else list(current_map.keys())
        )
        # Get the list of user-selectable currencies from the config
        user_currencies = self.config.get("user_currencies", COMMON_CURRENCIES.copy())

        updated_settings = AccountCurrencyDialog.get_settings(
            parent=self,
            current_map=current_map,
            current_default=current_default,
            all_accounts=accounts_to_show,
            # Pass the list of currencies to the dialog
            user_currencies=user_currencies,
        )

        if updated_settings:  # If user clicked Save
            new_map, new_default = updated_settings
            # Check if settings actually changed
            map_changed = new_map != self.config.get("account_currency_map", {})
            default_changed = new_default != self.config.get("default_currency", "USD")

            if map_changed or default_changed:
                logging.info(
                    "Account currency settings changed. Saving and refreshing..."
                )
                self.config["account_currency_map"] = new_map
                self.config["default_currency"] = new_default
                self.save_config()  # Save the updated config
                self.refresh_data()  # Trigger a full refresh as currencies impact calculations
            else:
                # logging.info("Account currency settings unchanged.")
                pass

    @contextlib.contextmanager
    def _programmatic_date_change(self):
        """Context manager to temporarily disable date change signals."""
        self._is_setting_dates_programmatically = True
        yield
        self._is_setting_dates_programmatically = False

    @Slot(str)
    def _set_graph_date_range(self, period: str):
        """Sets the graph start and end dates based on a preset period string."""
        end_date = date.today()
        start_date = None

        if period == "1W":
            start_date = end_date - timedelta(weeks=1)
        elif period == "MTD":
            start_date = end_date.replace(day=1)
        elif period == "1M":
            start_date = end_date - timedelta(days=30)
        elif period == "3M":
            start_date = end_date - timedelta(days=91)  # Approx 3 months
        elif period == "6M":
            start_date = end_date - timedelta(days=182)
        elif period == "YTD":
            start_date = end_date.replace(month=1, day=1)
        elif period == "1Y":
            start_date = end_date - timedelta(days=365)
        elif period == "3Y":
            start_date = end_date - timedelta(days=3 * 365)
        elif period == "5Y":
            start_date = end_date - timedelta(days=5 * 365)
        elif period == "10Y":
            start_date = end_date - timedelta(days=10 * 365)
        elif period == "All":
            # Use the full transaction history to find the true earliest date,
            # as self.full_historical_data only reflects the last calculated range.
            if (
                hasattr(self, "all_transactions_df_cleaned_for_logic")
                and not self.all_transactions_df_cleaned_for_logic.empty
            ):
                try:
                    start_date = (
                        self.all_transactions_df_cleaned_for_logic["Date"].min().date()
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not determine 'All' start date from transaction data: {e}"
                    )
                    start_date = None  # Fallback

            # Fallback if no transaction data is loaded yet
            if start_date is None:
                start_date = self.config.get(
                    "graph_start_date", DEFAULT_GRAPH_START_DATE
                )
                if isinstance(start_date, str):
                    start_date = date.fromisoformat(start_date)

        if start_date:
        # --- ADDED: Clamp start date to first transaction date of selected accounts ---
        # If the calculated start date (e.g. Jan 1 for YTD) is earlier than the
        # actual first transaction (e.g. Oct 22), use the transaction date.
            try:
                if (
                    hasattr(self, "all_transactions_df_cleaned_for_logic")
                    and not self.all_transactions_df_cleaned_for_logic.empty
                ):
                    # Get selected accounts
                    selected_accounts = getattr(self, "selected_accounts", [])
                    if not selected_accounts:
                        selected_accounts = self.config.get("selected_accounts", [])
                    
                    df = self.all_transactions_df_cleaned_for_logic
                    if selected_accounts and "Account" in df.columns:
                        df = df[df["Account"].isin(selected_accounts)]
                    
                    if not df.empty and "Date" in df.columns:
                        min_tx_date = df["Date"].min()
                        if pd.notna(min_tx_date):
                            if isinstance(min_tx_date, (pd.Timestamp, datetime)):
                                min_tx_date = min_tx_date.date()
                            
                            # If preset date is BEFORE the first transaction, clamp it.
                            # Exception: If "All" is selected, we already used min date above, so this is redundant but safe.
                            if start_date < min_tx_date:
                                logging.debug(f"DEBUG: Clamping preset '{period}' start date from {start_date} to first transaction {min_tx_date}")
                                start_date = min_tx_date
            except Exception as e:
                logging.error(f"Error clamping start date: {e}")
            # --- END ADDED ---

            with self._programmatic_date_change():
                logging.debug(f"DEBUG: _set_graph_date_range setting date to {start_date}")
                self.graph_start_date_edit.setDate(QDate(start_date))
                self.graph_end_date_edit.setDate(QDate(end_date))
                logging.info(
                    f"Graph date range set to '{period}': {start_date} to {end_date}"
                )
                # Automatically trigger graph update when a preset is clicked
                self.graph_update_button.click()
        else:
            logging.warning(f"Could not determine start date for period '{period}'.")

    @Slot(str)
    def _on_preset_selected(self, text: str):
        """Handles selection from the preset dropdown."""
        if text == "Presets...":
            return  # Do nothing for the placeholder

        # Block signals to prevent the change from being immediately cleared
        self.date_preset_combo.blockSignals(True)
        self._set_graph_date_range(text)
        self.date_preset_combo.setCurrentText(text)
        self.date_preset_combo.blockSignals(False)

    @Slot()
    def _clear_preset_button_selection(self):
        """Clears the selection of any preset date range button."""
        if self._is_setting_dates_programmatically:
            return

        # Reset the preset dropdown to the placeholder
        if self.date_preset_combo.currentIndex() != 0:
            self.date_preset_combo.blockSignals(True)
            self.date_preset_combo.setCurrentIndex(0)
            self.date_preset_combo.blockSignals(False)

    @Slot()
    def export_transactions_to_csv(self):
        """
        Exports all transactions from the current SQLite database to a new CSV file.
        The CSV will use standard "verbose" headers for better readability if possible,
        or the cleaned headers.
        """
        if not self.db_conn:
            self.show_warning(
                "No database open to export from.", popup=True, title="No Database"
            )
            return

        # Load all transactions from the database.
        # load_all_transactions_from_db returns a DataFrame with 'cleaned' column names
        # and 'original_index' (displayed as 'No.') as the DB id.
        # Use account_currency_map and default_currency from config for cleaning after DB load
        acc_map_config_export = self.config.get("account_currency_map", {})
        def_curr_config_export = self.config.get(
            "default_currency", config.DEFAULT_CURRENCY
        )

        df_to_export, success = load_all_transactions_from_db(
            self.db_conn, acc_map_config_export, def_curr_config_export
        )

        if not success or df_to_export is None:
            self.show_error(
                "Failed to load transactions from the database for export.",
                popup=True,
                title="Load Error",
            )
            return
        if df_to_export.empty:
            self.show_info(
                "No transactions in the database to export.",
                popup=True,
                title="No Data",
            )
            return

        # Suggest starting directory based on last export or Documents
        start_dir = self.config.get("last_csv_export_path", "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = (
                QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                or os.getcwd()
            )

        default_export_filename = f"{os.path.splitext(os.path.basename(self.DB_FILE_PATH))[0]}_export_{datetime.now().strftime('%Y%m%d')}.csv"

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export Transactions to CSV",
            os.path.join(start_dir, default_export_filename),
            "CSV Files (*.csv);;All Files (*)",
        )

        if fname:
            if not fname.lower().endswith(".csv"):
                fname += ".csv"

            # Save the directory for next time
            self.config["last_csv_export_path"] = os.path.dirname(fname)
            self.save_config()

            # logging.info(f"Attempting to export transactions to CSV: {fname}")
            self.set_status(f"Exporting transactions to {os.path.basename(fname)}...")
            QApplication.processEvents()

            try:
                # Prepare DataFrame for export:
                # 1. Map cleaned DB column names back to standard "verbose" CSV headers for user-friendliness.
                # 2. Order columns according to DESIRED_CLEANED_COLUMN_ORDER (which implies verbose headers).
                # 3. Format Date column to "MMM DD, YYYY".
                # 4. Drop the 'original_index' (internal DB ID) column or rename it for export.

                df_export_prepared = df_to_export.copy()

                # Drop 'original_index' (internal DB ID) or rename if you want to keep it
                if "original_index" in df_export_prepared.columns:
                    df_export_prepared.drop(columns=["original_index"], inplace=True)
                    # Or rename: df_export_prepared.rename(columns={'original_index': 'DatabaseID'}, inplace=True)

                # Map cleaned DB names to standard verbose CSV names for export
                # Create a reverse map from EXPECTED_CLEANED_COLUMNS to the keys of STANDARD_ORIGINAL_TO_CLEANED_MAP
                # This is a bit complex; simpler is to define the target CSV headers directly.
                # DESIRED_CLEANED_COLUMN_ORDER uses cleaned names. We need a map: Cleaned -> Verbose Original
                cleaned_to_verbose_map = {
                    v: k
                    for k, v in COLUMN_MAPPING_CSV_TO_INTERNAL.items()
                    if v in DESIRED_CLEANED_COLUMN_ORDER
                }
                # Prioritize the more verbose ones from STANDARD_ORIGINAL_TO_CLEANED_MAP in csv_utils.py
                # from csv_utils import STANDARD_ORIGINAL_TO_CLEANED_MAP as CSV_UTILS_STANDARD_MAP
                # cleaned_to_verbose_map = {v:k for k,v in CSV_UTILS_STANDARD_MAP.items()}

                # For simplicity, let's assume we export with the *cleaned* headers for now,
                # as mapping back to potentially ambiguous verbose headers can be tricky.
                # If specific verbose headers are desired, a direct rename map is better.
                # Example of renaming to verbose:
                # verbose_rename_map = {
                #     "Date": "Date (MMM DD, YYYY)",
                #     "Type": "Transaction Type",
                #     "Symbol": "Stock / ETF Symbol",
                #     "Quantity": "Quantity of Units",
                #     "Price/Share": "Amount per unit",
                #     # ... and so on for all columns in DESIRED_CLEANED_COLUMN_ORDER
                #     "Commission": "Fees",
                #     "Account": "Investment Account",
                #     "Split Ratio": "Split Ratio (new shares per old share)"
                # }
                # df_export_prepared.rename(columns=verbose_rename_map, inplace=True)
                # export_column_order = [verbose_rename_map.get(col, col) for col in DESIRED_CLEANED_COLUMN_ORDER if verbose_rename_map.get(col,col) in df_export_prepared.columns]

                # Using cleaned headers for export for now (matches DESIRED_CLEANED_COLUMN_ORDER)
                export_column_order = [
                    col
                    for col in DESIRED_CLEANED_COLUMN_ORDER
                    if col in df_export_prepared.columns
                ]
                # Add any other columns that might exist but are not in the desired order (e.g. Note if not listed)
                other_cols_in_export = [
                    col
                    for col in df_export_prepared.columns
                    if col not in export_column_order
                ]
                df_export_final = df_export_prepared[
                    export_column_order + other_cols_in_export
                ]

                # Format Date column to "MMM DD, YYYY" for CSV
                if (
                    "Date" in df_export_final.columns
                    and pd.api.types.is_datetime64_any_dtype(df_export_final["Date"])
                ):
                    try:
                        df_export_final["Date"] = df_export_final["Date"].dt.strftime(
                            CSV_DATE_FORMAT
                        )
                    except (
                        AttributeError
                    ):  # Handle cases where it might already be string after some ops
                        pass
                elif (
                    "Date" in df_export_final.columns
                ):  # If Date is string, try to parse and reformat
                    try:
                        df_export_final["Date"] = pd.to_datetime(
                            df_export_final["Date"], errors="coerce"
                        ).dt.strftime(CSV_DATE_FORMAT)
                    except:  # If parsing fails, leave as is
                        pass

                # Write to CSV via helper
                write_dataframe_to_csv(df_export_final, fname)
                self.show_info(
                    f"All transactions exported successfully to:\n{fname}",
                    popup=True,
                    title="Export Successful",
                )
                self.set_status(f"Exported to {os.path.basename(fname)}")
            except Exception as e:
                self.show_error(
                    f"Could not export transactions to CSV:\n{e}",
                    popup=True,
                    title="Export Error",
                )
                logging.error(
                    f"Error exporting transactions to CSV: {e}", exc_info=True
                )
                self.set_status("Export failed.")
        else:
            # logging.info("CSV export cancelled by user.")
            self.set_status("Export cancelled.")

    @Slot()
    def export_holdings_to_excel(self):
        """Exports the current main holdings table view to an Excel (.xlsx) file."""
        if not hasattr(self, "table_model") or self.table_model.rowCount() == 0:
            self.show_warning(
                "There is no holdings data to export.", popup=True, title="No Data"
            )
            return

        # Suggest starting directory and filename
        start_dir = self.config.get("last_excel_export_path", "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = (
                QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                or os.getcwd()
            )

        default_export_filename = (
            f"Investa_Holdings_{datetime.now().strftime('%Y%m%d')}.xlsx"
        )

        # Open file dialog
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Holdings to Excel",
            os.path.join(start_dir, default_export_filename),
            "Excel Files (*.xlsx);;All Files (*)",
        )

        if file_path:
            # Ensure correct extension
            if not file_path.lower().endswith(".xlsx"):
                file_path += ".xlsx"

            # Save the directory for next time
            self.config["last_excel_export_path"] = os.path.dirname(file_path)
            self.save_config()

            # logging.info(f"Attempting to export holdings to Excel: {file_path}")
            self.set_status(f"Exporting holdings to {os.path.basename(file_path)}...")
            QApplication.processEvents()

            try:
                # Get the DataFrame from the model
                df_to_export = self.table_model._data.copy()

                # The model's data might have internal columns like 'is_group_header'
                # and 'group_key'. We should remove these before exporting.
                internal_cols = ["is_group_header", "group_key"]
                cols_to_drop = [
                    col for col in internal_cols if col in df_to_export.columns
                ]
                if cols_to_drop:
                    df_to_export.drop(columns=cols_to_drop, inplace=True)

                # Write to Excel via helper
                write_dataframe_to_excel(df_to_export, file_path)

                self.show_info(
                    f"Holdings table exported successfully to:\n{file_path}",
                    popup=True,
                    title="Export Successful",
                )
                self.set_status(f"Exported to {os.path.basename(file_path)}")

            except ImportError:
                logging.error("Export to Excel failed: 'openpyxl' library not found.")
                self.show_error(
                    "Could not export to Excel because the 'openpyxl' library is not installed.\n\nPlease install it by running:\n<b>pip install openpyxl</b>",
                    popup=True,
                    title="Dependency Missing",
                )
                self.set_status("Export failed: openpyxl missing.")
            except Exception as e:
                logging.error(f"Error exporting holdings to Excel: {e}", exc_info=True)
                self.show_error(
                    f"Could not export holdings to Excel:\n{e}",
                    popup=True,
                    title="Export Error",
                )
                self.set_status("Export failed.")

    def export_to_pdf_html(self):
        """Exports the portfolio report to PDF or HTML."""
        if not hasattr(self, "table_model") or self.table_model.rowCount() == 0:
            self.show_warning(
                "There is no data to export.", popup=True, title="No Data"
            )
            return

        # Suggest filename
        current_date_str = datetime.now().strftime("%Y%m%d")
        default_filename = f"Investa_Report_{current_date_str}.pdf"
        
        start_dir = self.config.get("last_report_export_path", "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = (
                QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                or os.getcwd()
            )

        file_path, filter_selected = QFileDialog.getSaveFileName(
            self,
            "Export Portfolio Report",
            os.path.join(start_dir, default_filename),
            "PDF Files (*.pdf);;HTML Files (*.html);;All Files (*)",
        )

        if not file_path:
            return

        # Ensure correct extension
        if filter_selected.startswith("PDF") and not file_path.lower().endswith(".pdf"):
            file_path += ".pdf"
        elif filter_selected.startswith("HTML") and not file_path.lower().endswith(".html"):
            file_path += ".html"

        self.config["last_report_export_path"] = os.path.dirname(file_path)
        self.save_config()
        
        self.set_status(f"Generating report to {os.path.basename(file_path)}...")
        QApplication.processEvents()

        try:
            # --- Helper to capture figure as base64 image ---
            def fig_to_base64(fig):
                if not fig: return ""
                buf = BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode("utf-8")
                return f"data:image/png;base64,{img_str}"

            # --- Capture Charts ---
            value_chart_img = fig_to_base64(getattr(self, "abs_value_fig", None))
            return_chart_img = fig_to_base64(getattr(self, "perf_return_fig", None))
            drawdown_chart_img = fig_to_base64(getattr(self, "drawdown_fig", None))
            alloc_account_img = fig_to_base64(getattr(self, "account_fig", None))
            alloc_holding_img = fig_to_base64(getattr(self, "holdings_fig", None))

            # --- Gather Summary Data ---
            # Note: summary items are tuples (label_widget, value_widget, [percent_widget])
            # We need the text from the value_widget (index 1)
            # Some items are unpacked directly in __init__, so we access them directly.
            
            # Helper to safely get text
            def get_text(widget):
                return widget.text() if widget and hasattr(widget, "text") else "N/A"

            summary_data = {
                "Net Value": self.summary_net_value[1].text() if hasattr(self, "summary_net_value") else "N/A",
                "Day's G/L": get_text(getattr(self, "summary_day_change_value", None)),
                "Total G/L": self.summary_total_gain[1].text() if hasattr(self, "summary_total_gain") else "N/A",
                "Unrealized G/L": self.summary_unrealized_gain[1].text() if hasattr(self, "summary_unrealized_gain") else "N/A",
                "Realized G/L": self.summary_realized_gain[1].text() if hasattr(self, "summary_realized_gain") else "N/A",
                "Dividends": self.summary_dividends[1].text() if hasattr(self, "summary_dividends") else "N/A",
                "Fees": self.summary_commissions[1].text() if hasattr(self, "summary_commissions") else "N/A",
                "Cash Balance": self.summary_cash[1].text() if hasattr(self, "summary_cash") else "N/A",
                "Total Return %": self.summary_total_return_pct[1].text() if hasattr(self, "summary_total_return_pct") else "N/A",
                "Ann. TWR %": self.summary_annualized_twr[1].text() if hasattr(self, "summary_annualized_twr") else "N/A",
                "FX Gain/Loss": get_text(getattr(self, "summary_fx_gl_abs_value", None)),
                "FX G/L %": get_text(getattr(self, "summary_fx_gl_pct_value", None)),
            }

            # --- Generate HTML ---
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; color: #333; font-size: 10pt; margin: 20px; }}
                    h1 {{ color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 5px; font-size: 16pt; margin-bottom: 15px; }}
                    h2 {{ color: #2980b9; margin-top: 20px; font-size: 14pt; border-bottom: 1px solid #eee; padding-bottom: 3px; }}
                    h3 {{ font-size: 11pt; margin-bottom: 5px; color: #555; }}
                    
                    .summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-bottom: 20px; page-break-inside: avoid; }}
                    .summary-item {{ background: #f8f9fa; padding: 8px; border-radius: 4px; border: 1px solid #e9ecef; }}
                    .summary-label {{ font-size: 0.8em; color: #7f8c8d; display: block; }}
                    .summary-value {{ font-size: 1.0em; font-weight: bold; color: #2c3e50; }}
                    
                    /* Table styling for Landscape mode */
                    table {{ width: 100%; border-collapse: collapse; margin-top: 15px; font-size: 7pt; page-break-inside: auto; }}
                    th, td {{ padding: 4px 5px; text-align: right; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f2f2f2; text-align: center; font-weight: bold; white-space: nowrap; }}
                    td:first-child {{ text-align: left; white-space: nowrap; font-weight: bold; }}
                    tr {{ page-break-inside: avoid; page-break-after: auto; }}
                    
                    .chart-container {{ text-align: center; margin-bottom: 20px; page-break-inside: avoid; }}
                    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                    
                    .footer {{ margin-top: 30px; font-size: 8pt; color: #7f8c8d; text-align: center; border-top: 1px solid #eee; padding-top: 5px; }}
                </style>
            </head>
            <body>
                <h1>Portfolio Report</h1>
                <p style="font-size: 9pt; color: #666;">Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

                <h2>Summary</h2>
                <div class="summary-grid">
                    {''.join([f'<div class="summary-item"><span class="summary-label">{k}</span><span class="summary-value">{v}</span></div>' for k, v in summary_data.items()])}
                </div>

                <h2>Performance</h2>
                <div style="display: flex; justify-content: space-around; page-break-inside: avoid;">
                    <div class="chart-container" style="width: 48%;">
                        <h3>Portfolio Value</h3>
                        <img src="{value_chart_img}" style="max-height: 300px;" />
                    </div>
                    <div class="chart-container" style="width: 48%;">
                        <h3>Portfolio Return</h3>
                        <img src="{return_chart_img}" style="max-height: 300px;" />
                    </div>
                </div>
                <div class="chart-container">
                    <h3>Drawdown</h3>
                    <img src="{drawdown_chart_img}" style="max-height: 250px;" />
                </div>

                <h2>Allocation</h2>
                <div style="display: flex; justify-content: space-around; page-break-inside: avoid;">
                    <div class="chart-container" style="width: 48%;">
                        <h3>By Account</h3>
                        <img src="{alloc_account_img}" style="max-height: 250px;" />
                    </div>
                    <div class="chart-container" style="width: 48%;">
                        <h3>By Holding</h3>
                        <img src="{alloc_holding_img}" style="max-height: 250px;" />
                    </div>
                </div>

                <h2>Holdings</h2>
            """

            # Add Holdings Table
            if hasattr(self, "table_model"):
                df = self.table_model._data.copy()
                # Clean up for display - Drop technical and less critical columns for print
                cols_to_drop = [
                    "is_group_header", "group_key", "original_index",
                    "Sector", "Industry", "Currency", "Exchange",
                    "Realized G/L", "Unrealized G/L", "Total G/L", # Keep G/L % and Total Value
                    "Cost Basis", "Mkt Price" # Can be inferred or less critical than Value
                ]
                # Keep: Account, Symbol, Quantity, Mkt Value, Day's G/L, Day's G/L %, Total G/L %, Yield %
                
                df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)
                
                # Rename for brevity
                df.rename(columns={
                    "Account": "Acct",
                    "Quantity": "Qty",
                    "Mkt Value": "Value",
                    "Day's G/L": "Day G/L",
                    "Total G/L %": "Tot %",
                    "Yield (Cost) %": "Yld(C)%",
                    "Yield (Mkt) %": "Yld(M)%"
                }, inplace=True)
                
                # Convert to HTML table
                html_content += df.to_html(index=False, border=0, classes="table")

            html_content += """
                <div class="footer">
                    Generated by Investa Portfolio Dashboard
                </div>
            </body>
            </html>
            """

            # --- Save File ---
            if file_path.lower().endswith(".html"):
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(html_content)
            else:
                # PDF Export
                document = QTextDocument()
                document.setHtml(html_content)
                
                printer = QPrinter(QPrinter.HighResolution)
                printer.setOutputFormat(QPrinter.PdfFormat)
                printer.setOutputFileName(file_path)
                printer.setPageSize(QPageSize(QPageSize.A4))
                printer.setPageOrientation(QPageLayout.Landscape) # Switch to Landscape
                
                # IMPORTANT: Set document page size to match printer page rect
                # This ensures the HTML layout respects the PDF page width
                document.setPageSize(QSizeF(printer.pageRect(QPrinter.Unit.Point).size()))
                
                document.print_(printer)

            self.show_info(
                f"Report exported successfully to:\n{file_path}",
                popup=True,
                title="Export Successful",
            )
            self.set_status(f"Report exported to {os.path.basename(file_path)}")

        except Exception as e:
            logging.error(f"Error exporting report: {e}", exc_info=True)
            self.show_error(f"Failed to export report:\n{e}", popup=True)
            self.set_status("Export failed.")

    def show_about_dialog(self):
        # Placeholder - shows a simple message box with app info
        QMessageBox.about(
            self,
            "About Investa",
            "<b>Investa Portfolio Dashboard</b><br><br>"
            "Version: 0.3.0 (Features: Tx Edit/Delete, Ignored Log, Table Filter, Settings Edit)<br>"
            "Author: Kit Matan<br>"
            "License: MIT<br><br>"
            "Data provided by Yahoo Finance. Use at your own risk.",
        )

    @Slot()
    def show_manual_overrides_dialog(self):  # Renamed method
        """Shows the dialog to edit manual prices."""
        # self.manual_overrides_dict should be populated during __init__ by _load_manual_overrides
        if (
            not hasattr(self, "manual_overrides_dict")
            or not hasattr(self, "user_symbol_map_config")
            or not hasattr(self, "user_excluded_symbols_config")
        ):
            logging.error("Symbol settings dictionaries not initialized.")
            self.show_error(
                "Manual override data is not loaded.", popup=True, title="Error"
            )
            return

        # Call the renamed static method
        updated_settings = ManualPriceDialog.get_symbol_settings(
            parent=self,
            current_overrides=self.manual_overrides_dict,
            current_symbol_map=self.user_symbol_map_config,
            current_excluded_symbols=self.user_excluded_symbols_config,
        )

        if updated_settings is not None:  # User clicked Save and validation passed
            new_manual_overrides = updated_settings.get("manual_price_overrides", {})
            new_symbol_map = updated_settings.get("user_symbol_map", {})
            new_excluded_symbols = set(
                updated_settings.get("user_excluded_symbols", [])
            )  # Convert list to set

            if (
                new_manual_overrides != self.manual_overrides_dict
                or new_symbol_map != self.user_symbol_map_config
                or new_excluded_symbols != self.user_excluded_symbols_config
            ):
                # logging.info("Symbol settings changed. Saving and refreshing...")
                self.manual_overrides_dict = new_manual_overrides
                self.user_symbol_map_config = new_symbol_map
                self.user_excluded_symbols_config = new_excluded_symbols
                if self._save_manual_overrides_to_json():  # Save all parts to JSON
                    self.refresh_data()
            else:
                # logging.info("Symbol settings unchanged.")
                pass

    def _save_manual_overrides_to_json(
        self,
    ) -> bool:  # Renamed from _save_manual_prices_to_json
        """Saves all manual settings (prices, symbol map, exclusions) to MANUAL_OVERRIDES_FILE."""
        if (
            not hasattr(self, "manual_overrides_dict")
            or not hasattr(self, "user_symbol_map_config")
            or not hasattr(self, "user_excluded_symbols_config")
        ):
            logging.error("Cannot save symbol settings, dictionary attributes missing.")
            return False

        overrides_file_path = self.MANUAL_OVERRIDES_FILE
        # logging.info(f"Saving symbol settings to: {overrides_file_path}")

        data_to_save = {
            "manual_price_overrides": dict(sorted(self.manual_overrides_dict.items())),
            "user_symbol_map": dict(sorted(self.user_symbol_map_config.items())),
            "user_excluded_symbols": sorted(
                list(self.user_excluded_symbols_config)
            ),  # Save set as sorted list
        }

        try:
            overrides_dir = os.path.dirname(overrides_file_path)
            if overrides_dir:
                os.makedirs(overrides_dir, exist_ok=True)

            with open(overrides_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            # logging.info("Symbol settings saved successfully.")
            return True
        except TypeError as e:
            logging.error(f"TypeError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Data error saving symbol settings:\n{e}"
            )
        except IOError as e:
            logging.error(f"IOError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not write to file:\n{overrides_file_path}\n{e}",
            )
            return False
        except Exception as e:
            logging.exception("Unexpected error writing symbol settings JSON")
            QMessageBox.critical(
                self,
                "Save Error",
                f"An unexpected error occurred saving symbol settings:\n{e}",
            )
            return False

    def _format_tooltip_annotation(self, selection):
        """Callback function to format mplcursors annotations."""
        try:
            artist = selection.artist
            ax = artist.axes
            x_val_num = selection.target[0]  # X value (often matplotlib numeric date)
            y_val = selection.target[1]  # Y value

            # --- Convert X value (matplotlib date num) to datetime object ---
            try:
                # Use matplotlib's num2date for robust conversion
                dt_obj = mdates.num2date(x_val_num)
                # Format the date string
                date_str = dt_obj.strftime("%Y-%m-%d")  # Or '%a, %b %d, %Y' etc.
            except (ValueError, TypeError, OverflowError) as e_date:
                logging.debug(f"Tooltip date conversion error: {e_date}")
                date_str = f"Date Err ({x_val_num:.2f})"  # Show numeric value on error

            # --- Get Series Label ---
            label = artist.get_label()
            # Clean up internal labels like '_lineX' if they appear
            if label.startswith("_"):
                label = "Portfolio"  # Or derive more cleverly if needed

            # --- Format Y value based on Axis ---
            formatted_y = "N/A"
            if ax == self.perf_return_ax:  # Check if it's the return axis
                formatted_y = f"{y_val:+.2f}%"  # Format as signed percentage
            elif ax == self.abs_value_ax:  # Check if it's the value axis
                currency_symbol = self._get_currency_symbol()
                # Use a simplified version of the axis formatter logic
                if abs(y_val) >= 1e6:
                    formatted_y = f"{currency_symbol}{y_val/1e6:,.4f}M"
                elif abs(y_val) >= 1e3:
                    formatted_y = f"{currency_symbol}{y_val/1e3:,.2f}K"
                else:
                    formatted_y = (
                        f"{currency_symbol}{y_val:,.2f}"  # Changed to 2 decimal places
                    )
            else:  # Fallback if axis doesn't match expected ones
                formatted_y = f"{y_val:,.2f}"

            # --- Set Annotation Text ---
            # Use multiline text for clarity
            annotation_text = f"{date_str}\n{label}\n{formatted_y}"
            selection.annotation.set_text(annotation_text)

            # --- Optional: Customize annotation appearance ---
            selection.annotation.get_bbox_patch().set(
                facecolor="lightyellow", alpha=0.8, edgecolor="gray"
            )
            selection.annotation.set_fontsize(8)

        except Exception as e:
            logging.error(f"Error formatting tooltip annotation: {e}")
            # Fallback annotation text
            try:
                selection.annotation.set_text(
                    f"Err ({selection.target[0]:.1f}, {selection.target[1]:.1f})"
                )
            except Exception:
                pass  # Ignore error during error reporting

    def _format_intraday_tooltip_annotation(self, selection):
        """Callback function to format mplcursors annotations for the intraday chart."""
        try:
            # The artist can be a Line2D (price) or a Rectangle (volume bar)
            artist = selection.artist

            # The x,y coordinates of the selected data point
            x_val_num = selection.target[0]
            y_val = selection.target[1]

            # Retrieve the necessary data stored in the artist's GID
            plot_info = artist.get_gid()
            intraday_df = plot_info["df"]
            is_pct_change = plot_info["is_pct_change"]
            is_fx = plot_info["is_fx"]
            x_axis_type = plot_info["x_axis_type"]  # 'numeric' or 'datetime'
            symbol = plot_info["symbol"]
            is_volume_bar = plot_info.get("is_volume", False)  # Check our custom flag

            # Determine the timestamp based on how the x-axis was plotted
            if isinstance(intraday_df.index, pd.DatetimeIndex):
                if len(artist.get_xdata()) == len(intraday_df.index):
                    # The x-data is likely a numerical index [0, 1, 2, ...]
                    # Find the closest integer index
                    idx = int(np.round(x_val_num))
                    if x_axis_type == "numeric" and 0 <= idx < len(intraday_df.index):
                        dt_obj = intraday_df.index[idx]
                    else:  # Fallback for out of bounds or if x-axis is actual datetime numbers
                        dt_obj = mdates.num2date(x_val_num)
                elif isinstance(
                    intraday_df.index, pd.DatetimeIndex
                ):  # Fallback if x_axis_type is not set
                    dt_obj = mdates.num2date(x_val_num)
            else:  # Fallback if index is not datetime
                dt_obj = mdates.num2date(x_val_num)

            date_str = (
                dt_obj.strftime("%Y-%m-%d %H:%M")
                if isinstance(dt_obj, (datetime, pd.Timestamp))
                else "Invalid Date"
            )

            # Format the Y-value based on the plot type
            if is_volume_bar:  # If it's a volume bar
                formatted_y = f"Volume: {y_val:,.0f}"
            elif is_pct_change:
                formatted_y = f"{y_val:+.2f}%"
            else:
                currency_symbol = (
                    "Rate"
                    if is_fx
                    else self._get_currency_symbol(
                        currency_code=self._get_currency_for_symbol(symbol)
                    )
                )
                formatted_y = f"{currency_symbol} {y_val:,.2f}"

            selection.annotation.set_text(f"{date_str}\n{formatted_y}")

        except Exception as e:
            logging.debug(f"Error formatting intraday tooltip: {e}")

    # --- Initialization Method ---
    def __init__(self):
        """Initializes the main application window, loads config, and sets up UI."""
        # --- IMPORTANT: Call super().__init__() for QMainWindow ---
        super().__init__()
        logging.debug("--- PortfolioApp __init__: START ---")

        # --- ADDED: Setup in-memory log stream for error detection ---
        # This stream captures log messages so they can be inspected programmatically,
        # for example, to detect specific errors from yfinance.
        self.log_stream = StringIO()
        # Get the root logger, which is the ultimate destination for all log messages.
        root_logger = logging.getLogger()
        # Create a handler that will write log records to our StringIO object.
        string_io_handler = logging.StreamHandler(self.log_stream)
        # Optional: You can set a specific format for this handler if needed.
        # formatter = logging.Formatter('%(asctime)s [%(levelname)-8s] - %(message)s')
        # string_io_handler.setFormatter(formatter)
        # Add the new handler to the root logger.
        root_logger.addHandler(string_io_handler)
        # logging.info("In-memory log stream configured.")
        # --- END ADDED ---

        # --- Early Initialization of Themed QColor Attributes with Light Theme Defaults ---
        # These will be updated by _create_status_bar and apply_theme later.
        self.QCOLOR_GAIN_THEMED = QColor(config.COLOR_GAIN)
        self.QCOLOR_LOSS_THEMED = QColor(config.COLOR_LOSS)
        self.QCOLOR_TEXT_PRIMARY_THEMED = QColor(config.COLOR_TEXT_DARK)
        self.QCOLOR_TEXT_SECONDARY_THEMED = QColor(config.COLOR_TEXT_SECONDARY)
        self.QCOLOR_BACKGROUND_THEMED = QColor(config.COLOR_BG_DARK)
        self.QCOLOR_HEADER_BACKGROUND_THEMED = QColor(config.COLOR_BG_HEADER_LIGHT)
        self.QCOLOR_BORDER_THEMED = QColor(config.COLOR_BORDER_LIGHT)  # Initialize here
        self.QCOLOR_ACCENT_THEMED = QColor(config.COLOR_ACCENT_TEAL)
        self.QCOLOR_COLLAPSED_HEADER_BACKGROUND_THEMED = QColor(
            "#343a40"
        )  # A slightly different dark shade
        self.QCOLOR_INPUT_BACKGROUND_THEMED = QColor(config.COLOR_BG_DARK)
        self.QCOLOR_INPUT_TEXT_THEMED = QColor(config.COLOR_TEXT_DARK)
        self.QCOLOR_TABLE_ALT_ROW_THEMED = QColor("#fAfBff")
        logging.debug("Early initialization of QCOLOR_*_THEMED attributes completed.")

        # --- Path Initialization for DB and Config ---
        self.DB_FILE_PATH: Optional[str] = (
            None  # Will be set by get_database_path via load_config or direct call
        )
        self.CONFIG_FILE: Optional[str] = None
        self.MANUAL_OVERRIDES_FILE: Optional[str] = (
            None  # Will be set based on AppDataLocation
        )

        try:
            app_data_path = QStandardPaths.writableLocation(
                QStandardPaths.AppDataLocation
            )
            if (
                not app_data_path
            ):  # Fallback if AppDataLocation (often includes AppName) is empty
                app_config_path = QStandardPaths.writableLocation(
                    QStandardPaths.AppConfigLocation
                )
                if app_config_path:  # AppConfigLocation might also be app-specific
                    app_data_path = app_config_path
                else:  # Further fallback to user's home if both standard paths fail
                    logging.warning(
                        "QStandardPaths AppDataLocation and AppConfigLocation failed. Using user home directory."
                    )
                    # For home dir, we might want to explicitly create an app-named subfolder
                    home_dir = os.path.expanduser("~")
                    app_data_path = os.path.join(
                        home_dir, f".{config.APP_NAME.lower()}"
                    )  # e.g., ~/.investa

            if app_data_path:
                os.makedirs(app_data_path, exist_ok=True)  # Ensure directory exists
                self.CONFIG_FILE = os.path.join(app_data_path, "gui_config.json")
                self.MANUAL_OVERRIDES_FILE = os.path.join(
                    app_data_path, MANUAL_OVERRIDES_FILENAME
                )
            else:  # Last resort if all path finding fails (should be rare)
                logging.error(
                    "CRITICAL: Could not determine a writable application data directory. Using current working directory for config/overrides."
                )
                self.CONFIG_FILE = "gui_config.json"
                self.MANUAL_OVERRIDES_FILE = MANUAL_OVERRIDES_FILENAME
        except Exception as e_path_init:
            logging.exception(
                f"CRITICAL ERROR during config/manual file path initialization: {e_path_init}"
            )
            self.CONFIG_FILE = "gui_config.json"  # Fallback
            self.MANUAL_OVERRIDES_FILE = MANUAL_OVERRIDES_FILENAME  # Fallback
            QMessageBox.critical(
                self,
                "Path Error",
                f"Could not set up application paths for config files.\nError: {e_path_init}",
            )

        # DB_FILE_PATH is determined by load_config or set if a new DB is created/opened.
        # It will also use get_database_path() which has similar fallback logic.

        # logging.info(f"CONFIG_FILE path determined as: {self.CONFIG_FILE}")
        logging.info(
            f"MANUAL_OVERRIDES_FILE path determined as: {self.MANUAL_OVERRIDES_FILE}"
        )
        # DB_FILE_PATH will be logged after load_config
        # --- End Path Initialization ---

        self.app_font = QFont("Arial", 9)
        self.setFont(self.app_font)
        logging.info(
            f"Application base font set to: {self.app_font.family()} ({self.app_font.pointSize()}pt)"
        )
        self.base_window_title = "Investa Portfolio Dashboard"
        self.setWindowTitle(self.base_window_title)
        self.internal_to_yf_map = {}
        self.table_filter_timer = QTimer(self)
        self.table_filter_timer.setSingleShot(True)
        self.tx_filter_timer = QTimer(self)  # Timer for transactions tab filter
        self.tx_filter_timer.setSingleShot(True)
        self._initial_file_selection = False  # Used by select_database_file now
        self.worker_signals = WorkerSignals()
        self.market_data_provider = (
            MarketDataProvider()
        )  # Initialize MarketDataProvider

        # --- Configuration Loading ---
        self.config = (
            self.load_config()
        )  # This will set self.DB_FILE_PATH from config or default
        self.current_theme = self.config.get("theme", "light")  # Load theme preference
        # logging.info(f"DB_FILE_PATH set from/to config: {self.DB_FILE_PATH}")
        # logging.info(f"Initial theme set to: {self.current_theme}")

        # --- Initialize DB Connection ---
        # initialize_database will create the DB file and tables if they don't exist at self.DB_FILE_PATH
        self.db_conn = initialize_database(self.DB_FILE_PATH)
        if not self.db_conn:
            QMessageBox.critical(
                self,
                "Database Error",
                f"Could not initialize or connect to the database at:\n{self.DB_FILE_PATH}\n\nThe application may not function correctly.",
            )
        else:
            # logging.info(f"Database connection established: {self.DB_FILE_PATH}")
            self.setWindowTitle(
                f"{self.base_window_title} - {os.path.basename(self.DB_FILE_PATH)}"
            )

        # --- Load Manual Overrides & User Symbol Settings ---
        self.manual_overrides_dict: Dict[str, Dict[str, Any]] = {}
        self.user_symbol_map_config: Dict[str, str] = {}
        self.user_excluded_symbols_config: Set[str] = set()
        self._load_manual_overrides()  # Populates the above attributes

        self.fmp_api_key = self.config.get(
            "fmp_api_key",
            config.DEFAULT_API_KEY if hasattr(config, "DEFAULT_API_KEY") else "",
        )

        self.is_calculating = False
        self.last_calc_status = ""
        self.last_hist_twr_factor = np.nan
        self._is_setting_dates_programmatically = False
        # --- ADDED: Attribute to track first data load for header state restoration ---
        self._first_data_load_complete = False
        # --- END ADDED ---

        # --- ADDED: Attribute for frozen table view ---
        self.frozen_table_view = None
        # --- END ADDED ---
        self.holdings_data = pd.DataFrame()  # Initialize as empty DataFrame
        self.ignored_data = (
            pd.DataFrame()
            # This will store rows from DB load issues or processing issues
        )  # This will store rows from DB load issues or processing issues
        self.summary_metrics_data = {}
        self.account_metrics_data = {}
        self.historical_data = pd.DataFrame()
        self.index_quote_data: Dict[str, Dict[str, Any]] = {}
        self.full_historical_data = pd.DataFrame()
        self.periodic_returns_data: Dict[str, pd.DataFrame] = {}
        self.periodic_value_changes_data: Dict[str, pd.DataFrame] = (
            {}
        )  # ADDED for absolute value changes
        self.dividend_history_data = pd.DataFrame()
        self.historical_prices_yf_adjusted: Dict[str, pd.DataFrame] = {}
        self.historical_fx_yf: Dict[str, pd.DataFrame] = {}

        # Advanced Analysis Tab Attributes
        self.correlation_fig = None
        self.correlation_canvas = None
        self.correlation_table = None
        self.factor_model_combo = None
        self.run_factor_analysis_button = None
        self.factor_analysis_results_text = None
        self.scenario_input_line_edit = None
        self.run_scenario_button = None
        self.scenario_impact_label = None

        # Data attributes for Advanced Analysis
        self.correlation_matrix_df = pd.DataFrame()
        self.factor_analysis_results = None  # Store the regression summary object
        self.scenario_analysis_result = {}
        self.available_accounts: List[str] = []
        self.selected_accounts: List[str] = self.config.get("selected_accounts", [])
        self.selected_benchmarks = self.config.get(
            "graph_benchmarks", DEFAULT_GRAPH_BENCHMARKS
        )
        if not isinstance(self.selected_benchmarks, list) or not all(
            isinstance(item, str) for item in self.selected_benchmarks
        ):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS
        elif not self.selected_benchmarks and BENCHMARK_OPTIONS_DISPLAY:
            self.selected_benchmarks = (
                [BENCHMARK_OPTIONS_DISPLAY[0]] if BENCHMARK_OPTIONS_DISPLAY else []
            )

        self.column_visibility: Dict[str, bool] = self.config.get(
            "column_visibility", {}
        )
        self.group_expansion_states: Dict[str, bool] = {}
        # --- ADDED: Track account scope to reset grouping state ---
        # This is crucial to fix the bug where groups remain collapsed after changing accounts.
        self.last_selected_accounts_for_grouping: List[str] = (
            self.selected_accounts.copy()
        )
        # --- END ADDED ---
        self._ensure_all_columns_in_visibility()
        self.threadpool = QThreadPool()
        # logging.info(f"Max threads: {self.threadpool.maxThreadCount()}")

        # This will store the full, cleaned DataFrame loaded from the DB (or migrated CSV)
        # It's the primary source for portfolio_logic functions.
        self.all_transactions_df_cleaned_for_logic = pd.DataFrame()
        self.original_data = (
            pd.DataFrame()
        )  # Holds the most recently loaded full dataset
        # This stores the DataFrame representing the original source data, primarily for context
        # when displaying ignored rows if they originated from a CSV with different headers.

        self._create_status_bar()
        # If data is purely from DB, this might be similar/identical to all_transactions_df_cleaned_for_logic.
        self.original_transactions_df_for_ignored_context = pd.DataFrame()
        self.original_to_cleaned_header_map_from_csv: Dict[str, str] = (
            {}
        )  # Only relevant if CSV was imported

        logging.debug("--- PortfolioApp __init__: Before UI Structure/Widgets Init ---")
        
        # Ensure heavy GUI libs are loaded before widget creation
        _ensure_matplotlib()
        
        self._init_ui_structure()
        logging.debug("--- PortfolioApp __init__: After _init_ui_structure ---")
        
        self._init_ui_widgets()
        logging.debug("--- PortfolioApp __init__: After _init_ui_widgets ---")
        self._init_menu_bar()
        self._init_toolbar()
        self._connect_signals()
        self._apply_initial_styles_and_updates()  # This used to call self.apply_styles()

        # --- Initial Theme Application ---
        # apply_theme will also call save_config if triggered by user, but not here during init.
        # We set _config_already_saved_for_theme to bypass save_config inside apply_theme during init.
        self._config_already_saved_for_theme = True
        if hasattr(self, "apply_theme"):  # Check if method exists before calling
            self.apply_theme(self.current_theme)
            if self.current_theme == "light":
                if hasattr(self, "light_theme_action"):
                    self.light_theme_action.setChecked(True)
            elif hasattr(self, "dark_theme_action"):  # Ensure dark_theme_action exists
                self.dark_theme_action.setChecked(True)
        else:
            logging.error(
                "apply_theme method not found during __init__ post theme setup."
            )
        if hasattr(self, "_config_already_saved_for_theme"):  # Clean up temp attr
            del self._config_already_saved_for_theme
        # --- End Initial Theme Application ---

        # --- Initial Data Load / Migration Check ---
        if self.db_conn and self.config.get("load_on_startup", True):
            # Path to a CSV that *might* exist for migration if DB is new/empty
            csv_for_potential_migration = self.config.get(
                "transactions_file_csv_fallback", DEFAULT_CSV
            )
            logging.info(
                f"Startup: Checking DB ({self.DB_FILE_PATH}) and potential CSV for migration ({csv_for_potential_migration})"
            )

            if check_if_db_empty_and_csv_exists(
                self.db_conn, csv_for_potential_migration
            ):
                # logging.info("Startup: DB is empty and migration CSV exists.")
                reply = QMessageBox.question(
                    self,
                    "Migrate CSV to Database?",
                    f"The Investa database is currently empty. "
                    f"An existing CSV transaction file was found or configured:\n\n"
                    f"<b>{os.path.basename(csv_for_potential_migration)}</b>\n\n"
                    f"Would you like to import transactions from this CSV file into the new database?\n\n"
                    f"Database: {os.path.basename(self.DB_FILE_PATH)}",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes,
                )
                if reply == QMessageBox.Yes:
                    self._run_csv_migration(
                        csv_for_potential_migration
                    )  # This will also call refresh_data
                else:
                    self.set_status(
                        "Ready. Add transactions or import from CSV via File menu."
                    )
                    self._perform_initial_load_from_db_only()  # Attempt to load from (now confirmed empty) DB
            else:  # DB not empty or CSV doesn't exist for migration
                logging.info(
                    "Startup: DB not empty or no migration CSV found. Loading from DB directly."
                )
                self._perform_initial_load_from_db_only()
        elif not self.db_conn:
            self.set_status("Error: Database connection failed. Check logs.")
        else:  # Not loading on startup
            self.set_status("Ready. Click Refresh or Import CSV via File menu.")
            self._update_table_view_with_filtered_columns(pd.DataFrame())
            self.apply_column_visibility()
            self.update_performance_graphs(initial=True)
            self._update_account_button_text()
            self._update_table_title()
        logging.debug("--- PortfolioApp __init__: END ---")

    # --- Theme Application Method ---
    def apply_theme(
        self, theme_name: str
    ):  # Removed _config_already_saved_for_theme from signature
        """Applies the selected theme (light or dark) to the application."""
        # logging.info(f"Applying theme: {theme_name}")

        if (
            hasattr(self, "current_theme")
            and self.current_theme == theme_name
            and hasattr(self, "_style_sheet_applied_once")
        ):
            logging.debug(
                f"Theme '{theme_name}' is already applied and stylesheet was set. Skipping full re-application of QSS."
            )
            # If triggered by user action, still save config.
            if (
                hasattr(self, "light_theme_action")
                and hasattr(self, "dark_theme_action")
                and self.sender()
                in [
                    self.light_theme_action,
                    self.dark_theme_action,
                ]  # Only save if user action
            ):
                self.config["theme"] = theme_name
                self.save_config()
            return

        # --- Update Toolbar Icons based on Theme ---
        for action_attr, specs_by_theme in PortfolioApp.TOOLBAR_ICON_SPECS.items():
            action_widget = getattr(self, action_attr, None)
            if not action_widget:
                logging.debug(
                    f"Action attribute '{action_attr}' not found, skipping icon update."
                )
                continue

            # Default to light theme spec if current theme's spec is missing
            spec_to_use = specs_by_theme.get(theme_name, specs_by_theme["light"])
            icon_type, primary_spec, *fallback_spec_tuple = spec_to_use
            fallback_spec = fallback_spec_tuple[0] if fallback_spec_tuple else None

            icon = QIcon()  # Start with a null icon
            if icon_type == "theme":
                icon = QIcon.fromTheme(primary_spec)
                if icon.isNull() and fallback_spec:
                    icon = self.style().standardIcon(fallback_spec)
            elif icon_type == "sp":
                icon = self.style().standardIcon(primary_spec)
                if icon.isNull() and fallback_spec:
                    icon = self.style().standardIcon(fallback_spec)

            if icon.isNull():
                logging.warning(
                    f"Icon for action '{action_attr}' (theme: {theme_name}, spec: {spec_to_use}) is NULL. Action text: '{action_widget.text()}'"
                )
                # Set to a truly empty icon to clear any previous one if it's null
                action_widget.setIcon(QIcon())
            else:
                action_widget.setIcon(icon)

        # Handle lookup_button (QPushButton) separately
        if hasattr(self, "lookup_button"):
            spec_lookup = PortfolioApp.LOOKUP_BUTTON_ICON_SPECS.get(
                theme_name, PortfolioApp.LOOKUP_BUTTON_ICON_SPECS["light"]
            )
            icon_type_lookup, primary_spec_lookup, *_ = spec_lookup

            icon_lookup = QIcon()
            if icon_type_lookup == "theme":
                icon_lookup = QIcon.fromTheme(primary_spec_lookup)
            elif icon_type_lookup == "sp":
                icon_lookup = self.style().standardIcon(primary_spec_lookup)

            if icon_lookup.isNull():
                logging.warning(
                    f"Icon for lookup_button (theme: {theme_name}, spec: {spec_lookup}) is NULL."
                )
                self.lookup_button.setIcon(QIcon())  # Clear icon
            else:
                self.lookup_button.setIcon(icon_lookup)
        # --- End Toolbar Icon Update ---

        self.current_theme = theme_name
        self._style_sheet_applied_once = (
            True  # Mark that a stylesheet has been applied at least once
        )

        qss_file = "gui/style.qss" if theme_name == "light" else "gui/style_dark.qss"

        try:
            qss_path = resource_path(qss_file)
            if not os.path.exists(qss_path):
                logging.error(f"Stylesheet file NOT FOUND at: {qss_path}")
                self.setStyleSheet("")  # Fallback to default Qt style
                QMessageBox.warning(
                    self,
                    "Theme Error",
                    f"Stylesheet for {theme_name} theme not found. Using default style.",
                )
                return

            with open(qss_path, "r", encoding="utf-8") as f:  # Added encoding
                style_sheet_content = f.read()
                self.setStyleSheet(style_sheet_content)
            # logging.info(f"Successfully applied stylesheet: {qss_file}")

        except Exception as e:
            logging.error(
                f"ERROR applying stylesheet {qss_file}: {e}", exc_info=True
            )  # Added exc_info
            self.setStyleSheet("")  # Fallback
            QMessageBox.warning(
                self,
                "Theme Error",
                f"Could not load {theme_name} theme. Using default style.\nError: {e}",
            )

        # Update themed QColor objects
        if theme_name == "dark":
            self.QCOLOR_GAIN_THEMED = QColor(config.DARK_COLOR_GAIN)
            self.QCOLOR_LOSS_THEMED = QColor(config.DARK_COLOR_LOSS)
            self.QCOLOR_TEXT_PRIMARY_THEMED = QColor(config.DARK_COLOR_TEXT_DARK)
            self.QCOLOR_TEXT_SECONDARY_THEMED = QColor(config.DARK_COLOR_TEXT_LIGHT)
            self.QCOLOR_BACKGROUND_THEMED = QColor(config.DARK_COLOR_BG_DARK)
            self.QCOLOR_HEADER_BACKGROUND_THEMED = QColor(config.DARK_COLOR_BG_LIGHT)
            self.QCOLOR_BORDER_THEMED = QColor(config.DARK_COLOR_BORDER)
            self.QCOLOR_ACCENT_THEMED = QColor(
                config.DARK_COLOR_ACCENT_PRIMARY
            )  # Using primary accent
            self.QCOLOR_INPUT_BACKGROUND_THEMED = QColor(config.DARK_COLOR_INPUT_BG)
            self.QCOLOR_INPUT_TEXT_THEMED = QColor(config.DARK_COLOR_INPUT_TEXT)
            self.QCOLOR_TABLE_ALT_ROW_THEMED = QColor(config.DARK_COLOR_TABLE_ALT_ROW)

            # Matplotlib dark theme settings
            plt.style.use("dark_background")  # Use a predefined dark style as a base
            # Override specific rcParams for dark theme
            # (dark_background style already sets many of these)

            plt.rcParams.update(
                {
                    "axes.labelcolor": config.DARK_COLOR_TEXT_LIGHT,
                    "xtick.color": config.DARK_COLOR_TEXT_LIGHT,
                    "ytick.color": config.DARK_COLOR_TEXT_LIGHT,
                    "text.color": config.DARK_COLOR_TEXT_DARK,
                    "axes.edgecolor": config.DARK_COLOR_BORDER,
                    "figure.facecolor": config.DARK_COLOR_BG_DARK,
                    "axes.facecolor": config.DARK_COLOR_BG_DARK,
                    "savefig.facecolor": config.DARK_COLOR_BG_DARK,
                    "savefig.edgecolor": config.DARK_COLOR_BG_DARK,
                }
            )
        else:  # Light theme
            self.QCOLOR_GAIN_THEMED = QColor(config.COLOR_GAIN)
            self.QCOLOR_LOSS_THEMED = QColor(config.COLOR_LOSS)
            self.QCOLOR_TEXT_PRIMARY_THEMED = QColor(config.COLOR_TEXT_DARK)
            self.QCOLOR_TEXT_SECONDARY_THEMED = QColor(config.COLOR_TEXT_SECONDARY)
            self.QCOLOR_BACKGROUND_THEMED = QColor(config.COLOR_BG_DARK)
            self.QCOLOR_HEADER_BACKGROUND_THEMED = QColor(config.COLOR_BG_HEADER_LIGHT)
            self.QCOLOR_BORDER_THEMED = QColor(config.COLOR_BORDER_LIGHT)
            self.QCOLOR_ACCENT_THEMED = QColor(config.COLOR_ACCENT_TEAL)
            self.QCOLOR_COLLAPSED_HEADER_BACKGROUND_THEMED = QColor(
                "#E9ECEF"
            )  # A slightly darker light gray
            self.QCOLOR_INPUT_BACKGROUND_THEMED = QColor(
                config.COLOR_BG_DARK
            )  # White for light theme inputs
            self.QCOLOR_INPUT_TEXT_THEMED = QColor(
                config.COLOR_TEXT_DARK
            )  # Dark text for light theme inputs
            # Matplotlib light theme settings
            plt.style.use("default")  # Revert to Matplotlib default
            self.QCOLOR_TABLE_ALT_ROW_THEMED = QColor("#fAfBff")

            plt.rcParams.update(
                {
                    "axes.labelcolor": config.COLOR_TEXT_DARK,
                    "xtick.color": config.COLOR_TEXT_SECONDARY,
                    "ytick.color": config.COLOR_TEXT_SECONDARY,
                    "text.color": config.COLOR_TEXT_DARK,
                    "axes.edgecolor": config.COLOR_BORDER_DARK,  # Darker border for light theme charts
                    "figure.facecolor": config.COLOR_BG_DARK,  # White
                    "axes.facecolor": config.COLOR_BG_DARK,  # White
                    "savefig.facecolor": "white",
                    "savefig.edgecolor": "white",
                }
            )
        logging.debug(f"Matplotlib rcParams updated for {theme_name} theme.")
        logging.debug(f"Themed QColors updated for {theme_name}.")
        # Explicit QSpinBox theming removed, will be handled by style_dark.qss

        # Explicitly theme table viewports and tables
        table_view_attr_names = [
            "table_view",  # Main holdings table
            "stock_transactions_table_view",
            "cash_transactions_table_view",  # Log tab
            "dividend_table_view",
            "dividend_summary_table_view",  # Dividend tab
            "cg_table_view",
            "cg_summary_table_view",  # Capital Gains tab
        ]

        # Update the delegate with the new theme
        if hasattr(self, "table_view"):
            self.table_view.setItemDelegate(
                GroupHeaderDelegate(self.table_view, theme=self.current_theme)
            )

        for attr_name in table_view_attr_names:
            if hasattr(self, attr_name):
                table_view_widget = getattr(self, attr_name)
                if table_view_widget:  # Ensure the widget exists
                    # Theme the viewport (the actual scrollable area where items are drawn)
                    viewport = table_view_widget.viewport()
                    if viewport:
                        logging.debug(f"Theming viewport for {attr_name}")  # DEBUG
                        vp_palette = viewport.palette()
                        vp_palette.setColor(
                            QPalette.Base, self.QCOLOR_BACKGROUND_THEMED
                        )
                        vp_palette.setColor(
                            QPalette.Window, self.QCOLOR_BACKGROUND_THEMED
                        )
                        vp_palette.setColor(
                            QPalette.Text, self.QCOLOR_TEXT_PRIMARY_THEMED
                        )
                        viewport.setPalette(vp_palette)
                        viewport.setAutoFillBackground(True)

                    # Theme the table view itself
                    table_palette = table_view_widget.palette()
                    logging.debug(f"Theming table view {attr_name}")  # DEBUG
                    table_palette.setColor(QPalette.Base, self.QCOLOR_BACKGROUND_THEMED)
                    table_palette.setColor(
                        QPalette.Text, self.QCOLOR_TEXT_PRIMARY_THEMED
                    )
                    table_palette.setColor(
                        QPalette.AlternateBase, self.QCOLOR_TABLE_ALT_ROW_THEMED
                    )
                    table_view_widget.setPalette(table_palette)
                    table_view_widget.setAutoFillBackground(
                        True
                    )  # Important for palette to take effect
                    table_view_widget.setAlternatingRowColors(
                        True
                    )  # Ensure alternating rows are active

        # Refresh UI elements
        if hasattr(self, "table_model"):
            self.table_model.layoutChanged.emit()
        if hasattr(self, "stock_transactions_table_model"):
            self.stock_transactions_table_model.layoutChanged.emit()
        if hasattr(self, "cash_transactions_table_model"):
            self.cash_transactions_table_model.layoutChanged.emit()
        if hasattr(self, "dividend_table_model"):
            self.dividend_table_model.layoutChanged.emit()
        if hasattr(self, "dividend_summary_table_model"):
            self.dividend_summary_table_model.layoutChanged.emit()
        if hasattr(self, "cg_table_model"):
            self.cg_table_model.layoutChanged.emit()
        if hasattr(self, "cg_summary_table_model"):
            self.cg_summary_table_model.layoutChanged.emit()
        # IgnoredLogTable model is created dynamically, so it will pick up new theme when next opened.

        self.update_performance_graphs(
            initial=False
        )  # initial=False to redraw with current data
        self._update_periodic_bar_charts()
        self._update_dividend_bar_chart()
        self._update_capital_gains_display()  # This calls its own chart and table updates

        # Pie charts and asset allocation charts might need data, pass filtered data if available
        df_display_filtered_for_pies = (
            self._get_filtered_data(group_by_sector=False)
            if hasattr(self, "_get_filtered_data")
            else pd.DataFrame()
        )
        self.update_account_pie_chart(df_display_filtered_for_pies)
        self.update_holdings_pie_chart(df_display_filtered_for_pies)
        self._update_asset_allocation_charts()

        self.update_dashboard_summary(
            df_display_filtered_for_pies
        )  # Re-apply palette colors to summary labels
        self.update_header_info()  # Re-color header text

        self._update_periodic_value_change_display()  # Update new tab
        # logging.info(f"UI components refreshed for {theme_name} theme.")

        # Save the theme preference if triggered by menu action
        if hasattr(self, "light_theme_action") and self.sender() in [
            self.light_theme_action,
            self.dark_theme_action,
        ]:  # Only save if user action
            self.config["theme"] = theme_name
            self.save_config()

    # --- Benchmark Selection Methods (Define BEFORE initUI) ---
    def _update_benchmark_button_text(self):
        """Updates the benchmark selection button text to reflect current selections."""
        # Ensure selected_benchmarks attribute exists
        if not hasattr(self, "selected_benchmarks"):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS  # Initialize if missing

        if not self.selected_benchmarks:
            text = "Select Benchmarks"
        elif len(self.selected_benchmarks) == 1:
            text = f"Bench: {self.selected_benchmarks[0]}"  # This is now a display name
        else:
            # Show count and first few benchmarks
            text = f"Bench ({len(self.selected_benchmarks)}): {', '.join(self.selected_benchmarks[:2])}"
            if len(self.selected_benchmarks) > 2:
                text += ", ..."

        # Ensure benchmark_select_button exists before setting text
        if hasattr(self, "benchmark_select_button") and self.benchmark_select_button:
            self.benchmark_select_button.setText(text)
            # Set a tooltip showing all selected benchmarks
            tooltip_text = f"Selected: {', '.join(self.selected_benchmarks) if self.selected_benchmarks else 'None'}\nClick to change."
            self.benchmark_select_button.setToolTip(tooltip_text)

    @Slot(str, bool)
    def toggle_benchmark_selection(self, display_name: str, is_checked: bool):
        """
        Adds or removes a benchmark symbol from the selected list.

        Updates `self.selected_benchmarks` and the button text. Does NOT trigger
        an immediate data refresh; user must click 'Update Graphs'.

        Args:
            symbol (str): The benchmark symbol (e.g., "SPY") being toggled.
            is_checked (bool): True if the benchmark is being selected, False if deselected.
        """  # display_name is the user-friendly name
        # ... (IMPORTANT: This should NOT call refresh_data anymore) ...
        if not hasattr(self, "selected_benchmarks"):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS
        if is_checked:
            if display_name not in self.selected_benchmarks:
                self.selected_benchmarks.append(display_name)
        else:
            if display_name in self.selected_benchmarks:
                self.selected_benchmarks.remove(display_name)
        try:
            self.selected_benchmarks.sort(
                key=lambda b: (
                    BENCHMARK_OPTIONS_DISPLAY.index(
                        b
                    )  # Sort by the order in BENCHMARK_OPTIONS_DISPLAY
                    if b in BENCHMARK_OPTIONS_DISPLAY
                    else float("inf")
                )
            )
        except ValueError:
            logging.warning("Warn: Could not sort benchmarks based on options.")
        self._update_benchmark_button_text()
        # Inform user to click Update Graphs
        self.set_status("Benchmark selection changed. Click 'Update Graphs' to apply.")

    @Slot()  # Add Slot decorator if used with signals/slots consistently
    def show_benchmark_selection_menu(self):
        """Displays a context menu with checkable actions for benchmark selection."""
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())  # Apply style

        # Create a checkable action for each available benchmark option
        self._build_benchmark_menu_actions(menu)

        # Display the menu just below the benchmark selection button
        # Ensure benchmark_select_button exists before mapping position
        if hasattr(self, "benchmark_select_button") and self.benchmark_select_button:
            self._exec_menu_below_widget(self.benchmark_select_button, menu)
        else:
            logging.warning("Warn: Benchmark button not ready for menu.")

    # --- NEW: Account Selection Methods ---
    def _update_account_button_text(self):
        """Updates the account selection button text based on available and selected accounts."""
        button = getattr(self, "account_select_button", None)
        if not button:
            return

        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)

        if not self.available_accounts:
            text = "Accounts (N/A)"
            tooltip_text = "Load data to see accounts."
        elif not self.selected_accounts or num_selected == num_available:
            text = f"Accounts ({num_available}/{num_available})"
            tooltip_text = "All accounts selected.\nClick to change."
        elif num_selected == 1:
            text = f"Account: {self.selected_accounts[0]}"
            tooltip_text = f"Selected: {self.selected_accounts[0]}\nClick to change."
        else:
            text = f"Accounts ({num_selected}/{num_available})"
            tooltip_text = (
                f"Selected: {', '.join(self.selected_accounts)}\nClick to change."
            )

        button.setText(text)
        button.setToolTip(tooltip_text)

    @Slot(str, bool)
    def toggle_account_selection(self, account_name: str, is_checked: bool):
        """
        Adds or removes an account from the selected list.

        Updates `self.selected_accounts` and the button text. Handles the case where
        the selection becomes empty (defaults back to all). Does NOT trigger an
        immediate data refresh; user must click 'Update Accounts'.

        Args:
            account_name (str): The account name being toggled.
            is_checked (bool): True if the account is being selected, False if deselected.
        """
        if not hasattr(self, "selected_accounts"):
            self.selected_accounts = []
        if not hasattr(self, "available_accounts"):
            self.available_accounts = []

        if is_checked:
            if account_name not in self.selected_accounts:
                self.selected_accounts.append(account_name)
                # logging.info(f"Account added: {account_name}.")
        else:
            if account_name in self.selected_accounts:
                self.selected_accounts.remove(account_name)
                # logging.info(f"Account removed: {account_name}.")

        # If selection becomes empty, default back to selecting all available accounts
        if not self.selected_accounts and self.available_accounts:
            logging.warning(
                "Warn: No accounts selected. Defaulting to all available accounts."
            )
            self.selected_accounts = self.available_accounts.copy()
            # We need to re-check the menu items visually if the menu is open,
            # but since the menu closes on action, just updating the state is enough.

        # Sort selected list based on available accounts order for consistency
        self.selected_accounts.sort(
            key=lambda acc: (
                self.available_accounts.index(acc)
                if acc in self.available_accounts
                else float("inf")
            )
        )

        # logging.info(f"Selected Accounts: {self.selected_accounts}")
        self._update_account_button_text()

        # Inform user to click Update
        self.set_status("Account selection changed. Click 'Update Accounts' to apply.")

    @Slot()
    def show_account_selection_menu(self):
        """Displays a context menu with checkable actions for account selection."""
        if not self.available_accounts:
            self.show_info(
                "Load transaction data first to see available accounts.",
                popup=True,
                title="No Accounts",
            )
            return

        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())

        # Populate menu actions via helper
        self._build_account_menu_actions(menu)

        button = getattr(self, "account_select_button", None)
        if button:
            self._exec_menu_below_widget(button, menu)
        else:
            logging.warning("Warn: Account button not ready for menu.")

    @Slot()
    def manage_account_groups(self):
        """Opens the AccountGroupingDialog to manage account groups."""
        current_groups = self.config.get("account_groups", {})
        all_accounts = self.available_accounts

        dialog = AccountGroupingDialog(
            parent=self, current_groups=current_groups, all_accounts=all_accounts
        )
        if dialog.exec():
            new_groups = dialog.get_settings()
            if new_groups != current_groups:
                self.config["account_groups"] = new_groups
                self.save_config()
                self.set_status(
                    "Account groups updated. Re-open the account menu to see changes."
                )
                # No refresh needed, as this only affects the UI menu structure
            else:
                self.set_status("Account groups unchanged.")

    def _toggle_group_selection(self, group_name: str, select_all: bool):
        """Selects or deselects all accounts within a specific group."""
        account_groups = self.config.get("account_groups", {})
        accounts_in_group = account_groups.get(group_name, [])

        if not accounts_in_group:
            return

        if select_all:
            for acc in accounts_in_group:
                if acc not in self.selected_accounts:
                    self.selected_accounts.append(acc)
            # logging.info(f"Selected all accounts in group '{group_name}'.")
        else:  # Deselect all
            self.selected_accounts = [
                acc for acc in self.selected_accounts if acc not in accounts_in_group
            ]
            # logging.info(f"Deselected all accounts in group '{group_name}'.")

        # If selection becomes empty, default back to selecting all available accounts
        if not self.selected_accounts and self.available_accounts:
            self.selected_accounts = self.available_accounts.copy()

        # Sort selected list based on available accounts order for consistency
        self.selected_accounts.sort(
            key=lambda acc: (
                self.available_accounts.index(acc)
                if acc in self.available_accounts
                else float("inf")
            )
        )

        self._update_account_button_text()
        self.set_status("Account selection changed. Click 'Update Accounts' to apply.")

    def _toggle_all_accounts(self, select_all: bool):
        """
        Selects or deselects all available accounts.

        Args:
            select_all (bool): True to select all, False to deselect all.
        """
        if select_all:
            self.selected_accounts = self.available_accounts.copy()
            # logging.info("All accounts selected.")
        else:
            # Prevent deselecting all - keep at least one if possible, or default back to all later
            # For simplicity now, allow deselecting all, toggle_account_selection will handle the empty case.
            self.selected_accounts = []
            logging.info(
                "All accounts deselected (will default back to all on next action if empty)."
            )

        # Update button text
        self._update_account_button_text()

        # Inform user to click Update
        self.set_status("Account selection changed. Click 'Update Accounts' to apply.")

    # --- End Account Selection Methods ---

    def _build_account_menu_actions(self, menu: QMenu):
        """
        Builds the account selection menu, including groups as sub-menus.
        Overrides the method from UiHelpersMixin.
        """
        account_groups = self.config.get("account_groups", {})
        all_available_accounts = set(self.available_accounts)
        grouped_accounts = set()

        # --- "Select All" / "Deselect All" Actions ---
        select_all_action = QAction("Select All", self)
        select_all_action.triggered.connect(lambda: self._toggle_all_accounts(True))
        menu.addAction(select_all_action)

        deselect_all_action = QAction("Deselect All", self)
        deselect_all_action.triggered.connect(lambda: self._toggle_all_accounts(False))
        menu.addAction(deselect_all_action)
        menu.addSeparator()

        # --- Grouped Accounts (as Sub-menus) ---
        if account_groups:
            for group_name, accounts_in_group in sorted(account_groups.items()):
                valid_accounts_in_group = [
                    acc for acc in accounts_in_group if acc in all_available_accounts
                ]
                if not valid_accounts_in_group:
                    continue

                group_menu = menu.addMenu(f"📂 {group_name}")
                grouped_accounts.update(valid_accounts_in_group)

                # Add "Select All in Group" action
                select_group_action = QAction(f"Select All in '{group_name}'", self)
                select_group_action.triggered.connect(
                    partial(self._toggle_group_selection, group_name, True)
                )
                group_menu.addAction(select_group_action)
                group_menu.addSeparator()

                for account_name in sorted(valid_accounts_in_group):
                    action = QAction(account_name, self)
                    action.setCheckable(True)
                    action.setChecked(account_name in self.selected_accounts)
                    action.triggered.connect(
                        partial(
                            self.toggle_account_selection,
                            account_name,
                            action.isChecked,
                        )
                    )
                    group_menu.addAction(action)
            menu.addSeparator()

        # --- Ungrouped Accounts ---
        ungrouped_accounts = sorted(list(all_available_accounts - grouped_accounts))
        if ungrouped_accounts:
            for account_name in ungrouped_accounts:
                action = QAction(account_name, self)
                action.setCheckable(True)
                action.setChecked(account_name in self.selected_accounts)
                action.triggered.connect(
                    partial(
                        self.toggle_account_selection, account_name, action.isChecked
                    )
                )
                menu.addAction(action)

        # --- Footer Actions ---
        menu.addSeparator()
        manage_groups_action = QAction("Manage Groups...", self)
        manage_groups_action.triggered.connect(self.manage_account_groups)
        menu.addAction(manage_groups_action)

    def _load_manual_overrides(self):  # No longer returns, sets instance attributes
        """Loads manual overrides, symbol map, and excluded symbols from MANUAL_OVERRIDES_FILE.
        Handles migration from old price-only format if necessary for price overrides.
        """
        # Initialize with defaults from config.py
        self.manual_overrides_dict = {}  # For prices, sectors, etc.
        self.user_symbol_map_config = config.SYMBOL_MAP_TO_YFINANCE.copy()
        self.user_excluded_symbols_config = set(config.YFINANCE_EXCLUDED_SYMBOLS.copy())

        # Try loading new format first
        if os.path.exists(self.MANUAL_OVERRIDES_FILE):
            try:
                with open(
                    self.MANUAL_OVERRIDES_FILE, "r", encoding="utf-8"
                ) as f:  # Use instance attribute
                    loaded_data = json.load(f)
                if isinstance(loaded_data, dict):
                    # Load manual price overrides
                    price_overrides_loaded = loaded_data.get(
                        "manual_price_overrides", {}
                    )
                    if isinstance(price_overrides_loaded, dict):
                        valid_price_overrides: Dict[str, Dict[str, Any]] = {}
                        invalid_count = 0
                        for key, value_dict in price_overrides_loaded.items():
                            if isinstance(key, str) and isinstance(value_dict, dict):
                                entry: Dict[str, Any] = {}
                                price = value_dict.get("price")
                                if (
                                    price is not None
                                    and isinstance(price, (int, float))
                                    and price > 0
                                ):
                                    entry["price"] = float(price)

                                entry["asset_type"] = str(
                                    value_dict.get("asset_type", "")
                                ).strip()
                                entry["sector"] = str(
                                    value_dict.get("sector", "")
                                ).strip()
                                entry["geography"] = str(
                                    value_dict.get("geography", "")
                                ).strip()
                                entry["industry"] = str(
                                    value_dict.get("industry", "")
                                ).strip()
                                valid_price_overrides[key.upper().strip()] = entry
                            else:
                                invalid_count += 1
                                logging.warning(
                                    f"Warn: Invalid manual price override entry skipped: Key='{key}', Value='{value_dict}'"
                                )
                        self.manual_overrides_dict = valid_price_overrides
                        logging.info(
                            f"Loaded {len(self.manual_overrides_dict)} valid price overrides."
                        )
                        if invalid_count > 0:
                            logging.warning(
                                f"Skipped {invalid_count} invalid price override entries."
                            )
                    else:
                        logging.warning(
                            "Manual price overrides in JSON is not a dictionary. Using defaults."
                        )

                    # Load user symbol map
                    user_map_loaded = loaded_data.get("user_symbol_map", {})
                    if isinstance(user_map_loaded, dict) and all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in user_map_loaded.items()
                    ):
                        self.user_symbol_map_config = {
                            k.upper().strip(): v.upper().strip()
                            for k, v in user_map_loaded.items()
                        }
                        logging.info(
                            f"Loaded {len(self.user_symbol_map_config)} user symbol mappings."
                        )
                    else:
                        logging.warning(
                            "User symbol map in JSON is not a valid dictionary. Using defaults."
                        )

                    # Load user excluded symbols
                    user_excluded_loaded = loaded_data.get("user_excluded_symbols", [])
                    if isinstance(user_excluded_loaded, list) and all(
                        isinstance(s, str) for s in user_excluded_loaded
                    ):
                        self.user_excluded_symbols_config = {
                            s.upper().strip() for s in user_excluded_loaded
                        }
                        logging.info(
                            f"Loaded {len(self.user_excluded_symbols_config)} user excluded symbols."
                        )
                    else:
                        logging.warning(
                            "User excluded symbols in JSON is not a valid list. Using defaults."
                        )
                    return  # Successfully loaded from new format

                # --- Migration from old format (if new format load failed or file was old format) ---
                # This part is for migrating old "manual_prices.json" which only had symbol:price
                # If the current MANUAL_OVERRIDES_FILE was an old format, this will try to parse it.
                elif isinstance(loaded_data, dict) and all(
                    isinstance(v, (int, float)) for v in loaded_data.values()
                ):
                    # logging.info("Attempting to migrate old manual price format...")
                    migrated_prices: Dict[str, Dict[str, Any]] = {}
                    for key, price_val in loaded_data.items():
                        if (
                            isinstance(key, str)
                            and isinstance(price_val, (int, float))
                            and price_val > 0
                        ):
                            migrated_prices[key.upper().strip()] = {
                                "price": float(price_val)
                            }
                    if migrated_prices:
                        self.manual_overrides_dict = migrated_prices
                        logging.info(
                            f"Migrated {len(migrated_prices)} entries from old price format. Other settings use defaults."
                        )
                        # Save in new format immediately after migration
                        self._save_manual_overrides_to_json()
                    return
                else:
                    logging.warning(
                        f"Warn: Content of {self.MANUAL_OVERRIDES_FILE} is not a recognized dictionary format. Using defaults."
                    )
            except json.JSONDecodeError as e:
                logging.error(
                    f"Error decoding JSON from {self.MANUAL_OVERRIDES_FILE}: {e}. Using defaults."
                )
            except Exception as e:
                logging.error(
                    f"Error reading {self.MANUAL_OVERRIDES_FILE}: {e}. Using defaults."
                )

        # If new file didn't exist or load failed, and no migration happened, defaults are already set.
        logging.info(
            f"Using default symbol settings (or migrated prices if applicable). "
            f"Overrides: {len(self.manual_overrides_dict)}, Map: {len(self.user_symbol_map_config)}, Excluded: {len(self.user_excluded_symbols_config)}"
        )

    def _save_manual_overrides_to_json(
        self,
    ) -> bool:  # Renamed from _save_manual_prices_to_json
        """Saves all manual settings (prices, symbol map, exclusions) to MANUAL_OVERRIDES_FILE."""
        if (
            not hasattr(self, "manual_overrides_dict")
            or not hasattr(self, "user_symbol_map_config")
            or not hasattr(self, "user_excluded_symbols_config")
        ):
            logging.error("Cannot save symbol settings, dictionary attributes missing.")
            return False

        overrides_file_path = self.MANUAL_OVERRIDES_FILE
        # logging.info(f"Saving symbol settings to: {overrides_file_path}")

        data_to_save = {
            "manual_price_overrides": dict(sorted(self.manual_overrides_dict.items())),
            "user_symbol_map": dict(sorted(self.user_symbol_map_config.items())),
            "user_excluded_symbols": sorted(
                list(self.user_excluded_symbols_config)
            ),  # Save set as sorted list
        }

        try:
            overrides_dir = os.path.dirname(overrides_file_path)
            if overrides_dir:
                os.makedirs(overrides_dir, exist_ok=True)

            with open(overrides_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            # logging.info("Symbol settings saved successfully.")
            return True
        except TypeError as e:
            logging.error(f"TypeError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Data error saving symbol settings:\n{e}"
            )
            return False
        except IOError as e:
            logging.error(f"IOError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not write to file:\n{overrides_file_path}\n{e}",
            )
            return False
        except Exception as e:
            logging.exception("Unexpected error writing symbol settings JSON")
            QMessageBox.critical(
                self,
                "Save Error",
                f"An unexpected error occurred saving symbol settings:\n{e}",
            )
            return False

    def _apply_row_visibility_for_grouping(self):
        """Hides or shows data rows based on the group's expansion state."""
        if not self.group_by_sector_check.isChecked():
            # If grouping is off, ensure all rows are visible
            for i in range(self.table_model.rowCount()):
                # --- ADDED: Also show rows in frozen table ---
                if self.frozen_table_view and self.frozen_table_view.isRowHidden(i):
                    self.frozen_table_view.setRowHidden(i, False)
                # --- END ADDED ---
                if self.table_view.isRowHidden(i):
                    self.table_view.setRowHidden(i, False)
            return

        if "is_group_header" not in self.table_model._data.columns:
            return

        try:
            is_header_col_idx = self.table_model._data.columns.get_loc(
                "is_group_header"
            )
            group_key_col_idx = self.table_model._data.columns.get_loc("group_key")

            for i in range(self.table_model.rowCount()):
                is_header = self.table_model._data.iat[i, is_header_col_idx]
                if pd.notna(is_header) and is_header:
                    continue  # Always show header rows

                group_key = self.table_model._data.iat[i, group_key_col_idx]
                if pd.notna(group_key):
                    is_expanded = self.group_expansion_states.get(group_key, True)
                    if self.table_view.isRowHidden(i) == is_expanded:
                        # --- MODIFIED: Apply to both tables ---
                        self.table_view.setRowHidden(i, not is_expanded)
                        if self.frozen_table_view:
                            self.frozen_table_view.setRowHidden(i, not is_expanded)
                        # --- END MODIFIED ---
        except (KeyError, IndexError) as e:
            logging.error(f"Error applying group visibility: {e}")

    @Slot(QModelIndex)
    def on_table_view_clicked(self, index: QModelIndex):
        """Handles clicks on the table to expand/collapse groups."""
        if not index.isValid() or not self.group_by_sector_check.isChecked():
            return

        row = index.row()
        try:
            is_header = self.table_model._data.iat[
                row, self.table_model._data.columns.get_loc("is_group_header")
            ]
            if pd.notna(is_header) and is_header:
                group_key = self.table_model._data.iat[
                    row, self.table_model._data.columns.get_loc("group_key")
                ]
                current_state = self.group_expansion_states.get(group_key, True)
                self.group_expansion_states[group_key] = not current_state
                self._get_filtered_data(group_by_sector=True, update_view=True)
        except (KeyError, IndexError) as e:
            logging.warning(f"Could not process click for collapse/expand: {e}")

    # --- ADDED: Slots for synchronizing frozen table selection ---
    @Slot(QItemSelection, QItemSelection)
    def _sync_main_to_frozen_selection(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """When the main table selection changes, update the frozen table's selection."""
        if not hasattr(self, "frozen_table_view") or not self.frozen_table_view:
            return
        frozen_model = self.frozen_table_view.selectionModel()
        if not frozen_model:
            return
        frozen_model.blockSignals(True)
        try:
            frozen_model.select(selected, QItemSelectionModel.Select)
            frozen_model.select(deselected, QItemSelectionModel.Deselect)
        finally:
            frozen_model.blockSignals(False)

    @Slot(QItemSelection, QItemSelection)
    def _sync_frozen_to_main_selection(
        self, selected: QItemSelection, deselected: QItemSelection
    ):
        """When the frozen table selection changes, update the main table's selection."""
        if not hasattr(self, "table_view") or not self.table_view:
            return
        main_model = self.table_view.selectionModel()
        if not main_model:
            return
        main_model.blockSignals(True)
        try:
            main_model.select(selected, QItemSelectionModel.Select)
            main_model.select(deselected, QItemSelectionModel.Deselect)
        finally:
            main_model.blockSignals(False)

    # --- Helper Methods (Define BEFORE they are called in __init__) ---
    def _ensure_all_columns_in_visibility(self):
        """
        Ensures self.column_visibility covers all possible UI columns.

        Updates the `self.column_visibility` dictionary to include entries for all
        columns defined by `get_column_definitions` based on the current display
        currency, preserving existing visibility states where possible.
        """
        current_currency = "USD"
        if hasattr(self, "currency_combo") and self.currency_combo:
            current_currency = self.currency_combo.currentText()
        self.all_possible_ui_columns = list(
            get_column_definitions(current_currency).keys()
        )
        current_visibility = self.column_visibility.copy()
        self.column_visibility = {}
        for col_name in self.all_possible_ui_columns:
            is_visible = current_visibility.get(col_name, True)
            if not isinstance(is_visible, bool):
                is_visible = True
            self.column_visibility[col_name] = is_visible

    def _get_currency_symbol(
        self, get_name=False, currency_code=None
    ):  # <-- Add currency_code=None
        """
        Gets the currency symbol (e.g., "$") or 3-letter name (e.g., "USD").

        If currency_code is provided, it returns the symbol/name for that code.
        Otherwise, it uses the currently selected display currency from the UI or config.

        Args:
            get_name (bool, optional): If True, returns the 3-letter currency code.
                                       Defaults to False.
            currency_code (str | None, optional): If provided, get the symbol/name for
                                                  this specific code. Defaults to None.

        Returns:
            str: The currency symbol or name.
        """
        target_currency_code = None
        if currency_code and isinstance(currency_code, str):
            target_currency_code = currency_code.upper()
        else:
            # Fallback to display currency if no code provided
            if (
                hasattr(self, "currency_combo")
                and self.currency_combo
                and self.currency_combo.count() > 0
            ):
                target_currency_code = self.currency_combo.currentText()
            elif hasattr(self, "config"):
                target_currency_code = self.config.get("display_currency", "USD")
            else:
                target_currency_code = "USD"  # Final fallback

        if get_name:
            return target_currency_code  # Return the 3-letter code

        # Map code to symbol
        symbol_map = {
            "USD": "$",
            "THB": "฿",
            "EUR": "€",
            "GBP": "£",
            "JPY": "¥",
            "CAD": "$",
            "AUD": "$",
            "CHF": "Fr",
            "CNY": "¥",
            "HKD": "$",
            "SGD": "$",
        }  # Added more common ones
        return symbol_map.get(
            target_currency_code, target_currency_code
        )  # Return code itself if no symbol mapped

    def _calculate_annualized_twr(self, total_twr_factor, start_date, end_date):
        """
        Calculates the annualized Time-Weighted Return (TWR) percentage.

        Args:
            total_twr_factor (float | np.nan): The total TWR factor (1 + total TWR)
                                               over the period.
            start_date (date | None): The start date of the period.
            end_date (date | None): The end date of the period.

        Returns:
            float | np.nan: The annualized TWR as a percentage, or np.nan if inputs
                            are invalid or calculation fails.
        """
        if pd.isna(total_twr_factor) or total_twr_factor <= 0:
            return np.nan
        if start_date is None or end_date is None or start_date >= end_date:
            return np.nan
        try:
            num_days = (end_date - start_date).days
            if num_days <= 0:
                return np.nan
            annualized_twr_factor = total_twr_factor ** (365.25 / num_days)
            return (annualized_twr_factor - 1) * 100.0
        except (TypeError, ValueError, OverflowError):
            return np.nan

    # --- UI Update Methods (Define BEFORE __init__ calls them) ---

    def update_header_info(
        self, loading=False
    ):  # Keep loading param for now, though logic relies on self.index_quote_data
        """Updates the header label with index quotes or a loading message.

        Formats and displays data for indices specified in `INDICES_FOR_HEADER`
        using data stored in `self.index_quote_data`. Shows price, change, and
        percentage change with appropriate coloring for gains/losses.

        Args:
            loading (bool, optional): If True, displays a "Loading..." message
                                      instead of attempting to show quotes.
                                      Defaults to False.
        """
        if not hasattr(self, "header_info_label") or not self.header_info_label:
            return  # Label not ready

        if loading or not self.index_quote_data:
            self.header_info_label.setText("<i>Loading indices...</i>")
            return

        header_parts = []
        # logging.debug(f"DEBUG update_header_info: Data = {self.index_quote_data}") # Optional debug print
        for index_symbol in INDICES_FOR_HEADER:
            data = self.index_quote_data.get(index_symbol)
            if data and isinstance(data, dict):
                price = data.get("price")
                change = data.get("change")
                # --- FIX: Use 'changesPercentage' which should be decimal/fraction ---
                change_pct_decimal = data.get("changesPercentage")
                name = data.get("name", index_symbol).split(" ")[
                    0
                ]  # Use short name part

                price_str = f"{price:,.2f}" if pd.notna(price) else "N/A"
                change_str = "N/A"
                change_color = COLOR_TEXT_DARK  # Default color

                if pd.notna(change) and pd.notna(change_pct_decimal):
                    change_val = float(change)
                    change_pct_val = float(
                        change_pct_decimal
                    )  # already in decimal form %
                    sign = (
                        "+" if change_val >= -1e-9 else ""
                    )  # Add '+' for positive/zero change
                    change_str = (
                        f"{sign}{change_val:,.2f} ({sign}{change_pct_val:.2f}%)"
                    )
                    if change_val > 1e-9:
                        change_color = COLOR_GAIN
                    elif change_val < -1e-9:
                        change_color = COLOR_LOSS
                    # else: keep default color for zero change
                # --- END FIX ---

                # Use HTML for coloring
                header_parts.append(
                    f"<b>{name}:</b> {price_str} <font color='{change_color}'>{change_str}</font>"
                )
            else:
                # Handle case where index data is missing
                header_parts.append(
                    f"<b>{index_symbol.split('.')[0] if '.' in index_symbol else index_symbol}:</b> N/A"
                )

        if header_parts:
            self.header_info_label.setText(" | ".join(header_parts))
            # logging.debug(f"DEBUG update_header_info: Set text to: {' | '.join(header_parts)}") # Optional debug print
        else:
            self.header_info_label.setText("<i>Index data unavailable.</i>")

    @Slot()
    def show_choose_currencies_dialog(self):
        """Shows the dialog to choose currencies for the currency combo box."""
        current_managed_currencies = self.config.get(
            "user_currencies", COMMON_CURRENCIES.copy()
        )
        if not current_managed_currencies:  # Ensure it's not empty
            current_managed_currencies = COMMON_CURRENCIES.copy()

        dialog = QDialog(self)
        dialog.setWindowTitle("Manage Currencies for Dropdown")
        layout = QVBoxLayout(dialog)
        dialog.setMinimumWidth(300)

        # --- List of current currencies ---
        list_label = QLabel("Currencies available in the main dropdown:")
        layout.addWidget(list_label)
        self.currency_manage_list_widget = QListWidget()
        self.currency_manage_list_widget.addItems(current_managed_currencies)
        self.currency_manage_list_widget.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        layout.addWidget(self.currency_manage_list_widget)

        # --- Add new currency ---
        add_layout = QHBoxLayout()
        self.new_currency_edit = QLineEdit()
        self.new_currency_edit.setPlaceholderText("New Currency (e.g., NZD)")
        self.new_currency_edit.setMaxLength(3)
        add_layout.addWidget(self.new_currency_edit, 1)
        add_button = QPushButton("Add")
        add_button.clicked.connect(self._add_currency_to_manage_list)
        add_layout.addWidget(add_button)
        layout.addLayout(add_layout)

        # --- Delete selected currency ---
        delete_button = QPushButton("Delete Selected Currency")
        delete_button.clicked.connect(self._delete_currency_from_manage_list)
        layout.addWidget(delete_button)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # Store list widget as temp attribute for helper methods
        dialog.currency_list_widget = self.currency_manage_list_widget
        dialog.new_currency_edit = self.new_currency_edit

        if dialog.exec():
            chosen_currencies = []
            for i in range(self.currency_manage_list_widget.count()):
                chosen_currencies.append(
                    self.currency_manage_list_widget.item(i).text()
                )

            if not chosen_currencies:
                QMessageBox.warning(
                    self, "Selection Error", "At least one currency must be available."
                )
                return  # Keep dialog open or handle error

            # --- Temporarily disconnect the signal that triggers refresh ---
            disconnected_successfully = False
            try:
                self.currency_combo.currentTextChanged.disconnect(
                    self.filter_changed_refresh
                )
                disconnected_successfully = True
                logging.debug(
                    "Disconnected currency_combo.currentTextChanged from filter_changed_refresh."
                )
            except RuntimeError:  # Signal was not connected
                logging.warning(
                    "Could not disconnect currentTextChanged from currency_combo, was it connected?"
                )

            # Get the current display currency before clearing the combo box
            current_display_currency_in_combo = self.currency_combo.currentText()

            # Update the currency combo box in the UI
            self.currency_combo.clear()
            self.currency_combo.addItems(chosen_currencies)

            # Set the current text: restore previous if still valid, otherwise pick a new default
            new_selected_currency_for_combo = current_display_currency_in_combo
            if current_display_currency_in_combo not in chosen_currencies:
                new_selected_currency_for_combo = (
                    "USD" if "USD" in chosen_currencies else chosen_currencies[0]
                )
                logging.info(
                    f"Previous display currency '{current_display_currency_in_combo}' removed. "
                    f"Setting combo to '{new_selected_currency_for_combo}'."
                )

            self.currency_combo.setCurrentText(new_selected_currency_for_combo)

            # Update config to reflect the actual state of the combo box
            self.config["display_currency"] = self.currency_combo.currentText()
            self.config["user_currencies"] = chosen_currencies
            self.save_config()

            # --- Reconnect the signal if it was disconnected ---
            if disconnected_successfully:
                self.currency_combo.currentTextChanged.connect(
                    self.filter_changed_refresh
                )
                logging.debug(
                    "Reconnected currency_combo.currentTextChanged to filter_changed_refresh."
                )

            logging.info(
                f"User selected currencies: {chosen_currencies}. Main currency combo updated. No automatic refresh."
            )
            # self.refresh_data() # Explicitly removed to prevent immediate refresh
        else:
            # logging.info("Currency selection cancelled by user.")
            pass
        del self.currency_manage_list_widget  # Clean up temp attribute

    def _add_currency_to_manage_list(self):
        """Adds a new currency to the list in the 'Manage Currencies' dialog."""
        if not hasattr(self, "currency_manage_list_widget"):
            return

        new_code = self.new_currency_edit.text().strip().upper()
        if not new_code:
            return

        if len(new_code) != 3 or not new_code.isalpha():
            QMessageBox.warning(
                self.currency_manage_list_widget.parent(),
                "Invalid Code",
                "Currency code must be 3 alphabetic characters.",
            )
            return

        current_items = [
            self.currency_manage_list_widget.item(i).text()
            for i in range(self.currency_manage_list_widget.count())
        ]
        if new_code in current_items:
            QMessageBox.information(
                self.currency_manage_list_widget.parent(),
                "Duplicate",
                f"Currency '{new_code}' is already in the list.",
            )
            return

        self.currency_manage_list_widget.addItem(new_code)
        self.new_currency_edit.clear()
        # logging.info(f"Currency '{new_code}' added to management list.")

    def _delete_currency_from_manage_list(self):
        """Deletes the selected currency from the list in the 'Manage Currencies' dialog."""
        if not hasattr(self, "currency_manage_list_widget"):
            return

        selected_items = self.currency_manage_list_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self.currency_manage_list_widget.parent(),
                "Selection Error",
                "Please select a currency to delete.",
            )
            return

        currency_to_delete = selected_items[0].text()

        if currency_to_delete == "USD":
            QMessageBox.warning(
                self.currency_manage_list_widget.parent(),
                "Deletion Error",
                "Cannot delete the base currency 'USD'.",
            )
            return

        if self.currency_manage_list_widget.count() <= 1:
            QMessageBox.warning(
                self.currency_manage_list_widget.parent(),
                "Deletion Error",
                "Cannot delete the last currency.",
            )
            return

        self.currency_manage_list_widget.takeItem(
            self.currency_manage_list_widget.row(selected_items[0])
        )
        # logging.info(f"Currency '{currency_to_delete}' removed from management list.")

    # --- ADDED: Slot for CSV Format Help ---
    @Slot()
    def show_csv_format_help(self):
        """Displays a message box with information about the expected CSV format."""
        help_text = """
<b>Required CSV File Format</b>

The CSV file should contain the following columns (header names must match exactly):

<ol>
    <li><b>"Date (MMM DD, YYYY)"</b>: Transaction date (e.g., <i>Jan 01, 2023</i>, <i>Dec 31, 2024</i>)</li>
    <li><b>Transaction Type</b>: Type of transaction (e.g., <i>Buy, Sell, Dividend, Split, Deposit, Withdrawal, Fees</i>)</li>
    <li><b>Stock / ETF Symbol</b>: Ticker symbol (e.g., <i>AAPL, GOOG</i>). Use <i>$CASH</i> for cash deposits/withdrawals.</li>
    <li><b>Quantity of Units</b>: Number of shares/units (positive). Required for most types.</li>
    <li><b>Amount per unit</b>: Price per share/unit (positive). Required for Buy/Sell.</li>
    <li><b>Total Amount</b>: Total value of the transaction (optional, can be calculated for Buy/Sell). Required for some Dividends.</li>
    <li><b>Fees</b>: Transaction fees/commissions (positive).</li>
    <li><b>Investment Account</b>: Name of the account (e.g., <i>Brokerage A, IRA</i>).</li>
    <li><b>Split Ratio (new shares per old share)</b>: Required only for 'Split' type (e.g., <i>2</i> for 2-for-1).</li>
    <li><b>Note</b>: Optional text note for the transaction.</li>
</ol>

<b>Example Rows:</b>
<pre>
"Date (MMM DD, YYYY)",Transaction Type,Stock / ETF Symbol,Quantity of Units,Amount per unit,Total Amount,Fees,Investment Account,Split Ratio (new shares per old share),Note
"Jan 15, 2023",Buy,AAPL,10,150.25,1502.50,5.95,Brokerage A,,Bought Apple shares
"Feb 01, 2023",Dividend,MSFT,,,50.00,,Brokerage A,,Microsoft dividend received
"Mar 10, 2023",Deposit,$CASH,1000,,1000.00,,IRA,,Initial IRA contribution
</pre>
"""
        QMessageBox.information(self, "CSV Format Help", help_text)

    # --- END ADDED ---

    @Slot(str)  # Ensure Slot decorator is imported
    def _chart_holding_history(self, symbol: str):
        """Handles 'Chart History' context menu action by showing a price chart dialog."""
        # logging.info(f"Action triggered: Chart History for {symbol}")
        if is_cash_symbol(symbol):
            QMessageBox.information(self, "Info", "Cannot chart history for Cash.")
            return

        # --- Get required data ---
        yf_symbol = None
        price_data_df = None

        # --- Use the stored map ---
        if hasattr(self, "internal_to_yf_map") and isinstance(
            self.internal_to_yf_map, dict
        ):
            yf_symbol = self.internal_to_yf_map.get(symbol)
            if not yf_symbol:
                # Fallback: Check if the symbol itself is a valid-looking ticker
                if "." not in symbol and ":" not in symbol and " " not in symbol:
                    logging.debug(
                        f"Symbol '{symbol}' not in map, trying as direct YF ticker."
                    )
                    yf_symbol = symbol.upper()
                else:
                    logging.warning(
                        f"No YF symbol mapping found for internal symbol: {symbol}"
                    )
        else:
            logging.warning(
                "internal_to_yf_map attribute not found or invalid. Cannot look up YF ticker."
            )
            # Attempt to use symbol directly as YF symbol as a last resort
            if "." not in symbol and ":" not in symbol and " " not in symbol:
                logging.debug(
                    f"Symbol map missing, trying '{symbol}' as direct YF ticker."
                )
                yf_symbol = symbol.upper()
            else:
                QMessageBox.warning(
                    self,
                    "Mapping Error",
                    f"Could not determine Yahoo Finance ticker for '{symbol}'.",
                )
                return  # Cannot proceed without a YF ticker
        # --- End Use the stored map ---

        # Use ADJUSTED prices for single symbol history chart
        if (
            yf_symbol
            and hasattr(self, "historical_prices_yf_adjusted")
            and isinstance(self.historical_prices_yf_adjusted, dict)
        ):
            price_data_df = self.historical_prices_yf_adjusted.get(yf_symbol)
            if price_data_df is None:
                logging.warning(
                    f"No adjusted historical price data found for YF symbol: {yf_symbol} (Internal: {symbol})"
                )
        else:
            if not yf_symbol:
                logging.warning("No YF symbol determined.")  # Already logged above
            else:
                logging.warning(
                    "historical_prices_yf_adjusted attribute not found or invalid."
                )

        if price_data_df is None or price_data_df.empty:
            QMessageBox.warning(
                self,
                "No Data",
                f"Could not find historical price data for symbol '{symbol}' (Ticker: {yf_symbol or 'N/A'}).",
            )
            return

        # 2. Date Range (from main UI controls)
        try:
            start_date = self.graph_start_date_edit.date().toPython()
            end_date = self.graph_end_date_edit.date().toPython()

            # --- ADDED: Ensure end_date is not in the future ---
            today = date.today()
            if end_date > today:
                end_date = today
                self.graph_end_date_edit.setDate(QDate(end_date))  # Update UI
                logging.info(
                    f"Graph end date was in future, reset to today: {end_date}"
                )
            # --- END ADDED ---

            if start_date >= end_date:
                QMessageBox.warning(
                    self,
                    "Invalid Date Range",
                    "Graph start date must be before end date.",
                )
                return
        except Exception as e:
            logging.error(f"Error getting date range for symbol chart: {e}")
            QMessageBox.critical(
                self, "Error", "Could not get date range from UI controls."
            )
            return

        # 3. Get Local Currency Code for Labeling
        display_currency_code = self._get_currency_for_symbol(symbol)
        if (
            not display_currency_code
        ):  # Should return default, but handle None just in case
            display_currency_code = self.config.get("default_currency", "USD")

        # --- Create and Show Dialog ---
        try:
            dialog = SymbolChartDialog(
                symbol=symbol,  # Pass internal symbol for title
                start_date=start_date,
                end_date=end_date,
                price_data=price_data_df.copy(),  # Pass a copy of the data
                display_currency=display_currency_code,  # Pass currency CODE
                parent=self,
            )
            dialog.exec()  # Show modal

        except Exception as e_dialog:
            logging.exception(
                f"Error creating or showing SymbolChartDialog for {symbol}"
            )
            QMessageBox.critical(
                self,
                "Chart Error",
                f"Failed to display chart for {symbol}:\n{e_dialog}",
            )

    @Slot(QPoint, Figure, str)
    def _show_graph_context_menu(self, pos: QPoint, figure: Figure, graph_name: str):
        """Shows a context menu on a graph canvas to allow saving the graph."""
        canvas = figure.canvas
        if not canvas:
            return

        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())

        save_action = QAction("Save Graph as Image...", self)
        save_action.triggered.connect(
            lambda: self._save_figure_as_image(figure, graph_name)
        )
        menu.addAction(save_action)

        global_pos = canvas.mapToGlobal(pos)
        menu.exec(global_pos)

    def _save_figure_as_image(self, figure: Figure, default_name_prefix: str):
        """Opens a file dialog to save a Matplotlib figure as an image."""
        if not figure:
            return

        current_date_str = date.today().strftime("%Y%m%d")
        suggested_filename = f"{default_name_prefix}_{current_date_str}.png"
        start_dir = self.config.get(
            "last_image_export_path",
            QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
            or os.getcwd(),
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Graph",
            os.path.join(start_dir, suggested_filename),
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;SVG Vector Image (*.svg);;PDF Document (*.pdf);;All Files (*)",
        )

        if file_path:
            self.config["last_image_export_path"] = os.path.dirname(file_path)
            self.save_config()
            try:
                figure.savefig(
                    file_path,
                    dpi=300,
                    bbox_inches="tight",
                    facecolor=figure.get_facecolor(),
                )
                QMessageBox.information(
                    self,
                    "Save Successful",
                    f"Graph saved successfully to:\n{file_path}",
                )
            except Exception as e:
                logging.error(f"Error saving graph to {file_path}: {e}", exc_info=True)
                QMessageBox.critical(
                    self, "Save Error", f"Could not save the graph:\n{e}"
                )

    def _get_currency_for_symbol(self, symbol_to_find: str) -> Optional[str]:
        """
        Finds the local currency code associated with a symbol based on transaction data
        or the account currency map. Returns the default currency if not found.

        Args:
            symbol_to_find (str): The internal symbol (e.g., 'AAPL', 'SET:BKK').

        Returns:
            Optional[str]: The 3-letter currency code (e.g., 'USD', 'THB') or None if lookup fails badly.
                           Returns default currency as fallback.
        """  # 1. Check if it's a cash symbol
        if is_cash_symbol(symbol_to_find):
            # For a generic $CASH symbol, we can't determine currency from symbol alone.
            # The calling context should handle this. As a fallback, return app's default.
            return self.config.get("default_currency", "USD")
        # 2. Check Holdings Data (most reliable if data is loaded) # This requires holdings_data to have the 'Local Currency' column correctly populated.
        if (
            hasattr(self, "holdings_data")
            and not self.holdings_data.empty
            and "Symbol" in self.holdings_data.columns
        ):
            symbol_rows = self.holdings_data[
                self.holdings_data["Symbol"] == symbol_to_find
            ]
            if not symbol_rows.empty and "Local Currency" in symbol_rows.columns:
                first_currency = symbol_rows["Local Currency"].iloc[0]
                if (
                    pd.notna(first_currency)
                    and isinstance(first_currency, str)
                    and len(first_currency) == 3
                ):
                    # logging.debug(f"Found currency '{first_currency}' for symbol '{symbol_to_find}' via holdings_data.")
                    return first_currency.upper()

        # 3. Fallback: Check original transactions (less efficient but broader)
        # This requires original_data to have 'Investment Account' and 'Stock / ETF Symbol'
        if hasattr(self, "original_data") and not self.original_data.empty:
            if (
                "Stock / ETF Symbol" in self.original_data.columns
                and "Investment Account" in self.original_data.columns
            ):
                symbol_rows_orig = self.original_data[
                    self.original_data["Stock / ETF Symbol"] == symbol_to_find
                ]
                if not symbol_rows_orig.empty:
                    first_account = symbol_rows_orig["Investment Account"].iloc[0]
                    if pd.notna(first_account) and isinstance(first_account, str):
                        # Get currency from the account map
                        acc_map = self.config.get("account_currency_map", {})
                        currency_from_map = acc_map.get(first_account)
                        if (
                            currency_from_map
                            and isinstance(currency_from_map, str)
                            and len(currency_from_map) == 3
                        ):
                            # logging.debug(f"Found currency '{currency_from_map}' for symbol '{symbol_to_find}' via original_data/account_map (Account: {first_account}).")
                            return currency_from_map.upper()

        # 4. Final Fallback: Return application's default currency
        default_curr = self.config.get("default_currency", "USD")
        # logging.debug(f"Could not find specific currency for symbol '{symbol_to_find}'. Using default: {default_curr}")
        return default_curr

    # Add this NEW slot method to PortfolioApp
    @Slot()
    def show_account_currency_dialog(self):
        """Shows the dialog to edit account currencies."""
        current_map = self.config.get("account_currency_map", {})
        current_default = self.config.get("default_currency", "USD")
        # Use all accounts detected during the last load/refresh
        accounts_to_show = (
            self.available_accounts
            if self.available_accounts
            else list(current_map.keys())
        )
        # Get the list of user-selectable currencies from the config
        user_currencies = self.config.get("user_currencies", COMMON_CURRENCIES.copy())

        updated_settings = AccountCurrencyDialog.get_settings(
            parent=self,
            current_map=current_map,
            current_default=current_default,
            all_accounts=accounts_to_show,
            # Pass the list of currencies to the dialog
            user_currencies=user_currencies,
        )

        if updated_settings:  # If user clicked Save
            new_map, new_default = updated_settings
            # Check if settings actually changed
            map_changed = new_map != self.config.get("account_currency_map", {})
            default_changed = new_default != self.config.get("default_currency", "USD")

            if map_changed or default_changed:
                logging.info(
                    "Account currency settings changed. Saving and refreshing..."
                )
                self.config["account_currency_map"] = new_map
                self.config["default_currency"] = new_default
                self.save_config()  # Save the updated config
                self.refresh_data()  # Trigger a full refresh as currencies impact calculations
            else:
                logging.info("Account currency settings unchanged.")

    # --- Need to implement these methods used in the menu ---
    def save_transactions_as(self):
        # Placeholder - opens save dialog, writes self.original_data (or maybe filtered?) to new CSV
        if not hasattr(self, "original_data") or self.original_data.empty:
            QMessageBox.warning(self, "No Data", "No transaction data loaded to save.")
            return

        start_dir = (
            os.path.dirname(self.transactions_file)
            if self.transactions_file
            else os.getcwd()
        )
        fname, _ = QFileDialog.getSaveFileName(
            self, "Save Transactions As", start_dir, "CSV Files (*.csv)"
        )

        if fname:
            if not fname.lower().endswith(".csv"):
                fname += ".csv"
            logging.info(f"Attempting to save transactions to: {fname}")

            # --- Backup the ORIGINAL file FIRST ---
            original_file_to_backup = self.transactions_file
            if original_file_to_backup and os.path.exists(original_file_to_backup):
                backup_ok, backup_msg = self._backup_csv(
                    filename_to_backup=original_file_to_backup
                )  # Pass original filename
                if not backup_ok:
                    QMessageBox.critical(
                        self,
                        "Backup Error",
                        f"Failed to backup original CSV before saving:\n{backup_msg}",
                    )
                    return  # Stop if backup fails
            else:
                logging.info(
                    "Original transaction file not set or found, skipping backup for 'Save As'."
                )
            # --- End Backup ---

            # Use the same rewrite logic, but with the new filename
            temp_orig_file = self.transactions_file  # Store original
            self.transactions_file = fname  # Temporarily set filename for rewrite
            # Modify _rewrite_csv to optionally skip its internal backup call
            success = self._rewrite_csv(
                self.original_data.drop(columns=["original_index"], errors="ignore"),
                skip_backup=True,  # Add flag to skip internal backup
            )
            self.transactions_file = temp_orig_file  # Restore original filename

            if success:
                QMessageBox.information(
                    self, "Save Successful", f"Transactions saved to:\n{fname}"
                )
            # Error message shown by _rewrite_csv if it failed
            else:
                QMessageBox.critical(
                    self, "Save Failed", f"Could not save transactions to:\n{fname}"
                )

    def show_about_dialog(self):
        # Placeholder - shows a simple message box with app info
        QMessageBox.about(
            self,
            "About Investa",
            "<b>Investa Portfolio Dashboard</b><br><br>"
            "Version: 0.3.0 (Features: Tx Edit/Delete, Ignored Log, Table Filter, Settings Edit)<br>"
            "Author: Kit Matan<br>"
            "License: MIT<br><br>"
            "Data provided by Yahoo Finance. Use at your own risk.",
        )

    @Slot()
    def show_symbol_settings_dialog(self):  # Renamed from show_manual_overrides_dialog
        """Shows the dialog to edit symbol settings (overrides, map, exclusions)."""
        if (
            not hasattr(self, "manual_overrides_dict")
            or not hasattr(self, "user_symbol_map_config")
            or not hasattr(self, "user_excluded_symbols_config")
        ):
            logging.error("Symbol settings dictionaries not initialized.")
            QMessageBox.critical(self, "Error", "Symbol settings data is not loaded.")
            return

        updated_settings = ManualPriceDialog.get_symbol_settings(  # Renamed static call
            parent=self,
            current_overrides=self.manual_overrides_dict,
            current_symbol_map=self.user_symbol_map_config,
            current_excluded_symbols=self.user_excluded_symbols_config,
        )

        if updated_settings is not None:
            new_manual_overrides = updated_settings.get("manual_price_overrides", {})
            new_symbol_map = updated_settings.get("user_symbol_map", {})
            new_excluded_symbols = set(
                updated_settings.get("user_excluded_symbols", [])
            )

            settings_changed = (
                new_manual_overrides != self.manual_overrides_dict
                or new_symbol_map != self.user_symbol_map_config
                or new_excluded_symbols != self.user_excluded_symbols_config
            )

            if settings_changed:
                logging.info("Symbol settings changed. Saving and refreshing...")
                self.manual_overrides_dict = new_manual_overrides
                self.user_symbol_map_config = new_symbol_map
                self.user_excluded_symbols_config = new_excluded_symbols
                if self._save_manual_overrides_to_json():  # Save all parts
                    self.refresh_data()
            else:
                logging.info("Symbol settings unchanged.")

    def _load_manual_overrides(self):  # No longer returns, sets instance attributes
        """Loads manual overrides, symbol map, and excluded symbols from MANUAL_OVERRIDES_FILE.
        Handles migration from old price-only format if necessary for price overrides.
        """
        # Initialize with defaults from config.py
        # These will be overwritten if data is found in the JSON file.
        self.manual_overrides_dict = {}  # For prices, sectors, etc.
        self.user_symbol_map_config = config.SYMBOL_MAP_TO_YFINANCE.copy()
        self.user_excluded_symbols_config = set(config.YFINANCE_EXCLUDED_SYMBOLS.copy())

        if os.path.exists(self.MANUAL_OVERRIDES_FILE):
            try:
                with open(self.MANUAL_OVERRIDES_FILE, "r", encoding="utf-8") as f:
                    loaded_data = json.load(f)

                if isinstance(loaded_data, dict):
                    # Load manual price overrides (existing logic)
                    price_overrides_loaded = loaded_data.get(
                        "manual_price_overrides", {}
                    )
                    if isinstance(price_overrides_loaded, dict):
                        valid_price_overrides: Dict[str, Dict[str, Any]] = {}
                        # ... (validation logic for price_overrides_loaded as before) ...
                        for key, value_dict in price_overrides_loaded.items():
                            if isinstance(key, str) and isinstance(value_dict, dict):
                                entry: Dict[str, Any] = {}
                                price = value_dict.get("price")
                                if (
                                    price is not None
                                    and isinstance(price, (int, float))
                                    and price > 0
                                ):
                                    entry["price"] = float(price)
                                entry["asset_type"] = str(
                                    value_dict.get("asset_type", "")
                                ).strip()
                                entry["sector"] = str(
                                    value_dict.get("sector", "")
                                ).strip()
                                entry["geography"] = str(
                                    value_dict.get("geography", "")
                                ).strip()
                                entry["industry"] = str(
                                    value_dict.get("industry", "")
                                ).strip()
                                valid_price_overrides[key.upper().strip()] = entry
                        self.manual_overrides_dict = valid_price_overrides
                        logging.info(
                            f"Loaded {len(self.manual_overrides_dict)} manual price overrides."
                        )
                    else:
                        logging.warning(
                            "Manual price overrides in JSON is not a dictionary. Using defaults for prices."
                        )

                    # Load user symbol map
                    user_map_loaded = loaded_data.get(
                        "user_symbol_map", None
                    )  # Default to None to distinguish from empty dict
                    if isinstance(user_map_loaded, dict):
                        self.user_symbol_map_config = {
                            k.upper().strip(): v.upper().strip()
                            for k, v in user_map_loaded.items()
                        }
                        logging.info(
                            f"Loaded {len(self.user_symbol_map_config)} user symbol mappings from JSON."
                        )
                    elif (
                        user_map_loaded is None
                    ):  # Key not present, keep default from config
                        logging.info(
                            "No 'user_symbol_map' key in JSON, using defaults from config.py."
                        )
                    else:  # Invalid type
                        logging.warning(
                            "User symbol map in JSON is not a valid dictionary. Using defaults from config.py."
                        )

                    # Load user excluded symbols
                    user_excluded_loaded = loaded_data.get(
                        "user_excluded_symbols", None
                    )  # Default to None
                    if isinstance(user_excluded_loaded, list):
                        self.user_excluded_symbols_config = {
                            s.upper().strip()
                            for s in user_excluded_loaded
                            if isinstance(s, str)
                        }
                        logging.info(
                            f"Loaded {len(self.user_excluded_symbols_config)} user excluded symbols from JSON."
                        )
                    elif (
                        user_excluded_loaded is None
                    ):  # Key not present, keep default from config
                        logging.info(
                            "No 'user_excluded_symbols' key in JSON, using defaults from config.py."
                        )
                    else:  # Invalid type
                        logging.warning(
                            "User excluded symbols in JSON is not a valid list. Using defaults from config.py."
                        )

                # --- Migration from old format (if new format load failed or file was old format) ---
                elif isinstance(loaded_data, dict) and all(
                    isinstance(v, (int, float)) for v in loaded_data.values()
                ):
                    logging.info("Attempting to migrate old manual price format...")
                    migrated_prices: Dict[str, Dict[str, Any]] = {}
                    for key, price_val in loaded_data.items():
                        if (
                            isinstance(key, str)
                            and isinstance(price_val, (int, float))
                            and price_val > 0
                        ):
                            migrated_prices[key.upper().strip()] = {
                                "price": float(price_val)
                            }
                    if migrated_prices:
                        self.manual_overrides_dict = (
                            migrated_prices  # Only overrides prices
                        )
                        # user_symbol_map_config and user_excluded_symbols_config retain their defaults from config.py
                        logging.info(
                            f"Migrated {len(migrated_prices)} entries from old price format. Other settings use defaults."
                        )
                        self._save_manual_overrides_to_json()  # Save in new format immediately
                else:
                    logging.warning(
                        f"Content of {self.MANUAL_OVERRIDES_FILE} is not a recognized dictionary format. Using defaults for all settings."
                    )
            except json.JSONDecodeError as e:
                logging.error(
                    f"Error decoding JSON from {self.MANUAL_OVERRIDES_FILE}: {e}. Using defaults for all settings."
                )
            except Exception as e:
                logging.error(
                    f"Error reading {self.MANUAL_OVERRIDES_FILE}: {e}. Using defaults for all settings."
                )
        else:  # File does not exist
            logging.info(
                f"{self.MANUAL_OVERRIDES_FILE} not found. Using default symbol settings from config.py."
            )
        # Log final state after load/default
        logging.info(
            f"Final loaded settings: Overrides={len(self.manual_overrides_dict)}, Map={len(self.user_symbol_map_config)}, Excluded={len(self.user_excluded_symbols_config)}"
        )

    def _save_manual_overrides_to_json(
        self,
    ) -> bool:  # Renamed from _save_manual_prices_to_json
        """Saves all manual settings (prices, symbol map, exclusions) to MANUAL_OVERRIDES_FILE."""
        if (
            not hasattr(self, "manual_overrides_dict")
            or not hasattr(self, "user_symbol_map_config")
            or not hasattr(self, "user_excluded_symbols_config")
        ):
            logging.error("Cannot save symbol settings, dictionary attributes missing.")
            return False

        overrides_file_path = self.MANUAL_OVERRIDES_FILE
        logging.info(f"Saving symbol settings to: {overrides_file_path}")

        data_to_save = {
            "manual_price_overrides": dict(sorted(self.manual_overrides_dict.items())),
            "user_symbol_map": dict(sorted(self.user_symbol_map_config.items())),
            "user_excluded_symbols": sorted(
                list(self.user_excluded_symbols_config)
            ),  # Save set as sorted list
        }

        try:
            overrides_dir = os.path.dirname(overrides_file_path)
            if overrides_dir:
                os.makedirs(overrides_dir, exist_ok=True)

            with open(overrides_file_path, "w", encoding="utf-8") as f:
                json.dump(data_to_save, f, indent=4, ensure_ascii=False)
            logging.info("Symbol settings saved successfully.")
            return True
        except TypeError as e:
            logging.error(f"TypeError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Data error saving symbol settings:\n{e}"
            )
            return False
        except IOError as e:
            logging.error(f"IOError writing symbol settings JSON: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Could not write to file:\n{overrides_file_path}\n{e}",
            )
            return False
        except Exception as e:
            logging.exception("Unexpected error writing symbol settings JSON")
            QMessageBox.critical(
                self,
                "Save Error",
                f"An unexpected error occurred saving symbol settings:\n{e}",
            )
            return False

    # --- Helper to create summary items (moved from initUI) ---
    def create_summary_item(self, label_text, is_large=False, has_percentage=False):
        """
        Creates a QLabel pair (label and value) for the summary grid.

        Args:
            label_text (str): The text for the descriptive label (e.g., "Net Value").
            is_large (bool, optional): If True, uses slightly larger fonts for emphasis.
                                       Defaults to False.
            has_percentage (bool, optional): If True, creates an additional QLabel for percentage.
                                             Defaults to False.

        Returns:
            Tuple[QLabel, QLabel] or Tuple[QLabel, QLabel, QLabel]: A tuple containing
            the created label and value QLabels, and optionally a percentage QLabel.
        """
        label = QLabel(label_text + ":")
        label.setObjectName("SummaryLabelLarge" if is_large else "SummaryLabel")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        value = QLabel("N/A")  # Default text
        value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value.setObjectName("SummaryValueLarge" if is_large else "SummaryValue")
        value.setTextFormat(Qt.RichText)

        # Adjust font sizes based on is_large flag
        # Ensure self.app_font exists if called during initUI (should be fine if initUI called after font set)
        base_font_size = (
            self.app_font.pointSize() if hasattr(self, "app_font") else 10
        )  # Default size

        label_font = QFont(
            self.app_font if hasattr(self, "app_font") else QFont()
        )  # Use base app font or default
        label_font.setPointSize(base_font_size + (1 if is_large else 0))
        label.setFont(label_font)

        value_font = QFont(
            self.app_font if hasattr(self, "app_font") and self.app_font else QFont()
        )  # Use base app font or default
        # Increase font size more for large value display
        value_font.setPointSize(base_font_size + (4 if is_large else 1))
        value_font.setBold(True)
        value.setFont(value_font)

        if has_percentage:
            percentage_value = QLabel("N/A")
            percentage_value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            percentage_value.setObjectName("SummaryPercentageValue")
            percentage_value.setTextFormat(Qt.RichText)
            percentage_font = QFont(
                self.app_font if hasattr(self, "app_font") else QFont()
            )
            percentage_font.setPointSize(
                base_font_size - 1
            )  # Smaller font for percentage
            percentage_value.setFont(percentage_font)
            return label, value, percentage_value
        else:
            return label, value

    # --- Column Visibility Methods (Define BEFORE initUI) ---

    @Slot(QPoint)  # Add Slot decorator
    def show_header_context_menu(self, pos: QPoint):
        """
        Shows a context menu on the table header to toggle column visibility.

        Args:
            pos (QPoint): The position where the right-click occurred within the header.
        """
        header = self.sender()
        if not isinstance(header, QHeaderView):
            logging.warning(
                f"show_header_context_menu called by non-QHeaderView sender: {type(header)}"
            )
            return

        global_pos = header.mapToGlobal(
            pos
        )  # Map local position to global screen position
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())  # Apply application style to menu

        # Ensure possible columns are up-to-date based on current currency
        self._ensure_all_columns_in_visibility()

        # Create checkable actions for each possible column
        for col_name in self.all_possible_ui_columns:
            action = QAction(col_name, self)
            action.setCheckable(True)
            # Check the action based on the current visibility state
            action.setChecked(self.column_visibility.get(col_name, True))
            # Connect the action's triggered signal to the toggle function
            action.triggered.connect(self.toggle_column_visibility)
            menu.addAction(action)

        menu.exec(global_pos)  # Show the menu at the calculated position

    @Slot()  # Add Slot decorator
    def toggle_column_visibility(self):
        """
        Slot triggered by column visibility actions in the context menu.

        Updates the `self.column_visibility` state and calls `apply_column_visibility`.
        """
        action = self.sender()  # Get the QAction that triggered the signal
        if isinstance(action, QAction):
            col_name = action.text()
            is_visible = action.isChecked()
            if col_name in self.column_visibility:
                # Update the internal visibility state
                self.column_visibility[col_name] = is_visible
                logging.info(f"Column visibility changed: '{col_name}' -> {is_visible}")
                # Apply the change to the table view immediately
                self.apply_column_visibility()
            else:
                logging.warning(
                    f"Warning: Toggled column '{col_name}' not found in visibility config."
                )

    def apply_column_visibility(self):
        """Applies the current column visibility state to the QTableView."""
        # Ensure table_view and model exist
        # --- MODIFIED: Check for both table views ---
        if (
            not hasattr(self, "table_view")
            or not self.table_view
            or not hasattr(self, "frozen_table_view")
            or not self.frozen_table_view
            or not hasattr(self, "table_model")
            or not isinstance(self.table_model, PandasModel)
            or self.table_model.columnCount() == 0
        ):
            return  # Cannot apply if table/model not ready

        header = self.table_view.horizontalHeader()
        frozen_header = self.frozen_table_view.horizontalHeader()
        frozen_view_width = 0

        # Iterate through the columns currently in the model
        for col_index in range(self.table_model.columnCount()):
            # Get the header name displayed in the UI for this column index
            header_name = self.table_model.headerData(
                col_index, Qt.Horizontal, Qt.DisplayRole
            )
            if not header_name:
                continue

            is_frozen_col = str(header_name) in FROZEN_COLUMNS_UI
            is_visible = self.column_visibility.get(str(header_name), True)
            is_internal_col = str(header_name) in [
                "is_group_header",
                "group_key",
                "is_summary_row",
            ]

            if is_internal_col:
                self.table_view.setColumnHidden(col_index, True)
                self.frozen_table_view.setColumnHidden(col_index, True)
                continue

            if is_frozen_col:
                # This column belongs to the frozen view
                self.frozen_table_view.setColumnHidden(col_index, not is_visible)
                self.table_view.setColumnHidden(
                    col_index, True
                )  # Hide in scrollable view
                if is_visible:
                    # Use resizeColumnsToContents before getting width
                    self.frozen_table_view.resizeColumnToContents(col_index)
                    frozen_view_width += self.frozen_table_view.columnWidth(col_index)
            else:  # scrollable column
                # This column belongs to the scrollable view
                self.frozen_table_view.setColumnHidden(
                    col_index, True
                )  # Hide in frozen view
                self.table_view.setColumnHidden(col_index, not is_visible)

        # Add a small buffer for borders/etc.
        # The vertical header is not visible, so we don't need to add its width.
        self.frozen_table_view.setFixedWidth(frozen_view_width + 4)
        # --- END MODIFICATION ---

    def save_config(self):
        """Saves the current application configuration to gui_config.json."""
        if not self.CONFIG_FILE:  # Should have been set in __init__
            logging.error("Cannot save config: CONFIG_FILE path is not set.")
            QMessageBox.critical(
                self,
                "Config Error",
                "Cannot save settings: Configuration file path is not defined.",
            )
            return

        # Ensure self.config is a dict, even if it was somehow cleared
        if not isinstance(self.config, dict):
            logging.warning(
                "self.config was not a dict during save_config. Reinitializing."
            )
            self.config = (
                {}
            )  # Reinitialize to avoid error, though this indicates a prior issue.

        self.config["transactions_file"] = self.DB_FILE_PATH  # Store DB path
        # transactions_file_csv_fallback is already in self.config if it was set during load
        # or if a CSV was opened and then a new DB created/opened.
        self.config["last_csv_import_path"] = self.config.get(
            "last_csv_import_path", os.getcwd()
        )
        self.config["last_csv_export_path"] = self.config.get(
            "last_csv_export_path", os.getcwd()
        )
        self.config["last_excel_export_path"] = self.config.get(
            "last_excel_export_path", os.getcwd()
        )
        self.config["last_image_export_path"] = self.config.get(
            "last_image_export_path", os.getcwd()
        )

        self.config.pop("fmp_api_key", None)  # Ensure API key is not saved

        if hasattr(self, "currency_combo"):
            self.config["display_currency"] = self.currency_combo.currentText()
        # else: use existing value in self.config or default (handled by load_config next time)

        if hasattr(self, "show_closed_check"):
            self.config["show_closed"] = self.show_closed_check.isChecked()

        if hasattr(self, "group_by_sector_check"):
            self.config["group_by_sector"] = self.group_by_sector_check.isChecked()

        if hasattr(self, "selected_accounts") and isinstance(
            self.selected_accounts, list
        ):
            if (
                hasattr(self, "available_accounts")
                and self.available_accounts
                and len(self.selected_accounts) == len(self.available_accounts)
            ):
                self.config["selected_accounts"] = []  # Empty list means "All"
            else:
                self.config["selected_accounts"] = self.selected_accounts
        # else: keep existing self.config["selected_accounts"]

        # account_currency_map and default_currency are already in self.config, updated by dialogs
        # Ensure they exist from defaults if somehow missing
        self.config["account_currency_map"] = self.config.get(
            "account_currency_map", {"SET": "THB"}
        )
        self.config["default_currency"] = self.config.get(
            "default_currency",
            config.DEFAULT_CURRENCY if hasattr(config, "DEFAULT_CURRENCY") else "USD",
        )

        self.config["load_on_startup"] = self.config.get("load_on_startup", True)

        if hasattr(self, "graph_start_date_edit"):
            self.config["graph_start_date"] = (
                self.graph_start_date_edit.date().toString("yyyy-MM-dd")
            )
        if hasattr(self, "graph_end_date_edit"):
            self.config["graph_end_date"] = self.graph_end_date_edit.date().toString(
                "yyyy-MM-dd"
            )
        if hasattr(self, "graph_interval_combo"):
            self.config["graph_interval"] = self.graph_interval_combo.currentText()

        # --- ADDED: Save Performance Graph Tab Index ---
        if hasattr(self, "perf_graphs_tab_widget"):
            self.config["perf_graph_tab_index"] = self.perf_graphs_tab_widget.currentIndex()
        # --- END ADDED ---

        if hasattr(self, "selected_benchmarks") and isinstance(
            self.selected_benchmarks, list
        ):
            self.config["graph_benchmarks"] = self.selected_benchmarks
        # else: keep existing self.config["graph_benchmarks"]

        if hasattr(self, "column_visibility"):  # Should always exist after __init__
            self.config["column_visibility"] = self.column_visibility

        # --- ADDED: Save header state ---
        if hasattr(self, "table_view"):
            header_state = self.table_view.horizontalHeader().saveState()
            self.config["holdings_table_header_state"] = (
                header_state.toHex().data().decode()
            )
        # --- END ADDED ---

        # Ensure the current theme is saved
        if hasattr(self, "current_theme"):
            self.config["theme"] = self.current_theme
        else:  # Fallback if current_theme attribute somehow missing
            self.config["theme"] = self.config.get(
                "theme", "light"
            )  # Get existing or default

        # Save bar chart periods
        if hasattr(self, "annual_periods_spinbox"):
            self.config["bar_periods_annual"] = self.annual_periods_spinbox.value()
        if hasattr(self, "monthly_periods_spinbox"):
            self.config["bar_periods_monthly"] = self.monthly_periods_spinbox.value()
        # Save PVC tab spinbox values
        if hasattr(self, "weekly_periods_spinbox"):
            self.config["bar_periods_weekly"] = self.weekly_periods_spinbox.value()

        # Save dividend history settings
        if hasattr(self, "dividend_period_combo"):
            self.config["dividend_agg_period"] = (
                self.dividend_period_combo.currentText()
            )
        if hasattr(self, "dividend_periods_spinbox"):
            self.config["dividend_periods_to_show"] = (
                self.dividend_periods_spinbox.value()
            )

        # Save rebalancing targets from the table
        if hasattr(self, "target_allocation_table"):
            rebalancing_targets = {}
            for i in range(self.target_allocation_table.rowCount()):
                symbol_item = self.target_allocation_table.item(i, 0)
                target_pct_item = self.target_allocation_table.item(i, 4)
                if symbol_item and target_pct_item:
                    symbol = symbol_item.text()
                    try:
                        target_pct = float(target_pct_item.text().replace("%", ""))
                        rebalancing_targets[symbol] = target_pct
                    except (ValueError, AttributeError):
                        logging.warning(
                            f"Could not parse target % for '{symbol}' while saving config."
                        )
            self.config["rebalancing_targets"] = rebalancing_targets

        # Save PVC tab spinbox values
        if hasattr(self, "pvc_annual_graph_spinbox"):
            self.config["pvc_annual_periods"] = self.pvc_annual_graph_spinbox.value()
        if hasattr(self, "pvc_monthly_graph_spinbox"):
            self.config["pvc_monthly_periods"] = self.pvc_monthly_graph_spinbox.value()
        if hasattr(self, "pvc_annual_graph_spinbox"):
            self.config["pvc_annual_periods"] = self.pvc_annual_graph_spinbox.value()
        if hasattr(self, "pvc_monthly_graph_spinbox"):
            self.config["pvc_monthly_periods"] = self.pvc_monthly_graph_spinbox.value()
        if hasattr(self, "pvc_weekly_graph_spinbox"):
            self.config["pvc_weekly_periods"] = self.pvc_weekly_graph_spinbox.value()
        if hasattr(self, "pvc_daily_graph_spinbox"):
            self.config["pvc_daily_periods"] = self.pvc_daily_graph_spinbox.value()

        # Also save the default spinbox values (these are loaded by load_config if key exists)
        # self.config["dividend_chart_default_periods_annual"] = self.config.get("dividend_chart_default_periods_annual", 10)
        # self.config["dividend_chart_default_periods_quarterly"] = self.config.get("dividend_chart_default_periods_quarterly", 12)
        # self.config["dividend_chart_default_periods_monthly"] = self.config.get("dividend_chart_default_periods_monthly", 24)

        try:
            # Ensure directory for CONFIG_FILE exists
            config_dir = os.path.dirname(self.CONFIG_FILE)
            if (
                config_dir
            ):  # Check if dirname returned anything (it should for absolute paths)
                os.makedirs(config_dir, exist_ok=True)

            with open(self.CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Config saved to {self.CONFIG_FILE}")
        except Exception as e:
            logging.error(
                f"Error saving config to {self.CONFIG_FILE}: {e}", exc_info=True
            )
            QMessageBox.warning(
                self, "Config Save Error", f"Could not save settings:\n{e}"
            )

    def _apply_initial_styles_and_updates(self):
        """Applies styles and initial UI states after widgets are created."""
        # self.apply_styles() # Styles will be applied by apply_theme called from __init__
        # --- ADDED: Set initial date ---
        self.date_label.setText(f"<b>{datetime.now().strftime('%A, %B %d, %Y')}</b>")
        self.update_header_info(loading=True)
        self.update_performance_graphs(initial=True)
        self._update_account_button_text()
        self._update_benchmark_button_text()
        self._update_table_title(pd.DataFrame())
        # Header state is now restored in handle_results after data is loaded.

        # --- Style Tab Titles ---
        if hasattr(self, "main_tab_widget") and self.main_tab_widget:
            tab_bar = self.main_tab_widget.tabBar()
            if tab_bar:
                # Font styling for tab titles is now handled by QSS files (style.qss, style_dark.qss)
                # This ensures theme consistency and avoids overrides.
                logging.debug("Tab title font styling is now managed by QSS.")
        self._update_periodic_bar_charts()  # Draw empty bar charts initially

    def _init_ui_structure(self):
        """Sets up the main window layout using QFrames for structure."""
        logging.debug("--- _init_ui_structure: START ---")
        self.setWindowTitle(self.base_window_title)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # --- Header and Controls (remain outside tabs) ---
        self.header_frame = QFrame()
        self.header_frame.setObjectName("HeaderFrame")
        self.controls_frame = QFrame()
        self.controls_frame.setObjectName("ControlsFrame")
        main_layout.addWidget(self.header_frame)
        main_layout.addWidget(self.controls_frame)

        # --- Main Tab Widget ---
        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setObjectName("MainTabWidget")
        main_layout.addWidget(
            self.main_tab_widget, 1
        )  # Tab widget takes remaining space

        # --- Tab 1: Performance & Summary ---
        self.performance_summary_tab = QWidget()
        self.summary_and_graphs_frame = QFrame()
        self.summary_and_graphs_frame.setObjectName("SummaryAndGraphsFrame")

        # Create the bar_charts_frame here as it's now part of this tab's content
        self.bar_charts_frame = QFrame()
        self.bar_charts_frame.setObjectName("BarChartsFrame")
        self.bar_charts_frame.setVisible(False)  # Initially hidden

        # Create the content_frame (for pie charts and holdings table) here
        # It will be added to the performance_summary_tab's layout
        self.content_frame = QFrame()
        self.summary_and_graphs_frame.setObjectName("SummaryAndGraphsFrame")

        perf_summary_layout = QVBoxLayout(self.performance_summary_tab)
        perf_summary_layout.setContentsMargins(0, 0, 0, 0)
        perf_summary_layout.addWidget(
            self.summary_and_graphs_frame, 2  # Summary grid and line graphs
        )
        # --- MODIFIED ORDER: Bar charts now above content_frame ---
        perf_summary_layout.addWidget(
            self.content_frame, 3  # Pie charts and holdings table
        )
        # --- END MODIFIED ORDER ---
        # Tabs will be added in _add_main_tabs for clear ordering.
        self.performance_summary_tab.setObjectName(
            "performance_summary_tab"
        )  # Ensure object name is set for styling
        self.capital_gains_tab = QWidget()
        self.capital_gains_tab.setObjectName(
            "capital_gains_tab"
        )  # Ensure object name is set for styling
        self.intraday_chart_tab = QWidget()
        self.rebalancing_tab = QWidget()
        self.advanced_analysis_tab = QWidget()
        self.advanced_analysis_tab.setObjectName("advanced_analysis_tab")
        self._init_advanced_analysis_tab()

        # Tab 4 for Dividend History will be initialized in _init_ui_widgets
        # The "Holdings Overview" tab is now removed.
        logging.debug("--- _init_ui_structure: END ---")

    def _init_asset_change_tab_widgets(self):
        """Initializes the content for the Asset Change tab."""
        self.periodic_value_change_tab = QWidget()
        self.periodic_value_change_tab.setObjectName("AssetChangeTab")
        main_layout = QVBoxLayout(self.periodic_value_change_tab)
        main_layout.setContentsMargins(5, 5, 5, 5)

        # --- Period Settings --
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Annual Periods:"))
        self.pvc_annual_spinbox = QSpinBox()
        self.pvc_annual_spinbox.setMinimum(1)
        self.pvc_annual_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_ANNUAL)
        self.pvc_annual_spinbox.setFixedWidth(40)
        self.pvc_annual_spinbox.setValue(self.config.get("pvc_annual_periods", 10))
        controls_layout.addWidget(self.pvc_annual_spinbox)

        controls_layout.addWidget(QLabel("Monthly Periods:"))
        self.pvc_monthly_spinbox = QSpinBox()
        self.pvc_monthly_spinbox.setMinimum(1)
        self.pvc_monthly_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_MONTHLY)
        self.pvc_monthly_spinbox.setFixedWidth(40)
        self.pvc_monthly_spinbox.setValue(self.config.get("pvc_monthly_periods", 12))
        controls_layout.addWidget(self.pvc_monthly_spinbox)

        controls_layout.addWidget(QLabel("Weekly Periods:"))
        self.pvc_weekly_spinbox = QSpinBox()
        self.pvc_weekly_spinbox.setMinimum(1)
        self.pvc_weekly_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_WEEKLY)
        self.pvc_weekly_spinbox.setFixedWidth(40)
        self.pvc_weekly_spinbox.setValue(self.config.get("pvc_weekly_periods", 12))
        controls_layout.addWidget(self.pvc_weekly_spinbox)

        controls_layout.addWidget(QLabel("Daily Periods:"))
        self.pvc_daily_spinbox = QSpinBox()
        self.pvc_daily_spinbox.setMinimum(1)
        self.pvc_daily_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_DAILY)
        self.pvc_daily_spinbox.setFixedWidth(40)
        self.pvc_daily_spinbox.setValue(self.config.get("pvc_daily_periods", 30))
        controls_layout.addWidget(self.pvc_daily_spinbox)

        main_layout.addLayout(controls_layout)

        splitter = QSplitter(Qt.Vertical)
        main_layout.addWidget(splitter)

        # --- Top Pane: Value Change Graphs ---
        top_pane_widget = QWidget()
        top_pane_layout = QHBoxLayout(top_pane_widget)
        splitter.addWidget(top_pane_widget)

        graph_configs = [
            ("Annual", "pvc_annual_graph", ""),
            ("Monthly", "pvc_monthly_graph", ""),
            ("Weekly", "pvc_weekly_graph", ""),
            ("Daily", "pvc_daily_graph", ""),
        ]

        # graph_configs = [
        #     ("Annual", "pvc_annual_graph", "Annual Value Change Graph"),
        #     ("Monthly", "pvc_monthly_graph", "Monthly Value Change Graph"),
        #     ("Weekly", "pvc_weekly_graph", "Weekly Value Change Graph"),
        #     ("Daily", "pvc_daily_graph", "Daily Value Change Graph"),
        # ]

        for period_name, attr_prefix, group_title in graph_configs:
            group_box = QGroupBox(group_title)
            group_layout = QVBoxLayout(group_box)
            fig = Figure(figsize=(4.5, 2.5), dpi=CHART_DPI)
            ax = fig.add_subplot(111)
            canvas = FigureCanvas(fig)
            canvas_pixel_height = int(fig.get_figheight() * fig.dpi)
            canvas.setFixedHeight(canvas_pixel_height)
            canvas.setObjectName(f"{attr_prefix}_canvas")
            setattr(self, f"{attr_prefix}_fig", fig)
            setattr(self, f"{attr_prefix}_ax", ax)
            setattr(self, f"{attr_prefix}_canvas", canvas)
            group_layout.addWidget(canvas)
            top_pane_layout.addWidget(group_box)

        # --- Middle Pane: Performance Bar Graphs ---
        self.bar_charts_frame = QFrame()
        self.bar_charts_frame.setObjectName("BarChartsFrame")
        splitter.addWidget(self.bar_charts_frame)
        self._init_bar_charts_frame_widgets()

        # --- Bottom Pane: Tables ---
        bottom_pane_widget = QWidget()
        bottom_pane_layout = QHBoxLayout(bottom_pane_widget)
        splitter.addWidget(bottom_pane_widget)

        table_configs = [
            ("Annual", "pvc_annual_table", "Annual Value Change Table"),
            ("Monthly", "pvc_monthly_table", "Monthly Value Change Table"),
            ("Weekly", "pvc_weekly_table", "Weekly Value Change Table"),
            ("Daily", "pvc_daily_table", "Daily Value Change Table"),
        ]

        for period_name, attr_prefix, group_title in table_configs:
            group_box = QGroupBox(group_title)
            group_layout = QVBoxLayout(group_box)
            table_view = QTableView()
            table_view.setObjectName(f"{attr_prefix}_view")
            table_view.setAlternatingRowColors(True)
            table_view.setSelectionBehavior(QTableView.SelectRows)
            table_view.setWordWrap(False)
            table_view.setSortingEnabled(True)
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
            table_view.verticalHeader().setVisible(False)
            model = PandasModel(parent=self, log_mode=True)
            table_view.setModel(model)
            setattr(self, f"{attr_prefix}_view", table_view)
            setattr(self, f"{attr_prefix}_model", model)
            group_layout.addWidget(table_view)
            bottom_pane_layout.addWidget(group_box)

        # Set initial relative sizes for the splitter panes (Top, Middle, Bottom)
        splitter.setSizes([300, 200, 500])

        self.main_tab_widget.addTab(self.periodic_value_change_tab, "Asset Change")

    def _init_transactions_log_tab_widgets(self):
        """Initializes widgets for the Transactions Log tab."""
        # This method will be filled in _init_ui_widgets
        # It's called from there to set up the content of self.transactions_log_tab
        pass

    def _init_asset_allocation_tab_widgets(self):
        """Initializes widgets for the Asset Allocation tab."""
        # This method will be filled in _init_ui_widgets
        # It's called from there to set up the content of self.asset_allocation_tab
        pass

    def _init_header_frame_widgets(self):
        """Initializes widgets for the Header Frame."""
        if not hasattr(self, "header_frame"):
            return

        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(15, 8, 15, 8)
        self.main_title_label = QLabel("📈 <b>Investa Portfolio Dashboard</b> ✨")
        self.main_title_label.setObjectName("MainTitleLabel")
        self.main_title_label.setTextFormat(Qt.RichText)

        # --- ADDED: Date Label ---
        self.date_label = QLabel("")
        self.date_label.setObjectName("DateLabel")
        self.date_label.setTextFormat(Qt.RichText)
        self.date_label.setStyleSheet("font-size: 14pt;")
        self.date_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        self.header_info_label = QLabel("<i>Initializing...</i>")
        # --- ADDED: FX Rate Label in Header ---
        self.exchange_rate_display_label = QLabel("")
        self.exchange_rate_display_label.setObjectName("ExchangeRateLabel")
        self.exchange_rate_display_label.setVisible(False)
        self.exchange_rate_display_label.setTextFormat(Qt.RichText)
        self.exchange_rate_display_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # --- END ADDED ---

        self.header_info_label.setObjectName("HeaderInfoLabel")
        self.header_info_label.setTextFormat(Qt.RichText)
        self.header_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.main_title_label)
        header_layout.addSpacing(15)  # Add a small spacer
        header_layout.addWidget(self.date_label)
        header_layout.addStretch(1)  # Move stretch to after the date
        header_layout.addWidget(self.exchange_rate_display_label)
        # Add a small separator/spacer
        header_layout.addSpacing(15)
        header_layout.addWidget(self.header_info_label)

    def _init_controls_frame_widgets(self):
        """Initializes widgets for the Controls Frame."""
        if not hasattr(self, "controls_frame"):
            return
        controls_layout = QHBoxLayout(self.controls_frame)  # Use self.controls_frame
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(4)

        # Helper function to create a vertical separator
        def create_separator():
            separator = QFrame()
            separator.setFrameShape(QFrame.Shape.VLine)
            separator.setFrameShadow(QFrame.Shadow.Sunken)
            return separator

        # Buttons
        self.manage_transactions_button = QPushButton(
            "Manage Tx"
        )  # Keep button instance for menu/toolbar
        self.manage_transactions_button.setObjectName("ManageTransactionsButton")
        self.manage_transactions_button.setIcon(
            self.style().standardIcon(QStyle.SP_DialogSaveButton)
        )  # Example icon
        self.manage_transactions_button.setToolTip(
            "Edit or delete existing transactions"
        )
        # controls_layout.addWidget(self.manage_transactions_button) # REMOVED from controls bar

        self.view_ignored_button = QPushButton("View Log")
        self.view_ignored_button.setObjectName("ViewIgnoredButton")
        self.view_ignored_button.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        )  # Example icon
        self.view_ignored_button.setToolTip(
            "View transactions ignored during the last calculation"
        )
        self.view_ignored_button.setEnabled(False)  # Initially disabled
        # controls_layout.addWidget(self.view_ignored_button) # REMOVED from left group

        # --- Separator 1 ---
        controls_layout.addWidget(create_separator())

        self.account_select_button = QPushButton("Accounts")
        self.account_select_button.setObjectName("AccountSelectButton")
        self.account_select_button.setMinimumWidth(120)
        controls_layout.addWidget(self.account_select_button)
        self.update_accounts_button = QPushButton("Update Accounts")
        self.update_accounts_button.setObjectName("UpdateAccountsButton")
        self.update_accounts_button.setIcon(
            self.style().standardIcon(QStyle.SP_DialogApplyButton)
        )
        self.update_accounts_button.setToolTip(
            "Apply selected accounts and recalculate"
        )
        controls_layout.addWidget(self.update_accounts_button)

        # --- Separator 2 (Between Account Controls and Filters) ---
        controls_layout.addWidget(create_separator())

        # Filters & Combos
        controls_layout.addWidget(QLabel("Currency:"))
        self.currency_combo = QComboBox()
        self.currency_combo.setObjectName("CurrencyCombo")
        # Populate from config's user_currencies
        user_selected_currencies = self.config.get(
            "user_currencies", COMMON_CURRENCIES.copy()
        )
        if (
            not user_selected_currencies
        ):  # Ensure there's always something, fallback to COMMON_CURRENCIES
            user_selected_currencies = COMMON_CURRENCIES.copy()
        self.currency_combo.addItems(user_selected_currencies)
        self.currency_combo.setCurrentText(self.config.get("display_currency", "USD"))
        self.currency_combo.setMinimumWidth(80)
        controls_layout.addWidget(self.currency_combo)
        self.show_closed_check = QCheckBox("Show Closed")
        self.show_closed_check.setObjectName("ShowClosedCheck")
        self.show_closed_check.setChecked(self.config.get("show_closed", False))
        controls_layout.addWidget(self.show_closed_check)

        self.group_by_sector_check = QCheckBox("Group by Sector")
        self.group_by_sector_check.setObjectName("GroupBySectorCheck")
        self.group_by_sector_check.setChecked(self.config.get("group_by_sector", False))
        controls_layout.addWidget(self.group_by_sector_check)

        # --- Separator 3 (Between Account/Display and Graph Controls) ---
        controls_layout.addWidget(create_separator())

        # --- Graph Controls Group ---
        graph_controls_group = QGroupBox("")
        graph_controls_group.setObjectName("GraphControlsGroup")
        graph_v_layout = QVBoxLayout(graph_controls_group)
        graph_v_layout.setContentsMargins(2, 2, 2, 2)
        graph_v_layout.setSpacing(4)

        # Row 1: Date editors, interval, benchmarks, update button
        graph_controls_row1_layout = QHBoxLayout()
        graph_controls_row1_layout.addWidget(QLabel("Range:"))
        self.graph_start_date_edit = QDateEdit()
        self.graph_start_date_edit.setObjectName("GraphDateEdit")
        self.graph_start_date_edit.setCalendarPopup(True)
        self.graph_start_date_edit.setDisplayFormat("yyyy-MM-dd")
        logging.debug(f"DEBUG: _init_graph_controls setting start date from config: {self.config.get('graph_start_date')}")
        self.graph_start_date_edit.setDate(
            QDate.fromString(self.config.get("graph_start_date"), "yyyy-MM-dd")
        )
        graph_controls_row1_layout.addWidget(self.graph_start_date_edit)
        graph_controls_row1_layout.addWidget(QLabel("to"))
        self.graph_end_date_edit = QDateEdit()
        self.graph_end_date_edit.setObjectName("GraphDateEdit")
        self.graph_end_date_edit.setCalendarPopup(True)
        self.graph_end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.graph_end_date_edit.setDate(
            QDate.fromString(self.config.get("graph_end_date"), "yyyy-MM-dd")
        )
        graph_controls_row1_layout.addWidget(self.graph_end_date_edit)
        # --- MOVED: Preset dropdown is now on the main row ---
        self.date_preset_combo = QComboBox()
        self.date_preset_combo.setObjectName("DatePresetCombo")
        self.date_preset_combo.addItems(
            [
                "Presets...",
                "1W",
                "MTD",
                "1M",
                "3M",  # <-- ADDED
                "6M",
                "YTD",
                "1Y",
                "3Y",
                "5Y",
                "10Y",
                "All",
            ]
        )
        self.date_preset_combo.setToolTip("Select a preset date range")
        self.date_preset_combo.setMinimumWidth(90)
        self.date_preset_combo.textActivated.connect(self._on_preset_selected)
        graph_controls_row1_layout.addWidget(self.date_preset_combo)
        # --- END MOVED ---

        self.benchmark_select_button = QPushButton()
        self.benchmark_select_button.setObjectName("BenchmarkSelectButton")
        self.benchmark_select_button.setMinimumWidth(100)
        graph_controls_row1_layout.addWidget(self.benchmark_select_button)
        self.graph_update_button = QPushButton("Update Graphs")
        self.graph_update_button.setObjectName("GraphUpdateButton")
        self.graph_update_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self.graph_update_button.setToolTip(
            "Recalculate and redraw performance graphs."
        )
        # Connect date edits to clear preset selection (MOVED HERE)
        self.graph_start_date_edit.dateChanged.connect(
            self._clear_preset_button_selection
        )
        self.graph_end_date_edit.dateChanged.connect(
            self._clear_preset_button_selection
        )

        graph_controls_row1_layout.addWidget(self.graph_update_button)
        graph_v_layout.addLayout(graph_controls_row1_layout)

        # The preset dropdown has been moved to the row above. This layout is no longer needed.

        controls_layout.addWidget(graph_controls_group)

        # --- Separator 4 ---
        controls_layout.addWidget(create_separator())

        # Spacer & Right Aligned Controls
        controls_layout.addStretch(1)
        controls_layout.addWidget(self.view_ignored_button)  # ADDED to right group
        self.refresh_button = QPushButton("Refresh All")
        self.refresh_button.setObjectName("RefreshButton")
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        # controls_layout.addWidget(self.refresh_button) # REMOVED from controls bar

    def _init_summary_grid_widgets(self, summary_graphs_layout: QHBoxLayout):
        """Initializes the summary grid widgets and adds them to the provided layout."""
        if not hasattr(self, "summary_and_graphs_frame"):
            return

        summary_graphs_layout.setContentsMargins(10, 5, 10, 5)
        summary_graphs_layout.setSpacing(10)
        # Summary Grid
        summary_grid_widget = QWidget()
        summary_layout = QGridLayout(summary_grid_widget)
        summary_layout.setContentsMargins(10, 10, 10, 10)
        summary_layout.setHorizontalSpacing(15)
        summary_layout.setVerticalSpacing(30)
        self.summary_net_value = self.create_summary_item("Net Value", True)
        (
            self.summary_day_change_label,
            self.summary_day_change_value,
            self.summary_day_change_pct,
        ) = self.create_summary_item("Day's G/L", True, has_percentage=True)
        self.summary_total_gain = self.create_summary_item("Total G/L")
        self.summary_realized_gain = self.create_summary_item("Realized G/L")
        self.summary_unrealized_gain = self.create_summary_item("Unrealized G/L")
        self.summary_dividends = self.create_summary_item("Dividends")
        self.summary_commissions = self.create_summary_item("Fees")
        self.summary_cash = self.create_summary_item("Cash Balance")
        self.summary_total_return_pct = self.create_summary_item("Total Ret %")
        self.summary_annualized_twr = self.create_summary_item("Ann. TWR %")
        # --- Corrected initialization for FX G/L labels and values ---
        self.summary_fx_gl_abs_label, self.summary_fx_gl_abs_value = (
            self.create_summary_item("FX Gain/Loss")
        )
        self.summary_fx_gl_pct_label, self.summary_fx_gl_pct_value = (
            self.create_summary_item("FX G/L %")
        )

        summary_layout.addWidget(self.summary_net_value[0], 0, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_net_value[1], 0, 1)
        # Container for Daily G/L Value and Percentage
        daily_gl_value_pct_container = QVBoxLayout()
        daily_gl_value_pct_container.setContentsMargins(0, 0, 0, 0)
        daily_gl_value_pct_container.setSpacing(0)
        daily_gl_value_pct_container.addWidget(self.summary_day_change_value)
        daily_gl_value_pct_container.addWidget(self.summary_day_change_pct)

        summary_layout.addWidget(self.summary_day_change_label, 0, 2, Qt.AlignRight)
        summary_layout.addLayout(daily_gl_value_pct_container, 0, 3)
        summary_layout.addWidget(self.summary_total_gain[0], 1, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_total_gain[1], 1, 1)
        summary_layout.addWidget(self.summary_realized_gain[0], 1, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_realized_gain[1], 1, 3)
        summary_layout.addWidget(self.summary_unrealized_gain[0], 2, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_unrealized_gain[1], 2, 1)
        summary_layout.addWidget(self.summary_dividends[0], 2, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_dividends[1], 2, 3)
        summary_layout.addWidget(self.summary_commissions[0], 3, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_commissions[1], 3, 1)
        summary_layout.addWidget(self.summary_cash[0], 3, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_cash[1], 3, 3)
        summary_layout.addWidget(self.summary_total_return_pct[0], 4, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_total_return_pct[1], 4, 1)
        summary_layout.addWidget(self.summary_annualized_twr[0], 4, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_annualized_twr[1], 4, 3)
        # Add new FX G/L to row 5 using corrected variable names
        summary_layout.addWidget(self.summary_fx_gl_abs_label, 5, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_fx_gl_abs_value, 5, 1)
        summary_layout.addWidget(self.summary_fx_gl_pct_label, 5, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_fx_gl_pct_value, 5, 3)
        summary_layout.setColumnStretch(1, 1)
        summary_layout.setColumnStretch(3, 1)
        # Apply stretch to the row *after* the last content row (row index 5 is last content)
        # summary_layout.rowCount() will be 6 after adding items to row 5.
        summary_layout.setRowStretch(summary_layout.rowCount(), 1)
        
        # --- ADDED: Risk Metrics Button ---
        self.view_risk_metrics_button = QPushButton("View Risk Metrics")
        self.view_risk_metrics_button.setIcon(QIcon.fromTheme("utilities-system-monitor"))
        self.view_risk_metrics_button.setToolTip("Calculate and view risk metrics (Drawdown, Sharpe, etc.) based on historical performance.")
        self.view_risk_metrics_button.clicked.connect(self.show_risk_metrics_dialog)
        # Add to the bottom, spanning all columns
        summary_layout.addWidget(self.view_risk_metrics_button, 6, 0, 1, 4)
        # --- END ADDED ---

        summary_graphs_layout.addWidget(summary_grid_widget, 9)

    def _init_performance_graph_widgets(self, summary_graphs_layout: QHBoxLayout):
        """Initializes the performance graph widgets and adds them to the provided layout."""
        perf_graphs_container_widget = QWidget()
        if not hasattr(self, "summary_and_graphs_frame"):
            return
        perf_graphs_container_widget.setObjectName("PerfGraphsContainer")
        # This layout holds the two vertical (Graph+Toolbar) widgets side-by-side
        perf_graphs_main_layout = QHBoxLayout(perf_graphs_container_widget)
        perf_graphs_main_layout.setContentsMargins(0, 0, 0, 0)
        perf_graphs_main_layout.setSpacing(0)  # Spacing between the two graph columns

        # --- Restructured Performance Graphs with Tabs ---
        # Create a QTabWidget to hold the graphs
        self.perf_graphs_tab_widget = QTabWidget()
        self.perf_graphs_tab_widget.setObjectName("PerfGraphsTabWidget")
        self.perf_graphs_tab_widget.setTabPosition(QTabWidget.South) # Move tabs to bottom
        # Increase tab width to show full titles
        self.perf_graphs_tab_widget.setStyleSheet(
            "QTabBar::tab { min-width: 150px; padding: 5px; }"
        )

        # --- Tab 1: Standard Performance (Return & Value) ---
        perf_std_tab = QWidget()
        perf_std_layout = QHBoxLayout(perf_std_tab)
        perf_std_layout.setContentsMargins(0, 0, 0, 0)
        perf_std_layout.setSpacing(0)

        # -- Return Graph (Left Column) --
        self.perf_return_fig = Figure(figsize=PERF_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.perf_return_ax = self.perf_return_fig.add_subplot(111)
        self.perf_return_canvas = FigureCanvas(self.perf_return_fig)
        self.perf_return_canvas.setObjectName("PerfReturnCanvas")
        self.perf_return_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.perf_return_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        perf_std_layout.addWidget(self.perf_return_canvas, 1)

        # -- Absolute Value Graph (Right Column) --
        self.abs_value_fig = Figure(figsize=PERF_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.abs_value_ax = self.abs_value_fig.add_subplot(111)
        self.abs_value_canvas = FigureCanvas(self.abs_value_fig)
        self.abs_value_canvas.setObjectName("AbsValueCanvas")
        self.abs_value_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        self.abs_value_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        perf_std_layout.addWidget(self.abs_value_canvas, 1)
        
        self.perf_graphs_tab_widget.addTab(perf_std_tab, "Return & Value")

        # --- Tab 2: Drawdown ---
        drawdown_tab = QWidget()
        drawdown_layout = QVBoxLayout(drawdown_tab) # Vertical layout for single large chart
        drawdown_layout.setContentsMargins(0, 0, 0, 0)
        
        self.drawdown_fig = Figure(figsize=PERF_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.drawdown_ax = self.drawdown_fig.add_subplot(111)
        self.drawdown_canvas = FigureCanvas(self.drawdown_fig)
        self.drawdown_canvas.setObjectName("DrawdownCanvas")
        self.drawdown_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        # Context menu for drawdown (optional, can reuse graph context menu logic)
        self.drawdown_canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.drawdown_canvas.customContextMenuRequested.connect(
            lambda pos: self._show_graph_context_menu(
                pos, self.drawdown_fig, "Drawdown_Graph"
            )
        )
        
        drawdown_layout.addWidget(self.drawdown_canvas)
        self.perf_graphs_tab_widget.addTab(drawdown_tab, "Drawdown")
        self.perf_graphs_tab_widget.setTabToolTip(1, "Visualizes the percentage decline from the historical peak value (Drawdown).")

        # --- Drawdown Hover Event ---
        self.drawdown_canvas.mpl_connect("motion_notify_event", self._on_drawdown_hover)
        self.drawdown_annot = None # Will be initialized in _update_drawdown_chart

        # Add the tab widget to the main layout
        perf_graphs_main_layout.addWidget(self.perf_graphs_tab_widget)
        
        # --- Restore Saved Tab Index ---
        saved_tab_index = self.config.get("perf_graph_tab_index", 0)
        if isinstance(saved_tab_index, int) and 0 <= saved_tab_index < self.perf_graphs_tab_widget.count():
            self.perf_graphs_tab_widget.setCurrentIndex(saved_tab_index)
        # --- End Restore ---
        
        # Add the main performance graphs container to the summary/graphs frame
        summary_graphs_layout.addWidget(perf_graphs_container_widget, 20)
        # --- End Performance Graphs Container Modification ---

    def _init_bar_charts_frame_widgets(self):
        """Initializes widgets for the Bar Charts Frame."""
        if not hasattr(self, "bar_charts_frame"):
            return
        bar_charts_main_layout = QHBoxLayout(
            self.bar_charts_frame
        )  # Use self.bar_charts_frame
        bar_charts_main_layout.setContentsMargins(0, 0, 0, 0)
        bar_charts_main_layout.setSpacing(10)

        # Function to create a single bar chart widget
        def create_bar_chart_widget(  # Modified signature
            title,
            canvas_attr_name,
            fig_attr_name,
            ax_attr_name,
        ):
            widget = QWidget()
            layout = QVBoxLayout(widget)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(2)

            # --- Title and Input Row ---
            title_input_layout = QHBoxLayout()
            title_input_layout.setContentsMargins(0, 0, 0, 0)
            title_input_layout.setSpacing(5)

            title_label = QLabel(f"<b>{title}</b>")
            title_label.setObjectName("BarChartTitleLabel")
            title_input_layout.addWidget(title_label)
            title_input_layout.addStretch()  # Push input to the right

            layout.addLayout(title_input_layout)  # Add the title/input row

            setattr(
                self, fig_attr_name, Figure(figsize=(4, 2.5), dpi=CHART_DPI)
            )  # Smaller figsize
            fig = getattr(self, fig_attr_name)
            setattr(self, ax_attr_name, fig.add_subplot(111))
            setattr(self, canvas_attr_name, FigureCanvas(fig))
            canvas = getattr(self, canvas_attr_name)
            canvas.setObjectName(canvas_attr_name)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            layout.addWidget(canvas, 1)
            # No toolbar for bar charts for now
            return widget

        # Create the three bar chart widgets
        self.annual_bar_widget = create_bar_chart_widget(
            "Annual Returns",
            "annual_bar_canvas",
            "annual_bar_fig",
            "annual_bar_ax",
        )
        self.monthly_bar_widget = create_bar_chart_widget(
            "Monthly Returns",
            "monthly_bar_canvas",
            "monthly_bar_fig",
            "monthly_bar_ax",
        )
        self.weekly_bar_widget = create_bar_chart_widget(
            "Weekly Returns",
            "weekly_bar_canvas",
            "weekly_bar_fig",
            "weekly_bar_ax",
        )
        self.daily_bar_widget = create_bar_chart_widget(
            "Daily Returns",
            "daily_bar_canvas",
            "daily_bar_fig",
            "daily_bar_ax",
        )

        # Add widgets to the layout
        bar_charts_main_layout.addWidget(self.annual_bar_widget, 1)
        bar_charts_main_layout.addWidget(self.monthly_bar_widget, 1)
        bar_charts_main_layout.addWidget(self.weekly_bar_widget, 1)
        bar_charts_main_layout.addWidget(self.daily_bar_widget, 1)
        # --- End Bar Charts Frame Setup ---

    def _init_pie_chart_widgets(self, content_layout: QHBoxLayout):
        """Initializes pie chart widgets and adds them to the provided content_layout."""
        if not hasattr(self, "content_frame"):
            return
        pie_charts_container_widget = QWidget()
        pie_charts_container_widget.setObjectName("PieChartsContainer")
        pie_charts_layout = QVBoxLayout(pie_charts_container_widget)
        pie_charts_layout.setContentsMargins(0, 0, 0, 0)
        pie_charts_layout.setSpacing(10)
        account_chart_widget = QWidget()
        account_chart_layout = QVBoxLayout(account_chart_widget)
        account_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.account_pie_title_label = QLabel(
            "<b>Value by Account</b>"
        )  # <-- RESTORED Title
        self.account_pie_title_label.setObjectName("AccountPieTitleLabel")
        self.account_pie_title_label.setTextFormat(Qt.RichText)
        account_chart_layout.addWidget(
            self.account_pie_title_label, alignment=Qt.AlignCenter
        )  # <-- RESTORED Title
        self.account_fig = Figure(figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.account_ax = self.account_fig.add_subplot(111)
        self.account_canvas = FigureCanvas(self.account_fig)
        self.account_canvas.setObjectName("AccountPieCanvas")
        self.account_canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        account_chart_layout.addWidget(self.account_canvas)
        pie_charts_layout.addWidget(account_chart_widget)
        holdings_chart_widget = QWidget()
        holdings_chart_layout = QVBoxLayout(holdings_chart_widget)
        holdings_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.holdings_pie_title_label = QLabel("<b>Value by Holding</b>")
        self.holdings_pie_title_label.setObjectName("HoldingsPieTitleLabel")
        self.holdings_pie_title_label.setTextFormat(Qt.RichText)
        holdings_chart_layout.addWidget(
            self.holdings_pie_title_label, alignment=Qt.AlignCenter
        )  # <-- RESTORED Title AddWidget Call
        self.holdings_fig = Figure(figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.holdings_ax = self.holdings_fig.add_subplot(111)
        self.holdings_canvas = FigureCanvas(self.holdings_fig)
        self.holdings_canvas.setObjectName("HoldingsPieCanvas")
        self.holdings_canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        holdings_chart_layout.addWidget(self.holdings_canvas)
        pie_charts_layout.addWidget(holdings_chart_widget)
        content_layout.addWidget(pie_charts_container_widget, 1)

    def _init_table_panel_widgets(self, content_layout: QHBoxLayout):
        """Initializes the table panel widgets and adds them to the provided content_layout."""
        table_panel = QFrame()
        if not hasattr(self, "content_frame"):
            return
        table_panel.setObjectName("TablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(5)

        # --- Table Header Layout (MODIFIED for Filters) ---
        table_header_and_filter_layout = (
            QVBoxLayout()
        )  # Use QVBoxLayout to stack title and filters
        table_header_and_filter_layout.setContentsMargins(5, 5, 5, 0)  # Adjust margins
        table_header_and_filter_layout.setSpacing(4)  # Add spacing

        # Title Row (Original HBox)
        table_title_layout = QHBoxLayout()
        self.table_title_label_left = QLabel("")
        self.table_title_label_left.setObjectName("TableScopeLabel")
        self.table_title_label_left.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table_title_label_right = QLabel("Holdings Detail")
        self.table_title_label_right.setObjectName("TableTitleLabel")
        self.table_title_label_right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        table_title_layout.addWidget(self.table_title_label_left)
        table_title_layout.addStretch(1)
        table_title_layout.addWidget(self.table_title_label_right)
        table_header_and_filter_layout.addLayout(table_title_layout)  # Add title row

        # Filter Row (New HBox)
        table_filter_layout = QHBoxLayout()
        table_filter_layout.addWidget(QLabel("Filter:"))
        self.filter_symbol_table_edit = QLineEdit()
        self.filter_symbol_table_edit.setPlaceholderText("Symbol contains...")
        self.filter_symbol_table_edit.setClearButtonEnabled(True)
        table_filter_layout.addWidget(
            self.filter_symbol_table_edit, 1
        )  # Give symbol more stretch
        self.filter_account_table_edit = QLineEdit()
        self.filter_account_table_edit.setPlaceholderText("Account contains...")
        self.filter_account_table_edit.setClearButtonEnabled(True)
        table_filter_layout.addWidget(
            self.filter_account_table_edit, 1
        )  # Give account more stretch # REMOVED Apply Button
        # self.apply_table_filter_button = QPushButton("Apply")
        # self.apply_table_filter_button.setToolTip("Apply table filters")
        # self.apply_table_filter_button.setObjectName("ApplyTableFilterButton")
        # table_filter_layout.addWidget(self.apply_table_filter_button)
        self.clear_table_filter_button = QPushButton(
            "Clear"
        )  # <-- MOVED/ADDED CREATION HERE
        # self.clear_table_filter_button.setToolTip("Clear table filters")
        self.clear_table_filter_button.setToolTip("Clear table filters")
        self.clear_table_filter_button.setObjectName("ClearTableFilterButton")
        table_filter_layout.addWidget(self.clear_table_filter_button)
        table_header_and_filter_layout.addLayout(table_filter_layout)  # Add filter row
        # --- End Header/Filter Modification ---

        # Add the combined header/filter layout to the main table layout
        table_layout.addLayout(table_header_and_filter_layout)

        # --- MODIFIED: Setup for dual table view (frozen + scrollable) ---
        self.table_model = PandasModel(parent=self)

        # Main (scrollable) table view
        self.table_view = QTableView()
        self.table_view.setObjectName("HoldingsTable")
        self.table_view.setModel(self.table_model)
        self.table_view.setItemDelegate(
            GroupHeaderDelegate(self.table_view, theme=self.current_theme)
        )
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setWordWrap(False)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionsMovable(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Frozen table view for fixed columns
        self.frozen_table_view = QTableView()
        self.frozen_table_view.setObjectName("FrozenHoldingsTable")
        self.frozen_table_view.setModel(self.table_model)
        self.frozen_table_view.setItemDelegate(
            GroupHeaderDelegate(self.frozen_table_view, theme=self.current_theme)
        )
        self.frozen_table_view.setAlternatingRowColors(True)
        self.frozen_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.frozen_table_view.setWordWrap(False)
        self.frozen_table_view.setSortingEnabled(True)
        self.frozen_table_view.verticalHeader().setVisible(False)
        self.frozen_table_view.setFocusPolicy(Qt.NoFocus)
        self.frozen_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeToContents
        )
        self.frozen_table_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.frozen_table_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # --- ADDED: CONTEXT MENU SETUP for both tables ---
        self.table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table_view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.frozen_table_view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.frozen_table_view.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        # --- END ADDED ---

        # Layout for the two tables
        table_container_layout = QHBoxLayout()
        table_container_layout.setSpacing(0)
        table_container_layout.setContentsMargins(0, 0, 0, 0)
        table_container_layout.addWidget(self.frozen_table_view)
        table_container_layout.addWidget(
            self.table_view, 1
        )  # scrollable part takes stretch

        table_layout.addLayout(table_container_layout, 1)
        content_layout.addWidget(table_panel, 4)

        # Synchronize vertical scrolling
        self.frozen_table_view.verticalScrollBar().valueChanged.connect(
            self.table_view.verticalScrollBar().setValue
        )
        self.table_view.verticalScrollBar().valueChanged.connect(
            self.frozen_table_view.verticalScrollBar().setValue
        )

        # Synchronize selection - using explicit slots to prevent recursion and warnings
        self.table_view.selectionModel().selectionChanged.connect(
            self._sync_main_to_frozen_selection
        )
        self.frozen_table_view.selectionModel().selectionChanged.connect(
            self._sync_frozen_to_main_selection
        )

        self.view_ignored_button.clicked.connect(self.show_ignored_log)

    def _init_ui_widgets(self):
        """Orchestrates the initialization of all UI widgets within their frames."""
        logging.debug(
            "DEBUG PRINT: _init_ui_widgets method has been entered."
        )  # <-- ADD THIS PRINT STATEMENT
        logging.debug("--- _init_ui_widgets: START ---")
        self._init_header_frame_widgets()
        logging.debug("--- _init_ui_widgets: After _init_header_frame_widgets ---")
        self._init_controls_frame_widgets()
        logging.debug("--- _init_ui_widgets: After _init_controls_frame_widgets ---")
        # Summary & Graphs Frame (needs its own layout passed to helpers)
        summary_graphs_layout = QHBoxLayout(self.summary_and_graphs_frame)
        summary_graphs_layout.setContentsMargins(
            0, 0, 0, 0
        )  # Let content widgets handle margins
        self._init_summary_grid_widgets(summary_graphs_layout)
        self._init_performance_graph_widgets(summary_graphs_layout)
        logging.debug(
            "--- _init_ui_widgets: After _init_summary_grid_widgets & _init_performance_graph_widgets ---"
        )
        # Bar Charts Frame (Periodic Returns Tab)
        self._init_bar_charts_frame_widgets()
        logging.debug("--- _init_ui_widgets: After _init_bar_charts_frame_widgets ---")
        # Content Frame (needs its own layout passed to helpers)
        content_layout = QHBoxLayout(self.content_frame)
        content_layout.setContentsMargins(
            0, 0, 0, 0
        )  # Let content widgets handle margins
        self._init_pie_chart_widgets(content_layout)
        logging.debug("--- _init_ui_widgets: After _init_pie_chart_widgets ---")
        self._init_table_panel_widgets(content_layout)
        # Initialize Capital Gains tab widgets
        self._init_capital_gains_tab_widgets()
        logging.debug("--- _init_ui_widgets: After _init_table_panel_widgets ---")

        # --- Tab: Asset Change ---
        self._init_asset_change_tab_widgets()

        # --- Tab 4: Dividend History ---
        # (This will become Tab 4 after we add Transactions Log and Asset Allocation)
        logging.debug("--- _init_ui_widgets: Entering Dividend History Tab setup ---")
        self.dividend_history_tab = QWidget()
        dividend_history_layout = QVBoxLayout(self.dividend_history_tab)
        dividend_history_layout.setContentsMargins(10, 10, 10, 10)  # Add some padding
        dividend_history_layout.setSpacing(8)

        # --- TEMPORARY DIAGNOSTIC: Simplify tab content ---
        # Comment out the original complex content and add a simple label
        # to see if the tab itself can be rendered.
        # simple_test_label = QLabel("Dividend History Tab - Test Content")
        # simple_test_label.setAlignment(Qt.AlignCenter)
        # dividend_history_layout.addWidget(simple_test_label)
        # --- END TEMPORARY DIAGNOSTIC ---

        logging.debug("--- _init_ui_widgets: Dividend History Tab layout created ---")
        # Controls for Dividend Bar Graph
        dividend_controls_layout = QHBoxLayout()
        dividend_controls_layout.addWidget(QLabel("Aggregate by:"))
        logging.debug(
            "--- _init_ui_widgets: BEFORE self.dividend_period_combo creation ---"
        )
        # """ # Original Content Start (Commented out for testing) # Keep this line if you want to easily re-comment
        self.dividend_period_combo = QComboBox()
        logging.debug(
            "--- _init_ui_widgets: AFTER self.dividend_period_combo creation ---"
            "--- _init_ui_widgets: AFTER self.dividend_period_combo creation ---"
        )
        self.dividend_period_combo.setObjectName("DividendPeriodCombo")
        self.dividend_period_combo.addItems(["Annual", "Quarterly", "Monthly"])
        # --- MODIFIED: Set dividend period combo from config ---
        logging.debug(
            f"_INIT_UI_WIDGETS: self.config['dividend_agg_period'] from load_config = {self.config.get('dividend_agg_period')}"
        )
        saved_div_agg_period = self.config.get("dividend_agg_period", "Annual")
        logging.debug(
            f"_INIT_UI_WIDGETS: saved_div_agg_period = {saved_div_agg_period}"
        )
        dividend_controls_layout.addWidget(self.dividend_period_combo)

        dividend_controls_layout.addWidget(QLabel("  Periods to Show:"))
        self.dividend_period_combo.setMinimumWidth(100)  # <-- SET MINIMUM WIDTH
        self.dividend_periods_spinbox = QSpinBox()
        # ... (rest of spinbox setup) ...
        logging.debug(
            "--- _init_ui_widgets: AFTER self.dividend_periods_spinbox creation ---"
        )
        self.dividend_periods_spinbox.setObjectName("DividendPeriodsSpinbox")
        self.dividend_periods_spinbox.setMinimum(1)

        def update_dividend_max_periods(period_text):
            if period_text == "Annual":
                self.dividend_periods_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_ANNUAL)
            elif period_text == "Quarterly":
                self.dividend_periods_spinbox.setMaximum(
                    BAR_CHART_MAX_PERIODS_QUARTERLY
                )
            elif period_text == "Monthly":
                self.dividend_periods_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_MONTHLY)

        self.dividend_period_combo.currentTextChanged.connect(
            update_dividend_max_periods
        )
        update_dividend_max_periods(self.dividend_period_combo.currentText())
        # Initialize based on default period (Annual)
        # --- MODIFIED: Set dividend periods spinbox from config ---
        # The _update_dividend_spinbox_default will be called once due to setCurrentText above,
        # then we set the specific saved value.
        saved_div_periods_to_show = self.config.get(
            "dividend_periods_to_show", 10
        )  # Default to 10 if key missing
        # --- END MODIFICATION ---
        self.dividend_periods_spinbox.setFixedWidth(60)
        dividend_controls_layout.addWidget(self.dividend_periods_spinbox)

        # --- ADDED: Estimated Annual Dividend Income Display ---
        # Moved to the same line as "Aggregate by:"
        dividend_controls_layout.addStretch()  # This stretch pushes "Aggregate by:" to the left and "Estimated Dividend Income" to the right
        self.est_annual_income_label = QLabel("<b>Est. Next 12m Dividend Income:</b>")
        self.est_annual_income_label.setObjectName("EstAnnualIncomeLabel")
        dividend_controls_layout.addWidget(self.est_annual_income_label)

        self.est_annual_income_value_label = QLabel("N/A")
        self.est_annual_income_value_label.setObjectName("EstAnnualIncomeValueLabel")
        # Set font size for the value label
        self.est_annual_income_value_label.setStyleSheet("font-size: 16pt;")
        dividend_controls_layout.addWidget(self.est_annual_income_value_label)
        # No stretch needed here, as the previous stretch will push it to the right

        dividend_history_layout.addLayout(dividend_controls_layout)
        logging.debug("--- _init_ui_widgets: Dividend controls layout added ---")

        # Dividend Bar Graph Canvas
        self.dividend_bar_fig = Figure(
            figsize=(7, 3), dpi=CHART_DPI
        )  # Adjust size as needed
        # --- Set combo box text AFTER spinbox is created and its value potentially set by _update_dividend_spinbox_default ---
        if saved_div_agg_period in ["Annual", "Quarterly", "Monthly"]:
            self.dividend_period_combo.setCurrentText(saved_div_agg_period)
        else:
            self.dividend_period_combo.setCurrentText("Annual")  # Fallback
        logging.debug(
            f"_INIT_UI_WIDGETS: After setCurrentText('{saved_div_agg_period}'), combo.currentText() is now '{self.dividend_period_combo.currentText()}'"
        )
        self.dividend_periods_spinbox.setValue(
            saved_div_periods_to_show
        )  # Ensure saved value is set after combo triggers default
        self.dividend_bar_fig = (
            Figure(  # This line was duplicated, removing the second instance
                figsize=(7, 3), dpi=CHART_DPI
            )
        )  # Adjust size as needed
        self.dividend_bar_ax = self.dividend_bar_fig.add_subplot(111)
        self.dividend_bar_canvas = FigureCanvas(self.dividend_bar_fig)
        logging.debug("--- _init_ui_widgets: Dividend bar canvas created ---")
        self.dividend_bar_canvas.setObjectName("DividendBarCanvas")
        dividend_history_layout.addWidget(
            self.dividend_bar_canvas, 1
        )  # Stretch factor 1
        logging.debug("--- _init_ui_widgets: Dividend bar canvas added to layout ---")

        # --- ADDED: Dividend Summary Table ---
        self.dividend_summary_table_view = QTableView()
        self.dividend_summary_table_view.setObjectName("DividendSummaryTable")
        self.dividend_summary_table_model = PandasModel(
            parent=self, log_mode=True
        )  # New model, ADD log_mode=True
        logging.debug(
            "--- _init_ui_widgets: Setting model for Dividend Summary TableView ---"
        )
        self.dividend_summary_table_view.setModel(self.dividend_summary_table_model)
        logging.debug(
            "--- _init_ui_widgets: Configuring Dividend Summary TableView properties ---"
        )
        self.dividend_summary_table_view.setAlternatingRowColors(True)
        self.dividend_summary_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.dividend_summary_table_view.setWordWrap(False)
        self.dividend_summary_table_view.setSortingEnabled(True)  # Allow sorting
        self.dividend_summary_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )  # Stretch columns
        self.dividend_summary_table_view.verticalHeader().setVisible(False)
        # self.dividend_summary_table_view.setFixedHeight(150)  # REMOVE fixed height

        # Dividend Table View
        self.dividend_table_view = QTableView()
        self.dividend_table_view.setObjectName("DividendHistoryTable")
        self.dividend_table_model = PandasModel(
            parent=self, log_mode=True
        )  # ADD log_mode=True
        logging.debug("--- _init_ui_widgets: Dividend table model created ---")
        self.dividend_table_view.setModel(self.dividend_table_model)
        self.dividend_table_view.setAlternatingRowColors(True)
        self.dividend_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.dividend_table_view.setWordWrap(False)
        self.dividend_table_view.setSortingEnabled(True)
        self.dividend_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.dividend_table_view.verticalHeader().setVisible(False)

        # --- Horizontal Layout for Tables ---
        tables_horizontal_layout = QHBoxLayout()

        # Dividend Summary GroupBox
        summary_group = QGroupBox("Dividend Summary (as plotted)")
        summary_group_layout = QVBoxLayout(summary_group)
        summary_group_layout.addWidget(
            self.dividend_summary_table_view, 1
        )  # ADD stretch factor
        self.dividend_summary_table_view.setVisible(True)  # Ensure visible
        tables_horizontal_layout.addWidget(
            summary_group, 1
        )  # Stretch factor 1 for summary
        logging.debug(
            f"--- _init_ui_widgets: Dividend summary table group added. Visible: {self.dividend_summary_table_view.isVisible()}, Geometry: {self.dividend_summary_table_view.geometry()} ---"
        )

        # Dividend Transaction History GroupBox
        transaction_group = QGroupBox("Dividend Transaction History")
        self.dividend_transaction_group = transaction_group  # Store as attribute
        transaction_group_layout = QVBoxLayout(self.dividend_transaction_group)
        transaction_group_layout.addWidget(
            self.dividend_table_view, 1
        )  # Table takes all vertical space in group
        self.dividend_table_view.setVisible(True)  # Ensure visible
        tables_horizontal_layout.addWidget(
            transaction_group, 2
        )  # Stretch factor 2 for transaction history
        logging.debug(
            "--- _init_ui_widgets: Dividend transaction table group added ---"
        )

        # Add the horizontal layout of tables directly
        # to the main vertical layout for the tab.
        dividend_history_layout.addLayout(  # This now adds tables_horizontal_layout directly
            tables_horizontal_layout, 2
        )  # Give tables area a stretch factor of 2

        logging.debug(
            "--- _init_ui_widgets: Dividend transaction table view added to layout ---"
        )
        # """ # Original Content End (Commented out for testing) # Keep this line if you want to easily re-comment
        self._init_rebalancing_tab_widgets()
        # Explicitly set the tab content widget to visible before adding to QTabWidget
        self.dividend_history_tab.setVisible(
            True
        )  # Ensure the page widget itself is visible

        # --- End Dividend History Tab ---

        # --- Tab 2: Transactions Log ---
        # --- Tab: Intraday Chart ---
        self._init_intraday_chart_tab_widgets()
        logging.debug("--- _init_ui_widgets: Entering Transactions Log Tab setup ---")
        self.transactions_log_tab = QWidget()
        transactions_log_main_layout = QVBoxLayout(self.transactions_log_tab)
        transactions_log_main_layout.setContentsMargins(10, 10, 10, 10)
        transactions_log_main_layout.setSpacing(8)

        self._init_transactions_management_widgets(transactions_log_main_layout)

        # Create a splitter to divide the tab
        splitter = QSplitter(Qt.Horizontal)

        # Group for Stock/ETF Transactions
        stock_tx_group = QGroupBox("Stock/ETF Transactions")
        stock_tx_layout = QVBoxLayout(stock_tx_group)
        self.stock_transactions_table_view = QTableView()
        self.stock_transactions_table_view.setObjectName(
            "StockTransactionsLogTable"
        )  # Ensure object name is set
        self.stock_transactions_table_model = PandasModel(
            parent=self, log_mode=True
        )  # MODIFIED: parent=self
        self.stock_transactions_table_view.setModel(self.stock_transactions_table_model)
        self.stock_transactions_table_view.setSortingEnabled(True)
        self.stock_transactions_table_view.setAlternatingRowColors(True)
        self.stock_transactions_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.stock_transactions_table_view.verticalHeader().setVisible(False)
        stock_tx_layout.addWidget(self.stock_transactions_table_view)
        # Context menu for column visibility
        self.stock_transactions_table_view.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.stock_transactions_table_view.horizontalHeader().customContextMenuRequested.connect(
            lambda pos: self._show_column_context_menu(
                self.stock_transactions_table_view, pos, "stock_tx_columns"
            )
        )
        splitter.addWidget(stock_tx_group)

        # Group for $CASH Transactions
        cash_tx_group = QGroupBox(f"{CASH_SYMBOL_CSV} Transactions")
        cash_tx_layout = QVBoxLayout(cash_tx_group)
        self.cash_transactions_table_view = QTableView()  # Ensure object name is set
        self.cash_transactions_table_view.setObjectName(
            "CashTransactionsLogTable"
        )  # Ensure object name is set
        self.cash_transactions_table_model = PandasModel(
            parent=self, log_mode=True
        )  # MODIFIED: parent=self
        self.cash_transactions_table_view.setModel(self.cash_transactions_table_model)
        self.cash_transactions_table_view.setSortingEnabled(True)
        self.cash_transactions_table_view.setAlternatingRowColors(True)
        self.cash_transactions_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.cash_transactions_table_view.verticalHeader().setVisible(False)
        cash_tx_layout.addWidget(self.cash_transactions_table_view)
        # Context menu for column visibility
        self.cash_transactions_table_view.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.cash_transactions_table_view.horizontalHeader().customContextMenuRequested.connect(
            lambda pos: self._show_column_context_menu(
                self.cash_transactions_table_view, pos, "cash_tx_columns"
            )
        )
        splitter.addWidget(cash_tx_group)

        # Set initial sizes for the splitter (optional)
        splitter.setSizes(
            [self.height() // 2, self.height() // 2]
        )  # Roughly equal split

        transactions_log_main_layout.addWidget(splitter)
        self.transactions_log_tab.setVisible(True)

        # Insert the Transactions Log tab at index 1 (making it the second tab)
        self.main_tab_widget.insertTab(1, self.transactions_log_tab, "Transactions")
        logging.debug(
            "--- _init_ui_widgets: Transactions Log Tab added to main_tab_widget ---"
        )
        # --- End Transactions Log Tab ---

        # --- Tab 3: Asset Allocation ---
        logging.debug("--- _init_ui_widgets: Entering Asset Allocation Tab setup ---")
        self.asset_allocation_tab = QWidget()
        asset_allocation_main_layout = QVBoxLayout(self.asset_allocation_tab)
        asset_allocation_main_layout.setContentsMargins(10, 10, 10, 10)
        asset_allocation_main_layout.setSpacing(8)

        # --- Row 1 for Pie Charts ---
        row1_charts_container = QWidget()
        row1_charts_layout = QHBoxLayout(
            row1_charts_container
        )  # Arrange pie charts horizontally

        # Asset Type Pie Chart
        asset_type_group = QGroupBox("Allocation by Asset Type")
        asset_type_layout = QVBoxLayout(asset_type_group)
        self.asset_type_pie_fig = Figure(figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.asset_type_pie_ax = self.asset_type_pie_fig.add_subplot(111)
        self.asset_type_pie_canvas = FigureCanvas(self.asset_type_pie_fig)
        self.asset_type_pie_canvas.setObjectName("AssetTypePieCanvas")
        asset_type_layout.addWidget(self.asset_type_pie_canvas)
        row1_charts_layout.addWidget(asset_type_group, 1)  # Equal stretch factor

        # Sector Allocation Chart
        sector_allocation_group = QGroupBox("Allocation by Sector")  # Renamed
        sector_allocation_layout = QVBoxLayout(sector_allocation_group)
        self.sector_pie_fig = Figure(
            figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI
        )  # New Figure
        self.sector_pie_ax = self.sector_pie_fig.add_subplot(111)  # New Axes
        self.sector_pie_canvas = FigureCanvas(self.sector_pie_fig)  # New Canvas
        self.sector_pie_canvas.setObjectName("SectorPieCanvas")
        sector_allocation_layout.addWidget(
            self.sector_pie_canvas
        )  # Add canvas instead of placeholder
        row1_charts_layout.addWidget(sector_allocation_group, 1)  # Equal stretch factor
        asset_allocation_main_layout.addWidget(row1_charts_container)

        # --- Row 2 for Pie Charts (Geography) ---
        row2_charts_container = QWidget()
        row2_charts_layout = QHBoxLayout(row2_charts_container)

        # Placeholder for Geographical Allocation
        geo_allocation_group = QGroupBox("Allocation by Geography")
        geo_allocation_layout = QVBoxLayout(geo_allocation_group)
        self.geo_pie_fig = Figure(
            figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI
        )  # New Figure
        self.geo_pie_ax = self.geo_pie_fig.add_subplot(111)  # New Axes
        self.geo_pie_canvas = FigureCanvas(self.geo_pie_fig)  # New Canvas
        self.geo_pie_canvas.setObjectName("GeoPieCanvas")
        geo_allocation_layout.addWidget(self.geo_pie_canvas)  # Add canvas

        row2_charts_layout.addWidget(geo_allocation_group, 1)

        # Industry Allocation Chart (New)
        industry_allocation_group = QGroupBox("Allocation by Industry")
        industry_allocation_layout = QVBoxLayout(industry_allocation_group)
        self.industry_pie_fig = Figure(figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.industry_pie_ax = self.industry_pie_fig.add_subplot(111)
        self.industry_pie_canvas = FigureCanvas(self.industry_pie_fig)
        self.industry_pie_canvas.setObjectName("IndustryPieCanvas")
        industry_allocation_layout.addWidget(self.industry_pie_canvas)
        row2_charts_layout.addWidget(industry_allocation_group, 1)  # Add to row 2
        row2_charts_layout.addWidget(
            geo_allocation_group, 1
        )  # Takes up its row, adjust stretch if more items
        asset_allocation_main_layout.addWidget(row2_charts_container)

        asset_allocation_main_layout.addStretch(1)  # Add stretch at the bottom
        self.asset_allocation_tab.setVisible(True)

        # Insert the Asset Allocation tab at index 2 (making it the third tab)
        self.main_tab_widget.insertTab(2, self.asset_allocation_tab, "Asset Allocation")
        logging.debug(
            "--- _init_ui_widgets: Asset Allocation Tab added to main_tab_widget ---"
        )
        # --- End Asset Allocation Tab ---

        self._create_status_bar()
        logging.debug("--- _init_ui_widgets: After _create_status_bar ---")
        self._add_main_tabs()  # Add tabs in desired order
        logging.debug("--- _init_ui_widgets: END ---")

    def _add_main_tabs(self):
        """Adds all main tabs to the main_tab_widget in a defined order."""
        logging.debug("--- _add_main_tabs: START ---")
        # Clear existing tabs if any (useful for reordering or dynamic tab creation)
        while self.main_tab_widget.count() > 0:
            self.main_tab_widget.removeTab(0)

        # Define the desired order of tabs
        tab_order = [
            (self.performance_summary_tab, "Performance"),
            (self.transactions_log_tab, "Transactions"),
            (self.asset_allocation_tab, "Asset Allocation"),
            (self.periodic_value_change_tab, "Asset Change"),
            (self.capital_gains_tab, "Capital Gains"),
            (self.dividend_history_tab, "Dividend"),
            (self.intraday_chart_tab, "Intraday Chart"),
            (self.rebalancing_tab, "Rebalancing"),
            (self.advanced_analysis_tab, "Advanced Analysis"),
        ]

        for tab_widget, tab_name in tab_order:
            # All tab widgets should be initialized by this point
            self.main_tab_widget.addTab(tab_widget, tab_name)
            logging.debug(f"Added tab: {tab_name}")
        logging.debug("--- _add_main_tabs: END ---")

    def _init_intraday_chart_tab_widgets(self):
        """Initializes widgets for the Intraday Chart tab."""
        self.intraday_chart_tab.setObjectName("IntradayChartTab")
        main_layout = QVBoxLayout(self.intraday_chart_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # --- Controls Row ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Symbol:"))
        self.intraday_symbol_combo = QComboBox()
        self.intraday_symbol_combo.setEditable(True)
        self.intraday_symbol_combo.setMinimumWidth(150)
        controls_layout.addWidget(self.intraday_symbol_combo)

        controls_layout.addWidget(QLabel("Period:"))
        self.intraday_period_combo = QComboBox()
        # Valid periods for intraday data are limited
        self.intraday_period_combo.addItems(["1d", "5d", "1mo", "3mo", "6mo", "YTD"])
        self.intraday_period_combo.setMinimumWidth(75)
        self.intraday_period_combo.setCurrentText("5d")
        controls_layout.addWidget(self.intraday_period_combo)

        controls_layout.addWidget(QLabel("Interval:"))
        self.intraday_interval_combo = QComboBox()
        # Valid intervals depend on the period fetched
        self.intraday_interval_combo.setMinimumWidth(75)
        self.intraday_interval_combo.addItems(["1m", "2m", "5m", "15m", "30m", "1h"])
        self.intraday_interval_combo.setCurrentText("5m")
        controls_layout.addWidget(self.intraday_interval_combo)

        self.intraday_pct_change_check = QCheckBox("Show % Change")
        self.intraday_pct_change_check.setObjectName("IntradayPctChangeCheck")
        self.intraday_pct_change_check.setToolTip(
            "Show price change as a percentage from the start of the period"
        )
        controls_layout.addWidget(self.intraday_pct_change_check)

        self.intraday_update_button = QPushButton("Update Chart")
        self.intraday_update_button.setIcon(QIcon.fromTheme("view-refresh"))
        controls_layout.addWidget(self.intraday_update_button)
        controls_layout.addStretch()  # Keep stretch to align controls to the left

        main_layout.addLayout(controls_layout)

        # --- Chart Area ---
        self.intraday_fig = Figure(figsize=(8, 5), dpi=CHART_DPI)
        self.intraday_ax = self.intraday_fig.add_subplot(111)
        self.intraday_canvas = FigureCanvas(self.intraday_fig)
        self.intraday_canvas.setObjectName("IntradayChartCanvas")
        main_layout.addWidget(self.intraday_canvas, 1)

    @Slot(str)
    def _update_intraday_interval_options(self, period: str):
        """
        Dynamically updates the available intervals based on the selected period
        to prevent invalid yfinance API requests.
        """
        self.intraday_interval_combo.blockSignals(True)
        current_interval = self.intraday_interval_combo.currentText()
        self.intraday_interval_combo.clear()

        valid_intervals = []
        # yfinance intraday data limitations:
        # 1m data is only available for the last 7 days.
        # Data for intervals < 1h is only available for the last 60 days.
        if period in ["1d", "5d"]:
            valid_intervals = ["1m", "2m", "5m", "15m", "30m", "1h"]
        elif period in ["1mo"]:  # "1mo" is within the 60-day limit for <1h data
            valid_intervals = ["2m", "5m", "15m", "30m", "1h"]
        else:  # "3mo", "6mo", "ytd" are outside the 60-day limit for most fine-grained data
            valid_intervals = ["1h"]

        self.intraday_interval_combo.addItems(valid_intervals)

        # Try to restore the previous selection if it's still valid
        if current_interval in valid_intervals:
            self.intraday_interval_combo.setCurrentText(current_interval)
        elif valid_intervals:
            self.intraday_interval_combo.setCurrentIndex(
                0
            )  # Default to the first valid option
        self.intraday_interval_combo.blockSignals(False)

    @Slot()
    def _update_intraday_chart(self):
        """Fetches and plots intraday data for the selected symbol."""
        internal_symbol = self.intraday_symbol_combo.currentText()
        period = self.intraday_period_combo.currentText()
        interval = self.intraday_interval_combo.currentText()
        show_pct_change = self.intraday_pct_change_check.isChecked()
        market_hours_only = True  # Default to market hours only

        if not internal_symbol:
            self.show_warning("Please select a symbol.", popup=True)
            return

        # --- MODIFIED: Detect if input is a stock symbol or an FX pair ---
        yf_symbol = ""
        is_fx_pair = False
        # Case 1: User enters a yfinance-style FX pair like "EURUSD=X"
        if internal_symbol.upper().endswith("=X"):
            # Validate that the base is 6 characters before the =X
            if len(internal_symbol) == 8:
                yf_symbol = internal_symbol.upper()
                is_fx_pair = True
            else:
                self.show_warning(
                    f"Invalid FX pair format: '{internal_symbol}'. Expected 6 currency characters, e.g., 'EURUSD=X'.",
                    popup=True,
                )
                self.set_status("Error: Invalid FX pair format.")
                return
        # Case 2: User enters a common format like "USD/EUR" or "USD.EUR"
        elif "/" in internal_symbol or "." in internal_symbol:
            parts = re.split(r"[/.]", internal_symbol)
            if len(parts) == 2 and len(parts[0]) == 3 and len(parts[1]) == 3:
                yf_symbol = f"{parts[0].upper()}{parts[1].upper()}=X"
                is_fx_pair = True
            else:  # It looks like an FX pair but is malformed (e.g., USD.EURt)
                self.show_warning(
                    f"Invalid FX pair format: '{internal_symbol}'. Expected format like 'USD/EUR' or 'USD.EUR'.",
                    popup=True,
                )
                self.set_status("Error: Invalid FX pair format.")
                return

        # Case 3: If not detected as an FX pair, treat as a stock symbol
        if not yf_symbol:
            yf_symbol = self.internal_to_yf_map.get(internal_symbol, internal_symbol)
        # --- END MODIFIED ---

        self.set_status(f"Fetching {interval} intraday data for {internal_symbol}...")
        QApplication.processEvents()

        intraday_df = self.market_data_provider.get_intraday_data(
            yf_symbol,
            period=period,
            interval=interval,  # yf_symbol is now either stock or FX pair
        )

        ax = self.intraday_ax
        ax.clear()
        # --- ADDED: Clear secondary axes if they exist from previous plots ---
        # --- ADDED: Clear previous mplcursors if they exist ---
        if hasattr(self, "intraday_cursor") and self.intraday_cursor:
            try:
                self.intraday_cursor.disconnect_all()
            except Exception:
                pass
            self.intraday_cursor = None
        # --- END ADDED ---
        for other_ax in self.intraday_fig.get_axes():
            if other_ax is not ax:
                try:
                    other_ax.remove()
                except Exception:
                    pass  # Ignore if it fails

        if intraday_df is None or intraday_df.empty:
            # --- MODIFIED: Check for delisted/invalid symbol error ---
            # Check the log buffer for a specific yfinance error for this symbol
            log_contents = self.log_stream.getvalue()
            delisted_pattern = re.compile(
                rf"'{re.escape(yf_symbol)}'.*YFPricesMissingError.*no price data found",
                re.IGNORECASE,
            )
            if delisted_pattern.search(log_contents):
                error_message = f"No data found for '{internal_symbol}'.\nThe symbol may be invalid or delisted."
            else:
                error_message = f"No intraday data available for\n'{internal_symbol}' with selected period/interval."

            ax.text(
                0.5,
                0.5,
                error_message,
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            self.set_status(f"Failed to fetch intraday data for {internal_symbol}.")
        else:
            # Filter for market hours if checked (only for stocks, not FX)
            if market_hours_only:
                try:
                    # --- MODIFIED: Dynamic Market Hours ---
                    exchange = self.market_data_provider.get_exchange_for_symbol(
                        yf_symbol
                    )

                    # Only apply market hours filter if the exchange is in our map (i.e., it's a stock exchange)
                    if exchange in EXCHANGE_TRADING_HOURS:
                        open_time_str, close_time_str = EXCHANGE_TRADING_HOURS[exchange]

                        logging.info(
                            f"Applying market hours for exchange '{exchange}': {open_time_str} - {close_time_str}"
                        )

                        # The index from yfinance is timezone-aware. between_time works on the local time of the index.
                        asset_timezone = intraday_df.index.tz
                        if asset_timezone:
                            # Create naive time objects. Pandas' between_time correctly handles
                            # filtering on a timezone-aware index using naive wall times.
                            market_open = time.fromisoformat(open_time_str)
                            market_close = time.fromisoformat(close_time_str)

                            intraday_df = intraday_df.between_time(
                                market_open, market_close
                            )
                            logging.info(
                                f"Filtered intraday data for {internal_symbol} to market hours in timezone: {asset_timezone}"
                            )
                            if intraday_df.empty:
                                logging.warning(
                                    f"Intraday data for {internal_symbol} became empty after applying market hours filter."
                                )
                        else:
                            logging.warning(
                                f"Could not determine timezone for {internal_symbol} from intraday data. "
                                "Market hours filter may be inaccurate."
                            )
                    else:
                        logging.info(
                            f"'{exchange}' not in EXCHANGE_TRADING_HOURS map. Skipping market hours filter for '{internal_symbol}'."
                        )
                    # --- END MODIFIED ---
                except Exception as e_filter:
                    logging.warning(
                        f"Could not filter intraday data by time: {e_filter}"
                    )

            if intraday_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No data available for the selected market hours.",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.set_status(f"No market hours data for {internal_symbol}.")
                self.intraday_canvas.draw()
                return

            try:
                # Using mplfinance is a good option here, but for simplicity, let's do a simple line plot.
                # For a proper candlestick, you'd use a library like mplfinance.
                # --- MODIFIED: Plot against a numerical index if market_hours_only ---
                x_axis_data = (
                    np.arange(len(intraday_df.index))
                    if market_hours_only
                    else intraday_df.index
                )

                # --- MODIFIED: Plot either absolute price or percentage change ---
                if show_pct_change:
                    first_price = intraday_df["Close"].iloc[0]
                    if first_price > 0:
                        pct_change_data = (
                            (intraday_df["Close"] / first_price) - 1
                        ) * 100.0
                        (close_line,) = ax.plot(
                            x_axis_data,
                            pct_change_data,
                            label=f"{internal_symbol} % Change",
                        )
                        # Store data for tooltip formatter
                        close_line.set_gid(
                            {
                                "df": intraday_df,
                                "is_pct_change": True,
                                "is_fx": is_fx_pair,
                                "x_axis_type": (
                                    "numeric" if market_hours_only else "datetime"
                                ),
                                "symbol": internal_symbol,
                            }
                        )
                        ax.yaxis.set_major_formatter(mtick.PercentFormatter())
                    else:
                        # Cannot calculate % change if starting price is zero
                        (close_line,) = ax.plot(
                            x_axis_data,
                            intraday_df["Close"],
                            label=f"{internal_symbol} Close",
                        )
                else:
                    # Plot absolute Close price
                    (close_line,) = ax.plot(
                        x_axis_data,
                        intraday_df["Close"],
                        label=f"{internal_symbol} Close",
                    )
                    # Store data for tooltip formatter
                    close_line.set_gid(
                        {
                            "df": intraday_df,
                            "is_pct_change": False,
                            "is_fx": is_fx_pair,
                            "x_axis_type": (
                                "numeric" if market_hours_only else "datetime"
                            ),
                            "symbol": internal_symbol,
                        }
                    )
                # --- END MODIFIED ---

                # Plot Volume on a secondary y-axis
                if "Volume" in intraday_df.columns:
                    ax2 = ax.twinx()
                    ax2.bar(
                        x_axis_data,
                        intraday_df["Volume"],
                        color="grey",
                        alpha=0.3,
                        width=1.0,
                        label="Volume",
                    )
                    ax2.set_ylabel("Volume", fontsize=8)
                    ax2.tick_params(axis="y", labelsize=7)
                    ax2.spines["right"].set_position(("outward", 0))
                    # --- MODIFIED: Hide volume axis for FX pairs ---
                    ax2.spines["right"].set_visible(not is_fx_pair)
                    ax2.get_yaxis().set_visible(not is_fx_pair)
                    # --- END MODIFIED ---
                    ax2.yaxis.set_major_formatter(
                        mtick.FuncFormatter(
                            lambda x, p: (
                                ""
                                if is_fx_pair  # Don't show labels for FX volume
                                else f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K"
                            )
                        )
                    )

                ax.set_title(
                    f"Intraday Chart: {internal_symbol} ({period}, {interval})"
                )
                # --- MODIFIED: Change Y-axis label based on view type ---
                if show_pct_change:
                    ax.set_ylabel("Change (%)")
                else:
                    ax.set_ylabel(
                        "Rate"
                        if is_fx_pair
                        else f"Price ({self._get_currency_for_symbol(internal_symbol)})"
                    )
                # --- END MODIFIED ---

                # --- MODIFIED: Handle legend for multiple axes ---
                lines, labels = ax.get_legend_handles_labels()
                if "ax2" in locals():  # Check if secondary axis was created
                    # --- ADDED: Store data for volume bar tooltips ---
                    gid_data = {
                        "df": intraday_df,
                        "is_pct_change": False,  # Volume is not pct change
                        "is_fx": is_fx_pair,
                        "x_axis_type": "numeric" if market_hours_only else "datetime",
                        "symbol": internal_symbol,
                        "is_volume": True,  # Add a flag to identify volume bars
                    }
                    for (
                        container
                    ) in (
                        ax2.containers
                    ):  # ax2.containers is a list of BarContainer objects
                        for (
                            bar_patch
                        ) in (
                            container.patches
                        ):  # Iterate through the actual bars (patches)
                            bar_patch.set_gid(gid_data)
                    # --- END ADDED ---
                    lines2, labels2 = ax2.get_legend_handles_labels()
                    ax.legend(lines + lines2, labels + labels2, loc="upper left")
                else:
                    ax.legend(loc="upper left")

                ax.grid(True, linestyle="--", alpha=0.6)

                # --- MODIFIED: Custom tick formatting for market_hours_only ---
                if market_hours_only:
                    # Set custom tick labels to show timestamps without gaps
                    tick_indices = np.linspace(
                        0, len(intraday_df) - 1, num=6, dtype=int
                    )
                    # --- MODIFIED: Force display timezone to US Eastern Time ---
                    display_timezone = ZoneInfo("America/New_York")
                    tz_name = display_timezone.tzname(datetime.now()) or "EST/EDT"
                    ax.set_xlabel(f"Time ({tz_name})", fontsize=8)

                    # --- MODIFIED: Convert timestamp to the asset's timezone before formatting ---
                    tick_labels = [
                        intraday_df.index[i]
                        .tz_convert(display_timezone)
                        .strftime("%H:%M\n%b-%d")
                        for i in tick_indices
                    ]
                    ax.set_xticks(tick_indices)
                    ax.set_xticklabels(tick_labels)
                else:
                    ax.set_xlabel("Time", fontsize=8)
                    # --- ADDED: mplcursors for intraday chart ---
                    if MPLCURSORS_AVAILABLE:
                        try:
                            # --- MODIFIED: Create a single cursor targeting all axes in the figure ---
                            # This is the most robust way to handle multiple axes (primary for price, secondary for volume).
                            # mplcursors will automatically find all artists (lines, bars) on all axes.
                            self.intraday_cursor = mplcursors.cursor(
                                self.intraday_fig.get_axes(), hover=True
                            )
                            # --- END MODIFIED ---

                            # The 'add' signal will be emitted when the cursor is shown on any artist.
                            # Our formatter will then use the artist's GID to get the correct data.
                            self.intraday_cursor.connect(
                                "add", self._format_intraday_tooltip_annotation
                            )
                            logging.debug("mplcursors activated for Intraday chart.")
                        except Exception as e_cursor_intra:
                            logging.error(
                                f"Error activating mplcursors for intraday chart: {e_cursor_intra}"
                            )
                            self.intraday_cursor = None
                    # --- END ADDED ---

                    self.intraday_fig.autofmt_xdate()

                self.set_status(f"Intraday chart for {internal_symbol} updated.")
            except Exception as e:
                logging.error(
                    f"Error plotting intraday data for {internal_symbol}: {e}",
                    exc_info=True,
                )
                ax.text(
                    0.5,
                    0.5,
                    "Error plotting data.",
                    ha="center",
                    va="center",
                    color=COLOR_LOSS,
                )
                self.set_status(f"Error plotting intraday data for {internal_symbol}.")

        self.intraday_canvas.draw()

    def _populate_intraday_symbol_combo(self):
        """Populates the symbol combobox in the Intraday Chart tab."""
        if hasattr(self, "holdings_data") and not self.holdings_data.empty:
            # Get non-cash symbols from the current holdings
            symbols = sorted(
                [
                    s
                    for s in self.holdings_data["Symbol"].unique()
                    if not is_cash_symbol(s)
                ]
            )
            self.intraday_symbol_combo.clear()
            self.intraday_symbol_combo.addItems(symbols)

    def _init_transactions_management_widgets(self, parent_layout: QVBoxLayout):
        """Initializes widgets for managing transactions within the Transactions Log tab."""
        # --- Filter Widgets ---
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Symbol:"))
        self.filter_symbol_edit = QLineEdit()
        self.filter_symbol_edit.setPlaceholderText("Enter symbol to filter...")
        filter_layout.addWidget(self.filter_symbol_edit)

        filter_layout.addWidget(QLabel("Account:"))
        self.filter_account_edit = QLineEdit()
        self.filter_account_edit.setPlaceholderText("Enter account to filter...")
        filter_layout.addWidget(self.filter_account_edit)

        self.apply_filter_button = QPushButton("Apply Filter")
        self.apply_filter_button.setIcon(QIcon.fromTheme("edit-find"))
        filter_layout.addWidget(self.apply_filter_button)

        self.clear_filter_button = QPushButton("Clear Filters")
        self.clear_filter_button.setIcon(QIcon.fromTheme("edit-clear"))
        filter_layout.addWidget(self.clear_filter_button)
        filter_layout.addStretch(1)
        parent_layout.addLayout(filter_layout)

        # --- Table View ---
        self.transactions_management_table_view = QTableView()
        self.transactions_management_table_view.setObjectName(
            "ManageTransactionsDBTable"
        )
        self.transactions_management_table_model = PandasModel(
            parent=self, log_mode=True
        )
        self.transactions_management_table_view.setModel(
            self.transactions_management_table_model
        )

        self.transactions_management_table_view.setSelectionBehavior(
            QTableView.SelectRows
        )
        self.transactions_management_table_view.setSelectionMode(
            QTableView.SingleSelection
        )
        self.transactions_management_table_view.setAlternatingRowColors(True)
        self.transactions_management_table_view.setSortingEnabled(True)
        self.transactions_management_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.transactions_management_table_view.verticalHeader().setVisible(False)

        parent_layout.addWidget(self.transactions_management_table_view)
        self.transactions_management_table_view.resizeColumnsToContents()

        # Context menu for column visibility
        self.transactions_management_table_view.horizontalHeader().setContextMenuPolicy(
            Qt.CustomContextMenu
        )
        self.transactions_management_table_view.horizontalHeader().customContextMenuRequested.connect(
            lambda pos: self._show_column_context_menu(
                self.transactions_management_table_view,
                pos,
                "transactions_management_columns",
            )
        )

        # --- Action Buttons ---
        button_layout = QHBoxLayout()
        self.manage_tab_add_button = QPushButton("Add Transaction")
        self.manage_tab_add_button.setIcon(QIcon.fromTheme("list-add"))
        self.manage_tab_edit_button = QPushButton("Edit Selected")
        self.manage_tab_edit_button.setIcon(QIcon.fromTheme("document-edit"))
        self.manage_tab_delete_button = QPushButton("Delete Selected")
        self.manage_tab_delete_button.setIcon(QIcon.fromTheme("edit-delete"))
        self.manage_tab_export_button = QPushButton("Export to CSV")
        self.manage_tab_export_button.setIcon(QIcon.fromTheme("document-save"))

        button_layout.addWidget(self.manage_tab_add_button)
        button_layout.addWidget(self.manage_tab_edit_button)
        button_layout.addWidget(self.manage_tab_delete_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.manage_tab_export_button)
        parent_layout.addLayout(button_layout)

        # Initial data load for the table
        self._refresh_transactions_view()

    def _refresh_transactions_view(self):
        """Reloads data from DB and updates the transactions table view."""
        logging.debug("PortfolioApp: Refreshing transactions table from database...")
        acc_map_config_refresh = self.config.get("account_currency_map", {})
        def_curr_config_refresh = self.config.get(
            "default_currency", config.DEFAULT_CURRENCY
        )
        df_updated, success = load_all_transactions_from_db(
            self.db_conn,
            account_currency_map=acc_map_config_refresh,
            default_currency=def_curr_config_refresh,
        )
        if success and df_updated is not None:
            self._current_data_df = df_updated.copy()
            self._apply_filter_to_transactions_view()  # Re-apply any active filters to the new data
            self._apply_column_visibility(
                self.transactions_management_table_view,
                "transactions_management_columns",
            )
            logging.debug(
                f"PortfolioApp: Transactions table refreshed with {len(self._current_data_df)} rows."
            )
        else:
            QMessageBox.warning(
                self,
                "Refresh Error",
                "Could not refresh transaction list from database.",
            )
            self._current_data_df = pd.DataFrame()  # Clear data on error
            self.transactions_management_table_model.updateData(
                self._current_data_df
            )  # Update view with empty

    def _apply_filter_to_transactions_view(self):
        """Filters the transactions table view based on symbol/account input using DB column names."""
        symbol_filter_text = self.filter_symbol_edit.text().strip().upper()
        account_filter_text = (
            self.filter_account_edit.text().strip()
        )  # Keep case for account name
        df_to_display = self._current_data_df.copy()  # Start with full current data

        # Filter by Symbol (column name in DB is "Symbol")
        if symbol_filter_text and "Symbol" in df_to_display.columns:
            try:
                df_to_display = df_to_display[
                    df_to_display["Symbol"]
                    .astype(str)
                    .str.contains(symbol_filter_text, case=False, na=False)
                ]
            except Exception as e:
                logging.warning(f"Error applying symbol filter: {e}")

        # Filter by Account (column name in DB is "Account")
        if account_filter_text and "Account" in df_to_display.columns:
            try:
                # Check if account matches 'Account' OR 'To Account' column
                mask_account = df_to_display["Account"].astype(str).str.contains(account_filter_text, case=False, na=False)
                if "To Account" in df_to_display.columns:
                     mask_to_account = df_to_display["To Account"].astype(str).str.contains(account_filter_text, case=False, na=False)
                     mask_account = mask_account | mask_to_account
                
                df_to_display = df_to_display[mask_account]
            except Exception as e:
                logging.warning(f"Error applying account filter: {e}")

        self.transactions_management_table_model.updateData(df_to_display)
        self._apply_column_visibility(
            self.transactions_management_table_view, "transactions_management_columns"
        )
        self.transactions_management_table_view.resizeColumnsToContents()
        self.transactions_management_table_view.viewport().update()  # Force viewport repaint

        # --- ADDED: Re-apply sort after data update ---
        if "Date" in df_to_display.columns:
            try:
                date_col_idx = df_to_display.columns.get_loc("Date")
                self.transactions_management_table_view.sortByColumn(
                    date_col_idx, Qt.DescendingOrder
                )
                logging.debug("PortfolioApp: Re-applied sort by Date descending.")
            except KeyError:
                logging.warning("PortfolioApp: 'Date' column not found for re-sorting.")
        # --- END ADDED ---

    def get_selected_db_id(self) -> Optional[int]:
        """
        Gets the 'original_index' (internal DB ID) of the currently selected row
        from the transactions management table model.
        """
        selected_indexes = (
            self.transactions_management_table_view.selectionModel().selectedRows()
        )
        if not selected_indexes or len(selected_indexes) != 1:
            return None  # No single row selected

        selected_view_index = selected_indexes[
            0
        ]  # QModelIndex of the first cell in the selected row
        source_model = (
            self.transactions_management_table_model
        )  # This is the PandasModel instance

        try:
            # 'original_index' column in the model's DataFrame (_data) holds the internal DB ID
            # This column was created by load_all_transactions_from_db aliasing `id as original_index` (internal ID)
            if "original_index" not in source_model._data.columns:
                logging.error(
                    "'original_index' (internal ID) column not found in transactions management model data."
                )
                return None

            original_index_col_idx = source_model._data.columns.get_loc(
                "original_index"
            )

            # Get the model index for the specific cell in the 'original_index' column
            target_model_index = source_model.index(
                selected_view_index.row(), original_index_col_idx
            )

            db_id_val = source_model.data(
                target_model_index, Qt.DisplayRole
            )  # Get display data

            return (
                int(db_id_val)
                if db_id_val is not None and str(db_id_val).strip() != "-"
                else None
            )
        except (AttributeError, KeyError, IndexError, ValueError, TypeError) as e:
            logging.error(
                f"Error getting DB ID from transactions management selection: {e}",
                exc_info=True,
            )
            return None

    @Slot()
    def add_new_transaction_db(self):
        """Opens the AddTransactionDialog to add a new transaction to the database."""
        # Get available accounts and symbols from the current data for dialog autocompletion
        # Combine accounts from both 'Account' and 'To Account' columns
        unique_accounts = set()
        if "Account" in self._current_data_df.columns:
            unique_accounts.update(self._current_data_df["Account"].dropna().unique())
        if "To Account" in self._current_data_df.columns:
            unique_accounts.update(self._current_data_df["To Account"].dropna().unique())
        
        accounts_for_dialog = sorted(list(unique_accounts))
        symbols_for_dialog = (
            sorted(list(self._current_data_df["Symbol"].unique()))
            if "Symbol" in self._current_data_df.columns
            else []
        )

        add_dialog = AddTransactionDialog(
            existing_accounts=accounts_for_dialog,
            portfolio_symbols=symbols_for_dialog,
            parent=self,  # The dialog's parent is this PortfolioApp
        )

        if add_dialog.exec():  # True if Save was clicked and dialog validation passed
            new_data_dict_pytypes = (
                add_dialog.get_transaction_data()
            )  # Returns dict with Python types, CSV-like keys
            if new_data_dict_pytypes:
                # The parent app's save_new_transaction method handles the DB interaction
                # and the main app's refresh.
                if hasattr(self, "save_new_transaction"):
                    self.save_new_transaction(new_data_dict_pytypes)
                    # After the parent app saves, we can refresh our own view.
                    self._refresh_transactions_view()
                    # self.data_changed.emit()  # Signal to the main window that data has changed.
                else:
                    QMessageBox.critical(
                        self,
                        "Internal Error",
                        "The main application does not have a 'save_new_transaction' method.",
                    )
            # If validation fails, get_transaction_data shows its own message box, so no extra message is needed.

    @Slot()
    def edit_selected_transaction_db(self):
        db_id = self.get_selected_db_id()
        if db_id is None:
            QMessageBox.warning(
                self, "Selection Error", "Please select a single transaction to edit."
            )
            return

        # Fetch the row with DB column names from self._current_data_df
        try:
            transaction_row_series = self._current_data_df[
                self._current_data_df["original_index"] == db_id
            ].iloc[0]
        except IndexError:
            QMessageBox.warning(
                self,
                "Data Error",
                f"Could not find transaction with ID {db_id} for editing in the current view.",
            )
            return

        # Map DB data to the format AddTransactionDialog expects for pre-filling
        # (keys are CSV-like headers, values are strings or appropriate types for QDateEdit)
        dialog_prefill_data: Dict[str, Any] = {
            "Date (MMM DD, YYYY)": (
                pd.to_datetime(transaction_row_series.get("Date")).strftime(
                    CSV_DATE_FORMAT
                )
                if pd.notna(transaction_row_series.get("Date"))
                else ""
            ),
            "Transaction Type": transaction_row_series.get("Type", ""),
            "Stock / ETF Symbol": transaction_row_series.get("Symbol", ""),
            # For numeric, pass as string, AddTransactionDialog's _populate_fields_for_edit will format
            "Quantity of Units": str(
                transaction_row_series.get("Quantity", "")
                if pd.notna(transaction_row_series.get("Quantity"))
                else ""
            ),
            "Amount per unit": str(
                transaction_row_series.get("Price/Share", "")
                if pd.notna(transaction_row_series.get("Price/Share"))
                else ""
            ),
            "Total Amount": str(
                transaction_row_series.get("Total Amount", "")
                if pd.notna(transaction_row_series.get("Total Amount"))
                else ""
            ),
            "Fees": str(
                transaction_row_series.get("Commission", "")
                if pd.notna(transaction_row_series.get("Commission"))
                else ""
            ),
            "Investment Account": transaction_row_series.get("Account", ""),
            "Split Ratio (new shares per old share)": str(
                transaction_row_series.get("Split Ratio", "")
                if pd.notna(transaction_row_series.get("Split Ratio"))
                else ""
            ),
            "Note": transaction_row_series.get("Note", ""),
            "To Account": transaction_row_series.get("To Account", ""),  # <-- ADDED
            # Local Currency is not directly part of AddTransactionDialog fields, it's derived by PortfolioApp
        }

        # Combine accounts from both 'Account' and 'To Account' columns
        unique_accounts = set()
        if "Account" in self._current_data_df.columns:
            unique_accounts.update(self._current_data_df["Account"].dropna().unique())
        if "To Account" in self._current_data_df.columns:
            unique_accounts.update(self._current_data_df["To Account"].dropna().unique())

        accounts_for_dialog = sorted(list(unique_accounts))

        edit_dialog = AddTransactionDialog(
            existing_accounts=accounts_for_dialog,
            parent=self,
            edit_data=dialog_prefill_data,
        )

        if edit_dialog.exec():  # True if Save was clicked and dialog validation passed
            new_data_dict_pytypes = (
                edit_dialog.get_transaction_data()
            )  # Returns dict with Python types, CSV-like keys
            if new_data_dict_pytypes:
                # Call PortfolioApp's method to handle the DB update
                if self._edit_transaction_in_db(db_id, new_data_dict_pytypes):
                    self._mark_data_as_stale()  # Mark data as stale
                    self._refresh_transactions_view()  # Refresh the management table view
                else:
                    # Error message is shown by _edit_transaction_in_db
                    QMessageBox.critical(
                        self,
                        "Update Error",
                        "Failed to update transaction in the database. Check logs.",
                    )
            else:  # Dialog validation failed (get_transaction_data returned None)
                QMessageBox.warning(
                    self,
                    "Validation Error",
                    "Transaction data was not valid. No changes were saved.",
                )

    @Slot()
    def delete_selected_transaction_db(self):
        db_id = self.get_selected_db_id()
        if db_id is None:
            QMessageBox.warning(
                self, "Selection Error", "Please select a single transaction to delete."
            )
            return

        # Get some details of the transaction for the confirmation message
        tx_details_str = "this transaction"
        try:
            row_to_delete = self._current_data_df[
                self._current_data_df["original_index"] == db_id
            ].iloc[0]
            tx_date_str = (
                pd.to_datetime(row_to_delete.get("Date")).strftime("%Y-%m-%d")
                if pd.notna(row_to_delete.get("Date"))
                else "N/A"
            )
            tx_symbol = row_to_delete.get("Symbol", "N/A")
            tx_type = row_to_delete.get("Type", "N/A")
            tx_details_str = (
                f"Type: {tx_type}, Symbol: {tx_symbol}, Date: {tx_date_str}"
            )
        except Exception:
            pass  # Stick with generic message if details can't be fetched

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to permanently delete this transaction from the database?\n\n{tx_details_str}",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            # Call PortfolioApp's method to handle the DB delete
            if self._delete_transaction_in_db(db_id):
                self._mark_data_as_stale()  # Mark data as stale
                self._refresh_transactions_view()  # Refresh the management table view
            else:
                # Error message is shown by _delete_transaction_in_db
                QMessageBox.critical(
                    self,
                    "Delete Error",
                    "Failed to delete transaction from the database. Check logs.",
                )

    def _export_transactions_to_csv(self):
        """Exports the currently displayed transactions to a CSV file."""
        if self.transactions_management_table_model.rowCount() == 0:
            QMessageBox.information(self, "Export", "No data to export.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Transactions to CSV", "", "CSV Files (*.csv);;All Files (*)"
        )
        if file_path:
            try:
                df_to_export = self.transactions_management_table_model.dataFrame()
                # Exclude 'original_index' column if it exists, as it's an internal DB ID
                if "original_index" in df_to_export.columns:
                    df_to_export = df_to_export.drop(columns=["original_index"])
                df_to_export.to_csv(file_path, index=False)
                QMessageBox.information(
                    self, "Export Successful", f"Transactions exported to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Export Error", f"Failed to export transactions: {e}"
                )

    def _clear_filter_in_transactions_view(self):
        """Clears filters and re-applies to show all current data."""
        self.filter_symbol_edit.clear()
        self.filter_account_edit.clear()
        self._apply_filter_to_transactions_view()

    def _update_periodic_value_change_display(self):
        """Updates the graphs and tables in the Asset Change tab."""
        # This method will be called from handle_results and when spinboxes change.
        # It will iterate through Annual, Monthly, Weekly, get data from
        # self.periodic_returns_data, filter by spinbox value, and update
        # the corresponding graph and table.
        # For now, it's a placeholder. The detailed implementation will be complex.
        logging.info("Placeholder: _update_periodic_value_change_display called.")
        # Actual plotting and table updates will go here.
        # Example for one period (Annual):
        # self._plot_pvc_graph(self.pvc_annual_graph_ax, self.pvc_annual_graph_canvas, 'Y', self.pvc_annual_graph_spinbox.value())
        # self._update_pvc_table(self.pvc_annual_table_model, 'Y', self.pvc_annual_graph_spinbox.value())

    def _init_capital_gains_tab_widgets(self):
        """Initializes widgets for the Capital Gains tab."""
        if not hasattr(self, "capital_gains_tab"):
            return

        cg_main_layout = QVBoxLayout(self.capital_gains_tab)
        cg_main_layout.setContentsMargins(10, 10, 10, 10)
        cg_main_layout.setSpacing(8)

        # --- Controls Row ---
        cg_controls_layout = QHBoxLayout()
        cg_controls_layout.addWidget(QLabel("Aggregate by:"))
        self.cg_period_combo = QComboBox()
        self.cg_period_combo.setObjectName("CgPeriodCombo")
        self.cg_period_combo.addItems(["Annual", "Quarterly"])
        self.cg_period_combo.setCurrentText(self.config.get("cg_agg_period", "Annual"))
        self.cg_period_combo.setMinimumWidth(100)
        cg_controls_layout.addWidget(self.cg_period_combo)

        cg_controls_layout.addWidget(QLabel("  Periods to Show:"))
        self.cg_periods_spinbox = QSpinBox()
        self.cg_periods_spinbox.setObjectName("CgPeriodsSpinbox")
        self.cg_periods_spinbox.setMinimum(1)

        def update_cg_max_periods(period_text):
            if period_text == "Annual":
                self.cg_periods_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_ANNUAL)
            elif period_text == "Quarterly":
                self.cg_periods_spinbox.setMaximum(BAR_CHART_MAX_PERIODS_QUARTERLY)

        self.cg_period_combo.currentTextChanged.connect(update_cg_max_periods)
        update_cg_max_periods(self.cg_period_combo.currentText())

        self.cg_periods_spinbox.setValue(self.config.get("cg_periods_to_show", 10))
        self.cg_periods_spinbox.setFixedWidth(60)
        cg_controls_layout.addWidget(self.cg_periods_spinbox)
        cg_controls_layout.addStretch()
        cg_main_layout.addLayout(cg_controls_layout)

        # --- Capital Gains Summary Cards ---
        self.cg_summary_frame = QFrame()
        self.cg_summary_frame.setObjectName("CgSummaryFrame")
        cg_summary_layout = QHBoxLayout(self.cg_summary_frame)
        cg_summary_layout.setContentsMargins(10, 10, 10, 10)
        cg_summary_layout.setSpacing(40) # Increased spacing for better separation

        # Helper to create a card (just labels now)
        def create_summary_card(title, object_name):
            # Use a layout for the pair
            layout = QVBoxLayout()
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(5)
            
            title_label = QLabel(title)
            # Use SummaryLabel style from QSS (inherits font size etc)
            title_label.setObjectName("SummaryLabel") 
            title_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            
            value_label = QLabel("Loading...")
            value_label.setObjectName(object_name)
            # Use SummaryValue style from QSS
            # We can set the object name to SummaryValue for default style, 
            # but we need specific IDs for updating. 
            # So we will style these specific IDs in code or rely on generic QLabel styling + manual font updates if needed.
            # Actually, let's reuse the create_summary_item logic's styling approach manually here
            # or just set a class property if Qt supported it.
            # For now, let's stick to manual styling consistent with create_summary_item
            
            # Match create_summary_item styling
            font = QFont(self.app_font if hasattr(self, "app_font") else QFont())
            font.setPointSize(10) # Base size
            title_label.setFont(font)
            title_label.setStyleSheet("color: palette(text); font-weight: normal;") # Ensure color matches theme

            val_font = QFont(self.app_font if hasattr(self, "app_font") else QFont())
            val_font.setPointSize(14) # Larger for value
            val_font.setBold(True)
            value_label.setFont(val_font)
            value_label.setStyleSheet("color: palette(text);") # Default color

            layout.addWidget(title_label)
            layout.addWidget(value_label)
            return layout, value_label

        # Create cards
        self.cg_card_gain_layout, self.cg_card_gain_value = create_summary_card("Total Realized Gain", "CgTotalGainValue")
        self.cg_card_proceeds_layout, self.cg_card_proceeds_value = create_summary_card("Total Proceeds", "CgTotalProceedsValue")
        self.cg_card_cost_layout, self.cg_card_cost_value = create_summary_card("Total Cost Basis", "CgTotalCostValue")

        cg_summary_layout.addLayout(self.cg_card_gain_layout)
        cg_summary_layout.addLayout(self.cg_card_proceeds_layout)
        cg_summary_layout.addLayout(self.cg_card_cost_layout)
        cg_summary_layout.addStretch() # Push everything to the left

        cg_main_layout.addWidget(self.cg_summary_frame)

        # --- Chart Area ---
        self.cg_bar_fig = Figure(figsize=(7, 3), dpi=CHART_DPI)  # Adjust size as needed
        self.cg_bar_ax = self.cg_bar_fig.add_subplot(111)
        self.cg_bar_canvas = FigureCanvas(self.cg_bar_fig)
        self.cg_bar_canvas.setObjectName("CgBarCanvas")
        cg_main_layout.addWidget(self.cg_bar_canvas, 1)  # Stretch factor 1

        # --- Horizontal layout for the two tables ---
        tables_horizontal_layout = QHBoxLayout()

        # --- Capital Gains Summary Table (Data from Bar Graph) ---
        summary_table_group = QGroupBox("Summary of Plotted Gains")
        summary_table_group_layout = QVBoxLayout(summary_table_group)
        self.cg_summary_table_view = QTableView()
        self.cg_summary_table_view.setObjectName("CgSummaryTable")
        self.cg_summary_table_model = PandasModel(parent=self, log_mode=True)
        self.cg_summary_table_view.setModel(self.cg_summary_table_model)
        self.cg_summary_table_view.setAlternatingRowColors(True)
        self.cg_summary_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.cg_summary_table_view.setWordWrap(False)
        self.cg_summary_table_view.setSortingEnabled(True)
        self.cg_summary_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch  # Stretch columns
        )
        self.cg_summary_table_view.verticalHeader().setVisible(False)
        self.cg_summary_table_view.setMinimumHeight(
            80
        )  # Ensure it has some visible height
        summary_table_group_layout.addWidget(self.cg_summary_table_view)
        tables_horizontal_layout.addWidget(
            summary_table_group, 1
        )  # Summary table takes 1 part of horizontal space

        # --- Detailed Capital Gains History Table ---
        history_table_group = QGroupBox("Detailed Capital Gains History")
        history_table_group_layout = QVBoxLayout(history_table_group)
        self.cg_table_view = QTableView()
        self.cg_table_view.setObjectName("CgHistoryTable")
        self.cg_table_model = PandasModel(
            parent=self, log_mode=True
        )  # log_mode for direct column display
        self.cg_table_view.setModel(self.cg_table_model)
        self.cg_table_view.setAlternatingRowColors(True)
        self.cg_table_view.setSelectionBehavior(QTableView.SelectRows)
        self.cg_table_view.setWordWrap(False)
        self.cg_table_view.setSortingEnabled(True)
        self.cg_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Interactive
        )
        self.cg_table_view.verticalHeader().setVisible(False)
        history_table_group_layout.addWidget(self.cg_table_view)
        tables_horizontal_layout.addWidget(
            history_table_group, 2
        )  # History table takes 2 parts of horizontal space

        # Add the horizontal layout of tables to the main vertical layout
        cg_main_layout.addLayout(
            tables_horizontal_layout, 2
        )  # Tables area takes 2 parts of vertical space

        logging.debug("--- _init_ui_widgets: Capital Gains Tab widgets initialized ---")

    def _init_advanced_analysis_tab(self):
        """Initializes the Advanced Analysis tab with its sub-tabs and layouts."""
        logging.debug("Initializing Advanced Analysis tab.")
        aa_layout = QVBoxLayout(self.advanced_analysis_tab)
        self.advanced_analysis_tab.setLayout(aa_layout)

        self.advanced_analysis_sub_tab_widget = QTabWidget()
        aa_layout.addWidget(self.advanced_analysis_sub_tab_widget)

        # Correlation Matrix Tab
        self.correlation_matrix_tab = QWidget()
        self.advanced_analysis_sub_tab_widget.addTab(
            self.correlation_matrix_tab, "Correlation Matrix"
        )
        self._init_correlation_matrix_tab()

        # Factor Analysis Tab
        self.factor_analysis_tab = QWidget()
        self.advanced_analysis_sub_tab_widget.addTab(
            self.factor_analysis_tab, "Factor Analysis"
        )
        self._init_factor_analysis_tab()

        # Scenario Analysis Tab
        self.scenario_analysis_tab = QWidget()
        self.advanced_analysis_sub_tab_widget.addTab(
            self.scenario_analysis_tab, "Scenario Analysis"
        )
        self._init_scenario_analysis_tab()

        logging.debug("Advanced Analysis tab initialized.")

    def _update_rebalancing_tab(self):
        self._load_current_holdings_to_target_table()

    def _load_current_holdings_to_target_table(self):
        logging.debug(
            f"Loading current holdings. Holdings data shape: {self.holdings_data.shape}"
        )
        if not self.holdings_data.empty:
            logging.debug(f"Holdings data columns: {self.holdings_data.columns}")

        self.target_allocation_table.setRowCount(0)
        if self.holdings_data.empty:
            return

        display_currency = self.currency_combo.currentText()
        mkt_val_col = f"Market Value ({display_currency})"

        if mkt_val_col not in self.holdings_data.columns:
            logging.error(
                f"Required column '{mkt_val_col}' not found in holdings_data."
            )
            return

        total_portfolio_value = self.summary_metrics_data.get(
            f"Total Portfolio Value ({display_currency})"
        )
        if not total_portfolio_value or total_portfolio_value == 0:
            total_portfolio_value = self.holdings_data[mkt_val_col].sum()

        total_portfolio_value = self.summary_metrics_data.get(
            f"Total Portfolio Value ({display_currency})"
        )
        if not total_portfolio_value or total_portfolio_value == 0:
            total_portfolio_value = self.holdings_data[mkt_val_col].sum()

        saved_targets = self.config.get("rebalancing_targets", {})

        # Prepare data for the table, including CASH
        table_data = []
        cash_row = None

        for i, row in self.holdings_data.iterrows():
            symbol = row["Symbol"]
            current_value = row[mkt_val_col]
            current_pct = (
                (current_value / total_portfolio_value) * 100.0
                if total_portfolio_value > 0
                else 0.0
            )

            if symbol == CASH_SYMBOL_CSV:
                cash_row = {
                    "Symbol": symbol,
                    "Asset Class": "Cash",
                    "Current Value": current_value,
                    "Current %": current_pct,
                    "Target %": current_pct,  # Initial target is current
                    "Target %": saved_targets.get(symbol, current_pct),
                    "Target Value": current_value,
                    "Drift %": 0.0,
                }
            else:
                table_data.append(
                    {
                        "Symbol": symbol,
                        "Asset Class": row.get(
                            "quoteType", row.get("Sector", "Unknown")
                        ),
                        "Current Value": current_value,
                        "Current %": current_pct,
                        "Target %": current_pct,  # Initial target is current
                        "Target %": saved_targets.get(symbol, current_pct),
                        "Target Value": current_value,
                        "Drift %": 0.0,
                    }
                )

        # Sort non-cash holdings by current value descending, then add cash at the end
        table_data.sort(key=lambda x: x["Current Value"], reverse=True)
        if cash_row:  # Ensure cash is always at the bottom
            table_data.append(cash_row)

        self.target_allocation_table.setRowCount(len(table_data))

        # Disconnect signal to prevent infinite loop during population
        try:
            self.target_allocation_table.cellChanged.disconnect(
                self._update_target_total_label
            )
        except TypeError:  # Handle case where it might not be connected yet
            pass

        for i, data_row in enumerate(table_data):
            symbol = data_row["Symbol"]
            asset_class = data_row["Asset Class"]
            current_value = data_row["Current Value"]
            current_pct = data_row["Current %"]
            target_pct = data_row["Target %"]
            target_value = data_row["Target Value"]
            drift_pct = data_row["Drift %"]

            self.target_allocation_table.setItem(i, 0, QTableWidgetItem(symbol))
            self.target_allocation_table.setItem(
                i, 1, QTableWidgetItem(str(asset_class))
            )
            self.target_allocation_table.setItem(
                i,
                2,
                QTableWidgetItem(
                    format_currency_value(current_value, self._get_currency_symbol())
                ),
            )
            self.target_allocation_table.setItem(
                i, 3, QTableWidgetItem(format_percentage_value(current_pct))
            )
            self.target_allocation_table.setItem(
                i, 4, QTableWidgetItem(format_percentage_value(target_pct))
            )
            self.target_allocation_table.setItem(
                i,
                5,
                QTableWidgetItem(
                    format_currency_value(target_value, self._get_currency_symbol())
                ),
            )
            drift_item = QTableWidgetItem(format_percentage_value(drift_pct))
            self.target_allocation_table.setItem(i, 6, drift_item)

        # Reconnect signal after population
        self.target_allocation_table.cellChanged.connect(
            self._update_target_total_label
        )
        # Manually trigger update after initial load
        self._update_target_total_label()

    def _handle_rebalance_calculation(self):
        target_alloc_pct = {}
        for i in range(self.target_allocation_table.rowCount()):
            symbol = self.target_allocation_table.item(i, 0).text()
            target_pct = float(
                self.target_allocation_table.item(i, 4).text().replace("%", "")
            )
            target_alloc_pct[symbol] = target_pct

        new_cash = float(self.cash_to_add_line_edit.text() or 0.0)
        display_currency = self.currency_combo.currentText()

        trades_df, summary = calculate_rebalancing_trades(
            self.holdings_data, target_alloc_pct, new_cash, display_currency
        )

        self.suggested_trades_table.setRowCount(trades_df.shape[0])
        for i, row in trades_df.iterrows():
            action = row["Action"]
            symbol = row["Symbol"]
            account = row["Account"]
            quantity = row["Quantity"]
            price = row["Current Price"]
            trade_value = row["Trade Value"]
            note = row["Note"]

            action_item = QTableWidgetItem(action)
            quantity_item = QTableWidgetItem(format_float_with_commas(quantity, 4))
            trade_value_item = QTableWidgetItem(
                format_currency_value(trade_value, self._get_currency_symbol())
            )

            if action == "SELL":
                action_item.setForeground(self.QCOLOR_LOSS_THEMED)
                quantity_item.setForeground(self.QCOLOR_LOSS_THEMED)
                trade_value_item.setForeground(self.QCOLOR_LOSS_THEMED)
            elif action == "BUY":
                action_item.setForeground(self.QCOLOR_GAIN_THEMED)
                quantity_item.setForeground(self.QCOLOR_GAIN_THEMED)
                trade_value_item.setForeground(self.QCOLOR_GAIN_THEMED)

            self.suggested_trades_table.setItem(i, 0, action_item)
            self.suggested_trades_table.setItem(i, 1, QTableWidgetItem(symbol))
            self.suggested_trades_table.setItem(i, 2, QTableWidgetItem(account))
            self.suggested_trades_table.setItem(i, 3, quantity_item)
            self.suggested_trades_table.setItem(
                i,
                4,
                QTableWidgetItem(
                    format_currency_value(price, self._get_currency_symbol())
                ),
            )
            self.suggested_trades_table.setItem(i, 5, trade_value_item)
            self.suggested_trades_table.setItem(i, 6, QTableWidgetItem(note))

        currency_symbol = self._get_currency_symbol()
        for key, value in summary.items():
            if key == "Total Portfolio Value (After Rebalance)":
                self.total_portfolio_value_label.setText(
                    format_currency_value(value, currency_symbol)
                )
            elif key == "Total Value to Sell":
                self.total_value_to_sell_label.setText(
                    format_currency_value(value, currency_symbol)
                )
            elif key == "Total Value to Buy":
                self.total_value_to_buy_label.setText(
                    format_currency_value(value, currency_symbol)
                )
            elif key == "Net Cash Change":
                self.net_cash_change_label.setText(
                    format_currency_value(value, currency_symbol, show_plus_sign=True)
                )
            elif key == "Estimated Number of Trades":
                self.estimated_trades_label.setText(str(value))

    def _update_target_total_label(
        self, changed_row: int = -1, changed_column: int = -1
    ):
        # Block signals to prevent infinite loop during internal updates
        self.target_allocation_table.blockSignals(True)

        display_currency = self.currency_combo.currentText()
        try:
            new_cash_input = float(self.cash_to_add_line_edit.text() or 0.0)
        except ValueError:
            new_cash_input = 0.0

        mkt_val_col = f"Market Value ({display_currency})"
        current_total_market_value = self.holdings_data[mkt_val_col].sum()
        total_portfolio_value_after_new_cash = (
            current_total_market_value + new_cash_input
        )

        total_pct_sum = 0.0
        cash_row_index = -1

        # First pass: Collect all target percentages and find cash row
        # This loop will also update the target value and drift for the *changed* row if applicable
        # and for all other rows based on their existing target percentage.
        for i in range(self.target_allocation_table.rowCount()):
            item_symbol = self.target_allocation_table.item(i, 0)
            if item_symbol:
                symbol = item_symbol.text()
                if symbol == "$CASH":
                    cash_row_index = i
                    # We'll handle CASH separately after summing non-cash
                    continue

                try:
                    target_pct_str = (
                        self.target_allocation_table.item(i, 4).text().replace("%", "")
                    )
                    target_pct = float(target_pct_str)
                    total_pct_sum += target_pct

                    # Recalculate Target Value for this row based on its Target %
                    new_target_value = total_portfolio_value_after_new_cash * (
                        target_pct / 100.0
                    )
                    self.target_allocation_table.setItem(
                        i,
                        5,
                        QTableWidgetItem(
                            format_currency_value(
                                new_target_value, self._get_currency_symbol()
                            )
                        ),
                    )

                    # Recalculate Drift for this row
                    current_pct_item = self.target_allocation_table.item(i, 3)
                    current_pct_str = current_pct_item.text().replace("%", "")
                    current_pct = float(current_pct_str)
                    drift_pct = current_pct - target_pct
                    drift_item = self.target_allocation_table.item(i, 6)
                    drift_item.setText(
                        format_percentage_value(drift_pct, show_plus_sign=True)
                    )
                    if drift_pct > 0.01:
                        drift_item.setForeground(self.QCOLOR_GAIN_THEMED)
                    elif drift_pct < -0.01:
                        drift_item.setForeground(self.QCOLOR_LOSS_THEMED)
                    else:
                        drift_item.setForeground(self.QCOLOR_TEXT_PRIMARY_THEMED)

                except (ValueError, AttributeError, IndexError) as e:
                    logging.error(f"Error processing row {i} in rebalancing table: {e}")
                    pass

        # Adjust CASH target percentage and value if cash row exists
        if cash_row_index != -1:
            remaining_for_cash = 100.0 - total_pct_sum
            if remaining_for_cash < 0:
                remaining_for_cash = 0.0

            # Update CASH target percentage
            self.target_allocation_table.setItem(
                cash_row_index,
                4,
                QTableWidgetItem(format_percentage_value(remaining_for_cash)),
            )

            # Recalculate CASH target value
            cash_target_value = total_portfolio_value_after_new_cash * (
                remaining_for_cash / 100.0
            )
            self.target_allocation_table.setItem(
                cash_row_index,
                5,
                QTableWidgetItem(
                    format_currency_value(
                        cash_target_value, self._get_currency_symbol()
                    )
                ),
            )

            # Update CASH drift
            current_cash_pct_item = self.target_allocation_table.item(cash_row_index, 3)
            current_cash_pct_str = current_cash_pct_item.text().replace("%", "")
            current_cash_pct = float(current_cash_pct_str)
            cash_drift_pct = current_cash_pct - remaining_for_cash
            cash_drift_item = self.target_allocation_table.item(cash_row_index, 6)
            cash_drift_item.setText(
                format_percentage_value(cash_drift_pct, show_plus_sign=True)
            )
            if cash_drift_pct > 0.01:
                cash_drift_item.setForeground(self.QCOLOR_GAIN_THEMED)
            elif cash_drift_pct < -0.01:
                cash_drift_item.setForeground(self.QCOLOR_LOSS_THEMED)
            else:
                cash_drift_item.setForeground(self.QCOLOR_TEXT_PRIMARY_THEMED)

            # Add cash percentage to total_pct_sum for final label update
            total_pct_sum += remaining_for_cash

        # Update the total percentage label
        self.target_percent_total_label.setText(f"{total_pct_sum:.2f}%")

        # Color the total label based on whether it's exactly 100%
        if abs(total_pct_sum - 100.0) > 0.01:
            self.target_percent_total_label.setStyleSheet("color: red;")
        else:
            self.target_percent_total_label.setStyleSheet("")

        # Reconnect signal after all updates are done
        self.target_allocation_table.blockSignals(False)

    def _add_new_symbol_to_target_table(self):
        """Adds a new row to the target allocation table for a user-inputted symbol."""
        symbol, ok = QInputDialog.getText(self, "Add Symbol", "Enter Symbol:")
        if ok and symbol:
            symbol = symbol.upper().strip()
            # Check if symbol already exists
            for row in range(self.target_allocation_table.rowCount()):
                if self.target_allocation_table.item(row, 0).text() == symbol:
                    QMessageBox.warning(
                        self,
                        "Symbol Exists",
                        f"The symbol '{symbol}' is already in the table.",
                    )
                    return

            row_position = self.target_allocation_table.rowCount()
            self.target_allocation_table.insertRow(row_position)

            self.target_allocation_table.setItem(
                row_position, 0, QTableWidgetItem(symbol)
            )
            self.target_allocation_table.setItem(
                row_position, 1, QTableWidgetItem("N/A")
            )
            self.target_allocation_table.setItem(
                row_position,
                2,
                QTableWidgetItem(format_currency_value(0, self._get_currency_symbol())),
            )
            self.target_allocation_table.setItem(
                row_position, 3, QTableWidgetItem(format_percentage_value(0))
            )
            self.target_allocation_table.setItem(
                row_position, 4, QTableWidgetItem(format_percentage_value(0))
            )
            self.target_allocation_table.setItem(
                row_position,
                5,
                QTableWidgetItem(format_currency_value(0, self._get_currency_symbol())),
            )
            self.target_allocation_table.setItem(
                row_position, 6, QTableWidgetItem(format_percentage_value(0))
            )

    def _remove_selected_symbol_from_target_table(self):
        """Removes the selected row from the target allocation table."""
        selected_row = self.target_allocation_table.currentRow()
        if selected_row >= 0:
            self.target_allocation_table.removeRow(selected_row)
            self._update_target_total_label()  # Recalculate total
        else:
            QMessageBox.warning(
                self, "No Selection", "Please select a symbol to remove."
            )

    @Slot()
    def _clear_target_allocation_table(self):
        """Clears the contents of the target allocation table."""
        logging.info("Clearing target allocation table.")
        self.target_allocation_table.clearContents()
        self.target_allocation_table.setRowCount(0)
        # Removed call to _populate_target_allocation_table()

    def _init_rebalancing_tab_widgets(self):
        """Initializes widgets for the Rebalancing tab."""
        self.rebalancing_tab.setObjectName("RebalancingTab")
        main_layout = QVBoxLayout(self.rebalancing_tab)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)

        # Group 1: Rebalancing Summary (Top)
        summary_group = QGroupBox("Rebalancing Summary")
        summary_layout = QHBoxLayout(summary_group)
        summary_layout.setSpacing(10)

        self.total_portfolio_value_label = QLabel("N/A")
        self.total_value_to_sell_label = QLabel("N/A")
        self.total_value_to_buy_label = QLabel("N/A")
        self.net_cash_change_label = QLabel("N/A")
        self.estimated_trades_label = QLabel("N/A")

        summary_layout.addWidget(QLabel("<b>Total Value (After):</b>"))
        summary_layout.addWidget(self.total_portfolio_value_label)

        separator1 = QFrame()
        separator1.setFrameShape(QFrame.VLine)
        separator1.setFrameShadow(QFrame.Sunken)
        summary_layout.addWidget(separator1)

        summary_layout.addWidget(QLabel("<b>Sell:</b>"))
        summary_layout.addWidget(self.total_value_to_sell_label)

        separator2 = QFrame()
        separator2.setFrameShape(QFrame.VLine)
        separator2.setFrameShadow(QFrame.Sunken)
        summary_layout.addWidget(separator2)

        summary_layout.addWidget(QLabel("<b>Buy:</b>"))
        summary_layout.addWidget(self.total_value_to_buy_label)

        separator3 = QFrame()
        separator3.setFrameShape(QFrame.VLine)
        separator3.setFrameShadow(QFrame.Sunken)
        summary_layout.addWidget(separator3)

        summary_layout.addWidget(QLabel("<b>Net Cash:</b>"))
        summary_layout.addWidget(self.net_cash_change_label)

        separator4 = QFrame()
        separator4.setFrameShape(QFrame.VLine)
        separator4.setFrameShadow(QFrame.Sunken)
        summary_layout.addWidget(separator4)

        summary_layout.addWidget(QLabel("<b>Trades:</b>"))
        summary_layout.addWidget(self.estimated_trades_label)

        summary_layout.addStretch(1)
        main_layout.addWidget(summary_group, 0)  # Stretch factor 0 for summary

        # Bottom section with two tables
        bottom_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(bottom_splitter, 1)  # Stretch factor 1 for splitter

        # Group 2: Target Allocation & Controls (Left)
        target_group = QGroupBox("Target Allocation & Controls")
        target_layout = QVBoxLayout(target_group)
        bottom_splitter.addWidget(target_group)

        controls_layout = QHBoxLayout()
        self.load_current_holdings_button = QPushButton("Load Holdings")
        self.add_symbol_button = QPushButton("Add")
        self.remove_symbol_button = QPushButton("Remove")
        self.cash_to_add_line_edit = QLineEdit()
        # Allow positive and negative numbers for cash injection/withdrawal
        cash_validator = QDoubleValidator(-1e12, 1e12, 2, self)
        cash_validator.setNotation(QDoubleValidator.StandardNotation)
        self.cash_to_add_line_edit.setValidator(cash_validator)

        self.cash_to_add_line_edit.setPlaceholderText("Cash to Add/Withdraw (-)")
        self.calculate_rebalance_button = QPushButton("Calculate Rebalance")
        self.calculate_rebalance_button.setObjectName("CalculateRebalanceButton")

        self.add_symbol_button.clicked.connect(self._add_new_symbol_to_target_table)
        self.remove_symbol_button.clicked.connect(
            self._remove_selected_symbol_from_target_table
        )

        controls_layout.addWidget(self.load_current_holdings_button)
        controls_layout.addWidget(self.add_symbol_button)
        controls_layout.addWidget(self.remove_symbol_button)
        self.clear_targets_button = QPushButton("Clear")
        controls_layout.addWidget(self.clear_targets_button)
        self.clear_targets_button.clicked.connect(self._clear_target_allocation_table)
        controls_layout.addSpacerItem(
            QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        )
        controls_layout.addWidget(self.cash_to_add_line_edit)
        controls_layout.addWidget(self.calculate_rebalance_button)
        target_layout.addLayout(controls_layout)

        self.target_allocation_table = QTableWidget()
        self.target_allocation_table.setColumnCount(7)
        self.target_allocation_table.setHorizontalHeaderLabels(
            [
                "Symbol",
                "Asset Class",
                "Current Value",
                "Current %",
                "Target %",
                "Target Value",
                "Drift %",
            ]
        )
        self.target_allocation_table.verticalHeader().setVisible(False)
        target_layout.addWidget(self.target_allocation_table)

        target_summary_layout = QHBoxLayout()
        target_summary_layout.addStretch()
        target_summary_layout.addWidget(QLabel("Target Total:"))
        self.target_percent_total_label = QLabel("0.00%")
        target_summary_layout.addWidget(self.target_percent_total_label)
        target_layout.addLayout(target_summary_layout)

        # Group 3: Suggested Trades (Right)
        trades_group = QGroupBox("Suggested Trades")
        trades_layout = QVBoxLayout(trades_group)
        bottom_splitter.addWidget(trades_group)

        self.suggested_trades_table = QTableWidget()
        self.suggested_trades_table.setColumnCount(7)
        self.suggested_trades_table.setHorizontalHeaderLabels(
            [
                "Action",
                "Symbol",
                "Account",
                "Quantity",
                "Current Price",
                "Trade Value",
                "Note",
            ]
        )
        self.suggested_trades_table.verticalHeader().setVisible(False)
        trades_layout.addWidget(self.suggested_trades_table)

        # Defer setting the splitter sizes until the UI is shown to get accurate widths
        QTimer.singleShot(
            0, lambda: bottom_splitter.setSizes([self.width() // 2, self.width() // 2])
        )

    def _init_correlation_matrix_tab(self):
        """Initializes the Correlation Matrix sub-tab."""
        logging.debug("Initializing Correlation Matrix tab.")
        corr_layout = QVBoxLayout(self.correlation_matrix_tab)
        self.correlation_matrix_tab.setLayout(corr_layout)

        self.correlation_fig = Figure(figsize=(8, 8), dpi=CHART_DPI)
        self.correlation_canvas = FigureCanvas(self.correlation_fig)
        explanation_label = QLabel(
            """<p>The <b>Correlation Matrix</b> visualizes the statistical relationship between the historical daily returns of assets in your portfolio. Each value, ranging from -1 to +1, indicates how two assets move together:</p>
            <ul>
            <li><b>+1 (Perfect Positive Correlation):</b> Assets move in the exact same direction.</li>
            <li><b>-1 (Perfect Negative Correlation):</b> Assets move in opposite directions.</li>
            <li><b>0 (No Linear Correlation):</b> No consistent linear relationship.</li>
            </ul>
            <p><b>What is calculated:</b> This matrix calculates the Pearson correlation coefficient for the daily returns of each pair of assets.</p>
            <p><b>Why it's useful:</b> Understanding correlations is crucial for diversification. Combining assets with low or negative correlations can help reduce overall portfolio risk by offsetting losses in one asset with gains in another.</p>"""
        )
        explanation_label.setWordWrap(True)
        explanation_label.setObjectName("ExplanationLabel")  # For potential styling
        corr_layout.addWidget(explanation_label)

        # --- Center the canvas horizontally ---
        h_layout = QHBoxLayout()
        h_layout.addStretch(1)
        h_layout.addWidget(self.correlation_canvas)
        h_layout.addStretch(1)
        corr_layout.addLayout(h_layout)
        # --- End centering ---

        corr_layout.addStretch(1)

        logging.debug("Correlation Matrix tab initialized.")

    def _init_factor_analysis_tab(self):
        """Initializes the Factor Analysis sub-tab."""
        logging.debug("Initializing Factor Analysis tab.")
        factor_layout = QVBoxLayout(self.factor_analysis_tab)
        # Reduce margins to maximize space
        factor_layout.setContentsMargins(5, 5, 5, 5)
        self.factor_analysis_tab.setLayout(factor_layout)

        # --- 1. Explanation Area (Collapsible-like GroupBox) ---
        explanation_group = QGroupBox("Understanding Factor Analysis")
        explanation_layout = QVBoxLayout()
        explanation_label = QLabel(
            """<p>The <b>Factor Analysis</b> tab performs a regression to explain portfolio returns using market factors:</p>
            <ul>
            <li><b>Mkt-RF:</b> Sensitivity to the overall stock market.</li>
            <li><b>SMB:</b> Exposure to Small-Cap (vs Large-Cap).</li>
            <li><b>HML:</b> Exposure to Value (vs Growth).</li>
            <li><b>UMD:</b> Exposure to Momentum (Recent Winners).</li>
            </ul>
            <p><i>Click 'Run Analysis' to estimate Factor Betas and R-squared.</i></p>"""
        )
        explanation_label.setWordWrap(True)
        explanation_layout.addWidget(explanation_label)
        explanation_group.setLayout(explanation_layout)
        factor_layout.addWidget(explanation_group)

        # --- 2. Controls Area ---
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(QLabel("Factor Model:"))
        self.factor_model_combo = QComboBox()
        self.factor_model_combo.addItems(["Fama-French 3-Factor", "Carhart 4-Factor"])
        controls_layout.addWidget(self.factor_model_combo)
        self.run_factor_analysis_button = QPushButton("Run Analysis")
        self.run_factor_analysis_button.clicked.connect(self._run_factor_analysis)
        controls_layout.addWidget(self.run_factor_analysis_button)
        controls_layout.addStretch(1)
        factor_layout.addLayout(controls_layout)

        # --- 3. Main Content Area (Splitter for Chart & Results) ---
        splitter = QSplitter(Qt.Vertical)
        
        # Chart Canvas
        self.factor_fig = Figure(figsize=(8, 8), dpi=CHART_DPI)
        self.factor_canvas = FigureCanvas(self.factor_fig)
        self.factor_canvas.setObjectName("FactorCanvas")
        # Ensure canvas expands and has a minimum size
        self.factor_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.factor_canvas.setMinimumHeight(400) # Increased minimum height
        splitter.addWidget(self.factor_canvas)

        # Results Text
        self.factor_analysis_results_text = QTextEdit()
        self.factor_analysis_results_text.setReadOnly(True)
        self.factor_analysis_results_text.setPlaceholderText("Analysis results will appear here...")
        self.factor_analysis_results_text.setMinimumHeight(100) # Minimum height for text
        splitter.addWidget(self.factor_analysis_results_text)

        # Set initial splitter sizes (70% chart, 30% text)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        # Ensure splitter itself expands
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        factor_layout.addWidget(splitter, 1) # Give all remaining space to the splitter

        logging.debug("Factor Analysis tab initialized.")

    def _run_factor_analysis(self):
        """Runs the factor analysis regression and updates the UI."""
        if not hasattr(self, "portfolio_daily_returns") or self.portfolio_daily_returns.empty:
            QMessageBox.warning(self, "No Data", "Please load data and ensure portfolio returns are calculated first.")
            return

        model_name = self.factor_model_combo.currentText()
        logging.info(f"Running Factor Analysis with model: {model_name}")
        
        self.set_status("Running Factor Analysis...")
        self.run_factor_analysis_button.setEnabled(False)
        self.run_factor_analysis_button.setText("Running...")
        QApplication.processEvents() # Force UI update

        try:
            # We need to pass benchmark data if available to avoid re-fetching SPY if possible
            # Check if we have 'SPY' in our historical data
            benchmark_data = None
            # logic to extract benchmark data if it exists in self.historical_data_cache or similar
            # For now, let run_factor_regression handle fetching/using its internal logic.
            
            results = run_factor_regression(
                self.portfolio_daily_returns,
                model_name=model_name
            )

            if results is None:
                self.factor_analysis_results_text.setText("Regression failed. See logs for details (likely insufficient data or API error).")
                return

            # Update Text Results
            self.factor_analysis_results_text.setText(results.summary().as_text())
            
            # --- Update Charts ---
            self.factor_fig.clear()
            
            # 1. Bar Chart of Betas
            # params includes Intercept (Alpha) and factors
            params = results.params
            bse = results.bse # Standard errors
            
            ax_betas = self.factor_fig.add_subplot(211) # Top half
            # Exclude Intercept (Alpha) from the beta bar chart usually, or keep it distinct?
            # Alpha is return, Betas are sensitivity. Scale is different. 
            # Often better to show Betas. Alpha is shown in text.
            betas = params.drop("const") if "const" in params else params
            errors = bse.drop("const") if "const" in bse else bse
            
            # Colors for bars
            colors = [self.QCOLOR_ACCENT_THEMED.name()] * len(betas)
            
            betas.plot(kind='bar', ax=ax_betas, yerr=errors, capsize=4, color=colors, rot=0)
            ax_betas.set_title(f"Factor Betas ({model_name})")
            ax_betas.set_ylabel("Beta Coefficient")
            ax_betas.axhline(0, color='black', linewidth=0.8)
            ax_betas.grid(axis='y', linestyle='--', alpha=0.7)
            
            # 2. Scatter Plot (Model Fit) - Actual vs Predicted or Factor exposure
            # A common visual is Portfolio Excess Return vs Market Excess Return (Mkt-RF)
            # This visualizes the primary beta.
            ax_scatter = self.factor_fig.add_subplot(212)
            
            # We need the data used in regression to plot points. 
            # results.model.exog contains the inputs (Factors)
            # results.model.endog contains the target (Portfolio Excess Returns)
            
            # Extract Mkt-RF (usually index 1 if const is 0)
            exog_names = results.model.exog_names
            if "Mkt-RF" in exog_names:
                mkt_rf_idx = exog_names.index("Mkt-RF")
                mkt_rf_data = results.model.exog[:, mkt_rf_idx]
                portfolio_excess = results.model.endog
                
                ax_scatter.scatter(mkt_rf_data, portfolio_excess, alpha=0.6, label="Monthly Returns")
                
                # Plot regression line for Mkt-RF
                # partial regression plot or just simple univariable line? 
                # Let's plot the "Market Line" which is just slope = Beta_Mkt
                beta_mkt = params["Mkt-RF"]
                alpha = params["const"] if "const" in params else 0
                
                # Generate x range
                x_range = np.linspace(min(mkt_rf_data), max(mkt_rf_data), 100)
                y_pred_mkt = alpha + beta_mkt * x_range
                
                ax_scatter.plot(x_range, y_pred_mkt, color='red', linewidth=2, label=f"Beta: {beta_mkt:.2f}")
                
                ax_scatter.set_title("Portfolio Excess Returns vs. Market (Mkt-RF)")
                ax_scatter.set_xlabel("Market Excess Return")
                ax_scatter.set_ylabel("Portfolio Excess Return")
                ax_scatter.legend()
                ax_scatter.grid(True, linestyle='--', alpha=0.6)
            else:
                 ax_scatter.text(0.5, 0.5, "Mkt-RF factor not found for scatter plot.", ha='center')

            self.factor_fig.tight_layout()
            self.factor_canvas.draw()
            
            self.set_status(f"Factor Analysis complete ({model_name}).")

        except Exception as e:
            logging.error(f"Error in _run_factor_analysis: {e}")
            self.factor_analysis_results_text.setText(f"Error running analysis:\n{e}")
            QMessageBox.critical(self, "Analysis Error", f"An error occurred:\n{e}")
        finally:
            self.run_factor_analysis_button.setEnabled(True)
            self.run_factor_analysis_button.setText("Run Analysis")

    def _init_scenario_analysis_tab(self):
        """Initializes the Scenario Analysis sub-tab."""
        logging.debug("Initializing Scenario Analysis tab.")
        scenario_layout = QVBoxLayout(self.scenario_analysis_tab)
        self.scenario_analysis_tab.setLayout(scenario_layout)

        explanation_label = QLabel(
            """The <b>Scenario Analysis</b> tab allows you to simulate the potential impact of hypothetical market movements (scenarios) on your portfolio's value and returns.<br>
            <b>What is calculated:</b> You define a scenario by specifying changes in key market variables (e.g., a percentage change in a specific index or factor). The system then estimates the potential impact on your portfolio based on its current holdings and sensitivities to these variables.<br>            
            <b>Why it's useful:</b> This tool is valuable for stress-testing your portfolio against adverse market conditions, assessing potential downside risks, and making informed decisions about portfolio adjustments to mitigate those risks."""
        )
        explanation_label.setWordWrap(True)
        explanation_label.setObjectName("ExplanationLabel")  # For potential styling
        scenario_layout.addWidget(explanation_label)

        input_form_layout = QFormLayout()

        # Preset Scenarios
        preset_scenario_layout = QHBoxLayout()
        self.preset_scenario_combo = QComboBox()
        self.preset_scenario_combo.addItem("Select a Preset Scenario")
        self.preset_scenario_combo.addItem("S&P500 Change: +30% (Mkt-RF: 0.30)")
        self.preset_scenario_combo.addItem("S&P500 Change: +20% (Mkt-RF: 0.20)")
        self.preset_scenario_combo.addItem("S&P500 Change: +10% (Mkt-RF: 0.10)")
        self.preset_scenario_combo.addItem("S&P500 Change: +5% (Mkt-RF: 0.05)")
        self.preset_scenario_combo.addItem("S&P500 Change: -5% (Mkt-RF: -0.05)")
        self.preset_scenario_combo.addItem("S&P500 Change: -10% (Mkt-RF: -0.10)")
        self.preset_scenario_combo.addItem("S&P500 Change: -20% (Mkt-RF: -0.20)")
        self.preset_scenario_combo.addItem("S&P500 Change: -30% (Mkt-RF: -0.30)")
        self.preset_scenario_combo.insertSeparator(9)
        self.preset_scenario_combo.addItem("Market Downturn (Mkt-RF: -0.10, HML: 0.05)")
        self.preset_scenario_combo.addItem(
            "Interest Rate Hike (Mkt-RF: -0.05, SMB: -0.03, HML: 0.07)"
        )
        self.preset_scenario_combo.addItem(
            "Inflation Surge (Mkt-RF: -0.07, HML: 0.08, RMW: -0.02)"
        )
        preset_scenario_layout.addWidget(self.preset_scenario_combo)

        self.load_preset_button = QPushButton("Load Preset")
        self.load_preset_button.clicked.connect(self._load_preset_scenario)
        preset_scenario_layout.addWidget(self.load_preset_button)
        input_form_layout.addRow("Preset Scenarios:", preset_scenario_layout)

        self.scenario_input_line_edit = QLineEdit()
        self.scenario_input_line_edit.setPlaceholderText(
            "e.g., Mkt-RF: -0.10, SMB: 0.05"
        )
        self.scenario_input_line_edit.setMinimumWidth(400)  # Set a fixed minimum width
        self.scenario_input_line_edit.setSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.Preferred
        )
        input_form_layout.addRow(
            "Custom Scenario (Factor: Shock):", self.scenario_input_line_edit
        )
        scenario_layout.addLayout(input_form_layout)

        self.run_scenario_button = QPushButton("Run Scenario")
        scenario_layout.addWidget(self.run_scenario_button)

        results_group_box = QGroupBox("Scenario Results")
        results_layout = QVBoxLayout(results_group_box)
        self.scenario_impact_label = QLabel("Estimated Portfolio Impact: N/A")
        results_layout.addWidget(self.scenario_impact_label)
        results_group_box.setLayout(results_layout)
        scenario_layout.addWidget(results_group_box)

        scenario_layout.addStretch(1)

        logging.debug("Scenario Analysis tab initialized.")

    def _update_transaction_log_tables(self):
        """Populates the stock and cash transaction log tables."""
        # This method will be filled in handle_results or a dedicated update sequence
        # For now, it's a placeholder. The actual population logic will be added
        # where self.original_data is available and processed.
        pass

    def _get_scope_label_for_charts(self) -> str:
        """Helper to get a consistent scope label for chart titles."""
        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)
        scope_label = "Overall Portfolio"
        if (
            self.available_accounts
            and num_selected > 0
            and num_selected != num_available
        ):
            if num_selected == 1:
                scope_label = f"Account: {self.selected_accounts[0]}"
            else:
                scope_label = f"Selected Accounts ({num_selected}/{num_available})"
        elif not self.available_accounts:
            scope_label = "No Accounts Available"
        return scope_label

    def _update_dividend_spinbox_default(self):
        """Updates the dividend periods spinbox based on the selected aggregation period."""
        period = self.dividend_period_combo.currentText()
        if period == "Annual":
            self.dividend_periods_spinbox.setValue(
                self.config.get("dividend_chart_default_periods_annual", 10)
            )
        elif period == "Quarterly":
            self.dividend_periods_spinbox.setValue(
                self.config.get("dividend_chart_default_periods_quarterly", 12)
            )
        elif period == "Monthly":
            self.dividend_periods_spinbox.setValue(
                self.config.get("dividend_chart_default_periods_monthly", 24)
            )

    def _create_status_bar(self):
        """Creates and configures the application's status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        self.status_bar.addWidget(self.status_label, 1)

        # --- Themed QColor objects ---
        # Initialize with light theme colors; apply_theme will update them.
        self.QCOLOR_GAIN_THEMED = QColor(COLOR_GAIN)
        self.QCOLOR_LOSS_THEMED = QColor(COLOR_LOSS)
        self.QCOLOR_TEXT_PRIMARY_THEMED = QColor(COLOR_TEXT_DARK)  # Default main text
        self.QCOLOR_TEXT_SECONDARY_THEMED = QColor(
            COLOR_TEXT_SECONDARY
        )  # Default secondary text
        self.QCOLOR_BACKGROUND_THEMED = QColor(
            COLOR_BG_DARK
        )  # Default main background (QWidget, QMainWindow)
        self.QCOLOR_HEADER_BACKGROUND_THEMED = QColor(
            COLOR_BG_HEADER_LIGHT
        )  # Default header/frame background
        self.QCOLOR_BORDER_THEMED = QColor(COLOR_BORDER_LIGHT)  # Default border color
        self.QCOLOR_ACCENT_THEMED = QColor(COLOR_ACCENT_TEAL)  # Default accent color
        self.QCOLOR_INPUT_BACKGROUND_THEMED = QColor(
            COLOR_BG_DARK
        )  # For inputs like QLineEdit, QComboBox
        self.QCOLOR_INPUT_TEXT_THEMED = QColor(
            COLOR_TEXT_DARK
        )  # For text within inputs
        self.QCOLOR_TABLE_ALT_ROW_THEMED = QColor(
            "#fAfBff"
        )  # Light theme alternate row color

        self.yahoo_attribution_label = QLabel(  # RESTORED
            "Financial data provided by Yahoo Finance"
        )
        self.yahoo_attribution_label.setObjectName("YahooAttributionLabel")
        self.status_bar.addPermanentWidget(self.yahoo_attribution_label)  # RESTORED
        # --- ADD Progress Bar ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("StatusBarProgressBar")
        self.progress_bar.setVisible(False)  # Initially hidden
        self.status_bar.addPermanentWidget(self.progress_bar)  # RESTORED
    def _on_update_accounts_clicked(self):
        """
        Handles the 'Update Accounts' button click.
        Updates the graph date range based on the selected accounts, then refreshes data.
        """
        self._update_graph_date_range_for_accounts()
        self.refresh_data(force_historical_refresh=False)

    def _update_graph_date_range_for_accounts(self):
        """
        Updates the graph start date based on the earliest transaction of the selected accounts.
        If no transactions are found or data is not loaded, does nothing.
        """
        if (
            not hasattr(self, "all_transactions_df_cleaned_for_logic")
            or self.all_transactions_df_cleaned_for_logic.empty
        ):
            return

        # Use self.selected_accounts if available (which is updated by the menu),
        # otherwise fall back to config.
        selected_accounts = getattr(self, "selected_accounts", [])
        if not selected_accounts:
             selected_accounts = self.config.get("selected_accounts", [])
        
        df = self.all_transactions_df_cleaned_for_logic

        # Filter by selected accounts if any are selected
        if selected_accounts:
            # Filter df for transactions belonging to selected accounts
            # Note: 'Account' column name might vary, check get_column_definitions or standard name
            # Usually it's 'Account' in the cleaned DF.
            if "Account" in df.columns:
                df = df[df["Account"].isin(selected_accounts)]
            else:
                logging.warning(
                    "Could not filter by account for date range: 'Account' column missing."
                )
                return

        if df.empty:
            return

        # Find the earliest date
        if "Date" in df.columns:
            try:
                min_date = df["Date"].min()
                if pd.notna(min_date):
                    # Convert to python date if it's a timestamp
                    if isinstance(min_date, (pd.Timestamp, datetime)):
                        min_date = min_date.date()

                    # Use the exact start date as requested
                    new_start_date = min_date
                    
                    # Check current UI date
                    current_ui_date = self.graph_start_date_edit.date().toPython()
                    
                    # Check if we should force update (e.g. if "All" or "Presets..." is selected)
                    should_force_update = False
                    if hasattr(self, "date_preset_combo"):
                        current_preset = self.date_preset_combo.currentText()
                        if current_preset in ["Presets...", "All"]:
                            should_force_update = True

                    # Update if current date is LATER than new start date (we want to expand range)
                    # OR if we should force update (e.g. user selected "All")
                    if current_ui_date > new_start_date or should_force_update:
                        logging.debug(f"DEBUG: _update_graph_date_range_for_accounts updating date to {new_start_date}")
                        self.graph_start_date_edit.setDate(QDate(new_start_date))
                        
                        # Explicitly reset preset combo to "Presets..." to avoid confusion
                        # But if it was "All", maybe keep it? For now reset to 0 (Presets...)
                        if hasattr(self, "date_preset_combo"):
                            self.date_preset_combo.setCurrentIndex(0)
                            
                        logging.debug(
                            f"DEBUG: Auto-updated graph start date to {new_start_date} based on selected accounts (was {current_ui_date})."
                        )
                    else:
                        logging.debug(
                            f"DEBUG: Kept existing graph start date {current_ui_date} as it is valid (<= {new_start_date})."
                        )
            except Exception as e:
                logging.error(f"Error calculating min date for graph: {e}")
        else:
            logging.warning(
                "Could not determine min date: 'Date' column missing in transactions."
            )


    @Slot(bool)
    def refresh_data(self, force_historical_refresh: Optional[bool] = None):
        """
        Initiates the background calculation process via the worker thread.
        Loads data from the SQLite database.

        Gathers current UI settings (display currency, filters, graph parameters),
        prepares arguments for the portfolio logic functions, creates a
        `PortfolioCalculatorWorker`, and starts it in the thread pool. Disables UI
        controls during calculation.

        Args:
            force_historical_refresh (Optional[bool]): If True, forces a full re-fetch of historical data.
                                                        If False, uses cache for historical. If None (default),
                                                        the worker decides based on data staleness.
        """
        # --- ADDED: Reset group expansion state if account scope changes ---
        # This is the definitive fix for the bug where groups remain collapsed after
        if self.is_calculating:
            logging.info("Calculation already in progress. Ignoring refresh request.")
            return

        if not self.db_conn:  # Check DB connection first
            self.show_error(
                "No active database connection. Cannot refresh data.",
                popup=True,
                title="Database Error",
            )
            self.set_status("Error: Database not connected.")
            self.calculation_finished()  # Ensure UI is re-enabled
            return

        self.is_calculating = True
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.set_status(f"Refreshing data from database... ({now_str})")
        self.set_controls_enabled(False)
        if hasattr(self, "progress_bar"):  # Check if progress_bar exists
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

        # Mark data as fresh since we are starting a refresh
        self._mark_data_as_fresh()

        # --- Load data from DB ---
        # df_all_transactions will be the cleaned DataFrame from the DB
        # df_original_for_ignored will also be from the DB, as it's already structured.
        # Ignored indices/reasons from DB load itself will be empty unless load_all_transactions_from_db reports errors.
        # Use account_currency_map and default_currency from config for cleaning after DB load
        acc_map_config_load = self.config.get("account_currency_map", {})
        def_curr_config_load = self.config.get(
            "default_currency",
            config.DEFAULT_CURRENCY if hasattr(config, "DEFAULT_CURRENCY") else "USD",
        )

        df_all_transactions, load_success = load_all_transactions_from_db(
            self.db_conn, acc_map_config_load, def_curr_config_load
        )  # MODIFIED: Pass map and default
        # END MODIFIED

        # For context of ignored rows, if any are generated by portfolio_logic,
        # we use the same DataFrame initially. portfolio_logic itself might receive
        # the raw CSV data if migration was part of the flow leading to the calculation.
        # However, refresh_data now assumes data is IN the DB.
        df_original_for_ignored_context = (
            df_all_transactions.copy()
            if load_success and df_all_transactions is not None
            else pd.DataFrame()
        )

        ignored_indices_load_db = (
            set()
        )  # DB load itself usually doesn't produce these; they come from CSV parse
        ignored_reasons_load_db = {}  # or later processing.

        if not load_success or df_all_transactions is None:
            self.show_error(
                "Failed to load transactions from the database.",
                popup=True,
                title="Load Error",
            )
            self.all_transactions_df_cleaned_for_logic = (
                pd.DataFrame()
            )  # Ensure attribute is empty
            self.original_transactions_df_for_ignored_context = pd.DataFrame()
            self.internal_to_yf_map = {}
            self.calculation_finished()  # Re-enable UI
            return
        elif df_all_transactions.empty:
            logging.info("Database contains no transactions. Nothing to refresh.")
            self.all_transactions_df_cleaned_for_logic = pd.DataFrame()  # Store empty
            self.original_transactions_df_for_ignored_context = pd.DataFrame()
            self.clear_results()  # Clear UI
            self.set_status("Database is empty.")
            self.calculation_finished()  # Re-enable UI
            return
        else:
            # Store the loaded data for portfolio_logic functions
            self.all_transactions_df_cleaned_for_logic = df_all_transactions.copy()
            self.original_transactions_df_for_ignored_context = (
                df_original_for_ignored_context.copy()
            )
            # Ensure self.original_data is also updated with the latest from DB
            self.original_data = df_all_transactions.copy()
            # `original_to_cleaned_header_map` is not relevant for DB source in this context
            self.original_to_cleaned_header_map_from_csv = {}

        # --- MODIFIED: Mark that the first data load has completed ---
        if not self._first_data_load_complete:
            self._first_data_load_complete = True

        # --- Update available accounts and selection based on DB data ---
        # Combine accounts from both 'Account' and 'To Account' columns
        unique_accounts = set()
        has_account_col = "Account" in self.all_transactions_df_cleaned_for_logic.columns
        if has_account_col:
            unique_accounts.update(self.all_transactions_df_cleaned_for_logic["Account"].dropna().unique())
        if "To Account" in self.all_transactions_df_cleaned_for_logic.columns:
            unique_accounts.update(self.all_transactions_df_cleaned_for_logic["To Account"].dropna().unique())
        
        self.available_accounts = sorted(list(unique_accounts))
        
        if not has_account_col:
            logging.warning(
                "No 'Account' column in data loaded from DB. Account filtering may not be available."
            )
        self._update_account_button_text()  # Update button with new list of accounts

        # --- MODIFIED: Reset group expansion state if account scope changes ---
        # This logic now correctly handles the "All Accounts" case by comparing the
        # effective list of accounts, not just the selection list.
        current_scope_effective = self.selected_accounts or self.available_accounts

        if set(current_scope_effective) != set(
            self.last_selected_accounts_for_grouping
        ):
            logging.info("Account scope changed. Clearing group expansion states.")
            self.group_expansion_states.clear()
            # Store the new *effective* scope for the next comparison
            self.last_selected_accounts_for_grouping = current_scope_effective.copy()
        # --- END MODIFIED ---

        # --- Generate internal_to_yf_map based on the effectively filtered transactions ---
        # This needs to be done *after* account filtering for the current scope is considered.
        # For now, let's generate it based on all_transactions_df_cleaned_for_logic,
        # portfolio_logic will internally filter.
        try:
            if not self.all_transactions_df_cleaned_for_logic.empty:
                all_symbols_internal = list(
                    set(self.all_transactions_df_cleaned_for_logic["Symbol"].unique())
                )
                temp_internal_to_yf_map = {}
                for internal_sym in all_symbols_internal:
                    if internal_sym == CASH_SYMBOL_CSV:
                        continue
                    yf_sym = map_to_yf_symbol(
                        internal_sym,
                        self.user_symbol_map_config,
                        self.user_excluded_symbols_config,
                    )
                    if yf_sym:
                        temp_internal_to_yf_map[internal_sym] = yf_sym
                self.internal_to_yf_map = temp_internal_to_yf_map
                logging.info(
                    f"Generated internal_to_yf_map (full dataset): {len(self.internal_to_yf_map)} mappings."
                )
            else:
                self.internal_to_yf_map = {}
        except Exception as e_map_gen:
            logging.error(
                f"Error generating internal_to_yf_map in refresh_data: {e_map_gen}"
            )
            self.internal_to_yf_map = {}

        # --- Prepare for worker ---
        display_currency = self.currency_combo.currentText()
        show_closed = self.show_closed_check.isChecked()

        # Get dates from UI for validation and for line graph display filtering later
        start_date_ui = self.graph_start_date_edit.date().toPython()
        end_date_ui = self.graph_end_date_edit.date().toPython()

        # --- ADDED: Ensure end_date is not in the future ---
        today = date.today()
        if end_date_ui > today:
            end_date_ui = today
            self.graph_end_date_edit.setDate(QDate(end_date_ui))  # Update UI
            logging.info(f"Graph end date was in future, reset to today: {end_date_ui}")
        # --- END ADDED ---

        if start_date_ui >= end_date_ui:
            QMessageBox.warning(
                self, "Invalid Date Range", "Graph start date must be before end date."
            )
            self.calculation_finished()
            return

        # For historical calculation, use the full range of transactions.
        # The UI date pickers will be used later to filter the line graph display.
        if not self.all_transactions_df_cleaned_for_logic.empty:
            # Ensure 'Date' is datetime to find min/max
            if not pd.api.types.is_datetime64_any_dtype(
                self.all_transactions_df_cleaned_for_logic["Date"]
            ):
                self.all_transactions_df_cleaned_for_logic["Date"] = pd.to_datetime(
                    self.all_transactions_df_cleaned_for_logic["Date"]
                )

            start_date_hist_calc = (
                self.all_transactions_df_cleaned_for_logic["Date"].min().date()
            )
            logging.debug(
                f"DEBUG: Full Transaction Min Date is: {start_date_hist_calc}"
            )
            end_date_hist_calc = date.today()  # Always calculate up to today
        else:
            # Fallback if no transactions: use UI dates
            start_date_hist_calc = start_date_ui
            end_date_hist_calc = end_date_ui

        # The interval for the underlying historical data calculation should always be daily
        # to provide the most granular data for all other calculations (like periodic returns).
        # The UI interval combo will be used to resample this data for display.
        interval_hist_calc = "D"

        selected_benchmark_tickers = [
            BENCHMARK_MAPPING.get(name)
            for name in self.selected_benchmarks
            if BENCHMARK_MAPPING.get(name)
        ]
        api_key = self.fmp_api_key

        # Use the validated selected_accounts (or None for all)
        selected_accounts_for_worker = (
            self.selected_accounts if self.selected_accounts else None
        )

        # Account currency map and default currency from config
        acc_map_config = self.config.get("account_currency_map", {})
        def_curr_config = self.config.get(
            "default_currency",
            config.DEFAULT_CURRENCY if hasattr(config, "DEFAULT_CURRENCY") else "USD",
        )
        if not selected_benchmark_tickers:  # Fallback if no valid benchmarks selected
            selected_benchmark_tickers = (
                [BENCHMARK_MAPPING.get(DEFAULT_GRAPH_BENCHMARKS[0], "SPY")]
                if DEFAULT_GRAPH_BENCHMARKS
                else ["SPY"]
            )

        # --- Worker Setup ---
        portfolio_kwargs = {
            "all_transactions_df_cleaned": self.all_transactions_df_cleaned_for_logic.copy(),
            "original_transactions_df_for_ignored": self.original_transactions_df_for_ignored_context.copy(),
            "ignored_indices_from_load": ignored_indices_load_db,  # From DB load (likely empty)
            "ignored_reasons_from_load": ignored_reasons_load_db,  # From DB load (likely empty)
            "display_currency": display_currency,
            "show_closed_positions": show_closed,
            "account_currency_map": acc_map_config,
            "default_currency": def_curr_config,
            "fmp_api_key": api_key,
            "include_accounts": selected_accounts_for_worker,
            "manual_overrides_dict": self.manual_overrides_dict,
            "user_symbol_map": self.user_symbol_map_config,
            "user_excluded_symbols": self.user_excluded_symbols_config,
            "all_transactions_df_for_worker": self.all_transactions_df_cleaned_for_logic.copy(),  # Pass for correlation
        }
        current_cache_dir_base = QStandardPaths.writableLocation(
            QStandardPaths.CacheLocation
        )
        if current_cache_dir_base:
            current_cache_dir = current_cache_dir_base
            os.makedirs(current_cache_dir, exist_ok=True)
            portfolio_kwargs["cache_file_path"] = os.path.join(
                current_cache_dir, "portfolio_cache_yf.json"
            )
        else:
            portfolio_kwargs["cache_file_path"] = "portfolio_cache_yf.json"

        historical_kwargs = {
            "all_transactions_df_cleaned": self.all_transactions_df_cleaned_for_logic.copy(),
            "original_transactions_df_for_ignored": self.original_transactions_df_for_ignored_context.copy(),
            "ignored_indices_from_load": ignored_indices_load_db,
            "ignored_reasons_from_load": ignored_reasons_load_db,
            "start_date": start_date_hist_calc,
            "end_date": end_date_hist_calc,
            "interval": interval_hist_calc,
            "benchmark_symbols_yf": selected_benchmark_tickers,
            "display_currency": display_currency,
            "account_currency_map": acc_map_config,
            "default_currency": def_curr_config,
            "include_accounts": selected_accounts_for_worker,
            "manual_overrides_dict": self.manual_overrides_dict,
            "user_symbol_map": self.user_symbol_map_config,
            "user_excluded_symbols": self.user_excluded_symbols_config,
            "original_csv_file_path": None,  # No single CSV path when data is from DB for cache key hash
        }
        if HISTORICAL_FN_SUPPORTS_EXCLUDE:
            accounts_to_exclude_hist = [
                acc
                for acc in self.available_accounts
                if selected_accounts_for_worker
                and acc not in selected_accounts_for_worker
            ]
            historical_kwargs["exclude_accounts"] = accounts_to_exclude_hist

        logging.info(f"Starting calculation & data fetch (DB source):")
        logging.info(
            f"DB='{os.path.basename(self.DB_FILE_PATH)}', Currency='{display_currency}', ShowClosed={show_closed}, SelectedAccounts={selected_accounts_for_worker if selected_accounts_for_worker else 'All'}"
        )
        logging.info(
            f"Default Currency: {def_curr_config}, Account Map: {acc_map_config}"
        )
        exclude_log_msg = (
            f", ExcludeHist={historical_kwargs.get('exclude_accounts', [])}"
            if HISTORICAL_FN_SUPPORTS_EXCLUDE
            and historical_kwargs.get("exclude_accounts")
            else ""
        )
        logging.info(
            f"Graph Params: Start={start_date_hist_calc}, End={end_date_hist_calc}, Interval={interval_hist_calc}, Benchmarks (Tickers)={selected_benchmark_tickers}{exclude_log_msg}"
        )
        logging.debug(
            f"DEBUG: all_transactions_df_cleaned_for_logic shape before worker init: {self.all_transactions_df_cleaned_for_logic.shape}"
        )

        worker = PortfolioCalculatorWorker(
            portfolio_fn=calculate_portfolio_summary,
            portfolio_args=(),
            portfolio_kwargs=portfolio_kwargs,
            historical_fn=calculate_historical_performance,
            historical_args=(),
            historical_kwargs=historical_kwargs,
            worker_signals=self.worker_signals,  # Use the instance's signals object
            manual_overrides_dict=self.manual_overrides_dict,
            user_symbol_map=self.user_symbol_map_config,
            user_excluded_symbols=self.user_excluded_symbols_config,
            market_data_provider=self.market_data_provider,
            force_historical_refresh=force_historical_refresh,  # Pass the tri-state value
            historical_fn_supports_exclude=HISTORICAL_FN_SUPPORTS_EXCLUDE,
            market_provider_available=MARKET_PROVIDER_AVAILABLE,
            factor_model_name=self.factor_model_combo.currentText(),
            scenario_shocks=self._get_scenario_shocks_from_input(),
        )
        # Signals are already connected in __init__ to self.worker_signals
        self.threadpool.start(worker)

    def _perform_initial_load_from_db_only(self):
        """Loads data from DB on startup if load_on_startup is true and DB is not empty."""
        if self.db_conn:
            # Check if DB has any data before refreshing
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM transactions")
                count_result = cursor.fetchone()
                count = count_result[0] if count_result else 0

                if count > 0:
                    logging.info(
                        "DB has data. Triggering initial data refresh on startup..."
                    )
                    # QTimer.singleShot(150, self.refresh_data) # Already called by __init__ logic path if needed
                    self.refresh_data()  # Call directly if this path is taken
                else:
                    logging.info("DB is empty. No initial data refresh from DB.")
                    self.set_status(
                        "Database is empty. Add transactions or import from CSV via File menu."
                    )
                    # Clear UI elements as no data will be loaded
                    self.clear_results()  # This handles UI clearing
            except sqlite3.Error as e:
                logging.error(f"Error checking database transaction count: {e}")
                QMessageBox.warning(
                    self, "Database Error", f"Could not check database content: {e}"
                )
                self.set_status("Error accessing database. Check logs.")
                self.clear_results()
            except Exception as e_unknown:  # Catch any other unexpected error
                logging.error(
                    f"Unexpected error during initial DB check: {e_unknown}",
                    exc_info=True,
                )
                QMessageBox.critical(
                    self,
                    "Unexpected Error",
                    f"An unexpected error occurred: {e_unknown}",
                )
                self.set_status("Unexpected error. Check logs.")
                self.clear_results()
        else:
            self.set_status("Database not connected. Cannot load data.")
            self.clear_results()

    def _perform_initial_load(self):
        """Performs the initial data load on startup if configured."""
        if self.config.get("load_on_startup", True):
            if self.transactions_file and os.path.exists(self.transactions_file):
                logging.info("Triggering initial data refresh on startup...")

                QTimer.singleShot(150, self.refresh_data)
            elif not self.transactions_file or not os.path.exists(
                self.transactions_file
            ):
                logging.info(
                    f"Startup TX file invalid or not found: '{self.transactions_file}'. Prompting user."
                )
                self.set_status("Info: Please select your transactions CSV file.")
                self._initial_file_selection = True

                QTimer.singleShot(100, self.select_file)
            else:
                startup_file_msg = f"Warn: Startup TX file invalid or not found: '{self.transactions_file}'. Load skipped."
                self.set_status(startup_file_msg)
                logging.info(startup_file_msg)
                self._update_table_view_with_filtered_columns(pd.DataFrame())
                self.apply_column_visibility()
                self.update_performance_graphs(initial=True)
                self._update_account_button_text()
                self._update_table_title()
        else:
            self.set_status("Ready. Select CSV file and click Refresh.")
            self._update_table_view_with_filtered_columns(pd.DataFrame())
            self.apply_column_visibility()
            self.update_performance_graphs(initial=True)
            self._update_account_button_text()
            self._update_table_title()

    def _update_ui_components_after_calculation(self):
        """Updates all relevant UI components after data processing."""
        logging.debug("Entering _update_ui_components_after_calculation...")
        logging.debug("Updating UI elements after receiving results...")
        try:
            logging.debug(
                f"  Ignored data shape: {self.ignored_data.shape if isinstance(self.ignored_data, pd.DataFrame) else 'Not a DF'}"
            )
            if hasattr(self, "view_ignored_button"):
                self.view_ignored_button.setEnabled(not self.ignored_data.empty)

            # --- REFACTOR: Call _get_filtered_data once per required configuration ---
            logging.debug(
                "  Calling _get_filtered_data to generate display DataFrames..."
            )
            df_for_pies = self._get_filtered_data(group_by_sector=False)

            is_grouped = self.group_by_sector_check.isChecked()
            if is_grouped:
                df_for_table = self._get_filtered_data(group_by_sector=True)
            else:
                df_for_table = df_for_pies  # No need to call again if not grouping

            logging.debug(
                f"  _get_filtered_data (for pies) returned DataFrame shape: {df_for_pies.shape}"
            )
            logging.debug(
                f"  _get_filtered_data (for table, grouped={is_grouped}) returned DataFrame shape: {df_for_table.shape}"
            )
            # --- END REFACTOR ---

            # Update UI components
            logging.debug("  Calling _update_table_title...")
            self._update_table_title(df_for_table)  # Pass the generated DataFrame
            logging.debug(
                "  Calling update_dashboard_summary..."
            )  # MODIFIED: Pass df_for_pies
            self.update_dashboard_summary(
                df_for_pies
            )  # Pass filtered data for cash calculation
            # Account pie needs data grouped by account *within the selected scope*
            # We can derive this from the df_for_pies
            logging.debug("  Calling update_account_pie_chart...")
            self.update_account_pie_chart(df_for_pies)
            logging.debug("  Calling update_holdings_pie_chart...")
            self.update_holdings_pie_chart(df_for_pies)  # Uses filtered data
            logging.debug("  Calling _update_table_view_with_filtered_columns...")

            # --- MODIFIED: Restore header state AFTER model is populated ---
            # This should only run on the very first successful data load.
            if not self._first_data_load_complete and self.config.get(
                "holdings_table_header_state"
            ):
                try:
                    header_state_hex = self.config[
                        "holdings_table_header_state"
                    ].encode()
                    self.table_view.horizontalHeader().restoreState(
                        QByteArray.fromHex(header_state_hex)
                    )
                    logging.info(
                        "Restored holdings table header state (column order/sizes)."
                    )
                except Exception as e:
                    logging.warning(
                        f"Could not restore holdings table header state: {e}"
                    )

            self._update_table_view_with_filtered_columns(df_for_table)  # Update table
            logging.debug("  Calling apply_column_visibility...")
            self.apply_column_visibility()  # Re-apply visibility
            self.update_performance_graphs()  # Uses self.historical_data (which reflects scope)
            self.update_header_info()  # Uses self.index_quote_data

            # Update Advanced Analysis tabs
            self._update_correlation_matrix_display()
            self._update_factor_analysis_display()
            self._update_scenario_analysis_display()

            logging.debug(
                f"  Calling _update_fx_rate_display with currency: {self.currency_combo.currentText()}"
            )

            self._update_fx_rate_display(
                self.currency_combo.currentText()
            )  # Uses self.summary_metrics_data
            # --- Make bar charts visible and plot ---
            # --- MODIFY Visibility Check ---
            # --- ADDED: Log state right before check ---
            logging.debug(
                f"[Handle Results] PRE-CHECK: self.periodic_returns_data is type {type(self.periodic_returns_data)}"
            )
            logging.debug(
                f"[Handle Results] PRE-CHECK: bool(self.periodic_returns_data) = {bool(self.periodic_returns_data)}"
            )
            if isinstance(self.periodic_returns_data, dict):
                logging.debug(
                    f"[Handle Results] PRE-CHECK: periodic_returns_data keys: {list(self.periodic_returns_data.keys())}"
                )
                logging.debug(
                    f"[Handle Results] PRE-CHECK: any(not df.empty...) = {any(not df.empty for df in self.periodic_returns_data.values() if isinstance(df, pd.DataFrame))}"
                )
            # Check if the dictionary itself is non-empty AND if it contains data for *any* interval
            bar_charts_have_data = bool(self.periodic_returns_data) and any(
                not df.empty
                for df in self.periodic_returns_data.values()
                if isinstance(df, pd.DataFrame)
            )
            logging.debug(
                f"[Handle Results] Bar charts visibility check: bar_charts_have_data = {bar_charts_have_data}"
            )  # ADDED LOG
            self.bar_charts_frame.setVisible(bar_charts_have_data)
            if bar_charts_have_data:  # Only call plot if there's data
                logging.debug("  Calling _update_periodic_bar_charts...")
                self._update_periodic_bar_charts()
                logging.debug("  Calling _update_dividend_bar_chart...")
                self._update_dividend_bar_chart()  # Also update dividend chart
                # _update_dividend_summary_table will be called by _update_dividend_bar_chart
                logging.debug("  Calling _update_dividend_table...")
                self._update_dividend_table()  # And dividend table
                self._update_all_transaction_tables()  # Update all three transaction tables
                logging.debug("  Calling _update_asset_allocation_charts...")
                self._update_asset_allocation_charts()  # Update new allocation charts
                logging.debug("  Calling _update_capital_gains_display...")
                self._update_capital_gains_display()  # Update Capital Gains tab

                self._update_periodic_value_change_display()  # Update new tab

            else:
                logging.info(
                    "Hiding bar charts frame as no periodic data is available."
                )

        except Exception as ui_update_e:
            logging.error(
                f"--- CRITICAL ERROR during UI update in handle_results: {ui_update_e} ---"
            )
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "UI Update Error",
                f"Failed to update display after calculation:\n{ui_update_e}",
            )

        logging.debug("Exiting handle_results.")

    @Slot(
        str, str
    )  # Ensure Slot decorator is imported: from PySide6.QtCore import Slot
    def _view_transactions_for_holding(self, symbol: str, account: str):
        """Placeholder/Handler for 'View Transactions' context menu action."""
        logging.info(f"Action triggered: View Transactions for {symbol} in {account}")

        if not hasattr(self, "original_data") or self.original_data.empty:
            QMessageBox.warning(
                self, "No Data", "Original transaction data not loaded."
            )
            return

        try:
            # Filter original data (using original CSV headers)
            symbol_to_filter = CASH_SYMBOL_CSV if is_cash_symbol(symbol) else symbol

            filtered_df = self.original_data[
                (
                    self.original_data["Symbol"] == symbol_to_filter
                )  # Use cleaned column name
                & (self.original_data["Account"] == account)  # Use cleaned column name
            ].copy()

            if filtered_df.empty:
                QMessageBox.information(
                    self,
                    "No Transactions",
                    f"No transactions found for {symbol} in account {account}.",
                )
            else:
                # Reuse LogViewerDialog
                dialog = LogViewerDialog(filtered_df, self)
                dialog.setWindowTitle(f"Transactions for: {symbol} / {account}")
                dialog.exec()

        except KeyError as e:
            logging.error(f"Missing column in original_data for filtering: {e}")
            QMessageBox.critical(
                self,
                "Error",
                "Could not filter transactions due to missing data columns.",
            )
        except Exception as e:
            logging.exception(
                f"Error filtering/displaying transactions for {symbol}/{account}"
            )
            QMessageBox.critical(
                self, "Error", f"An error occurred while viewing transactions:\n{e}"
            )

    @Slot(QPoint)
    def show_table_context_menu(self, pos: QPoint):
        """Shows a context menu for the row right-clicked in the holdings table."""
        # --- MODIFIED: Use sender() to determine which table was clicked ---
        sender_view = self.sender()
        if not isinstance(sender_view, QTableView):
            return

        # Get the model index corresponding to the click position within the table view
        index = sender_view.indexAt(pos)

        if not index.isValid():
            logging.debug("Right-click outside valid table rows.")
            return  # Click wasn't on a valid item row

        # Get the row number in the VIEW
        view_row = index.row()

        # Get Symbol and Account from the model for the clicked row
        symbol = None
        account = None
        try:
            # Find column indices in the current model
            symbol_col_idx = self.table_model._data.columns.get_loc("Symbol")
            account_col_idx = self.table_model._data.columns.get_loc("Account")

            # Get model indices for symbol and account in the clicked row
            symbol_model_idx = self.table_model.index(view_row, symbol_col_idx)
            account_model_idx = self.table_model.index(view_row, account_col_idx)

            # Retrieve data using the model's data() method
            symbol = self.table_model.data(symbol_model_idx, Qt.DisplayRole)
            account = self.table_model.data(account_model_idx, Qt.DisplayRole)

            # Handle potential "Cash (CUR)" display name
            if isinstance(symbol, str) and symbol.startswith("Cash ("):
                symbol = CASH_SYMBOL_CSV  # Use internal symbol

        except (KeyError, IndexError, AttributeError) as e:
            logging.error(
                f"Error retrieving symbol/account from table model for context menu: {e}"
            )
            # Optionally show a generic menu or nothing? Let's show nothing on error.
            return

        if symbol is None or account is None:
            logging.warning("Could not determine symbol or account for context menu.")
            return

        # Create the context menu
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())  # Apply style

        # --- Define Actions ---
        # Action 1: View Transactions
        view_tx_action = QAction(f"View Transactions for {symbol}/{account}", self)
        # Disable if original data isn't available
        view_tx_action.setEnabled(
            hasattr(self, "original_data") and not self.original_data.empty
        )
        # Use lambda to pass current symbol/account to the slot
        view_tx_action.triggered.connect(
            lambda checked=False, s=symbol, a=account: self._view_transactions_for_holding(
                s, a
            )
        )
        menu.addAction(view_tx_action)

        # Separator
        menu.addSeparator()

        # Action 2: Chart History (Placeholder)
        # Disable for Cash symbol
        chart_action = QAction(f"Chart History for {symbol}", self)
        chart_action.setEnabled(not is_cash_symbol(symbol))  # Disable for cash
        chart_action.triggered.connect(
            lambda checked=False, s=symbol: self._chart_holding_history(s)
        )
        menu.addAction(chart_action)

        # --- Show Menu ---
        # Map the click position within the table's viewport to global screen coordinates
        global_pos = sender_view.viewport().mapToGlobal(pos)

        # --- ADD Fundamental Data Action to Context Menu ---
        if not is_cash_symbol(symbol):  # Only for actual stocks/ETFs
            menu.addSeparator()
            fundamentals_action = QAction(f"View Fundamentals for {symbol}", self)
            fundamentals_action.triggered.connect(
                lambda checked=False, s=symbol: self._handle_context_menu_fundamental_lookup(
                    s
                )
            )
            menu.addAction(fundamentals_action)
        # --- END ADD ---

        menu.exec(global_pos)

    def _connect_signals(self):
        """Connects signals from UI widgets (buttons, combos, etc.) to their slots."""
        self.account_select_button.clicked.connect(self.show_account_selection_menu)
        self.refresh_action.triggered.connect(
            lambda: self.refresh_data(force_historical_refresh=True)
        )
        self.update_accounts_button.clicked.connect(self._on_update_accounts_clicked)
        self.currency_combo.currentTextChanged.connect(self.filter_changed_refresh)
        self.show_closed_check.stateChanged.connect(self.filter_changed_refresh)
        # --- MODIFIED: Decouple grouping from full refresh ---
        # Grouping by sector is a UI-only view change and should not trigger a full
        # data recalculation. It should only re-process the currently loaded holdings data
        # for display in the table. This fixes a bug where the historical graph would
        # change when grouping was toggled.
        self.group_by_sector_check.stateChanged.connect(self._update_table_display)
        self.graph_start_date_edit.dateChanged.connect(
            lambda: self.set_status("Graph dates changed. Click 'Update Graphs'.")
        )
        self.graph_end_date_edit.dateChanged.connect(
            lambda: self.set_status("Graph dates changed. Click 'Update Graphs'.")
        )
        self.benchmark_select_button.clicked.connect(self.show_benchmark_selection_menu)
        # MODIFIED: "Update Graphs" no longer forces a historical refresh. It lets the worker decide.
        # The main "Refresh" action (F5/toolbar) is now the primary way to force a full re-fetch.
        self.graph_update_button.clicked.connect(lambda: self.refresh_data())
        self.refresh_button.clicked.connect(
            lambda: self.refresh_data(force_historical_refresh=True)
        )
        self.table_view.horizontalHeader().customContextMenuRequested.connect(
            self.show_header_context_menu
        )
        # --- ADDED: Connect frozen table header context menu ---
        if self.frozen_table_view:
            self.frozen_table_view.horizontalHeader().customContextMenuRequested.connect(
                self.show_header_context_menu
            )
        # --- END ADDED ---

        # Table Filter Connections
        # self.apply_table_filter_button.clicked.connect(self._apply_table_filter)
        self.clear_table_filter_button.clicked.connect(self._clear_table_filter)
        # --- ADDED: Connect the button from the Transactions Log tab ---
        # This button was previously connected in _init_transactions_management_widgets
        # It is now correctly connected here with all other signals.
        # self.add_transaction_button.clicked.connect(self.open_add_transaction_dialog)
        # --- END ADDED ---
        logging.debug(
            "Connected add_transaction_button.clicked signal to open_add_transaction_dialog."
        )
        # Removed some signals to find if the button only triggers once
        # Other signals are still in the code, just commented out for now.

        # Optional: Trigger apply on pressing Enter in the line edits
        self.filter_symbol_table_edit.returnPressed.connect(self._apply_table_filter)
        self.filter_account_table_edit.returnPressed.connect(self._apply_table_filter)
        # Optional: Trigger clear if the clear button (X) inside QLineEdit is clicked
        self.filter_symbol_table_edit.textChanged.connect(
            self._on_table_filter_text_changed
        )
        self.filter_account_table_edit.textChanged.connect(
            self._on_table_filter_text_changed
        )
        self.table_filter_timer.timeout.connect(
            self._apply_table_filter
        )  # Timer timeout applies filter

        # --- Transactions Management Filter Connections ---
        self.apply_filter_button.clicked.connect(
            self._apply_filter_to_transactions_view
        )
        self.clear_filter_button.clicked.connect(
            self._clear_filter_in_transactions_view
        )
        self.filter_symbol_edit.textChanged.connect(self._on_tx_filter_text_changed)
        self.filter_account_edit.textChanged.connect(self._on_tx_filter_text_changed)
        self.tx_filter_timer.timeout.connect(self._apply_filter_to_transactions_view)
        self.filter_symbol_edit.returnPressed.connect(
            self._apply_filter_to_transactions_view
        )
        self.filter_account_edit.returnPressed.connect(
            self._apply_filter_to_transactions_view
        )

        # --- Transactions Management Tab Connections ---
        self.manage_tab_add_button.clicked.connect(self.add_new_transaction_db)
        self.manage_tab_edit_button.clicked.connect(self.edit_selected_transaction_db)
        self.manage_tab_delete_button.clicked.connect(
            self.delete_selected_transaction_db
        )
        self.manage_tab_export_button.clicked.connect(self.export_transactions_to_csv)
        # --- End Transactions Management Tab Connections ---

        # Fundamental Lookup Connections
        self.lookup_button.clicked.connect(self._handle_direct_symbol_lookup)
        self.lookup_symbol_edit.returnPressed.connect(self._handle_direct_symbol_lookup)
        self.worker_signals.fundamental_data_ready.connect(
            self._show_fundamental_data_dialog_from_worker
        )
        # --- ADDED: Connect main worker signals ---
        self.worker_signals.result.connect(self.handle_results)
        self.worker_signals.error.connect(self.handle_error)
        self.worker_signals.finished.connect(self.calculation_finished)
        self.worker_signals.progress.connect(self.update_progress)

        # Table Context Menu Connection
        self.table_view.customContextMenuRequested.connect(
            self.show_table_context_menu
        )  # Connect table's signal
        # --- ADDED: Connect frozen table context menu ---
        if self.frozen_table_view:
            self.frozen_table_view.customContextMenuRequested.connect(
                self.show_table_context_menu
            )
        # --- END ADDED ---
        self.table_view.clicked.connect(
            self.on_table_view_clicked
        )  # For collapse/expand
        # --- ADDED: Connect frozen table click for expand/collapse ---
        if self.frozen_table_view:
            self.frozen_table_view.clicked.connect(self.on_table_view_clicked)
        # --- END ADDED ---

        # Performance Graph Context Menus
        self.perf_return_canvas.customContextMenuRequested.connect(
            lambda pos: self._show_graph_context_menu(
                pos, self.perf_return_fig, "TWR_Graph"
            )
        )
        self.abs_value_canvas.customContextMenuRequested.connect(
            lambda pos: self._show_graph_context_menu(
                pos, self.abs_value_fig, "Value_Graph"
            )
        )

        # Bar Chart Period Spinbox Connections
        self.pvc_annual_spinbox.valueChanged.connect(self._update_periodic_bar_charts)
        self.pvc_annual_spinbox.valueChanged.connect(
            self._update_periodic_value_change_display
        )
        self.pvc_monthly_spinbox.valueChanged.connect(self._update_periodic_bar_charts)
        self.pvc_monthly_spinbox.valueChanged.connect(
            self._update_periodic_value_change_display
        )
        self.pvc_weekly_spinbox.valueChanged.connect(self._update_periodic_bar_charts)
        self.pvc_weekly_spinbox.valueChanged.connect(
            self._update_periodic_value_change_display
        )
        self.pvc_daily_spinbox.valueChanged.connect(self._update_periodic_bar_charts)
        self.pvc_daily_spinbox.valueChanged.connect(
            self._update_periodic_value_change_display
        )

        # --- ADDED: Connect tab change signal for PVC tab default sort ---
        if hasattr(self, "main_tab_widget") and self.main_tab_widget:
            self.main_tab_widget.currentChanged.connect(
                self._handle_pvc_tab_visibility_change
            )
        # --- END ADDED ---

        # Rebalancing Tab Connections
        self.load_current_holdings_button.clicked.connect(
            self._load_current_holdings_to_target_table
        )
        self.calculate_rebalance_button.clicked.connect(
            self._handle_rebalance_calculation
        )
        self.target_allocation_table.cellChanged.connect(
            self._update_target_total_label
        )
        self.cash_to_add_line_edit.textChanged.connect(
            lambda: self._update_target_total_label()
        )

        # Intraday Chart Connections
        self.intraday_update_button.clicked.connect(self._update_intraday_chart)
        self.intraday_period_combo.currentTextChanged.connect(
            self._update_intraday_interval_options
        )

    def _update_table_display(self):
        """Updates the table view, pie charts, and title based on current filters."""
        logging.debug("Updating table display due to filter change...")
        # 1. Get data filtered by Account, Show Closed, AND Table Filters
        df_display_filtered = self._get_filtered_data(
            group_by_sector=self.group_by_sector_check.isChecked()
        )  # This now includes table filters

        # 2. Update the table view itself
        self._update_table_view_with_filtered_columns(df_display_filtered)
        self.apply_column_visibility()  # Re-apply visibility

        # 3. Update the holdings pie chart.
        #    It should always be based on ungrouped data.
        #    If the data for the table was grouped, filter out the group headers.
        df_for_pie = df_display_filtered
        if "is_group_header" in df_for_pie.columns:
            df_for_pie = df_for_pie[df_for_pie["is_group_header"] != True].copy()

        self.update_holdings_pie_chart(df_for_pie)

        # 4. Update the table title to reflect the number of items *shown*.
        self._update_table_title(df_display_filtered)

    @Slot()
    def _apply_table_filter(self):
        """Applies the current text filters to the table view."""
        self._update_table_display()

    @Slot()
    def _clear_table_filter(self):  # Slot for "Clear" button
        """Clears the text filters. The table view will update via textChanged signal."""
        self.filter_symbol_table_edit.clear()
        self.filter_account_table_edit.clear()
        # Clearing the text will trigger _on_table_filter_text_changed, which starts the timer.

    @Slot(str)  # Slot for textChanged signals from filter QLineEdits
    def _on_table_filter_text_changed(self, text: str):
        """Restarts the debounce timer when filter text changes."""
        logging.debug(
            f"Filter text changed: '{text}', restarting debounce timer ({DEBOUNCE_INTERVAL_MS}ms)."
        )
        self.table_filter_timer.start(DEBOUNCE_INTERVAL_MS)

    @Slot(str)
    def _on_tx_filter_text_changed(self, text: str):
        """Restarts the debounce timer for the transactions tab filter."""
        logging.debug(
            f"Transaction filter text changed: '{text}', restarting debounce timer ({DEBOUNCE_INTERVAL_MS}ms)."
        )
        self.tx_filter_timer.start(DEBOUNCE_INTERVAL_MS)

    # --- Styling Method (Reads External File) ---
    def apply_styles(self):
        """DEPRECATED: This method is now largely handled by apply_theme and QSS.
        Kept for potential direct Matplotlib rcParams if needed, but QSS is preferred.
        The rcParams update is now within apply_theme.
        """
        logging.info(
            "apply_styles() called. Functionality moved to apply_theme() and QSS."
        )
        # The core stylesheet application is now in apply_theme.
        # Matplotlib rcParams are also updated in apply_theme.
        # This method can be kept if there are any specific non-QSS, non-rcParams
        # styling adjustments needed, or removed if fully superseded.

        # Example of how chart backgrounds might be set if not using QSS for FigureCanvas:
        # try:
        #     if hasattr(self, 'account_fig'): # and other chart figures
        #         bg_color = self.QCOLOR_BACKGROUND_THEMED.name()
        #         header_bg_color = self.QCOLOR_HEADER_BACKGROUND_THEMED.name()
        #
        #         for fig_attr, ax_attr, color_to_use in [
        #             ('account_fig', 'account_ax', bg_color),
        #             ('holdings_fig', 'holdings_ax', bg_color),
        #             ('asset_type_pie_fig', 'asset_type_pie_ax', bg_color),
        #             ('sector_pie_fig', 'sector_pie_ax', bg_color),
        #             ('geo_pie_fig', 'geo_pie_ax', bg_color),
        #             ('industry_pie_fig', 'industry_pie_ax', bg_color),
        #             ('perf_return_fig', 'perf_return_ax', header_bg_color), # Typically match container
        #             ('abs_value_fig', 'abs_value_ax', header_bg_color),     # Typically match container
        #             ('annual_bar_fig', 'annual_bar_ax', bg_color),
        #             ('monthly_bar_fig', 'monthly_bar_ax', bg_color),
        #             ('weekly_bar_fig', 'weekly_bar_ax', bg_color),
        #             ('dividend_bar_fig', 'dividend_bar_ax', bg_color),
        #             ('cg_bar_fig', 'cg_bar_ax', bg_color),
        #         ]:
        #             if hasattr(self, fig_attr) and getattr(self, fig_attr):
        #                 getattr(self, fig_attr).patch.set_facecolor(color_to_use)
        #             if hasattr(self, ax_attr) and getattr(self, ax_attr):
        #                 getattr(self, ax_attr).patch.set_facecolor(color_to_use)
        #         logging.debug("Chart figure/axes backgrounds updated based on theme.")
        # except Exception as e_chart_bg:
        #     logging.warning(f"Warning: Failed to update chart backgrounds in apply_styles: {e_chart_bg}")

    # --- Chart Update Methods ---
    def _adjust_pie_labels(self, label_positions, vertical_threshold=0.15):
        """
        Adjusts pie chart label positions vertically to minimize overlap.

        Internal helper for pie chart drawing.

        Args:
            label_positions (List[Dict]): List of dictionaries, each containing label
                                          info ('x', 'y', 'label', 'index', etc.).
            vertical_threshold (float, optional): Minimum vertical distance between labels.
                                                  Defaults to 0.15.

        Returns:
            List[Dict]: The list of label position dictionaries with adjusted 'y_nudge'.
        """  # ... (implementation as provided before) ...
        if not label_positions:
            return label_positions

        def calculate_vertical_nudges(sorted_labels, pos_map_ref):
            if len(sorted_labels) < 2:
                return
            for i in range(len(sorted_labels) - 1):
                curr_label_info = sorted_labels[i]
                next_label_info = sorted_labels[i + 1]
                curr_y = (
                    curr_label_info["final_y"]
                    + pos_map_ref[curr_label_info["index"]]["y_nudge"]
                )
                next_y = (
                    next_label_info["final_y"]
                    + pos_map_ref[next_label_info["index"]]["y_nudge"]
                )
                if abs(next_y - curr_y) < vertical_threshold:
                    overlap = vertical_threshold - abs(next_y - curr_y)
                    shift_amount = overlap / 2.0
                    pos_map_ref[next_label_info["index"]]["y_nudge"] += shift_amount
                    pos_map_ref[curr_label_info["index"]]["y_nudge"] -= shift_amount

        left_initial = sorted(
            [p for p in label_positions if p["original_side"] == "left"],
            key=lambda item: item["y"],
        )
        right_initial = sorted(
            [p for p in label_positions if p["original_side"] == "right"],
            key=lambda item: item["y"],
        )

        def check_initial_side_for_flip(sorted_labels):
            flip_indices = set()
            for i in range(1, len(sorted_labels)):
                if (
                    abs(sorted_labels[i]["y"] - sorted_labels[i - 1]["y"])
                    < vertical_threshold
                ):
                    flip_indices.add(sorted_labels[i]["index"])
            return flip_indices

        all_flip_indices = check_initial_side_for_flip(left_initial).union(
            check_initial_side_for_flip(right_initial)
        )
        for pos in label_positions:
            pos["final_x"] = pos["x"]
            pos["final_y"] = pos["y"]
            pos["y_nudge"] = 0.0
            if pos["index"] in all_flip_indices:
                pos["final_x"] = -pos["x"]
        final_left = sorted(
            [p for p in label_positions if p["final_x"] < 0],
            key=lambda item: item["final_y"],
        )
        final_right = sorted(
            [p for p in label_positions if p["final_x"] >= 0],
            key=lambda item: item["final_y"],
        )
        pos_map = {p["index"]: p for p in label_positions}
        for _ in range(4):  # Multiple nudge passes
            calculate_vertical_nudges(final_left, pos_map)
            calculate_vertical_nudges(final_right, pos_map)
            final_left = sorted(
                final_left, key=lambda p: p["final_y"] + pos_map[p["index"]]["y_nudge"]
            )
            final_right = sorted(
                final_right, key=lambda p: p["final_y"] + pos_map[p["index"]]["y_nudge"]
            )
        return label_positions

    # Add this method to the PortfolioApp class
    @Slot()
    def show_ignored_log(self):
        """Shows the dialog displaying ignored transactions."""
        if hasattr(self, "ignored_data") and isinstance(
            self.ignored_data, pd.DataFrame
        ):
            # Make sure the Reason Ignored column exists if possible
            df_to_show = self.ignored_data.copy()
            if "Reason Ignored" not in df_to_show.columns:
                # If the main ignored_data doesn't have it (e.g., from older logic), add a placeholder
                if not df_to_show.empty:
                    df_to_show["Reason Ignored"] = "Reason not captured"
                else:
                    # If df is empty, create the column anyway for the dialog
                    df_to_show = pd.DataFrame(columns=["Reason Ignored"])

            dialog = LogViewerDialog(df_to_show, self)
            dialog.exec()  # Show as a modal dialog
        else:
            QMessageBox.information(
                self, "Info", "No ignored transaction data available."
            )

    # --- ADDED: Risk Metrics Dialog ---
    @Slot()
    def show_risk_metrics_dialog(self):
        """Calculates and displays risk metrics in a dialog."""
        if not hasattr(self, "full_historical_data") or self.full_historical_data.empty:
            QMessageBox.warning(
                self,
                "Data Unavailable",
                "No historical data available. Please ensure data is loaded and historical performance is calculated.",
            )
            return

        if "Portfolio Value" not in self.full_historical_data.columns:
            QMessageBox.warning(
                self,
                "Data Error",
                "Historical data does not contain 'Portfolio Value' column.",
            )
            return

        try:
            # Extract portfolio value series
            portfolio_values = self.full_historical_data["Portfolio Value"].sort_index()
            
            # Calculate metrics
            metrics = calculate_all_risk_metrics(portfolio_values)
            
            if not metrics:
                QMessageBox.information(self, "Risk Metrics", "Could not calculate metrics (insufficient data?).")
                return

            # Format for display
            msg = "<h3>Portfolio Risk Metrics</h3>"
            msg += "<table border='0' cellpadding='5'>"
            
            # Max Drawdown
            mdd = metrics.get("Max Drawdown", 0.0)
            msg += f"<tr><td><b>Max Drawdown:</b></td><td><font color='red'>{mdd:.2%}</font></td></tr>"
            
            # Volatility
            vol = metrics.get("Volatility (Ann.)", 0.0)
            msg += f"<tr><td><b>Volatility (Ann.):</b></td><td>{vol:.2%}</td></tr>"
            
            # Sharpe Ratio
            sharpe = metrics.get("Sharpe Ratio", 0.0)
            sharpe_color = "green" if sharpe > 1 else "black"
            msg += f"<tr><td><b>Sharpe Ratio:</b></td><td><font color='{sharpe_color}'>{sharpe:.2f}</font></td></tr>"
            
            # Sortino Ratio
            sortino = metrics.get("Sortino Ratio", 0.0)
            sortino_color = "green" if sortino > 1 else "black"
            msg += f"<tr><td><b>Sortino Ratio:</b></td><td><font color='{sortino_color}'>{sortino:.2f}</font></td></tr>"
            
            msg += "</table>"
            msg += "<br><i>Note: Metrics are based on daily returns from the loaded historical period. Risk-free rate assumed 2%.</i>"

            QMessageBox.information(self, "Risk Metrics", msg)

        except Exception as e:
            logging.error(f"Error calculating/displaying risk metrics: {e}", exc_info=True)
            QMessageBox.critical(self, "Error", f"An error occurred while calculating metrics:\n{e}")


    def _update_drawdown_chart(self, drawdown_series: pd.Series):
        """Updates the drawdown chart with the provided drawdown series."""
        if not hasattr(self, "drawdown_ax"):
            return

        self.drawdown_ax.clear()
        
        # Set background color
        if hasattr(self, "QCOLOR_BACKGROUND_THEMED"):
             self.drawdown_ax.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
             self.drawdown_fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())

        if drawdown_series.empty:
            self.drawdown_canvas.draw()
            return

        try:
            # Plot
            dates = drawdown_series.index
            values = drawdown_series.values
            
            self.drawdown_ax.fill_between(dates, values, 0, color='red', alpha=0.3)
            self.drawdown_line, = self.drawdown_ax.plot(dates, values, color='red', linewidth=1)
            
            # --- Formatting to match Value Graphs ---
            # Spines
            self.drawdown_ax.spines["top"].set_visible(False)
            self.drawdown_ax.spines["right"].set_visible(False)
            if hasattr(self, "QCOLOR_BORDER_THEMED"):
                self.drawdown_ax.spines["bottom"].set_color(self.QCOLOR_BORDER_THEMED.name())
                self.drawdown_ax.spines["left"].set_color(self.QCOLOR_BORDER_THEMED.name())
            
            # Ticks and Labels
            if hasattr(self, "QCOLOR_TEXT_SECONDARY_THEMED"):
                self.drawdown_ax.tick_params(
                    axis="x", colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name(), labelsize=8
                )
                self.drawdown_ax.tick_params(
                    axis="y", colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name(), labelsize=8
                )
            
            if hasattr(self, "QCOLOR_TEXT_PRIMARY_THEMED"):
                self.drawdown_ax.xaxis.label.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name())
                self.drawdown_ax.yaxis.label.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name())
                # self.drawdown_ax.title.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name()) # Title removed
                
            # Grid
            grid_color = "gray"
            if hasattr(self, "QCOLOR_BORDER_THEMED"):
                 grid_color = self.QCOLOR_BORDER_THEMED.name()
                 
            self.drawdown_ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color=grid_color,
                alpha=1.0 # Use alpha 1.0 because color handles transparency/lightness
            )
            
            # Format Y-axis as percentage
            import matplotlib.ticker as mtick
            self.drawdown_ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
            
            # Rotate date labels
            self.drawdown_fig.autofmt_xdate()

            # --- Initialize Annotation ---
            self.drawdown_annot = self.drawdown_ax.annotate(
                "",
                xy=(0, 0),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=dict(boxstyle="round", fc="white", alpha=0.8),
                arrowprops=dict(arrowstyle="->"),
            )
            self.drawdown_annot.set_visible(False)
            
            self.drawdown_canvas.draw()
            
        except Exception as e:
            logging.error(f"Error updating drawdown chart: {e}")

    def _on_drawdown_hover(self, event):
        """Handles mouse hover events on the drawdown chart."""
        # logging.debug(f"Hover event: inaxes={event.inaxes}, x={event.xdata}, y={event.ydata}")
        if event.inaxes != self.drawdown_ax:
            if hasattr(self, "drawdown_annot") and self.drawdown_annot and self.drawdown_annot.get_visible():
                self.drawdown_annot.set_visible(False)
                self.drawdown_canvas.draw_idle()
            return

        if not hasattr(self, "drawdown_line"):
            return

        # Find closest data point
        try:
            # Convert event x (date) to numerical format used by matplotlib
            x_mouse = event.xdata
            if x_mouse is None: return
            
            # Get line data
            x_data = self.drawdown_line.get_xdata()
            y_data = self.drawdown_line.get_ydata()
            
            # Find index of nearest date
            import matplotlib.dates as mdates
            
            # Check type of x_data to handle both numeric and datetime arrays
            if np.issubdtype(x_data.dtype, np.datetime64):
                # x_data is datetime64, convert x_mouse (float) to datetime64
                dt_mouse = mdates.num2date(x_mouse)
                # Remove timezone info if present to match typical numpy datetime64[ns]
                if dt_mouse.tzinfo is not None:
                    dt_mouse = dt_mouse.replace(tzinfo=None)
                x_mouse_val = np.datetime64(dt_mouse)
            else:
                # x_data is likely float/numeric
                x_mouse_val = x_mouse

            # Find closest index
            idx = (np.abs(x_data - x_mouse_val)).argmin()
            
            x_val = x_data[idx]
            y_val = y_data[idx]
            
            # Update annotation
            # Note: annotate xy expects the data coordinates. 
            # If x_val is datetime64, matplotlib's plot usually handles it, 
            # but for xy we might need the float representation if the axis is date-based?
            # Actually, if we pass the same type as the data, it should work.
            self.drawdown_annot.xy = (x_val, y_val)
            
            # Format date and value
            # If x_val is datetime, format it directly
            if isinstance(x_val, (np.datetime64, pd.Timestamp, datetime)):
                # Convert to python datetime for strftime
                # Use pd.to_datetime for robust conversion handling numpy/pandas types
                date_obj = pd.to_datetime(x_val).to_pydatetime()
                date_str = date_obj.strftime("%Y-%m-%d")
            else:
                # It's a float, convert using mdates
                date_str = mdates.num2date(x_val).strftime("%Y-%m-%d")

            text = f"{date_str}\nDrawdown: {y_val:.2%}"
            
            self.drawdown_annot.set_text(text)
            self.drawdown_annot.set_visible(True)
            self.drawdown_canvas.draw_idle()
            
        except Exception as e:
            logging.error(f"Error in hover: {e}")
            # Suppress errors during hover to avoid spamming logs
            pass

    def update_account_pie_chart(self, df_account_data=None):
        """Updates the 'Value by Account' pie chart.

        Uses the provided DataFrame (filtered for selected accounts) or falls back
        to `self.holdings_data`. Groups data by account and plots market values.

        Args:
            df_account_data (pd.DataFrame, optional): The DataFrame containing data
                filtered for the accounts to be displayed in this pie chart. If None,
                uses `self.holdings_data` filtered by `self.selected_accounts`.
                Defaults to None.
        """
        self.account_ax.clear()
        self.account_ax.axis("off")  # Turn off by default

        # Explicitly set background colors based on current theme
        fig = self.account_fig
        if fig:
            fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
        self.account_ax.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())

        # Use passed data if available, otherwise fallback to self.holdings_data
        df_to_use = (
            df_account_data
            if isinstance(df_account_data, pd.DataFrame)
            else self.holdings_data
        )

        # Check if data is valid BEFORE accessing columns
        if not isinstance(df_to_use, pd.DataFrame) or df_to_use.empty:
            self.account_canvas.draw()
            return

        # Proceed with column checks and plotting logic...
        display_currency = self.currency_combo.currentText()
        col_defs = get_column_definitions(display_currency)
        value_col_ui = "Mkt Val"
        value_col_actual = col_defs.get(value_col_ui)

        if (
            not value_col_actual
            or value_col_actual not in df_to_use.columns
            or "Account" not in df_to_use.columns
        ):
            self.account_canvas.draw()
            return

        account_values = df_to_use.groupby("Account")[value_col_actual].sum()
        # --- FIX: Filter for positive values for pie chart ---
        account_values = account_values[account_values > 1e-3]

        if account_values.empty:
            self.account_canvas.draw()
            return
        else:
            # --- Turn axis back on only if we have data to plot ---
            self.account_ax.axis("on")  # Turn axis back on for plotting
            # self.account_ax.axis("equal")  # REMOVED - Pie chart handles aspect ratio

            account_values = account_values.sort_values(ascending=False)
            labels = account_values.index.tolist()
            values = account_values.values

            if len(values) > CHART_MAX_SLICES:
                top_v = values[: CHART_MAX_SLICES - 1]
                top_l = labels[: CHART_MAX_SLICES - 1]
                other_v = values[CHART_MAX_SLICES - 1 :].sum()
                values = np.append(top_v, other_v)
                labels = top_l + ["Other"]

            cmap = plt.get_cmap("Spectral")
            colors = cmap(np.linspace(0, 1, len(values)))
            pie_radius = 0.9
            label_offset_multiplier = 1.1
            VERTICAL_THRESHOLD = 0.15  # Adjust as needed

            wedges, _ = self.account_ax.pie(
                values,
                labels=None,
                autopct=None,
                startangle=90,
                counterclock=False,
                colors=colors,
                radius=pie_radius,
                wedgeprops={"edgecolor": "white", "linewidth": 0.5},
            )

            label_positions = []
            total_value = np.sum(values)
            for i, p in enumerate(wedges):
                ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
                y_edge = pie_radius * np.sin(np.deg2rad(ang))
                x_edge = pie_radius * np.cos(np.deg2rad(ang))
                y_text = label_offset_multiplier * np.sin(np.deg2rad(ang))
                x_text = label_offset_multiplier * np.cos(np.deg2rad(ang))
                original_side = "right" if x_text >= 0 else "left"
                percent = (values[i] / total_value) * 100 if total_value > 1e-9 else 0
                label_text = f"{labels[i]} ({percent:.0f}%)"
                label_positions.append(
                    {
                        "x": x_text,
                        "y": y_text,
                        "label": label_text,
                        "index": i,
                        "original_side": original_side,
                        "ang": ang,
                        "x_edge": x_edge,
                        "y_edge": y_edge,
                    }
                )

            label_positions = self._adjust_pie_labels(
                label_positions, VERTICAL_THRESHOLD
            )

            arrowprops = dict(
                arrowstyle="-",
                color=COLOR_TEXT_SECONDARY,
                shrinkA=0,
                shrinkB=0,
                relpos=(0.5, 0.5),
            )
            kw = dict(arrowprops=arrowprops, zorder=0, va="center")

            for pos in label_positions:
                x_t = pos["final_x"]
                y_t = pos["final_y"] + pos.get("y_nudge", 0.0)
                x_e, y_e = pos["x_edge"], pos["y_edge"]
                lbl = pos["label"]
                ang = pos["ang"]
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_t))]
                connectionstyle = f"arc,angleA={180 if x_t<0 else 0},angleB={ang},armA=0,armB=40,rad=0"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                self.account_ax.annotate(
                    lbl,
                    xy=(x_e, y_e),
                    xytext=(x_t, y_t),
                    horizontalalignment=horizontalalignment,
                    **kw,
                    fontsize=8,
                    color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                )

        # Adjust subplot parameters to make space for labels
        # Increase left/right margins, decrease top/bottom slightly if needed
        self.account_fig.subplots_adjust(left=0.15, right=0.85, top=0.92, bottom=0.05)
        # Remove tight_layout as it can interfere with subplots_adjust and annotations
        # self.account_fig.tight_layout(pad=0.1)
        self.account_canvas.draw()

    def update_holdings_pie_chart(self, df_display):
        """Updates the 'Value by Holding' pie chart.

        Uses the provided DataFrame (already filtered by account and 'Show Closed').
        Groups data by symbol and plots market values. Handles grouping small slices
        into 'Other'.

        Args:
            df_display (pd.DataFrame): The DataFrame containing the holdings data
                                       currently displayed in the main table.
        """
        self.holdings_ax.clear()
        # --- Turn axis off by default, only turn on if plotting ---
        # Explicitly set background colors based on current theme
        fig = self.holdings_fig
        if fig:
            fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
        self.holdings_ax.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())

        self.holdings_ax.axis("off")

        if not isinstance(df_display, pd.DataFrame) or df_display.empty:
            # self.holdings_ax.text(0.5, 0.5, 'No Holdings Data', ha='center', va='center', transform=self.holdings_ax.transAxes, fontsize=9, color=COLOR_TEXT_SECONDARY)
            self.holdings_canvas.draw()
            return

        display_currency = self.currency_combo.currentText()
        col_defs = get_column_definitions(display_currency)
        value_col_ui = "Mkt Val"
        value_col_actual = col_defs.get(value_col_ui)

        if (
            not value_col_actual
            or value_col_actual not in df_display.columns
            or "Symbol" not in df_display.columns
        ):
            # self.holdings_ax.text(0.5, 0.5, 'Missing Value/Symbol', ha='center', va='center', transform=self.holdings_ax.transAxes, fontsize=9, color=COLOR_TEXT_SECONDARY)
            self.holdings_canvas.draw()
            return

        holdings_values = df_display.groupby("Symbol")[value_col_actual].sum()
        # Filter out the "TOTALS" summary row if it exists, as it's not a holding
        if "is_summary_row" in df_display.columns and not df_display.empty:
            df_for_pie = df_display[df_display["is_summary_row"] != True].copy()
        else:
            df_for_pie = df_display.copy()

        holdings_values = df_for_pie.groupby("Symbol")[value_col_actual].sum()
        # --- FIX: Filter for positive values for pie chart ---
        holdings_values = holdings_values[holdings_values > 1e-3]

        if holdings_values.empty:
            # self.holdings_ax.text(0.5, 0.5, 'No Holdings > 0', ha='center', va='center', transform=self.holdings_ax.transAxes, fontsize=9, color=COLOR_TEXT_SECONDARY)
            self.holdings_canvas.draw()
            return
        else:
            # --- Turn axis back on only if we have data to plot ---
            self.holdings_ax.axis("on")  # Turn axis back on for plotting
            # self.holdings_ax.axis("equal")  # REMOVED - Pie chart handles aspect ratio

            holdings_values = holdings_values.sort_values(ascending=False)
            # --- MODIFIED: Group all cash symbols into one "Cash" slice ---
            cash_value = holdings_values[
                holdings_values.index.map(is_cash_symbol)
            ].sum()
            non_cash_values = holdings_values[
                ~holdings_values.index.map(is_cash_symbol)
            ]

            if cash_value > 1e-3:
                non_cash_values.loc["Cash"] = cash_value

            holdings_values = non_cash_values.sort_values(ascending=False)
            labels = holdings_values.index.tolist()
            values = holdings_values.values

            if len(values) > CHART_MAX_SLICES:
                top_v = values[: CHART_MAX_SLICES - 1]
                top_l = labels[: CHART_MAX_SLICES - 1]
                other_v = values[CHART_MAX_SLICES - 1 :].sum()
                values = np.append(top_v, other_v)
                labels = top_l + ["Other"]

            cmap = plt.get_cmap("Spectral")
            colors = cmap(np.linspace(0.1, 0.9, len(values)))
            pie_radius = 0.9
            label_offset_multiplier = 1.1
            VERTICAL_THRESHOLD = 0.15  # Adjust as needed

            wedges, _ = self.holdings_ax.pie(
                values,
                labels=None,
                autopct=None,
                startangle=90,
                counterclock=False,
                colors=colors,
                radius=pie_radius,
                wedgeprops={"edgecolor": "white", "linewidth": 0.5},
            )

            label_positions = []
            total_value = np.sum(values)
            for i, p in enumerate(wedges):
                current_label = labels[i]
                ang = (p.theta2 - p.theta1) / 2.0 + p.theta1
                y_edge = pie_radius * np.sin(np.deg2rad(ang))
                x_edge = pie_radius * np.cos(np.deg2rad(ang))
                y_text = label_offset_multiplier * np.sin(np.deg2rad(ang))
                x_text = label_offset_multiplier * np.cos(np.deg2rad(ang))
                original_side = "right" if x_text >= 0 else "left"
                percent = (values[i] / total_value) * 100 if total_value > 1e-9 else 0
                label_text = f"{current_label} ({percent:.0f}%)"
                label_positions.append(
                    {
                        "x": x_text,
                        "y": y_text,
                        "label": label_text,
                        "index": i,
                        "original_side": original_side,
                        "ang": ang,
                        "x_edge": x_edge,
                        "y_edge": y_edge,
                    }
                )

            label_positions = self._adjust_pie_labels(
                label_positions, VERTICAL_THRESHOLD
            )

            arrowprops = dict(
                arrowstyle="-",
                color=COLOR_TEXT_SECONDARY,
                shrinkA=0,
                shrinkB=0,
                relpos=(0.5, 0.5),
            )
            kw = dict(arrowprops=arrowprops, zorder=0, va="center")

            for pos in label_positions:
                x_t = pos["final_x"]
                y_t = pos["final_y"] + pos.get("y_nudge", 0.0)
                x_e, y_e = pos["x_edge"], pos["y_edge"]
                lbl = pos["label"]
                ang = pos["ang"]
                horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x_t))]
                connectionstyle = f"arc,angleA={180 if x_t<0 else 0},angleB={ang},armA=0,armB=40,rad=0"
                kw["arrowprops"].update({"connectionstyle": connectionstyle})
                self.holdings_ax.annotate(
                    lbl,
                    xy=(x_e, y_e),
                    xytext=(x_t, y_t),
                    horizontalalignment=horizontalalignment,
                    **kw,
                    fontsize=8,
                    color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                )

        # Adjust subplot parameters to make space for labels
        # Increase left/right margins, decrease top/bottom slightly if needed
        self.holdings_fig.subplots_adjust(left=0.15, right=0.85, top=0.95, bottom=0.05)
        # Remove tight_layout as it can interfere with subplots_adjust and annotations
        # self.holdings_fig.tight_layout(pad=0.1)
        self.holdings_canvas.draw()

    def update_performance_graphs(self, initial=False):
        """
        Updates the historical performance line graphs (Accumulated Gain and Value).

        Uses data stored in `self.historical_data`. Filters data based on the selected
        date range from the UI. Plots portfolio performance and selected benchmarks.
        Adds annotations like the total TWR factor.

        Args:
            initial (bool, optional): If True, draws empty graphs with titles, typically
                                      used during initialization before data is loaded.
                                      Defaults to False.
        """
        # --- Determine Titles & Scope ---
        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)
        scope_label = "Overall Portfolio"
        if (
            self.available_accounts
            and num_selected > 0
            and num_selected != num_available
        ):
            if num_selected == 1:
                scope_label = f"Account: {self.selected_accounts[0]}"
            else:
                scope_label = f"Selected Accounts ({num_selected}/{num_available})"
        elif not self.available_accounts:
            scope_label = "No Accounts Available"

        # logging.info(
        #     f"Updating performance graphs for scope: {scope_label}... Initial: {initial}, Benchmarks: {self.selected_benchmarks}"
        # )
        self.perf_return_ax.clear()
        self.abs_value_ax.clear()

        # --- ADDED: Remove any previously created secondary y-axis (twinx) ---
        # This prevents stacking multiple FX rate axes when switching currencies.
        if hasattr(self, "abs_value_fig"):
            for ax in self.abs_value_fig.get_axes():
                if ax is not self.abs_value_ax:
                    ax.remove()
        # --- END ADDED ---

        # Explicitly set backgrounds for performance line graphs
        # These graphs are in PerfGraphsContainer, which has its own QSS background.
        # Both figure and axes should match the main dashboard background.
        perf_fig_bg = self.QCOLOR_BACKGROUND_THEMED.name()
        perf_ax_bg = self.QCOLOR_BACKGROUND_THEMED.name()

        if self.perf_return_fig:
            self.perf_return_fig.patch.set_facecolor(perf_fig_bg)
        if self.perf_return_ax:
            self.perf_return_ax.patch.set_facecolor(perf_ax_bg)
        if self.abs_value_fig:
            self.abs_value_fig.patch.set_facecolor(perf_fig_bg)
        if self.abs_value_ax:
            self.abs_value_ax.patch.set_facecolor(perf_ax_bg)

        # --- Clear existing mplcursors if they exist ---
        # Disconnect signals to prevent errors if objects persist unexpectedly
        if hasattr(self, "return_cursor") and self.return_cursor:
            try:
                self.return_cursor.disconnect_all()
            except Exception:
                pass  # Ignore errors during disconnect
            self.return_cursor = None
        if hasattr(self, "value_cursor") and self.value_cursor:
            try:
                self.value_cursor.disconnect_all()
            except Exception:
                pass
            self.value_cursor = None
        # --- End Clear cursors ---

        # --- Base Styling & Aspect ---
        for ax in [self.perf_return_ax, self.abs_value_ax]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(self.QCOLOR_BORDER_THEMED.name())
            ax.spines["left"].set_color(self.QCOLOR_BORDER_THEMED.name())
            ax.tick_params(
                axis="x", colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name(), labelsize=8
            )
            ax.tick_params(
                axis="y", colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name(), labelsize=8
            )
            ax.xaxis.label.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name())
            ax.yaxis.label.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name())
            ax.title.set_color(self.QCOLOR_TEXT_PRIMARY_THEMED.name())
            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color=self.QCOLOR_BORDER_THEMED.name(),  # Use themed border for grid
            )
            ax.set_aspect("auto")  # Ensure aspect is auto after clearing

        # --- Dynamic Titles and Currency ---
        benchmark_display_name = (
            ", ".join(self.selected_benchmarks)
            if self.selected_benchmarks
            else "None"  # self.selected_benchmarks are display names
        )
        display_currency = self._get_currency_symbol(get_name=True)
        currency_symbol = self._get_currency_symbol()
        return_graph_title = f"{scope_label} - Accumulated Gain (TWR)"
        value_graph_title = f"{scope_label} - Value ({display_currency})"

        # --- Get Selected Date Range for Limits ---
        plot_start_date = None
        plot_end_date = None
        try:
            plot_start_date = self.graph_start_date_edit.date().toPython()
            plot_end_date = self.graph_end_date_edit.date().toPython()
        except Exception as e:
            logging.error(f"Error getting plot dates: {e}")

        # --- Handle Initial State or Missing Data ---
        if (
            initial
            or not isinstance(self.historical_data, pd.DataFrame)
            or self.historical_data.empty
        ):
            self.perf_return_ax.text(
                0.5,
                0.5,
                "No Performance Data",
                ha="center",
                va="center",
                transform=self.perf_return_ax.transAxes,
                fontsize=10,
                color=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
            )
            self.abs_value_ax.text(
                0.5,
                0.5,
                "No Value Data",
                ha="center",
                va="center",
                transform=self.abs_value_ax.transAxes,
                fontsize=10,
                color=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
            )

            if plot_start_date and plot_end_date:
                try:
                    self.perf_return_ax.set_xlim(plot_start_date, plot_end_date)
                    self.abs_value_ax.set_xlim(plot_start_date, plot_end_date)
                    self.perf_return_ax.set_ylim(-10, 10)  # Default Y range
                    self.abs_value_ax.set_ylim(0, 100)  # Default Y range
                except Exception as e:
                    logging.warning(f"Warn setting initial graph limits: {e}")
            self.perf_return_canvas.draw()
            self.abs_value_canvas.draw()
            return

        # --- Data Prep (Use Full Data stored in self.historical_data) ---
        results_df = self.historical_data.copy()
        if not isinstance(results_df.index, pd.DatetimeIndex):
            try:
                results_df.index = pd.to_datetime(results_df.index)
            except Exception as e:
                logging.error(
                    f"ERROR converting historical index to DatetimeIndex: {e}"
                )
                # Attempt to draw empty graphs if index fails
                self.update_performance_graphs(initial=True)
                return
        results_df.sort_index(inplace=True)

        # --- Filter data to the selected date range for Y limit calculation ONLY ---
        results_visible_df = results_df.copy()  # Default to full data for range calc
        if plot_start_date and plot_end_date:
            try:
                pd_start = pd.Timestamp(plot_start_date)
                pd_end = pd.Timestamp(plot_end_date)
                # Ensure index is timezone-naive before comparison if needed
                if results_visible_df.index.tz is not None:
                    results_visible_df.index = results_visible_df.index.tz_localize(
                        None
                    )

                results_visible_df = results_df.loc[pd_start:pd_end]  # Use .loc slicing
                logging.debug(
                    f"[Graph Update] Using date range {plot_start_date} to {plot_end_date} for Y-limit calculation ({len(results_visible_df)} rows)"
                )
            except Exception as e_filter:
                logging.warning(
                    f"Error filtering DataFrame by date for Y limits: {e_filter}. Using full range."
                )
                results_visible_df = results_df  # Fallback to full range
        else:
            logging.warning(
                "[Graph Update] No valid date range from UI for Y-limit filtering. Using full range."
            )
            results_visible_field = results_df

        # --- Calculate Y Ranges from VISIBLE Data ---
        min_y_ret, max_y_ret = np.inf, -np.inf
        min_y_val, max_y_val = np.inf, -np.inf
        port_plotted_visible, bench_plotted_visible_count, val_data_plotted_visible = (
            False,
            0,
            False,
        )
        port_col = "Portfolio Accumulated Gain"
        if port_col in results_visible_df.columns:
            vg_vis = results_visible_df[port_col].dropna()
            if not vg_vis.empty:
                pct_vis = (vg_vis - 1) * 100
                min_y_ret = min(min_y_ret, pct_vis.min())
                max_y_ret = max(max_y_ret, pct_vis.max())
                port_plotted_visible = True

        for b in self.selected_benchmarks:
            bc = f"{b} Accumulated Gain"
            if bc in results_visible_df.columns:
                vgb_vis = results_visible_df[bc].dropna()
                if not vgb_vis.empty:
                    pctb_vis = (vgb_vis - 1) * 100
                    min_y_ret = min(min_y_ret, pctb_vis.min())
                    max_y_ret = max(max_y_ret, pctb_vis.max())
                    bench_plotted_visible_count += 1

        val_col = "Portfolio Value"
        vv_vis = (
            results_visible_df[val_col].dropna()
            if val_col in results_visible_df.columns
            else pd.Series(dtype=float)
        )
        val_data_plotted_visible = not vv_vis.empty
        # FIX: Check for finite values before calculating min/max
        vv_vis_finite = vv_vis[np.isfinite(vv_vis)]
        if not vv_vis_finite.empty:
            min_y_val = min(min_y_val, vv_vis_finite.min())
            max_y_val = max(max_y_val, vv_vis_finite.max())
            val_data_plotted_visible = True  # Mark as plotted if finite values exist

        logging.debug(
            f"[Graph Update] Visible RETURN Y Range (data): Min={min_y_ret}, Max={max_y_ret}"
        )
        logging.debug(
            f"[Graph Update] Visible VALUE Y Range (data): Min={min_y_val}, Max={max_y_val}"
        )

        # --- Plotting Setup ---
        # Use the first color for the portfolio, the rest for benchmarks
        all_colors = [
            "red",
            "blue",
            "green",
            "orange",
            "purple",
            "brown",
            "magenta",
            "cyan",
        ]
        portfolio_color = all_colors[0]
        return_lines_plotted = []  # Store plotted lines for the return graph
        value_lines_plotted = []  # Store plotted lines for the value graph

        # --- Plot 1: Accumulated Gain (TWR %) ---
        port_plotted_full = False  # Track if portfolio line was plotted from full data
        if port_col in results_df.columns:
            vg_full = results_df[port_col].dropna()  # Plot FULL data
            if not vg_full.empty:
                pct_full = (vg_full - 1) * 100
                lbl = f"{scope_label}"
                (line,) = self.perf_return_ax.plot(
                    pct_full.index,
                    pct_full,
                    label=lbl,
                    linewidth=2.0,
                    color=portfolio_color,  # Use defined portfolio color
                    zorder=10,
                )
                return_lines_plotted.append(line)
                port_plotted_full = True

        bench_plotted_count_full = 0
        for i, display_name in enumerate(
            self.selected_benchmarks
        ):  # Iterate display names
            yf_ticker = BENCHMARK_MAPPING.get(display_name)
            if yf_ticker:
                benchmark_col_name = f"{yf_ticker} Accumulated Gain"  # Column name in results_df uses ticker
                if benchmark_col_name in results_df.columns:
                    vgb_full = results_df[benchmark_col_name].dropna()
                    if not vgb_full.empty:
                        pctb_full = (vgb_full - 1) * 100
                        bcol = all_colors[(i + 1) % len(all_colors)]
                        (line,) = self.perf_return_ax.plot(
                            pctb_full.index,
                            pctb_full,
                            label=f"{display_name}",  # Use display_name for the label
                            linewidth=1.5,
                            color=bcol,
                            alpha=0.8,
                        )
                        return_lines_plotted.append(line)
                        bench_plotted_count_full += 1

        # Add TWR Annotation
        # --- MODIFIED: Calculate TWR for the visible period ---
        period_twr_factor = np.nan
        if port_col in results_df.columns:
            # The 'Portfolio Accumulated Gain' in results_df is already re-normalized for the period
            series_for_period_twr = results_df[port_col].dropna()
            if not series_for_period_twr.empty:
                period_twr_factor = series_for_period_twr.iloc[-1]

        if pd.notna(period_twr_factor):
            try:
                logging.debug(
                    f"[Graph Update] Using period_twr_factor for annotation: {period_twr_factor}"
                )
                tfv = float(period_twr_factor)
                tpg = (tfv - 1) * 100.0
                tt = f"Period TWR: {tpg:+.2f}%"  # Changed label to "Period TWR"
                tc_color = QCOLOR_GAIN if tpg >= -1e-9 else QCOLOR_LOSS
                self.perf_return_ax.text(
                    0.98,  # X-coordinate (near right edge)
                    0.05,  # Y-coordinate (near bottom edge)
                    tt,
                    transform=self.perf_return_ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    color=tc_color.name(),
                    va="bottom",  # Vertical alignment
                    ha="right",  # Horizontal alignment
                    bbox=dict(
                        boxstyle="round,pad=0.3",
                        fc=self.QCOLOR_BACKGROUND_THEMED.name(),
                        alpha=0.7,
                        ec="none",
                    ),
                )
            except Exception as e:
                logging.warning(f"Warn adding TWR annotation: {e}")
        # --- END MODIFICATION ---

        # Format Return Plot
        if port_plotted_full or bench_plotted_count_full > 0:
            self.perf_return_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            nl = bench_plotted_count_full + (1 if port_plotted_full else 0)
            if nl > 0:
                self.perf_return_ax.legend(
                    fontsize=8,
                    ncol=min(3, nl),
                    loc="upper left",
                    bbox_to_anchor=(0, 0.95),
                    facecolor=self.QCOLOR_BACKGROUND_THEMED.name(),
                    edgecolor=self.QCOLOR_BORDER_THEMED.name(),
                )  # Adjusted legend position

            self.perf_return_ax.set_ylabel(
                "Accumulated Gain (%)",
                fontsize=9,
                color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
            )
            self.perf_return_ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color=self.QCOLOR_BORDER_THEMED.name(),
            )

            # SET RETURN Y LIMITS (using VISIBLE range min/max)
            try:
                if (
                    np.isfinite(min_y_ret)
                    and np.isfinite(max_y_ret)
                    and max_y_ret >= min_y_ret
                ):
                    yr = max_y_ret - min_y_ret
                    pad = max(yr * 0.05, 1.0)  # Add at least 1% padding
                    fmin = min_y_ret - pad
                    fmax = max_y_ret + pad
                    # Ensure range is not zero
                    if abs(fmax - fmin) < 1e-6:
                        fmax = fmin + 1
                    logging.debug(
                        f"[Graph Update] Setting RETURN ylim based on visible range: {fmin:.2f} to {fmax:.2f}"
                    )
                    self.perf_return_ax.set_ylim(fmin, fmax)
                elif (
                    port_plotted_visible or bench_plotted_visible_count > 0
                ):  # Fallback if range calc failed but visible data existed
                    logging.debug(
                        "[Graph Update] RETURN visible ylim invalid, using autoscale."
                    )
                    self.perf_return_ax.autoscale(enable=True, axis="y", tight=False)
                else:  # Nothing plotted in visible range
                    self.perf_return_ax.set_ylim(-10, 10)
            except Exception as e:
                logging.warning(f"Warn setting RETURN ylim: {e}")
                self.perf_return_ax.autoscale(
                    enable=True, axis="y", tight=False
                )  # Final fallback

        else:  # Nothing plotted from full data
            self.perf_return_ax.text(
                0.5,
                0.5,
                "No Return Data",
                ha="center",
                va="center",
                transform=self.perf_return_ax.transAxes,
                fontsize=10,
                color=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
            )

            self.perf_return_ax.set_ylim(-10, 10)  # Default Y range if no data

        # --- Plot 2: Absolute Value ---
        value_data_plotted_full = (
            False  # Track if value line was plotted from full data
        )
        if val_col in results_df.columns:
            vv_full = results_df[val_col].dropna()  # Plot FULL data
            if not vv_full.empty:
                # Plot the area under the curve first
                self.abs_value_ax.fill_between(
                    vv_full.index,
                    vv_full.values,  # Use .values for y1
                    0,  # Fill down to the x-axis
                    color=portfolio_color,
                    alpha=0.3,  # Adjust transparency as needed
                    label="_nolegend_",  # Avoid duplicate legend entry if line has label
                )
                # Then plot the line on top
                (line,) = self.abs_value_ax.plot(
                    vv_full.index,
                    vv_full,
                    label=f"{scope_label} Value ({currency_symbol})",
                    color=portfolio_color,  # Use portfolio color for value line too
                    linewidth=1.5,
                )
                value_lines_plotted.append(line)
                value_data_plotted_full = True

                # --- ADDED: Add annotation for value change ---
                if len(vv_full) >= 2:
                    start_value = vv_full.iloc[0]
                    end_value = vv_full.iloc[-1]
                    abs_change = end_value - start_value
                    pct_change = (
                        (abs_change / start_value) * 100.0
                        if abs(start_value) > 1e-9
                        else np.nan
                    )

                    change_text = f"Period Change: {currency_symbol}{abs_change:,.0f}"
                    if pd.notna(pct_change):
                        change_text += f" ({pct_change:+.2f}%)"

                    change_color = (
                        self.QCOLOR_GAIN_THEMED
                        if abs_change >= -1e-9
                        else self.QCOLOR_LOSS_THEMED
                    )

                    self.abs_value_ax.text(
                        0.98,
                        0.05,
                        change_text,
                        transform=self.abs_value_ax.transAxes,
                        fontsize=9,
                        fontweight="bold",
                        color=change_color.name(),
                        va="bottom",
                        ha="right",
                        bbox=dict(
                            boxstyle="round,pad=0.3",
                            fc=self.QCOLOR_BACKGROUND_THEMED.name(),
                            alpha=0.7,
                            ec="none",
                        ),
                    )
                # --- END ADDED ---

                # --- Define currency formatter ---
                def currency_formatter(x, pos):
                    local_currency_symbol = self._get_currency_symbol()
                    if pd.isna(x):
                        return "N/A"
                    try:
                        if abs(x) >= 1e6:
                            return f"{local_currency_symbol}{x/1e6:,.1f}M"
                        if abs(x) >= 1e3:
                            return f"{local_currency_symbol}{x/1e3:,.0f}K"
                        return f"{local_currency_symbol}{x:,.0f}"
                    except TypeError:
                        return "Err"

                formatter = mtick.FuncFormatter(currency_formatter)
                self.abs_value_ax.yaxis.set_major_formatter(formatter)
                # --- End currency formatter ---

                # --- ADDED: Secondary Y-Axis for Historical FX Rate ---
                if display_currency != "USD":
                    fx_ax = self.abs_value_ax.twinx()
                    fx_ticker = f"{display_currency}=X"

                    if (
                        hasattr(self, "historical_fx_yf")
                        and fx_ticker in self.historical_fx_yf
                    ):
                        fx_data = self.historical_fx_yf[fx_ticker]
                        if not fx_data.empty:
                            # Plot historical FX rate
                            (fx_line,) = fx_ax.plot(
                                fx_data.index,
                                fx_data["price"],
                                color="#9b59b6",  # A distinct purple color
                                linestyle="--",
                                linewidth=1.2,
                                label=f"USD/{display_currency} Rate",
                                alpha=0.8,
                            )
                            # Style the secondary axis
                            fx_ax.set_ylabel(
                                f"USD/{display_currency} Rate",
                                color="#9b59b6",
                                fontsize=8,
                            )
                            fx_ax.tick_params(
                                axis="y", labelcolor="#9b59b6", labelsize=7
                            )
                            fx_ax.spines["right"].set_color("#9b59b6")
                            # Add the FX line to the list for tooltips
                            value_lines_plotted.append(fx_line)
                            # --- ADDED: Create combined legend for both axes ---
                            lines, labels = (
                                self.abs_value_ax.get_legend_handles_labels()
                            )
                            lines2, labels2 = fx_ax.get_legend_handles_labels()
                            self.abs_value_ax.legend(
                                lines + lines2,
                                labels + labels2,
                                loc="upper left",
                                fontsize=8,
                                facecolor=self.QCOLOR_BACKGROUND_THEMED.name(),
                                edgecolor=self.QCOLOR_BORDER_THEMED.name(),
                            )

                # self.abs_value_ax.set_title(
                #     value_graph_title,
                #     fontsize=10,
                #     weight="bold",
                #     color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                # )
                self.abs_value_ax.set_ylabel(
                    f"Value ({currency_symbol})",
                    fontsize=9,
                    color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                )
                self.abs_value_ax.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=0.5,
                    color=self.QCOLOR_BORDER_THEMED.name(),
                )

                # --- SET VALUE Y LIMITS (using VISIBLE range min/max) ---
                try:
                    if (
                        val_data_plotted_visible  # Check if any valid data was found
                        and np.isfinite(min_y_val)
                        and np.isfinite(max_y_val)
                        and max_y_val >= min_y_val
                    ):
                        yrv = max_y_val - min_y_val
                        padv = max(yrv * 0.05, 10.0)  # Add reasonable padding
                        fminv = min_y_val - padv
                        fminv = max(0, fminv)  # Value likely shouldn't go below zero
                        fmaxv = max_y_val + padv
                        if abs(fmaxv - fminv) < 1e-6:
                            fmaxv = fminv + 10  # Ensure some range
                        logging.debug(
                            f"[Graph Update] Setting VALUE ylim based on visible range: {fminv:.2f} to {fmaxv:.2f}"
                        )
                        self.abs_value_ax.set_ylim(fminv, fmaxv)
                    elif (
                        val_data_plotted_visible
                    ):  # Fallback if range calc failed but visible data existed
                        logging.debug(
                            "[Graph Update] VALUE visible ylim invalid, using autoscale."
                        )
                        self.abs_value_ax.autoscale(enable=True, axis="y", tight=False)
                    else:  # Nothing plotted in visible range
                        self.abs_value_ax.set_ylim(0, 100)  # Default Y range
                except Exception as e:
                    logging.warning(f"Warn setting VALUE ylim: {e}")
                    self.abs_value_ax.autoscale(
                        enable=True, axis="y", tight=False
                    )  # Final fallback
            else:  # No valid value data in full range
                self.abs_value_ax.text(
                    0.5,
                    0.5,
                    "No Value Data",
                    ha="center",
                    va="center",
                    transform=self.abs_value_ax.transAxes,
                    fontsize=10,
                    color=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
                )
                # self.abs_value_ax.set_title(
                #     value_graph_title,
                #     fontsize=10,
                #     weight="bold",
                #     color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                # )
                self.abs_value_ax.set_ylim(0, 100)  # Default Y range

        else:  # Value column doesn't even exist in full data
            self.abs_value_ax.text(
                0.5,
                0.5,
                "No Value Data",
                ha="center",
                va="center",
                transform=self.abs_value_ax.transAxes,
                fontsize=10,
                color=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
            )

            self.abs_value_ax.set_ylim(0, 100)

        # --- Apply Final Layout Adjustments and X Limits ---
        for fig, ax in [
            (self.perf_return_fig, self.perf_return_ax),
            (self.abs_value_fig, self.abs_value_ax),
        ]:
            try:
                # Option 1: Increase padding for tight_layout
                # fig.tight_layout(
                #     pad=0.5, h_pad=0.7
                # )  # Increased pad and added h_pad for height

                # Option 2: Or, more explicitly, adjust subplot parameters
                # You might need to experiment with the 'top' value (0.0 to 1.0)
                fig.subplots_adjust(
                    top=1.0, bottom=0.0, left=0.17, right=1.0
                )  # Maximize plotting area by removing all margins

                fig.autofmt_xdate(rotation=15)
                # --- MODIFIED: Adjust x-axis start limit ---
                if (
                    plot_start_date
                    and plot_end_date
                    and isinstance(self.historical_data, pd.DataFrame)
                    and not self.historical_data.empty
                ):
                    try:
                        first_data_date = (
                            self.historical_data.index.min().date()
                        )  # Get first date from filtered data
                        effective_start_date = max(
                            plot_start_date, first_data_date
                        )  # Use the later date
                        logging.debug(
                            f"Setting xlim: Effective Start={effective_start_date}, UI End={plot_end_date}"
                        )
                        ax.set_xlim(effective_start_date, plot_end_date)
                    except Exception as e_xlim_calc:
                        logging.warning(
                            f"Could not determine effective start date for xlim: {e_xlim_calc}. Using UI dates."
                        )
                        ax.set_xlim(
                            plot_start_date, plot_end_date
                        )  # Fallback to UI dates
                elif plot_start_date and plot_end_date:  # If no data, just use UI dates
                    ax.set_xlim(plot_start_date, plot_end_date)
            except Exception as e:
                logging.warning(f"Warn setting layout/xlim: {e}")

        # --- Activate mplcursors AFTER plotting and formatting ---
        if MPLCURSORS_AVAILABLE:
            try:
                # Clear previous cursors explicitly again just before creating new ones
                if hasattr(self, "return_cursor") and self.return_cursor:
                    self.return_cursor.disconnect_all()
                if hasattr(self, "value_cursor") and self.value_cursor:
                    self.value_cursor.disconnect_all()

                if return_lines_plotted:
                    self.return_cursor = mplcursors.cursor(
                        return_lines_plotted, hover=mplcursors.HoverMode.Transient
                    )  # Use Transient for hover
                    self.return_cursor.connect("add", self._format_tooltip_annotation)
                    logging.debug("mplcursors activated for Return graph.")
                else:
                    self.return_cursor = None

                if value_lines_plotted:
                    self.value_cursor = mplcursors.cursor(
                        value_lines_plotted, hover=mplcursors.HoverMode.Transient
                    )  # Use Transient for hover
                    self.value_cursor.connect("add", self._format_tooltip_annotation)
                    logging.debug("mplcursors activated for Value graph.")
                else:
                    self.value_cursor = None

            except Exception as e_cursor:
                logging.error(f"Error activating mplcursors: {e_cursor}")
                self.return_cursor = None  # Ensure cursors are None on error
                self.value_cursor = None
        # --- End mplcursors Activation ---

        # --- Draw Canvases ---
        try:
            self.perf_return_canvas.draw_idle()  # Use draw_idle for potentially smoother updates
            self.abs_value_canvas.draw_idle()
        except Exception as e_draw:
            logging.error(f"Error drawing graph canvas: {e_draw}")

        # --- Re-apply Backgrounds based on theme ---
        try:
            # Pie charts container and content frame background are handled by QSS.
            # Figure and Axes patch colors for Matplotlib plots:
            fig_face_color = self.QCOLOR_BACKGROUND_THEMED.name()
            # Use input background for axes of pie charts for consistency if desired,
            # or main background if they should blend more with the figure.
            # For pies, usually figure and axes are same.
            ax_pie_face_color = self.QCOLOR_BACKGROUND_THEMED.name()

            # Performance graphs are in a QWidget#PerfGraphsContainer which has its own BG from QSS.
            # So, make Matplotlib figure/axes transparent for these if QSS is to show through.
            # Or, explicitly set their BG to match the PerfGraphsContainer's QSS BG.
            # For now, let's try to match the QSS intent for PerfGraphsContainer.
            # If QSS sets PerfGraphsContainer to DARK_COLOR_BG_HEADER, then use that.
            perf_graph_bg_color_to_match_qss = (
                self.QCOLOR_HEADER_BACKGROUND_THEMED.name()
            )

            for fig, ax in [
                (self.account_fig, self.account_ax),
                (self.holdings_fig, self.holdings_ax),
                (
                    self.asset_type_pie_fig,
                    self.asset_type_pie_ax,
                ),  # Asset Allocation Tab
                (self.sector_pie_fig, self.sector_pie_ax),  # Asset Allocation Tab
                (self.geo_pie_fig, self.geo_pie_ax),  # Asset Allocation Tab
                (self.industry_pie_fig, self.industry_pie_ax),  # Asset Allocation Tab
            ]:
                if fig:
                    fig.patch.set_facecolor(fig_face_color)
                if ax:
                    ax.patch.set_facecolor(
                        ax_pie_face_color
                    )  # Use specific pie axes color

            # For performance line graphs, axes are often different from figure
            ax_line_graph_face_color = self.QCOLOR_BACKGROUND_THEMED.name()
            for fig, ax in [
                (self.perf_return_fig, self.perf_return_ax),
                (self.abs_value_fig, self.abs_value_ax),
            ]:
                if fig:
                    fig.patch.set_facecolor(perf_graph_bg_color_to_match_qss)
                if ax:
                    ax.patch.set_facecolor(
                        ax_line_graph_face_color
                    )  # Use input-like background for axes

            # Bar charts (Periodic, Dividend, Capital Gains)
            # Figure background matches main, axes background matches input style
            bar_fig_bg_color = self.QCOLOR_BACKGROUND_THEMED.name()
            bar_ax_bg_color = self.QCOLOR_BACKGROUND_THEMED.name()

            for fig, ax in [
                (self.annual_bar_fig, self.annual_bar_ax),
                (self.monthly_bar_fig, self.monthly_bar_ax),
                (self.weekly_bar_fig, self.weekly_bar_ax),
                (self.dividend_bar_fig, self.dividend_bar_ax),
                (self.cg_bar_fig, self.cg_bar_ax),
            ]:
                if fig:
                    fig.patch.set_facecolor(bar_fig_bg_color)
                if ax:
                    ax.patch.set_facecolor(bar_ax_bg_color)

        except Exception as e_bg:
            logging.warning(f"Warn setting graph background: {e_bg}")

    # --- Data Handling and UI Update Methods ---

    def update_dashboard_summary(self, df_filtered_for_cash: pd.DataFrame):
        """
        Updates the summary grid labels with aggregated portfolio metrics.

        Uses data from `self.summary_metrics_data` (which reflects the selected
        account scope) and the passed `df_filtered_for_cash` DataFrame to determine
        the cash balance. Formats values with currency symbols and applies gain/loss
        coloring. Calculates and displays
        Total Return % and Annualized TWR %.
        """
        # --- Determine Scope ---
        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)
        is_all_accounts_selected = (
            not self.available_accounts or num_selected == num_available
        )  # Treat no accounts available as "all"

        display_currency = self.currency_combo.currentText()
        currency_symbol = self._get_currency_symbol()
        col_defs = get_column_definitions(display_currency)
        cash_val_col_actual = col_defs.get("Mkt Val")

        summary_widgets = {
            "net_value": self.summary_net_value,
            "day_change": (
                self.summary_day_change_label,
                self.summary_day_change_value,
            ),
            "total_gain": self.summary_total_gain,
            "realized_gain": self.summary_realized_gain,
            "unrealized_gain": self.summary_unrealized_gain,
            "dividends": self.summary_dividends,
            "commissions": self.summary_commissions,
            "cash": self.summary_cash,
        }
        for key, (label_widget, value_widget) in summary_widgets.items():
            self.update_summary_value(value_widget, None, metric_type="clear")
        self.update_summary_value(
            self.summary_total_return_pct[1], None, metric_type="clear"
        )  # Use new name
        self.update_summary_value(
            self.summary_annualized_twr[1], None, metric_type="clear"
        )

        # --- ADDED: Clear Estimated Annual Dividend Income ---
        if hasattr(self, "est_annual_income_value_label"):
            self.update_summary_value(
                self.est_annual_income_value_label, None, metric_type="clear"
            )
        # --- END ADDED ---

        # Clear FX G/L
        self.update_summary_value(  # Corrected
            self.summary_fx_gl_abs_value, None, metric_type="clear"
        )
        self.update_summary_value(  # Corrected
            self.summary_fx_gl_pct_value, None, metric_type="clear"
        )

        # Data source is always the overall summary metrics, which now reflect the selected scope
        data_source_current = self.summary_metrics_data

        # Update common summary items using data_source_current
        if data_source_current:
            # --- Cash calculation needs to use the filtered holdings_data ---
            overall_cash_value = None
            if (
                isinstance(df_filtered_for_cash, pd.DataFrame)
                and not df_filtered_for_cash.empty
                and cash_val_col_actual
                and "Symbol" in df_filtered_for_cash.columns
                and cash_val_col_actual in df_filtered_for_cash.columns
            ):
                try:
                    # Sum up all cash symbols
                    cash_mask = df_filtered_for_cash["Symbol"].apply(is_cash_symbol)

                    overall_cash_value = (
                        pd.to_numeric(  # Ensure it's numeric before summing
                            df_filtered_for_cash.loc[cash_mask, cash_val_col_actual],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .sum()
                        if cash_mask.any()
                        else 0.0
                    )
                except Exception:
                    overall_cash_value = None
            # --- End Cash Calc Update ---

            self.update_summary_value(
                self.summary_net_value[1],
                data_source_current.get("market_value"),
                currency_symbol,
                True,
                False,
                "net_value",
            )

            # --- FIX: Correctly scale and format day change percentage ---
            day_change_val = data_source_current.get("day_change_display")
            day_change_pct = data_source_current.get(
                "day_change_percent"
            )  # This is the % value from backend

            if pd.notna(day_change_val):
                # Format the absolute currency change
                day_change_abs_val_str = f"{currency_symbol}{abs(day_change_val):,.2f}"
                day_change_pct_str = ""  # Initialize percentage string

                # Format the percentage change (use directly, add sign)
                if pd.notna(day_change_pct) and np.isfinite(day_change_pct):
                    pct_val = day_change_pct  # Use the value directly
                    sign = (
                        "+" if pct_val >= -1e-9 else ""
                    )  # Add '+' sign for positive/zero
                    day_change_pct_str = f" ({sign}{pct_val:,.2f}%)"  # Format with sign
                elif np.isinf(day_change_pct):
                    day_change_pct_str = " (Inf%)"  # Handle infinity

                # Combine the strings

            # --- END FIX ---

            self.update_summary_value(
                self.summary_day_change_value,
                day_change_val,
                currency_symbol,
                True,
                False,
                "day_change",
            )
            self.update_summary_value(
                self.summary_day_change_pct,
                day_change_pct,
                currency_symbol,
                False,
                True,
                "day_change_pct",
            )

            self.update_summary_value(
                self.summary_total_gain[1],
                data_source_current.get("total_gain"),
                currency_symbol,
                True,
                False,
                "total_gain",
            )
            self.update_summary_value(
                self.summary_realized_gain[1],
                data_source_current.get("realized_gain"),
                currency_symbol,
                True,
                False,
                "realized_gain",
            )
            self.update_summary_value(
                self.summary_unrealized_gain[1],
                data_source_current.get("unrealized_gain"),
                currency_symbol,
                True,
                False,
                "unrealized_gain",
            )
            self.update_summary_value(
                self.summary_dividends[1],
                data_source_current.get("dividends"),
                currency_symbol,
                True,
                False,
                "dividends",
            )
            self.update_summary_value(
                self.summary_commissions[1],
                data_source_current.get("commissions"),
                currency_symbol,
                True,
                False,
                "fees",
            )
            # Use the recalculated overall_cash_value
            self.update_summary_value(
                self.summary_cash[1],
                overall_cash_value,
                currency_symbol,
                False,
                False,
                "cash",
            )

        # --- Handle Performance Metrics Display ---
        # Slot 1: Total Return %
        total_return_pct_value = (
            data_source_current.get("total_return_pct")
            if data_source_current
            else np.nan
        )
        self.update_summary_value(
            self.summary_total_return_pct[1],
            total_return_pct_value,
            "",
            False,
            True,
            "total_return_pct",
        )
        # Set label text (now static)
        self.summary_total_return_pct[0].setText("Total Ret %:")
        self.summary_total_return_pct[0].setVisible(True)
        self.summary_total_return_pct[1].setVisible(True)

        # Slot 2: Annualized TWR %
        twr_factor = (
            self.last_hist_twr_factor
        )  # This factor now reflects the selected scope

        # --- FIX: Use the actual date range from the FULL historical data for TWR annualization ---
        start_date_val = None
        end_date_val = None
        if (
            hasattr(self, "full_historical_data")
            and isinstance(self.full_historical_data, pd.DataFrame)
            and not self.full_historical_data.empty
            and isinstance(self.full_historical_data.index, pd.DatetimeIndex)
        ):
            start_date_val = self.full_historical_data.index.min().date()
            end_date_val = self.full_historical_data.index.max().date()
            logging.debug(
                f"Using full_historical_data date range for TWR annualization: {start_date_val} to {end_date_val}"
            )
        else:  # Fallback to UI controls if historical_data is not ready
            start_date_val = (
                self.graph_start_date_edit.date().toPython()
                if hasattr(self, "graph_start_date_edit")
                else None
            )
            end_date_val = (
                self.graph_end_date_edit.date().toPython()
                if hasattr(self, "graph_end_date_edit")
                else None
            )
            # Only log a warning if we expected data to be present (i.e., not in a cleared/initial state)
            if self.summary_metrics_data:
                logging.warning(
                    f"Fallback to UI date range for TWR annualization: {start_date_val} to {end_date_val}"
                )
        # --- END FIX ---

        annualized_twr_pct = self._calculate_annualized_twr(
            twr_factor, start_date_val, end_date_val
        )
        self.update_summary_value(
            self.summary_annualized_twr[1],
            annualized_twr_pct,
            "",
            False,
            True,
            "annualized_twr",
        )
        # Set label text (now static)
        self.summary_annualized_twr[0].setText("Ann. TWR %:")
        self.summary_annualized_twr[0].setVisible(True)
        self.summary_annualized_twr[1].setVisible(True)

        # --- ADDED: Update FX Gain/Loss in Summary Panel ---
        fx_gain_loss_display_val = data_source_current.get("fx_gain_loss_display")
        fx_gain_loss_pct_val = data_source_current.get("fx_gain_loss_pct")

        self.update_summary_value(
            self.summary_fx_gl_abs_value,  # Corrected
            fx_gain_loss_display_val,
            currency_symbol,
            True,  # is_currency
            False,  # is_percent
            "fx_gain_loss",  # metric_type for coloring
        )
        self.update_summary_value(
            self.summary_fx_gl_pct_value,
            fx_gain_loss_pct_val,
            "",  # no currency symbol for percentage
            False,  # is_currency
            True,  # is_percent
            "fx_gain_loss_pct",  # metric_type for coloring
        )
        # --- END ADDED ---

        # --- ADDED: Update Estimated Annual Dividend Income in Dividend History Tab ---
        if hasattr(self, "est_annual_income_value_label") and data_source_current:
            est_annual_income_val = data_source_current.get("est_annual_income_display")
            self.update_summary_value(
                self.est_annual_income_value_label,
                est_annual_income_val,
                currency_symbol,
                True,  # is_currency
                False,  # is_percent
                "dividends",  # Use 'dividends' metric type for coloring (green if positive)
            )
        # --- END ADDED ---

    def update_summary_value(
        self,
        value_label,
        value,
        currency_symbol="$",
        is_currency=True,
        is_percent=False,
        metric_type=None,
    ):
        """Formats and updates a single summary value QLabel.

        Handles formatting for currency, percentages, NaN/None values, and applies
        appropriate text color based on the metric type and value (gain/loss).

        Args:
            value_label (QLabel): The QLabel widget to update.
            value (float | int | None | np.nan): The numeric value to display.
            currency_symbol (str, optional): The currency symbol to prepend if `is_currency` is True. Defaults to "$".
            is_currency (bool, optional): If True, format as currency. Defaults to True.
            is_percent (bool, optional): If True, format as percentage. Defaults to False.
            metric_type (str | None, optional): A string identifying the metric type (e.g.,
                'net_value', 'total_gain', 'fees') to help determine text color. Defaults to None.

        """
        # (Keep implementation as before)
        text = "N/A"
        original_value_float = None
        target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
        if value is not None and pd.notna(value):
            try:
                original_value_float = float(value)
            except (ValueError, TypeError):
                text = "Error"
                target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
        if original_value_float is not None:
            if metric_type in ["net_value", "cash"]:
                target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
            elif metric_type == "dividends":
                target_color = (
                    self.QCOLOR_GAIN_THEMED
                    if original_value_float > 1e-9
                    else self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
                )
            # Fees and commissions are usually displayed as positive numbers but represent losses/costs
            elif metric_type in ["fees", "commissions"]:
                target_color = (
                    self.QCOLOR_LOSS_THEMED
                    if original_value_float > 1e-9
                    else self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
                )
            elif metric_type in [
                "total_gain",
                "realized_gain",
                "unrealized_gain",
                "day_change",
                "portfolio_mwr",
                "period_twr",
            ]:  # Added MWR/TWR here
                if original_value_float > 1e-9:
                    target_color = self.QCOLOR_GAIN_THEMED  # Use themed color
                elif original_value_float < -1e-9:
                    target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
                else:
                    target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
            # Explicitly color Total Return %
            elif metric_type in [
                "account_total_return_pct",
                "overall_total_return_pct",
                "annualized_twr",
                "total_return_pct",
            ]:  # Added total_return_pct
                if original_value_float > 1e-9:
                    target_color = self.QCOLOR_GAIN_THEMED  # Use themed color
                elif original_value_float < -1e-9:
                    target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
                else:
                    target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
            elif metric_type in [
                "fx_gain_loss",
                "fx_gain_loss_pct",
            ]:  # Coloring for FX G/L
                if original_value_float > 1e-9:
                    target_color = self.QCOLOR_GAIN_THEMED  # Use themed color
                elif original_value_float < -1e-9:
                    target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
                else:
                    # For FX G/L, if it's zero, keep it dark, not green.
                    target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color
            # Default coloring
            else:
                if original_value_float > 1e-9:
                    target_color = self.QCOLOR_GAIN_THEMED  # Use themed color
                elif original_value_float < -1e-9:
                    target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
                else:
                    target_color = self.QCOLOR_TEXT_PRIMARY_THEMED  # Use themed color

        if original_value_float is not None:
            value_float_abs = abs(original_value_float)
            if abs(original_value_float) < 1e-9:
                value_float_abs = 0.0
            try:  # Keep sign for percentages
                if np.isinf(original_value_float):
                    text = "Inf %" if is_percent else "Inf"
                elif is_percent:
                    text = f"{original_value_float:,.2f}%"  # Show sign for percentages like MWR/TWR
                elif is_currency:
                    text = f"{currency_symbol}{value_float_abs:,.2f}"
                else:
                    text = f"{value_float_abs:,.2f}"
            except (ValueError, TypeError):
                text = "Format Error"
                target_color = self.QCOLOR_LOSS_THEMED  # Use themed color
        elif metric_type == "clear":
            text = "N/A"
            target_color = self.QCOLOR_TEXT_PRIMARY_THEMED

        # Apply color using rich text
        value_label.setText(f"<font color='{target_color.name()}'>{text}</font>")

    def _edit_transaction_in_db(
        self, transaction_id: int, new_data_dict_from_dialog_pytypes: Dict[str, Any]
    ) -> bool:
        """
        Updates an existing transaction in the database.
        `new_data_dict_from_dialog_pytypes` comes from AddTransactionDialog.get_transaction_data()
        and contains Python types, keyed by CSV-like headers (e.g., "Date (MMM DD, YYYY)").
        This method maps these to DB column names and appropriate data types for `db_utils`.

        Args:
            transaction_id (int): The database ID of the transaction to update.
            new_data_dict_from_dialog_pytypes (Dict[str, Any]): Data from the edit dialog.

        Returns:
            bool: True if the update was successful, False otherwise.
        """
        if not self.db_conn:
            logging.error("DB Edit Error: No active database connection.")
            return False

        logging.info(
            f"Preparing to edit DB transaction ID {transaction_id}. Raw dialog data: {new_data_dict_from_dialog_pytypes}"
        )
        data_for_db_update: Dict[str, Any] = {}
        try:
            # Map and format data from dialog (CSV-like headers) to DB schema
            date_obj = new_data_dict_from_dialog_pytypes.get("Date (MMM DD, YYYY)")
            if isinstance(date_obj, (datetime, date)):
                data_for_db_update["Date"] = date_obj.strftime("%Y-%m-%d")
            elif (
                date_obj is None
                and "Date (MMM DD, YYYY)" in new_data_dict_from_dialog_pytypes
            ):  # Explicit None
                data_for_db_update["Date"] = (
                    None  # This should be validated by dialog if Date is mandatory
                )
            elif date_obj is not None:
                logging.error(
                    f"Invalid date type for DB edit: {type(date_obj)} for value {date_obj}"
                )
                QMessageBox.critical(
                    self,
                    "Data Error",
                    f"Invalid date type from edit dialog: {type(date_obj)}. Update failed.",
                )
                return False

            data_for_db_update["Type"] = new_data_dict_from_dialog_pytypes.get(
                "Transaction Type"
            )
            data_for_db_update["Symbol"] = new_data_dict_from_dialog_pytypes.get(
                "Stock / ETF Symbol"
            )

            # Numeric fields are already float or None from get_transaction_data
            data_for_db_update["Quantity"] = new_data_dict_from_dialog_pytypes.get(
                "Quantity of Units"
            )
            data_for_db_update["Price/Share"] = new_data_dict_from_dialog_pytypes.get(
                "Amount per unit"
            )
            data_for_db_update["Total Amount"] = new_data_dict_from_dialog_pytypes.get(
                "Total Amount"
            )
            data_for_db_update["Commission"] = new_data_dict_from_dialog_pytypes.get(
                "Fees"
            )
            data_for_db_update["Split Ratio"] = new_data_dict_from_dialog_pytypes.get(
                "Split Ratio (new shares per old share)"
            )

            data_for_db_update["Account"] = new_data_dict_from_dialog_pytypes.get(
                "Investment Account"
            )
            data_for_db_update["Note"] = new_data_dict_from_dialog_pytypes.get("Note")

            # --- ADDED: Handle "To Account" for transfers ---
            if data_for_db_update.get("Type", "").lower() == "transfer":
                data_for_db_update["To Account"] = (
                    new_data_dict_from_dialog_pytypes.get("To Account")
                )
                logging.info(f"Edit Transaction: Setting 'To Account' to '{data_for_db_update['To Account']}' for ID {transaction_id}")
                # --- BUG FIX: Ensure date is passed for the deposit part of the transfer ---
                # When a transfer is edited, the corresponding deposit record for the 'To Account'
                # needs to be updated with the same date. The update_transaction_in_db function
                # in db_utils handles finding and updating this second record if the date is provided.
                data_for_db_update["Date"] = new_data_dict_from_dialog_pytypes.get(
                    "Date (MMM DD, YYYY)"
                ).strftime("%Y-%m-%d")

            # Determine Local Currency based on account
            acc_name_for_currency_update = data_for_db_update.get("Account")
            if acc_name_for_currency_update:
                app_config_account_map = self.config.get("account_currency_map", {})
                app_config_default_curr = self.config.get(
                    "default_currency",
                    (
                        config.DEFAULT_CURRENCY
                        if hasattr(config, "DEFAULT_CURRENCY")
                        else "USD"
                    ),
                )
                data_for_db_update["Local Currency"] = app_config_account_map.get(
                    str(acc_name_for_currency_update), app_config_default_curr
                )

            else:  # Should be caught by dialog if Account is mandatory
                data_for_db_update["Local Currency"] = self.config.get(
                    "default_currency",
                    (
                        config.DEFAULT_CURRENCY
                        if hasattr(config, "DEFAULT_CURRENCY")
                        else "USD"
                    ),
                )
                logging.warning(
                    "Account name missing when determining local currency for transaction update."
                )

            # Remove keys with None values if db_utils.update_transaction_in_db expects only provided fields
            # However, db_utils.update_transaction_in_db is designed to handle None for nullable fields.
            # So, we pass the full dict with None values.
            data_for_db_update_cleaned = {
                k: v
                for k, v in data_for_db_update.items()
                if k
                in [  # Ensure only valid DB columns
                    "Date",
                    "Type",
                    "Symbol",
                    "Quantity",
                    "Price/Share",
                    "Total Amount",
                    "Commission",
                    "Account",
                    "Split Ratio",
                    "Note",
                    "Local Currency",
                ]
            }

            # Validate that required fields for DB are present before calling update
            for required_db_col in [
                "Date",
                "Type",
                "Symbol",
                "Account",
                "Local Currency",
            ]:
                if (
                    data_for_db_update_cleaned.get(required_db_col) is None
                ):  # Check if value is None after processing
                    logging.error(
                        f"Critical field '{required_db_col}' is None after mapping. Transaction update failed for ID {transaction_id}."
                    )
                    QMessageBox.critical(
                        self,
                        "Data Error",
                        f"Required field '{required_db_col}' is missing or invalid. Update failed.",
                    )
                    return False

            logging.debug(
                f"Data prepared for DB update (ID: {transaction_id}): {data_for_db_update_cleaned}"
            )

        except Exception as e_map_edit:
            logging.error(
                f"Error mapping dialog data for DB update (ID: {transaction_id}): {e_map_edit}",
                exc_info=True,
            )
            QMessageBox.critical(
                self, "Data Error", "Internal error preparing data for database update."
            )
            return False

        success = update_transaction_in_db(
            self.db_conn, transaction_id, data_for_db_update_cleaned
        )
        if success:
            logging.info(f"Successfully updated transaction ID {transaction_id} in DB.")
            new_account_name_edited = data_for_db_update_cleaned.get("Account")
            if new_account_name_edited:
                self._add_new_account_if_needed(
                    str(new_account_name_edited)
                )  # Update GUI's available accounts

        return success

    def _backup_csv(self, filename_to_backup=None):  # Add optional argument
        """Creates a timestamped backup of the specified transactions CSV."""
        file_to_backup = (
            filename_to_backup if filename_to_backup else self.transactions_file
        )  # Use provided or default
        if not file_to_backup or not os.path.exists(file_to_backup):
            return (
                False,
                f"CSV file to backup not found: {file_to_backup}",
            )  # Use actual filename in message
        try:
            # --- MODIFIED: Use QStandardPaths for backups ---
            # This places backups in a standard user-specific application data location
            app_data_dir_base = QStandardPaths.writableLocation(
                QStandardPaths.AppDataLocation
            )
            if not app_data_dir_base:  # Fallback if AppDataLocation is not available
                app_data_dir_base = QStandardPaths.writableLocation(
                    QStandardPaths.DocumentsLocation
                )
                if not app_data_dir_base:  # Further fallback
                    app_data_dir_base = os.path.expanduser("~")

            investa_app_data_dir = os.path.join(app_data_dir_base, "Investa")
            backup_dir = os.path.join(investa_app_data_dir, "csv_backups")
            # --- END MODIFICATION ---

            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(file_to_backup)  # Use correct base name
            name, ext = os.path.splitext(base_name)
            backup_filename = f"{name}_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            shutil.copy2(
                file_to_backup, backup_path  # Use correct source file
            )  # copy2 preserves metadata
            logging.info(f"CSV backup created: {backup_path}")
            return True, f"Backup saved to: {backup_path}"
        except Exception as e:
            logging.error(f"ERROR creating CSV backup for {file_to_backup}: {e}")
            return False, str(e)

    def _rewrite_csv(
        self, df_to_write: pd.DataFrame, skip_backup=False
    ) -> bool:  # Add skip_backup flag
        """Rewrites the entire CSV file from the DataFrame."""
        if not self.transactions_file:
            QMessageBox.critical(self, "Save Error", "CSV file path is not set.")
            return False

        # Define headers in the correct order
        # This order should ideally match DESIRED_CLEANED_COLUMN_ORDER from csv_utils.py
        # For now, let's define it here.
        cleaned_csv_headers_in_order = [
            "Date",
            "Type",
            "Symbol",
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Account",
            "Split Ratio",
            "Note",
        ]

        df_ordered = pd.DataFrame()
        # df_to_write contains the actual headers from the CSV (which are cleaned if standardized)
        for header in cleaned_csv_headers_in_order:
            if header in df_to_write.columns:
                df_ordered[header] = df_to_write[header]
            else:
                # If a standard column is missing, add it as empty.
                # This could happen if the original CSV was missing a column.
                df_ordered[header] = ""
                logging.warning(
                    f"_rewrite_csv: Standard CSV header '{header}' not found in df_to_write. Added as empty column."
                )

        # --- Ensure correct data types before saving ---
        try:
            # Convert Date column ONLY (needed for date_format) - Use CLEANED name
            date_col_name = "Date"
            if date_col_name in df_ordered.columns:
                original_date_strings = df_ordered[
                    date_col_name
                ].copy()  # Keep original strings

                # Attempt to convert to datetime
                converted_dates = pd.to_datetime(original_date_strings, errors="coerce")

                # Identify where conversion failed (became NaT) but original was not NaT/None/empty string
                failed_conversion_mask = (
                    converted_dates.isna()
                    & original_date_strings.notna()
                    & (original_date_strings != "")
                )

                if failed_conversion_mask.any():
                    num_failed = failed_conversion_mask.sum()
                    logging.warning(
                        f"_rewrite_csv: {num_failed} date string(s) could not be parsed by pd.to_datetime and will be written as original strings."
                    )
                    # For failed conversions, revert to the original string.
                    # For successful conversions or original NaNs/empty strings, keep the (converted) datetime object or NaT.
                    df_ordered[date_col_name] = converted_dates.where(
                        ~failed_conversion_mask, original_date_strings
                    )
                else:
                    # All conversions successful or original values were already NaN/None/empty
                    df_ordered[date_col_name] = converted_dates

                # DO NOT DROP ROWS WITH NaT DATES ANYMORE.
                # NaT values (from originally empty cells or truly unparseable strings not reverted)
                # will be written as empty strings by to_csv.
            else:
                logging.warning(
                    f"_rewrite_csv: Date column '{date_col_name}' not found in df_ordered."
                )
        except Exception as e_dtype:
            logging.error(f"Error converting Date dtype before CSV rewrite: {e_dtype}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Internal data type error before saving:\n{e_dtype}",
            )
            return False
        # --- End dtype conversion ---

        try:
            # --- Backup BEFORE writing ---
            if not skip_backup:  # Only backup if not skipped
                backup_ok, backup_msg = (
                    self._backup_csv()
                )  # Backup the current self.transactions_file
                if not backup_ok:
                    QMessageBox.critical(
                        self,
                        # MODIFIED: Use self as parent for QMessageBox
                        self,
                        "Backup Error",
                        f"Failed to backup CSV before saving:\n{backup_msg}",
                    )
                    return False

            # --- Write to CSV ---
            df_ordered.to_csv(
                # MODIFIED: Use self.transactions_file
                self.transactions_file,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_MINIMAL,  # Keep minimal quoting
                date_format=CSV_DATE_FORMAT,  # Use the specific date format
            )
            logging.info(f"Successfully rewrote CSV: {self.transactions_file}")
            return True
        except PermissionError as e:
            logging.error(f"Permission error rewriting CSV: {e}")
            QMessageBox.critical(
                self,
                "Save Error",
                f"Permission denied writing to CSV file:\n{e}\n\nIs the file open in another program?",
            )
            return False
        except Exception as e:
            logging.error(f"ERROR rewriting CSV: {e}")
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "Save Error",
                f"An unexpected error occurred while saving CSV:\n{e}",
            )
            return False

    def _edit_transaction_in_csv(
        self, original_index_to_edit: int, new_data_dict: Dict[str, str]
    ) -> bool:
        """
        Finds a transaction by its 'original_index' (internal DB ID), updates it with new_data_dict,
        and rewrites the entire CSV file.
        Triggers a data refresh via ManageTransactionsDialog's data_changed signal if successful.
        """
        """Finds a transaction by original_index (internal DB ID), updates it, and rewrites the CSV."""
        if not hasattr(self, "original_data") or self.original_data.empty:
            QMessageBox.critical(
                self,
                "Data Error",
                "Original transaction data not available for editing.",
            )
            return False

        logging.info(
            f"Attempting to edit transaction with original_index (internal DB ID): {original_index_to_edit}"
        )
        # Work on a copy
        df_modified = self.original_data.copy()

        # Find the row index in the DataFrame corresponding to the original_index (internal DB ID)
        row_mask = df_modified["original_index"] == original_index_to_edit
        if not row_mask.any():
            QMessageBox.critical(
                self,
                "Edit Error",
                "Could not find the transaction to edit (original index mismatch).",
            )
            return False

        df_row_index = df_modified.index[row_mask].tolist()[
            0
        ]  # Get the DataFrame index

        # Update the row
        for csv_header, new_value_str in new_data_dict.items():
            # csv_header is the standard original CSV header, e.g., "Date (MMM DD, YYYY)"
            # self.original_data (and thus df_modified) has actual headers from the CSV file.
            # self.original_to_cleaned_header_map maps standard original CSV headers to cleaned names.
            # If the CSV uses cleaned names, then df_modified.columns contains these cleaned names.

            # Get the cleaned name corresponding to the standard CSV header
            # This map should have been populated correctly by data_loader.py
            actual_col_name_in_df = self.original_to_cleaned_header_map.get(csv_header)

            if actual_col_name_in_df:
                if actual_col_name_in_df in df_modified.columns:
                    col_dtype = df_modified[actual_col_name_in_df].dtype
                    try:
                        if pd.api.types.is_numeric_dtype(col_dtype):
                            # Attempt to convert to float, if it fails, set to NaN and handle drop
                            numeric_value = pd.to_numeric(
                                new_value_str, errors="coerce"
                            )
                            if pd.isna(numeric_value):
                                # If it's not a valid numeric, it goes to NaN and will effectively be set to "" when we save the CSV (which saves all float cols)
                                df_modified.loc[df_row_index, actual_col_name_in_df] = (
                                    numeric_value
                                )
                            else:
                                df_modified.loc[df_row_index, actual_col_name_in_df] = (
                                    numeric_value
                                )
                            logging.debug(
                                f"Updated numeric column '{actual_col_name_in_df}' (from std header '{csv_header}') with value '{new_value_str}' (converted to float)"
                            )
                        else:
                            # String column
                            df_modified.loc[df_row_index, actual_col_name_in_df] = (
                                new_value_str
                            )
                            logging.debug(
                                f"Updated string column '{actual_col_name_in_df}' (from std header '{csv_header}') with value '{new_value_str}'"
                            )
                    except Exception as e_set_value:
                        logging.error(
                            f"Error setting value for column '{actual_col_name_in_df}' (from std header '{csv_header}') at index {df_row_index}: {e_set_value}"
                        )
                        # Continue with other columns
                else:
                    logging.warning(
                        f"Mapped column name '{actual_col_name_in_df}' (from std header '{csv_header}') not found in df_modified columns: {df_modified.columns.tolist()}."
                    )
            else:
                # This case means the standard_csv_header from new_data_dict wasn't in original_to_cleaned_header_map
                # which would be unusual if original_to_cleaned_header_map is built from the same standard mapping.
                # However, it's also possible the CSV has a completely unexpected header that doesn't map.
                # More robust: try direct match if map fails, as a fallback if CSV *did* use standard headers.
                if csv_header in df_modified.columns:
                    logging.warning(
                        f"Standard header '{csv_header}' not in original_to_cleaned_header_map, but found directly in df_modified.columns. Updating directly."
                    )
                    try:
                        df_modified.loc[df_row_index, csv_header] = new_value_str
                    except Exception as e_set_value_direct:
                        logging.error(
                            f"Error setting value directly for column '{csv_header}' at index {df_row_index}: {e_set_value_direct}"
                        )
                else:
                    logging.warning(
                        f"Header '{csv_header}' from edit dialog could not be mapped to a column in original_data and was not found directly."
                    )

        # Rewrite the entire CSV
        # L9896: # The _rewrite_csv method should handle dropping 'original_index' (internal DB ID) if it's still there
        rewrite_successful = self._rewrite_csv(df_modified)

        if rewrite_successful:
            # Update the main app's original_data immediately
            self.original_data = df_modified.copy()
            QMessageBox.information(
                self, "Success", "Transaction updated successfully."
            )
            # The ManageTransactionsDialog will emit data_changed, which triggers self.refresh_data()
            return True
        else:
            # _rewrite_csv would have shown its own error message
            return False

    def _delete_transaction_in_db(self, transaction_id: int) -> bool:
        """
        Deletes a transaction from the database using its ID.


        Args:
            transaction_id (int): The database ID of the transaction to delete.

        Returns:
            bool: True if the deletion was successful, False otherwise.
        """
        if not self.db_conn:
            logging.error("DB Delete Error: No active database connection.")
            QMessageBox.critical(
                self,
                "Database Error",
                "Cannot delete transaction: No active database connection.",
            )
            return False

        logging.info(f"Attempting to delete transaction with DB ID: {transaction_id}")

        success = delete_transaction_from_db(self.db_conn, transaction_id)

        if success:
            logging.info(
                f"Successfully deleted transaction ID {transaction_id} from DB."
            )
        return success

    def _delete_transactions_from_csv(
        self, original_indices_to_delete: List[int]
    ) -> bool:
        """Removes transactions by original_index (internal DB ID) and rewrites the CSV."""
        if not hasattr(self, "original_data") or self.original_data.empty:
            QMessageBox.critical(
                self,
                "Data Error",
                "Original transaction data not available for deletion.",
            )
            return False
        if not original_indices_to_delete:
            return True  # Nothing to delete

        # Work on a copy
        df_original = self.original_data.copy()

        # Filter OUT the rows to delete
        rows_to_keep_mask = ~df_original["original_index"].isin(
            original_indices_to_delete
        )
        df_filtered = df_original[rows_to_keep_mask]

        if len(df_filtered) == len(df_original):
            QMessageBox.warning(
                self,
                "Delete Error",
                "Could not find the transaction(s) to delete (original index mismatch).",
            )
            return False  # None of the indices were found

        # Rewrite the CSV with the filtered data
        if self._rewrite_csv(df_filtered):
            QMessageBox.information(
                self,
                "Success",
                f"Successfully deleted {len(original_indices_to_delete)} transaction(s).",
            )
            self.refresh_data()  # Refresh data after successful delete
            return True
        else:
            # Rewrite failed, error message shown in _rewrite_csv
            return False

    @Slot()
    def select_database_file(self):
        """Opens a file dialog to select a new SQLite database file."""
        current_db_dir = os.getcwd()  # Default start directory
        if self.DB_FILE_PATH and os.path.exists(os.path.dirname(self.DB_FILE_PATH)):
            current_db_dir = os.path.dirname(self.DB_FILE_PATH)
        elif self.config and self.config.get(
            "transactions_file"
        ):  # Check config if self.DB_FILE_PATH is not yet set
            config_path = self.config.get("transactions_file")
            if config_path and os.path.exists(os.path.dirname(config_path)):
                current_db_dir = os.path.dirname(config_path)

        fname, _ = QFileDialog.getOpenFileName(
            self,
            "Open Investa Database File",
            current_db_dir,
            "Database Files (*.db *.sqlite *.sqlite3);;All Files (*)",
        )

        if fname:
            if fname != self.DB_FILE_PATH:
                logging.info(f"User selected new database file: {fname}")
                # Close existing connection if open
                if self.db_conn:
                    try:
                        self.db_conn.close()
                        logging.info(
                            f"Closed existing DB connection to: {self.DB_FILE_PATH}"
                        )
                    except Exception as e_close:
                        logging.error(
                            f"Error closing existing DB connection: {e_close}"
                        )
                    self.db_conn = None

                # Attempt to initialize/connect to the new DB
                # initialize_database will create tables if the DB is new/empty.
                new_conn = initialize_database(fname)
                if not new_conn:
                    self.show_error(
                        f"Could not open or initialize selected database:\n{fname}",
                        popup=True,
                        title="Database Error",
                    )
                    # Optionally, try to revert to the previous valid DB path if one existed
                    # For now, we won't auto-revert here, user can re-select.
                    self.DB_FILE_PATH = None  # Mark as no valid DB
                    self.setWindowTitle(f"{self.base_window_title} - (No Database)")
                    self.clear_results()  # Clear UI
                    self.set_status("Error: Failed to open selected database.")
                    return

                self.db_conn = new_conn
                self.DB_FILE_PATH = fname
                self.config["transactions_file"] = (
                    fname  # Update config with new DB path
                )
                # When opening a new DB, it's safer to reset account-specific settings
                # or prompt the user if they want to keep them.
                # For simplicity now, let's reset them.
                self.config["account_currency_map"] = {}
                self.config["selected_accounts"] = []
                self.save_config()  # Save the new DB path and reset account settings

                self.clear_results()  # Clear UI for the new database
                self.setWindowTitle(
                    f"{self.base_window_title} - {os.path.basename(self.DB_FILE_PATH)}"
                )
                self.set_status(
                    f"Opened database: {os.path.basename(self.DB_FILE_PATH)}. Refreshing..."
                )
                self.refresh_data()  # Refresh data from the newly opened/initialized database
            else:
                logging.info(
                    "Selected database file is the same as current. No change."
                )
        else:
            logging.info("Database file selection cancelled by user.")

    @Slot()
    def create_new_database_file(self):
        """Prompts the user to create a new, empty SQLite database file and initializes its schema."""
        start_dir = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
        if not start_dir:  # Fallback if DocumentsLocation is not available
            start_dir = os.getcwd()
        default_filename = DB_FILENAME  # e.g., "investa_transactions.db"

        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Create New Investa Database File",
            os.path.join(start_dir, default_filename),
            "Database Files (*.db *.sqlite *.sqlite3);;All Files (*)",
        )

        if fname:
            # Ensure a .db extension if none provided or if it's different
            if not any(
                fname.lower().endswith(ext) for ext in [".db", ".sqlite", ".sqlite3"]
            ):
                fname += ".db"  # Append .db by default if no valid extension

            logging.info(f"User selected to create new database file at: {fname}")

            if os.path.exists(fname):
                reply = QMessageBox.question(
                    self,
                    "File Exists",
                    f"The file '{os.path.basename(fname)}' already exists.\n"
                    "Do you want to overwrite it? All existing data in that file will be lost.",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No,
                )
                if reply == QMessageBox.No:
                    logging.info(
                        "New database creation cancelled by user (file exists and chose not to overwrite)."
                    )
                    return
                else:  # User chose to overwrite
                    try:
                        os.remove(fname)
                        logging.info(f"Existing file '{fname}' removed for overwrite.")
                    except Exception as e_del:
                        self.show_error(
                            f"Could not remove existing file '{fname}':\n{e_del}",
                            popup=True,
                            title="Error",
                        )
                        return

            # Close existing connection if open
            if self.db_conn:
                try:
                    self.db_conn.close()
                    logging.info(
                        f"Closed existing DB connection to: {self.DB_FILE_PATH or 'previous DB'}"
                    )
                except Exception as e_close:
                    logging.error(f"Error closing existing DB connection: {e_close}")
                self.db_conn = None

            # Initialize the new database (this will create the file and tables)
            new_conn = initialize_database(fname)
            if new_conn:
                self.db_conn = new_conn
                self.DB_FILE_PATH = fname
                self.config["transactions_file"] = (
                    fname  # Update config with the new DB path
                )
                # Reset account-specific configurations for a new database
                self.config["account_currency_map"] = {}
                self.config["selected_accounts"] = []
                self.config["transactions_file_csv_fallback"] = (
                    ""  # Clear any CSV fallback
                )
                self.save_config()

                self.clear_results()  # Clear UI from any previous data
                self.setWindowTitle(
                    f"{self.base_window_title} - {os.path.basename(self.DB_FILE_PATH)}"
                )
                self.set_status(
                    f"New database created: {os.path.basename(fname)}. Ready to add transactions or import."
                )
                self.show_info(
                    f"New database file created successfully:\n{fname}",
                    popup=True,
                    title="Database Created",
                )
                # Do not automatically refresh_data as the DB is empty.
                # User can now add transactions or import.
            else:
                self.show_error(
                    f"Could not create or initialize new database at:\n{fname}",
                    popup=True,
                    title="Database Creation Error",
                )
                # Attempt to revert to a previously known valid DB path from config if possible
                previous_db_path = self.config.get("transactions_file")
                if (
                    previous_db_path
                    and previous_db_path != fname
                    and os.path.exists(previous_db_path)
                ):
                    self.DB_FILE_PATH = previous_db_path
                    self.db_conn = initialize_database(
                        self.DB_FILE_PATH
                    )  # Re-initialize old one
                    if self.db_conn:
                        self.setWindowTitle(
                            f"{self.base_window_title} - {os.path.basename(self.DB_FILE_PATH)}"
                        )
                    else:  # Failed to reopen previous
                        self.DB_FILE_PATH = None
                        self.setWindowTitle(f"{self.base_window_title} - (No Database)")
                else:  # No valid previous path or it was the one that failed
                    self.DB_FILE_PATH = None
                    self.setWindowTitle(f"{self.base_window_title} - (No Database)")
        else:
            logging.info("New database file creation cancelled by user.")

    @Slot()
    def import_transactions_from_csv(self):
        """
        Prompts the user to select a CSV file and imports its transactions
        into the currently open SQLite database.
        """
        if not self.db_conn:
            self.show_warning(
                "Please open or create a database file first before importing from CSV.",
                popup=True,
                title="No Database Open",
            )
            return

        # Suggest starting directory based on last import or Documents
        start_dir = self.config.get("last_csv_import_path", "")
        if not start_dir or not os.path.isdir(start_dir):
            start_dir = (
                QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation)
                or os.getcwd()
            )

        csv_fname, _ = QFileDialog.getOpenFileName(
            self,
            "Select CSV File to Import Transactions",
            start_dir,
            "CSV Files (*.csv);;All Files (*)",
        )

        if csv_fname:
            # Save the directory of the selected CSV for next time
            self.config["last_csv_import_path"] = os.path.dirname(csv_fname)
            self.save_config()  # Save this path preference

            logging.info(f"User selected CSV for import: {csv_fname}")
            confirm_msg = (
                f"This will import transactions from:\n<b>{os.path.basename(csv_fname)}</b>\n\n"
                f"Into the current database:\n<b>{os.path.basename(self.DB_FILE_PATH)}</b>\n\n"
                "Existing transactions in the database will NOT be deleted. "
                "If transactions from this CSV (or similar data) have already been imported, "
                "this may create duplicates.\n\n"
                "It's recommended to import into an empty or carefully managed database.\n\n"
                "Proceed with import?"
            )
            reply = QMessageBox.question(
                self,
                "Confirm CSV Import",
                confirm_msg,
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes,
            )

            if reply == QMessageBox.No:
                logging.info("CSV import cancelled by user.")
                return

            self.set_status(
                f"Importing transactions from {os.path.basename(csv_fname)}..."
            )
            self.set_controls_enabled(False)  # Disable UI during import
            QApplication.processEvents()  # Ensure UI updates

            # Use current account_currency_map and default_currency from app's config for migration
            current_account_map = self.config.get("account_currency_map", {})
            current_default_curr = self.config.get(
                "default_currency",
                (
                    config.DEFAULT_CURRENCY
                    if hasattr(config, "DEFAULT_CURRENCY")
                    else "USD"
                ),
            )

            mig_count, err_count = migrate_csv_to_db(
                csv_fname, self.db_conn, current_account_map, current_default_curr
            )

            self.set_controls_enabled(True)  # Re-enable UI

            if err_count > 0:
                self.show_warning(
                    f"Imported {mig_count} transaction(s) with {err_count} error(s).\nPlease check the application logs for details.",
                    popup=True,
                    title="Import Issues",
                )
            else:
                self.show_info(
                    f"Successfully imported {mig_count} transaction(s) from '{os.path.basename(csv_fname)}'.",
                    popup=True,
                    title="Import Successful",
                )

            final_status_msg = f"Import from '{os.path.basename(csv_fname)}' complete. Migrated: {mig_count}, Errors: {err_count}."
            if (
                mig_count > 0 or err_count > 0
            ):  # Refresh if anything changed or tried to change
                final_status_msg += " Refreshing data..."
            self.set_status(final_status_msg)
            logging.info(final_status_msg)

            if mig_count > 0:  # Only refresh if actual transactions were migrated
                self.refresh_data()  # Refresh data to show newly imported transactions
            elif (
                err_count > 0
            ):  # If only errors, still might be good to ensure UI is in a clean state
                self.set_status(
                    f"Import from '{os.path.basename(csv_fname)}' had {err_count} errors. No data changed."
                )
            else:  # No rows migrated and no errors (e.g. empty CSV)
                self.set_status(
                    f"No transactions imported from '{os.path.basename(csv_fname)}'."
                )

        else:
            logging.info("CSV import file selection cancelled by user.")

    def _run_csv_migration(self, csv_path_to_migrate: str):
        """
        Internal helper to run CSV to DB migration, update UI, and save config.
        This is typically called on first startup if an empty DB and a fallback CSV are found.
        """
        if not self.db_conn:
            self.show_error(
                "Database connection not available for migration.",
                popup=True,
                title="Migration Error",
            )
            logging.error("Migration Error: DB connection not available.")
            self.set_status("Error: DB connection lost before migration.")
            return

        if not os.path.exists(csv_path_to_migrate):
            self.show_error(
                f"CSV file for migration not found:\n{csv_path_to_migrate}",
                popup=True,
                title="Migration Error",
            )
            logging.error(
                f"Migration Error: CSV file '{csv_path_to_migrate}' not found."
            )
            self.set_status(f"Error: Migration CSV not found.")
            return

        logging.info(f"Starting automatic CSV migration from: {csv_path_to_migrate}")
        self.set_status(
            f"Migrating data from {os.path.basename(csv_path_to_migrate)} to database..."
        )
        self.set_controls_enabled(False)
        QApplication.processEvents()

        # Use current account_currency_map and default_currency from app's config for migration
        # These should be the defaults if it's a first run.
        current_account_map = self.config.get("account_currency_map", {})
        current_default_curr = self.config.get(
            "default_currency",
            config.DEFAULT_CURRENCY if hasattr(config, "DEFAULT_CURRENCY") else "USD",
        )

        mig_count, err_count = migrate_csv_to_db(
            csv_path_to_migrate, self.db_conn, current_account_map, current_default_curr
        )

        self.set_controls_enabled(True)

        if err_count > 0:
            self.show_warning(
                f"Migrated {mig_count} transaction(s) with {err_count} error(s) from '{os.path.basename(csv_path_to_migrate)}'.\nPlease check the application logs for details.",
                popup=True,
                title="Migration Issues",
            )
        else:
            self.show_info(
                f"Successfully migrated {mig_count} transaction(s) from '{os.path.basename(csv_path_to_migrate)}' to the database.",
                popup=True,
                title="Migration Successful",
            )

        # Update config: clear the csv_fallback path as migration has been attempted/completed.
        if self.config.get("transactions_file_csv_fallback") == csv_path_to_migrate:
            self.config["transactions_file_csv_fallback"] = ""  # Clear it
            logging.info(
                f"Cleared 'transactions_file_csv_fallback' after migration attempt from '{csv_path_to_migrate}'."
            )
        self.save_config()  # Save updated config

        final_status_msg = f"Migration from '{os.path.basename(csv_path_to_migrate)}' complete. Migrated: {mig_count}, Errors: {err_count}."
        if mig_count > 0 or err_count > 0:
            final_status_msg += " Refreshing data..."
        self.set_status(final_status_msg)
        logging.info(final_status_msg)

        if mig_count > 0:  # Refresh data to show newly migrated transactions
            self.refresh_data()
        elif err_count > 0:
            self.set_status(
                f"Migration from '{os.path.basename(csv_path_to_migrate)}' had {err_count} errors. No data changed."
            )
        else:  # No rows migrated and no errors (e.g. empty CSV)
            self.set_status(
                f"No transactions migrated from '{os.path.basename(csv_path_to_migrate)}'."
            )

    def clear_results(self):
        """Clears all displayed data, resets UI elements, and clears internal state."""
        # (Keep implementation as before, but also clear account lists)
        logging.info("Clearing results display...")
        self.holdings_data = pd.DataFrame()
        self.ignored_data = pd.DataFrame()
        self.summary_metrics_data = {}
        self.account_metrics_data = {}
        self.index_quote_data = {}
        self.last_calc_status = ""
        self.historical_data = pd.DataFrame()
        self.last_hist_twr_factor = np.nan
        self.available_accounts = []  # Clear available accounts
        self.capital_gains_history_data = pd.DataFrame()  # Already present from Phase 2

        self.dividend_history_data = pd.DataFrame()  # Clear dividend data
        # Keep selected_accounts as loaded from config, validation happens on load
        self._update_table_view_with_filtered_columns(pd.DataFrame())
        # Clear PVC tab
        if hasattr(self, "pvc_annual_graph_ax"):
            self.pvc_annual_graph_ax.clear()
            self.pvc_annual_graph_canvas.draw()
            self.pvc_monthly_graph_ax.clear()
            self.pvc_monthly_graph_canvas.draw()
            self.pvc_weekly_graph_ax.clear()
            self.pvc_weekly_graph_canvas.draw()
            self.pvc_annual_table_model.updateData(pd.DataFrame())
            self.pvc_monthly_table_model.updateData(pd.DataFrame())
            self.pvc_weekly_table_model.updateData(pd.DataFrame())
        self.apply_column_visibility()
        self.update_dashboard_summary(pd.DataFrame())  # Pass empty DF on clear
        self.update_account_pie_chart()
        self.update_holdings_pie_chart(pd.DataFrame())
        self.update_performance_graphs(initial=True)
        self.set_status("Ready")
        self._update_table_title(pd.DataFrame())  # Update table title
        self._update_account_button_text()  # Update button text
        self.stock_transactions_table_model.updateData(
            pd.DataFrame()
        )  # Clear log table
        self.cash_transactions_table_model.updateData(pd.DataFrame())  # Clear log table
        self._update_fx_rate_display(self.currency_combo.currentText())
        self.update_header_info(loading=True)
        self._clear_asset_allocation_charts()  # Clear allocation charts
        self._update_capital_gains_display()  # Clear Capital Gains tab
        if hasattr(self, "view_ignored_button"):
            self.view_ignored_button.setEnabled(False)  # Disable when clearing
        self._update_dividend_bar_chart()  # Clear dividend chart
        self._update_dividend_table()  # Clear dividend table

    def _update_all_transaction_tables(self):
        """
        Refreshes all three transaction tables in the 'Transactions' tab
        using the main application data (self.original_data).
        """
        logging.debug("Updating all transaction tables...")
        # This updates the top "management" table
        self._refresh_transactions_view()
        # This updates the bottom two "log" tables (Stock/ETF and $CASH)
        self._update_transaction_log_tables_content()

    def _update_transaction_log_tables_content(self):
        """Populates the stock and cash transaction log tables using self.original_data."""
        logging.debug("Updating transaction log tables...")

        if (
            not hasattr(self, "original_data")
            or self.original_data is None
            or self.original_data.empty
        ):
            logging.info(
                "Original data not available, clearing transaction log tables."
            )
            self.stock_transactions_table_model.updateData(pd.DataFrame())
            self.cash_transactions_table_model.updateData(pd.DataFrame())
            return

        # Ensure required columns exist in original_data
        # self.original_data is all_transactions_df_cleaned_for_logic, which has cleaned names.
        df_for_logs = self.original_data.copy()

        # Filter by selected accounts
        # self.selected_accounts contains the list of accounts to show. If empty, show all.
        if self.selected_accounts and "Account" in df_for_logs.columns:
            logging.debug(
                f"Filtering transaction log for accounts: {self.selected_accounts}"
            )
            df_for_logs = df_for_logs[
                df_for_logs["Account"].isin(self.selected_accounts)
            ]
        elif not self.selected_accounts:
            logging.debug("No accounts selected for log filter, showing all.")
        else:  # self.selected_accounts is not None but "Account" column is missing
            logging.warning(
                "Cannot filter transaction log by account: 'Account' column missing in original_data."
            )

        # Determine column names for symbol and date (should be cleaned names from DB)
        symbol_col_name_to_use = "Symbol"  # Expect cleaned name "Symbol"
        date_col_name_to_use_for_sort = "Date"  # Expect cleaned name "Date"

        if symbol_col_name_to_use not in df_for_logs.columns:
            logging.error(
                f"Missing expected column '{symbol_col_name_to_use}' in original_data. Cannot populate transaction logs."
            )
            self.stock_transactions_table_model.updateData(pd.DataFrame())
            self.cash_transactions_table_model.updateData(pd.DataFrame())
            return

        if date_col_name_to_use_for_sort not in df_for_logs.columns:
            logging.error(
                f"Missing expected column '{date_col_name_to_use_for_sort}' in original_data. Cannot populate transaction logs or sort."
            )
            self.stock_transactions_table_model.updateData(pd.DataFrame())
            self.cash_transactions_table_model.updateData(pd.DataFrame())
            return

        # Get the column index for the date column in the DataFrame passed to the model
        # The DataFrames passed to updateData are slices of self.original_data,
        # so they retain the original column order and names.

        # Filter for stock transactions (anything not $CASH)
        stock_tx_df = df_for_logs[
            df_for_logs[symbol_col_name_to_use] != CASH_SYMBOL_CSV
        ].copy()
        # Filter for cash transactions
        cash_tx_df = df_for_logs[
            df_for_logs[symbol_col_name_to_use] == CASH_SYMBOL_CSV
        ].copy()

        # Ensure Date column is formatted as string "YYYY-MM-DD" for display
        if date_col_name_to_use_for_sort in stock_tx_df.columns:
            stock_tx_df[date_col_name_to_use_for_sort] = pd.to_datetime(
                stock_tx_df[date_col_name_to_use_for_sort], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            stock_tx_df[date_col_name_to_use_for_sort] = stock_tx_df[
                date_col_name_to_use_for_sort
            ].replace(
                "NaT", "-", regex=False
            )  # Handle potential NaT from coerce
        if date_col_name_to_use_for_sort in cash_tx_df.columns:
            cash_tx_df[date_col_name_to_use_for_sort] = pd.to_datetime(
                cash_tx_df[date_col_name_to_use_for_sort], errors="coerce"
            ).dt.strftime("%Y-%m-%d")
            cash_tx_df[date_col_name_to_use_for_sort] = cash_tx_df[
                date_col_name_to_use_for_sort
            ].replace(
                "NaT", "-", regex=False
            )  # Handle potential NaT

        self.stock_transactions_table_model.updateData(stock_tx_df)
        self._apply_column_visibility(
            self.stock_transactions_table_view, "stock_tx_columns"
        )
        self.cash_transactions_table_model.updateData(cash_tx_df)
        self._apply_column_visibility(
            self.cash_transactions_table_view, "cash_tx_columns"
        )

        # --- ADDED: Apply default sort by Date (Descending) ---
        # So, we find the index of `date_col_name_to_use_for_sort` in the respective DataFrames.
        try:
            if (
                not stock_tx_df.empty
                and date_col_name_to_use_for_sort in stock_tx_df.columns
            ):
                stock_date_col_idx = stock_tx_df.columns.get_loc(
                    date_col_name_to_use_for_sort
                )
                self.stock_transactions_table_view.sortByColumn(
                    stock_date_col_idx, Qt.DescendingOrder
                )
            if (
                not cash_tx_df.empty
                and date_col_name_to_use_for_sort in cash_tx_df.columns
            ):
                cash_date_col_idx = cash_tx_df.columns.get_loc(
                    date_col_name_to_use_for_sort
                )
                self.cash_transactions_table_view.sortByColumn(
                    cash_date_col_idx, Qt.DescendingOrder
                )
        except KeyError:
            logging.error(
                f"Could not find column index for '{date_col_name_to_use_for_sort}' in log tables. Cannot apply default sort."
            )
        except Exception as e_sort:
            logging.error(f"Error during log table sort: {e_sort}")
        # --- END ADDED ---

        self.stock_transactions_table_view.resizeColumnsToContents()
        self.cash_transactions_table_view.resizeColumnsToContents()
        # logging.info(
        #     f"Transaction log tables updated. Stock Txs: {len(stock_tx_df)}, Cash Txs: {len(cash_tx_df)}"
        # )

    # Ensure this method is part of the PortfolioApp class and correctly indented
    def _clear_asset_allocation_charts(self):
        """Clears the asset allocation pie charts."""
        if hasattr(self, "asset_type_pie_ax"):
            self.asset_type_pie_ax.clear()
            self.asset_type_pie_ax.axis("on")  # Default to axis 'on' after clearing
            # Clear any previous title or legend explicitly if needed
            # self.asset_type_pie_ax.set_title("")
            # if self.asset_type_pie_ax.legend_ is not None:
            #     self.asset_type_pie_ax.legend_.remove()
            self.asset_type_pie_canvas.draw()

        if hasattr(self, "sector_pie_ax"):  # Clear sector pie chart
            self.sector_pie_ax.clear()
            self.sector_pie_ax.axis("on")
            self.sector_pie_canvas.draw()

        if hasattr(self, "geo_pie_ax"):  # Clear geo pie chart
            self.geo_pie_ax.clear()
            self.geo_pie_ax.axis("on")
            self.geo_pie_canvas.draw()

        if hasattr(self, "industry_pie_ax"):  # ADDED: Clear industry pie chart
            self.industry_pie_ax.clear()
            self.industry_pie_ax.axis("on")
            self.industry_pie_canvas.draw()

    def _update_asset_allocation_charts(self):  # Method name was already correct
        """Populates the asset allocation charts using self.holdings_data."""
        logging.debug(
            "Updating asset allocation charts (Asset Type, Sector, Geography)..."
        )
        # Explicitly set background colors for all asset allocation pie charts
        for fig_attr, ax_attr in [
            ("asset_type_pie_fig", "asset_type_pie_ax"),
            ("sector_pie_fig", "sector_pie_ax"),  # These are in AssetAllocationTab
            ("geo_pie_fig", "geo_pie_ax"),  # which should match main background
            ("industry_pie_fig", "industry_pie_ax"),  #
        ]:
            if hasattr(self, fig_attr) and hasattr(self, ax_attr):
                fig = getattr(self, fig_attr)
                ax = getattr(self, ax_attr)
                if fig:
                    fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
                if ax:
                    ax.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
        self._clear_asset_allocation_charts()  # Clear existing charts first

        if not hasattr(self, "holdings_data") or self.holdings_data.empty:
            logging.info(
                "Holdings data not available, cannot update asset allocation charts."
            )
            if hasattr(self, "asset_type_pie_ax"):
                self.asset_type_pie_ax.axis("off")  # Turn off axis for "No Data" text
                self.asset_type_pie_ax.text(
                    0.5,
                    0.5,
                    "No Data Available",
                    ha="center",
                    va="center",
                    transform=self.asset_type_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.asset_type_pie_canvas.draw()
            if hasattr(self, "sector_pie_ax"):  # Also show No Data for sector chart
                self.sector_pie_ax.axis("off")
                self.sector_pie_ax.text(
                    0.5,
                    0.5,
                    "No Data Available",
                    ha="center",
                    va="center",
                    transform=self.sector_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.sector_pie_canvas.draw()
            if hasattr(self, "geo_pie_ax"):  # Also show No Data for geo chart
                self.geo_pie_ax.axis("off")
                self.geo_pie_ax.text(
                    0.5,
                    0.5,
                    "No Data Available",
                    ha="center",
                    va="center",
                    transform=self.geo_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.geo_pie_canvas.draw()
            if hasattr(self, "industry_pie_ax"):  # ADDED
                self.industry_pie_ax.axis("off")
                self.industry_pie_ax.text(
                    0.5,
                    0.5,
                    "No Data Available",
                    ha="center",
                    va="center",
                    transform=self.industry_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.industry_pie_canvas.draw()
            return

        df_alloc = self.holdings_data.copy()
        display_currency = self.currency_combo.currentText()
        col_defs = get_column_definitions(display_currency)
        value_col_actual = col_defs.get("Mkt Val")

        # --- MOVED: Log df_alloc columns and head AFTER it's defined ---
        logging.debug(
            f"[_update_asset_allocation_charts] Base df_alloc (from self.holdings_data) columns: {df_alloc.columns.tolist()}"
        )

        # Filter df_alloc by selected accounts.
        # self.holdings_data should already be filtered by the worker.
        # However, to be robust or if that assumption is wrong, explicit filtering here is safer.
        df_alloc_filtered = df_alloc.copy()  # Start with potentially scoped data
        if self.selected_accounts and "Account" in df_alloc_filtered.columns:
            # If specific accounts are selected, ensure we only use data from those accounts.
            # The _AGGREGATE_CASH_ACCOUNT_NAME_ row in self.holdings_data is special;
            # it represents the cash sum for the *selected scope*. So, if it's present
            # in self.holdings_data (which is already supposed to be scoped), it should be included.
            account_filter_mask = df_alloc_filtered["Account"].isin(
                self.selected_accounts
            )
            aggregate_cash_mask = (
                df_alloc_filtered["Account"] == _AGGREGATE_CASH_ACCOUNT_NAME_
            )
            df_alloc_filtered = df_alloc_filtered[
                account_filter_mask | aggregate_cash_mask
            ].copy()
            logging.debug(
                f"Asset allocation charts filtered for accounts: {self.selected_accounts}. Resulting df_alloc_filtered shape: {df_alloc_filtered.shape}"
            )
        elif not self.selected_accounts:
            logging.debug(
                "Asset allocation charts using data for all accounts (no specific selection)."
            )
        # df_alloc is now the correctly scoped DataFrame for allocation charts.
        df_alloc = df_alloc_filtered

        logging.debug(
            f"[_update_asset_allocation_charts] Filtered df_alloc columns: {df_alloc.columns.tolist()}"
        )
        if not df_alloc.empty:  # Log head only if not empty
            logging.debug(
                f"[_update_asset_allocation_charts] df_alloc head (first 5 rows):\n{df_alloc.head().to_string()}"
            )
            for col_check in ["Sector", "quoteType", "Country"]:
                logging.debug(
                    f"  Column '{col_check}' in df_alloc: {col_check in df_alloc.columns}"
                )
        # --- END MOVED ---

        if not value_col_actual or value_col_actual not in df_alloc.columns:
            logging.error(
                f"Market Value column '{value_col_actual}' not found for asset allocation."
            )
            if hasattr(self, "asset_type_pie_ax"):
                self.asset_type_pie_ax.axis("off")  # Turn off axis for error text
                self.asset_type_pie_ax.text(
                    0.5,
                    0.5,
                    "Data Error",
                    ha="center",
                    va="center",
                    transform=self.asset_type_pie_ax.transAxes,
                    color=COLOR_LOSS,
                )
                self.asset_type_pie_canvas.draw()
            if hasattr(self, "sector_pie_ax"):  # Also show Data Error for sector chart
                self.sector_pie_ax.axis("off")
                self.sector_pie_ax.text(
                    0.5,
                    0.5,
                    "Data Error",
                    ha="center",
                    va="center",
                    transform=self.sector_pie_ax.transAxes,
                    color=COLOR_LOSS,
                )
                self.sector_pie_canvas.draw()
            if hasattr(self, "geo_pie_ax"):  # Also show Data Error for geo chart
                self.geo_pie_ax.axis("off")
                self.geo_pie_ax.text(
                    0.5,
                    0.5,
                    "Data Error",
                    ha="center",
                    va="center",
                    transform=self.geo_pie_ax.transAxes,
                    color=COLOR_LOSS,
                )
                self.geo_pie_canvas.draw()
            return

        # --- Simple Asset Type Classification ---
        def classify_asset(symbol):
            # Assumes df_alloc has a 'quoteType' column from portfolio_logic
            if symbol == CASH_SYMBOL_CSV or symbol.startswith("Cash ("):
                return "Cash"

            quote_type_series = df_alloc.loc[df_alloc["Symbol"] == symbol, "quoteType"]
            if not quote_type_series.empty:
                quote_type = quote_type_series.iloc[0]
                if quote_type == "STOCK":
                    return "STOCK"
                elif quote_type == "ETF":
                    return "ETF"
                elif quote_type:  # Other known quote types
                    return quote_type.capitalize()
            return "Other Assets"  # Fallback for missing quoteType or unhandled types

        df_alloc["Asset Type"] = df_alloc["Symbol"].apply(classify_asset)
        asset_type_values = df_alloc.groupby("Asset Type", observed=False)[
            value_col_actual
        ].sum()
        # --- FIX: Filter for positive values for pie chart ---
        asset_type_values = asset_type_values[asset_type_values > 1e-3]

        if not asset_type_values.empty:
            # Axis should be 'on' from the _clear_asset_allocation_charts call
            labels = asset_type_values.index.tolist()
            values = asset_type_values.values
            total_value = np.sum(values)

            # Use a predefined color map or generate colors
            cmap_asset = plt.get_cmap("Spectral")
            colors = cmap_asset(np.linspace(0, 1, len(values)))

            wedges, texts, autotexts = self.asset_type_pie_ax.pie(
                values,
                labels=None,  # We'll use a legend or custom labels
                autopct="",  # CHANGED: Remove autopct for internal labels
                startangle=90,
                counterclock=False,
                colors=colors,
                wedgeprops={
                    "edgecolor": self.QCOLOR_BACKGROUND_THEMED.name(),
                    "linewidth": 0.7,
                },  # Edge matches bg
            )
            # plt.setp(autotexts, size=8, weight="bold", color="black") # No longer needed for internal

            # Create legend from labels and values
            legend_labels = [
                f"{l} ({self._get_currency_symbol()}{v:,.0f})"
                for l, v in asset_type_values.items()
            ]
            self.asset_type_pie_ax.legend(
                wedges,
                legend_labels,
                title="Asset Types",
                loc="upper left",  # Anchor point of the legend box
                bbox_to_anchor=(1.10, 1.05),
                fontsize=8,
            )

            # Add percentage labels outside the pie
            for i, wedge in enumerate(wedges):
                ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                x = wedge.r * 1.15 * np.cos(np.deg2rad(ang))  # Position outside
                y = wedge.r * 1.15 * np.sin(np.deg2rad(ang))
                percent = (values[i] / total_value) * 100
                if percent > 0.5:  # Only label significant slices
                    horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                    self.asset_type_pie_ax.text(
                        x,
                        y,
                        f"{percent:.1f}%",
                        ha=horizontalalignment,
                        va="center",
                        fontsize=7,
                        color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                    )

            self.asset_type_pie_fig.subplots_adjust(
                left=0.05, right=0.75
            )  # Adjust to make space for legend
            self.asset_type_pie_canvas.draw()
        else:
            logging.info("No asset type values to plot after filtering.")
            if hasattr(self, "asset_type_pie_ax"):
                self.asset_type_pie_ax.axis(
                    "off"
                )  # Turn off axis for "No Allocation Data" text
                self.asset_type_pie_ax.text(
                    0.5,
                    0.5,
                    "No Allocation Data",
                    ha="center",
                    va="center",
                    transform=self.asset_type_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.asset_type_pie_canvas.draw()

        # --- Sector Allocation Chart ---
        # For now, we'll assume 'Sector' column might exist in self.holdings_data
        # or we'll fetch it. If not, we show a placeholder message.
        if "Sector" in df_alloc.columns:  # Check if 'Sector' column exists
            # Prepare sector data: fill NaNs and empty strings, ensure string type for robust grouping
            df_alloc_for_sector_chart = df_alloc.copy()
            df_alloc_for_sector_chart["Sector"] = (
                df_alloc_for_sector_chart["Sector"]
                .astype(str)
                .fillna("Unknown Sector")
                .str.strip()
            )
            df_alloc_for_sector_chart.loc[
                df_alloc_for_sector_chart["Sector"] == "", "Sector"
            ] = "Unknown Sector"

            sector_values = df_alloc_for_sector_chart.groupby("Sector", observed=False)[
                value_col_actual
            ].sum()
            sector_values = sector_values[sector_values > 1e-3].sort_values(
                ascending=False
            )  # FIX: Only positive values

            if not sector_values.empty:
                self.sector_pie_ax.axis("on")  # Turn axis on for plotting

                # Group small slices into "Other"
                if len(sector_values) > CHART_MAX_SLICES:
                    top_sectors = sector_values.head(CHART_MAX_SLICES - 1)
                    other_value = sector_values.iloc[CHART_MAX_SLICES - 1 :].sum()
                    if other_value > 1e-3:  # Only add "Other" if it's significant
                        top_sectors.loc["Other"] = other_value
                    sector_values_to_plot = top_sectors
                else:
                    sector_values_to_plot = sector_values

                labels = sector_values_to_plot.index.tolist()
                values = sector_values_to_plot.values

                cmap_sector = plt.get_cmap("Spectral")  # CHANGED colormap
                colors_sector = cmap_sector(np.linspace(0, 1, len(values)))

                wedges_s, texts_s, autotexts_s = self.sector_pie_ax.pie(
                    values,
                    labels=None,
                    autopct="",  # CHANGED: Remove autopct for internal labels
                    startangle=90,
                    counterclock=False,
                    colors=colors_sector,
                    wedgeprops={
                        "edgecolor": self.QCOLOR_BACKGROUND_THEMED.name(),
                        "linewidth": 0.7,
                    },
                )
                # plt.setp(autotexts_s, size=8, weight="bold", color="black") # No longer needed

                legend_labels_s = [
                    f"{l} ({self._get_currency_symbol()}{v:,.0f})"
                    for l, v in sector_values_to_plot.items()
                ]
                self.sector_pie_ax.legend(
                    wedges_s,
                    legend_labels_s,
                    title="Sectors",
                    loc="upper left",
                    bbox_to_anchor=(1.1, 1.05),
                    fontsize=8,
                )

                # Add percentage labels outside the pie for sector chart
                total_sector_value = np.sum(values)
                for i, wedge in enumerate(wedges_s):
                    ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                    x = wedge.r * 1.15 * np.cos(np.deg2rad(ang))  # Position outside
                    y = wedge.r * 1.15 * np.sin(np.deg2rad(ang))
                    percent = (values[i] / total_sector_value) * 100
                    if percent > 0.5:  # Only label significant slices
                        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                        self.sector_pie_ax.text(
                            x,
                            y,
                            f"{percent:.1f}%",
                            ha=horizontalalignment,
                            va="center",
                            fontsize=7,
                            color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                        )
                self.sector_pie_fig.subplots_adjust(left=0.05, right=0.75)
                self.sector_pie_canvas.draw()
            else:
                self.sector_pie_ax.axis("off")
                self.sector_pie_ax.text(
                    0.5,
                    0.5,
                    "No Sector Data",
                    ha="center",
                    va="center",
                    transform=self.sector_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.sector_pie_canvas.draw()
        else:
            self.sector_pie_ax.axis("off")
            self.sector_pie_ax.text(
                0.5,
                0.5,
                "Sector Data Unavailable\n(Requires Fundamentals)",
                ha="center",
                va="center",
                transform=self.sector_pie_ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
                wrap=True,
            )
            self.sector_pie_canvas.draw()

        # --- Geographical Allocation Chart ---
        if "Country" in df_alloc.columns:
            df_alloc_for_geo_chart = df_alloc.copy()
            df_alloc_for_geo_chart["Country"] = (  # type: ignore
                df_alloc_for_geo_chart["Country"]
                .astype(str)
                .fillna("Unknown Region")
                .str.strip()
            )
            df_alloc_for_geo_chart.loc[
                df_alloc_for_geo_chart["Country"] == "", "Country"
            ] = "Unknown Region"

            # Special handling for "Cash" country to keep it separate if desired
            # Or map it to a specific region based on display currency, e.g., "Domestic Cash"
            # For now, "Cash" will be its own slice if it exists as a "Country".

            country_values = df_alloc_for_geo_chart.groupby("Country", observed=False)[
                value_col_actual
            ].sum()
            country_values = country_values[country_values > 1e-3].sort_values(
                ascending=False
            )  # FIX: Only positive values

            if not country_values.empty:
                self.geo_pie_ax.axis("on")

                if len(country_values) > CHART_MAX_SLICES:
                    top_countries = country_values.head(CHART_MAX_SLICES - 1)
                    other_value_geo = country_values.iloc[CHART_MAX_SLICES - 1 :].sum()
                    if other_value_geo > 1e-3:
                        top_countries.loc["Other Regions"] = other_value_geo
                    country_values_to_plot = top_countries
                else:
                    country_values_to_plot = country_values

                labels_g = country_values_to_plot.index.tolist()
                values_g = country_values_to_plot.values

                cmap_geo = plt.get_cmap(
                    "Spectral"
                )  # Using Spectral again, or choose another like "Paired"
                colors_geo = cmap_geo(np.linspace(0, 1, len(values_g)))

                wedges_g, _, _ = self.geo_pie_ax.pie(
                    values_g,
                    labels=None,
                    autopct="",
                    startangle=90,
                    counterclock=False,
                    colors=colors_geo,
                    wedgeprops={
                        "edgecolor": self.QCOLOR_BACKGROUND_THEMED.name(),
                        "linewidth": 0.7,
                    },
                )
                legend_labels_g = [
                    f"{l} ({self._get_currency_symbol()}{v:,.0f})"
                    for l, v in country_values_to_plot.items()
                ]
                self.geo_pie_ax.legend(
                    wedges_g,
                    legend_labels_g,
                    title="Geography",
                    loc="upper left",
                    bbox_to_anchor=(1.1, 1.05),
                    fontsize=8,
                )

                total_geo_value = np.sum(values_g)
                for i, wedge in enumerate(wedges_g):
                    ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                    x = wedge.r * 1.15 * np.cos(np.deg2rad(ang))
                    y = wedge.r * 1.15 * np.sin(np.deg2rad(ang))
                    percent = (values_g[i] / total_geo_value) * 100
                    if percent > 0.5:
                        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                        self.geo_pie_ax.text(
                            x,
                            y,
                            f"{percent:.1f}%",
                            ha=horizontalalignment,
                            va="center",
                            fontsize=7,
                            color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                        )
                self.geo_pie_fig.subplots_adjust(left=0.05, right=0.75)
                self.geo_pie_canvas.draw()
            else:  # No country values to plot
                self.geo_pie_ax.axis("off")
                self.geo_pie_ax.text(
                    0.5,
                    0.5,
                    "No Geographical Data",
                    ha="center",
                    va="center",
                    transform=self.geo_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.geo_pie_canvas.draw()
        else:  # "Country" column not in df_alloc
            self.geo_pie_ax.axis("off")
            self.geo_pie_ax.text(
                0.5,
                0.5,
                "Geographical Data Unavailable\n(Requires Fundamentals)",
                ha="center",
                va="center",
                transform=self.geo_pie_ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
                wrap=True,
            )
            self.geo_pie_canvas.draw()

        # --- Industry Allocation Chart (New) ---
        if "Industry" in df_alloc.columns:
            df_alloc_for_industry_chart = df_alloc.copy()
            df_alloc_for_industry_chart["Industry"] = (  # type: ignore
                df_alloc_for_industry_chart["Industry"]
                .astype(str)
                .fillna("Unknown Industry")
                .str.strip()
            )
            df_alloc_for_industry_chart.loc[
                df_alloc_for_industry_chart["Industry"] == "", "Industry"
            ] = "Unknown Industry"

            industry_values = df_alloc_for_industry_chart.groupby(
                "Industry", observed=False
            )[value_col_actual].sum()
            industry_values = industry_values[industry_values > 1e-3].sort_values(
                ascending=False
            )  # FIX: Only positive values

            if not industry_values.empty:
                self.industry_pie_ax.axis("on")

                if len(industry_values) > CHART_MAX_SLICES:
                    top_industries = industry_values.head(CHART_MAX_SLICES - 1)
                    other_value_ind = industry_values.iloc[CHART_MAX_SLICES - 1 :].sum()
                    if other_value_ind > 1e-3:
                        top_industries.loc["Other Industries"] = other_value_ind
                    industry_values_to_plot = top_industries
                else:
                    industry_values_to_plot = industry_values

                labels_i = industry_values_to_plot.index.tolist()
                values_i = industry_values_to_plot.values

                cmap_ind = plt.get_cmap("Spectral")  # Or "tab20", "Set3"
                colors_ind = cmap_ind(np.linspace(0, 1, len(values_i)))

                wedges_i, _, _ = self.industry_pie_ax.pie(
                    values_i,
                    labels=None,
                    autopct="",
                    startangle=90,
                    counterclock=False,
                    colors=colors_ind,
                    wedgeprops={
                        "edgecolor": self.QCOLOR_BACKGROUND_THEMED.name(),
                        "linewidth": 0.7,
                    },
                )

                legend_labels_i = [
                    f"{l} ({self._get_currency_symbol()}{v:,.0f})"
                    for l, v in industry_values_to_plot.items()
                ]
                self.industry_pie_ax.legend(
                    wedges_i,
                    legend_labels_i,
                    title="Industry",
                    loc="upper left",
                    bbox_to_anchor=(1.1, 1.05),
                    fontsize=8,
                )

                total_industry_value = np.sum(values_i)
                for i, wedge in enumerate(wedges_i):
                    ang = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
                    x = wedge.r * 1.15 * np.cos(np.deg2rad(ang))
                    y = wedge.r * 1.15 * np.sin(np.deg2rad(ang))
                    percent = (values_i[i] / total_industry_value) * 100
                    if percent > 0.5:
                        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
                        self.industry_pie_ax.text(
                            x,
                            y,
                            f"{percent:.1f}%",
                            ha=horizontalalignment,
                            va="center",
                            fontsize=7,
                            color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                        )
                self.industry_pie_fig.subplots_adjust(left=0.05, right=0.75)
                self.industry_pie_canvas.draw()
            else:  # No industry values to plot
                self.industry_pie_ax.axis("off")
                self.industry_pie_ax.text(
                    0.5,
                    0.5,
                    "No Industry Data",
                    ha="center",
                    va="center",
                    transform=self.industry_pie_ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.industry_pie_canvas.draw()
        else:  # "Industry" column not in df_alloc
            self.industry_pie_ax.axis("off")
            self.industry_pie_ax.text(
                0.5,
                0.5,
                "Industry Data Unavailable\n(Requires Fundamentals)",
                ha="center",
                va="center",
                transform=self.industry_pie_ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
                wrap=True,
            )
            self.industry_pie_canvas.draw()  # Corrected canvas to draw on

    # --- Filter Change Handlers ---
    @Slot()
    def filter_changed_refresh(self):
        """
        Slot triggered by changes requiring a full data recalculation.

        Connected to 'Currency' combo box and 'Show Closed' checkbox changes.
        Calls `refresh_data`.
        """
        sender = self.sender()
        # Reset initial selection flag if any filter change happens after startup
        self._initial_file_selection = False

        changed_control = "Unknown Filter"
        if sender == self.currency_combo:
            changed_control = "Currency"
            self._ensure_all_columns_in_visibility()
        elif sender == self.show_closed_check:
            changed_control = "'Show Closed' Checkbox"
        logging.info(f"Filter change ({changed_control}) requires full refresh...")
        self.refresh_data()  # Trigger the main refresh function, let worker decide on historical

    # --- UI Update Helpers ---
    def _update_table_view_with_filtered_columns(self, df_source_data):
        """
        Updates the main table view with the provided DataFrame.

        Renames columns from internal names to UI-friendly headers based on the
        current display currency. Applies column visibility settings and adjusts
        column widths.

        Args:
            df_source_data (pd.DataFrame): The DataFrame containing the data to display.
        """
        df_for_table = pd.DataFrame()
        if df_source_data is not None and not df_source_data.empty:
            display_currency = self.currency_combo.currentText()
            col_defs = get_column_definitions(display_currency)
            preferred_ui_cols_order = list(col_defs.keys())
            cols_to_keep_actual = []
            actual_to_ui_map = {}
            for ui_name in preferred_ui_cols_order:
                actual_col_name = col_defs.get(ui_name)
                if actual_col_name and actual_col_name in df_source_data.columns:
                    if actual_col_name not in cols_to_keep_actual:
                        cols_to_keep_actual.append(actual_col_name)
                        actual_to_ui_map[actual_col_name] = ui_name
            if cols_to_keep_actual:
                df_intermediate = df_source_data[cols_to_keep_actual].copy()
                # --- ADDED: Re-attach the 'is_group_header' column if it exists ---
                # This column is used for styling and is dropped by the column selection logic above.
                if "is_group_header" in df_source_data.columns:
                    df_intermediate["is_group_header"] = df_source_data[
                        "is_group_header"
                    ]
                # This column is used for stable group sorting and is also dropped.
                if "group_key" in df_source_data.columns:
                    df_intermediate["group_key"] = df_source_data["group_key"]
                # --- ADDED: Re-attach the 'is_summary_row' column ---
                # This column is used for styling the totals row.
                if "is_summary_row" in df_source_data.columns:
                    df_intermediate["is_summary_row"] = df_source_data["is_summary_row"]
                # --- END ADDED ---
                # --- END ADDED ---
                df_for_table = df_intermediate.rename(columns=actual_to_ui_map)
        self.table_model.updateData(df_for_table)
        # --- ADDED: Apply row visibility for groups ---
        self._apply_row_visibility_for_grouping()
        # --- END ADDED ---
        # --- MODIFIED: Resize columns for both tables ---
        if not df_for_table.empty:
            self.table_view.resizeColumnsToContents()
            self.frozen_table_view.resizeColumnsToContents()

        if not df_for_table.empty:
            self.table_view.resizeColumnsToContents()
            # The fixed-width setting logic has been removed to allow dynamic resizing.
            self.table_view.horizontalHeader().setStretchLastSection(False)
        else:
            try:
                self.table_view.horizontalHeader().setStretchLastSection(False)
            except Exception as e:  # pragma: no cover
                logging.warning(f"Warning: Could not unset stretch last section: {e}")

    def _apply_grouping(self, df: pd.DataFrame) -> pd.DataFrame:
        """Groups the DataFrame by sector and adds header/summary rows."""
        if not self.group_by_sector_check.isChecked() or "Sector" not in df.columns:
            return df

        df_grouped = df.copy()
        # --- MODIFIED: More robust cleaning of the 'Sector' column ---
        # This ensures that any non-string, NaN, None, or empty string values
        # are consistently handled and grouped under "Unknown", preventing
        # holdings from being missed by the groupby operation.
        if "Sector" not in df_grouped.columns:
            # This should not happen if data comes from portfolio_logic, but as a safeguard:
            df_grouped["Sector"] = "Unknown"

        # Coerce to string, then fill any resulting 'nan' or empty strings.
        df_grouped["Sector"] = df_grouped["Sector"].astype(str).fillna("Unknown")
        df_grouped.loc[df_grouped["Sector"].str.strip() == "", "Sector"] = "Unknown"
        df_grouped.loc[
            df_grouped["Sector"].str.upper().isin(["NAN", "NONE", "<NA>"]), "Sector"
        ] = "Unknown"
        # --- END MODIFICATION ---

        grouped_data = []

        currency = self._get_currency_symbol(get_name=True)
        all_col_defs = get_column_definitions(currency)
        cols_to_sum_ui = [
            "Mkt Val",
            # "% of Total" is handled separately for groups to avoid summation errors
            "Day Chg",
            "Unreal. G/L",
            "Real. G/L",
            "Divs",
            "Fees",
            "Total G/L",
            "FX G/L",
            "Est. Income",
            "Cost Basis",
            "Total Buy Cost",
        ]
        cols_to_sum_actual = [
            all_col_defs[ui_name]
            for ui_name in cols_to_sum_ui
            if ui_name in all_col_defs and all_col_defs[ui_name] in df_grouped.columns
        ]

        for sector, group in df_grouped.groupby("Sector", sort=True):
            is_expanded = self.group_expansion_states.get(sector, True)
            indicator = "▾" if is_expanded else "▸"

            summed_values = {
                col: group[col].sum()
                for col in cols_to_sum_actual
                if pd.api.types.is_numeric_dtype(group[col])
            }

            group_header_data = {
                "Symbol": f"{indicator} {sector}",
                "is_group_header": True,
                "group_key": sector,
            }

            # --- ADDED: Calculate % of Total for the group ---
            total_portfolio_value = self.summary_metrics_data.get("market_value")
            mkt_val_col_actual = all_col_defs.get("Mkt Val")

            if (
                total_portfolio_value
                and total_portfolio_value > 0
                and mkt_val_col_actual
                and mkt_val_col_actual in summed_values
            ):
                group_header_data["pct_of_total"] = (
                    summed_values[mkt_val_col_actual] / total_portfolio_value
                ) * 100.0

            group_header_data.update(summed_values)
            group_header_data.update(
                self._calculate_summary_percentages(group, summed_values)
            )

            grouped_data.append(pd.DataFrame([group_header_data]))

            group_with_key = group.copy()
            group_with_key["group_key"] = sector
            grouped_data.append(group_with_key)

        return (
            pd.concat(grouped_data, ignore_index=True)
            if grouped_data
            else pd.DataFrame()
        )

    def _add_summary_row(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculates and appends a 'TOTALS' summary row to the DataFrame."""
        if df.empty:
            return df

        df_with_summary = df.copy()

        # Exclude group headers from summation
        if "is_group_header" in df_with_summary.columns:
            data_rows = df_with_summary[df_with_summary["is_group_header"] != True]
        else:
            data_rows = df_with_summary

        if data_rows.empty:
            return df  # Return original if only headers exist

        currency = self._get_currency_symbol(get_name=True)
        all_col_defs = get_column_definitions(currency)
        cols_to_sum_ui = [
            "Mkt Val",
            "% of Total",
            "Day Chg",
            "Unreal. G/L",
            "Real. G/L",
            "Divs",
            "Fees",
            "Total G/L",
            "FX G/L",
            "Est. Income",
            "Cost Basis",
            "Total Buy Cost",
        ]
        cols_to_sum_actual = [
            all_col_defs[ui_name]
            for ui_name in cols_to_sum_ui
            if ui_name in all_col_defs and all_col_defs[ui_name] in data_rows.columns
        ]

        summary_row = pd.Series(index=df.columns, dtype=object)
        summary_row["Symbol"] = "TOTALS"
        summary_row["is_summary_row"] = True

        summed_values_total = {
            col: data_rows[col].sum()
            for col in cols_to_sum_actual
            if pd.api.types.is_numeric_dtype(data_rows[col])
        }

        for col, total in summed_values_total.items():
            summary_row[col] = total

        total_percentages = self._calculate_summary_percentages(
            data_rows, summed_values_total
        )
        for col, pct_val in total_percentages.items():
            summary_row[col] = pct_val

        return pd.concat(
            [df_with_summary, pd.DataFrame([summary_row])], ignore_index=True
        )

    def _get_filtered_data(self, group_by_sector=False, update_view=False):
        """
        Filters and processes the main holdings DataFrame through a pipeline of helper methods.

        Args:
            group_by_sector (bool): If True, groups the data by sector.
            update_view (bool): If True, updates the table view directly.

        Returns:
            pd.DataFrame: The fully processed DataFrame ready for display.
        """
        if not isinstance(self.holdings_data, pd.DataFrame) or self.holdings_data.empty:
            return pd.DataFrame()

        # Start the pipeline
        df_processed = self._apply_base_filters(self.holdings_data)
        df_processed = self._apply_text_filters(df_processed)

        if group_by_sector:
            df_processed = self._apply_grouping(df_processed)

        df_processed = self._add_summary_row(df_processed)

        # This path is no longer used but kept for safety

        if update_view:
            self._update_table_view_with_filtered_columns(df_processed)
            self.apply_column_visibility()
            return None  # Explicitly return None as view is updated

        return df_processed

    def _update_fx_rate_display(self, display_currency):
        """
        Updates the FX rate label in the status bar.

        Shows the rate between the base currency (from config) and the selected
        display currency, if they differ. Uses the rate stored in `self.summary_metrics_data`.

        Args:
            display_currency (str): The currently selected display currency code.
        """
        show_rate = False
        rate_text = ""
        # Get the default/base currency from the config
        base_currency = self.config.get("default_currency", "USD")  # <-- Use config

        if display_currency != base_currency:
            rate = None
            # The summary metrics should contain the rate relative to the default currency
            if self.summary_metrics_data and isinstance(
                self.summary_metrics_data, dict
            ):
                rate = self.summary_metrics_data.get(
                    "exchange_rate_to_display"
                )  # This key holds BASE->DISPLAY rate

            if rate is not None and pd.notna(rate):
                try:
                    rate_float = float(rate)
                    # Display the rate: 1 BASE = X DISPLAY
                    rate_text = f"<b>FX:</b> 1 {base_currency} = {display_currency} {abs(rate_float):,.4f}"
                    show_rate = True
                except (ValueError, TypeError):
                    rate_text = f"<b>FX:</b> Invalid Rate"
                    show_rate = True
            else:
                rate_text = f"<b>FX:</b> ({base_currency}->{display_currency}) Unavailable"  # Indicate which rate is missing
                show_rate = True

        self.exchange_rate_display_label.setText(rate_text)
        self.exchange_rate_display_label.setVisible(show_rate)

    # --- New Helper: Update Table Title ---
    def _update_table_title(self, df_display_filtered: pd.DataFrame):
        """Updates the title above the main holdings table to reflect the current scope."""
        title_right_label = getattr(self, "table_title_label_right", None)
        title_left_label = getattr(self, "table_title_label_left", None)

        if not title_right_label or not title_left_label:
            return

        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)
        # --- MODIFIED: Count only stock rows, excluding cash, group headers, and summary rows. ---
        num_rows_displayed = 0
        if not df_display_filtered.empty:
            # Create a boolean mask for rows that are NOT special rows
            stock_rows_mask = pd.Series(True, index=df_display_filtered.index)

            if "is_group_header" in df_display_filtered.columns:
                stock_rows_mask &= df_display_filtered["is_group_header"] != True

            if "is_summary_row" in df_display_filtered.columns:
                stock_rows_mask &= df_display_filtered["is_summary_row"] != True

            if "Symbol" in df_display_filtered.columns:
                # is_cash_symbol is imported from finutils
                stock_rows_mask &= ~df_display_filtered["Symbol"].apply(
                    lambda x: is_cash_symbol(str(x))
                )

            num_rows_displayed = int(stock_rows_mask.sum())

        title_right_text = f"Holdings Detail ({num_rows_displayed} stocks shown)"
        scope_text = ""

        if not self.available_accounts:
            scope_text = "Scope: N/A (Load Data)"
        elif not self.selected_accounts or num_selected == num_available:
            scope_text = f"Scope: All Accounts ({num_available})"
        elif num_selected == 1:
            scope_text = f"Scope: Account '{self.selected_accounts[0]}'"
        else:
            # Limit displayed accounts if list is very long
            max_accounts_in_title = 4
            if num_selected <= max_accounts_in_title:
                scope_text = f"Scope: Accounts ({num_selected}/{num_available})"
                # scope_text = f"Scope: {', '.join(self.selected_accounts)}" # Alternative: List names
            else:
                scope_text = f"Scope: {num_selected} / {num_available} Accounts"
                # scope_text = f"Scope: {', '.join(self.selected_accounts[:max_accounts_in_title])}, ..." # Alternative

        title_right_label.setText(title_right_text)
        title_left_label.setText(scope_text)

    # --- Filtering and Grouping Helpers for _get_filtered_data ---

    def _calculate_summary_percentages(
        self, df_subset: pd.DataFrame, summed_values: dict
    ) -> dict:
        """Calculates percentage metrics for a given DataFrame subset and its summed values."""
        percentages = {}
        currency = self._get_currency_symbol(get_name=True)

        # Define column names
        mkt_val_col = f"Market Value ({currency})"
        day_chg_col = f"Day Change ({currency})"
        unreal_gain_col = f"Unreal. Gain ({currency})"
        cost_basis_col = f"Cost Basis ({currency})"
        total_gain_col = f"Total Gain ({currency})"
        total_buy_cost_col = f"Total Buy Cost ({currency})"
        est_income_col = f"Est. Ann. Income ({currency})"
        irr_col = "IRR (%)"

        # Get summed values
        sum_mkt_val = summed_values.get(mkt_val_col, 0.0)
        sum_day_chg = summed_values.get(day_chg_col, 0.0)
        sum_unreal_gain = summed_values.get(unreal_gain_col, 0.0)
        sum_cost_basis = summed_values.get(cost_basis_col, 0.0)
        sum_total_gain = summed_values.get(total_gain_col, 0.0)
        sum_total_buy_cost = summed_values.get(total_buy_cost_col, 0.0)
        sum_est_income = summed_values.get(est_income_col, 0.0)

        def safe_division_pct(numerator, denominator):
            if abs(denominator) > 1e-9:
                return (numerator / denominator) * 100.0
            elif abs(numerator) < 1e-9:
                return 0.0
            else:
                return np.inf

        # Day Chg %
        prev_day_val = sum_mkt_val - sum_day_chg
        percentages["Day Change %"] = safe_division_pct(sum_day_chg, prev_day_val)

        # Unreal. G/L %
        percentages["Unreal. Gain %"] = safe_division_pct(
            sum_unreal_gain, sum_cost_basis
        )

        # Total Ret. %
        is_cash_sector = False
        if "Sector" in df_subset.columns and not df_subset.empty:
            unique_sectors = df_subset["Sector"].dropna().unique()
            if len(unique_sectors) == 1 and unique_sectors[0] == "Cash":
                is_cash_sector = True

        if is_cash_sector:
            percentages["Total Return %"] = np.nan
        else:
            # Use cost basis of held assets + cash as the denominator for a more intuitive return on capital.
            percentages["Total Return %"] = safe_division_pct(
                sum_total_gain, sum_cost_basis
            )

        # Yield (Cost) % & Yield (Mkt) %
        percentages["Div. Yield (Cost) %"] = safe_division_pct(
            sum_est_income, sum_cost_basis
        )
        percentages["Div. Yield (Current) %"] = safe_division_pct(
            sum_est_income, sum_mkt_val
        )

        # IRR (%) - Weighted Average by Market Value
        if (
            irr_col in df_subset.columns
            and mkt_val_col in df_subset.columns
            and abs(sum_mkt_val) > 1e-9
        ):
            weighted_irr_sum = (
                df_subset[irr_col].fillna(0) * df_subset[mkt_val_col].fillna(0)
            ).sum()
            percentages[irr_col] = weighted_irr_sum / sum_mkt_val
        else:
            percentages[irr_col] = np.nan

        return percentages

    def _apply_base_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies account and 'show closed' filters to the holdings data."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()

        df_filtered = df.copy()

        # --- FIX: Re-enable UI-level account filtering ---
        # The worker filters the data, but self.holdings_data holds the result of that
        # worker run. If the user changes the account selection in the UI *without*
        # clicking "Update Accounts", this filter ensures that subsequent UI-only
        # actions (like text filtering or grouping) still respect the *current* UI selection.
        if (
            hasattr(self, "selected_accounts")
            and self.selected_accounts
            and hasattr(self, "available_accounts")
            and len(self.selected_accounts) != len(self.available_accounts)
            and "Account" in df_filtered.columns
        ):
            account_filter_mask = df_filtered["Account"].isin(self.selected_accounts)

            # --- ADD THIS LINE ---
            # Always include the special aggregated cash row
            cash_mask = df_filtered["Account"] == _AGGREGATE_CASH_ACCOUNT_NAME_
            # --- END ADD ---

            # Keep special rows (group headers, totals)
            special_rows_mask = pd.Series(False, index=df_filtered.index)
            if "is_group_header" in df_filtered.columns:
                special_rows_mask |= df_filtered["is_group_header"] == True
            if "is_summary_row" in df_filtered.columns:
                special_rows_mask |= df_filtered["is_summary_row"] == True

            # Keep rows that match the account filter OR are special rows OR are the cash row
            df_filtered = df_filtered[
                account_filter_mask | special_rows_mask | cash_mask
            ].copy()  # <-- MODIFIED
        # --- END FIX ---

        # Filter by 'Show Closed'
        show_closed = self.show_closed_check.isChecked()
        if (
            not show_closed
            and "Quantity" in df_filtered.columns
            and "Symbol" in df_filtered.columns
        ):
            try:
                numeric_quantity = pd.to_numeric(
                    df_filtered["Quantity"], errors="coerce"
                ).fillna(0)
                keep_mask = (numeric_quantity.abs() > 1e-9) | (
                    df_filtered["Symbol"].apply(is_cash_symbol)
                )
                df_filtered = df_filtered[keep_mask]
            except Exception as e:
                logging.warning(f"Warning: Error filtering 'Show Closed': {e}")

        return df_filtered

    def _apply_text_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies live text filters from the UI to the DataFrame."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return df

        df_filtered = df.copy()

        symbol_filter_text = ""
        account_filter_text = ""
        if hasattr(self, "filter_symbol_table_edit"):
            symbol_filter_text = self.filter_symbol_table_edit.text().strip()
        if hasattr(self, "filter_account_table_edit"):
            account_filter_text = self.filter_account_table_edit.text().strip()

        if symbol_filter_text and "Symbol" in df_filtered.columns:
            try:
                cash_display_symbol = (
                    f"Cash ({self._get_currency_symbol(get_name=True)})"
                )
                symbol_mask = (
                    df_filtered["Symbol"]
                    .astype(str)
                    .str.contains(symbol_filter_text, case=False, na=False)
                )
                if "CASH" in symbol_filter_text.upper():
                    symbol_mask |= (
                        df_filtered["Symbol"]
                        .astype(str)
                        .str.contains(cash_display_symbol, case=False, na=False)
                    )
                df_filtered = df_filtered[symbol_mask]
            except Exception as e_sym_filt:
                logging.warning(
                    f"Warning: Error applying symbol table filter: {e_sym_filt}"
                )

        if account_filter_text and "Account" in df_filtered.columns:
            try:
                df_filtered = df_filtered[
                    df_filtered["Account"]
                    .astype(str)
                    .str.contains(account_filter_text, case=False, na=False)
                ]
            except Exception as e_acc_filt:
                logging.warning(
                    f"Warning: Error applying account table filter: {e_acc_filt}"
                )

        return df_filtered

    def _mark_data_as_stale(self):
        """Updates the UI to indicate that a manual refresh is required."""
        if hasattr(self, "stale_data_indicator_label"):
            self.stale_data_indicator_label.setVisible(True)
        if hasattr(self, "status_label"):
            self.set_status(
                "Data changed. Click 'Refresh All' to update portfolio calculations."
            )
        logging.info("UI marked as stale, refresh required.")

    def _mark_data_as_fresh(self):
        """Resets the UI state after a refresh, hiding the 'stale data' indicator."""
        if hasattr(self, "stale_data_indicator_label"):
            self.stale_data_indicator_label.setVisible(False)
        # The status label will be updated by the refresh process itself,
        # so we don't need to reset it here.
        logging.info("UI marked as fresh after data refresh.")

    # --- End New Helper ---

    # --- Control Enabling/Disabling ---
    def set_controls_enabled(self, enabled: bool):
        """
        Enables or disables main UI controls during calculations.

        Args:
            enabled (bool): True to enable controls, False to disable.
        """
        controls_to_toggle = [
            self.account_select_button,  # Use new button
            self.update_accounts_button,  # <-- ADDED Update Accounts button
            self.currency_combo,
            self.show_closed_check,
            self.graph_start_date_edit,
            self.graph_end_date_edit,
            self.benchmark_select_button,
            self.graph_update_button,
            self.refresh_button,
            # Add fundamental lookup controls here
            self.lookup_symbol_edit,
            self.lookup_button,
        ]
        for control in controls_to_toggle:
            try:
                control.setEnabled(enabled)
            except AttributeError:
                pass
        self.setCursor(Qt.WaitCursor if not enabled else Qt.ArrowCursor)

    # --- Signal Handlers from Worker ---
    @Slot(  # MODIFIED: Signature matches WorkerSignals.result (15 args)
        dict,
        pd.DataFrame,
        dict,
        dict,
        pd.DataFrame,
        dict,
        dict,
        set,
        dict,
        pd.DataFrame,
        pd.DataFrame,  # capital_gains_history_df
        pd.DataFrame,  # correlation_matrix_df
        dict,  # factor_analysis_results
        dict,  # scenario_analysis_result
    )
    def handle_results(
        self,
        summary_metrics,
        holdings_df,
        account_metrics,
        index_quotes,
        full_historical_data_df,
        hist_prices_adj,
        hist_fx,
        combined_ignored_indices,
        combined_ignored_reasons,
        dividend_history_df,
        capital_gains_history_df,
        correlation_matrix_df,
        factor_analysis_results,
        scenario_analysis_result,
    ):
        """
        Slot to process results received from the worker.
        This orchestrator method now contains the logic to store, process, and
        prepare all data before triggering the final UI component updates.
        """
        # logging.info("HANDLE_RESULTS: Orchestrator entered.")  # Changed to INFO for visibility

        try:
            # --- Part 1: Store Worker Data ---
            # logging.info("HANDLE_RESULTS: Storing worker data...")
            portfolio_status = summary_metrics.pop("status_msg", "Status Unknown")
            historical_status = summary_metrics.pop(
                "historical_status_msg", "Status Unknown"
            )
            self.last_calc_status = f"{portfolio_status} | {historical_status}"
            self.last_hist_twr_factor = np.nan
            if "|||TWR_FACTOR:" in historical_status:  # Parse total TWR from worker
                try:
                    twr_factor_str = historical_status.split("|||TWR_FACTOR:")[1]
                    self.last_hist_twr_factor = float(twr_factor_str)
                    logging.info(
                        f"Parsed total TWR factor from worker: {self.last_hist_twr_factor}"
                    )
                except (ValueError, IndexError) as e:
                    logging.warning(
                        f"Could not parse TWR factor from status string: {e}"
                    )
                    self.last_hist_twr_factor = np.nan
            self.summary_metrics_data = summary_metrics if summary_metrics else {}
            self.holdings_data = (
                holdings_df if holdings_df is not None else pd.DataFrame()
            )

            # --- ADDED: Calculate '% of Total' column ---
            if not self.holdings_data.empty:
                total_portfolio_value = self.summary_metrics_data.get("market_value")
                display_currency = self.currency_combo.currentText()
                mkt_val_col = f"Market Value ({display_currency})"

                if (
                    total_portfolio_value
                    and total_portfolio_value > 0
                    and mkt_val_col in self.holdings_data.columns
                ):
                    self.holdings_data["pct_of_total"] = (
                        self.holdings_data[mkt_val_col] / total_portfolio_value
                    ) * 100.0
                else:
                    self.holdings_data["pct_of_total"] = 0.0
            # --- END ADDED ---

            self.periodic_returns_data: Dict[str, pd.DataFrame] = {}
            self.account_metrics_data = account_metrics if account_metrics else {}
            self.index_quote_data = index_quotes if index_quotes else {}
            self.dividend_history_data = (
                dividend_history_df
                if dividend_history_df is not None
                else pd.DataFrame()
            )
            self.capital_gains_history_data = (
                capital_gains_history_df
                if capital_gains_history_df is not None
                else pd.DataFrame()
            )
            self.correlation_matrix_df = (
                correlation_matrix_df
                if correlation_matrix_df is not None
                else pd.DataFrame()
            )
            self.factor_analysis_results = (
                factor_analysis_results if factor_analysis_results is not None else {}
            )
            self.scenario_analysis_result = (
                scenario_analysis_result if scenario_analysis_result is not None else {}
            )
            self.full_historical_data = (
                full_historical_data_df
                if full_historical_data_df is not None
                else pd.DataFrame()
            )
            # Ensure 'Portfolio Value' column exists or rename 'value' if present (just in case logic differs)
            if not self.full_historical_data.empty:
                 if "Portfolio Value" not in self.full_historical_data.columns and "value" in self.full_historical_data.columns:
                     self.full_historical_data.rename(columns={"value": "Portfolio Value"}, inplace=True)

            self.historical_prices_yf_adjusted = (
                hist_prices_adj if hist_prices_adj is not None else {}
            )
            self.historical_fx_yf = hist_fx if hist_fx is not None else {}
            self.combined_ignored_indices = (
                combined_ignored_indices if combined_ignored_indices else []
            )
            self.combined_ignored_reasons = (
                combined_ignored_reasons if combined_ignored_reasons else []
            )
            

            self.ignored_data = pd.DataFrame()
            if (
                combined_ignored_indices
                and hasattr(self, "original_data")
                and not self.original_data.empty
            ):
                logging.info(
                    f"Processing {len(combined_ignored_indices)} ignored row indices..."
                )
                try:
                    if "original_index" in self.original_data.columns:
                        indices_to_check = {
                            int(i) for i in combined_ignored_indices if pd.notna(i)
                        }
                        valid_indices_mask = self.original_data["original_index"].isin(
                            indices_to_check
                        )
                        ignored_rows_df = self.original_data[valid_indices_mask].copy()
                        if not ignored_rows_df.empty:
                            reasons_mapped = (
                                ignored_rows_df["original_index"]
                                .map(combined_ignored_reasons)
                                .fillna("Unknown Reason")
                            )
                            ignored_rows_df["Reason Ignored"] = reasons_mapped
                            self.ignored_data = ignored_rows_df.sort_values(
                                by="original_index"
                            )
                except Exception as e_ignored:
                    logging.error(
                        f"Error constructing ignored_data DataFrame: {e_ignored}",
                        exc_info=True,
                    )
                    self.ignored_data = pd.DataFrame()

            # --- Part 2: Process Historical and Periodic Data ---
            # logging.info("HANDLE_RESULTS: Processing historical and periodic data...")
            if (
                isinstance(self.full_historical_data, pd.DataFrame)
                and not self.full_historical_data.empty
            ):
                selected_benchmark_tickers_for_periodic = [
                    BENCHMARK_MAPPING.get(name)
                    for name in self.selected_benchmarks
                    if BENCHMARK_MAPPING.get(name)
                ]
                self.periodic_returns_data = calculate_periodic_returns(
                    self.full_historical_data, selected_benchmark_tickers_for_periodic
                )
                intervals_map_abs_val = {"Y": "YE", "M": "ME", "W": "W-FRI"}
                self.periodic_value_changes_data = {}
                if "Portfolio Value" in self.full_historical_data.columns:
                    for interval_key, freq_code in intervals_map_abs_val.items():
                        period_end_values = (
                            self.full_historical_data["Portfolio Value"]
                            .resample(freq_code)
                            .last()
                        )
                        period_start_values = period_end_values.shift(1)
                        period_net_flows = (
                            self.full_historical_data["net_flow"]
                            .resample(freq_code)
                            .sum()
                            .fillna(0.0)
                            if "net_flow" in self.full_historical_data.columns
                            else pd.Series(0.0, index=period_end_values.index)
                        )
                        df_aligned = pd.DataFrame(
                            {
                                "end_value": period_end_values,
                                "start_value": period_start_values,
                                "net_flow": period_net_flows,
                            }
                        )
                        value_change_series = (
                            df_aligned["end_value"]
                            - df_aligned["start_value"]
                            - df_aligned["net_flow"]
                        )
                        self.periodic_value_changes_data[interval_key] = pd.DataFrame(
                            {"Portfolio Value Change": value_change_series}
                        ).dropna(subset=["Portfolio Value Change"])
                daily_df = self.full_historical_data.copy()
                if "daily_gain" in daily_df.columns:
                    self.periodic_value_changes_data["D"] = pd.DataFrame(
                        {"Portfolio Value Change": daily_df["daily_gain"]}
                    )
                daily_returns_df = pd.DataFrame(index=daily_df.index)
                if "daily_return" in daily_df.columns:
                    daily_returns_df["Portfolio D-Return"] = (
                        daily_df["daily_return"] * 100.0
                    )
                for yf_ticker in selected_benchmark_tickers_for_periodic:
                    price_col = f"{yf_ticker} Price"
                    if price_col in daily_df.columns:
                        daily_returns_df[f"{yf_ticker} D-Return"] = (
                            daily_df[price_col].pct_change(fill_method=None) * 100.0
                        )
                self.periodic_returns_data["D"] = daily_returns_df

            self.historical_data = pd.DataFrame()
            if (
                isinstance(self.full_historical_data, pd.DataFrame)
                and not self.full_historical_data.empty
            ):
                plot_start_date = self.graph_start_date_edit.date().toPython()
                plot_end_date = self.graph_end_date_edit.date().toPython()
                pd_start = pd.Timestamp(plot_start_date)
                pd_end = pd.Timestamp(plot_end_date)
                temp_df = self.full_historical_data
                if not isinstance(temp_df.index, pd.DatetimeIndex):
                    temp_df.index = pd.to_datetime(temp_df.index)
                if temp_df.index.tz is not None:
                    temp_df.index = temp_df.index.tz_localize(None)
                filtered_by_date_df = temp_df.loc[pd_start:pd_end].copy()

                logging.debug(
                    f"Filtered historical data to {len(filtered_by_date_df)} rows for date range."
                )

                # --- Re-normalize all accumulated gain columns to start at 1.0 for the selected period ---
                if not filtered_by_date_df.empty:
                    gain_cols = [
                        col
                        for col in filtered_by_date_df.columns
                        if "Accumulated Gain" in col
                    ]
                    logging.debug(
                        f"Re-normalizing accumulated gain columns: {gain_cols}"
                    )
                    for col in gain_cols:
                        # Find the first valid (non-NaN and non-zero) value in the series to use as the base factor
                        series_for_norm = filtered_by_date_df[col].dropna()
                        # Filter out zeros effectively
                        series_for_norm = series_for_norm[abs(series_for_norm) > 1e-9]

                        if not series_for_norm.empty:
                            start_factor = series_for_norm.iloc[0]
                            # start_factor is guaranteed non-zero by the filter above
                            # Divide the whole column by the start_factor to re-base it to 1.0
                            filtered_by_date_df[col] = (
                                filtered_by_date_df[col] / start_factor
                            )
                            logging.debug(
                                f"  Normalized '{col}' using start factor: {start_factor:.4f}"
                            )
                        else:
                            # If there are no valid values (all NaNs or Zeros), we can't normalize.
                            # We leave it as is (likely zeros or NaNs) or set to NaN if strictly required.
                            # If it was all zeros, dividing by ? is impossible. 0 is fine.
                            logging.debug(
                                f"  No valid non-zero data to normalize for '{col}' in selected range."
                            )
                            # If we strictly want to hide invalid data:
                            if filtered_by_date_df[col].dropna().empty:
                                 # It was all NaNs
                                 pass
                            else:
                                 # It was all Zeros (or NaNs)
                                 # Normalized 0 is 0. So leaving it is fine.
                                 pass
                            # If there are no valid values in the filtered range, the column is all NaN.
                            logging.debug(
                                f"  No data to normalize for '{col}' in the selected range."
                            )
                # --- End Re-normalization ---

                # Resample based on interval
                interval = "D"  # Interval is now hardcoded to Daily
                if interval in ["W", "M"]:  # This block will now be skipped
                    logging.info(f"Resampling historical data to interval: {interval}")
                    resample_freq = "W-FRI" if interval == "W" else "ME"

                    # Define what to aggregate. We need 'Portfolio Value' and benchmark prices.
                    agg_dict = {}
                    if "Portfolio Value" in filtered_by_date_df.columns:
                        agg_dict["Portfolio Value"] = "last"

                    selected_benchmark_tickers = [
                        BENCHMARK_MAPPING.get(name)
                        for name in self.selected_benchmarks
                        if BENCHMARK_MAPPING.get(name)
                    ]
                    for ticker in selected_benchmark_tickers:
                        price_col = f"{ticker} Price"
                        if price_col in filtered_by_date_df.columns:
                            agg_dict[price_col] = "last"

                    if not agg_dict:
                        logging.warning(
                            "No columns to aggregate for resampling. Using simple .last()."
                        )
                        resampled_df = filtered_by_date_df.resample(
                            resample_freq
                        ).last()
                    else:
                        resampled_df = filtered_by_date_df.resample(resample_freq).agg(
                            agg_dict
                        )
                        resampled_df.dropna(how="all", inplace=True)

                        # Recalculate 'Portfolio Accumulated Gain'
                        if (
                            "Portfolio Value" in resampled_df.columns
                            and not resampled_df["Portfolio Value"].dropna().empty
                        ):
                            portfolio_returns = resampled_df[
                                "Portfolio Value"
                            ].pct_change()
                            portfolio_gain_factors = 1 + portfolio_returns.fillna(0.0)
                            resampled_df["Portfolio Accumulated Gain"] = (
                                portfolio_gain_factors.cumprod()
                            )
                            if not resampled_df.empty:
                                resampled_df.iloc[
                                    0,
                                    resampled_df.columns.get_loc(
                                        "Portfolio Accumulated Gain"
                                    ),
                                ] = np.nan

                        # Recalculate benchmark accumulated gains
                        for ticker in selected_benchmark_tickers:
                            price_col = f"{ticker} Price"
                            accum_col = f"{ticker} Accumulated Gain"
                            if (
                                price_col in resampled_df.columns
                                and not resampled_df[price_col].dropna().empty
                            ):
                                bench_returns = resampled_df[price_col].pct_change()
                                bench_gain_factors = 1 + bench_returns.fillna(0.0)
                                resampled_df[accum_col] = bench_gain_factors.cumprod()
                                if not resampled_df.empty:
                                    resampled_df.iloc[
                                        0, resampled_df.columns.get_loc(accum_col)
                                    ] = np.nan
                            else:
                                resampled_df[accum_col] = np.nan  # Ensure column exists

                    self.historical_data = resampled_df
                else:
                    self.historical_data = filtered_by_date_df

            # --- Part 3: Update Available Accounts ---
            # logging.info("HANDLE_RESULTS: Updating available accounts...")
            available_accounts_from_backend = self.summary_metrics_data.get(
                "_available_accounts", []
            )
            if available_accounts_from_backend and isinstance(
                available_accounts_from_backend, list
            ):
                self.available_accounts = available_accounts_from_backend
            elif (
                not self.holdings_data.empty and "Account" in self.holdings_data.columns
            ):
                self.available_accounts = sorted(
                    self.holdings_data["Account"].unique().tolist()
                )
            else:
                self.available_accounts = []
            if self.selected_accounts:
                self.selected_accounts = [
                    acc
                    for acc in self.selected_accounts
                    if acc in self.available_accounts
                ]
                if not self.selected_accounts and self.available_accounts:
                    self.selected_accounts = []
            self._update_account_button_text()

            # --- Part 4: Trigger UI Component Updates ---
            # logging.info(
            #     "HANDLE_RESULTS: Calling _update_ui_components_after_calculation..."
            # )
            # --- ADDED: Update Drawdown Chart (Moved here to use filtered historical_data) ---
            # Calculate drawdown on FULL history to preserve global peak context
            if not self.full_historical_data.empty and "Portfolio Value" in self.full_historical_data.columns:
                 full_drawdown = calculate_drawdown_series(self.full_historical_data["Portfolio Value"])
                 
                 # Now slice to the selected date range
                 if hasattr(self, "historical_data") and not self.historical_data.empty:
                     # Align indices
                     # Ensure indices are datetime
                     if not isinstance(full_drawdown.index, pd.DatetimeIndex):
                         full_drawdown.index = pd.to_datetime(full_drawdown.index)
                     
                     start_date = self.historical_data.index.min()
                     end_date = self.historical_data.index.max()
                     
                     sliced_drawdown = full_drawdown.loc[start_date:end_date]
                     self._update_drawdown_chart(sliced_drawdown)
                 else:
                     self._update_drawdown_chart(full_drawdown)
            else:
                 # Clear chart if no data
                 self._update_drawdown_chart(pd.Series(dtype=float))
            # --- END ADDED ---

            # --- ADDED: Populate intraday symbol combo ---
            self._populate_intraday_symbol_combo()
            self._update_ui_components_after_calculation()
            # logging.info(
            #     "HANDLE_RESULTS: Finished _update_ui_components_after_calculation."
            # )
            # logging.info("HANDLE_RESULTS: Successfully processed and updated UI.")

        except Exception as e:
            logging.critical(f"CRITICAL ERROR in handle_results: {e}", exc_info=True)
            # Try to set error status even if other UI updates failed
            try:
                self._update_rebalancing_tab()
                self.show_error(f"Error processing results: {e}")
            except Exception as e_status:
                logging.error(
                    f"Failed to set status_label in handle_results error handler: {e_status}"
                )
            # It's important that calculation_finished still runs to re-enable UI
            # self.calculation_finished(f"Error in handle_results: {e}") # This might be redundant if finished signal is always processed

        # logging.info("HANDLE_RESULTS: Exiting.")

    @Slot(str)
    def handle_error(self, error_message):
        """
        Slot to handle error messages received from the PortfolioCalculatorWorker.
        Logs the error and updates the status bar.
        Args:
            error_message (str): The error description string.
        """
        logging.info(f"HANDLE_ERROR: Slot entered with message: {error_message}")
        try:
            self.is_calculating = False  # Ensure this is reset
            logging.error(  # Changed to error for more visibility
                f"--- Calculation/Fetch Error Reported by Worker ---\n{error_message}\n--- End Error Report ---"
            )
            # Ensure status update even if label missing
            base_msg = (
                error_message.split("|||")[0]
                if isinstance(error_message, str)
                else str(error_message)
            )
            if hasattr(self, "status_label") and self.status_label:
                self.set_status(f"Error: {base_msg}")
            else:
                logging.warning("status_label not available in handle_error")
            # calculation_finished will be called by the 'finished' signal separately.
            # Do not call it directly here as it might lead to double execution or race conditions.
            # self.calculation_finished(error_message) # Pass the error message to calculation_finished
        except Exception as e:
            logging.critical(
                f"CRITICAL ERROR in handle_error itself: {e}", exc_info=True
            )
            # Fallback UI updates if possible
            if hasattr(self, "status_label") and self.status_label:
                self.set_status("Critical error handling worker error.")
            if hasattr(self, "set_controls_enabled"):
                self.set_controls_enabled(True)  # Try to re-enable controls

    def show_status_popup(self, status_message):
        """(Currently unused) Placeholder for potentially showing status popups."""
        # (Keep implementation as before)
        # ... (no changes to this method's internal logic)

    @Slot()
    def calculation_finished(
        self, error_message: Optional[str] = None
    ):  # Add optional error_message
        # logging.info(
        #     f"CALC_FINISHED: Slot entered. Error message from worker: {error_message}"
        # )  # Changed to INFO
        """
        Slot called when the worker thread finishes (success or error).
        Re-enables UI controls and updates the status bar with the final status message.
        """
        try:
            self.is_calculating = False
            logging.info(
                f"Worker thread finished. Error message received by calculation_finished: {error_message}"
            )
            self.set_controls_enabled(True)

            final_status_text = "Finished (Status Unknown)"

            network_fetch_error_detected = False
            if error_message and (
                "DNSError" in error_message
                or "Could not resolve host" in error_message
                or "Failed to fetch" in error_message.lower()
                or "Failed to perform, curl" in error_message
            ):
                network_fetch_error_detected = True
            elif self.last_calc_status and (
                "DNSError" in self.last_calc_status
                or "Could not resolve host" in self.last_calc_status
                or "Fetch Error" in self.last_calc_status
                or "Failed to perform, curl" in self.last_calc_status
            ):
                network_fetch_error_detected = True

            if network_fetch_error_detected:
                final_status_text = "Error: Could not connect to market data provider. Check network/DNS."
                QMessageBox.warning(
                    self,
                    "Network Error",
                    "Could not connect to the market data provider (e.g., Yahoo Finance).\n"
                    "Please check your internet connection, DNS settings, or firewall.\n"
                    "Market data will be incomplete or unavailable.",
                )
            elif self.last_calc_status:
                cleaned_status = self.last_calc_status.split("|||TWR_FACTOR:")[
                    0
                ].strip()
                if "Error" in cleaned_status or "Crit" in cleaned_status:
                    final_status_text = "Finished (with calculation errors)"
                else:
                    now_str = QDateTime.currentDateTime().toString("hh:mm:ss")
                    final_status_text = f"Finished ({now_str})"
            elif error_message:
                final_status_text = "Finished (with errors)"

            if hasattr(self, "status_label") and self.status_label:
                self.set_status(final_status_text)
            # logging.info(f"CALC_FINISHED: Status label set to: '{final_status_text}'")

            if error_message and (
                isinstance(error_message, str)
                and (
                    "fundamental" in error_message.lower()
                    or "fundamentals" in error_message.lower()
                )
            ):
                if hasattr(self, "lookup_symbol_edit"):
                    self.lookup_symbol_edit.setEnabled(True)
                if hasattr(self, "lookup_button"):
                    self.lookup_button.setEnabled(True)

            if hasattr(self, "progress_bar"):
                self.progress_bar.setVisible(False)
                # logging.info("CALC_FINISHED: Progress bar hidden.")
            # logging.info("CALC_FINISHED: Slot exited successfully.")

        except Exception as e:
            logging.critical(
                f"CRITICAL ERROR in calculation_finished: {e}", exc_info=True
            )
            # Ensure UI is reset to a usable state even if calculation_finished itself errors
            self.is_calculating = False
            try:
                self.set_controls_enabled(True)
                if hasattr(self, "status_label") and self.status_label:
                    self.set_status(f"Critical error in finish step: {e}")
                if hasattr(self, "progress_bar"):
                    self.progress_bar.setVisible(False)
            except Exception as e_reset:
                logging.error(
                    f"Failed to reset UI in calculation_finished error handler: {e_reset}"
                )

    @Slot(int)
    def update_progress(self, percent):
        """Updates the progress bar and status label."""
        if not hasattr(self, "progress_bar"):
            return

        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)

        self.progress_bar.setValue(percent)
        # Optionally update status label too
        self.set_status(f"Calculating historical data... {percent}%")

    def _update_periodic_bar_charts(self):
        """Updates the weekly, monthly, and annual return bar charts."""
        logging.debug(
            "--- Entering _update_periodic_bar_charts ---"
        )  # Changed to DEBUG

        intervals_config = {
            "Y": {
                "data_key": "Y",
                "ax": self.annual_bar_ax,
                "canvas": self.annual_bar_canvas,
                "title": "Annual Returns",  # Title used for logging/errors
                "spinbox": self.pvc_annual_spinbox,  # Reference to the spinbox
                "date_format": "%Y",
            },
            "M": {
                "data_key": "M",
                "ax": self.monthly_bar_ax,
                "canvas": self.monthly_bar_canvas,
                "title": "Monthly Returns",
                "spinbox": self.pvc_monthly_spinbox,
                "date_format": "%Y-%m",
            },
            "W": {
                "data_key": "W",
                "ax": self.weekly_bar_ax,
                "canvas": self.weekly_bar_canvas,
                "title": "Weekly Returns",
                "spinbox": self.pvc_weekly_spinbox,
                "date_format": "%Y-%m-%d",
            },
            "D": {
                "data_key": "D",
                "ax": self.daily_bar_ax,
                "canvas": self.daily_bar_canvas,
                "title": "Daily Returns",
                "spinbox": self.pvc_daily_spinbox,
                "date_format": "%Y-%m-%d",
            },
        }

        # Get portfolio and benchmark column names based on current selection
        portfolio_return_base = "Portfolio"  # Base name used in calculate_periodic_returns renaming  # This should be fine as portfolio_logic.py likely standardizes this.

        # For benchmarks, we need their YF tickers to match columns from calculate_periodic_returns
        benchmark_yf_tickers_for_cols = [
            BENCHMARK_MAPPING.get(name)
            for name in self.selected_benchmarks
            if BENCHMARK_MAPPING.get(name)
        ]

        for config in intervals_config.values():
            ax = config["ax"]
            canvas = config["canvas"]
            interval_key = config["data_key"]
            ax.clear()  # Clear previous plot

            # Explicitly set background colors based on current theme
            # Bar charts are in BarChartsFrame, which should match main background
            fig = ax.get_figure()
            if fig:
                fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
            # Use the input-specific background for the axes plotting area
            ax.patch.set_facecolor(
                self.QCOLOR_BACKGROUND_THEMED.name()  # Match dashboard background)
            )

            returns_df = self.periodic_returns_data.get(interval_key)

            # --- FILTER: Remove weekends for daily interval (Returns Graph) ---
            if interval_key == "D" and returns_df is not None and not returns_df.empty:
                if isinstance(returns_df.index, pd.DatetimeIndex):
                    returns_df = returns_df[returns_df.index.dayofweek < 5]
                else:
                    try:
                        temp_index = pd.to_datetime(returns_df.index)
                        returns_df = returns_df[temp_index.dayofweek < 5]
                    except Exception:
                        pass
            # --- END FILTER ---

            # --- ADDED: Log the DataFrame being used ---
            logging.debug(f"Plotting interval '{interval_key}':")
            if isinstance(returns_df, pd.DataFrame):
                logging.debug(
                    f"  DataFrame shape: {returns_df.shape}, IsEmpty: {returns_df.empty}"
                )
            else:
                logging.debug(f"  Data is not a DataFrame (Type: {type(returns_df)})")
            # --- END ADDED ---

            if returns_df is None or returns_df.empty:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                ax.set_title(config["title"], fontsize=9, weight="bold")
                canvas.draw()
                continue

            # Select relevant columns and limit bars
            portfolio_col_name = f"{portfolio_return_base} {interval_key}-Return"
            # Construct column names using YF tickers for data access
            benchmark_col_names = [
                f"{yf_ticker} {interval_key}-Return"
                for yf_ticker in benchmark_yf_tickers_for_cols
            ]
            cols_to_plot = [
                portfolio_col_name
            ] + benchmark_col_names  # These are ticker-based names
            valid_cols = [col for col in cols_to_plot if col in returns_df.columns]

            if not valid_cols:
                ax.text(
                    0.5,
                    0.5,
                    "No Data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                ax.set_title(config["title"], fontsize=9, weight="bold")
                canvas.draw()
                continue

            # --- Get number of periods from spinbox ---
            try:
                num_periods = config["spinbox"].value()
            except Exception:
                num_periods = 10  # Fallback default
            logging.debug(f"  Using num_periods = {num_periods} from spinbox.")
            plot_data = (
                returns_df[valid_cols].tail(num_periods).copy()
            )  # Use value from spinbox

            # Plotting
            n_series = len(plot_data.columns)
            bar_width = 0.9 / n_series  # Adjust width based on number of series
            index = np.arange(len(plot_data))

            # --- Store x-axis labels before clearing ticks ---
            x_tick_labels = plot_data.index.strftime(config["date_format"])
            # --- End Store x-axis labels ---

            # Define colors for bars
            # Use the first color for the portfolio, the rest for benchmarks
            all_colors = [
                "red",  # Portfolio
                "blue",  # 1st Benchmark
                "green",  # 2nd Benchmark
                "orange",
                "purple",
                "brown",
                "magenta",
                "cyan",
            ]
            portfolio_color = all_colors[0]
            color_map = {portfolio_col_name: portfolio_color}
            for i, bm_col in enumerate(benchmark_col_names):
                if bm_col in plot_data.columns:
                    # Map the YF ticker based column name to its display name for legend and color assignment
                    # Find which display_name corresponds to this bm_col (which uses yf_ticker)
                    display_name_for_bm = ""
                    for dn, yft in BENCHMARK_MAPPING.items():
                        if f"{yft} {interval_key}-Return" == bm_col:
                            display_name_for_bm = dn
                            break

                    color_map[bm_col] = all_colors[(i + 1) % len(all_colors)]

            for i, col in enumerate(plot_data.columns):
                offset = (i - (n_series - 1) / 2) * bar_width
                current_bar_data = plot_data[col].fillna(0.0)
                # If the interval is Annual ('Y'), assume the data might be a factor (e.g., 0.15 for 15%)
                # and multiply by 100 to convert to percentage points (e.g., 15.0).
                bars = ax.bar(
                    index + offset,
                    current_bar_data,  # Use potentially scaled data
                    bar_width,
                    # For label, if it's a benchmark, find its display name. If portfolio, use "Portfolio".
                    label=(
                        next(
                            (
                                dn
                                for dn, yft in BENCHMARK_MAPPING.items()
                                if f"{yft} {interval_key}-Return" == col
                            ),
                            col.replace(f" {interval_key}-Return", ""),
                        )
                        if col != portfolio_col_name
                        else portfolio_return_base  # Use "Portfolio" for the portfolio bar
                    ),
                    # label=col.replace(
                    #     f" {interval_key}-Return", ""
                    # ),  # Clean label for legend
                    color=color_map.get(
                        col, "gray"
                    ),  # Use mapped color, fallback to gray
                )
                # Add value labels on top/bottom of bars (optional, can get crowded)
                # ax.bar_label(bars, fmt='%.1f%%', padding=2, fontsize=6, rotation=90)

            # Formatting
            # ax.set_ylabel("Return (%)", fontsize=8)
            # ax.set_title(config["title"], fontsize=9, weight="bold") # <-- REMOVED Title from plot axes
            ax.set_xticks([])
            ax.set_yticks([])
            ax.yaxis.set_major_formatter(
                mtick.PercentFormatter(xmax=100.0)
            )  # Format Y axis as %
            # --- ADDED: Explicitly set xlim based on the number of data points ---
            if not plot_data.empty:
                ax.set_xlim(-0.5, len(plot_data) - 0.5)
            else:
                # Default for an empty plot, though "No Data" text should be shown
                ax.set_xlim(-0.5, 0.5)
            # --- END ADDED ---
            ax.grid(
                True, axis="y", linestyle="--", linewidth=0.5, color=COLOR_BORDER_LIGHT
            )
            ax.axhline(0, color=COLOR_BORDER_DARK, linewidth=0.6)  # Zero line

            # Legend
            if (
                n_series > 1
                and interval_key == "Y"  # Use interval_key which is 'Y', 'M', 'W'
            ):  # Only show legend for Annual chart
                ax.legend(
                    fontsize=7,
                    loc="upper left",  # Change location to bottom left
                    bbox_to_anchor=(0, 1.25),  # Anchor at the bottom left corner
                    ncol=n_series,
                )

            # --- Add mplcursors Tooltip ---
            if MPLCURSORS_AVAILABLE:
                try:
                    # Use lambda to capture the correct x_tick_labels for this specific axis
                    cursor = mplcursors.cursor(
                        ax.containers, hover=mplcursors.HoverMode.Transient
                    )  # Use Transient hover

                    @cursor.connect("add")
                    def on_add_bar(sel, labels=x_tick_labels):  # Capture labels
                        bar_index = int(sel.index)
                        series_label = sel.artist.get_label()
                        if bar_index < len(labels):
                            period_label = labels[bar_index]
                        else:
                            period_label = "Unknown Period"  # Fallback

                        # FIX: Get height from sel.target[1] (y-coordinate) for bars
                        return_value = sel.target[1]
                        annotation_text = f"{period_label}\n{series_label}\nReturn: {return_value:+.2f}%"

                        sel.annotation.set_text(annotation_text)
                        sel.annotation.get_bbox_patch().set(
                            facecolor="lightyellow", alpha=0.9, edgecolor="gray"
                        )
                        sel.annotation.set_fontsize(8)

                except Exception as e_cursor_bar:
                    logging.error(
                        f"Error activating mplcursors for {config['title']} bar chart: {e_cursor_bar}"
                    )
            # --- End mplcursors ---

            # Style tweaks - Hide all spines (axes lines)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_visible(False)
            ax.spines["left"].set_visible(False)
            # ax.tick_params(axis="x", colors=COLOR_TEXT_SECONDARY, labelsize=7)
            # ax.tick_params(axis="y", colors=COLOR_TEXT_SECONDARY, labelsize=7)

            fig = config["ax"].get_figure()
            fig.tight_layout(pad=0.2)
            canvas.draw()

    def _update_dividend_bar_chart(self):
        """Updates the dividend history bar chart based on selected period and count."""
        logging.debug("Updating dividend bar chart...")
        ax = self.dividend_bar_ax
        canvas = self.dividend_bar_canvas
        ax.clear()

        # Explicitly set background colors based on current theme
        fig = ax.get_figure()
        if fig:
            fig.patch.set_facecolor(
                self.QCOLOR_BACKGROUND_THEMED.name()
            )  # Main background for figure
        ax.patch.set_facecolor(
            self.QCOLOR_BACKGROUND_THEMED.name()  # Match dashboard background
        )  # Input-like background for axes

        if (
            not hasattr(self, "dividend_history_data")
            or self.dividend_history_data.empty
            or not hasattr(
                self, "dividend_summary_table_model"
            )  # Check if summary table model exists
        ):
            logging.debug(
                "  _update_dividend_bar_chart: No dividend_history_data or empty."
            )
            ax.text(
                0.5,
                0.5,
                "No Dividend Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            ax.set_title("Dividend History", fontsize=9, weight="bold")
            canvas.draw()
            self._update_dividend_summary_table(
                pd.Series(dtype=float)
            )  # Clear summary table
            return
        logging.debug(
            f"  _update_dividend_bar_chart: self.dividend_history_data (shape {self.dividend_history_data.shape}):"
        )
        if not self.dividend_history_data.empty:
            logging.debug(f"    Head:\n{self.dividend_history_data.head().to_string()}")

        if (
            not hasattr(self, "dividend_history_data")
            or self.dividend_history_data.empty
        ):

            # This case is handled above, but defensive check
            return

        df_dividends = self.dividend_history_data.copy()
        if (
            "Date" not in df_dividends.columns
            or "DividendAmountDisplayCurrency" not in df_dividends.columns
        ):
            logging.error("Dividend data missing required Date or Amount columns.")
            ax.text(
                0.5,
                0.5,
                "Invalid Dividend Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            ax.set_title("Dividend History", fontsize=9, weight="bold")
            canvas.draw()
            self._update_dividend_summary_table(
                pd.Series(dtype=float)
            )  # Clear summary table
            return

        df_dividends["Date"] = pd.to_datetime(df_dividends["Date"])
        df_dividends.set_index("Date", inplace=True)

        period_type = self.dividend_period_combo.currentText()
        num_periods_to_show = self.dividend_periods_spinbox.value()
        display_currency_symbol = self._get_currency_symbol()

        resample_freq = "YE"  # Annual
        date_format_str = "%Y"
        if period_type == "Quarterly":
            resample_freq = "QE"
            date_format_str = "%Y-Q%q"
        elif period_type == "Monthly":
            resample_freq = "ME"
            date_format_str = "%Y-%m"

        try:
            # Resample and sum dividends
            aggregated_dividends = (
                df_dividends["DividendAmountDisplayCurrency"]
                .resample(resample_freq)
                .sum()
                .dropna()
            )
            logging.debug(
                f"    Aggregated dividends before positive filter (summed & dropna):\n{aggregated_dividends.to_string()}"
            )
            aggregated_dividends = aggregated_dividends[
                aggregated_dividends > 1e-9
            ]  # Keep only positive sums
            # logging.debug(
            #     f"    Aggregated dividends after positive filter (>1e-9):\n{aggregated_dividends.to_string()}"
            # )

            if aggregated_dividends.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No Dividends for {period_type} Periods",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                plot_data_for_table = pd.Series(dtype=float)  # Empty series for table
            else:
                plot_data = aggregated_dividends.tail(num_periods_to_show)
                if plot_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No Dividends for Last {num_periods_to_show} {period_type} Periods",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color=COLOR_TEXT_SECONDARY,
                    )
                    plot_data_for_table = pd.Series(
                        dtype=float
                    )  # Empty series for table
                else:
                    # Ensure plot_data_for_table is initialized before potentially being used
                    plot_data_for_table = pd.Series(dtype=float)
                    # --- MODIFIED: Generate x_labels for quarterly directly ---
                    if period_type == "Quarterly":
                        x_labels = [
                            f"{dt.year}-Q{dt.quarter}" for dt in plot_data.index  # type: ignore
                        ]
                    else:
                        x_labels = plot_data.index.strftime(date_format_str)
                    # --- END MODIFICATION ---
                    bars = ax.bar(
                        x_labels, plot_data.values, color=COLOR_ACCENT_TEAL, width=0.6
                    )

                    # Add value labels on top of bars
                    for bar in bars:
                        yval = bar.get_height()
                        if (
                            pd.notna(yval) and abs(yval) > 1e-9
                        ):  # Only label non-zero bars
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                yval + (plot_data.max() * 0.01),  # Slight offset
                                f"{display_currency_symbol}{yval:,.0f}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                            )

                    ax.yaxis.set_major_formatter(
                        mtick.FuncFormatter(
                            lambda x, p: f"{display_currency_symbol}{x:,.0f}"
                        )
                    )
                    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    plot_data_for_table = plot_data  # Data for summary table

        except Exception as e:
            logging.error(f"Error generating dividend bar chart: {e}")
            traceback.print_exc()  # Add traceback
            ax.text(
                0.5,
                0.5,
                "Error Plotting Dividends",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_LOSS,
            )

        self._update_capital_gains_summary_table(plot_data_for_table)
        scope_display_label = self._get_scope_label_for_charts()
        ax.set_title(
            f"{scope_display_label} - {period_type} Dividend Totals ({display_currency_symbol})",
            fontsize=9,
            weight="bold",
        )
        ax.set_xlabel("")  # Clear x-axis label as dates are on ticks
        ax.set_ylabel(f"Total Dividends ({display_currency_symbol})", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color=COLOR_BORDER_LIGHT)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.dividend_bar_fig.tight_layout(pad=0.5)
        canvas.draw()

        # Update the summary table with the plotted data
        self._update_dividend_summary_table(plot_data_for_table)

    def _plot_pvc_graph(self, ax, canvas, interval_key, num_periods):
        """Helper to plot a single graph for the Asset Change tab."""
        ax.clear()
        fig = ax.get_figure()
        if fig:
            fig.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())
        ax.patch.set_facecolor(self.QCOLOR_BACKGROUND_THEMED.name())

        returns_df = self.periodic_returns_data.get(interval_key)
        # --- ADDED: Clear previous mplcursors if they exist for this axis ---
        if (
            interval_key == "Y"
            and hasattr(self, "pvc_annual_cursor")
            and self.pvc_annual_cursor
        ):
            self.pvc_annual_cursor.remove()
        elif (
            interval_key == "M"
            and hasattr(self, "pvc_monthly_cursor")
            and self.pvc_monthly_cursor
        ):
            self.pvc_monthly_cursor.remove()
        elif (
            interval_key == "W"
            and hasattr(self, "pvc_weekly_cursor")
            and self.pvc_weekly_cursor
        ):
            self.pvc_weekly_cursor.remove()
        # --- END ADDED ---
        value_change_df = self.periodic_value_changes_data.get(interval_key)

        if value_change_df is None or value_change_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            canvas.draw()
            return

        portfolio_value_change_col_name = (
            "Portfolio Value Change"  # Column name from new dict
        )
        if portfolio_value_change_col_name not in value_change_df.columns:
            ax.text(
                0.5,
                0.5,
                "Portfolio Return Data Missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            canvas.draw()
            return

        plot_data = (
            value_change_df[[portfolio_value_change_col_name]].tail(num_periods).copy()
        )
        # --- FILTER: Remove weekends for daily interval (Graph) ---
        if interval_key == "D" and not plot_data.empty:
             if isinstance(plot_data.index, pd.DatetimeIndex):
                 plot_data = plot_data[plot_data.index.dayofweek < 5]
             else:
                 try:
                     temp_index = pd.to_datetime(plot_data.index)
                     plot_data = plot_data[temp_index.dayofweek < 5]
                 except Exception:
                     pass # Ignore filtering errors
        # --- END FILTER ---

        if plot_data.empty:
            ax.text(
                0.5,
                0.5,
                "No Data for Selected Periods",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            canvas.draw()
            return

        date_format_map = {"Y": "%Y", "M": "%Y-%m", "W": "%Y-%m-%d"}
        x_tick_labels = plot_data.index.strftime(
            date_format_map.get(interval_key, "%Y-%m-%d")
        )

        bar_colors = [
            (
                self.QCOLOR_GAIN_THEMED.name()
                if val >= 0
                else self.QCOLOR_LOSS_THEMED.name()
            )
            for val in plot_data[portfolio_value_change_col_name]
        ]

        bars = ax.bar(
            np.arange(len(plot_data)),  # Use numerical index for x-axis
            plot_data[portfolio_value_change_col_name],
            color=bar_colors,
            width=0.9,  # Increased bar width
        )

        # Format Y axis as currency
        currency_symbol_for_axis = self._get_currency_symbol()
        ax.yaxis.set_major_formatter(
            mtick.FuncFormatter(lambda x, p: f"{currency_symbol_for_axis}{x:,.0f}")
        )

        ax.set_xticks(np.arange(len(plot_data)))  # Set tick positions
        ax.set_xticklabels(x_tick_labels)  # Set tick labels
        ax.tick_params(
            axis="x",
            labelrotation=45,
            labelsize=7,
            colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name(),
        )
        ax.tick_params(
            axis="y", labelsize=7, colors=self.QCOLOR_TEXT_SECONDARY_THEMED.name()
        )
        ax.grid(
            True,
            axis="y",
            linestyle="--",
            linewidth=0.5,
            color=self.QCOLOR_BORDER_THEMED.name(),
        )
        ax.axhline(0, color=self.QCOLOR_BORDER_THEMED.name(), linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(self.QCOLOR_BORDER_THEMED.name())
        ax.spines["left"].set_color(self.QCOLOR_BORDER_THEMED.name())

        # --- ADDED: mplcursors for hover annotations ---
        if MPLCURSORS_AVAILABLE and not plot_data.empty:
            try:
                # Define the callback function within the scope to capture x_tick_labels and currency_symbol_for_axis
                def on_add_pvc_bar(
                    sel,
                    local_x_labels=x_tick_labels,
                    local_currency_sym=currency_symbol_for_axis,
                ):
                    bar_index = int(
                        sel.index
                    )  # sel.index is the index of the bar in the container
                    period_label = (
                        local_x_labels[bar_index]
                        if bar_index < len(local_x_labels)
                        else "Unknown Period"
                    )
                    value = sel.artist.patches[
                        sel.index
                    ].get_height()  # Height of the specific bar

                    annotation_text = (
                        f"{period_label}\nValue: {local_currency_sym}{value:,.0f}"
                    )
                    sel.annotation.set_text(annotation_text)
                    sel.annotation.get_bbox_patch().set(
                        facecolor="lightyellow", alpha=0.9, edgecolor="gray"
                    )
                    sel.annotation.set_fontsize(8)

                cursor = mplcursors.cursor(bars, hover=mplcursors.HoverMode.Transient)
                cursor.connect("add", on_add_pvc_bar)

                # Store the cursor object to keep it alive
                if interval_key == "Y":
                    self.pvc_annual_cursor = cursor
                elif interval_key == "M":
                    self.pvc_monthly_cursor = cursor
                elif interval_key == "W":
                    self.pvc_weekly_cursor = cursor

            except Exception as e_cursor_pvc:
                logging.error(
                    f"Error activating mplcursors for PVC graph (Interval: {interval_key}): {e_cursor_pvc}"
                )
        # --- END ADDED ---

        fig.tight_layout(pad=0.5)
        canvas.draw()

    def _update_pvc_table(
        self, table_model: PandasModel, interval_key: str, num_periods: int
    ):
        """Helper to update a single table in the Asset Change tab."""
        original_index_name = None  # Initialize to None
        percent_returns_df_full = self.periodic_returns_data.get(interval_key)
        value_changes_df_full = self.periodic_value_changes_data.get(interval_key)

        if (percent_returns_df_full is None or percent_returns_df_full.empty) and (
            value_changes_df_full is None or value_changes_df_full.empty
        ):
            table_model.updateData(pd.DataFrame())
            return

        # Select data for the last N periods
        df_percent_returns_table = pd.DataFrame()
        if percent_returns_df_full is not None and not percent_returns_df_full.empty:
            df_percent_returns_table = percent_returns_df_full.tail(num_periods).copy()

        df_value_changes_table = pd.DataFrame()
        if value_changes_df_full is not None and not value_changes_df_full.empty:
            df_value_changes_table = value_changes_df_full.tail(num_periods).copy()

        # Combine the dataframes based on index (Period)
        if not df_percent_returns_table.empty and not df_value_changes_table.empty:
            df_for_table = pd.merge(
                df_value_changes_table,
                df_percent_returns_table,
                left_index=True,
                right_index=True,
                how="outer",
            )
        elif not df_value_changes_table.empty:
            df_for_table = df_value_changes_table
        elif not df_percent_returns_table.empty:
            df_for_table = df_percent_returns_table
        else:
            table_model.updateData(pd.DataFrame())
            return

        # --- FILTER: Remove weekends for daily interval ---
        if interval_key == "D" and not df_for_table.empty:
            if isinstance(df_for_table.index, pd.DatetimeIndex):
                 # Filter out Saturday(5) and Sunday(6)
                 df_for_table = df_for_table[df_for_table.index.dayofweek < 5]
            else:
                 # Try to convert to datetime
                 try:
                     temp_index = pd.to_datetime(df_for_table.index)
                     df_for_table = df_for_table[temp_index.dayofweek < 5]
                 except Exception as e_date_filt:
                     logging.warning(f"Could not filter weekends for PVC daily table: {e_date_filt}")
        # --- END FILTER ---

        # Step 1: Reset index to move DatetimeIndex to a column
        # The name of the column that holds the original index/DB ID
        original_index_name = (
            df_for_table.index.name
        )  # Store original index name, could be None
        df_for_table.reset_index(inplace=True)

        # Identify the column that came from the index
        # If original_index_name was None, reset_index creates 'index'. Otherwise, it uses original_index_name (which is the internal DB ID).
        date_column_name_after_reset = (
            original_index_name if original_index_name is not None else "index"
        )

        # Ensure this column exists, if not, try to find a 'Date' column as a fallback
        if date_column_name_after_reset not in df_for_table.columns:
            if (
                "Date" in df_for_table.columns
            ):  # Common fallback if index was named 'Date'
                date_column_name_after_reset = "Date"
            elif (
                "index" in df_for_table.columns
            ):  # If original_index_name was something else but 'index' was created (and is the internal DB ID)
                date_column_name_after_reset = "index"
            else:
                if not df_for_table.empty:
                    date_column_name_after_reset = df_for_table.columns[0]
                    logging.warning(
                        f"PVC Table ({interval_key}): Could not reliably identify the date column after reset_index. "
                        f"Original index name was '{original_index_name}'. Assuming first column '{date_column_name_after_reset}' is the date column (and original_index is handled by PandasModel)."
                    )
                else:
                    logging.error(
                        f"PVC Table ({interval_key}): DataFrame is empty after reset_index, cannot identify date column."
                    )
                    table_model.updateData(pd.DataFrame())
                    return

        # Rename columns for display (remove interval key suffix, map benchmark tickers to names)
        # The date column (currently `date_column_name_after_reset`) still holds datetime objects.
        new_column_names = {}
        currency_sym = self._get_currency_symbol()

        for col in df_for_table.columns:
            if col == date_column_name_after_reset:  # This is our date column
                new_column_names[col] = "Period"  # Target name
            elif col == "Portfolio Value Change":
                new_column_names[col] = f"Value Change ({currency_sym})"  # MODIFIED
            elif col.startswith("Portfolio") and f" {interval_key}-Return" in col:
                new_column_names[col] = "Portfolio (%)"
            else:  # Benchmark column
                if (
                    f" {interval_key}-Return" in col
                ):  # Check if it's a benchmark return column
                    yf_ticker_part = col.split(f" {interval_key}-Return")[0]
                    display_name = yf_ticker_part
                    for name, ticker_in_map in BENCHMARK_MAPPING.items():
                        if ticker_in_map == yf_ticker_part:
                            display_name = name
                            break
                    new_column_names[col] = f"{display_name} (%)"
                # else: # Other columns, if any, keep their names or handle as needed
                #    new_column_names[col] = col # Implicitly handled if not in conditions

        df_for_table.rename(columns=new_column_names, inplace=True)

        # Step 3: Now that the "Period" column is correctly named and contains datetime objects, format it to string.
        if "Period" in df_for_table.columns:
            date_format_map = {"Y": "%Y", "M": "%Y-%m", "W": "%Y-%m-%d"}
            df_for_table["Period"] = pd.to_datetime(
                df_for_table["Period"], errors="coerce"
            )
            df_for_table["Period"] = df_for_table["Period"].dt.strftime(
                date_format_map.get(interval_key, "%Y-%m-%d")
            )
            df_for_table["Period"] = df_for_table["Period"].replace(
                {"NaT": "-"}, regex=False
            )  # Handle any NaT from coerce
        else:
            logging.warning(
                f"PVC Table ({interval_key}): 'Period' column not found after renaming, for date string formatting."
            )

        table_model.updateData(df_for_table)

        # --- ADDED: Apply default sort by Period (Descending) ---
        # Find the QTableView associated with this model to apply sort
        table_view_to_sort = None
        if table_model == self.pvc_annual_table_model:
            table_view_to_sort = self.pvc_annual_table_view
        elif table_model == self.pvc_monthly_table_model:
            table_view_to_sort = self.pvc_monthly_table_view
        elif table_model == self.pvc_weekly_table_model:
            table_view_to_sort = self.pvc_weekly_table_view
        elif table_model == self.pvc_daily_table_model:
            table_view_to_sort = self.pvc_daily_table_view

        if table_view_to_sort and not df_for_table.empty:
            try:
                # Find the column index in the *model* based on the UI name "Period"
                # This is safer if the model reorders/renames columns internally (though unlikely for PandasModel here)
                model_period_col_idx = -1
                for i in range(table_model.columnCount()):
                    header_ui_name = table_model.headerData(
                        i, Qt.Horizontal, Qt.DisplayRole
                    )
                    if str(header_ui_name).strip() == "Period":
                        model_period_col_idx = (
                            i  # This should be the index of the "Period" column
                        )
                        break
                if model_period_col_idx != -1:
                    logging.debug(
                        f"PVC Table ({interval_key}): Found 'Period' in model at index {model_period_col_idx}. Sorting view."
                    )
                    # Directly apply sort. Visibility handler will also cover it.
                    table_view_to_sort.sortByColumn(
                        model_period_col_idx, Qt.DescendingOrder
                    )
                    logging.debug(
                        f"PVC Table ({interval_key}): sortByColumn called directly for 'Period' index {model_period_col_idx} Descending."
                    )
                else:
                    logging.warning(
                        f"PVC Table ({interval_key}): Could not find 'Period' column in the model's headers to sort view."
                    )  # This log might appear if the column name isn't "Period" after rename

            except KeyError:
                logging.warning(
                    f"PVC Table ({interval_key}): KeyError finding 'Period' column in df_for_table for sorting view."
                )
            except Exception as e_sort_pvc:
                logging.error(
                    f"PVC Table ({interval_key}): Error applying default sort to view: {e_sort_pvc}"
                )
        else:
            if not table_view_to_sort:
                logging.warning(
                    f"PVC Table ({interval_key}): table_view_to_sort is None."
                )
            if df_for_table.empty:
                logging.debug(
                    f"PVC Table ({interval_key}): df_for_table is empty, skipping view sort."
                )
        # --- END ADDED ---

    # --- ADDED: Methods for PVC tab visibility and default sort ---
    @Slot(int)
    def _handle_pvc_tab_visibility_change(self, index: int):
        """
        Slot connected to main_tab_widget.currentChanged signal.
        If the Asset Change tab becomes visible, applies default sort to its tables.
        """
        try:
            current_tab_widget = self.main_tab_widget.widget(index)
            if current_tab_widget == self.periodic_value_change_tab:
                logging.debug(
                    "Asset Change tab became visible. Applying default sorts."
                )
                self._apply_default_sort_to_pvc_tables()
        except Exception as e:
            logging.error(
                f"Error in _handle_pvc_tab_visibility_change: {e}", exc_info=True
            )

    def _apply_default_sort_to_pvc_tables(self):
        """Applies default sort (Period Descending) to all tables in the PVC tab."""
        pvc_tables_config = [
            (self.pvc_annual_table_view, self.pvc_annual_table_model, "Annual"),
            (self.pvc_monthly_table_view, self.pvc_monthly_table_model, "Monthly"),
            (self.pvc_weekly_table_view, self.pvc_weekly_table_model, "Weekly"),
            (self.pvc_daily_table_view, self.pvc_daily_table_model, "Daily"),
        ]

        for table_view, table_model, interval_key_debug in pvc_tables_config:
            if table_view and table_model and not table_model._data.empty:
                try:
                    model_period_col_idx = -1
                    # ---- ADDED LOGGING ----
                    model_headers_from_visibility_handler = [
                        table_model.headerData(i, Qt.Horizontal, Qt.DisplayRole)
                        for i in range(table_model.columnCount())
                    ]
                    logging.debug(
                        f"PVC Tab Visible Sort ({interval_key_debug}): Model headers found by visibility handler: {model_headers_from_visibility_handler}"
                    )
                    # ---- END ADDED LOGGING ----
                    for i in range(table_model.columnCount()):
                        header_ui_name = table_model.headerData(
                            i, Qt.Horizontal, Qt.DisplayRole
                        )
                        if str(header_ui_name).strip() == "Period":
                            model_period_col_idx = (
                                i  # This should be the index of the "Period" column
                            )
                            break

                    if model_period_col_idx != -1:
                        logging.debug(
                            f"PVC Tab Visible Sort ({interval_key_debug}): Applying sort by 'Period' index {model_period_col_idx} Descending."
                        )
                        table_view.sortByColumn(
                            model_period_col_idx,
                            Qt.DescendingOrder,  # This should sort by "Period"
                        )
                    else:
                        logging.warning(
                            f"PVC Tab Visible Sort ({interval_key_debug}): Could not find 'Period' column in model headers. (Actual headers: {model_headers_from_visibility_handler})"
                        )
                except Exception as e_sort:
                    logging.error(
                        f"Error applying default sort to PVC table ({interval_key_debug}) on visibility: {e_sort}",
                        exc_info=True,
                    )
            elif table_model and table_model._data.empty:
                logging.debug(
                    f"PVC Tab Visible Sort ({interval_key_debug}): Model is empty, skipping sort."
                )
            elif not table_view:
                logging.warning(
                    f"PVC Tab Visible Sort ({interval_key_debug}): TableView is None."
                )
            elif not table_model:
                logging.warning(
                    f"PVC Tab Visible Sort ({interval_key_debug}): TableModel is None."
                )

    # --- END ADDED ---

    def _update_periodic_value_change_display(self):
        """Updates all graphs and tables in the Asset Change tab."""
        self._plot_pvc_graph(
            self.pvc_annual_graph_ax,
            self.pvc_annual_graph_canvas,
            "Y",
            self.pvc_annual_spinbox.value(),
        )
        self._update_pvc_table(
            self.pvc_annual_table_model, "Y", self.pvc_annual_spinbox.value()
        )
        self._plot_pvc_graph(
            self.pvc_monthly_graph_ax,
            self.pvc_monthly_graph_canvas,
            "M",
            self.pvc_monthly_spinbox.value(),
        )
        self._update_pvc_table(
            self.pvc_monthly_table_model, "M", self.pvc_monthly_spinbox.value()
        )
        self._plot_pvc_graph(
            self.pvc_weekly_graph_ax,
            self.pvc_weekly_graph_canvas,
            "W",
            self.pvc_weekly_spinbox.value(),
        )
        self._update_pvc_table(
            self.pvc_weekly_table_model, "W", self.pvc_weekly_spinbox.value()
        )
        self._plot_pvc_graph(
            self.pvc_daily_graph_ax,
            self.pvc_daily_graph_canvas,
            "D",
            self.pvc_daily_spinbox.value(),
        )
        self._update_pvc_table(
            self.pvc_daily_table_model, "D", self.pvc_daily_spinbox.value()
        )

    def _update_dividend_summary_table(self, plot_data: pd.Series):
        """Updates the dividend summary table view with aggregated data."""
        logging.debug("Updating dividend summary table...")
        if not hasattr(self, "dividend_summary_table_model"):
            logging.error("_update_dividend_summary_table: Model not initialized.")
            return

        if plot_data is None or plot_data.empty:
            logging.debug("  _update_dividend_summary_table: No plot_data or empty.")
            self.dividend_summary_table_model.updateData(pd.DataFrame())
            return

        # Convert Series to DataFrame for table display
        # Ensure index is named for clarity if it's a DatetimeIndex
        df_summary = plot_data.to_frame(
            name=f"Total Dividends ({self._get_currency_symbol()})"
        )
        # --- MODIFIED: Direct formatting for quarterly period string ---
        if isinstance(df_summary.index, pd.DatetimeIndex):
            period_type = self.dividend_period_combo.currentText()
            if period_type == "Quarterly":
                df_summary.index = [
                    f"{dt.year}-Q{dt.quarter}" for dt in df_summary.index
                ]
            else:  # Monthly or Annual
                date_format_str_table = "%Y"
                if period_type == "Monthly":  # Should be "Monthly" from combo box
                    date_format_str_table = "%Y-%m"
                df_summary.index = df_summary.index.strftime(date_format_str_table)
        # --- END MODIFICATION ---

        df_summary.index.name = "Period"
        df_summary = df_summary.reset_index()  # Make 'Period' a column

        logging.debug(
            f"  _update_dividend_summary_table: df_summary to be set in model (shape {df_summary.shape}):"
        )
        if not df_summary.empty:
            logging.debug(f"    Head:\n{df_summary.head().to_string()}")

        # --- CORRECTED: Call updateData ONCE, then sort ---
        self.dividend_summary_table_model.updateData(df_summary)

        try:
            if (
                not df_summary.empty
            ):  # Check if df_summary has data before trying to get column index
                period_col_index = df_summary.columns.get_loc("Period")
                self.dividend_summary_table_view.sortByColumn(
                    period_col_index, Qt.DescendingOrder
                )
                logging.debug("  Applied default sort to dividend summary table.")
            else:
                logging.debug(
                    "  df_summary is empty, skipping sort for dividend summary table."
                )
        except KeyError:
            logging.warning(
                "  'Period' column not found in dividend summary table, cannot apply default sort."
            )
        except Exception as e_sort:
            logging.error(f"  Error during dividend summary table sort: {e_sort}")
        # --- END CORRECTION ---
        self.dividend_summary_table_view.resizeColumnsToContents()

    def _update_dividend_table(self):
        """Updates the dividend history table view."""
        logging.debug("Updating dividend table...")
        if (
            not hasattr(self, "dividend_history_data")
            or self.dividend_history_data.empty
        ):
            logging.debug(
                "  _update_dividend_table: No dividend_history_data or empty."
            )
            self.dividend_table_model.updateData(pd.DataFrame())
            return

        logging.debug(
            f"  _update_dividend_table: self.dividend_history_data (shape {self.dividend_history_data.shape}):"
        )
        if not self.dividend_history_data.empty:
            logging.debug(f"    Head:\n{self.dividend_history_data.head().to_string()}")

        # self.dividend_history_data should already be filtered by account scope from the worker.
        # If it's not, the bug is in portfolio_analyzer.extract_dividend_history.
        # For the table display, we use self.dividend_history_data as is.
        # The title of the group box will be updated to reflect the scope.

        df_display = self.dividend_history_data.copy()

        # Format columns for display if needed (e.g., date, currency)
        if "Date" in df_display.columns:
            df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime(
                "%Y-%m-%d"
            )

        # Rename columns for UI friendliness
        rename_map = {
            "DividendAmountLocal": f'Amount ({self._get_currency_symbol(currency_code=df_display["LocalCurrency"].iloc[0] if not df_display.empty and "LocalCurrency" in df_display.columns and not df_display["LocalCurrency"].empty else self.config.get("default_currency"))})',
            "FXRateUsed": "FX Rate",
            "DividendAmountDisplayCurrency": f"Amount ({self._get_currency_symbol()})",
        }
        df_display.rename(columns=rename_map, inplace=True)

        # Update the title of the transaction group box
        scope_display_label_tx = self._get_scope_label_for_charts()
        if (
            hasattr(self, "dividend_transaction_group")
            and self.dividend_transaction_group
        ):
            self.dividend_transaction_group.setTitle(
                f"{scope_display_label_tx} - Dividend Transaction History"
            )
        # --- CORRECTED: Call updateData ONCE, then sort ---
        self.dividend_table_model.updateData(df_display)

        try:
            if not df_display.empty:  # Check if df_display has data
                date_col_index = df_display.columns.get_loc("Date")
                self.dividend_table_view.sortByColumn(
                    date_col_index, Qt.DescendingOrder
                )
                logging.debug("  Applied default sort to dividend history table.")
            else:
                logging.debug(
                    "  df_display is empty, skipping sort for dividend history table."
                )
        except KeyError:
            logging.warning(
                "  'Date' column not found in dividend history table, cannot apply default sort."
            )
        except Exception as e_sort:
            logging.error(f"  Error during dividend history table sort: {e_sort}")
        # --- END CORRECTION ---
        self.dividend_table_view.resizeColumnsToContents()

    # --- NEW: Capital Gains Display Methods ---
    @Slot()
    def _update_capital_gains_display(self):
        """Updates the Capital Gains tab (bar chart and table)."""
        logging.debug("Updating Capital Gains display...")
        self._update_capital_gains_bar_chart()
        # _update_capital_gains_summary_table will be called by _update_capital_gains_bar_chart
        self._update_capital_gains_table()
        self._update_capital_gains_summary_cards()

    def _update_capital_gains_summary_cards(self):
        """Updates the Capital Gains summary cards with total values."""
        if (
            not hasattr(self, "capital_gains_history_data")
            or self.capital_gains_history_data.empty
        ):
            self.cg_card_gain_value.setText("N/A")
            self.cg_card_proceeds_value.setText("N/A")
            self.cg_card_cost_value.setText("N/A")
            return

        try:
            df = self.capital_gains_history_data
            total_gain = df["Realized Gain (Display)"].sum()
            total_proceeds = df["Total Proceeds (Display)"].sum()
            total_cost = df["Total Cost Basis (Display)"].sum()
            
            currency_symbol = self._get_currency_symbol()
            
            # Helper to format
            def format_val(val):
                return f"{currency_symbol}{val:,.2f}"
            
            self.cg_card_gain_value.setText(format_val(total_gain))
            self.cg_card_proceeds_value.setText(format_val(total_proceeds))
            self.cg_card_cost_value.setText(format_val(total_cost))
            
            # Color code the gain
            # We need to maintain the font size/weight when setting stylesheet for color
            base_style = "font-weight: bold;" 
            # Note: Font size is set via QFont in init, but stylesheet might override if not careful.
            # Let's be explicit in stylesheet to be safe, matching init.
            
            if total_gain >= 0:
                self.cg_card_gain_value.setStyleSheet(f"color: {COLOR_GAIN}; {base_style}")
            else:
                self.cg_card_gain_value.setStyleSheet(f"color: {COLOR_LOSS}; {base_style}")
                
        except Exception as e:
            logging.error(f"Error updating capital gains summary cards: {e}")
            self.cg_card_gain_value.setText("Error")
            self.cg_card_proceeds_value.setText("Error")
            self.cg_card_cost_value.setText("Error")

    def _update_correlation_matrix_display(self):
        """Updates the Correlation Matrix display."""
        logging.debug("Updating Correlation Matrix display...")
        # Clear the entire figure to remove old axes and color bars
        self.correlation_fig.clear()
        # Add a new subplot for the new heatmap
        ax = self.correlation_fig.add_subplot(111)

        # Explicitly set background and text colors to match the current theme
        bg_color = self.QCOLOR_BACKGROUND_THEMED.name()
        text_color = self.QCOLOR_TEXT_PRIMARY_THEMED.name()
        secondary_text_color = self.QCOLOR_TEXT_SECONDARY_THEMED.name()

        self.correlation_fig.patch.set_facecolor(bg_color)
        ax.patch.set_facecolor(bg_color)
        ax.tick_params(colors=text_color, which="both")

        if self.correlation_matrix_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Correlation Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=secondary_text_color,
            )

        else:
            import seaborn as sns

            # Sort the index and columns alphabetically for consistent display
            sorted_corr_matrix = self.correlation_matrix_df.sort_index(
                axis=0
            ).sort_index(axis=1)
            sns.heatmap(
                sorted_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax
            )
            ax.set_title("Asset Correlation Matrix", color=text_color)
            ax.set_xticklabels(
                ax.get_xticklabels(), rotation=90, ha="right"
            )  # Rotate x-axis labels for better readability
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, va="center")

        self.correlation_fig.tight_layout()
        self.correlation_canvas.draw()

    def _update_factor_analysis_display(self):
        """Updates the Factor Analysis display."""
        logging.debug("Updating Factor Analysis display...")
        if self.factor_analysis_results:
            params = self.factor_analysis_results.get("params", {})
            pvalues = self.factor_analysis_results.get("pvalues", {})
            rsquared = self.factor_analysis_results.get("rsquared", None)

            model_name = self.factor_analysis_results.get(
                "model_name", "Unknown Factor Model"
            )

            display_text = f"<h3>Factor Analysis Results ({model_name})</h3>"
            display_text += "<p>This analysis helps understand how much of your portfolio's returns can be explained by common risk factors.</p>"

            if params:
                display_text += "<h4>Factor Betas (Coefficients):</h4>"
                display_text += "<table border='1' cellpadding='5' cellspacing='0'><tr><th>Factor</th><th>Beta</th><th>P-Value</th></tr>"
                for factor, beta in params.items():
                    p_value = pvalues.get(factor, float("nan"))
                    display_text += f"<tr><td>{factor}</td><td>{beta:.4f}</td><td>{p_value:.4f}</td></tr>"
                display_text += "</table>"
                display_text += "<p><b>Interpretation:</b><br>"
                display_text += "- <b>Mkt-RF (Market Risk Premium):</b> Sensitivity to overall market movements. A beta > 1 means more volatile than the market.<br>"
                display_text += "- <b>SMB (Small Minus Big):</b> Sensitivity to small-cap vs. large-cap stocks. Positive beta means exposure to small-cap.<br>"
                display_text += "- <b>HML (High Minus Low):</b> Sensitivity to value vs. growth stocks. Positive beta means exposure to value stocks.<br>"
                if model_name == "Carhart 4-Factor":
                    display_text += "- <b>UMD (Up Minus Down / Momentum):</b> Sensitivity to momentum factor. Positive beta means exposure to past winning stocks.<br>"
                display_text += "- <b>const (Alpha):</b> The portfolio's excess return not explained by the factors. Ideally positive and statistically significant (low P-value).</p>"

            if rsquared is not None:
                display_text += f"<h4>R-squared: {rsquared:.4f}</h4>"
                display_text += "<p><b>Interpretation:</b> R-squared indicates the proportion of the variance in your portfolio's returns that can be explained by the factors. A higher R-squared means the model explains more of your portfolio's movements.</p>"

            if not params and rsquared is None:
                display_text += "<p>No detailed factor analysis results available. This might happen if there's insufficient historical data or an error during calculation.</p>"

            self.factor_analysis_results_text.setHtml(display_text)
        else:
            self.factor_analysis_results_text.setText(
                "No Factor Analysis Results. Ensure you have sufficient historical data and valid transactions."
            )

    def _update_scenario_analysis_display(self):
        """Updates the Scenario Analysis display."""
        logging.debug("Updating Scenario Analysis display...")
        if self.scenario_analysis_result:
            estimated_return = self.scenario_analysis_result.get(
                "estimated_portfolio_return", np.nan
            )
            estimated_impact = self.scenario_analysis_result.get(
                "estimated_portfolio_impact", np.nan
            )
            display_currency_symbol = self._get_currency_symbol()

            # Determine color for return and impact
            return_color = COLOR_GAIN if estimated_return >= 0 else COLOR_LOSS
            impact_color = COLOR_GAIN if estimated_impact >= 0 else COLOR_LOSS

            # Format return and impact, handling NaN values
            formatted_return = (
                f"{estimated_return:+.2%}" if pd.notna(estimated_return) else "N/A"
            )
            formatted_impact = (
                f"{display_currency_symbol}{estimated_impact:+,.2f}"
                if pd.notna(estimated_impact)
                else "N/A"
            )

            self.scenario_impact_label.setText(
                f"<b>Estimated Portfolio Return:</b> <font color='{return_color}'>{formatted_return}</font><br>"
                f"<b>Estimated Portfolio Impact:</b> <font color='{impact_color}'>{formatted_impact}</font>"
            )
        else:
            self.scenario_impact_label.setText("No Scenario Analysis Results")

    @Slot()
    def _run_factor_analysis(self):
        """Triggers a full data refresh, which includes factor analysis calculation."""
        logging.info(
            "Run Factor Analysis button clicked. Triggering full data refresh."
        )
        self.refresh_data()

    @Slot()
    def _run_scenario_analysis(self):
        """Triggers a full data refresh, which includes scenario analysis calculation."""
        logging.info(
            "Run Scenario Analysis button clicked. Triggering full data refresh."
        )
        self.refresh_data()

    @Slot()
    def _load_preset_scenario(self):
        """Loads the selected preset scenario into the custom scenario input field."""
        selected_text = self.preset_scenario_combo.currentText()
        if "Select a Preset Scenario" in selected_text:
            self.scenario_input_line_edit.clear()
            return

        # Extract the scenario string from the preset text
        # Example: "Market Downturn (Mkt-RF: -0.10, HML: 0.05)"
        # We want: "Mkt-RF: -0.10, HML: 0.05"
        match = re.search(r"\((.*)\)", selected_text)
        if match:
            scenario_string = match.group(1)
            self.scenario_input_line_edit.setText(scenario_string)
        else:
            self.scenario_input_line_edit.clear()
            logging.warning(f"Could not parse scenario from preset: {selected_text}")

    def _get_scenario_shocks_from_input(self) -> Optional[Dict[str, float]]:
        """Parses the scenario input field and returns a dictionary of shocks."""
        shocks_text = self.scenario_input_line_edit.text().strip()
        if not shocks_text:
            return None

        shocks = {}
        try:
            parts = shocks_text.split(",")
            for part in parts:
                factor, value = part.split(":")
                factor = factor.strip()
                value = float(value.strip())
                shocks[factor] = value
            return shocks
        except ValueError:
            logging.error(f"Invalid scenario format: {shocks_text}")
            QMessageBox.warning(
                self,
                "Invalid Scenario",
                "Invalid scenario format. Please use 'Factor: Value, Factor: Value'.",
            )
            return None

    def _update_capital_gains_bar_chart(self):
        """Updates the Capital Gains bar chart."""
        logging.debug("Updating Capital Gains bar chart...")
        ax = self.cg_bar_ax
        canvas = self.cg_bar_canvas
        ax.clear()

        # Explicitly set background colors based on current theme
        fig = ax.get_figure()
        if fig:
            fig.patch.set_facecolor(
                self.QCOLOR_BACKGROUND_THEMED.name()
            )  # Main background for figure
        ax.patch.set_facecolor(
            self.QCOLOR_BACKGROUND_THEMED.name()
        )  # Input-like background for axes

        if (
            not hasattr(self, "capital_gains_history_data")
            or self.capital_gains_history_data.empty
        ):
            logging.debug("  _update_capital_gains_bar_chart: No data or empty.")
            ax.text(
                0.5,
                0.5,
                "No Capital Gains Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            # ax.set_title("Realized Capital Gains", fontsize=9, weight="bold")
            canvas.draw()
            self._update_capital_gains_summary_table(
                pd.Series(dtype=float)
            )  # Clear summary table
            return

        df_cg = self.capital_gains_history_data.copy()
        if (
            "Date" not in df_cg.columns
            or "Realized Gain (Display)" not in df_cg.columns
        ):
            logging.error("Capital gains data missing required Date or Gain columns.")
            ax.text(
                0.5,
                0.5,
                "Invalid Capital Gains Data",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_TEXT_SECONDARY,
            )
            # ax.set_title("Realized Capital Gains", fontsize=9, weight="bold")
            canvas.draw()
            self._update_capital_gains_summary_table(
                pd.Series(dtype=float)
            )  # Clear summary table
            return

        df_cg["Date"] = pd.to_datetime(df_cg["Date"])
        df_cg.set_index("Date", inplace=True)

        period_type = self.cg_period_combo.currentText()
        num_periods_to_show = self.cg_periods_spinbox.value()
        display_currency_symbol = self._get_currency_symbol()

        resample_freq = "YE"  # Annual
        date_format_str = "%Y"
        if period_type == "Quarterly":
            resample_freq = "QE"
            # For quarterly, strftime %q is not standard across all systems for pandas index directly
            # We will handle quarterly labeling during x_labels generation.
            date_format_str = "%Y-Q%q"

        try:
            aggregated_gains = (
                df_cg["Realized Gain (Display)"].resample(resample_freq).sum().dropna()
            )

            # Initialize plot_data_for_table as an empty Series.
            # It will be updated only if valid plot_data is generated.
            plot_data_for_table = pd.Series(dtype=float)

            if aggregated_gains.empty:
                ax.text(
                    0.5,
                    0.5,
                    f"No Realized Gains/Losses for {period_type} Periods",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    color=COLOR_TEXT_SECONDARY,
                )
                # plot_data_for_table remains empty here
            else:
                plot_data = aggregated_gains.tail(num_periods_to_show)
                if plot_data.empty:
                    ax.text(
                        0.5,
                        0.5,
                        f"No Realized Gains/Losses for Last {num_periods_to_show} {period_type} Periods",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        color=COLOR_TEXT_SECONDARY,
                    )
                    # plot_data_for_table remains empty here
                else:
                    # >>> KEY CHANGE: Assign plot_data to plot_data_for_table <<<
                    plot_data_for_table = plot_data
                    # --- Generate x_labels ---
                    if period_type == "Quarterly":
                        x_labels = [f"{dt.year}-Q{dt.quarter}" for dt in plot_data.index]  # type: ignore
                    else:  # Annual
                        x_labels = plot_data.index.strftime(date_format_str)
                    # --- End Generate x_labels ---

                    colors = [
                        COLOR_GAIN if val >= 0 else COLOR_LOSS
                        for val in plot_data.values
                    ]
                    bars = ax.bar(x_labels, plot_data.values, color=colors, width=0.6)

                    # Add value labels on top/bottom of bars
                    for bar in bars:
                        yval = bar.get_height()
                        if pd.notna(yval) and abs(yval) > 1e-9:
                            va = "bottom" if yval >= 0 else "top"
                            offset = (
                                (plot_data.max() * 0.01)
                                if yval >= 0
                                else -(plot_data.min() * 0.01)
                            )  # Small offset
                            ax.text(
                                bar.get_x() + bar.get_width() / 2.0,
                                yval + offset,
                                f"{display_currency_symbol}{yval:,.2f}",
                                ha="center",
                                va=va,
                                fontsize=7,
                                color=self.QCOLOR_TEXT_PRIMARY_THEMED.name(),
                            )

                    ax.yaxis.set_major_formatter(
                        mtick.FuncFormatter(
                            lambda x, p: f"{display_currency_symbol}{x:,.0f}"
                        )
                    )
                    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    ax.grid(
                        True,
                        axis="y",
                        linestyle="--",
                        linewidth=0.5,
                        color=COLOR_BORDER_LIGHT,
                    )
                    ax.axhline(0, color=COLOR_BORDER_DARK, linewidth=0.6)  # Zero line

        except Exception as e:
            logging.error(f"Error generating capital gains bar chart: {e}")
            traceback.print_exc()
            ax.text(
                0.5,
                0.5,
                "Error Plotting Gains",
                ha="center",
                va="center",
                transform=ax.transAxes,
                color=COLOR_LOSS,
            )

        self._update_capital_gains_summary_table(
            plot_data_for_table
            if "plot_data_for_table" in locals()
            else pd.Series(dtype=float)
        )
        scope_display_label = self._get_scope_label_for_charts()
        # ax.set_title(
        #     f"{scope_display_label} - Realized Capital Gains ({display_currency_symbol})",
        #     fontsize=9,
        #     weight="bold",
        # )
        ax.set_xlabel("")  # Clear x-axis label
        ax.set_ylabel(f"Total Gain/Loss ({display_currency_symbol})", fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        self.cg_bar_fig.tight_layout(pad=0.5)
        canvas.draw()

    def _update_capital_gains_summary_table(self, plot_data: pd.Series):
        """Updates the capital gains summary table view with aggregated data."""
        logging.debug("Updating capital gains summary table...")
        if not hasattr(self, "cg_summary_table_model"):
            logging.error("_update_capital_gains_summary_table: Model not initialized.")
            return

        if plot_data is None or plot_data.empty:
            logging.debug(
                "  _update_capital_gains_summary_table: No plot_data or empty."
            )
            self.cg_summary_table_model.updateData(pd.DataFrame())
            return

        df_summary = plot_data.to_frame(
            name=f"Realized Gain/Loss ({self._get_currency_symbol()})"
        )

        period_type = self.cg_period_combo.currentText()
        if isinstance(df_summary.index, pd.DatetimeIndex):
            if period_type == "Quarterly":
                df_summary.index = [
                    f"{dt.year}-Q{dt.quarter}" for dt in df_summary.index
                ]
            else:  # Annual
                date_format_str_table = "%Y"
                df_summary.index = df_summary.index.strftime(date_format_str_table)

        df_summary.index.name = "Period"
        df_summary = df_summary.reset_index()

        self.cg_summary_table_model.updateData(df_summary)

        try:
            if not df_summary.empty:
                period_col_index = df_summary.columns.get_loc("Period")
                self.cg_summary_table_view.sortByColumn(
                    period_col_index, Qt.DescendingOrder
                )
                logging.debug("  Applied default sort to capital gains summary table.")
            else:
                logging.debug(
                    "  df_summary is empty, skipping sort for capital gains summary table."
                )
        except KeyError:
            logging.warning(
                "  'Period' column not found in capital gains summary table, cannot apply default sort."
            )
        except Exception as e_sort:
            logging.error(f"  Error during capital gains summary table sort: {e_sort}")

        self.cg_summary_table_view.resizeColumnsToContents()
        self.cg_summary_table_view.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch
        )

    def _update_capital_gains_table(self):
        """Updates the Capital Gains history table view."""
        logging.debug("Updating Capital Gains table...")
        if not hasattr(self, "cg_table_model"):
            logging.error("_update_capital_gains_table: Model not initialized.")
            return

        if (
            not hasattr(self, "capital_gains_history_data")
            or self.capital_gains_history_data.empty
        ):
            logging.debug("  _update_capital_gains_table: No data or empty.")
            self.cg_table_model.updateData(pd.DataFrame())
            return

        df_display = self.capital_gains_history_data.copy()

        display_currency_symbol = self._get_currency_symbol()

        # Format currency columns before renaming
        local_currency_cols = [
            "Avg Sale Price (Local)",
            "Total Proceeds (Local)",
            "Total Cost Basis (Local)",
            "Realized Gain (Local)",
        ]
        display_currency_cols = [
            "Total Proceeds (Display)",
            "Total Cost Basis (Display)",
            "Realized Gain (Display)",
        ]

        if "LocalCurrency" in df_display.columns:
            for col_name in local_currency_cols:
                if col_name in df_display.columns:
                    df_display[col_name] = df_display.apply(
                        lambda r: (
                            f"{self._get_currency_symbol(currency_code=r.get('LocalCurrency'))}{r[col_name]:,.2f}"
                            if pd.notna(r[col_name])
                            and pd.notna(r.get("LocalCurrency"))
                            and isinstance(r[col_name], (float, int))
                            else ("-" if pd.isna(r[col_name]) else str(r[col_name]))
                        ),
                        axis=1,
                    )
        else:  # Fallback if LocalCurrency column is missing
            for col_name in local_currency_cols:
                if col_name in df_display.columns:
                    df_display[col_name] = df_display[col_name].apply(
                        lambda x: (
                            f"{x:,.2f}"
                            if pd.notna(x) and isinstance(x, (float, int))
                            else ("-" if pd.isna(x) else str(x))
                        )
                    )

        for col_name in display_currency_cols:
            if col_name in df_display.columns:
                df_display[col_name] = df_display[col_name].apply(
                    lambda x: (
                        f"{display_currency_symbol}{x:,.2f}"
                        if pd.notna(x) and isinstance(x, (float, int))
                        else ("-" if pd.isna(x) else str(x))
                    )
                )

        # Format other numeric columns
        if "Quantity" in df_display.columns:
            df_display["Quantity"] = df_display["Quantity"].apply(
                lambda x: (
                    f"{x:,.4f}"
                    if pd.notna(x) and isinstance(x, (float, int))
                    else ("-" if pd.isna(x) else str(x))
                )
            )
        if "Sale/Cover FX Rate" in df_display.columns:
            df_display["Sale/Cover FX Rate"] = df_display["Sale/Cover FX Rate"].apply(
                lambda x: (
                    f"{x:,.4f}"
                    if pd.notna(x) and isinstance(x, (float, int))
                    else ("-" if pd.isna(x) else str(x))
                )
            )

        # Ensure original_tx_id is string
        if "original_tx_id" in df_display.columns:
            df_display["original_tx_id"] = (
                df_display["original_tx_id"]
                .astype(str)
                .replace("nan", "-")
                .replace("<NA>", "-")
            )

        # Drop LocalCurrency column after formatting, before renaming, if it's not part of UI
        if "LocalCurrency" in df_display.columns:
            df_display = df_display.drop(columns=["LocalCurrency"])

        # Format columns for display
        if "Date" in df_display.columns:
            df_display["Date"] = pd.to_datetime(df_display["Date"]).dt.strftime(
                "%Y-%m-%d"
            )

        # Rename columns for UI friendliness
        rename_map = {
            "Quantity": "Quantity Sold",
            "Avg Sale Price (Local)": "Avg Sale Price",  # Cell value already has local symbol
            "Total Proceeds (Local)": "Proceeds (Local)",
            "Total Cost Basis (Local)": "Cost Basis (Local)",
            "Realized Gain (Local)": "Gain/Loss (Local)",
            "Sale/Cover FX Rate": "FX Rate",
            "Total Proceeds (Display)": f"Proceeds ({display_currency_symbol})",
            "Total Cost Basis (Display)": f"Cost Basis ({display_currency_symbol})",
            "Realized Gain (Display)": f"Gain/Loss ({display_currency_symbol})",
            "original_tx_id": "Original Tx ID",
        }
        df_display.rename(columns=rename_map, inplace=True)

        # --- CORRECTED: Call updateData ONCE, then sort ---
        self.cg_table_model.updateData(df_display)

        try:
            if not df_display.empty:  # Check if df_display has data
                date_col_index = df_display.columns.get_loc("Date")
                self.cg_table_view.sortByColumn(date_col_index, Qt.DescendingOrder)
                logging.debug("  Applied default sort to capital gains table.")
            else:
                logging.debug(
                    "  df_display is empty, skipping sort for capital gains table."
                )
        except KeyError:
            logging.warning(
                "  'Date' column not found in capital gains table, cannot apply default sort."
            )
        except Exception as e_sort:
            logging.error(f"  Error during capital gains table sort: {e_sort}")
        # --- END CORRECTION ---

        self.cg_table_view.resizeColumnsToContents()

    # --- End Capital Gains Display Methods ---

    @Slot()
    def open_add_transaction_dialog(self):
        """Opens the AddTransactionDialog for manual transaction entry into the database."""
        if not self.db_conn:
            QMessageBox.warning(
                self,
                "No Database Connection",
                "Please open or create a database file first to add transactions.",
            )
            return

        # Get available accounts from the current application state
        # self.available_accounts should be populated by refresh_data or after DB initialization.
        accounts_for_dialog = self.available_accounts if self.available_accounts else []

        # If available_accounts is empty, try to fetch from DB one last time (e.g., if app just started with empty DB)
        if not accounts_for_dialog and self.db_conn:
            # Use account_currency_map and default_currency from config for cleaning after DB load
            acc_map_config_add_tx = self.config.get("account_currency_map", {})
            def_curr_config_add_tx = self.config.get(
                "default_currency",
                (
                    config.DEFAULT_CURRENCY
                    if hasattr(config, "DEFAULT_CURRENCY")
                    else "USD"
                ),
            )
            # MODIFIED: Pass map and default to load_all_transactions_from_db
            temp_df, success = load_all_transactions_from_db(
                self.db_conn,
                account_currency_map=acc_map_config_add_tx,
                default_currency=def_curr_config_add_tx,
            )
            # END MODIFIED
            if (
                success and temp_df is not None and "Account" in temp_df.columns
            ):  # Keep the rest of the logic
                # Combine accounts from both 'Account' and 'To Account' columns
                unique_accounts = set()
                if "Account" in temp_df.columns:
                    unique_accounts.update(temp_df["Account"].dropna().unique())
                if "To Account" in temp_df.columns:
                    unique_accounts.update(temp_df["To Account"].dropna().unique())
                accounts_for_dialog = sorted(list(unique_accounts))
                self.available_accounts = accounts_for_dialog  # Update app's list
            elif not success:
                QMessageBox.warning(
                    self, "DB Error", "Could not fetch account list from database."
                )
                # Proceed with empty account list, user can type new account.

        # --- Get portfolio symbols for autocompletion ---
        portfolio_symbols_for_dialog = []
        if (
            hasattr(self, "original_data")
            and not self.original_data.empty
            and "Symbol" in self.original_data.columns
        ):
            # Use original_data to include all symbols ever used, not just current holdings
            portfolio_symbols_for_dialog = list(self.original_data["Symbol"].unique())
            # <-- MODIFIED: Removed $CASH exclusion block
        # --- End Get portfolio symbols ---

        dialog = AddTransactionDialog(
            existing_accounts=accounts_for_dialog,
            portfolio_symbols=portfolio_symbols_for_dialog,  # <-- Pass symbols
            parent=self,
        )
        if (
            dialog.exec()
        ):  # True if dialog was accepted (Save clicked and validation passed)
            # get_transaction_data now returns a dict with Python types, keys are CSV-like headers
            new_data_pytypes_from_dialog = dialog.get_transaction_data()
            if new_data_pytypes_from_dialog:
                # save_new_transaction will handle mapping to DB columns and saving
                self.save_new_transaction(new_data_pytypes_from_dialog)
            # If get_transaction_data returned None, it means dialog validation failed, so do nothing.
        else:
            logging.info("Add transaction dialog was cancelled.")

    def _add_new_account_if_needed(self, account_name: str):
        """Checks if an account is new and adds it to available_accounts and config."""
        if account_name not in self.available_accounts:
            self.available_accounts.append(account_name)
            self.available_accounts.sort()  # Keep it sorted
            logging.info(f"New account '{account_name}' added to available accounts.")
            # Optionally add to config with default currency
            if account_name not in self.config.get("account_currency_map", {}):
                default_curr = self.config.get("default_currency", "USD")
                self.config.setdefault("account_currency_map", {})[
                    account_name
                ] = default_curr
                logging.info(
                    f"New account '{account_name}' added to config map with currency '{default_curr}'."
                )
                # self.save_config() # Decide if config should be saved immediately or on app close

    def save_new_transaction(self, transaction_data_pytypes: Dict[str, Any]):
        """
        Adds a new transaction to the database.
        Formats data from Python types (received from AddTransactionDialog)
        to what db_utils.add_transaction_to_db expects (DB column names and appropriate types).

        Args:
            transaction_data_pytypes (Dict[str, Any]): A dictionary containing the validated
                transaction data with Python types (float, date, str), keyed by CSV-like header names
                (e.g., "Date (MMM DD, YYYY)", "Quantity of Units").
        """
        logging.info(
            f"Attempting to save new transaction to DB. Raw dialog data: {transaction_data_pytypes}"
        )
        if not self.db_conn:
            QMessageBox.critical(
                self,
                "Save Error",
                "Cannot save transaction. No active database connection.",
            )
            return

        # Map CSV-like headers from dialog to DB column names and prepare types for DB
        data_for_db: Dict[str, Any] = {}
        try:
            # Date: AddTransactionDialog returns a datetime.date object via get_transaction_data
            date_obj = transaction_data_pytypes.get("Date (MMM DD, YYYY)")
            if isinstance(date_obj, (datetime, date)):
                data_for_db["Date"] = date_obj.strftime(
                    "%Y-%m-%d"
                )  # DB expects YYYY-MM-DD string
            elif (
                date_obj is None and "Date (MMM DD, YYYY)" in transaction_data_pytypes
            ):  # Explicit None means clear
                data_for_db["Date"] = (
                    None  # Should be caught by dialog validation if mandatory
                )
            elif (
                date_obj is not None
            ):  # If not date/datetime and not None (e.g. unexpected type)
                logging.error(
                    f"Invalid date type from AddTransactionDialog: {type(date_obj)} for value {date_obj}"
                )
                QMessageBox.critical(
                    self,
                    "Data Error",
                    "Invalid date type received from dialog. Transaction not saved.",
                )
                return
            # If key is missing, it wasn't relevant for the transaction type (e.g. Split Date)

            data_for_db["Type"] = transaction_data_pytypes.get("Transaction Type")
            data_for_db["Symbol"] = transaction_data_pytypes.get("Stock / ETF Symbol")

            # Numeric fields are already floats or None from get_transaction_data
            data_for_db["Quantity"] = transaction_data_pytypes.get("Quantity of Units")
            data_for_db["Price/Share"] = transaction_data_pytypes.get("Amount per unit")
            data_for_db["Total Amount"] = transaction_data_pytypes.get("Total Amount")
            data_for_db["Commission"] = transaction_data_pytypes.get("Fees")
            data_for_db["Split Ratio"] = transaction_data_pytypes.get(
                "Split Ratio (new shares per old share)"
            )

            data_for_db["Account"] = transaction_data_pytypes.get("Investment Account")
            data_for_db["Note"] = transaction_data_pytypes.get("Note")

            # --- ADDED: Handle "To Account" for transfers ---
            if data_for_db.get("Type", "").lower() == "transfer":
                data_for_db["To Account"] = transaction_data_pytypes.get("To Account")

            # Determine Local Currency based on account
            acc_name_for_currency = data_for_db.get("Account")
            if (
                acc_name_for_currency
            ):  # Account name should always be present due to dialog validation
                # Ensure account_currency_map and default_currency are correctly fetched from config
                app_config_account_map = self.config.get("account_currency_map", {})
                app_config_default_curr = self.config.get(
                    "default_currency",
                    (
                        config.DEFAULT_CURRENCY
                        if hasattr(config, "DEFAULT_CURRENCY")
                        else "USD"
                    ),
                )

                data_for_db["Local Currency"] = app_config_account_map.get(
                    str(
                        acc_name_for_currency
                    ),  # Ensure acc_name is string for map lookup
                    app_config_default_curr,
                )

            else:  # Fallback, though account should be mandatory from dialog
                data_for_db["Local Currency"] = self.config.get(
                    "default_currency",
                    (
                        config.DEFAULT_CURRENCY
                        if hasattr(config, "DEFAULT_CURRENCY")
                        else "USD"
                    ),
                )
                logging.warning(
                    "Account name was missing when determining local currency for new transaction."
                )

            # Ensure critical fields for DB are not None if they are NOT NULL in schema
            # db_utils.add_transaction_to_db will handle None for nullable fields.
            # Date, Type, Symbol, Account, Local Currency are NOT NULL in schema.
            for required_db_col in [
                "Date",
                "Type",
                "Symbol",
                "Account",
                "Local Currency",
            ]:
                if data_for_db.get(required_db_col) is None:
                    logging.error(
                        f"Critical field '{required_db_col}' is None. Transaction cannot be saved."
                    )
                    QMessageBox.critical(
                        self,
                        "Data Error",
                        f"Required field '{required_db_col}' is missing. Transaction not saved.",
                    )
                    return

            # Log the data being sent to DB for clarity
            logging.debug(f"Data prepared for DB insertion: {data_for_db}")

        except Exception as e_map:
            logging.error(
                f"Error mapping dialog data to DB format: {e_map}", exc_info=True
            )
            QMessageBox.critical(
                self, "Data Error", "Internal error preparing data for database."
            )
            return

        # Call db_utils function to add to database
        success, new_id = add_transaction_to_db(self.db_conn, data_for_db)

        if success:
            logging.info(f"Transaction successfully added to DB with ID: {new_id}")
            new_account_name_added = data_for_db.get("Account")
            if new_account_name_added:
                self._add_new_account_if_needed(
                    str(new_account_name_added)
                )  # Update available accounts list in GUI
            self._mark_data_as_stale()  # Mark data as stale
            QMessageBox.information(
                self,
                "Success",
                "Transaction added successfully to database.\nRefreshing data...",
            )
        else:
            QMessageBox.critical(
                self,
                "Save Error",
                "Failed to add transaction to database. Check logs for details.",
            )

    def closeEvent(self, event):
        """
        Handles the main window close event.

        Saves the current configuration and closes the database connection
        before closing. Ensures background threads are properly terminated.

        Args:
            event (QCloseEvent): The close event object.
        """
        logging.info(
            "Close event triggered. Saving config and closing database connection..."
        )
        try:
            self.save_config()
        except Exception as e:
            # Log the error but don't prevent closing if config save fails.
            # User might want to close even if saving config has issues.
            logging.error(f"Error saving settings on exit: {e}", exc_info=True)
            QMessageBox.warning(
                self,
                "Config Save Error",
                f"Could not save settings on exit:\n{e}\n\nApplication will still close.",
            )

        if self.db_conn:  # Close DB connection if it's open
            try:
                self.db_conn.close()
                logging.info(f"Database connection to '{self.DB_FILE_PATH}' closed.")
            except Exception as e_db_close:
                logging.error(
                    f"Error closing database connection: {e_db_close}", exc_info=True
                )
                # Continue closing even if DB close fails.

        logging.info("Exiting Investa Portfolio Dashboard application.")
        # Clean up worker threads
        if hasattr(self, "threadpool"):
            self.threadpool.clear()  # Clear any queued tasks
            # Wait for active threads to finish, with a timeout
            if not self.threadpool.waitForDone(2000):  # Wait up to 2 seconds
                logging.warning(
                    "Warning: Worker threads did not finish closing gracefully within the timeout."
                )

        event.accept()  # Accept the close event to allow the window to close

    # --- Toolbar Initialization ---
    def _init_toolbar(self):
        """Initializes the main application toolbar."""
        self.toolbar = self.addToolBar("Main Toolbar")
        self.toolbar.setObjectName("MainToolBar")
        self.toolbar.setIconSize(QSize(20, 20))  # Standard icon size

        if not isinstance(self.toolbar, QToolBar):
            logging.critical("CRITICAL: self.toolbar is NOT a QToolBar instance!")
            return

        # Add DB-related actions first
        if hasattr(self, "select_db_action") and self.select_db_action:
            self.toolbar.addAction(self.select_db_action)
        if hasattr(self, "new_database_file_action") and self.new_database_file_action:
            self.toolbar.addAction(self.new_database_file_action)  # Add New DB action

        # Import CSV action
        if hasattr(self, "import_csv_action") and self.import_csv_action:
            import_icon = self.import_csv_action.icon()
            if not import_icon or import_icon.isNull():
                logging.warning(
                    f"Toolbar: Icon for '{self.import_csv_action.text()}' is NULL. Using SP_ArrowDown as fallback."
                )
                import_icon = self.style().standardIcon(QStyle.SP_ArrowDown)
                self.import_csv_action.setIcon(import_icon)
            self.toolbar.addAction(self.import_csv_action)

        # NEW: Add Excel export action to toolbar
        if hasattr(self, "export_excel_action") and self.export_excel_action:
            self.toolbar.addAction(self.export_excel_action)

        if hasattr(self, "refresh_action"):
            self.toolbar.addAction(self.refresh_action)

        self.toolbar.addSeparator()  # Separator

        # Add Transaction action
        if hasattr(self, "add_transaction_action") and self.add_transaction_action:
            self.toolbar.addAction(self.add_transaction_action)

        # Manage Transactions action
        if (
            hasattr(self, "manage_transactions_action")
            and self.manage_transactions_action
        ):
            manage_icon = self.manage_transactions_action.icon()
            if not manage_icon or manage_icon.isNull():
                logging.warning(
                    f"Toolbar: Icon for '{self.manage_transactions_action.text()}' is NULL. Using SP_DirIcon as fallback."
                )
                manage_icon = self.style().standardIcon(QStyle.SP_DirIcon)
                self.manage_transactions_action.setIcon(manage_icon)
            self.toolbar.addAction(self.manage_transactions_action)

        # Spacer to push subsequent items to the right
        spacer_right = QWidget()
        spacer_right.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.toolbar.addWidget(spacer_right)

        # Fundamental lookup widgets (remain on the right)
        self.lookup_symbol_edit = QLineEdit(self)
        self.lookup_symbol_edit.setPlaceholderText(
            "Symbol for Fundamentals (e.g. AAPL)"
        )
        self.lookup_symbol_edit.setMinimumWidth(150)
        self.lookup_symbol_edit.setMaximumWidth(200)
        self.toolbar.addWidget(self.lookup_symbol_edit)

        self.lookup_button = QPushButton("Get Fundamentals")  # This was here before
        self.toolbar.addWidget(self.lookup_button)

        self.stale_data_indicator_label = QLabel(" <b>Refresh Required</b>")
        self.stale_data_indicator_label.setObjectName("StaleDataIndicator")
        self.stale_data_indicator_label.setToolTip(
            "Transaction data has changed. Click 'Refresh All' to update calculations."
        )
        self.stale_data_indicator_label.setVisible(False)  # Initially hidden
        self.toolbar.addWidget(self.stale_data_indicator_label)

    # --- Show Window and Run Event Loop ---
    # main_window.show() # This should be in __main__
    # sys.exit(app.exec()) # This should be in __main__

    # --- Slots for Fundamental Data Lookup ---
    @Slot()
    def _handle_direct_symbol_lookup(self):
        """Handles the 'Get Fundamentals' button click or Enter press."""
        input_symbol = self.lookup_symbol_edit.text().strip().upper()
        if not input_symbol:
            QMessageBox.warning(self, "Input Error", "Please enter a stock symbol.")
            return

        # For direct lookup, we use the user-defined map
        yf_symbol = map_to_yf_symbol(
            input_symbol, self.user_symbol_map_config, self.user_excluded_symbols_config
        )

        if not yf_symbol:  # Check after applying user map
            QMessageBox.warning(
                self,
                "Symbol Error",
                f"Could not map '{input_symbol}' to a valid Yahoo Finance ticker.",
            )
            return

        logging.info(
            f"Direct fundamental lookup requested for: {input_symbol} (YF: {yf_symbol})"
        )
        self.set_status(f"Fetching fundamentals for {input_symbol}...")
        self.lookup_symbol_edit.setEnabled(False)
        self.lookup_button.setEnabled(False)

        worker = FundamentalDataWorker(
            yf_symbol,
            input_symbol,  # The display symbol is the one the user typed
            self.worker_signals,
            FINANCIAL_RATIOS_AVAILABLE,
        )
        self.threadpool.start(worker)

    @Slot()
    def _show_column_context_menu(
        self, table_view: QTableView, pos: QPoint, config_key: str
    ):
        """
        Displays a context menu for table headers to toggle column visibility.
        The visibility state is stored in self.config under the given config_key.
        """
        menu = QMenu(self)
        header = table_view.horizontalHeader()
        model = table_view.model()

        # Ensure the config structure exists for this table
        if config_key not in self.config:
            self.config[config_key] = {}

        for i in range(model.columnCount()):
            column_name = model.headerData(i, Qt.Horizontal)
            action = menu.addAction(column_name)
            action.setCheckable(True)
            # Get initial state from config, fallback to current view state
            initial_checked_state = self.config[config_key].get(
                column_name, not table_view.isColumnHidden(i)
            )
            action.setChecked(initial_checked_state)

            # Connect action to a lambda that toggles visibility and saves config
            action.triggered.connect(
                lambda checked, col_idx=i, col_name=column_name, tv=table_view, ck=config_key: self._toggle_column_visibility(
                    tv, col_idx, col_name, checked, ck
                )
            )
        menu.exec(header.mapToGlobal(pos))

    @Slot()
    def _toggle_column_visibility(
        self,
        table_view: QTableView,
        column_index: int,
        column_name: str,
        visible: bool,
        config_key: str,
    ):
        """
        Toggles the visibility of a column and saves the state to config.
        """
        table_view.setColumnHidden(column_index, not visible)
        self.config[config_key][column_name] = visible
        self.save_config()  # Save config immediately after change
        logging.debug(
            f"Column '{column_name}' visibility set to {visible} for {config_key}."
        )

    def _apply_column_visibility(self, table_view: QTableView, config_key: str):
        """
        Applies saved column visibility settings from config to the given table view.
        """
        if config_key not in self.config:
            # If no config for this table, all columns are visible by default
            return

        column_visibility_settings = self.config[config_key]
        logging.debug(
            f"Applying column visibility for {config_key}. Settings: {column_visibility_settings}"
        )
        model = table_view.model()

        for i in range(model.columnCount()):
            column_name = model.headerData(i, Qt.Horizontal)
            # Default to visible if not found in settings
            is_visible = column_visibility_settings.get(column_name, True)
            table_view.setColumnHidden(i, not is_visible)
            logging.debug(
                f"  Column '{column_name}' (index {i}): is_visible={is_visible}, set hidden={not is_visible}"
            )
        logging.debug(f"Finished applying column visibility for {config_key}.")

    @Slot(str, dict)
    def _show_fundamental_data_dialog_from_worker(
        self, display_symbol: str, data: dict
    ):
        """Shows the FundamentalDataDialog with data received from the worker."""
        self.lookup_symbol_edit.setEnabled(True)
        self.lookup_button.setEnabled(True)
        self.set_status("Ready")  # Reset status

        if (
            not data
        ):  # data is an empty dict if yfinance found nothing or error during fetch by worker
            QMessageBox.warning(
                self,
                "Data Not Found",
                f"Could not retrieve fundamental data for {display_symbol}.",
            )
            return

        dialog = FundamentalDataDialog(display_symbol, data, self)
        dialog.exec()

    @Slot(str)
    def _handle_context_menu_fundamental_lookup(self, internal_symbol: str):
        """Handles fundamental lookup triggered from the table's context menu."""
        if not internal_symbol or internal_symbol == CASH_SYMBOL_CSV:
            return

        yf_symbol = self.internal_to_yf_map.get(internal_symbol)
        if not yf_symbol:
            # Fallback if not in map (e.g., if map wasn't populated for some reason) - use user map
            yf_symbol_fallback = map_to_yf_symbol(
                internal_symbol,
                self.user_symbol_map_config,
                self.user_excluded_symbols_config,
            )
            if not yf_symbol_fallback:
                QMessageBox.warning(
                    self,
                    "Symbol Error",
                    f"Could not map '{internal_symbol}' to a Yahoo Finance ticker for fundamentals.",
                )
                return
            yf_symbol = yf_symbol_fallback
            logging.warning(
                f"Used fallback YF symbol mapping for context menu fundamental lookup: {internal_symbol} -> {yf_symbol}"
            )

        logging.info(
            f"Context menu fundamental lookup for: {internal_symbol} (YF: {yf_symbol})"
        )
        self.set_status(f"Fetching fundamentals for {internal_symbol}...")
        self.lookup_symbol_edit.setEnabled(
            False
        )  # Disable direct lookup while context menu lookup is active
        self.lookup_button.setEnabled(False)

        worker = FundamentalDataWorker(
            yf_symbol,
            internal_symbol,
            self.worker_signals,
            FINANCIAL_RATIOS_AVAILABLE,  # Pass the module-level flag
        )
        self.threadpool.start(worker)

    # --- End Fundamental Data Slots ---

    # --- End Fundamental Data Slots ---

    @Slot()
    def clear_cache_files_action_triggered(self):
        # Ask for confirmation to delete all cache files
        reply = QMessageBox.question(
            self,
            "Confirm Clear Cache",
            "Are you sure you want to delete ALL application cache files?\n\nThis action cannot be undone.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            cache_dir_base = QStandardPaths.writableLocation(
                QStandardPaths.CacheLocation
            )
            if not cache_dir_base:
                QMessageBox.critical(
                    self,
                    "Clear Cache Error",
                    "Could not determine cache directory path.",
                )
                logging.error(
                    "Failed to clear cache: QStandardPaths.CacheLocation is not writable or available."
                )
                return
            try:
                if os.path.exists(cache_dir_base):
                    shutil.rmtree(cache_dir_base)
                    logging.info(
                        f"Successfully deleted cache directory: {cache_dir_base}"
                    )
                os.makedirs(cache_dir_base, exist_ok=True)
                logging.info(
                    f"Successfully recreated cache directory: {cache_dir_base}"
                )
                QMessageBox.information(
                    self,
                    "Clear Cache",
                    "All application cache files have been cleared.",
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "Clear Cache Error", f"Failed to clear cache: {e}"
                )
                logging.error(
                    f"Error clearing cache directory {cache_dir_base}: {e}",
                    exc_info=True,
                )
        else:
            QMessageBox.information(self, "Clear Cache", "Cache clearing cancelled.")
            logging.info("Cache clearing cancelled by user.")

    # This block handles:
    # - Checking for required library dependencies.
    # - Setting up basic logging.


class AccountGroupingDialog(QDialog):
    """A dialog to manage account groups."""

    def __init__(
        self, parent, current_groups: Dict[str, List[str]], all_accounts: List[str]
    ):
        super().__init__(parent)
        self.setWindowTitle("Manage Account Groups")
        self.setMinimumSize(700, 500)

        self.groups = {
            k: list(v) for k, v in current_groups.items()
        }  # Make a mutable copy
        self.all_accounts = set(all_accounts)

        main_layout = QVBoxLayout(self)
        h_layout = QHBoxLayout()
        main_layout.addLayout(h_layout)

        # --- Left: Groups ---
        groups_layout = QVBoxLayout()
        groups_layout.addWidget(QLabel("<b>Groups</b>"))
        self.groups_list = QListWidget()
        self.groups_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.groups_list.customContextMenuRequested.connect(self.show_group_context_menu)
        self.groups_list.itemSelectionChanged.connect(self.on_group_selected)
        groups_layout.addWidget(self.groups_list)

        group_buttons_layout = QHBoxLayout()
        add_group_btn = QPushButton("Add")
        add_group_btn.clicked.connect(self.add_group)
        rename_group_btn = QPushButton("Rename")
        rename_group_btn.clicked.connect(self.rename_group)
        del_group_btn = QPushButton("Delete")
        del_group_btn.clicked.connect(self.delete_group)
        
        group_buttons_layout.addWidget(add_group_btn)
        group_buttons_layout.addWidget(rename_group_btn)
        group_buttons_layout.addWidget(del_group_btn)
        groups_layout.addLayout(group_buttons_layout)
        h_layout.addLayout(groups_layout, 1)

        # --- Middle: Accounts in Group ---
        accounts_in_group_layout = QVBoxLayout()
        accounts_in_group_layout.addWidget(QLabel("<b>Accounts in Selected Group</b>"))
        self.accounts_in_group_list = QListWidget()
        self.accounts_in_group_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.accounts_in_group_list.itemDoubleClicked.connect(self.move_accounts_out)
        self.accounts_in_group_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.accounts_in_group_list.customContextMenuRequested.connect(
            lambda pos: self.show_account_context_menu(pos, is_grouped=True)
        )
        accounts_in_group_layout.addWidget(self.accounts_in_group_list)
        h_layout.addLayout(accounts_in_group_layout, 1)

        # --- Arrow buttons ---
        arrow_layout = QVBoxLayout()
        arrow_layout.addStretch()
        self.move_in_btn = QPushButton("<<")
        self.move_in_btn.setToolTip("Add selected account(s) to group")
        self.move_in_btn.clicked.connect(self.move_accounts_in)
        self.move_out_btn = QPushButton(">>")
        self.move_out_btn.setToolTip("Remove selected account(s) from group")
        self.move_out_btn.clicked.connect(self.move_accounts_out)
        arrow_layout.addWidget(self.move_in_btn)
        arrow_layout.addWidget(self.move_out_btn)
        arrow_layout.addStretch()
        h_layout.addLayout(arrow_layout)

        # --- Right: Ungrouped Accounts ---
        ungrouped_layout = QVBoxLayout()
        ungrouped_layout.addWidget(QLabel("<b>Ungrouped Accounts</b>"))
        self.ungrouped_accounts_list = QListWidget()
        self.ungrouped_accounts_list.setSelectionMode(
            QAbstractItemView.ExtendedSelection
        )
        self.ungrouped_accounts_list.itemDoubleClicked.connect(self.move_accounts_in)
        self.ungrouped_accounts_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.ungrouped_accounts_list.customContextMenuRequested.connect(
            lambda pos: self.show_account_context_menu(pos, is_grouped=False)
        )
        ungrouped_layout.addWidget(self.ungrouped_accounts_list)
        h_layout.addLayout(ungrouped_layout, 1)

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)

        self.populate_lists()

    def populate_lists(self):
        # Remember selection if possible
        current_row = self.groups_list.currentRow()
        self.groups_list.clear()
        self.groups_list.addItems(sorted(self.groups.keys()))
        if current_row >= 0 and current_row < self.groups_list.count():
            self.groups_list.setCurrentRow(current_row)
        elif self.groups_list.count() > 0:
            self.groups_list.setCurrentRow(0)
        else:
            self.update_account_lists()

    def update_account_lists(self):
        self.accounts_in_group_list.clear()
        self.ungrouped_accounts_list.clear()

        selected_group_items = self.groups_list.selectedItems()
        
        # Calculate all currently grouped accounts across ALL groups
        grouped_accounts = set(
            acc for acc_list in self.groups.values() for acc in acc_list
        )
        
        # Ungrouped is everything else
        ungrouped = sorted(list(self.all_accounts - grouped_accounts))
        self.ungrouped_accounts_list.addItems(ungrouped)

        if selected_group_items:
            group_name = selected_group_items[0].text()
            self.accounts_in_group_list.addItems(
                sorted(self.groups.get(group_name, []))
            )
            self.move_in_btn.setEnabled(True)
            self.move_out_btn.setEnabled(True)
            self.accounts_in_group_list.setEnabled(True)
        else:
            self.move_in_btn.setEnabled(False)
            self.move_out_btn.setEnabled(False)
            self.accounts_in_group_list.setEnabled(False)

    def on_group_selected(self):
        self.update_account_lists()

    def add_group(self):
        group_name, ok = QInputDialog.getText(
            self, "Add Group", "Enter new group name:"
        )
        if ok and group_name:
            group_name = group_name.strip()
            if not group_name:
                return
            if group_name in self.groups:
                QMessageBox.warning(
                    self, "Duplicate", "A group with this name already exists."
                )
            else:
                self.groups[group_name] = []
                self.populate_lists()
                # Select the new group
                items = self.groups_list.findItems(group_name, Qt.MatchExactly)
                if items:
                    self.groups_list.setCurrentItem(items[0])

    def rename_group(self):
        selected_items = self.groups_list.selectedItems()
        if not selected_items:
            return
        old_name = selected_items[0].text()
        new_name, ok = QInputDialog.getText(
            self, "Rename Group", "Enter new group name:", text=old_name
        )
        if ok and new_name:
            new_name = new_name.strip()
            if not new_name or new_name == old_name:
                return
            if new_name in self.groups:
                QMessageBox.warning(
                    self, "Duplicate", "A group with this name already exists."
                )
                return
            
            # Preserve the list of accounts
            self.groups[new_name] = self.groups.pop(old_name)
            self.populate_lists()
            # Reselect
            items = self.groups_list.findItems(new_name, Qt.MatchExactly)
            if items:
                self.groups_list.setCurrentItem(items[0])

    def delete_group(self):
        selected_items = self.groups_list.selectedItems()
        if not selected_items:
            return
        group_name = selected_items[0].text()
        
        reply = QMessageBox.question(
            self, "Confirm Delete", 
            f"Are you sure you want to delete group '{group_name}'?\nAccounts will become ungrouped.",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            del self.groups[group_name]
            self.populate_lists()

    def move_accounts_in(self):
        selected_group_items = self.groups_list.selectedItems()
        selected_accounts = self.ungrouped_accounts_list.selectedItems()
        if not selected_group_items or not selected_accounts:
            return
        group_name = selected_group_items[0].text()
        for item in selected_accounts:
            self.groups[group_name].append(item.text())
        self.update_account_lists()

    def move_accounts_out(self):
        selected_group_items = self.groups_list.selectedItems()
        selected_accounts = self.accounts_in_group_list.selectedItems()
        if not selected_group_items or not selected_accounts:
            return
        group_name = selected_group_items[0].text()
        for item in selected_accounts:
            if item.text() in self.groups[group_name]:
                self.groups[group_name].remove(item.text())
        self.update_account_lists()

    def show_group_context_menu(self, pos):
        item = self.groups_list.itemAt(pos)
        if not item:
            return
        
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        delete_action = menu.addAction("Delete")
        
        action = menu.exec(self.groups_list.mapToGlobal(pos))
        
        if action == rename_action:
            self.rename_group()
        elif action == delete_action:
            self.delete_group()

    def show_account_context_menu(self, pos, is_grouped):
        list_widget = self.accounts_in_group_list if is_grouped else self.ungrouped_accounts_list
        item = list_widget.itemAt(pos)
        if not item:
            return

        menu = QMenu(self)
        if is_grouped:
            action_move = menu.addAction("Remove from Group")
            action_move.triggered.connect(self.move_accounts_out)
        else:
            action_move = menu.addAction("Add to Group")
            action_move.triggered.connect(self.move_accounts_in)
            
            # Only enable if a group is selected
            if not self.groups_list.selectedItems():
                action_move.setEnabled(False)

        menu.exec(list_widget.mapToGlobal(pos))

    def get_settings(self) -> Dict[str, List[str]]:
        return self.groups

    # - Initializing the QApplication.
    # - Creating and showing the main PortfolioApp window.
    # - Starting the Qt event loop.


if __name__ == "__main__":
    # --- ADDED: multiprocessing.freeze_support() ---
    # This is CRUCIAL for PyInstaller on macOS/Windows when using multiprocessing
    import multiprocessing

    multiprocessing.freeze_support()
    # --- END ADDED ---
    # --- Dependency Check ---
    # Check for essential libraries before starting the GUI
    missing_libs = []
    try:
        import yfinance
    except ImportError:
        missing_libs.append("yfinance")
    try:
        import matplotlib
    except ImportError:
        missing_libs.append("matplotlib")
    try:
        import scipy
    except ImportError:
        missing_libs.append("scipy")
    try:
        import pandas
    except ImportError:
        missing_libs.append("pandas")
    try:
        import numpy
    except ImportError:
        missing_libs.append("numpy")
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        missing_libs.append("PySide6")
    # --- MODIFICATION: Add inspect to dependency list (used for signature check) ---
    try:
        import inspect
    except ImportError:
        missing_libs.append("inspect")  # Should be built-in, but good practice

    logging.info("Investa GUI Application Starting")
    if missing_libs:
        logging.info(f"\nERROR: Missing required libraries: {', '.join(missing_libs)}")
        logging.info("Please install dependencies, for example using pip:")
        logging.info("pip install yfinance matplotlib scipy pandas numpy PySide6")
        # Show a simple message box if possible
        try:
            # Prevent creating multiple QApplications if run again after fixing
            app_check = QApplication.instance()
            if app_check is None:
                app_check = QApplication(sys.argv)  # Create if doesn't exist
            else:
                logging.info("QApplication instance already exists.")

            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Critical)
            msg_box.setWindowTitle("Dependency Error")
            msg_box.setText(f"Required libraries missing: {', '.join(missing_libs)}")
            msg_box.setInformativeText(
                "Please install required libraries (see console output for details)."
            )
            msg_box.setStandardButtons(QMessageBox.Ok)
            msg_box.exec()
        except Exception as e_msg:
            logging.info(
                f"Could not display GUI error message: {e_msg}"
            )  # Fallback print
        sys.exit(1)  # Exit if dependencies are missing

    # --- Application Setup ---
    # Ensure only one QApplication instance exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    # --->>> SET ORG AND APP NAME EARLY <<<---
    # This ensures QStandardPaths uses the correct names when PortfolioApp initializes its paths.
    app.setOrganizationName("StockAlchemist")  # Or your desired organization name
    app.setApplicationName(config.APP_NAME)  # Use APP_NAME from config.py
    # --->>> ------------ <<<---

    # --->>> IT MUST BE HERE <<<---s
    # Name does not show up in the menu bar
    app.setApplicationName(config.APP_NAME)  # <--- Use the constant
    # --->>> ------------ <<<---
    logging.debug(f"DEBUG: QApplication name set to: {app.applicationName()}")

    main_window = PortfolioApp()

    # Set initial size and minimum size
    main_window.resize(1600, 1000)  # Adjust initial size as needed
    main_window.setMinimumSize(1200, 700)  # Set minimum reasonable size

    # --- Center the window on the primary screen ---
    try:
        screen_geometry = QApplication.primaryScreen().availableGeometry()
        window_geometry = main_window.frameGeometry()
        center_point = screen_geometry.center()
        window_geometry.moveCenter(center_point)
        main_window.move(window_geometry.topLeft())
        logging.info(f"Centering window at: {window_geometry.topLeft()}")
    except Exception as e:
        logging.warning(f"Warning: Could not center the window: {e}")
        # Optionally set a default position if centering fails
        # main_window.move(100, 100)

    # --- Show Window and Run Event Loop ---
    main_window.show()
    sys.exit(app.exec())


# --- END OF FILE main_gui.py ---
