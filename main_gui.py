# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          main_gui.py
 Purpose:       Main application window and GUI logic for Investa Portfolio Dashboard.
                Handles UI elements, user interaction, background tasks, and visualization.

 Author:        Kit Matan
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
import shutil
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Set  # Added Set
import logging

# --- Configure Logging Globally (as early as possible) ---
# Set the desired global level here (e.g., logging.INFO, logging.DEBUG)
LOGGING_LEVEL = logging.INFO  # Or logging.DEBUG for more detail

logging.basicConfig(
    level=LOGGING_LEVEL,
    format="%(asctime)s [%(levelname)-8s] %(name)-15s %(module)s:%(lineno)d - %(message)s",  # Added logger name
    datefmt="%Y-%m-%d %H:%M:%S",
    # Use force=True (Python 3.8+) to ensure this config takes precedence
    # Remove if using older Python or facing issues with existing handlers
    force=True,
)

# Quieten overly verbose libraries (optional, but often useful)
logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)
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
    QStyle,
    QDateEdit,
    QLineEdit,
    QMenu,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
)

from PySide6.QtGui import QColor, QPalette, QFont, QIcon, QPixmap, QAction
from PySide6.QtCore import (
    Qt,
    QAbstractTableModel,
    QThreadPool,
    QRunnable,
    Signal,
    Slot,
    QObject,
    QDateTime,
    QDate,
    QPoint,
)

import matplotlib

matplotlib.use("QtAgg")
import matplotlib.pyplot as plt

# --- Matplotlib Font Configuration (unchanged) ---
try:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "Arial",  # Or Arial, Tahoma, Verdana, Segoe UI
            "axes.labelcolor": "#333333",
            "xtick.color": "#666666",
            "ytick.color": "#666666",
            "text.color": "#333333",
        }
    )
    logging.debug("Matplotlib default font configured.")
except Exception as e:
    logging.warning(f"Warning: Could not configure Matplotlib font: {e}")
# --- End Matplotlib Font Configuration ---


from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.ticker as mtick

# --- Import Business Logic from portfolio_logic ---
try:
    # --- MODIFICATION: Check if the imported function actually supports exclude_accounts ---
    # We assume it might not, based on the error. The GUI will pass include_accounts,
    # but we will NOT pass exclude_accounts for now.
    from portfolio_logic import (
        calculate_portfolio_summary,
        fetch_index_quotes_yfinance,
        CASH_SYMBOL_CSV,
        calculate_historical_performance,
        load_and_clean_transactions,
    )

    LOGIC_AVAILABLE = True
    # Check if the imported function signature actually supports 'exclude_accounts'
    import inspect

    sig = inspect.signature(calculate_historical_performance)
    HISTORICAL_FN_SUPPORTS_EXCLUDE = "exclude_accounts" in sig.parameters
    if not HISTORICAL_FN_SUPPORTS_EXCLUDE:
        logging.warning(
            "Warn: Imported 'calculate_historical_performance' does NOT support 'exclude_accounts' argument. Exclusion logic in GUI will be ignored."
        )

except ImportError as import_err:
    logging.error(
        f"ERROR: portfolio_logic.py not found or missing required functions: {import_err}"
    )
    logging.info(
        "Ensure portfolio_logic.py contains the v1.0 calculate_historical_performance with include_accounts."
    )
    LOGIC_AVAILABLE = False
    HISTORICAL_FN_SUPPORTS_EXCLUDE = False  # Assume false if import fails
    CASH_SYMBOL_CSV = "__CASH__"

    def calculate_portfolio_summary(*args, **kwargs):
        return {}, pd.DataFrame(), pd.DataFrame(), {}, "Error: Logic missing"

    def fetch_index_quotes_yfinance(*args, **kwargs):
        return {}

    def calculate_historical_performance(*args, **kwargs):
        # Remove unexpected arg if present in dummy function call
        kwargs.pop("exclude_accounts", None)
        logging.warning(
            "Warn: Using dummy calculate_historical_performance (v1.0 signature)"
        )
        dummy_cols = ["Portfolio Value", "Portfolio Accumulated Gain"]
        if "benchmark_symbols_yf" in kwargs and isinstance(
            kwargs["benchmark_symbols_yf"], list
        ):
            for sym in kwargs["benchmark_symbols_yf"]:
                dummy_cols.extend([f"{sym} Price", f"{sym} Accumulated Gain"])
        return pd.DataFrame(columns=dummy_cols), "Error: Logic missing"

    logging.warning(f"Warning: Using fallback CASH_SYMBOL_CSV: {CASH_SYMBOL_CSV}")
except Exception as import_err:
    logging.error(
        f"ERROR: Unexpected error importing from portfolio_logic.py: {import_err}"
    )
    traceback.print_exc()
    LOGIC_AVAILABLE = False
    HISTORICAL_FN_SUPPORTS_EXCLUDE = False  # Assume false on error
    CASH_SYMBOL_CSV = "__CASH__"

    def calculate_portfolio_summary(*args, **kwargs):
        return {}, pd.DataFrame(), pd.DataFrame(), {}, "Error: Logic import failed"

    def fetch_index_quotes_yfinance(*args, **kwargs):
        return {}

    def calculate_historical_performance(*args, **kwargs):
        # Remove unexpected arg if present in dummy function call
        kwargs.pop("exclude_accounts", None)
        return pd.DataFrame(), "Error: Logic import failed"

    logging.warning(f"Warning: Using fallback CASH_SYMBOL_CSV: {CASH_SYMBOL_CSV}")

# --- Constants ---
DEFAULT_CSV = "my_transactions.csv"  # Default transaction file name
DEFAULT_API_KEY = os.getenv("FMP_API_KEY")  # Optional API key from environment
CONFIG_FILE = "gui_config.json"  # Configuration file name
CHART_MAX_SLICES = 10  # Max slices before grouping into 'Other' in pie charts
PIE_CHART_FIG_SIZE = (5.0, 2.5)  # Figure size for pie charts
PERF_CHART_FIG_SIZE = (7.5, 3.0)  # Figure size for performance graphs
CHART_DPI = 95  # Dots per inch for charts
INDICES_FOR_HEADER = [".DJI", "IXIC", ".INX"]  # Indices to display in the header
CSV_DATE_FORMAT = "%b %d, %Y"  # Define the specific date format used in the CSV

# --- Graph Defaults ---
DEFAULT_GRAPH_START_DATE = date.today() - timedelta(
    days=365 * 2
)  # Default start 2 years ago
DEFAULT_GRAPH_END_DATE = date.today()  # Default end today
DEFAULT_GRAPH_INTERVAL = "W"  # Default interval Weekly
DEFAULT_GRAPH_BENCHMARKS = ["SPY"]  # Default to a list with SPY benchmark
# --- Benchmark Options ---
BENCHMARK_OPTIONS = [
    "SPY",
    "QQQ",
    "DIA",
    "^SP500TR",
    "^GSPC",
    "VT",
    "IWM",
    "EFA",
    "TLT",
    "AGG",
]  # Available benchmark choices

# --- Theme Colors ---
# Define color palette for styling the UI (Minimal Theme)
COLOR_BG_DARK = "#FFFFFF"
COLOR_BG_HEADER_LIGHT = "#F8F9FA"
COLOR_BG_HEADER_ORIGINAL = "#495057"  # Dark grey for header text/borders
COLOR_BG_CONTROLS = "#FFFFFF"
COLOR_BG_SUMMARY = "#F8F9FA"  # Light grey for summary background
COLOR_BG_CONTENT = "#FFFFFF"
COLOR_TEXT_LIGHT = "#FFFFFF"
COLOR_TEXT_DARK = "#212529"  # Dark text
COLOR_TEXT_SECONDARY = "#6C757D"  # Grey text for labels, status
COLOR_ACCENT_TEAL = "#6C757D"  # Grey/teal accent
COLOR_ACCENT_TEAL_LIGHT = "#F8F9FA"  # Very light grey for headers
COLOR_ACCENT_AMBER = "#E9ECEF"  # Light grey for buttons, selection background
COLOR_ACCENT_AMBER_DARK = "#CED4DA"  # Slightly darker grey for button hover
COLOR_GAIN = "#198754"  # Green for gains
COLOR_LOSS = "#DC3545"  # Red for losses
COLOR_BORDER_LIGHT = "#DEE2E6"  # Light grey border
COLOR_BORDER_DARK = "#ADB5BD"
# Medium grey border

# Convert hex colors to QColor objects for easier use in Qt palettes
QCOLOR_GAIN = QColor(COLOR_GAIN)
QCOLOR_LOSS = QColor(COLOR_LOSS)
QCOLOR_TEXT_DARK = QColor(COLOR_TEXT_DARK)
QCOLOR_TEXT_SECONDARY = QColor(COLOR_TEXT_SECONDARY)


# --- Column Definition Helper ---
def get_column_definitions(display_currency="USD"):
    """Generates a mapping from UI table headers to DataFrame column names.

    This allows the UI header text to be independent of the underlying column names
    generated by the portfolio logic, especially handling currency-specific columns.

    Args:
        display_currency (str, optional): The currency code (e.g., "USD", "THB")
            to incorporate into currency-dependent column names. Defaults to "USD".

    Returns:
        Dict[str, str]: A dictionary where keys are user-friendly UI header names
            (e.g., 'Mkt Val') and values are the corresponding actual DataFrame
            column names expected from `portfolio_logic` (e.g., 'Market Value (USD)').
    """
    return {
        # UI Header Name : Actual DataFrame Column Name (potentially with currency)
        "Account": "Account",
        "Symbol": "Symbol",
        "Quantity": "Quantity",
        f"Day Chg": f"Day Change ({display_currency})",
        "Day Chg %": "Day Change %",
        "Avg Cost": f"Avg Cost ({display_currency})",
        "Price": f"Price ({display_currency})",
        "Cost Basis": f"Cost Basis ({display_currency})",
        "Mkt Val": f"Market Value ({display_currency})",
        "Unreal. G/L": f"Unreal. Gain ({display_currency})",
        "Unreal. G/L %": "Unreal. Gain %",
        "Real. G/L": f"Realized Gain ({display_currency})",
        "Divs": f"Dividends ({display_currency})",
        "Fees": f"Commissions ({display_currency})",
        "Total G/L": f"Total Gain ({display_currency})",
        "Total Ret %": "Total Return %",  # Calculated using Cumulative Investment
        "IRR (%)": "IRR (%)",
        # Optional columns that might be added for debugging or other features:
        # 'Cumulative Investment': f'Cumulative Investment ({display_currency})',
        # 'Price Source': 'Price Source',
    }


# --- Helper Classes for Background Processing ---


class WorkerSignals(QObject):
    """Defines signals available from a running worker thread (QRunnable).

    Signals:
        finished: Emitted when the worker task has completed, regardless of success.
        error: Emitted when an error occurs during the worker task. Passes a
               string describing the error.
        result: Emitted upon successful completion of the task. Passes the
                calculated results: summary metrics (dict), holdings DataFrame,
                ignored transactions DataFrame, account-level metrics (dict),
                index quotes (dict), and historical performance DataFrame.
    """

    finished = Signal()
    error = Signal(str)
    # Args: summary_metrics, holdings_df, ignored_df, account_metrics, index_quotes, historical_data_df
    result = Signal(dict, pd.DataFrame, pd.DataFrame, dict, dict, pd.DataFrame)


class PortfolioCalculatorWorker(QRunnable):
    """
    Worker thread (QRunnable) for performing portfolio calculations.

    Executes portfolio summary, index quote fetching, and historical performance
    calculations in a separate thread to avoid blocking the main GUI thread.
    Uses signals defined in WorkerSignals to communicate results or errors back.
    """

    def __init__(
        self,
        portfolio_fn,
        portfolio_args,
        portfolio_kwargs,
        index_fn,
        historical_fn,
        historical_args,
        historical_kwargs,
    ):
        """
        Initializes the worker with calculation functions and arguments.

        Args:
            portfolio_fn (callable): The function to calculate the current portfolio summary.
            portfolio_args (tuple): Positional arguments for portfolio_fn.
            portfolio_kwargs (dict): Keyword arguments for portfolio_fn.
            index_fn (callable): The function to fetch index quotes.
            historical_fn (callable): The function to calculate historical performance.
            historical_args (tuple): Positional arguments for historical_fn.
            historical_kwargs (dict): Keyword arguments for historical_fn.
        """
        super().__init__()
        self.portfolio_fn = portfolio_fn
        self.portfolio_args = portfolio_args
        # portfolio_kwargs will contain account_currency_map and default_currency
        self.portfolio_kwargs = portfolio_kwargs
        self.index_fn = index_fn
        self.historical_fn = historical_fn
        self.historical_args = historical_args
        # historical_kwargs will contain account_currency_map and default_currency
        self.historical_kwargs = historical_kwargs
        self.signals = WorkerSignals()
        self.original_data = pd.DataFrame()

    @Slot()
    def run(self):
        """
        Executes the portfolio, index, and historical calculations sequentially.

        Calls the provided functions, handles potential exceptions during each step,
        and emits the 'result' signal with all collected data upon successful
        completion, or the 'error' signal if any critical step fails. Finally,
        emits the 'finished' signal.

        Signals:
            Emits signals defined in WorkerSignals (result, error, finished).
        """
        portfolio_summary_metrics = {}
        holdings_df = pd.DataFrame()
        ignored_df = pd.DataFrame()
        account_metrics = {}
        index_quotes = {}
        historical_data_df = pd.DataFrame()
        portfolio_status = "Error: Portfolio calculation did not run"
        historical_status = "Info: Historical calculation pending"
        overall_status = "Error: Worker did not complete initialization"

        try:
            # --- 1. Run Portfolio Summary Calculation ---
            try:
                # No changes needed here, kwargs are passed directly
                logging.debug(
                    f"DEBUG Worker: Calling portfolio_fn with kwargs keys: {list(self.portfolio_kwargs.keys())}"
                )
                p_summary, p_holdings, p_ignored, p_account, p_status = (
                    self.portfolio_fn(*self.portfolio_args, **self.portfolio_kwargs)
                )
                portfolio_summary_metrics = p_summary if p_summary is not None else {}
                holdings_df = p_holdings if p_holdings is not None else pd.DataFrame()
                ignored_df = p_ignored if p_ignored is not None else pd.DataFrame()
                account_metrics = p_account if p_account is not None else {}
                portfolio_status = (
                    p_status if p_status else "Error: Unknown portfolio status"
                )
                if isinstance(portfolio_summary_metrics, dict):
                    portfolio_summary_metrics["status_msg"] = portfolio_status
            except Exception as port_e:
                logging.info(
                    f"--- Error during portfolio calculation in worker: {port_e} ---"
                )
                traceback.print_exc()
                portfolio_status = f"Error in Portfolio Calc: {port_e}"
                portfolio_summary_metrics, holdings_df, ignored_df, account_metrics = (
                    {},
                    pd.DataFrame(),
                    pd.DataFrame(),
                    {},
                )

            # --- 2. Fetch Index Quotes (unchanged) ---
            try:
                logging.debug("DEBUG Worker: Fetching index quotes...")
                index_quotes = self.index_fn()
                logging.debug(
                    f"DEBUG Worker: Index quotes fetched ({len(index_quotes)} items)."
                )
            except Exception as idx_e:
                logging.info(
                    f"--- Error during index quote fetch in worker: {idx_e} ---"
                )
                traceback.print_exc()
                index_quotes = {}

            # --- 3. Run Historical Performance Calculation (unchanged wrt currency map) ---
            try:
                # --- MODIFICATION: Conditionally remove exclude_accounts if not supported ---
                current_historical_kwargs = self.historical_kwargs.copy()
                if (
                    not HISTORICAL_FN_SUPPORTS_EXCLUDE
                    and "exclude_accounts" in current_historical_kwargs
                ):
                    logging.debug(
                        "DEBUG Worker: Removing 'exclude_accounts' from historical_fn call as it's not supported by the imported function."
                    )
                    current_historical_kwargs.pop("exclude_accounts")
                # --- End Modification ---

                # No changes needed here, kwargs are passed directly
                logging.debug(
                    f"DEBUG Worker: Calling historical_fn with kwargs keys: {list(current_historical_kwargs.keys())}"
                )
                hist_df, hist_status = self.historical_fn(
                    *self.historical_args,
                    **current_historical_kwargs,  # Use the potentially modified kwargs
                )
                historical_data_df = hist_df if hist_df is not None else pd.DataFrame()
                historical_status = (
                    hist_status if hist_status else "Error: Unknown historical status"
                )
                if isinstance(portfolio_summary_metrics, dict):
                    portfolio_summary_metrics["historical_status_msg"] = (
                        historical_status
                    )
                logging.debug(
                    f"DEBUG Worker: Historical calculation finished. Status: {historical_status}"
                )
            except Exception as hist_e:
                # ... (existing error handling for historical calc) ...
                if isinstance(
                    hist_e, TypeError
                ) and "unexpected keyword argument 'exclude_accounts'" in str(hist_e):
                    logging.info(
                        f"--- Error during historical performance calculation in worker: {hist_e} ---"
                    )
                    logging.info(
                        "--- This likely means the portfolio_logic.py version doesn't support 'exclude_accounts'. ---"
                    )
                    historical_status = (
                        f"Error: Hist. Calc doesn't support 'exclude_accounts'"
                    )
                else:
                    logging.info(
                        f"--- Error during historical performance calculation in worker: {hist_e} ---"
                    )
                    traceback.print_exc()
                    historical_status = f"Error in Hist. Calc: {hist_e}"
                historical_data_df = pd.DataFrame()

            # --- 4. Prepare and Emit Combined Results (unchanged) ---
            overall_status = (
                f"Portfolio: {portfolio_status} | Historical: {historical_status}"
            )
            if any(err in portfolio_status for err in ["Error", "Crit"]) or any(
                err in historical_status for err in ["Error", "Crit", "Fail", "Halt"]
            ):
                self.signals.error.emit(overall_status)

            self.signals.result.emit(
                portfolio_summary_metrics,
                holdings_df,
                ignored_df,
                account_metrics,
                index_quotes,
                historical_data_df,
            )
        except Exception as e:
            logging.info(f"--- Critical Error in Worker Thread run method: {e} ---")
            traceback.print_exc()
            overall_status = f"CritErr in Worker: {e}"
            self.signals.result.emit(
                {}, pd.DataFrame(), pd.DataFrame(), {}, {}, pd.DataFrame()
            )
            self.signals.error.emit(overall_status)
        finally:
            logging.debug("DEBUG Worker: Emitting finished signal.")
            self.signals.finished.emit()


# --- Pandas Model for TableView ---
class PandasModel(QAbstractTableModel):
    """A Qt Table Model for displaying pandas DataFrames in a QTableView.

    Handles data access, header information, sorting (with special handling for
    cash rows), and custom cell formatting (alignment, color, currency/percentage).
    """

    def __init__(self, data=pd.DataFrame(), parent=None):
        """
        Initializes the model with optional data and a parent reference.

        Args:
            data (pd.DataFrame, optional): The initial DataFrame to display.
                                           Defaults to an empty DataFrame.
            parent (QWidget, optional): The parent widget, typically the main
                                        application window, used to access shared
                                        information like the current currency symbol.
                                        Defaults to None.
        """
        super().__init__(parent)
        self._data = data
        self._parent = (
            parent  # Store reference to PortfolioApp for currency symbol etc.
        )
        self._default_text_color = QCOLOR_TEXT_DARK
        self._currency_symbol = "$"  # Default currency symbol

    def updateCurrencySymbol(self, symbol):
        """
        Updates the currency symbol used for formatting monetary values.

        Args:
            symbol (str): The currency symbol (e.g., "$", "฿").
        """
        self._currency_symbol = symbol

    def rowCount(self, parent=None):
        """Returns the number of rows in the model (DataFrame)."""
        return self._data.shape[0]

    def columnCount(self, parent=None):
        """Returns the number of columns in the model (DataFrame)."""
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        """
        Returns data or display properties for a specific cell.

        Handles various roles:
        - Qt.DisplayRole: Returns the formatted text to display.
        - Qt.TextAlignmentRole: Returns the alignment for the cell content.
        - Qt.ForegroundRole: Returns the text color (e.g., green for gain, red for loss).

        Args:
            index (QModelIndex): The index of the cell for which data is requested.
            role (Qt.ItemDataRole): The role for which data is requested.

        Returns:
            Any: The requested data (e.g., str, QColor, Qt.AlignmentFlag) or None
                 if the index is invalid or the role is not handled.
        """
        if not index.isValid():
            return None
        col = index.column()
        row = index.row()
        # Bounds check
        if row >= self.rowCount() or col >= self.columnCount():
            return None

        # --- Text Alignment ---
        # ... (Alignment logic remains the same) ...
        if role == Qt.TextAlignmentRole:
            alignment = int(Qt.AlignLeft | Qt.AlignVCenter)  # Default left
            try:
                col_name = self._data.columns[col]  # UI Column Name
                col_data = self._data.iloc[:, col]
                if col_name in ["Account", "Symbol", "Price Source"]:
                    alignment = int(Qt.AlignLeft | Qt.AlignVCenter)
                elif pd.api.types.is_numeric_dtype(col_data.dtype):
                    alignment = int(Qt.AlignRight | Qt.AlignVCenter)
                elif col_data.dtype == "object":
                    is_potentially_numeric_by_name = (
                        any(
                            indicator in col_name
                            for indicator in [
                                "%",
                                " G/L",
                                " Price",
                                " Cost",
                                " Val",
                                " Divs",
                                " Fees",
                                " Basis",
                                " Avg",
                                " Chg",
                                "Quantity",
                                "IRR",
                                " Mkt",
                                " Ret %",
                            ]
                        )
                        or f"({self._parent.currency_combo.currentText()})" in col_name
                    )
                    if is_potentially_numeric_by_name:
                        alignment = int(Qt.AlignRight | Qt.AlignVCenter)
            except (IndexError, AttributeError, KeyError) as e:
                pass
            return alignment

        # --- Text Color (Foreground) ---
        # ... (Coloring logic remains the same) ...
        if role == Qt.ForegroundRole:
            try:
                col_name = self._data.columns[col]
                value = self._data.iloc[row, col]
                target_color = self._default_text_color
                if pd.api.types.is_number(value) and pd.notna(value):
                    value_float = float(value)
                    gain_loss_color_cols = [
                        "Gain",
                        "Return",
                        "IRR",
                        "Day Change",
                        "Day Chg",
                        "Total G/L",
                        "Unreal. G/L",
                        "Real. G/L",
                        "Total Ret %",
                        "Unreal. G/L %",
                        "Day Chg %",
                        "IRR (%)",
                    ]  # Added IRR (%) here for coloring
                    if any(indicator in col_name for indicator in gain_loss_color_cols):
                        if value_float > 1e-9:
                            target_color = QCOLOR_GAIN
                        elif value_float < -1e-9:
                            target_color = QCOLOR_LOSS
                    elif "Dividend" in col_name or "Divs" in col_name:
                        if value_float > 1e-9:
                            target_color = QCOLOR_GAIN
                    elif (
                        "Commission" in col_name
                        or "Fee" in col_name
                        or "Fees" in col_name
                    ):
                        if value_float > 1e-9:
                            target_color = QCOLOR_LOSS
                return target_color
            except Exception as e:
                # logging.info(f"Coloring Error (Row:{row}, Col:{col}, Name:'{self._data.columns[col]}'): {e}")
                return self._default_text_color

        # --- Display Text ---
        if role == Qt.DisplayRole or role == Qt.EditRole:
            original_value = "ERR"  # Default in case of early error
            try:
                original_value = self._data.iloc[row, col]
                col_name = self._data.columns[col]  # UI Column Name

                # --- Special Formatting for CASH Symbol ---
                # ... (Cash formatting remains the same) ...
                try:
                    symbol_col_idx = self._data.columns.get_loc("Symbol")
                    symbol_value = self._data.iloc[row, symbol_col_idx]
                    if col_name == "Total Ret %" and symbol_value == CASH_SYMBOL_CSV:
                        return "-"
                    # Specific handling for IRR % for cash: It should be N/A
                    if col_name == "IRR (%)" and symbol_value == CASH_SYMBOL_CSV:
                        return "-"  # Display '-' for cash IRR
                    if col_name == "Symbol" and symbol_value == CASH_SYMBOL_CSV:
                        display_currency_name = (
                            self._parent._get_currency_symbol(get_name=True)
                            if self._parent
                            else "CUR"
                        )
                        return f"Cash ({display_currency_name})"
                except (KeyError, IndexError):
                    pass

                # --- Handle NaN/None values ---
                if pd.isna(original_value):
                    return "-"

                # --- Formatting based on value type and column name ---
                if isinstance(original_value, (int, float, np.number)):
                    value_float = float(original_value)
                    display_value_float = abs(value_float)
                    if abs(value_float) < 1e-9:
                        display_value_float = 0.0

                    if "Quantity" in col_name:
                        return f"{value_float:,.4f}"  # Keep original sign for Quantity

                    # Combined Percentage and IRR formatting
                    elif (
                        "%" in col_name
                    ):  # Check if it's a percentage column (incl. IRR(%))
                        if np.isinf(value_float):
                            return "Inf %"
                        # Use original signed value for percentages (already scaled if IRR)
                        return f"{value_float:,.2f}%"  # Add % sign

                    # --- MODIFIED Currency Check & Formatting ---
                    elif self._parent and hasattr(self._parent, "_get_currency_symbol"):
                        currency_ui_names = [
                            "Avg Cost",
                            "Price",
                            "Cost Basis",
                            "Mkt Val",
                            "Unreal. G/L",
                            "Real. G/L",
                            "Divs",
                            "Fees",
                            "Total G/L",
                            f"Day Chg",
                        ]
                        is_currency_col = col_name in currency_ui_names
                        if is_currency_col:
                            current_currency_symbol = self._get_currency_symbol_safe()
                            return (
                                f"{current_currency_symbol}{display_value_float:,.2f}"
                            )
                    # --- END MODIFIED Currency Check & Formatting ---

                    # Fallback for other numeric types (should be very rare now)
                    # This should theoretically not be hit often if col names are consistent
                    else:
                        return f"{value_float:,.2f}"  # Default formatting, keeps sign

                # General fallback for non-numeric types
                return str(original_value)

            except Exception as e:
                col_name_str = "OOB"
                try:
                    col_name_str = self._data.columns[col]
                except IndexError:
                    pass
                val_repr = (
                    repr(original_value) if "original_value" in locals() else "N/A"
                )
                logging.info(
                    f"Display Format Error (Row:{row}, Col:{col}, Name:'{col_name_str}', Value:'{val_repr}'): {e}"
                )
                return "FmtErr"

        return None  # Default return for unhandled roles

    def _get_currency_symbol_safe(self):
        """
        Safely retrieves the currency symbol from the parent widget.

        Returns:
            str: The currency symbol (e.g., "$") or a default "$" if retrieval fails.
        """
        if self._parent and hasattr(self._parent, "_get_currency_symbol"):
            try:
                return self._parent._get_currency_symbol()
            except Exception:  # Catch potential errors during symbol retrieval
                pass
        return "$"  # Default fallback symbol

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Returns the header data (column names or row numbers).

        Args:
            section (int): The column or row index.
            orientation (Qt.Orientation): Qt.Horizontal for columns, Qt.Vertical for rows.
            role (Qt.ItemDataRole): The role requested (typically Qt.DisplayRole).

        Returns:
            str | None: The header text or None if the role is not handled.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:  # Column headers
                try:
                    # The model's internal _data DataFrame now has UI-friendly column names
                    return str(self._data.columns[section])
                except IndexError:
                    return ""  # Return empty string if index out of bounds
            if orientation == Qt.Vertical:  # Row headers (row numbers)
                return str(section + 1)
        return None

    def updateData(self, data):
        """
        Updates the model with a new DataFrame.

        Notifies the view that the model is resetting and updates the internal
        DataFrame. Handles None or non-DataFrame inputs gracefully.

        Args:
            data (pd.DataFrame | None): The new DataFrame to display, or None to clear.
        """
        self.beginResetModel()  # Notify view that model is about to change drastically
        if data is None:
            self._data = pd.DataFrame()  # Use empty DataFrame if None provided
        elif isinstance(data, pd.DataFrame):
            self._data = data.copy()  # Use a copy to prevent external modification
        else:
            # Attempt conversion if not a DataFrame, default to empty on failure
            try:
                self._data = pd.DataFrame(data)
            except Exception:
                self._data = pd.DataFrame()

        # Update currency symbol from parent app when data changes
        if self._parent and hasattr(self._parent, "_get_currency_symbol"):
            self.updateCurrencySymbol(self._parent._get_currency_symbol())

        self.endResetModel()  # Notify view that model change is complete

    def sort(self, column, order):
        """
        Sorts the underlying DataFrame based on a column.

        Separates cash rows to keep them at the bottom. Attempts numeric sorting
        for columns likely containing numbers, otherwise sorts as strings.
        Handles NaNs appropriately. Notifies the view before and after sorting.

        Args:
            column (int): The index of the column to sort by.
            order (Qt.SortOrder): The sort order (Qt.AscendingOrder or Qt.DescendingOrder).
        """
        if self._data.empty:
            logging.debug("Sort called on empty model. Skipping.")
            return  # Nothing to sort

        try:
            # Basic validation of column index
            if column < 0 or column >= self.columnCount():
                logging.warning(
                    f"Warning: Sort called with invalid column index {column}"
                )
                return

            col_name = self._data.columns[
                column
            ]  # Get the UI/Actual column name being sorted
            ascending_order = order == Qt.AscendingOrder
            logging.info(
                f"Sorting by column: '{col_name}' (Index: {column}), Ascending: {ascending_order}"
            )

            self.layoutAboutToBeChanged.emit()  # Notify view about layout change start

            # --- Separate Cash Rows (MODIFIED to handle both symbol column names) ---
            cash_rows = pd.DataFrame()
            non_cash_rows = self._data.copy()  # Start assuming all rows are non-cash

            # --- Determine the correct Symbol column name ---
            symbol_col_name_internal = "Symbol"
            symbol_col_name_original = "Stock / ETF Symbol"
            actual_symbol_col = None
            if symbol_col_name_internal in self._data.columns:
                actual_symbol_col = symbol_col_name_internal
            elif symbol_col_name_original in self._data.columns:
                actual_symbol_col = symbol_col_name_original
            # --- End Symbol column determination ---

            if actual_symbol_col:  # Check if we found a symbol column
                try:
                    base_cash_symbol = CASH_SYMBOL_CSV
                    # Check for both possible cash representations
                    cash_mask = (
                        self._data[actual_symbol_col].astype(str) == base_cash_symbol
                    )

                    # Add check for display format "Cash (CUR)" only if using internal name "Symbol"
                    # and if the parent application context is available to get the currency name.
                    if actual_symbol_col == symbol_col_name_internal and self._parent:
                        try:
                            # Ensure _get_currency_symbol exists and is callable
                            if hasattr(
                                self._parent, "_get_currency_symbol"
                            ) and callable(self._parent._get_currency_symbol):
                                display_currency_name = (
                                    self._parent._get_currency_symbol(get_name=True)
                                )
                                cash_display_symbol = f"Cash ({display_currency_name})"
                                cash_mask |= (
                                    self._data[actual_symbol_col].astype(str)
                                    == cash_display_symbol
                                )
                            else:
                                logging.debug(
                                    "Parent lacks _get_currency_symbol method for cash sort check."
                                )
                        except Exception as e_disp_name:
                            logging.warning(
                                f"Could not get currency name for cash sort check: {e_disp_name}"
                            )

                    if cash_mask.any():
                        cash_rows = self._data[cash_mask].copy()
                        non_cash_rows = self._data[~cash_mask].copy()
                        logging.debug(
                            f"DEBUG Sort: Separated {len(cash_rows)} cash rows using '{actual_symbol_col}'."
                        )
                except Exception as e_cash_sep:
                    logging.warning(
                        f"Warning: Error separating cash rows during sort: {e_cash_sep}"
                    )
                    cash_rows = pd.DataFrame()  # Ensure cash_rows is empty
                    non_cash_rows = (
                        self._data.copy()
                    )  # Reset non_cash_rows to full data
            else:
                # This warning will now only appear if NEITHER symbol column name is found
                logging.warning(
                    f"Warning: Could not find '{symbol_col_name_internal}' or '{symbol_col_name_original}' column, cannot separate cash for sorting."
                )
            # --- End Cash Row Separation Modification ---

            # --- Sort Non-Cash Rows using pandas sort_values (MODIFIED Date Handling) ---
            sorted_non_cash_rows = pd.DataFrame()  # Initialize empty
            if not non_cash_rows.empty:
                try:
                    # Determine if the column *should* be numeric, date, or string based on name/content
                    col_data_for_check = non_cash_rows[col_name]

                    # --- Column Type Heuristics ---
                    # Check based on name first
                    is_potentially_numeric_by_name = any(
                        indicator in col_name
                        for indicator in [
                            "%",
                            " G/L",
                            " Price",
                            " Cost",
                            " Val",
                            " Divs",
                            " Fees",
                            " Basis",
                            " Avg",
                            " Chg",
                            "Quantity",
                            "IRR",
                            " Mkt",
                            " Ret %",
                            " Ratio",
                        ]
                    ) or (
                        self._parent
                        and hasattr(self._parent, "_get_currency_symbol")
                        and callable(self._parent._get_currency_symbol)
                        and f"({self._parent._get_currency_symbol(get_name=True)})"
                        in col_name
                    )

                    # Handle original CSV headers that should be numeric
                    original_numeric_headers = [
                        "Quantity of Units",
                        "Amount per unit",
                        "Total Amount",
                        "Fees",
                        "Split Ratio (new shares per old share)",
                    ]
                    if col_name in original_numeric_headers:
                        is_potentially_numeric_by_name = True

                    # Check actual dtype if name didn't trigger
                    is_numeric_dtype = pd.api.types.is_numeric_dtype(
                        col_data_for_check.dtype
                    )
                    is_potentially_numeric = (
                        is_potentially_numeric_by_name or is_numeric_dtype
                    )

                    # Explicitly treat Account and Symbol columns as strings for sorting
                    string_col_names = [
                        "Account",
                        "Symbol",
                        "Investment Account",
                        "Stock / ETF Symbol",
                        "Transaction Type",
                        "Price Source",
                        "Note",
                        "Local Currency",
                        "Reason Ignored",
                    ]
                    is_string_col = col_name in string_col_names

                    # Check for Date Column (using both possible names)
                    date_col_name_internal = "Date"
                    date_col_name_original = "Date (MMM DD, YYYY)"
                    is_date_col = col_name in [
                        date_col_name_internal,
                        date_col_name_original,
                    ]
                    # --- End Column Type Heuristics ---

                    # Define key functions
                    # Attempt to strip common currency symbols and commas before numeric conversion
                    def numeric_key_func(x_series):
                        if not pd.api.types.is_string_dtype(x_series):
                            x_series = x_series.astype(
                                str
                            )  # Convert to string if not already
                        # Remove currency symbols (add more if needed) and commas
                        cleaned_series = x_series.str.replace(
                            r"[$,฿€£¥]", "", regex=True
                        ).str.replace(",", "", regex=False)
                        return pd.to_numeric(cleaned_series, errors="coerce")

                    def date_key_func(x_series):
                        # Try specific format first if applicable
                        date_format_to_try = (
                            CSV_DATE_FORMAT
                            if col_name == date_col_name_original
                            else None
                        )
                        return pd.to_datetime(
                            x_series, errors="coerce", format=date_format_to_try
                        )

                    # --- Sorting Logic ---
                    if is_date_col:
                        logging.debug(
                            f"DEBUG Sort: Using DATE sort strategy for '{col_name}'."
                        )
                        sorted_non_cash_rows = non_cash_rows.sort_values(
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",
                            key=date_key_func,  # Apply datetime conversion
                            kind="mergesort",  # Use stable sort
                        )
                    elif is_potentially_numeric and not is_string_col:
                        logging.debug(
                            f"DEBUG Sort: Using NUMERIC sort strategy for '{col_name}'."
                        )
                        # Attempt numeric sort using the key
                        sorted_non_cash_rows = non_cash_rows.sort_values(
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",  # Keep NaNs at the bottom
                            key=numeric_key_func,  # Apply cleaning + numeric conversion before sorting
                            kind="mergesort",  # Use stable sort
                        )
                    else:  # Default to string sort (covers is_string_col and other cases)
                        logging.debug(
                            f"DEBUG Sort: Using STRING sort strategy for '{col_name}'."
                        )
                        # Sort as strings, fillna ensures consistent sorting
                        sorted_non_cash_rows = non_cash_rows.sort_values(
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",  # Keep NaNs at the bottom
                            key=lambda x: x.astype(str).fillna(
                                ""
                            ),  # Ensure string comparison
                            kind="mergesort",  # Use stable sort
                        )

                except Exception as e_sort:
                    logging.error(
                        f"ERROR during primary sorting logic for '{col_name}': {e_sort}"
                    )
                    traceback.print_exc()
                    # Fallback: Attempt simple sort without key if specific sort failed
                    try:
                        logging.warning(
                            f"Attempting fallback string sort for '{col_name}' after error."
                        )
                        sorted_non_cash_rows = non_cash_rows.sort_values(
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",
                            key=lambda x: x.astype(str).fillna(
                                ""
                            ),  # Use string key for fallback
                            kind="mergesort",
                        )
                    except Exception as e_fallback_sort:
                        logging.error(
                            f"Fallback sort also failed for '{col_name}': {e_fallback_sort}"
                        )
                        sorted_non_cash_rows = (
                            non_cash_rows  # Fallback to original order on error
                        )
            else:
                logging.debug("DEBUG Sort: No non-cash rows to sort.")
                pass  # sorted_non_cash_rows remains empty

            # --- Combine Sorted Rows ---
            # Concatenate sorted non-cash rows with cash rows (always at the bottom)
            self._data = pd.concat([sorted_non_cash_rows, cash_rows], ignore_index=True)

            self.layoutChanged.emit()  # Notify view about layout change end
            logging.info(f"Sorting finished for '{col_name}'.")

        except Exception as e:
            # Catch-all for unexpected errors in the sort method
            col_name_str = "OOB"
            try:
                # Check if 'column' is a valid index before accessing
                if 0 <= column < len(self._data.columns):
                    col_name_str = self._data.columns[column]
                else:
                    col_name_str = f"Invalid Index {column}"
            except IndexError:
                col_name_str = f"IndexError {column}"
            except (
                AttributeError
            ):  # Handle case where self._data might not be a DataFrame
                col_name_str = "Model Data Invalid"

            logging.error(
                f"CRITICAL Error in sort method (Col:{column}, Name:'{col_name_str}'): {e}"
            )
            traceback.print_exc()
            # Try to emit layoutChanged even on critical error to prevent UI freeze
            try:
                self.layoutChanged.emit()
            except Exception as e_emit:
                logging.error(
                    f"ERROR emitting layoutChanged after sort error: {e_emit}"
                )


class LogViewerDialog(QDialog):
    """Dialog to display ignored transactions and reasons."""

    def __init__(self, ignored_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ignored Transactions Log")
        self.setMinimumSize(800, 400)  # Make it reasonably sized

        layout = QVBoxLayout(self)

        if ignored_df is None or ignored_df.empty:
            label = QLabel("No transactions were ignored during the last calculation.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        else:
            self.table_view = QTableView()
            self.table_view.setObjectName("IgnoredLogTable")
            # Use a fresh PandasModel instance
            self.table_model = PandasModel(
                ignored_df.copy(), parent=parent
            )  # Pass parent for currency maybe? Or just display raw
            self.table_view.setModel(self.table_model)

            # Configure table view appearance (optional but good)
            self.table_view.setAlternatingRowColors(True)
            self.table_view.setSelectionBehavior(QTableView.SelectRows)
            self.table_view.setWordWrap(False)
            self.table_view.setSortingEnabled(True)  # Allow sorting
            self.table_view.horizontalHeader().setSectionResizeMode(
                QHeaderView.Interactive
            )
            self.table_view.horizontalHeader().setStretchLastSection(False)
            self.table_view.verticalHeader().setVisible(False)
            self.table_view.resizeColumnsToContents()  # Initial resize

            # Make the "Reason Ignored" column wider if it exists
            try:
                reason_col_idx = ignored_df.columns.get_loc("Reason Ignored")
                self.table_view.setColumnWidth(reason_col_idx, 250)
            except KeyError:
                pass  # Column might not exist if no reasons were added

            layout.addWidget(self.table_view)

        # Add a close button
        button_box = QDialogButtonBox(QDialogButtonBox.Close)
        button_box.rejected.connect(self.reject)  # Close maps to reject
        layout.addWidget(button_box)

        # Apply parent's stylesheet if possible
        if parent and hasattr(parent, "styleSheet"):
            self.setStyleSheet(parent.styleSheet())
        # Apply parent's font if possible
        if parent and hasattr(parent, "font"):
            self.setFont(parent.font())


# Add this class near other dialogs in main_gui.py


class ManageTransactionsDialog(QDialog):
    """Dialog to view, edit, and delete transactions from the source CSV."""

    # Signal to notify main window to refresh after changes
    data_changed = Signal()

    def __init__(self, original_df: pd.DataFrame, csv_filepath: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Manage Transactions")
        self.setMinimumSize(1000, 600)
        self._parent_app = parent  # Store reference to main app
        self._original_data = original_df.copy()  # Keep a local copy
        self._csv_filepath = csv_filepath  # Needed for saving changes

        layout = QVBoxLayout(self)

        # --- Add filtering (optional but helpful) ---
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("Filter Symbol:"))
        self.filter_symbol_edit = QLineEdit()
        filter_layout.addWidget(self.filter_symbol_edit)
        filter_layout.addWidget(QLabel("Account:"))
        self.filter_account_edit = QLineEdit()
        filter_layout.addWidget(self.filter_account_edit)
        filter_button = QPushButton("Apply Filter")
        filter_button.clicked.connect(self._apply_filter)
        filter_layout.addWidget(filter_button)
        clear_button = QPushButton("Clear Filter")
        clear_button.clicked.connect(self._clear_filter)
        filter_layout.addWidget(clear_button)
        filter_layout.addStretch(1)
        layout.addLayout(filter_layout)
        # --- End filtering ---

        self.table_view = QTableView()
        self.table_view.setObjectName("ManageTransactionsTable")
        # Use the original data with original column names
        self.table_model = PandasModel(self._original_data, parent=parent)
        self.table_view.setModel(self.table_model)

        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setSelectionMode(
            QTableView.SingleSelection
        )  # Enforce single selection for Edit
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSortingEnabled(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.resizeColumnsToContents()
        layout.addWidget(self.table_view)

        button_layout = QHBoxLayout()
        self.edit_button = QPushButton("Edit Selected")
        self.delete_button = QPushButton("Delete Selected")
        self.close_button = QPushButton("Close")
        button_layout.addWidget(self.edit_button)
        button_layout.addWidget(self.delete_button)
        button_layout.addStretch(1)
        button_layout.addWidget(self.close_button)
        layout.addLayout(button_layout)

        # Connections
        self.edit_button.clicked.connect(self.edit_selected_transaction)
        self.delete_button.clicked.connect(self.delete_selected_transaction)
        self.close_button.clicked.connect(self.reject)  # Close button rejects

        # Apply parent's style and font
        if parent:
            if hasattr(parent, "styleSheet"):
                self.setStyleSheet(parent.styleSheet())
            if hasattr(parent, "font"):
                self.setFont(parent.font())

    def _apply_filter(self):
        """Filters the table view based on symbol/account input."""
        symbol_filter = self.filter_symbol_edit.text().strip().upper()
        account_filter = self.filter_account_edit.text().strip()
        df_filtered = self._original_data.copy()

        if symbol_filter:
            try:
                df_filtered = df_filtered[
                    df_filtered["Stock / ETF Symbol"].str.contains(
                        symbol_filter, case=False, na=False
                    )
                ]
            except KeyError:
                pass  # Ignore if column doesn't exist
        if account_filter:
            try:
                df_filtered = df_filtered[
                    df_filtered["Investment Account"].str.contains(
                        account_filter, case=False, na=False
                    )
                ]
            except KeyError:
                pass

        self.table_model.updateData(df_filtered)
        self.table_view.resizeColumnsToContents()

    def _clear_filter(self):
        """Clears filters and shows all original data."""
        self.filter_symbol_edit.clear()
        self.filter_account_edit.clear()
        self.table_model.updateData(self._original_data)
        self.table_view.resizeColumnsToContents()

    def get_selected_original_index(self) -> Optional[int]:
        """
        Gets the 'original_index' of the currently selected row in the table view.
        Handles potential filtering/sorting by querying the model directly.
        """
        selected_indexes = self.table_view.selectionModel().selectedRows()
        if not selected_indexes or len(selected_indexes) != 1:
            logging.debug("get_selected_original_index: No single row selected.")
            # Don't show a message box here, let the calling function handle it.
            return None  # Indicate no valid selection

        selected_view_index = selected_indexes[
            0
        ]  # This is the QModelIndex for the selected view row
        source_model = self.table_model  # Get the model associated with the table

        # --- Find the column index for 'original_index' in the model's data ---
        try:
            # Use the actual DataFrame currently backing the model to find the column
            original_index_col_name = "original_index"  # The column name we stored
            if original_index_col_name not in source_model._data.columns:
                logging.error(
                    f"'{original_index_col_name}' column not found in the ManageTransactionsDialog model's data."
                )
                QMessageBox.warning(
                    self,
                    "Internal Error",
                    f"Required column '{original_index_col_name}' is missing.",
                )
                return None

            original_index_col_idx = source_model._data.columns.get_loc(
                original_index_col_name
            )

            # --- Create a QModelIndex specifically for the 'original_index' column of the selected row ---
            target_model_index = source_model.index(
                selected_view_index.row(), original_index_col_idx
            )

            # --- Ask the model for the data at that specific index ---
            # Using DisplayRole should give us the value as displayed (likely a string number)
            original_index_val = source_model.data(target_model_index, Qt.DisplayRole)

            if original_index_val is None:
                logging.warning(
                    f"Model returned None for original_index at view row {selected_view_index.row()}, col idx {original_index_col_idx}"
                )
                QMessageBox.warning(
                    self,
                    "Selection Error",
                    "Could not retrieve data for the selected row's original index.",
                )
                return None

            # --- Convert the retrieved value to an integer ---
            try:
                return int(original_index_val)
            except (ValueError, TypeError) as e:
                logging.error(
                    f"Could not convert retrieved original_index '{original_index_val}' to int: {e}"
                )
                QMessageBox.warning(
                    self,
                    "Data Error",
                    f"Invalid original index value found for selected row: {original_index_val}",
                )
                return None

        except (AttributeError, KeyError, IndexError) as e:
            logging.error(
                f"Error accessing model data/columns in get_selected_original_index: {e}"
            )
            QMessageBox.warning(
                self, "Internal Error", "Error accessing table model data."
            )
            return None
        except Exception as e:  # Catch any other unexpected errors
            logging.exception("Unexpected error in get_selected_original_index")
            QMessageBox.critical(
                self,
                "Unexpected Error",
                "An unexpected error occurred while getting the selected row index.",
            )
            return None

    @Slot()
    def edit_selected_transaction(self):
        original_index = self.get_selected_original_index()
        if original_index is None:
            QMessageBox.warning(
                self, "Selection Error", "Please select a single transaction to edit."
            )
            return

        # Find the row data from the original full dataset
        try:
            transaction_row = self._original_data[
                self._original_data["original_index"] == original_index
            ].iloc[0]
            transaction_dict_for_dialog = transaction_row.to_dict()
        except (IndexError, KeyError):
            QMessageBox.warning(
                self, "Data Error", "Could not find the selected transaction data."
            )
            return

        # Reuse AddTransactionDialog
        # Need the list of existing accounts for the dropdown
        accounts = (
            list(self._original_data["Investment Account"].unique())
            if "Investment Account" in self._original_data
            else []
        )
        edit_dialog = AddTransactionDialog(existing_accounts=accounts, parent=self)
        edit_dialog.setWindowTitle("Edit Transaction")

        # --- Pre-fill the dialog ---
        try:
            # Date
            date_str = transaction_dict_for_dialog.get("Date (MMM DD, YYYY)")
            if date_str:
                # Try parsing with the specific CSV format first
                qdate = QDate()
                parsed_dt = datetime.strptime(date_str, CSV_DATE_FORMAT)
                qdate.setDate(parsed_dt.year, parsed_dt.month, parsed_dt.day)
                if qdate.isValid():
                    edit_dialog.date_edit.setDate(qdate)
                else:  # Fallback parsing
                    qdate = QDate.fromString(
                        date_str, "yyyy-MM-dd"
                    )  # Try another common format
                    if qdate.isValid():
                        edit_dialog.date_edit.setDate(qdate)

            # Type (match case-insensitively)
            tx_type_str = transaction_dict_for_dialog.get("Transaction Type", "")
            for i in range(edit_dialog.type_combo.count()):
                if edit_dialog.type_combo.itemText(i).lower() == tx_type_str.lower():
                    edit_dialog.type_combo.setCurrentIndex(i)
                    break

            edit_dialog.symbol_edit.setText(
                str(transaction_dict_for_dialog.get("Stock / ETF Symbol", ""))
            )

            # Account (exact match)
            acc_str = transaction_dict_for_dialog.get("Investment Account", "")
            index = edit_dialog.account_combo.findText(acc_str, Qt.MatchFixedString)
            if index >= 0:
                edit_dialog.account_combo.setCurrentIndex(index)

            # Numeric fields (handle potential formatting issues)
            def format_for_edit(value, precision=8):
                if pd.isna(value):
                    return ""
                try:
                    # Format with desired precision, remove trailing zeros/decimal if integer
                    formatted = f"{float(value):.{precision}f}".rstrip("0").rstrip(".")
                    return (
                        formatted if formatted else "0"
                    )  # Return "0" if it became empty
                except (ValueError, TypeError):
                    return str(value)  # Fallback

            edit_dialog.quantity_edit.setText(
                format_for_edit(transaction_dict_for_dialog.get("Quantity of Units"))
            )
            edit_dialog.price_edit.setText(
                format_for_edit(transaction_dict_for_dialog.get("Amount per unit"))
            )
            edit_dialog.total_amount_edit.setText(
                format_for_edit(
                    transaction_dict_for_dialog.get("Total Amount"), precision=2
                )
            )
            edit_dialog.commission_edit.setText(
                format_for_edit(transaction_dict_for_dialog.get("Fees"), precision=2)
            )
            edit_dialog.split_ratio_edit.setText(
                format_for_edit(
                    transaction_dict_for_dialog.get(
                        "Split Ratio (new shares per old share)"
                    )
                )
            )
            edit_dialog.note_edit.setText(
                str(transaction_dict_for_dialog.get("Note", ""))
            )

            edit_dialog._update_field_states(
                edit_dialog.type_combo.currentText()
            )  # Ensure fields are correctly enabled/disabled

        except Exception as e_fill:
            QMessageBox.critical(
                self, "Dialog Error", f"Error pre-filling edit dialog:\n{e_fill}"
            )
            return
        # --- End Pre-fill ---

        if edit_dialog.exec():
            new_data_dict = (
                edit_dialog.get_transaction_data()
            )  # Validation happens here
            if new_data_dict:
                # Call parent's method to handle CSV update and refresh
                if self._parent_app and hasattr(
                    self._parent_app, "_edit_transaction_in_csv"
                ):
                    if self._parent_app._edit_transaction_in_csv(
                        original_index, new_data_dict
                    ):
                        # Refresh the data in *this* dialog's table if successful
                        # Need to get updated original data from parent
                        if hasattr(self._parent_app, "original_data"):
                            self._original_data = self._parent_app.original_data.copy()
                            self._apply_filter()  # Re-apply filter to show updated data
                        self.data_changed.emit()  # Signal main window to refresh

    @Slot()
    def delete_selected_transaction(self):
        original_index = self.get_selected_original_index()
        if original_index is None:
            QMessageBox.warning(
                self, "Selection Error", "Please select a single transaction to delete."
            )
            return

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            "Are you sure you want to permanently delete this transaction?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            # Call parent's method to handle CSV update and refresh
            if self._parent_app and hasattr(
                self._parent_app, "_delete_transactions_from_csv"
            ):
                if self._parent_app._delete_transactions_from_csv([original_index]):
                    # Refresh the data in *this* dialog's table if successful
                    if hasattr(self._parent_app, "original_data"):
                        self._original_data = self._parent_app.original_data.copy()
                        self._apply_filter()  # Re-apply filter
                    self.data_changed.emit()  # Signal main window to refresh


# --- Add/Edit Transaction Dialog ---
class AddTransactionDialog(QDialog):
    """Dialog window for manually adding a new transaction entry."""

    def __init__(self, existing_accounts: List[str], parent=None):
        """
        Initializes the dialog with fields for transaction details.

        Args:
            existing_accounts (List[str]): A list of known account names to populate
                                           the account dropdown.
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setWindowTitle("Add New Transaction")
        self.setMinimumWidth(300)

        # --- Set the font for the ENTIRE dialog and its children ---
        # All widgets, including labels created by QFormLayout, will inherit this.
        dialog_font = QFont("Arial", 10)  # Set desired font and size HERE
        self.setFont(dialog_font)
        # --- Removed separate label_font and self.label_font ---

        self.transaction_types = [
            "Buy",
            "Sell",
            "Dividend",
            "Split",
            "Deposit",
            "Withdrawal",
            "Fees",
        ]

        # --- Layout Modifications ---
        main_dialog_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(0, 0, 0, 0)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        # --- End Layout Modifications ---

        input_min_width = 150

        # Widgets (Creation remains the same)
        self.date_edit = QDateEdit(date.today())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setMinimumWidth(input_min_width)

        self.type_combo = QComboBox()
        self.type_combo.addItems(self.transaction_types)
        self.type_combo.setMinimumWidth(input_min_width)

        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText(" e.g., AAPL, GOOG, $CASH")
        self.symbol_edit.setMinimumWidth(input_min_width)

        self.account_combo = QComboBox()
        self.account_combo.addItems(sorted(list(set(existing_accounts))))
        self.account_combo.setMinimumWidth(input_min_width)

        self.quantity_edit = QLineEdit()
        self.quantity_edit.setPlaceholderText(" e.g., 100.5 (required for most types)")
        self.quantity_edit.setMinimumWidth(input_min_width)

        self.price_edit = QLineEdit()
        self.price_edit.setPlaceholderText(" Per unit (required for buy/sell)")
        self.price_edit.setMinimumWidth(input_min_width)

        self.total_amount_edit = QLineEdit()
        self.total_amount_edit.setPlaceholderText(" Optional (used for some dividends)")
        self.total_amount_edit.setMinimumWidth(input_min_width)

        self.commission_edit = QLineEdit()
        self.commission_edit.setPlaceholderText(" e.g., 6.95 (optional)")
        self.commission_edit.setMinimumWidth(input_min_width)

        self.split_ratio_edit = QLineEdit()
        self.split_ratio_edit.setPlaceholderText(
            " New shares per old (e.g., 2 for 2:1)"
        )
        # The label is created implicitly by addRow below, inheriting the dialog font
        self.split_ratio_label = QLabel(
            "Split Ratio:"
        )  # Keep reference if needed by _update_field_states
        self.split_ratio_edit.setMinimumWidth(input_min_width)

        self.note_edit = QLineEdit()
        self.note_edit.setPlaceholderText(" Optional note")
        self.note_edit.setMinimumWidth(input_min_width)

        # --- Add widgets to the QFormLayout using simple string labels ---
        # The labels will be created by QFormLayout and inherit the dialog's font.
        form_layout.addRow("Date:", self.date_edit)
        form_layout.addRow("Type:", self.type_combo)
        form_layout.addRow("Symbol:", self.symbol_edit)
        form_layout.addRow("Account:", self.account_combo)
        form_layout.addRow("Quantity:", self.quantity_edit)
        form_layout.addRow("Price/Unit:", self.price_edit)
        form_layout.addRow("Total Amount:", self.total_amount_edit)
        form_layout.addRow("Commission:", self.commission_edit)
        # Pass the QLabel object directly if you need to enable/disable it later
        form_layout.addRow(self.split_ratio_label, self.split_ratio_edit)
        form_layout.addRow("Note:", self.note_edit)
        # --- Removed the loop and the create_label helper ---

        # Buttons (Creation remains the same)
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        # --- Layout Modifications ---
        main_dialog_layout.addLayout(form_layout)
        main_dialog_layout.addWidget(self.button_box)
        # --- End Layout Modifications ---

        # Connect type change (remains the same)
        self.type_combo.currentTextChanged.connect(self._update_field_states)
        self._update_field_states(self.type_combo.currentText())

    def _update_field_states(self, tx_type):
        """Enables/disables input fields based on the selected transaction type.

        For example, 'Split Ratio' is only enabled for 'Split' type, 'Price/Unit'
        is disabled for cash deposits/withdrawals. Clears disabled fields.

        Args:
            tx_type (str): The currently selected transaction type (e.g., "Buy", "Split").
        """
        # ... (logic remains the same) ...
        tx_type = tx_type.lower()
        is_split = tx_type == "split"
        is_fee = tx_type == "fees"
        # Check symbol text directly, case-insensitive compare
        is_cash_flow = (tx_type in ["deposit", "withdrawal"]) and (
            self.symbol_edit.text().strip().upper() == CASH_SYMBOL_CSV
        )
        is_dividend = tx_type == "dividend"

        # Enable/disable based on type
        self.quantity_edit.setEnabled(not is_split and not is_fee)
        self.price_edit.setEnabled(not is_split and not is_fee and not is_cash_flow)
        self.total_amount_edit.setEnabled(
            is_dividend
        )  # Primarily for dividends if price/qty not given
        self.split_ratio_edit.setEnabled(is_split)
        self.split_ratio_label.setEnabled(is_split)  # Also enable/disable label

        # Clear disabled fields
        if not self.quantity_edit.isEnabled():
            self.quantity_edit.clear()
        if not self.price_edit.isEnabled():
            self.price_edit.clear()
        if not self.total_amount_edit.isEnabled():
            self.total_amount_edit.clear()
        if not self.split_ratio_edit.isEnabled():
            self.split_ratio_edit.clear()

        # Symbol handling for cash flow
        if is_cash_flow:
            # Don't force symbol if user might be changing type *from* cash flow
            # self.symbol_edit.setText(CASH_SYMBOL_CSV)
            self.price_edit.clear()
            self.price_edit.setEnabled(False)
        else:
            # Re-enable price edit if not cash flow
            self.price_edit.setEnabled(not is_split and not is_fee)

    def get_transaction_data(self) -> Optional[Dict[str, str]]:
        """Validates user input and returns transaction data formatted for CSV saving.

        Performs checks based on the transaction type (e.g., quantity and price
        required for buys/sells). Shows warning messages for invalid input.

        Returns:
            Optional[Dict[str, str]]: A dictionary where keys are the expected CSV
                header names and values are the validated and formatted user input
                strings. Returns None if validation fails.
        """
        # ... (Validation logic remains exactly the same as before) ...
        data = {}
        tx_type = self.type_combo.currentText().lower()
        symbol = self.symbol_edit.text().strip().upper()
        account = self.account_combo.currentText()
        date_val = self.date_edit.date().toPython()
        if not symbol:
            QMessageBox.warning(self, "Input Error", "Symbol cannot be empty.")
            return None
        if not account:
            QMessageBox.warning(self, "Input Error", "Account cannot be empty.")
            return None
        qty_str = self.quantity_edit.text().strip().replace(",", "")
        price_str = self.price_edit.text().strip().replace(",", "")
        total_str = self.total_amount_edit.text().strip().replace(",", "")
        comm_str = self.commission_edit.text().strip().replace(",", "")
        split_str = self.split_ratio_edit.text().strip().replace(",", "")
        note_str = self.note_edit.text().strip()
        qty, price, total, comm, split = None, None, None, 0.0, None
        if comm_str:
            try:
                comm = float(comm_str)
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Commission must be a valid number."
                )
                return None
        else:
            comm = 0.0
        if tx_type in ["buy", "sell", "short sell", "buy to cover"]:
            if not qty_str or not price_str:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Quantity and Price/Unit are required for '{tx_type}'.",
                )
                return None
            try:
                qty = float(qty_str)
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Quantity must be a valid number."
                )
                return None
            try:
                price = float(price_str)
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Price/Unit must be a valid number."
                )
                return None
            if qty <= 0:
                QMessageBox.warning(
                    self, "Input Error", "Quantity must be positive for buy/sell."
                )
                return None
            if price <= 0:
                QMessageBox.warning(
                    self, "Input Error", "Price/Unit must be positive for buy/sell."
                )
                return None
        elif tx_type in ["deposit", "withdrawal"]:
            if symbol != CASH_SYMBOL_CSV:
                if not qty_str or not price_str:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        f"Quantity and Price/Unit (cost basis) are required for stock '{tx_type}'.",
                    )
                    return None
                try:
                    qty = float(qty_str)
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", "Quantity must be a valid number."
                    )
                    return None
                try:
                    price = float(price_str)
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Price/Unit (cost basis) must be a valid number.",
                    )
                    return None
                if qty <= 0:
                    QMessageBox.warning(
                        self, "Input Error", "Quantity must be positive."
                    )
                    return None
                if price < 0:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Price/Unit (cost basis) cannot be negative.",
                    )
                    return None
            else:
                if not qty_str:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        f"Quantity (amount) is required for cash '{tx_type}'.",
                    )
                    return None
                try:
                    qty = float(qty_str)
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", "Quantity (amount) must be a valid number."
                    )
                    return None
                if qty <= 0:
                    QMessageBox.warning(
                        self, "Input Error", "Quantity (amount) must be positive."
                    )
                    return None
                price = 1.0
        elif tx_type == "dividend":
            qty_ok, price_ok, total_ok = False, False, False
            if qty_str and price_str:
                try:
                    qty = float(qty_str)
                    qty_ok = True
                except ValueError:
                    pass
                try:
                    price = float(price_str)
                    price_ok = True
                except ValueError:
                    pass
            if total_str:
                try:
                    total = float(total_str)
                    total_ok = True
                except ValueError:
                    pass
            if not ((qty_ok and price_ok) or total_ok):
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Dividend requires Quantity & Price/Unit OR Total Amount.",
                )
                return None
            if qty is not None and qty < 0:
                QMessageBox.warning(
                    self, "Input Error", "Dividend quantity cannot be negative."
                )
                return None
            if price is not None and price < 0:
                QMessageBox.warning(
                    self, "Input Error", "Dividend price/unit cannot be negative."
                )
                return None
            if total is not None and total < 0:
                QMessageBox.warning(
                    self, "Input Error", "Dividend total amount cannot be negative."
                )
                return None
        elif tx_type == "split":
            if not split_str:
                QMessageBox.warning(
                    self, "Input Error", "Split Ratio is required for 'split'."
                )
                return None
            try:
                split = float(split_str)
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Split Ratio must be a valid number."
                )
                return None
            if split <= 0:
                QMessageBox.warning(
                    self, "Input Error", "Split Ratio must be positive."
                )
                return None
            qty_str, price_str, total_str = "", "", ""
        elif tx_type == "fees":
            qty_str, price_str, total_str, split_str = "", "", "", ""

        # Format for CSV (using corrected version)
        data = {
            "Date (MMM DD, YYYY)": date_val.strftime(CSV_DATE_FORMAT),
            "Transaction Type": self.type_combo.currentText(),
            "Stock / ETF Symbol": symbol,
            "Quantity of Units": f"{qty:.8f}" if qty is not None else "",
            "Amount per unit": f"{price:.8f}" if price is not None else "",
            "Total Amount": f"{total:.2f}" if total is not None else "",
            "Fees": f"{comm:.2f}",
            "Investment Account": account,
            "Split Ratio (new shares per old share)": (
                f"{split:.8f}" if split is not None else ""
            ),
            "Note": note_str,
        }
        # Return only the validated data dictionary
        return data

    def accept(self):
        """
        Overrides the default accept behavior to validate input before closing.

        Calls `get_transaction_data` for validation. If validation passes, the
        dialog is accepted; otherwise, it remains open.
        """
        if self.get_transaction_data():  # Validation happens here
            super().accept()


# --- Main Application Window ---
class PortfolioApp(QMainWindow):
    """Main application window for the Investa Portfolio Dashboard."""

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

    def _get_currency_symbol(self, get_name=False):
        """
        Gets the currency symbol (e.g., "$") or 3-letter name (e.g., "USD").

        Retrieves the currently selected display currency from the UI or config.

        Args:
            get_name (bool, optional): If True, returns the 3-letter currency code
                                       (e.g., "USD"). Otherwise, returns the symbol
                                       (e.g., "$"). Defaults to False.

        Returns:
            str: The currency symbol or name.
        """
        display_currency = "USD"
        if (
            hasattr(self, "currency_combo")
            and self.currency_combo
            and self.currency_combo.count() > 0
        ):
            display_currency = self.currency_combo.currentText()
        elif hasattr(self, "config"):
            display_currency = self.config.get("display_currency", "USD")
        if get_name:
            return display_currency
        symbol_map = {"USD": "$", "THB": "฿", "EUR": "€", "GBP": "£", "JPY": "¥"}
        return symbol_map.get(display_currency, display_currency)

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

    def load_config(self):
        """
        Loads application configuration from a JSON file (gui_config.json).

        Loads settings like the transaction file path, display currency, selected
        accounts, graph parameters, and column visibility. Uses default values if
        the file doesn't exist or if specific settings are missing or invalid.
        Performs validation on loaded values.

        Returns:
            dict: The loaded (and potentially validated/defaulted) configuration dictionary.
        """
        default_display_currency = "USD"
        default_column_visibility = {
            col: True for col in get_column_definitions(default_display_currency).keys()
        }
        # --- Define Default Account Currency Map ---
        default_account_currency_map = {"SET": "THB"}  # Example default

        config_defaults = {
            "transactions_file": DEFAULT_CSV,
            "display_currency": default_display_currency,
            "show_closed": False,
            "selected_accounts": [],
            "load_on_startup": True,
            "fmp_api_key": DEFAULT_API_KEY,
            "account_currency_map": default_account_currency_map,  # <-- ADDED
            "default_currency": "USD",  # <-- ADDED Default base currency
            "graph_start_date": DEFAULT_GRAPH_START_DATE.strftime("%Y-%m-%d"),
            "graph_end_date": DEFAULT_GRAPH_END_DATE.strftime("%Y-%m-%d"),
            "graph_interval": DEFAULT_GRAPH_INTERVAL,
            "graph_benchmarks": DEFAULT_GRAPH_BENCHMARKS,
            "column_visibility": default_column_visibility,
        }
        config = config_defaults.copy()
        if os.path.exists(CONFIG_FILE):
            try:
                with open(CONFIG_FILE, "r") as f:
                    loaded_config = json.load(f)
                config.update(loaded_config)
                logging.info(f"Config loaded from {CONFIG_FILE}")

                # --- Validate Benchmarks (unchanged) ---
                if "graph_benchmarks" in config:
                    if isinstance(config["graph_benchmarks"], list):
                        valid_benchmarks = [
                            b
                            for b in config["graph_benchmarks"]
                            if isinstance(b, str) and b in BENCHMARK_OPTIONS
                        ]
                        config["graph_benchmarks"] = valid_benchmarks
                    else:
                        config["graph_benchmarks"] = DEFAULT_GRAPH_BENCHMARKS

                # --- Validate Selected Accounts (unchanged) ---
                if "selected_accounts" in config and not isinstance(
                    config["selected_accounts"], list
                ):
                    logging.warning(
                        "Warn: Invalid 'selected_accounts' type in config. Resetting to empty list."
                    )
                    config["selected_accounts"] = []

                # --- Validate Column Visibility (unchanged) ---
                if "column_visibility" in config and isinstance(
                    config["column_visibility"], dict
                ):
                    # Ensure keys are strings and values are booleans
                    validated_visibility = {}
                    all_cols = get_column_definitions(
                        config.get("display_currency", default_display_currency)
                    ).keys()
                    for col in all_cols:
                        # Get value from loaded config, default to True if missing or invalid type
                        val = config["column_visibility"].get(col)
                        validated_visibility[col] = (
                            val if isinstance(val, bool) else True
                        )
                    config["column_visibility"] = validated_visibility
                else:
                    config["column_visibility"] = default_column_visibility

                # --- Validate Account Currency Map ---
                if "account_currency_map" in config:
                    if not isinstance(config["account_currency_map"], dict) or not all(
                        isinstance(k, str) and isinstance(v, str)
                        for k, v in config["account_currency_map"].items()
                    ):
                        logging.warning(
                            "Warn: Invalid 'account_currency_map' type in config. Resetting to default."
                        )
                        config["account_currency_map"] = default_account_currency_map
                else:
                    config["account_currency_map"] = default_account_currency_map

                # --- Validate Default Currency ---
                if "default_currency" in config:
                    if (
                        not isinstance(config["default_currency"], str)
                        or len(config["default_currency"]) != 3
                    ):
                        logging.warning(
                            "Warn: Invalid 'default_currency' type/format in config. Resetting to 'USD'."
                        )
                        config["default_currency"] = "USD"
                else:
                    config["default_currency"] = "USD"

            except Exception as e:
                logging.warning(
                    f"Warn: Load config failed ({CONFIG_FILE}): {e}. Using defaults."
                )
        else:
            logging.info(f"Config file {CONFIG_FILE} not found. Using defaults.")

        # Ensure all default keys exist (modified to include new keys)
        for key, default_value in config_defaults.items():
            if key not in config:
                config[key] = default_value
            # Type check (allow list for selected_accounts, dict for map)
            elif not isinstance(config[key], type(default_value)):
                if key not in [
                    "fmp_api_key",
                    "selected_accounts",
                    "account_currency_map",
                    "graph_benchmarks",
                ] or (
                    config[key] is not None
                    and not isinstance(config[key], (str, list, dict))
                ):
                    logging.warning(
                        f"Warn: Config type mismatch for '{key}'. Loaded: {type(config[key])}, Default: {type(default_value)}. Using default."
                    )
                    config[key] = default_value

        # Final date format validation (unchanged)
        try:
            QDate.fromString(config["graph_start_date"], "yyyy-MM-dd")
        except:
            config["graph_start_date"] = DEFAULT_GRAPH_START_DATE.strftime("%Y-%m-%d")
        try:
            QDate.fromString(config["graph_end_date"], "yyyy-MM-dd")
        except:
            config["graph_end_date"] = DEFAULT_GRAPH_END_DATE.strftime("%Y-%m-%d")

        return config

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
                    change_pct_val = float(change_pct_decimal) * 100.0  # Scale to %
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

    # --- Initialization Method ---
    def __init__(self):
        """Initializes the main application window, loads config, and sets up UI."""
        super().__init__()
        self.app_font = QFont("Arial", 9)  # Or your chosen common font
        logging.info(
            f"Application base font set to: {self.app_font.family()} ({self.app_font.pointSize()}pt)"
        )  # Add this log
        self.setFont(self.app_font)
        self.base_window_title = "Investa Portfolio Dashboard"
        self.index_quote_data: Dict[str, Dict[str, Any]] = {}
        self.setWindowTitle(self.base_window_title)
        self.config = self.load_config()
        self.transactions_file = self.config.get("transactions_file", DEFAULT_CSV)
        self.fmp_api_key = self.config.get("fmp_api_key", DEFAULT_API_KEY)
        self.is_calculating = False

        # --- Account Selection State ---
        self.available_accounts: List[str] = []  # Populated after data load
        # Load selected accounts, default to empty list (meaning all)
        self.selected_accounts: List[str] = self.config.get("selected_accounts", [])
        # --- End Account Selection State ---

        self.selected_benchmarks = self.config.get(
            "graph_benchmarks", DEFAULT_GRAPH_BENCHMARKS
        )
        if not isinstance(self.selected_benchmarks, list):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS
        elif not self.selected_benchmarks and BENCHMARK_OPTIONS:
            self.selected_benchmarks = [BENCHMARK_OPTIONS[0]]

        self.all_possible_ui_columns = list(get_column_definitions().keys())
        self.column_visibility: Dict[str, bool] = self.config.get(
            "column_visibility", {}
        )
        self._ensure_all_columns_in_visibility()
        self.threadpool = QThreadPool()
        logging.info(f"Max threads: {self.threadpool.maxThreadCount()}")
        self.holdings_data = pd.DataFrame()
        self.ignored_data = pd.DataFrame()
        self.summary_metrics_data = {}
        self.account_metrics_data = {}
        self.historical_data = pd.DataFrame()
        self.last_calc_status = ""
        self.last_hist_twr_factor = np.nan
        self.app_font = QFont("Arial", 9)
        self.setFont(self.app_font)
        self.initUI()
        self.apply_styles()
        self.update_header_info(loading=True)
        self.update_performance_graphs(initial=True)

        # --- Initial Load Logic ---
        if self.config.get("load_on_startup", True):
            if self.transactions_file and os.path.exists(self.transactions_file):
                # Need to get available accounts before potentially filtering in refresh_data
                # Let's trigger a preliminary load just for accounts if needed, or handle in refresh_data
                logging.info("Triggering initial data refresh on startup...")
                from PySide6.QtCore import QTimer

                QTimer.singleShot(150, self.refresh_data)
            else:
                self.status_label.setText(
                    "Warn: Startup TX file invalid. Load skipped."
                )
                self._update_table_view_with_filtered_columns(pd.DataFrame())
                self.apply_column_visibility()
                self.update_performance_graphs(initial=True)
                self._update_account_button_text()  # Update button text even if no data
        else:
            self.status_label.setText("Ready. Select CSV file and click Refresh.")
            self._update_table_view_with_filtered_columns(pd.DataFrame())
            self.apply_column_visibility()
            self.update_performance_graphs(initial=True)
            self._update_account_button_text()  # Update button text

    # --- Benchmark Selection Methods (Define BEFORE initUI) ---
    def _update_benchmark_button_text(self):
        """Updates the benchmark selection button text to reflect current selections."""
        # Ensure selected_benchmarks attribute exists
        if not hasattr(self, "selected_benchmarks"):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS  # Initialize if missing

        if not self.selected_benchmarks:
            text = "Select Benchmarks"
        elif len(self.selected_benchmarks) == 1:
            text = f"Bench: {self.selected_benchmarks[0]}"
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
    def toggle_benchmark_selection(self, symbol: str, is_checked: bool):
        """
        Adds or removes a benchmark symbol from the selected list.

        Updates `self.selected_benchmarks` and the button text. Does NOT trigger
        an immediate data refresh; user must click 'Update Graphs'.

        Args:
            symbol (str): The benchmark symbol (e.g., "SPY") being toggled.
            is_checked (bool): True if the benchmark is being selected, False if deselected.
        """
        # ... (IMPORTANT: This should NOT call refresh_data anymore) ...
        if not hasattr(self, "selected_benchmarks"):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS
        if is_checked:
            if symbol not in self.selected_benchmarks:
                self.selected_benchmarks.append(symbol)
        else:
            if symbol in self.selected_benchmarks:
                self.selected_benchmarks.remove(symbol)
        try:
            self.selected_benchmarks.sort(
                key=lambda b: (
                    BENCHMARK_OPTIONS.index(b)
                    if b in BENCHMARK_OPTIONS
                    else float("inf")
                )
            )
        except ValueError:
            logging.warning("Warn: Could not sort benchmarks based on options.")
        self._update_benchmark_button_text()
        # Inform user to click Update Graphs
        self.status_label.setText(
            "Benchmark selection changed. Click 'Update Graphs' to apply."
        )

    @Slot()  # Add Slot decorator if used with signals/slots consistently
    def show_benchmark_selection_menu(self):
        """Displays a context menu with checkable actions for benchmark selection."""
        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())  # Apply style

        # Create a checkable action for each available benchmark option
        for benchmark_symbol in BENCHMARK_OPTIONS:
            action = QAction(benchmark_symbol, self)
            action.setCheckable(True)
            # Check the action if the symbol is currently in our selected list
            action.setChecked(benchmark_symbol in self.selected_benchmarks)
            # Connect the action's triggered signal to the toggle function
            # Use a lambda to pass the specific benchmark symbol being toggled
            action.triggered.connect(
                lambda checked, symbol=benchmark_symbol: self.toggle_benchmark_selection(
                    symbol, checked
                )
            )
            menu.addAction(action)

        # Display the menu just below the benchmark selection button
        # Ensure benchmark_select_button exists before mapping position
        if hasattr(self, "benchmark_select_button") and self.benchmark_select_button:
            button_pos = self.benchmark_select_button.mapToGlobal(
                QPoint(0, self.benchmark_select_button.height())
            )
            menu.exec(button_pos)
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
                logging.info(f"Account added: {account_name}.")
        else:
            if account_name in self.selected_accounts:
                self.selected_accounts.remove(account_name)
                logging.info(f"Account removed: {account_name}.")

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

        logging.info(f"Selected Accounts: {self.selected_accounts}")
        self._update_account_button_text()

        # Inform user to click Update
        self.status_label.setText(
            "Account selection changed. Click 'Update Accounts' to apply."
        )

    @Slot()
    def show_account_selection_menu(self):
        """Displays a context menu with checkable actions for account selection."""
        if not self.available_accounts:
            QMessageBox.information(
                self,
                "No Accounts",
                "Load transaction data first to see available accounts.",
            )
            return

        menu = QMenu(self)
        menu.setStyleSheet(self.styleSheet())

        # Action to select/deselect all
        action_all = QAction("Select/Deselect All", self)
        is_all_selected = len(self.selected_accounts) == len(self.available_accounts)
        action_all.triggered.connect(
            lambda: self._toggle_all_accounts(not is_all_selected)
        )
        menu.addAction(action_all)
        menu.addSeparator()

        # Create checkable actions for each available account
        for (
            account_name
        ) in self.available_accounts:  # Use the stored available accounts
            action = QAction(account_name, self)
            action.setCheckable(True)
            action.setChecked(account_name in self.selected_accounts)
            action.triggered.connect(
                lambda checked, name=account_name: self.toggle_account_selection(
                    name, checked
                )
            )
            menu.addAction(action)

        button = getattr(self, "account_select_button", None)
        if button:
            button_pos = button.mapToGlobal(QPoint(0, button.height()))
            menu.exec(button_pos)
        else:
            logging.warning("Warn: Account button not ready for menu.")

    def _toggle_all_accounts(self, select_all: bool):
        """
        Selects or deselects all available accounts.

        Args:
            select_all (bool): True to select all, False to deselect all.
        """
        if select_all:
            self.selected_accounts = self.available_accounts.copy()
            logging.info("All accounts selected.")
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
        self.status_label.setText(
            "Account selection changed. Click 'Update Accounts' to apply."
        )

    # --- End Account Selection Methods ---

    # --- Helper to create summary items (moved from initUI) ---
    def create_summary_item(self, label_text, is_large=False):
        """
        Creates a QLabel pair (label and value) for the summary grid.

        Args:
            label_text (str): The text for the descriptive label (e.g., "Net Value").
            is_large (bool, optional): If True, uses slightly larger fonts for emphasis.
                                       Defaults to False.

        Returns:
            Tuple[QLabel, QLabel]: A tuple containing the created label and value QLabels.
        """
        label = QLabel(label_text + ":")
        label.setObjectName("SummaryLabelLarge" if is_large else "SummaryLabel")
        label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        value = QLabel("N/A")  # Default text
        value.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        value.setObjectName("SummaryValueLarge" if is_large else "SummaryValue")

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
            self.app_font if hasattr(self, "app_font") else QFont()
        )  # Use base app font or default
        value_font.setPointSize(base_font_size + (2 if is_large else 1))
        value_font.setBold(True)
        value.setFont(value_font)

        return label, value

    # --- Column Visibility Methods (Define BEFORE initUI) ---

    @Slot(QPoint)  # Add Slot decorator
    def show_header_context_menu(self, pos: QPoint):
        """
        Shows a context menu on the table header to toggle column visibility.

        Args:
            pos (QPoint): The position where the right-click occurred within the header.
        """
        header = self.table_view.horizontalHeader()
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
        if (
            not hasattr(self, "table_view")
            or not self.table_view
            or not hasattr(self, "table_model")
            or not isinstance(self.table_model, PandasModel)
            or self.table_model.columnCount() == 0
        ):
            return  # Cannot apply if table/model not ready

        header = self.table_view.horizontalHeader()
        # Iterate through the columns currently in the model
        for col_index in range(self.table_model.columnCount()):
            # Get the header name displayed in the UI for this column index
            header_name = self.table_model.headerData(
                col_index, Qt.Horizontal, Qt.DisplayRole
            )
            if header_name:
                # Look up the visibility state for this header name
                is_visible = self.column_visibility.get(
                    str(header_name), True
                )  # Default to visible
                # Hide or show the section (column) in the header/table
                header.setSectionHidden(col_index, not is_visible)

    def save_config(self):
        """Saves the current application configuration to gui_config.json."""
        self.config["transactions_file"] = self.transactions_file
        if self.fmp_api_key:
            self.config["fmp_api_key"] = self.fmp_api_key
        else:
            self.config.pop("fmp_api_key", None)
        self.config["display_currency"] = self.currency_combo.currentText()
        self.config["show_closed"] = self.show_closed_check.isChecked()
        # Save list of selected accounts
        if hasattr(self, "selected_accounts") and isinstance(
            self.selected_accounts, list
        ):
            if hasattr(self, "available_accounts") and len(
                self.selected_accounts
            ) == len(self.available_accounts):
                self.config["selected_accounts"] = (
                    []
                )  # Save empty list if all are selected
            else:
                self.config["selected_accounts"] = self.selected_accounts
        else:
            logging.warning(
                "Warning: selected_accounts attribute missing or invalid during save. Saving default (all)."
            )
            self.config["selected_accounts"] = []

        # --- Save account_currency_map and default_currency ---
        # Assume self.config holds the current map (e.g., loaded initially, maybe editable later)
        # Ensure the keys exist before saving, using the loaded defaults if necessary.
        self.config["account_currency_map"] = self.config.get(
            "account_currency_map", {"SET": "THB"}
        )
        self.config["default_currency"] = self.config.get("default_currency", "USD")
        # --------------------------------------------------------

        self.config["load_on_startup"] = self.config.get(
            "load_on_startup", True
        )  # Keep existing
        self.config["graph_start_date"] = self.graph_start_date_edit.date().toString(
            "yyyy-MM-dd"
        )
        self.config["graph_end_date"] = self.graph_end_date_edit.date().toString(
            "yyyy-MM-dd"
        )
        self.config["graph_interval"] = self.graph_interval_combo.currentText()
        if hasattr(self, "selected_benchmarks") and isinstance(
            self.selected_benchmarks, list
        ):
            self.config["graph_benchmarks"] = self.selected_benchmarks
        else:
            self.config["graph_benchmarks"] = DEFAULT_GRAPH_BENCHMARKS
        self.config["column_visibility"] = self.column_visibility

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(self.config, f, indent=4)
            logging.info(f"Config saved to {CONFIG_FILE}")
        except Exception as e:
            logging.warning(f"Warn: Save config failed: {e}")
            QMessageBox.warning(
                self, "Config Save Error", f"Could not save settings: {e}"
            )

    # --- Initialization Method ---
    def __init__(self):
        super().__init__()

        # --- Core App Setup ---
        self.base_window_title = "Investa Portfolio Dashboard"
        self.setWindowTitle(self.base_window_title)
        self.app_font = QFont("Arial", 9)
        self.setFont(self.app_font)

        # --- Configuration Loading ---
        self.config = self.load_config()
        self.transactions_file = self.config.get("transactions_file", DEFAULT_CSV)
        self.fmp_api_key = self.config.get(
            "fmp_api_key", DEFAULT_API_KEY
        )  # Keep for potential future use

        # --- State Variables ---
        self.is_calculating = False
        self.last_calc_status = ""
        self.last_hist_twr_factor = np.nan

        # --- Data Placeholders ---
        self.holdings_data = pd.DataFrame()
        self.ignored_data = pd.DataFrame()
        self.summary_metrics_data = {}
        self.account_metrics_data = {}
        self.historical_data = pd.DataFrame()
        self.index_quote_data: Dict[str, Dict[str, Any]] = {}

        # --- Account Selection State ---
        self.available_accounts: List[str] = []
        # Load selected accounts, default to empty list (meaning all)
        self.selected_accounts: List[str] = self.config.get("selected_accounts", [])

        # --- Benchmark Selection State ---
        self.selected_benchmarks = self.config.get(
            "graph_benchmarks", DEFAULT_GRAPH_BENCHMARKS
        )
        # Validate and default benchmarks if needed
        if not isinstance(self.selected_benchmarks, list):
            self.selected_benchmarks = DEFAULT_GRAPH_BENCHMARKS
        elif (
            not self.selected_benchmarks and BENCHMARK_OPTIONS
        ):  # Ensure default if list is empty
            self.selected_benchmarks = [BENCHMARK_OPTIONS[0]]

        # --- Column Visibility State ---
        # Initialize self.column_visibility before calling _ensure_all_columns...
        self.column_visibility: Dict[str, bool] = self.config.get(
            "column_visibility", {}
        )
        # Now ensure all possible columns are covered
        self._ensure_all_columns_in_visibility()  # This needs self.column_visibility to exist

        # --- Background Processing ---
        self.threadpool = QThreadPool()
        logging.info(f"Max threads: {self.threadpool.maxThreadCount()}")

        # --- Initialize UI ---
        self._init_ui_structure()  # Setup frames and main layout
        self._init_ui_widgets()  # Create widgets within frames
        self._connect_signals()  # Connect buttons etc.

        # --- Apply Styles & Initial Updates ---
        self.apply_styles()
        self.update_header_info(loading=True)  # Show loading state initially
        self.update_performance_graphs(initial=True)  # Draw empty graphs initially
        self._update_account_button_text()  # Set initial account button text based on loaded config/defaults
        self._update_benchmark_button_text()  # Set initial benchmark button text
        self._update_table_title()  # Set initial table title

        # --- Initial Data Load Logic ---
        if self.config.get("load_on_startup", True):
            if self.transactions_file and os.path.exists(self.transactions_file):
                logging.info("Triggering initial data refresh on startup...")
                from PySide6.QtCore import QTimer

                QTimer.singleShot(
                    150, self.refresh_data
                )  # Delay slightly to ensure UI is fully shown
            else:
                # Provide specific feedback if the configured file is bad
                startup_file_msg = f"Warn: Startup TX file invalid or not found: '{self.transactions_file}'. Load skipped."
                self.status_label.setText(startup_file_msg)
                logging.info(startup_file_msg)
                # Update UI elements to reflect no data state
                self._update_table_view_with_filtered_columns(pd.DataFrame())
                self.apply_column_visibility()
                self.update_performance_graphs(initial=True)
                self._update_account_button_text()
                self._update_table_title()
        else:
            # User chose not to load on startup
            self.status_label.setText("Ready. Select CSV file and click Refresh.")
            self._update_table_view_with_filtered_columns(pd.DataFrame())
            self.apply_column_visibility()
            self.update_performance_graphs(initial=True)
            self._update_account_button_text()
            self._update_table_title()

    def _init_ui_structure(self):
        """Sets up the main window layout using QFrames for structure."""
        self.setWindowTitle(self.base_window_title)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create main frames/panels
        self.header_frame = QFrame()
        self.header_frame.setObjectName("HeaderFrame")
        self.controls_frame = QFrame()
        self.controls_frame.setObjectName("ControlsFrame")
        self.summary_and_graphs_frame = QFrame()
        self.summary_and_graphs_frame.setObjectName("SummaryAndGraphsFrame")
        self.content_frame = QFrame()
        self.content_frame.setObjectName("ContentFrame")

        # Add frames to main layout
        main_layout.addWidget(self.header_frame)
        main_layout.addWidget(self.controls_frame)
        main_layout.addWidget(self.summary_and_graphs_frame)
        main_layout.addWidget(
            self.content_frame, 1
        )  # Content frame takes remaining space

    def _init_ui_widgets(self):
        """Creates and places all UI widgets within their respective frames."""
        # --- Header ---
        header_layout = QHBoxLayout(self.header_frame)
        header_layout.setContentsMargins(15, 8, 15, 8)
        self.main_title_label = QLabel("📈 <b>Investa Portfolio Dashboard</b> ✨")
        self.main_title_label.setObjectName("MainTitleLabel")
        self.main_title_label.setTextFormat(Qt.RichText)
        self.header_info_label = QLabel("<i>Initializing...</i>")
        self.header_info_label.setObjectName("HeaderInfoLabel")
        self.header_info_label.setTextFormat(Qt.RichText)
        self.header_info_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        header_layout.addWidget(self.main_title_label)
        header_layout.addStretch(1)
        header_layout.addWidget(self.header_info_label)

        # --- Controls ---
        controls_layout = QHBoxLayout(self.controls_frame)
        controls_layout.setContentsMargins(10, 8, 10, 8)
        controls_layout.setSpacing(8)
        # Buttons
        self.select_file_button = QPushButton("Select CSV")
        self.select_file_button.setObjectName("SelectFileButton")
        self.select_file_button.setIcon(
            self.style().standardIcon(QStyle.SP_DirOpenIcon)
        )
        controls_layout.addWidget(self.select_file_button)
        self.add_transaction_button = QPushButton("Add Tx")
        self.add_transaction_button.setObjectName("AddTransactionButton")
        self.add_transaction_button.setIcon(
            self.style().standardIcon(QStyle.SP_FileIcon)
        )
        self.manage_transactions_button = QPushButton("Manage Tx")
        self.manage_transactions_button.setObjectName("ManageTransactionsButton")
        self.manage_transactions_button.setIcon(
            self.style().standardIcon(QStyle.SP_DialogSaveButton)
        )  # Example icon
        self.manage_transactions_button.setToolTip(
            "Edit or delete existing transactions"
        )
        controls_layout.addWidget(self.manage_transactions_button)

        self.view_ignored_button = QPushButton("View Log")
        self.view_ignored_button.setObjectName("ViewIgnoredButton")
        self.view_ignored_button.setIcon(
            self.style().standardIcon(QStyle.SP_MessageBoxWarning)
        )  # Example icon
        self.view_ignored_button.setToolTip(
            "View transactions ignored during the last calculation"
        )
        self.view_ignored_button.setEnabled(False)  # Initially disabled
        controls_layout.addWidget(self.view_ignored_button)

        self.add_transaction_button.setToolTip("Manually add a new transaction")
        controls_layout.addWidget(self.add_transaction_button)
        self.account_select_button = QPushButton("Accounts")
        self.account_select_button.setObjectName("AccountSelectButton")
        self.account_select_button.setMinimumWidth(130)
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
        # Filters & Combos
        controls_layout.addWidget(QLabel("Currency:"))
        self.currency_combo = QComboBox()
        self.currency_combo.setObjectName("CurrencyCombo")
        self.currency_combo.addItems(["USD", "THB", "JPY", "EUR", "GBP"])
        self.currency_combo.setCurrentText(self.config.get("display_currency", "USD"))
        self.currency_combo.setMinimumWidth(80)
        controls_layout.addWidget(self.currency_combo)
        self.show_closed_check = QCheckBox("Show Closed")
        self.show_closed_check.setObjectName("ShowClosedCheck")
        self.show_closed_check.setChecked(self.config.get("show_closed", False))
        controls_layout.addWidget(self.show_closed_check)
        # Graph Controls
        controls_layout.addWidget(QLabel("Graphs:"))
        self.graph_start_date_edit = QDateEdit()
        self.graph_start_date_edit.setObjectName("GraphDateEdit")
        self.graph_start_date_edit.setCalendarPopup(True)
        self.graph_start_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.graph_start_date_edit.setDate(
            QDate.fromString(self.config.get("graph_start_date"), "yyyy-MM-dd")
        )
        controls_layout.addWidget(self.graph_start_date_edit)
        controls_layout.addWidget(QLabel("to"))
        self.graph_end_date_edit = QDateEdit()
        self.graph_end_date_edit.setObjectName("GraphDateEdit")
        self.graph_end_date_edit.setCalendarPopup(True)
        self.graph_end_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.graph_end_date_edit.setDate(
            QDate.fromString(self.config.get("graph_end_date"), "yyyy-MM-dd")
        )
        controls_layout.addWidget(self.graph_end_date_edit)
        self.graph_interval_combo = QComboBox()
        self.graph_interval_combo.setObjectName("GraphIntervalCombo")
        self.graph_interval_combo.addItems(["D", "W", "M"])
        self.graph_interval_combo.setCurrentText(
            self.config.get("graph_interval", DEFAULT_GRAPH_INTERVAL)
        )
        self.graph_interval_combo.setMinimumWidth(60)
        controls_layout.addWidget(self.graph_interval_combo)
        self.benchmark_select_button = QPushButton()
        self.benchmark_select_button.setObjectName("BenchmarkSelectButton")
        self.benchmark_select_button.setMinimumWidth(100)
        controls_layout.addWidget(self.benchmark_select_button)  # Text set later
        self.graph_update_button = QPushButton("Update Graphs")
        self.graph_update_button.setObjectName("GraphUpdateButton")
        self.graph_update_button.setIcon(
            self.style().standardIcon(QStyle.SP_BrowserReload)
        )
        self.graph_update_button.setToolTip(
            "Recalculate and redraw performance graphs."
        )
        controls_layout.addWidget(self.graph_update_button)
        # Spacer & Right Aligned Controls
        controls_layout.addStretch(1)
        self.exchange_rate_display_label = QLabel("")
        self.exchange_rate_display_label.setObjectName("ExchangeRateLabel")
        self.exchange_rate_display_label.setVisible(False)
        controls_layout.addWidget(self.exchange_rate_display_label)
        self.refresh_button = QPushButton("Refresh All")
        self.refresh_button.setObjectName("RefreshButton")
        self.refresh_button.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        controls_layout.addWidget(self.refresh_button)

        # --- Summary & Graphs ---
        summary_graphs_layout = QHBoxLayout(self.summary_and_graphs_frame)
        summary_graphs_layout.setContentsMargins(10, 5, 10, 5)
        summary_graphs_layout.setSpacing(10)
        # Summary Grid
        summary_grid_widget = QWidget()
        summary_layout = QGridLayout(summary_grid_widget)
        summary_layout.setContentsMargins(10, 25, 10, 10)
        summary_layout.setHorizontalSpacing(15)
        summary_layout.setVerticalSpacing(30)
        self.summary_net_value = self.create_summary_item("Net Value", True)
        self.summary_day_change = self.create_summary_item("Day's G/L", True)
        self.summary_total_gain = self.create_summary_item("Total G/L")
        self.summary_realized_gain = self.create_summary_item("Realized G/L")
        self.summary_unrealized_gain = self.create_summary_item("Unrealized G/L")
        self.summary_dividends = self.create_summary_item("Dividends")
        self.summary_commissions = self.create_summary_item("Fees")
        self.summary_cash = self.create_summary_item("Cash Balance")
        self.summary_total_return_pct = self.create_summary_item("Total Ret %")
        self.summary_annualized_twr = self.create_summary_item("Ann. TWR %")
        summary_layout.addWidget(self.summary_net_value[0], 0, 0, Qt.AlignRight)
        summary_layout.addWidget(self.summary_net_value[1], 0, 1)
        summary_layout.addWidget(self.summary_day_change[0], 0, 2, Qt.AlignRight)
        summary_layout.addWidget(self.summary_day_change[1], 0, 3)
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
        summary_layout.setColumnStretch(1, 1)
        summary_layout.setColumnStretch(3, 1)
        summary_layout.setRowStretch(5, 1)
        summary_graphs_layout.addWidget(summary_grid_widget, 1)
        # Performance Graphs
        perf_graphs_container_widget = QWidget()
        perf_graphs_container_widget.setObjectName("PerfGraphsContainer")
        perf_graphs_layout = QHBoxLayout(perf_graphs_container_widget)
        perf_graphs_layout.setContentsMargins(0, 0, 0, 0)
        perf_graphs_layout.setSpacing(8)
        self.perf_return_fig = Figure(figsize=PERF_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.perf_return_ax = self.perf_return_fig.add_subplot(111)
        self.perf_return_canvas = FigureCanvas(self.perf_return_fig)
        self.perf_return_canvas.setObjectName("PerfReturnCanvas")
        self.perf_return_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        perf_graphs_layout.addWidget(self.perf_return_canvas)
        self.abs_value_fig = Figure(figsize=PERF_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.abs_value_ax = self.abs_value_fig.add_subplot(111)
        self.abs_value_canvas = FigureCanvas(self.abs_value_fig)
        self.abs_value_canvas.setObjectName("AbsValueCanvas")
        self.abs_value_canvas.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )
        perf_graphs_layout.addWidget(self.abs_value_canvas)
        summary_graphs_layout.addWidget(perf_graphs_container_widget, 2)

        # --- Content (Pies & Table) ---
        content_layout = QHBoxLayout(self.content_frame)
        content_layout.setContentsMargins(10, 5, 10, 10)
        content_layout.setSpacing(10)
        # Pie Charts
        pie_charts_container_widget = QWidget()
        pie_charts_container_widget.setObjectName("PieChartsContainer")
        pie_charts_layout = QVBoxLayout(pie_charts_container_widget)
        pie_charts_layout.setContentsMargins(0, 0, 0, 0)
        pie_charts_layout.setSpacing(10)
        account_chart_widget = QWidget()
        account_chart_layout = QVBoxLayout(account_chart_widget)
        account_chart_layout.setContentsMargins(0, 0, 0, 0)
        self.account_pie_title_label = QLabel("<b>Value by Account</b>")
        self.account_pie_title_label.setObjectName("AccountPieTitleLabel")
        self.account_pie_title_label.setTextFormat(Qt.RichText)
        account_chart_layout.addWidget(
            self.account_pie_title_label, alignment=Qt.AlignCenter
        )
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
        )
        self.holdings_fig = Figure(figsize=PIE_CHART_FIG_SIZE, dpi=CHART_DPI)
        self.holdings_ax = self.holdings_fig.add_subplot(111)
        self.holdings_canvas = FigureCanvas(self.holdings_fig)
        self.holdings_canvas.setObjectName("HoldingsPieCanvas")
        self.holdings_canvas.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        holdings_chart_layout.addWidget(self.holdings_canvas)
        pie_charts_layout.addWidget(holdings_chart_widget)
        content_layout.addWidget(pie_charts_container_widget, 1)
        # Table Panel
        table_panel = QFrame()
        table_panel.setObjectName("TablePanel")
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.setSpacing(5)
        table_header_layout = QHBoxLayout()
        table_header_layout.setContentsMargins(5, 5, 5, 3)
        self.table_title_label_left = QLabel("")
        self.table_title_label_left.setObjectName("TableScopeLabel")
        self.table_title_label_left.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self.table_title_label_right = QLabel("Holdings Detail")
        self.table_title_label_right.setObjectName("TableTitleLabel")
        self.table_title_label_right.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        table_header_layout.addWidget(self.table_title_label_left)
        table_header_layout.addStretch(1)
        table_header_layout.addWidget(self.table_title_label_right)
        table_layout.addLayout(table_header_layout)
        self.table_view = QTableView()
        self.table_view.setObjectName("HoldingsTable")
        self.table_view.setAlternatingRowColors(True)
        self.table_view.setSelectionBehavior(QTableView.SelectRows)
        self.table_view.setWordWrap(False)
        self.table_view.setSortingEnabled(True)
        self.table_model = PandasModel(parent=self)
        self.table_view.setModel(self.table_model)
        table_font = QFont(self.app_font)
        table_font.setPointSize(self.app_font.pointSize() + 1)
        self.table_view.setFont(table_font)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.horizontalHeader().setStretchLastSection(False)
        self.table_view.verticalHeader().setVisible(False)
        self.table_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table_view.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        table_layout.addWidget(self.table_view, 1)
        content_layout.addWidget(table_panel, 3)

        self.view_ignored_button.clicked.connect(self.show_ignored_log)
        self.manage_transactions_button.clicked.connect(
            self.show_manage_transactions_dialog
        )

        # --- Status Bar ---
        self._create_status_bar()

    def _create_status_bar(self):
        """Creates and configures the application's status bar."""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("StatusLabel")
        self.status_bar.addWidget(self.status_label, 1)
        self.yahoo_attribution_label = QLabel(
            "Financial data provided by Yahoo Finance"
        )
        self.yahoo_attribution_label.setObjectName("YahooAttributionLabel")
        self.status_bar.addPermanentWidget(self.yahoo_attribution_label)
        # Use the exchange rate label defined in _create_controls_widget
        self.status_bar.addPermanentWidget(self.exchange_rate_display_label)

    def _connect_signals(self):
        """Connects signals from UI widgets (buttons, combos, etc.) to their slots."""
        self.select_file_button.clicked.connect(self.select_file)
        self.add_transaction_button.clicked.connect(self.open_add_transaction_dialog)
        self.account_select_button.clicked.connect(self.show_account_selection_menu)
        self.update_accounts_button.clicked.connect(self.refresh_data)
        self.currency_combo.currentTextChanged.connect(self.filter_changed_refresh)
        self.show_closed_check.stateChanged.connect(self.filter_changed_refresh)
        self.graph_start_date_edit.dateChanged.connect(
            lambda: self.status_label.setText(
                "Graph dates changed. Click 'Update Graphs'."
            )
        )
        self.graph_end_date_edit.dateChanged.connect(
            lambda: self.status_label.setText(
                "Graph dates changed. Click 'Update Graphs'."
            )
        )
        self.graph_interval_combo.currentTextChanged.connect(
            lambda: self.status_label.setText(
                "Graph interval changed. Click 'Update Graphs'."
            )
        )
        self.benchmark_select_button.clicked.connect(self.show_benchmark_selection_menu)
        self.graph_update_button.clicked.connect(self.refresh_data)
        self.refresh_button.clicked.connect(self.refresh_data)
        self.table_view.horizontalHeader().customContextMenuRequested.connect(
            self.show_header_context_menu
        )

    # --- Styling Method (Reads External File) ---
    def apply_styles(self):
        """Loads and applies the stylesheet from style.qss."""
        logging.info("Applying styles from style.qss...")
        qss_file = "style.qss"  # Expect file in the same directory
        try:
            with open(qss_file, "r") as f:
                style_sheet = f.read()
                self.setStyleSheet(style_sheet)
            logging.info("Styles applied successfully.")

            # Re-apply chart background colors after loading stylesheet
            try:
                # Ensure figure/axes objects exist before setting colors
                if (
                    hasattr(self, "account_fig")
                    and hasattr(self, "holdings_fig")
                    and hasattr(self, "perf_return_fig")
                    and hasattr(self, "abs_value_fig")
                ):
                    pie_chart_bg_color = "#FFFFFF"  # Match COLOR_BG_CONTENT if possible
                    perf_chart_bg_color = (
                        "#F8F9FA"  # Match COLOR_BG_SUMMARY if possible
                    )
                    for fig in [self.account_fig, self.holdings_fig]:
                        fig.patch.set_facecolor(pie_chart_bg_color)
                    for ax in [self.account_ax, self.holdings_ax]:
                        ax.patch.set_facecolor(pie_chart_bg_color)
                    for fig in [self.perf_return_fig, self.abs_value_fig]:
                        fig.patch.set_facecolor(perf_chart_bg_color)
                    for ax in [self.perf_return_ax, self.abs_value_ax]:
                        ax.patch.set_facecolor(perf_chart_bg_color)
                else:
                    logging.warning(
                        "Warn: Chart figures/axes not fully initialized before style application."
                    )
            except Exception as e:
                logging.warning(f"Warning: Failed chart background style: {e}")

        except FileNotFoundError:
            logging.warning(
                f"Warning: Stylesheet file '{qss_file}' not found. Using default Qt styles."
            )
            # Clear any previously set stylesheet to revert to default
            self.setStyleSheet("")
        except Exception as e:
            logging.error(f"ERROR applying stylesheet: {e}")
            # Clear any potentially broken stylesheet
            self.setStyleSheet("")

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
        account_values = account_values[
            account_values.abs() > 1e-3
        ]  # Filter out near-zero values

        if account_values.empty:
            self.account_canvas.draw()
            return
        else:
            # --- Turn axis back on only if we have data to plot ---
            self.account_ax.axis("on")  # Turn axis back on for plotting
            self.account_ax.axis("equal")  # Ensure pie is circular

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
            label_offset_multiplier = 1.15
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
                    color=COLOR_TEXT_DARK,
                )

        self.account_fig.tight_layout(pad=0.1)
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
        holdings_values = holdings_values[
            holdings_values.abs() > 1e-3
        ]  # Filter out near-zero values

        if holdings_values.empty:
            # self.holdings_ax.text(0.5, 0.5, 'No Holdings > 0', ha='center', va='center', transform=self.holdings_ax.transAxes, fontsize=9, color=COLOR_TEXT_SECONDARY)
            self.holdings_canvas.draw()
            return
        else:
            # --- Turn axis back on only if we have data to plot ---
            self.holdings_ax.axis("on")  # Turn axis back on for plotting
            self.holdings_ax.axis("equal")  # Ensure pie is circular

            holdings_values = holdings_values.sort_values(ascending=False)
            labels_internal = holdings_values.index.tolist()
            values = holdings_values.values
            labels_display = [
                (
                    f"Cash ({display_currency})"
                    if symbol == CASH_SYMBOL_CSV
                    else str(symbol)
                )
                for symbol in labels_internal
            ]

            if len(values) > CHART_MAX_SLICES:
                top_v = values[: CHART_MAX_SLICES - 1]
                top_l = labels_display[: CHART_MAX_SLICES - 1]
                other_v = values[CHART_MAX_SLICES - 1 :].sum()
                values = np.append(top_v, other_v)
                labels = top_l + ["Other"]
            else:
                labels = labels_display

            cmap = plt.get_cmap("Spectral")
            colors = cmap(np.linspace(0.1, 0.9, len(values)))
            pie_radius = 0.9
            label_offset_multiplier = 1.15
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
                    color=COLOR_TEXT_DARK,
                )

        self.holdings_fig.tight_layout(pad=0.1)
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

        logging.info(
            f"Updating performance graphs for scope: {scope_label}... Initial: {initial}, Benchmarks: {self.selected_benchmarks}"
        )
        self.perf_return_ax.clear()
        self.abs_value_ax.clear()

        # --- Base Styling & Aspect ---
        for ax in [self.perf_return_ax, self.abs_value_ax]:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["bottom"].set_color(COLOR_BORDER_DARK)
            ax.spines["left"].set_color(COLOR_BORDER_DARK)
            ax.tick_params(axis="x", colors=COLOR_TEXT_SECONDARY, labelsize=8)
            ax.tick_params(axis="y", colors=COLOR_TEXT_SECONDARY, labelsize=8)
            ax.xaxis.label.set_color(COLOR_TEXT_DARK)
            ax.yaxis.label.set_color(COLOR_TEXT_DARK)
            ax.title.set_color(COLOR_TEXT_DARK)
            ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color=COLOR_BORDER_LIGHT,
            )
            # Set aspect ratio explicitly here after clearing
            ax.set_aspect("auto")

        # --- Dynamic Titles and Currency ---
        benchmark_display_name = (
            ", ".join(self.selected_benchmarks) if self.selected_benchmarks else "None"
        )
        display_currency = self._get_currency_symbol(get_name=True)
        currency_symbol = (
            self._get_currency_symbol()
        )  # Define the symbol to be used consistently
        return_graph_title = f"{scope_label} - Accumulated Gain (TWR)"
        value_graph_title = (
            f"{scope_label} - Value ({display_currency})"  # Use 3-letter code in title
        )

        # --- Get Selected Date Range for Limits ---
        plot_start_date = None
        plot_end_date = None
        try:
            plot_start_date = self.graph_start_date_edit.date().toPython()
            plot_end_date = self.graph_end_date_edit.date().toPython()
        except Exception as e:
            logging.error(f"Error get plot dates: {e}")  # Use logging.error

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
                color=COLOR_TEXT_SECONDARY,
            )
            self.abs_value_ax.text(
                0.5,
                0.5,
                "No Value Data",
                ha="center",
                va="center",
                transform=self.abs_value_ax.transAxes,
                fontsize=10,
                color=COLOR_TEXT_SECONDARY,
            )
            self.perf_return_ax.set_title(
                return_graph_title, fontsize=10, weight="bold"
            )
            self.abs_value_ax.set_title(value_graph_title, fontsize=10, weight="bold")
            if plot_start_date and plot_end_date:
                try:
                    self.perf_return_ax.set_xlim(plot_start_date, plot_end_date)
                    self.abs_value_ax.set_xlim(plot_start_date, plot_end_date)
                    self.perf_return_ax.set_ylim(-10, 10)  # Set default Y range
                    self.abs_value_ax.set_ylim(0, 100)  # Set default Y range
                except Exception as e:
                    logging.warning(f"Warn init limits: {e}")
            self.perf_return_canvas.draw()
            self.abs_value_canvas.draw()
            return

        # --- Data Prep (Use Full Data) ---
        results_df = self.historical_data.copy()
        if not isinstance(results_df.index, pd.DatetimeIndex):
            try:
                results_df.index = pd.to_datetime(results_df.index)
            except Exception as e:
                logging.error(f"ERROR index conv: {e}")
                return  # Exit if index is bad
        results_df.sort_index(inplace=True)

        # --- Filter data to the selected date range for Y limit calculation ---
        results_visible_df = results_df.copy()  # Default to full data
        if plot_start_date and plot_end_date:
            try:
                pd_start = pd.Timestamp(plot_start_date)
                pd_end = pd.Timestamp(plot_end_date)
                results_visible_df = results_df[
                    (results_df.index >= pd_start) & (results_df.index <= pd_end)
                ]
                logging.debug(
                    f"[Graph Update] Using date range {plot_start_date} to {plot_end_date} for Y-limit calculation ({len(results_visible_df)} rows)"
                )
            except Exception as e_filter:
                logging.warning(
                    f"Error filtering DataFrame by date for Y limits: {e_filter}"
                )
                results_visible_df = results_df  # Fallback to full range
        else:
            logging.warning(
                "[Graph Update] No date range specified for Y-limit filtering."
            )
            results_visible_df = results_df  # Fallback to full range

        # --- Calculate Y Ranges from VISIBLE Data ---
        min_y_ret, max_y_ret = np.inf, -np.inf
        min_y_val, max_y_val = np.inf, -np.inf
        port_plotted, bench_plotted_count, val_data_plotted = False, 0, False
        port_col = "Portfolio Accumulated Gain"
        if port_col in results_visible_df.columns:
            vg = results_visible_df[port_col].dropna()
            if not vg.empty:
                pct = (vg - 1) * 100
                min_y_ret = min(min_y_ret, pct.min())
                max_y_ret = max(max_y_ret, pct.max())
                port_plotted = True
        for b in self.selected_benchmarks:
            bc = f"{b} Accumulated Gain"
            if bc in results_visible_df.columns:
                vgb = results_visible_df[bc].dropna()
                if not vgb.empty:
                    pctb = (vgb - 1) * 100
                    min_y_ret = min(min_y_ret, pctb.min())
                    max_y_ret = max(max_y_ret, pctb.max())
                    bench_plotted_count += 1
        val_col = "Portfolio Value"
        # Use the already defined 'vv' variable consistently if needed later
        vv = (
            results_visible_df[val_col].dropna()
            if val_col in results_visible_df.columns
            else pd.Series(dtype=float)
        )
        val_data_plotted = not vv.empty  # Check if the resulting Series is empty
        if val_data_plotted:
            min_y_val = min(min_y_val, vv.min())
            max_y_val = max(max_y_val, vv.max())  # Use vv here
        logging.debug(
            f"[Graph Update] Visible RETURN Y Range (data): Min={min_y_ret}, Max={max_y_ret}"
        )
        logging.debug(
            f"[Graph Update] Visible VALUE Y Range (data): Min={min_y_val}, Max={max_y_val}"
        )

        # --- Plotting Setup ---
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        # --- Plot 1: Accumulated Gain (TWR %) ---
        self.perf_return_ax.clear()  # Clear axis
        self.perf_return_ax.set_aspect("auto")  # Set aspect again after clear
        port_plotted_full = False  # Track if anything was plotted from full data
        if (
            port_col in results_visible_df.columns
        ):  # Check if column exists in (potentially filtered) visible data
            vg_full = results_df[port_col].dropna()  # Get full data series for plotting
            if not vg_full.empty:
                pct_full = (vg_full - 1) * 100
                lbl = f"{scope_label}"
                self.perf_return_ax.plot(
                    pct_full.index,
                    pct_full,
                    label=lbl,
                    linewidth=2.0,
                    color=COLOR_ACCENT_TEAL,
                    zorder=10,
                )
                port_plotted_full = True  # Mark that we plotted this line

        bench_plotted_count_full = 0
        for i, b in enumerate(self.selected_benchmarks):
            bc = f"{b} Accumulated Gain"
            if bc in results_visible_df.columns:  # Check if column exists
                vgb_full = results_df[bc].dropna()  # Get full data series for plotting
                if not vgb_full.empty:
                    pctb_full = (vgb_full - 1) * 100
                    ci = i % len(colors)
                    # Ensure bcol is different from portfolio color if possible
                    bcol = (
                        colors[ci]
                        if colors[ci] != COLOR_ACCENT_TEAL
                        else colors[(ci + 1) % len(colors)]
                    )
                    self.perf_return_ax.plot(
                        pctb_full.index,
                        pctb_full,
                        label=f"{b}",
                        linewidth=1.5,
                        color=bcol,
                        alpha=0.8,
                    )
                    bench_plotted_count_full += 1

        # Add TWR Annotation (remains the same)
        if hasattr(self, "last_hist_twr_factor") and pd.notna(
            self.last_hist_twr_factor
        ):
            try:
                tfv = float(self.last_hist_twr_factor)
                tpg = (tfv - 1) * 100.0
                tt = f"Total TWR: {tpg:+.2f}%"
                # Use QColor to get the name string for QSS compatibility if needed, or directly use hex
                tc_color = QCOLOR_GAIN if tpg >= -1e-9 else QCOLOR_LOSS
                self.perf_return_ax.text(
                    0.02,
                    0.75,
                    tt,
                    transform=self.perf_return_ax.transAxes,
                    fontsize=9,
                    fontweight="bold",
                    color=tc_color.name(),  # Use .name() for hex string
                    va="top",
                    ha="left",
                    bbox=dict(
                        boxstyle="round,pad=0.3", fc="white", alpha=0.7, ec="none"
                    ),
                )
            except Exception as e:
                logging.warning(f"Warn TWR text: {e}")

        # Format Return Plot
        # Use the flags indicating if *anything* was plotted from the full data
        if port_plotted_full or bench_plotted_count_full > 0:
            self.perf_return_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
            # Calculate number of legend items based on what was actually plotted
            nl = bench_plotted_count_full + (1 if port_plotted_full else 0)
            if nl > 0:  # Only show legend if there are items
                self.perf_return_ax.legend(
                    fontsize=8,
                    ncol=min(3, nl),
                    loc="upper center",
                    bbox_to_anchor=(0.33, 0.98),
                )
            self.perf_return_ax.set_title(
                return_graph_title, fontsize=10, weight="bold", color=COLOR_TEXT_DARK
            )
            self.perf_return_ax.set_ylabel(
                "Accumulated Gain (%)", fontsize=9, color=COLOR_TEXT_DARK
            )
            self.perf_return_ax.grid(
                True,
                which="major",
                linestyle="--",
                linewidth=0.5,
                color=COLOR_BORDER_LIGHT,
            )

            # --- SET RETURN Y LIMITS (using VISIBLE range min/max calculated earlier) ---
            try:
                # Use min_y_ret and max_y_ret calculated from results_visible_df
                if (
                    np.isfinite(min_y_ret)
                    and np.isfinite(max_y_ret)
                    and max_y_ret >= min_y_ret
                ):
                    yr = max_y_ret - min_y_ret
                    pad = max(yr * 0.05, 1.0)
                    fmin = min_y_ret - pad
                    fmax = max_y_ret + pad
                    logging.debug(
                        f"[Graph Update] Setting RETURN ylim based on visible range: {fmin:.2f} to {fmax:.2f}"
                    )
                    self.perf_return_ax.set_ylim(fmin, fmax)
                # --- Correction: Use port_plotted and bench_plotted_count (based on visible data) for fallback check ---
                elif port_plotted or bench_plotted_count > 0:
                    logging.debug(
                        "[Graph Update] RETURN visible ylim invalid, using autoscale."
                    )
                    self.perf_return_ax.autoscale(enable=True, axis="y", tight=False)
                else:  # Neither visible portfolio nor visible benchmarks had data
                    self.perf_return_ax.set_ylim(
                        -10, 10
                    )  # Default if nothing plotted in visible range
            except Exception as e:
                logging.warning(f"Warn RET ylim: {e}")
                self.perf_return_ax.autoscale(
                    enable=True, axis="y", tight=False
                )  # Fallback
            # --- END SET RETURN Y LIMITS ---

        else:  # Case where nothing was plotted from the full data either
            self.perf_return_ax.text(
                0.5,
                0.5,
                "No Return Data",
                ha="center",
                va="center",
                transform=self.perf_return_ax.transAxes,
                fontsize=10,
                color=COLOR_TEXT_SECONDARY,
            )
            self.perf_return_ax.set_title(
                return_graph_title, fontsize=10, weight="bold", color=COLOR_TEXT_DARK
            )
            self.perf_return_ax.set_ylim(-10, 10)  # Default Y range if no data

        # --- Apply Layout Adjustments ---
        self.perf_return_fig.tight_layout(pad=0.3)
        self.perf_return_fig.autofmt_xdate(rotation=15)
        # --- SET RETURN X LIMITS (based on GUI edits) ---
        try:
            if plot_start_date and plot_end_date:
                logging.debug(
                    f"[Graph Update] Setting RETURN xlim: {plot_start_date} to {plot_end_date}"
                )
                self.perf_return_ax.set_xlim(plot_start_date, plot_end_date)
        except Exception as e:
            logging.warning(f"Warn RET xlim: {e}")
        # --- Draw Canvas ---
        self.perf_return_canvas.draw()

        # --- Plot 2: Absolute Value ---
        self.abs_value_ax.clear()  # Clear axis
        self.abs_value_ax.set_aspect("auto")  # Set aspect again after clear
        if (
            val_data_plotted
        ):  # Check if there was valid data in the visible range to justify plotting
            vv_full = results_df[val_col].dropna()  # Get full value data series
            if not vv_full.empty:
                # Use currency_symbol defined earlier in the function
                self.abs_value_ax.plot(
                    vv_full.index,
                    vv_full,
                    label=f"{scope_label} Value ({currency_symbol})",
                    color="green",
                    linewidth=1.5,
                )  # Plot full data

                # Define the formatter function HERE, ensuring currency_symbol is accessible
                def currency_formatter(x, pos):
                    # Need to access currency_symbol defined in the outer scope
                    local_currency_symbol = (
                        self._get_currency_symbol()
                    )  # Or pass currency_symbol if needed
                    if pd.isna(x):
                        return "N/A"
                    try:
                        if abs(x) >= 1e6:
                            return f"{local_currency_symbol}{x/1e6:,.1f}M"
                        if abs(x) >= 1e3:
                            return f"{local_currency_symbol}{x/1e3:,.0f}K"
                        return f"{local_currency_symbol}{x:,.0f}"
                    except TypeError:
                        return "Err"  # Catch potential type errors during formatting

                formatter = mtick.FuncFormatter(
                    currency_formatter
                )  # Create formatter instance
                self.abs_value_ax.yaxis.set_major_formatter(
                    formatter
                )  # Apply formatter
                self.abs_value_ax.set_title(
                    value_graph_title, fontsize=10, weight="bold", color=COLOR_TEXT_DARK
                )
                self.abs_value_ax.set_ylabel(
                    f"Value ({currency_symbol})", fontsize=9, color=COLOR_TEXT_DARK
                )  # Use currency_symbol here too
                self.abs_value_ax.grid(
                    True,
                    which="major",
                    linestyle="--",
                    linewidth=0.5,
                    color=COLOR_BORDER_LIGHT,
                )

                # --- SET VALUE Y LIMITS (using VISIBLE range min/max) ---
                try:
                    if (
                        np.isfinite(min_y_val)
                        and np.isfinite(max_y_val)
                        and max_y_val >= min_y_val
                    ):
                        yrv = max_y_val - min_y_val
                        padv = max(yrv * 0.05, 10.0)
                        fminv = min_y_val - padv
                        if min_y_val >= -1e-6:
                            fminv = max(0, fminv)
                        fmaxv = max_y_val + padv
                        if abs(fmaxv - fminv) < 1e-6:
                            fmaxv += 1
                        logging.debug(
                            f"[Graph Update] Setting VALUE ylim based on visible range: {fminv:.2f} to {fmaxv:.2f}"
                        )
                        self.abs_value_ax.set_ylim(fminv, fmaxv)
                    else:
                        logging.info(
                            "[Graph Update] VALUE visible ylim invalid, using autoscale."
                        )
                        self.abs_value_ax.autoscale(enable=True, axis="y", tight=False)
                except Exception as e:
                    logging.warning(f"Warn VAL ylim: {e}")
                    self.abs_value_ax.autoscale(enable=True, axis="y", tight=False)
                # --- END SET VALUE Y LIMITS ---
            else:  # Full data was empty after dropna
                self.abs_value_ax.text(
                    0.5,
                    0.5,
                    "No Value Data",
                    ha="center",
                    va="center",
                    transform=self.abs_value_ax.transAxes,
                    fontsize=10,
                    color=COLOR_TEXT_SECONDARY,
                )
                self.abs_value_ax.set_title(
                    value_graph_title, fontsize=10, weight="bold", color=COLOR_TEXT_DARK
                )
                self.abs_value_ax.set_ylim(0, 100)
        else:  # No valid data in visible range
            self.abs_value_ax.text(
                0.5,
                0.5,
                "No Value Data in Range",
                ha="center",
                va="center",
                transform=self.abs_value_ax.transAxes,
                fontsize=10,
                color=COLOR_TEXT_SECONDARY,
            )
            self.abs_value_ax.set_title(
                value_graph_title, fontsize=10, weight="bold", color=COLOR_TEXT_DARK
            )
            self.abs_value_ax.set_ylim(0, 100)
        # --- Apply Layout Adjustments ---
        self.abs_value_fig.tight_layout(pad=0.3)
        self.abs_value_fig.autofmt_xdate(rotation=15)
        # --- SET VALUE X LIMITS (based on GUI edits) ---
        try:
            if plot_start_date and plot_end_date:
                logging.debug(
                    f"[Graph Update] Setting VALUE xlim: {plot_start_date} to {plot_end_date}"
                )
                self.abs_value_ax.set_xlim(plot_start_date, plot_end_date)
        except Exception as e:
            logging.warning(f"Warn VAL xlim: {e}")
        # --- Draw Canvas ---
        self.abs_value_canvas.draw()

        # Re-apply backgrounds
        # ... (background setting code remains the same) ...
        try:
            pc = "#FFFFFF"
            pf = "#F8F9FA"
            for f in [self.account_fig, self.holdings_fig]:
                f.patch.set_facecolor(pc)
            for a in [self.account_ax, self.holdings_ax]:
                a.patch.set_facecolor(pc)
            for f in [self.perf_return_fig, self.abs_value_fig]:
                f.patch.set_facecolor(pf)
            for a in [self.perf_return_ax, self.abs_value_ax]:
                a.patch.set_facecolor(pf)
        except Exception as e:
            logging.warning(f"Warn bg: {e}")

    # --- Data Handling and UI Update Methods ---

    def update_dashboard_summary(self):
        """
        Updates the summary grid labels with aggregated portfolio metrics.

        Uses data from `self.summary_metrics_data` (which reflects the selected
        account scope) and `self.holdings_data` (for cash balance). Formats values
        with currency symbols and applies gain/loss coloring. Calculates and displays
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
            "day_change": self.summary_day_change,
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

        # Data source is always the overall summary metrics, which now reflect the selected scope
        data_source_current = self.summary_metrics_data

        # Update common summary items using data_source_current
        if data_source_current:
            # --- Cash calculation needs to use the filtered holdings_data ---
            overall_cash_value = None
            if (
                isinstance(self.holdings_data, pd.DataFrame)
                and not self.holdings_data.empty
                and cash_val_col_actual
                and "Symbol" in self.holdings_data.columns
                and cash_val_col_actual in self.holdings_data.columns
            ):
                try:
                    # Filter holdings_data based on selected accounts before summing cash
                    df_filtered_for_cash = (
                        self._get_filtered_data()
                    )  # Use existing filter logic
                    cash_mask = df_filtered_for_cash["Symbol"] == CASH_SYMBOL_CSV
                    # Also handle the display format "Cash (CUR)"
                    cash_display_symbol = (
                        f"Cash ({self._get_currency_symbol(get_name=True)})"
                    )
                    cash_mask |= df_filtered_for_cash["Symbol"] == cash_display_symbol

                    overall_cash_value = (
                        pd.to_numeric(
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
            day_change_text_override = "N/A"

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
                day_change_text_override = day_change_abs_val_str + day_change_pct_str
            # --- END FIX ---

            self.update_summary_value(
                self.summary_day_change[1],
                day_change_val,
                currency_symbol,
                True,
                False,
                "day_change",
                day_change_text_override,
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
        # Update Label Text
        if is_all_accounts_selected:
            self.summary_total_return_pct[0].setText("Total Ret %:")
        else:
            self.summary_total_return_pct[0].setText(f"Sel. Ret %:")
        self.summary_total_return_pct[0].setVisible(True)
        self.summary_total_return_pct[1].setVisible(True)

        # Slot 2: Annualized TWR %
        twr_factor = (
            self.last_hist_twr_factor
        )  # This factor now reflects the selected scope
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
        # Update Label Text
        if is_all_accounts_selected:
            self.summary_annualized_twr[0].setText("Ann. TWR %:")
        else:
            self.summary_annualized_twr[0].setText(f"Sel. TWR %:")
        self.summary_annualized_twr[0].setVisible(True)
        self.summary_annualized_twr[1].setVisible(True)

    def update_summary_value(
        self,
        value_label,
        value,
        currency_symbol="$",
        is_currency=True,
        is_percent=False,
        metric_type=None,
        override_text=None,
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
            override_text (str | None, optional): If provided, sets the label's text directly,
                bypassing formatting. Used for combined values like day change. Defaults to None.
        """
        # (Keep implementation as before)
        text = "N/A"
        original_value_float = None
        target_color = QCOLOR_TEXT_DARK
        if value is not None and pd.notna(value):
            try:
                original_value_float = float(value)
            except (ValueError, TypeError):
                text = "Error"
                target_color = QCOLOR_LOSS
        if original_value_float is not None:
            if metric_type in ["net_value", "cash"]:
                target_color = QCOLOR_TEXT_DARK
            elif metric_type == "dividends":
                target_color = (
                    QCOLOR_GAIN if original_value_float > 1e-9 else QCOLOR_TEXT_DARK
                )
            # Fees and commissions are usually displayed as positive numbers but represent losses/costs
            elif metric_type in ["fees", "commissions"]:
                target_color = (
                    QCOLOR_LOSS if original_value_float > 1e-9 else QCOLOR_TEXT_DARK
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
                    target_color = QCOLOR_GAIN
                elif original_value_float < -1e-9:
                    target_color = QCOLOR_LOSS
                else:
                    target_color = QCOLOR_TEXT_DARK
            # Explicitly color Total Return %
            elif metric_type in [
                "account_total_return_pct",
                "overall_total_return_pct",
                "annualized_twr",
                "total_return_pct",
            ]:  # Added total_return_pct
                if original_value_float > 1e-9:
                    target_color = QCOLOR_GAIN
                elif original_value_float < -1e-9:
                    target_color = QCOLOR_LOSS
                else:
                    target_color = QCOLOR_TEXT_DARK
            # Default coloring
            else:
                if original_value_float > 1e-9:
                    target_color = QCOLOR_GAIN
                elif original_value_float < -1e-9:
                    target_color = QCOLOR_LOSS
                else:
                    target_color = QCOLOR_TEXT_DARK

        if override_text is not None:
            text = override_text
        elif original_value_float is not None:
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
                target_color = QCOLOR_LOSS
        elif metric_type == "clear":
            text = "N/A"
            target_color = QCOLOR_TEXT_DARK
        value_label.setText(text)
        palette = value_label.palette()
        palette.setColor(QPalette.WindowText, target_color)
        value_label.setPalette(palette)

    @Slot()
    def show_manage_transactions_dialog(self):
        """Opens the dialog to manage transactions."""
        if not hasattr(self, "original_data") or self.original_data.empty:
            # Try loading it now if not loaded on startup/refresh
            account_map = self.config.get("account_currency_map", {"SET": "THB"})
            def_currency = self.config.get("default_currency", "USD")
            logging.info("Loading original transaction data for management dialog...")
            (_, orig_df_temp, _, _, err_load_orig, _) = (
                load_and_clean_transactions(  # <--- Direct call
                    self.transactions_file, account_map, def_currency
                )
            )
            if err_load_orig or orig_df_temp is None:
                QMessageBox.critical(
                    self,
                    "Load Error",
                    "Failed to load transaction data for management.",
                )
                return
            else:
                self.original_data = orig_df_temp.copy()

        if self.original_data.empty:
            QMessageBox.information(
                self, "No Data", "No transaction data loaded to manage."
            )
            return

        dialog = ManageTransactionsDialog(
            self.original_data, self.transactions_file, self
        )
        # Connect the dialog's signal back to the main window's refresh
        # dialog.data_changed.connect(self.refresh_data) # Let the _edit/_delete methods trigger refresh
        dialog.exec()

    def _backup_csv(self):
        """Creates a timestamped backup of the transactions CSV."""
        if not self.transactions_file or not os.path.exists(self.transactions_file):
            return False, "CSV file not found."
        try:
            backup_dir = "csv_backups"
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.basename(self.transactions_file)
            name, ext = os.path.splitext(base_name)
            backup_filename = f"{name}_{timestamp}{ext}"
            backup_path = os.path.join(backup_dir, backup_filename)
            shutil.copy2(
                self.transactions_file, backup_path
            )  # copy2 preserves metadata
            logging.info(f"CSV backup created: {backup_path}")
            return True, backup_path
        except Exception as e:
            logging.error(f"ERROR creating CSV backup: {e}")
            return False, str(e)

    def _rewrite_csv(self, df_to_write: pd.DataFrame) -> bool:
        """Rewrites the entire CSV file from the DataFrame."""
        if not self.transactions_file:
            QMessageBox.critical(self, "Save Error", "CSV file path is not set.")
            return False

        # Define headers in the correct order
        csv_headers = [
            "Date (MMM DD, YYYY)",
            "Transaction Type",
            "Stock / ETF Symbol",
            "Quantity of Units",
            "Amount per unit",
            "Total Amount",
            "Fees",
            "Investment Account",
            "Split Ratio (new shares per old share)",
            "Note",
        ]
        # Select and reorder columns to match headers
        df_ordered = pd.DataFrame(
            columns=csv_headers
        )  # Start with an empty df with correct columns
        for header in csv_headers:
            # Find the corresponding column in df_to_write (handle potential renames/missing)
            # This assumes df_to_write uses the *original* CSV headers
            if header in df_to_write.columns:
                df_ordered[header] = df_to_write[header]
            else:
                df_ordered[header] = ""  # Add empty column if missing in source

        try:
            # --- Backup BEFORE writing ---
            backup_ok, backup_msg = self._backup_csv()
            if not backup_ok:
                QMessageBox.critical(
                    self,
                    "Backup Error",
                    f"Failed to backup CSV before saving:\n{backup_msg}",
                )
                return False

            # --- Write to CSV ---
            # Use specific float formatting to avoid excessive decimals
            float_format = "%.8f"  # Adjust precision as needed
            df_ordered.to_csv(
                self.transactions_file,
                index=False,
                encoding="utf-8",
                quoting=csv.QUOTE_NONNUMERIC,  # Quote non-numeric fields
                date_format=CSV_DATE_FORMAT,  # Use the specific date format
                float_format=float_format,
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
        """Finds a transaction by original_index, updates it, and rewrites the CSV."""
        if not hasattr(self, "original_data") or self.original_data.empty:
            QMessageBox.critical(
                self,
                "Data Error",
                "Original transaction data not available for editing.",
            )
            return False

        # Work on a copy
        df_modified = self.original_data.copy()

        # Find the row index in the DataFrame corresponding to the original_index
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
            if csv_header in df_modified.columns:
                # Convert back to appropriate type before setting?
                # Or just set the string value? Pandas might handle it on write.
                # Let's set the string value directly for simplicity, to_csv handles quoting.
                df_modified.loc[df_row_index, csv_header] = new_value_str
            # else: Column from dialog doesn't exist in original_data? Should not happen.

        # Rewrite the entire CSV
        if self._rewrite_csv(df_modified):
            QMessageBox.information(
                self, "Success", "Transaction updated successfully."
            )
            self.refresh_data()  # Refresh data after successful edit
            return True
        else:
            # Rewrite failed, error message shown in _rewrite_csv
            return False

    def _delete_transactions_from_csv(
        self, original_indices_to_delete: List[int]
    ) -> bool:
        """Removes transactions by original_index and rewrites the CSV."""
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
    def select_file(self):
        """Opens a file dialog to select a new transactions CSV file."""
        # (Keep implementation as before)
        start_dir = (
            os.path.dirname(self.transactions_file)
            if self.transactions_file
            and os.path.exists(os.path.dirname(self.transactions_file))
            else os.getcwd()
        )
        fname, _ = QFileDialog.getOpenFileName(
            self, "Open Transactions CSV", start_dir, "CSV Files (*.csv)"
        )
        if fname and fname != self.transactions_file:
            self.transactions_file = fname
            self.config["transactions_file"] = fname
            logging.info(f"Selected new file: {self.transactions_file}")
            self.clear_results()  # Clear results including available accounts
            self.refresh_data()  # Refresh will load new data and accounts

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
        # Keep selected_accounts as loaded from config, validation happens on load
        self._update_table_view_with_filtered_columns(pd.DataFrame())
        self.apply_column_visibility()
        self.update_dashboard_summary()
        self.update_account_pie_chart()
        self.update_holdings_pie_chart(pd.DataFrame())
        self.update_performance_graphs(initial=True)
        self.status_label.setText("Ready")
        self._update_table_title()  # Update table title
        self._update_account_button_text()  # Update button text
        self._update_fx_rate_display(self.currency_combo.currentText())
        self.update_header_info(loading=True)
        if hasattr(self, "view_ignored_button"):
            self.view_ignored_button.setEnabled(False)  # Disable when clearing

    # --- Filter Change Handlers ---
    @Slot()
    def filter_changed_refresh(self):
        """
        Slot triggered by changes requiring a full data recalculation.

        Connected to 'Currency' combo box and 'Show Closed' checkbox changes.
        Calls `refresh_data`.
        """
        sender = self.sender()
        changed_control = "Unknown Filter"
        if sender == self.currency_combo:
            changed_control = "Currency"
            self._ensure_all_columns_in_visibility()
        elif sender == self.show_closed_check:
            changed_control = "'Show Closed' Checkbox"
        logging.info(f"Filter change ({changed_control}) requires full refresh...")
        self.refresh_data()  # Trigger the main refresh function

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
                df_for_table = df_intermediate.rename(columns=actual_to_ui_map)
        self.table_model.updateData(df_for_table)
        if not df_for_table.empty:
            self.table_view.resizeColumnsToContents()
            try:
                display_currency = self.currency_combo.currentText()
                col_widths = {
                    "Symbol": 100,
                    "Account": 90,
                    "Quantity": 90,
                    f"Day Chg": 95,
                    "Day Chg %": 75,
                    "Mkt Val": 110,
                    "Unreal. G/L": 95,
                    "Unreal. G/L %": 95,
                    "Total Ret %": 80,
                    "Real. G/L": 95,
                    "IRR (%)": 70,
                    "Fees": 70,
                    "Divs": 80,
                    "Avg Cost": 70,
                    "Price": 70,
                    "Cost Basis": 100,
                }
                for ui_header_name, width in col_widths.items():
                    if ui_header_name in df_for_table.columns:
                        col_index = df_for_table.columns.get_loc(ui_header_name)
                        self.table_view.setColumnWidth(col_index, width)
                self.table_view.horizontalHeader().setStretchLastSection(False)
            except Exception as e:
                logging.warning(f"Warning: Could not set specific column width: {e}")
        else:
            try:
                self.table_view.horizontalHeader().setStretchLastSection(False)
            except Exception as e:
                logging.warning(f"Warning: Could not unset stretch last section: {e}")

    def _get_filtered_data(self):
        """
        Filters the main holdings DataFrame based on selected accounts and 'Show Closed'.

        Uses `self.holdings_data`, `self.selected_accounts`, `self.available_accounts`,
        and the state of the 'Show Closed' checkbox.

        Returns:
            pd.DataFrame: The filtered DataFrame ready for display or charting.
        """
        df_filtered = pd.DataFrame()
        if (
            isinstance(self.holdings_data, pd.DataFrame)
            and not self.holdings_data.empty
        ):
            df_to_filter = self.holdings_data.copy()

            # --- Filter by selected accounts ---
            # Use self.selected_accounts. If empty or matches all available, no filtering needed.
            all_selected_or_empty = not self.selected_accounts or (
                set(self.selected_accounts) == set(self.available_accounts)
                if self.available_accounts
                else True
            )

            if not all_selected_or_empty and "Account" in df_to_filter.columns:
                df_filtered = df_to_filter[
                    df_to_filter["Account"].isin(self.selected_accounts)
                ].copy()
            else:  # Use all accounts if selection is empty/all or Account column missing
                df_filtered = df_to_filter
            # --- End Account Filter ---

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
                        df_filtered["Symbol"] == CASH_SYMBOL_CSV
                    )
                    df_filtered = df_filtered[keep_mask]
                except Exception as e:
                    logging.warning(f"Warning: Error filtering 'Show Closed': {e}")
        return df_filtered

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
                    rate_text = f"FX: 1 {base_currency} = {display_currency} {abs(rate_float):,.4f}"
                    show_rate = True
                except (ValueError, TypeError):
                    rate_text = f"FX: Invalid Rate"
                    show_rate = True
            else:
                rate_text = f"FX: ({base_currency}->{display_currency}) Unavailable"  # Indicate which rate is missing
                show_rate = True

        self.exchange_rate_display_label.setText(rate_text)
        self.exchange_rate_display_label.setVisible(show_rate)

    # --- New Helper: Update Table Title ---
    def _update_table_title(self):
        """Updates the title above the main holdings table to reflect the current scope."""
        title_right_label = getattr(self, "table_title_label_right", None)
        title_left_label = getattr(self, "table_title_label_left", None)

        if not title_right_label or not title_left_label:
            return

        num_available = len(self.available_accounts)
        num_selected = len(self.selected_accounts)
        df_display_filtered = self._get_filtered_data()  # Get currently displayed data
        num_rows_displayed = len(df_display_filtered)

        title_right_text = f"Holdings Detail ({num_rows_displayed} items shown)"
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

    # --- End New Helper ---

    # --- Main Data Refresh Logic ---
    @Slot()
    def refresh_data(self):
        """
        Initiates the background calculation process via the worker thread.

        Gathers current UI settings (file path, currency, filters, graph parameters),
        prepares arguments for the portfolio logic functions, creates a
        `PortfolioCalculatorWorker`, and starts it in the thread pool. Disables
        UI controls during calculation.
        """
        sender = self.sender()
        trigger_source = "Unknown"
        if sender == self.refresh_button:
            trigger_source = "'Refresh All' Button"
        elif sender == self.update_accounts_button:
            trigger_source = "'Update Accounts' Button"  # Catch the new button
        elif sender == self.graph_update_button:
            trigger_source = "'Update Graphs' Button"
        elif sender == self.currency_combo:
            trigger_source = "Currency Change"
        elif sender == self.show_closed_check:
            trigger_source = "'Show Closed' Change"
        elif isinstance(sender, QAction) and sender.parent() == self:
            trigger_source = "Menu Action"
        elif sender is None:
            trigger_source = "Programmatic Call"
        logging.info(f"Refresh triggered by: {trigger_source}")

        if not self.transactions_file or not os.path.exists(self.transactions_file):
            QMessageBox.warning(
                self,
                "Missing File",
                f"Transactions CSV file not found:\n{self.transactions_file}",
            )
            self.status_label.setText("Error: Select a valid transactions CSV file.")
            return

        # --- Load original data FIRST ---
        # Use the maps/defaults from config stored in self.config
        account_map = self.config.get("account_currency_map", {"SET": "THB"})
        def_currency = self.config.get("default_currency", "USD")
        logging.info("Loading original transaction data for potential management...")
        (
            all_tx_df_temp,  # Temp var for all tx
            orig_df_temp,  # This is what we want to store
            _,  # ignored indices - not needed here again
            _,  # ignored reasons - not needed here again
            err_load_orig,
            warn_load_orig,
        ) = load_and_clean_transactions(
            self.transactions_file, account_map, def_currency
        )  # <-- Direct call
        if err_load_orig or orig_df_temp is None:
            QMessageBox.critical(
                self,
                "Load Error",
                "Failed to load original transaction data for management.",
            )
            # Don't store potentially bad data
            self.original_data = pd.DataFrame()
            # Allow proceeding maybe? Or stop? Let's stop for now.
            self.calculation_finished()  # Re-enable controls
            return
        else:
            # Store the original DataFrame with its index
            self.original_data = orig_df_temp.copy()
            logging.info(f"Stored original data with {len(self.original_data)} rows.")
            if warn_load_orig:
                self.status_label.setText("Warning during initial data load check.")
        # --- End Load original data ---

        self.is_calculating = True
        now_str = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        self.status_label.setText(f"Refreshing data... ({now_str})")
        self.set_controls_enabled(False)
        display_currency = self.currency_combo.currentText()
        show_closed = self.show_closed_check.isChecked()
        start_date = self.graph_start_date_edit.date().toPython()
        end_date = self.graph_end_date_edit.date().toPython()
        interval = self.graph_interval_combo.currentText()
        selected_benchmarks_list = self.selected_benchmarks
        api_key = self.fmp_api_key

        # --- Get currency map and default currency from config ---
        account_map = self.config.get(
            "account_currency_map", {"SET": "THB"}
        )  # Use default if missing
        def_currency = self.config.get(
            "default_currency", "USD"
        )  # Use default if missing
        # ---------------------------------------------------------

        # Use self.selected_accounts. If it's empty, it means "All", so pass None or empty list to logic
        # The backend logic handles empty list as "all"
        selected_accounts_for_logic = (
            self.selected_accounts if self.selected_accounts else None
        )

        if start_date >= end_date:
            QMessageBox.warning(
                self, "Invalid Date Range", "Graph start date must be before end date."
            )
            self.calculation_finished()
            return
        if not selected_benchmarks_list:
            QMessageBox.warning(
                self, "No Benchmark Selected", "Please select at least one benchmark."
            )
            self.calculation_finished()
            return

        # Determine accounts to exclude for historical calculation if feature supported
        accounts_to_exclude = []
        if HISTORICAL_FN_SUPPORTS_EXCLUDE:
            # Example logic: If "All" are selected, exclude 'SET' by default
            # if not selected_accounts_for_logic: # i.e., All are selected
            #     accounts_to_exclude = ["SET"] # Replace with your default exclusion logic
            # else:
            #     accounts_to_exclude = [] # Don't exclude if specific accounts are selected
            pass  # Keep exclusion logic minimal for now, user doesn't control it via GUI yet

        logging.info(f"Starting calculation & data fetch:")
        logging.info(
            f"File='{os.path.basename(self.transactions_file)}', Currency='{display_currency}', ShowClosed={show_closed}, SelectedAccounts={selected_accounts_for_logic if selected_accounts_for_logic else 'All'}"
        )
        logging.info(
            f"Default Currency: {def_currency}, Account Map: {account_map}"
        )  # Log currency info
        exclude_log_msg = (
            f", ExcludeHist={accounts_to_exclude}"
            if HISTORICAL_FN_SUPPORTS_EXCLUDE and accounts_to_exclude
            else ""
        )
        logging.info(
            f"Graph Params: Start={start_date}, End={end_date}, Interval={interval}, Benchmarks={selected_benchmarks_list}{exclude_log_msg}"
        )

        # --- Pass account_map and def_currency to worker ---
        portfolio_args = ()
        portfolio_kwargs = {
            "transactions_csv_file": self.transactions_file,
            "display_currency": display_currency,
            "show_closed_positions": show_closed,
            "account_currency_map": account_map,  # <-- Pass map
            "default_currency": def_currency,  # <-- Pass default
            "cache_file_path": "portfolio_cache_yf.json",
            "fmp_api_key": api_key,
            "include_accounts": selected_accounts_for_logic,
        }
        historical_args = ()
        historical_kwargs = {
            "transactions_csv_file": self.transactions_file,
            "start_date": start_date,
            "end_date": end_date,
            "interval": interval,
            "benchmark_symbols_yf": selected_benchmarks_list,
            "display_currency": display_currency,
            "account_currency_map": account_map,  # <-- Pass map
            "default_currency": def_currency,  # <-- Pass default
            "use_raw_data_cache": True,
            "use_daily_results_cache": True,
            "include_accounts": selected_accounts_for_logic,
        }
        if HISTORICAL_FN_SUPPORTS_EXCLUDE:
            historical_kwargs["exclude_accounts"] = accounts_to_exclude
        # ----------------------------------------------------

        worker = PortfolioCalculatorWorker(
            portfolio_fn=calculate_portfolio_summary,
            portfolio_args=portfolio_args,
            portfolio_kwargs=portfolio_kwargs,
            index_fn=fetch_index_quotes_yfinance,
            historical_fn=calculate_historical_performance,
            historical_args=historical_args,
            historical_kwargs=historical_kwargs,
        )
        worker.signals.result.connect(self.handle_results)
        worker.signals.error.connect(self.handle_error)
        worker.signals.finished.connect(self.calculation_finished)
        self.threadpool.start(worker)

    # --- Control Enabling/Disabling ---
    def set_controls_enabled(self, enabled: bool):
        """
        Enables or disables main UI controls during calculations.

        Args:
            enabled (bool): True to enable controls, False to disable.
        """
        controls_to_toggle = [
            self.select_file_button,
            self.add_transaction_button,
            self.account_select_button,  # Use new button
            self.update_accounts_button,  # <-- ADDED Update Accounts button
            self.currency_combo,
            self.show_closed_check,
            self.graph_start_date_edit,
            self.graph_end_date_edit,
            self.graph_interval_combo,
            self.benchmark_select_button,
            self.graph_update_button,
            self.refresh_button,
        ]
        for control in controls_to_toggle:
            try:
                control.setEnabled(enabled)
            except AttributeError:
                pass
        self.setCursor(Qt.WaitCursor if not enabled else Qt.ArrowCursor)

    # --- Signal Handlers from Worker ---
    @Slot(dict, pd.DataFrame, pd.DataFrame, dict, dict, pd.DataFrame)
    def handle_results(
        self,
        summary_metrics,
        holdings_df,
        ignored_df,
        account_metrics,
        index_quotes,
        historical_data_df,
    ):
        """
        Slot to process results received from the PortfolioCalculatorWorker.

        Updates internal data attributes (`self.holdings_data`, `self.summary_metrics_data`, etc.),
        updates the list of available accounts, validates the current account selection,
        and triggers updates for all UI elements (summary, charts, table).

        Args:
            summary_metrics (dict): Aggregated metrics for the overall portfolio.
            holdings_df (pd.DataFrame): Detailed holdings data.
            ignored_df (pd.DataFrame): DataFrame of transactions ignored during processing.
            account_metrics (dict): Dictionary of metrics aggregated per account.
            index_quotes (dict): Fetched data for header indices.
            historical_data_df (pd.DataFrame): Calculated historical performance data.
        """
        portfolio_status = summary_metrics.pop("status_msg", "Status Unknown")
        historical_status = summary_metrics.pop(
            "historical_status_msg", "Status Unknown"
        )
        self.last_calc_status = f"{portfolio_status} | {historical_status}"
        self.last_hist_twr_factor = np.nan
        if "|||TWR_FACTOR:" in historical_status:
            try:
                twr_part = historical_status.split("|||TWR_FACTOR:")[1]
                self.last_hist_twr_factor = float(twr_part)
            except (IndexError, ValueError, TypeError) as e_twr:
                logging.warning(
                    f"Warn: Could not parse TWR factor from status: {e_twr}"
                )

        self.summary_metrics_data = summary_metrics if summary_metrics else {}
        self.holdings_data = (
            holdings_df if holdings_df is not None else pd.DataFrame()
        )  # Store unfiltered holdings
        self.ignored_data = ignored_df if ignored_df is not None else pd.DataFrame()
        if hasattr(self, "view_ignored_button"):
            self.view_ignored_button.setEnabled(not self.ignored_data.empty)

        self.account_metrics_data = account_metrics if account_metrics else {}
        self.index_quote_data = index_quotes if index_quotes else {}

        # --- STORE THE FULL HISTORICAL DATA ---
        self.historical_data = (
            historical_data_df if historical_data_df is not None else pd.DataFrame()
        )
        # --- ADD DEBUG PRINT HERE ---
        if (
            isinstance(self.historical_data, pd.DataFrame)
            and not self.historical_data.empty
        ):
            logging.info(
                f"[Handle Results] Stored historical data from {self.historical_data.index.min()} to {self.historical_data.index.max()} ({len(self.historical_data)} rows)"
            )
        else:
            logging.info("[Handle Results] Stored historical data is EMPTY or None.")
        # --- END DEBUG PRINT ---

        # --- Update Available Accounts & Validate Selection ---
        # ... (account handling remains the same) ...
        available_accounts_from_backend = self.summary_metrics_data.pop(
            "_available_accounts", None
        )
        if available_accounts_from_backend and isinstance(
            available_accounts_from_backend, list
        ):
            self.available_accounts = available_accounts_from_backend
        else:
            if not self.holdings_data.empty and "Account" in self.holdings_data.columns:
                self.available_accounts = sorted(
                    self.holdings_data["Account"].unique().tolist()
                )
            else:
                self.available_accounts = []

        if self.selected_accounts:
            original_selection = self.selected_accounts.copy()
            self.selected_accounts = [
                acc for acc in self.selected_accounts if acc in self.available_accounts
            ]
            if len(self.selected_accounts) != len(original_selection):
                logging.warning(
                    f"Warn: Some previously selected accounts are no longer available. Updated selection: {self.selected_accounts}"
                )
            if not self.selected_accounts and self.available_accounts:
                logging.warning(
                    "Warn: Validation resulted in empty selection. Defaulting to all available accounts."
                )
                self.selected_accounts = self.available_accounts.copy()
        elif not self.selected_accounts and self.available_accounts:
            self.selected_accounts = self.available_accounts.copy()
        self._update_account_button_text()

        logging.debug("DEBUG: Updating UI elements...")
        try:
            # --- Get Filtered Data for Display (for table/holdings pie) ---
            df_display_filtered = self._get_filtered_data()

            # --- Update Table Title ---
            self._update_table_title()

            # --- Update Rest of UI ---
            self.update_dashboard_summary()
            account_pie_data = pd.DataFrame()
            if not self.holdings_data.empty and "Account" in self.holdings_data.columns:
                # Filter holdings_data by selected accounts for the account pie chart
                account_pie_data = (
                    self.holdings_data[
                        self.holdings_data["Account"].isin(self.selected_accounts)
                    ].copy()
                    if self.selected_accounts
                    else self.holdings_data.copy()
                )
            self.update_account_pie_chart(account_pie_data)  # Pass scoped data
            self.update_holdings_pie_chart(
                df_display_filtered
            )  # Uses account+closed filtered data
            self._update_table_view_with_filtered_columns(df_display_filtered)
            self.apply_column_visibility()
            self.update_performance_graphs()  # <--- THIS SHOULD USE THE FULL self.historical_data
            self.update_header_info()
            self._update_fx_rate_display(self.currency_combo.currentText())

        except Exception as ui_update_e:
            logging.info(
                f"--- CRITICAL ERROR during UI update in handle_results: {ui_update_e} ---"
            )
            traceback.print_exc()
            QMessageBox.critical(
                self,
                "UI Update Error",
                f"Failed to update display after calculation:\n{ui_update_e}",
            )

        logging.debug("DEBUG: Exiting handle_results.")

    @Slot(str)
    def handle_error(self, error_message):
        """
        Slot to handle error messages received from the PortfolioCalculatorWorker.

        Logs the error and updates the status bar.

        Args:
            error_message (str): The error description string.
        """
        self.is_calculating = False
        logging.info(
            f"--- Calculation/Fetch Error Reported ---\n{error_message}\n--- End Error Report ---"
        )
        self.status_label.setText(
            f"Error: {error_message.split('|||')[0]}"
        )  # Show primary error
        self.calculation_finished()  # Call finished even on error

    def show_status_popup(self, status_message):
        """(Currently unused) Placeholder for potentially showing status popups."""
        # (Keep implementation as before)
        if not status_message:
            return
        cleaned_status = status_message.split("|||TWR_FACTOR:")[
            0
        ].strip()  # Use cleaned status
        if "Error" in cleaned_status or "Critical" in cleaned_status:
            pass
        elif "Success" in cleaned_status:
            pass
        elif "Warning" in cleaned_status or "ignored" in cleaned_status:
            logging.info(f"Status indicates warning/info: '{cleaned_status}'.")

    @Slot()
    def calculation_finished(self):
        """
        Slot called when the worker thread finishes (success or error).

        Re-enables UI controls and updates the status bar with the final status message.
        """
        self.is_calculating = False
        logging.info("Worker thread finished.")
        self.set_controls_enabled(True)
        # Update status label based on the final combined status
        if self.last_calc_status:
            cleaned_status = self.last_calc_status.split("|||TWR_FACTOR:")[0].strip()
            if "Error" in cleaned_status or "Crit" in cleaned_status:
                self.status_label.setText(f"Finished with Errors: {cleaned_status}")
            elif "Warn" in cleaned_status:
                self.status_label.setText(f"Finished with Warnings: {cleaned_status}")
            else:
                now_str = QDateTime.currentDateTime().toString("hh:mm:ss")
                self.status_label.setText(f"Finished ({now_str}): {cleaned_status}")
        else:
            self.status_label.setText("Finished (Status Unknown)")

    # --- Transaction Dialog Methods ---
    @Slot()
    def open_add_transaction_dialog(self):
        """Opens the AddTransactionDialog for manual transaction entry."""
        # Use self.available_accounts if populated, otherwise fallback
        if not self.transactions_file or not os.path.exists(self.transactions_file):
            QMessageBox.warning(
                self,
                "File Not Set",
                "Please select a valid transactions CSV file first.",
            )
            return
        accounts = (
            self.available_accounts
            if self.available_accounts
            else list(self.config.get("account_currency_map", {"SET": "THB"}).keys())
        )

        dialog = AddTransactionDialog(existing_accounts=accounts, parent=self)
        if dialog.exec():
            new_data = dialog.get_transaction_data()
            if new_data:
                self.save_new_transaction(new_data)

    def save_new_transaction(self, transaction_data: Dict[str, str]):
        """
        Appends a new transaction row to the selected CSV file.

        Handles writing the header if the file is new/empty. Shows success or
        error messages and triggers a data refresh upon successful save.

        Args:
            transaction_data (Dict[str, str]): A dictionary containing the validated
                transaction data, keyed by CSV header names.
        """
        # ... (logic remains the same) ...
        logging.info(f"Attempting to save new transaction to: {self.transactions_file}")
        if not self.transactions_file or not os.path.exists(self.transactions_file):
            QMessageBox.critical(
                self,
                "Save Error",
                f"Cannot save transaction. CSV file not found:\n{self.transactions_file}",
            )
            return
        # --- FIX: Define headers WITHOUT the initial blank column ---
        csv_headers = [
            "Date (MMM DD, YYYY)",
            "Transaction Type",
            "Stock / ETF Symbol",
            "Quantity of Units",
            "Amount per unit",
            "Total Amount",
            "Fees",
            "Investment Account",
            "Split Ratio (new shares per old share)",
            "Note",
        ]
        # --- END FIX ---
        new_row = [transaction_data.get(h, "") for h in csv_headers]
        try:
            # Check if file is empty to write header
            file_exists = os.path.exists(self.transactions_file)
            is_empty = not file_exists or os.path.getsize(self.transactions_file) == 0
            # Open in append mode ('a'), it creates if not exists
            with open(self.transactions_file, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                # Write header only if file was empty before opening
                if is_empty:
                    writer.writerow(csv_headers)
                writer.writerow(new_row)
            logging.info("Transaction successfully appended to CSV.")
            QMessageBox.information(
                self, "Success", "Transaction added successfully.\nRefreshing data..."
            )
            self.refresh_data()
        except IOError as e:
            logging.error(f"ERROR writing to CSV file: {e}")
            QMessageBox.critical(
                self, "Save Error", f"Failed to write transaction to CSV file:\n{e}"
            )
        except Exception as e:
            logging.error(f"ERROR saving transaction: {e}")
            traceback.print_exc()
            QMessageBox.critical(
                self, "Save Error", f"An unexpected error occurred while saving:\n{e}"
            )

    # --- Event Handlers ---
    def closeEvent(self, event):
        """
        Handles the main window close event.

        Saves the current configuration before closing. Ensures background threads
        are properly terminated.

        Args:
            event (QCloseEvent): The close event object.
        """
        # (Keep implementation as before)
        logging.info("Close event triggered. Saving config...")
        try:
            self.save_config()
        except Exception as e:
            QMessageBox.warning(
                self, "Save Error", f"Could not save settings on exit:\n{e}"
            )
        logging.info("Exiting application.")
        self.threadpool.clear()
        if not self.threadpool.waitForDone(2000):
            logging.warning("Warning: Worker threads did not finish closing.")
        event.accept()


# --- Run Application Entry Point ---
# (No function here, but the `if __name__ == "__main__":` block executes)
# This block handles:
# - Checking for required library dependencies.
# - Setting up basic logging.
# - Initializing the QApplication.
# - Creating and showing the main PortfolioApp window.
# - Starting the Qt event loop.
if __name__ == "__main__":
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

    main_window = PortfolioApp()

    # Set initial size and minimum size
    main_window.resize(1600, 900)  # Adjust initial size as needed
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
