# -*- coding: utf-8 -*-
"""
PandasModel for QTableView.
"""
import pandas as pd
import numpy as np
import logging
import re
from typing import Any, Optional, Dict, List, Tuple # Added List, Tuple

from PySide6.QtCore import QAbstractTableModel, Qt, Slot
from PySide6.QtGui import QColor

# It's better if config is imported where used, or values passed in.
# For now, to keep it simple, let's import specific items needed.
import config
from config import CSV_DATE_FORMAT # For sort key_func

# These are tricky. PandasModel ideally shouldn't directly import from portfolio_logic.
# It's better if these are passed in or the logic using them is handled by the parent.
# For now, to get it working, we'll import. This is a refactoring point.
try:
    from portfolio_logic import CASH_SYMBOL_CSV, _AGGREGATE_CASH_ACCOUNT_NAME_
except ImportError:
    logging.warning("PandasModel: Could not import CASH_SYMBOL_CSV or _AGGREGATE_CASH_ACCOUNT_NAME_ from portfolio_logic. Using fallbacks.")
    CASH_SYMBOL_CSV = "__CASH__"
    _AGGREGATE_CASH_ACCOUNT_NAME_ = "CASH (Aggregated)"


class PandasModel(QAbstractTableModel):
    """A Qt Table Model for displaying pandas DataFrames in a QTableView.

    Handles data access, header information, sorting (with special handling for
    cash rows), and custom cell formatting (alignment, color, currency/percentage).
    """

    def __init__(
        self, data=pd.DataFrame(), parent=None, log_mode=False
    ):  # ADDED log_mode
        """
        Initializes the model with optional data and a parent reference.

        Args:
            data (pd.DataFrame, optional): The initial DataFrame to display.
                                           Defaults to an empty DataFrame.
            parent (QWidget, optional): The parent widget, typically the main
                                        application window, used to access shared
                                        information like the current currency symbol.
                                        Defaults to None.
            log_mode (bool, optional): If True, disables special cash row handling during sort.
                                       Defaults to False.
        """
        super().__init__(parent)
        self._data = data
        self._parent = parent
        self._log_mode = log_mode  # STORE log_mode
        # In PandasModel, access themed colors via the parent (PortfolioApp instance)
        if (
            parent
            and hasattr(parent, "QCOLOR_TEXT_PRIMARY_THEMED")
            and hasattr(parent, "QCOLOR_GAIN_THEMED")
            and hasattr(parent, "QCOLOR_LOSS_THEMED")
        ):
            self._default_text_color = parent.QCOLOR_TEXT_PRIMARY_THEMED
            self._gain_color = parent.QCOLOR_GAIN_THEMED
            self._loss_color = parent.QCOLOR_LOSS_THEMED
            logging.debug("PandasModel initialized with themed colors from parent.")
        else:  # Fallback if parent or themed colors are not available
            self._default_text_color = QColor(
                config.COLOR_TEXT_DARK
            )  # Fallback to default config
            self._gain_color = QColor(config.COLOR_GAIN)
            self._loss_color = QColor(config.COLOR_LOSS)
            logging.warning("PandasModel initialized with fallback colors.")
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
                # Ensure col_name is treated as a string for subsequent checks
                col_name_str = str(col_name)
                col_data = self._data.iloc[:, col]
                if col_name_str in ["Account", "Symbol", "Price Source"]:
                    alignment = int(Qt.AlignLeft | Qt.AlignVCenter)
                elif pd.api.types.is_numeric_dtype(
                    col_data.dtype
                ) and not pd.api.types.is_datetime64_any_dtype(
                    col_data.dtype
                ):  # Exclude datetime
                    alignment = int(Qt.AlignRight | Qt.AlignVCenter)
                elif col_data.dtype == "object":
                    is_potentially_numeric_by_name = (
                        any(
                            indicator in col_name_str
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
                                "Yield (Cost) %",  # New
                                "Yield (Mkt) %",  # New
                                "Est. Income",  # New
                            ]
                        )
                        or f"({self._parent.currency_combo.currentText()})" in col_name if self._parent and hasattr(self._parent, 'currency_combo') else False
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
                col_name_orig_type_fg = self._data.columns[col]  # Original type
                col_name_fg = str(
                    col_name_orig_type_fg
                )  # String version for comparisons
                raw_cell_value = self._data.iloc[
                    row, col
                ]  # Get raw value for numeric check
                # Specific handling for Gain/Loss columns in Capital Gains table (and potentially others if named similarly)
                if "Gain/Loss" in col_name_fg:  # For Capital Gains table
                    value_str = str(raw_cell_value)
                    cleaned_value_str = value_str.replace(",", "")  # Remove commas
                    # Try to extract numeric part for coloring
                    match = re.search(r"([+-]?\d*\.?\d+)", cleaned_value_str)
                    if match:
                        numeric_part_str = match.group(1)
                        try:
                            numeric_value = float(numeric_part_str)
                            if numeric_value > 1e-9:
                                return self._gain_color
                            if numeric_value < -1e-9:
                                return self._loss_color
                        except ValueError:
                            pass  # Fall through to default (None)
                    return (
                        None  # Default to QSS if not clearly gain/loss or parse error
                    )

                # Existing coloring logic for other tables/columns (where value is numeric, not pre-formatted string)
                if pd.api.types.is_number(raw_cell_value) and pd.notna(raw_cell_value):
                    value_float = float(raw_cell_value)

                    # General Gain/Loss indicators
                    gain_loss_color_cols = [
                        "G/L",  # Catches "Unreal. G/L", "Real. G/L", "Total G/L", "FX G/L", and their % versions
                        "Ret %",  # Catches "Total Ret %"
                        "IRR",
                        "Day Chg",  # Catches "Day Chg" and "Day Chg %"
                        "Yield",  # Catches "Yield (Cost) %" and "Yield (Mkt) %"
                        "Income",  # Catches "Est. Income"
                    ]
                    if any(
                        indicator in col_name_fg for indicator in gain_loss_color_cols
                    ):
                        if (
                            "Yield" in col_name_fg  # Check against the UI name
                        ):  # Yields are typically shown as positive, color green if > 0
                            if value_float > 1e-9:
                                return self._gain_color
                        else:  # For other G/L type columns
                            if value_float > 1e-9:
                                return self._gain_color
                            if value_float < -1e-9:
                                return self._loss_color
                        return (
                            None  # Zero value or non-applicable yield, let QSS handle
                        )

                    # Dividends
                    elif "Dividend" in col_name_fg or "Divs" in col_name_fg:
                        if value_float > 1e-9:
                            return self._gain_color  # Positive dividends are green
                        return None  # Zero or negative dividend, let QSS handle

                    # Commissions/Fees
                    elif (
                        "Commission" in col_name_fg
                        or "Fee" in col_name_fg
                        or "Fees" in col_name_fg
                    ):
                        if value_float > 1e-9:
                            return (
                                self._loss_color
                            )  # Fees are costs, shown red if positive
                        return None  # Zero fee, let QSS handle

                # If not a special column or not numeric, or numeric but zero in a special column
                return None  # Let QSS handle color

            except Exception as e:
                # logging.info(f"Coloring Error (Row:{row}, Col:{col}, Name:'{self._data.columns[col]}'): {e}")
                return None  # Fallback to QSS on any error

        # --- Display Text ---
        if role == Qt.DisplayRole or role == Qt.EditRole:
            original_value = "ERR"  # Default in case of early error
            try:
                original_value = self._data.iloc[row, col]
                col_name_orig_type = self._data.columns[
                    col
                ]  # Original type (can be Timestamp)
                col_name = str(col_name_orig_type)  # Use string version for comparisons

                # --- Special Formatting for CASH Symbol ---
                # ... (Cash formatting remains the same) ...
                # This part should use 'col_name' (string version) if it compares column names
                try:
                    symbol_col_idx = self._data.columns.get_loc("Symbol")
                    symbol_value = self._data.iloc[row, symbol_col_idx]

                    if col_name == "Symbol" and symbol_value == CASH_SYMBOL_CSV:
                        # Check if this is the aggregated cash row
                        account_val_idx = self._data.columns.get_loc("Account")
                        account_val = self._data.iloc[row, account_val_idx]
                        if account_val == _AGGREGATE_CASH_ACCOUNT_NAME_:
                            return f"Total Cash ({self._parent._get_currency_symbol(get_name=True) if self._parent else 'CUR'})"
                    elif symbol_value == CASH_SYMBOL_CSV and col_name in [
                        "Total Ret %",
                        "IRR (%)",
                        "Yield (Cost) %",
                        "Yield (Mkt) %",
                        f"Est. Income",
                    ]:
                        # For the aggregated cash row, these are typically not applicable or zero
                        return "-"
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
                        return f"{value_float:,.4f}"  # Keep original sign for Quantity, up to 10 decimal places

                    # Combined Percentage and IRR formatting
                    elif (
                        "%" in col_name
                    ):  # Check if it's a percentage column (incl. IRR(%))
                        if np.isinf(value_float):
                            return "Inf %"
                        # Use original signed value for percentages (already scaled if IRR)
                        return f"{value_float:,.2f}%"  # Add % sign

                    # --- ADDED: FX Gain/Loss Formatting ---
                    elif col_name == "FX G/L":
                        current_currency_symbol = self._get_currency_symbol_safe()
                        return f"{current_currency_symbol}{value_float:,.2f}"
                    elif col_name == "FX G/L %":
                        return f"{value_float:,.2f}%"
                    # --- END ADDED ---

                    # --- MODIFIED Currency Check & Formatting ---
                    elif self._parent and hasattr(self._parent, "_get_currency_symbol"):
                        currency_ui_names = (
                            [  # These are UI-facing names that PandasModel receives
                                "Avg Cost", "Price", "Cost Basis", "Est. Income",
                                "Mkt Val", "Unreal. G/L", "Real. G/L",
                                "Divs", "Fees", "Total G/L", "Day Chg",
                            ]
                        )
                        is_currency_col = col_name in currency_ui_names
                        current_display_currency_symbol_for_check = (
                            self._get_currency_symbol_safe()
                        )
                        if not is_currency_col and (
                            f"({current_display_currency_symbol_for_check})" in col_name
                            or (
                                col_name.startswith("Total Dividends")
                                and col_name.endswith(
                                    f"({current_display_currency_symbol_for_check})"
                                )
                            )
                        ):
                            is_currency_col = True
                        if is_currency_col:
                            return f"{current_display_currency_symbol_for_check}{value_float:,.2f}"
                    # --- END MODIFIED Currency Check & Formatting ---
                    else:
                        return f"{value_float:,.2f}"
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
        return None

    def _get_currency_symbol_safe(self):
        """
        Safely retrieves the currency symbol from the parent widget.

        Returns:
            str: The currency symbol (e.g., "$") or a default "$" if retrieval fails.
        """
        if self._parent and hasattr(self._parent, "_get_currency_symbol"):
            try:
                return self._parent._get_currency_symbol()
            except Exception:
                pass
        return "$"

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        """
        Returns the header data (column names or row numbers).
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                try:
                    return str(self._data.columns[section])
                except IndexError:
                    return ""
            if orientation == Qt.Vertical:
                return str(section + 1)
        return None

    def updateData(self, data):
        """
        Updates the model with a new DataFrame.
        """
        self.beginResetModel()
        if data is None:
            self._data = pd.DataFrame()
        elif isinstance(data, pd.DataFrame):
            self._data = data.copy()
        else:
            try:
                self._data = pd.DataFrame(data)
            except Exception:
                self._data = pd.DataFrame()
        if self._parent and hasattr(self._parent, "_get_currency_symbol"):
            self.updateCurrencySymbol(self._parent._get_currency_symbol())
        self.endResetModel()

    def sort(self, column, order):
        """
        Sorts the underlying DataFrame based on a column.
        """
        if self._data.empty:
            logging.debug("Sort called on empty model. Skipping.")
            return

        try:
            if column < 0 or column >= self.columnCount():
                logging.warning(
                    f"Warning: Sort called with invalid column index {column}"
                )
                return
            col_name = self._data.columns[column]
            ascending_order = order == Qt.AscendingOrder
            logging.info(
                f"Sorting by column: '{col_name}' (Index: {column}), Ascending: {ascending_order}"
            )
            self.layoutAboutToBeChanged.emit()
            symbol_col_name_internal = "Symbol"
            symbol_col_name_original = "Stock / ETF Symbol"
            actual_symbol_col = None
            if symbol_col_name_internal in self._data.columns:
                actual_symbol_col = symbol_col_name_internal
            elif symbol_col_name_original in self._data.columns:
                actual_symbol_col = symbol_col_name_original
            col_data_for_check = self._data[col_name]
            is_potentially_numeric_by_name = any(
                indicator in col_name
                for indicator in [
                    "%", " G/L", " Price", " Cost", " Val", " Divs", " Fees",
                    " Basis", " Avg", " Chg", "Quantity", "IRR", " Mkt", " Ret %", " Ratio",
                ]
            ) or (
                self._parent
                and hasattr(self._parent, "_get_currency_symbol")
                and callable(self._parent._get_currency_symbol)
                and f"({self._parent._get_currency_symbol(get_name=True)})" in col_name
            )
            original_numeric_headers = [
                "Quantity of Units", "Amount per unit", "Total Amount", "Fees",
                "Split Ratio (new shares per old share)",
            ]
            if col_name in original_numeric_headers:
                is_potentially_numeric_by_name = True
            is_numeric_dtype = pd.api.types.is_numeric_dtype(col_data_for_check.dtype)
            is_potentially_numeric = is_potentially_numeric_by_name or is_numeric_dtype
            string_col_names = [
                "Account", "Symbol", "Investment Account", "Stock / ETF Symbol",
                "Transaction Type", "Price Source", "Note", "Local Currency", "Reason Ignored",
            ]
            is_string_col = col_name in string_col_names
            date_col_name_internal = "Date"
            date_col_name_original = "Date (MMM DD, YYYY)"
            is_date_col = col_name in [date_col_name_internal, date_col_name_original]

            def numeric_key_func(x_series):
                if not pd.api.types.is_string_dtype(x_series):
                    x_series = x_series.astype(str)
                cleaned_series = x_series.str.replace(
                    r"[$,฿€£¥]", "", regex=True
                ).str.replace(",", "", regex=False)
                return pd.to_numeric(cleaned_series, errors="coerce")

            def date_key_func(x_series):
                date_format_to_try = (
                    CSV_DATE_FORMAT if col_name == date_col_name_original else None
                )
                return pd.to_datetime(
                    x_series, errors="coerce", format=date_format_to_try
                )

            key_func = None
            if is_date_col:
                key_func = date_key_func
            elif is_potentially_numeric and not is_string_col:
                key_func = numeric_key_func
            else:
                key_func = lambda x: x.astype(str).fillna("")

            if self._log_mode:
                logging.debug(
                    f"DEBUG Sort (Log Mode): Sorting by '{col_name}' using determined key_func."
                )
                try:
                    self._data.sort_values(
                        by=col_name, ascending=ascending_order, na_position="last",
                        key=key_func, kind="mergesort", inplace=True,
                    )
                except Exception as e_log_sort:
                    logging.error(
                        f"Error during log mode sort for '{col_name}': {e_log_sort}"
                    )
                    try:
                        self._data.sort_values(
                            by=col_name, ascending=ascending_order, na_position="last",
                            key=lambda x: x.astype(str).fillna(""), kind="mergesort", inplace=True,
                        )
                    except Exception as e_fallback_sort:
                        logging.error(
                            f"Fallback log mode sort also failed for '{col_name}': {e_fallback_sort}"
                        )
            else:
                cash_rows = pd.DataFrame()
                non_cash_rows = self._data.copy()
                if actual_symbol_col is None:
                    logging.debug(
                        f"DEBUG Sort (Non-Log Mode): Symbol column not found. Current columns: {self._data.columns.tolist()}"
                    )
                if actual_symbol_col:
                    try:
                        base_cash_symbol = CASH_SYMBOL_CSV
                        cash_mask = (
                            self._data[actual_symbol_col].astype(str)
                            == base_cash_symbol
                        )
                        if (
                            actual_symbol_col == symbol_col_name_internal
                            and self._parent
                            and hasattr(self._parent, "_get_currency_symbol")
                            and callable(self._parent._get_currency_symbol)
                        ):
                            display_currency_name = self._parent._get_currency_symbol(
                                get_name=True
                            )
                            cash_display_symbol = f"Cash ({display_currency_name})"
                            cash_mask |= (
                                self._data[actual_symbol_col].astype(str)
                                == cash_display_symbol
                            )
                        if cash_mask.any():
                            cash_rows = self._data[cash_mask].copy()
                            non_cash_rows = self._data[~cash_mask].copy()
                    except Exception as e_cash_sep:
                        logging.warning(
                            f"Warning: Error separating cash rows during sort: {e_cash_sep}"
                        )
                        cash_rows = pd.DataFrame()
                        non_cash_rows = self._data.copy()
                else:
                    logging.warning(
                        f"Warning: Could not find symbol column for cash separation in sort."
                    )
                sorted_non_cash_rows = pd.DataFrame()
                if not non_cash_rows.empty:
                    try:
                        sorted_non_cash_rows = non_cash_rows.sort_values(
                            by=col_name, ascending=ascending_order, na_position="last",
                            key=key_func, kind="mergesort",
                        )
                    except Exception as e_sort:
                        logging.error(
                            f"ERROR during non-log mode sorting for '{col_name}': {e_sort}"
                        )
                        try:
                            sorted_non_cash_rows = non_cash_rows.sort_values(
                                by=col_name, ascending=ascending_order, na_position="last",
                                key=lambda x: x.astype(str).fillna(""), kind="mergesort",
                            )
                        except Exception as e_fallback_sort:
                            logging.error(
                                f"Fallback non-log sort also failed for '{col_name}': {e_fallback_sort}"
                            )
                            sorted_non_cash_rows = non_cash_rows
                self._data = pd.concat(
                    [sorted_non_cash_rows, cash_rows], ignore_index=True
                )
            self.layoutChanged.emit()
            logging.info(f"Sorting finished for '{col_name}'.")
        except Exception as e:
            col_name_str = "OOB"
            try:
                if 0 <= column < len(self._data.columns):
                    col_name_str = self._data.columns[column]
                else:
                    col_name_str = f"Invalid Index {column}"
            except IndexError:
                col_name_str = f"IndexError {column}"
            except AttributeError:
                col_name_str = "Model Data Invalid"
            logging.error(
                f"CRITICAL Error in sort method (Col:{column}, Name:'{col_name_str}'): {e}"
            )
            traceback.print_exc()
            try:
                self.layoutChanged.emit()
            except Exception as e_emit:
                logging.error(
                    f"ERROR emitting layoutChanged after sort error: {e_emit}"
                )
