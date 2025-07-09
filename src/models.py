# Auto-generated from main_gui.py modularization
from PySide6.QtCore import QAbstractTableModel, Qt, QModelIndex
from PySide6.QtGui import QColor
from typing import Any, Optional
import pandas as pd
import numpy as np
import re
import logging
import config
import traceback

from finutils import (
    format_currency_value,
    format_percentage_value,
    format_large_number_display,
    format_integer_with_commas,
    format_float_with_commas,
)

from config import CASH_SYMBOL_CSV, CSV_DATE_FORMAT, _AGGREGATE_CASH_ACCOUNT_NAME_
from datetime import datetime, date


class PandasModel(QAbstractTableModel):
    """A Qt Table Model for displaying pandas DataFrames in a QTableView.

    Handles data access, header information, sorting, and custom cell formatting.
    Special formatting can be applied for financial statement tables.

    Attributes:
        _is_financial_statement_model (bool): True if this model displays financial statement data
                                              requiring values in millions.
        _currency_symbol_override (Optional[str]): Specific currency symbol for this model instance.
    """

    def __init__(
        self, data=pd.DataFrame(), parent=None, log_mode=False, **kwargs
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
            is_financial_statement_model (bool): Flag for financial statement specific formatting.
            currency_symbol_override (Optional[str]): Specific currency symbol for this model.
            log_mode (bool, optional): If True, disables special cash row handling during sort.
                                       Defaults to False.
        """
        super().__init__(parent)
        self._data = data
        self._parent = parent
        self._log_mode = log_mode  # STORE log_mode
        # --- ADDED for financial statement formatting ---
        self._is_financial_statement_model = kwargs.get(
            "is_financial_statement_model", False
        )
        self._currency_symbol_override = kwargs.get("currency_symbol_override", None)
        # --- END ADDED ---
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
                    alignment = int(
                        Qt.AlignLeft | Qt.AlignVCenter
                    )  # Keep left alignment for these
                elif (
                    col_name_str == "Period"
                ):  # For PVC, Dividend Summary, CG Summary tables
                    alignment = int(
                        Qt.AlignCenter | Qt.AlignVCenter
                    )  # Center align "Period" column
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

                    # General Gain/Loss indicators for positive/negative coloring
                    gain_loss_color_cols = [
                        "G/L",  # Catches "Unreal. G/L", "Real. G/L", "Total G/L", "FX G/L", and their % versions
                        "Ret %",  # Catches "Total Ret %"
                        "IRR",
                        "Day Chg",  # Catches "Day Chg" and "Day Chg %"
                        "Value Change",  # Added for Asset Change table's absolute change column
                        "(%)",  # Added for general percentage columns like "Portfolio (%)" in PVC
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
                col_name_orig_type = self._data.columns[col]
                col_name = str(col_name_orig_type)

                # --- ADDED: Specific handling for 'original_index' ---
                if col_name == "original_index":
                    if pd.isna(original_value):
                        return "-"
                    try:
                        # Ensure it's an integer before converting to string for display
                        return str(
                            int(float(original_value))
                        )  # float() handles if it's already float-like string or int
                    except (ValueError, TypeError):
                        return str(original_value)  # Fallback if not convertible to int
                # --- END ADDED ---

                # --- MODIFIED: Financial Statement Formatting ---
                if self._is_financial_statement_model:
                    active_currency_symbol = (
                        self._currency_symbol_override
                        if self._currency_symbol_override
                        else self._get_currency_symbol_safe()
                    )

                    metric_column_name = "Metric"
                    if "Financial Ratio" in self._data.columns:
                        metric_column_name = "Financial Ratio"

                    if col_name != metric_column_name and isinstance(
                        original_value, (int, float, np.number)
                    ):
                        if pd.notna(original_value):
                            metric_name_for_row = ""
                            try:
                                metric_name_for_row = str(self._data.iloc[row, 0])
                            except IndexError:
                                pass

                            ratio_percent_words = [
                                "rate",
                                "ratio",
                                "margin",
                                "yield",
                                "payout",
                                "%",
                            ]
                            integer_count_words = ["shares", "employees"]
                            currency_exception_words = ["eps"]

                            ratio_percent_pattern = re.compile(
                                f"\\b({'|'.join(ratio_percent_words)})\\b",
                                re.IGNORECASE,
                            )
                            integer_count_pattern = re.compile(
                                f"\\b({'|'.join(integer_count_words)})\\b",
                                re.IGNORECASE,
                            )
                            currency_exception_pattern = re.compile(
                                f"\\b({'|'.join(currency_exception_words)})\\b",
                                re.IGNORECASE,
                            )

                            metric_name_lower = metric_name_for_row.lower()

                            # --- START OF THE FIX ---
                            # Use regex search, but add a specific exclusion for "exchange rate" metrics.
                            is_currency_exception = bool(
                                currency_exception_pattern.search(metric_name_lower)
                            )
                            is_ratio_metric = (
                                bool(ratio_percent_pattern.search(metric_name_lower))
                                and not is_currency_exception
                                and "exchange rate"
                                not in metric_name_lower  # <-- ADDED EXCEPTION
                            )
                            is_count_metric = (
                                bool(integer_count_pattern.search(metric_name_lower))
                                and not is_currency_exception
                            )
                            # --- END OF THE FIX ---

                            value_float = float(original_value)

                            if is_ratio_metric:
                                if abs(value_float) <= 1.0:
                                    return f"{(value_float * 100):.2f}%"
                                else:
                                    return f"{value_float:.2f}%"
                            elif is_count_metric:
                                return format_integer_with_commas(int(value_float))
                            else:
                                if abs(value_float) >= 1_000_000.0:
                                    value_in_millions = value_float / 1_000_000.0
                                    return f"{active_currency_symbol}{value_in_millions:,.1f}M"
                                else:
                                    return format_currency_value(
                                        value_float, active_currency_symbol, decimals=2
                                    )
                        else:
                            return "-"

                # --- END MODIFIED ---

                # Original formatting logic continues below if not handled by financial statement formatting
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
                            return f"Total Cash ({self._parent._get_currency_symbol(get_name=True)})"
                        # Fallback for any other cash symbol (should not happen with new logic but safe)
                        # else:
                        #     display_currency_name = self._parent._get_currency_symbol(get_name=True) if self._parent else "CUR"
                        #     return f"Cash ({display_currency_name})"
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

                # --- START OF THE FIX ---
                # ADDED: Explicitly format date/datetime objects to remove the time component.
                # This check comes before the generic numeric/string checks.
                if isinstance(original_value, (datetime, date, pd.Timestamp)):
                    return original_value.strftime("%Y-%m-%d")
                # --- END OF THE FIX ---

                # --- Formatting based on value type and column name ---
                if isinstance(original_value, (int, float, np.number)):
                    value_float = float(original_value)
                    display_value_float = abs(
                        value_float
                    )  # This variable is unused and can be removed
                    if abs(value_float) < 1e-9:
                        display_value_float = (
                            0.0  # This variable is unused and can be removed
                        )

                    if "Quantity" in col_name:
                        return f"{value_float:,.4f}"

                    elif "%" in col_name:
                        if np.isinf(value_float):
                            return "Inf %"
                        return f"{value_float:,.2f}%"

                    elif col_name == "FX G/L":
                        current_currency_symbol = self._get_currency_symbol_safe()
                        return f"{current_currency_symbol}{value_float:,.2f}"
                    elif col_name == "FX G/L %":
                        return f"{value_float:,.2f}%"

                    # --- MODIFIED: Restructure the currency check ---
                    # Check for currency, but don't let it block the fallback
                    if self._parent and hasattr(self._parent, "_get_currency_symbol"):
                        currency_ui_names = [
                            "Avg Cost",
                            "Price",
                            "Cost Basis",
                            "Est. Income",
                            "Mkt Val",
                            "Unreal. G/L",
                            "Real. G/L",
                            "Divs",
                            "Fees",
                            "Total G/L",
                            "Day Chg",
                        ]
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
                    # --- END MODIFICATION ---

                    # --- CORRECTED FALLBACK ---
                    # This is now the guaranteed fallback for any numeric type that wasn't
                    # a Quantity, Percentage, or specific Currency column.
                    return f"{value_float:,.2f}"

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

        try:  # Outer try for the whole sort method
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

            # --- Determine key_func (hoisted to be available for both modes) ---
            symbol_col_name_internal = "Symbol"
            symbol_col_name_original = "Stock / ETF Symbol"
            actual_symbol_col = None
            if symbol_col_name_internal in self._data.columns:
                actual_symbol_col = symbol_col_name_internal
            elif symbol_col_name_original in self._data.columns:
                actual_symbol_col = symbol_col_name_original
            # --- End Symbol column determination ---
            # --- Column Type Heuristics (applied to self._data[col_name]) ---
            col_data_for_check = self._data[col_name]
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
                and f"({self._parent._get_currency_symbol(get_name=True)})" in col_name
            )
            original_numeric_headers = [
                "Quantity of Units",
                "Amount per unit",
                "Total Amount",
                "Fees",
                "Split Ratio (new shares per old share)",
            ]
            if col_name in original_numeric_headers:
                is_potentially_numeric_by_name = True
            is_numeric_dtype = pd.api.types.is_numeric_dtype(col_data_for_check.dtype)
            is_potentially_numeric = is_potentially_numeric_by_name or is_numeric_dtype
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
            date_col_name_internal = "Date"
            date_col_name_original = "Date (MMM DD, YYYY)"
            is_date_col = col_name in [date_col_name_internal, date_col_name_original]

            # Define key functions
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
            else:  # Default to string sort key
                key_func = lambda x: x.astype(str).fillna("")
            # --- End key_func determination ---

            if self._log_mode:
                logging.debug(
                    f"DEBUG Sort (Log Mode): Sorting by '{col_name}' using determined key_func."
                )
                try:
                    self._data.sort_values(
                        by=col_name,
                        ascending=ascending_order,
                        na_position="last",
                        key=key_func,
                        kind="mergesort",
                        inplace=True,
                    )
                except Exception as e_log_sort:
                    logging.error(
                        f"Error during log mode sort for '{col_name}': {e_log_sort}"
                    )
                    # Fallback to simple string sort on error
                    try:
                        self._data.sort_values(
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",
                            key=lambda x: x.astype(str).fillna(""),  # Basic string key
                            kind="mergesort",  # Use stable sort
                            inplace=True,
                        )
                    except Exception as e_fallback_sort:
                        logging.error(
                            f"Fallback log mode sort also failed for '{col_name}': {e_fallback_sort}"
                        )
            else:  # Original logic for main table (non-log_mode)
                cash_rows = pd.DataFrame()
                non_cash_rows = self._data.copy()

                if actual_symbol_col is None:
                    # ADDED DEBUG LOG: Log columns when symbol column is not found
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
                            by=col_name,
                            ascending=ascending_order,
                            na_position="last",
                            key=key_func,  # Use hoisted key_func
                            kind="mergesort",
                        )
                    except Exception as e_sort:
                        logging.error(
                            f"ERROR during non-log mode sorting for '{col_name}': {e_sort}"
                        )
                        try:  # Fallback string sort for non_cash_rows
                            sorted_non_cash_rows = non_cash_rows.sort_values(
                                by=col_name,
                                ascending=ascending_order,
                                na_position="last",
                                key=lambda x: x.astype(str).fillna(""),
                                kind="mergesort",
                            )
                        except Exception as e_fallback_sort:
                            logging.error(
                                f"Fallback non-log sort also failed for '{col_name}': {e_fallback_sort}"
                            )
                            sorted_non_cash_rows = (
                                non_cash_rows  # Keep original order on error
                            )

                self._data = pd.concat(
                    [sorted_non_cash_rows, cash_rows], ignore_index=True
                )

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
