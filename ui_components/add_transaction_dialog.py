import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QFormLayout,
    QDateEdit,
    QComboBox,
    QLineEdit,
    QDialogButtonBox,
    QLabel,
    QMessageBox,
    QCompleter
)
from PySide6.QtGui import QFont, QDoubleValidator, QValidator
from PySide6.QtCore import Qt, Slot, QDate, QStringListModel
from typing import List, Optional, Dict, Any
from datetime import date, datetime # Added datetime
import logging

# Assuming config.py is in the path.
try:
    from config import CASH_SYMBOL_CSV, CSV_DATE_FORMAT
except ImportError:
    logging.error("AddTransactionDialog: Could not import constants from config. Using fallbacks.")
    CASH_SYMBOL_CSV = "$CASH" # Ensure this matches the one in config.py
    CSV_DATE_FORMAT = "%b %d, %Y"


class AddTransactionDialog(QDialog):
    """Dialog window for manually adding or editing a transaction entry."""

    def __init__(
        self,
        existing_accounts: List[str],
        parent=None,
        portfolio_symbols: Optional[List[str]] = None,
        edit_data: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(
            "Add New Transaction" if not edit_data else "Edit Transaction"
        )
        self.setMinimumWidth(350)

        dialog_font = QFont("Arial", 10)
        if parent and hasattr(parent, 'font'): # Inherit font if possible
            dialog_font = parent.font()
        self.setFont(dialog_font)

        self.transaction_types = [
            "Buy", "Sell", "Dividend", "Split", "Deposit", "Withdrawal",
            "Fees", "Short Sell", "Buy to Cover",
        ]
        self.total_amount_locked_by_user = False
        self.cash_symbol_csv = CASH_SYMBOL_CSV # Store for use in _update_field_states

        main_dialog_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(5, 5, 5, 5)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        input_min_width = 180

        self.date_edit = QDateEdit(date.today())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.date_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Date:", self.date_edit)

        self.type_combo = QComboBox()
        self.type_combo.addItems(self.transaction_types)
        self.type_combo.setMinimumWidth(input_min_width)
        form_layout.addRow("Type:", self.type_combo)

        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText("e.g., AAPL, GOOG, $CASH")
        self.symbol_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Symbol:", self.symbol_edit)

        if portfolio_symbols:
            self.symbol_completer_model = QStringListModel(self)
            self.symbol_completer_model.setStringList(sorted(list(set(portfolio_symbols))))
            self.symbol_completer = QCompleter(self.symbol_completer_model, self)
            self.symbol_completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.symbol_completer.setFilterMode(Qt.MatchContains)
            self.symbol_edit.setCompleter(self.symbol_completer)

        self.account_combo = QComboBox()
        self.account_combo.addItems(sorted(list(set(existing_accounts))))
        self.account_combo.setEditable(True)
        self.account_combo.setInsertPolicy(QComboBox.NoInsert)
        self.account_combo.setMinimumWidth(input_min_width)
        form_layout.addRow("Account:", self.account_combo)

        self.quantity_edit = QLineEdit()
        self.quantity_edit.setPlaceholderText("e.g., 100.5")
        self.quantity_edit.setMinimumWidth(input_min_width)
        self.quantity_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.quantity_validator.setNotation(QDoubleValidator.StandardNotation)
        self.quantity_edit.setValidator(self.quantity_validator)
        form_layout.addRow("Quantity:", self.quantity_edit)

        self.price_edit = QLineEdit()
        self.price_edit.setPlaceholderText("Per unit")
        self.price_edit.setMinimumWidth(input_min_width)
        self.price_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.price_validator.setNotation(QDoubleValidator.StandardNotation)
        self.price_edit.setValidator(self.price_validator)
        form_layout.addRow("Price/Unit:", self.price_edit)

        self.total_amount_edit = QLineEdit()
        self.total_amount_edit.setPlaceholderText("Auto for Buy/Sell/Short or manual")
        self.total_amount_edit.setMinimumWidth(input_min_width)
        self.total_validator = QDoubleValidator(0.0, 1e12, 2, self)
        self.total_validator.setNotation(QDoubleValidator.StandardNotation)
        self.total_amount_edit.setValidator(self.total_validator)
        form_layout.addRow("Total Amount:", self.total_amount_edit)

        self.commission_edit = QLineEdit()
        self.commission_edit.setPlaceholderText("e.g., 6.95 (optional, default 0)")
        self.commission_edit.setMinimumWidth(input_min_width)
        self.commission_validator = QDoubleValidator(0.0, 1e12, 2, self)
        self.commission_validator.setNotation(QDoubleValidator.StandardNotation)
        self.commission_edit.setValidator(self.commission_validator)
        form_layout.addRow("Commission:", self.commission_edit)

        self.split_ratio_label = QLabel("Split Ratio:")
        self.split_ratio_edit = QLineEdit()
        self.split_ratio_edit.setPlaceholderText("New shares per old (e.g., 2 for 2:1)")
        self.split_ratio_edit.setMinimumWidth(input_min_width)
        self.split_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.split_validator.setNotation(QDoubleValidator.StandardNotation)
        self.split_ratio_edit.setValidator(self.split_validator)
        form_layout.addRow(self.split_ratio_label, self.split_ratio_edit)

        self.note_edit = QLineEdit()
        self.note_edit.setObjectName("NoteEdit")
        self.note_edit.setPlaceholderText("Optional note")
        self.note_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Note:", self.note_edit)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_dialog_layout.addLayout(form_layout)
        main_dialog_layout.addWidget(self.button_box)

        self.type_combo.currentTextChanged.connect(self._update_field_states_wrapper)
        self.symbol_edit.textChanged.connect(self._update_field_states_wrapper_symbol)
        self.quantity_edit.textChanged.connect(self._validate_numeric_input)
        self.price_edit.textChanged.connect(self._validate_numeric_input)
        self.total_amount_edit.textChanged.connect(self._validate_numeric_input)
        self.commission_edit.textChanged.connect(self._validate_numeric_input)
        self.split_ratio_edit.textChanged.connect(self._validate_numeric_input)
        self.quantity_edit.textChanged.connect(self._auto_calculate_total)
        self.price_edit.textChanged.connect(self._auto_calculate_total)
        self.total_amount_edit.textEdited.connect(lambda text: setattr(self, "total_amount_locked_by_user", bool(text)))

        if edit_data:
            self._populate_fields_for_edit(edit_data)
        else:
            self._update_field_states(self.type_combo.currentText(), self.symbol_edit.text())

    @Slot()
    def _validate_numeric_input(self):
        sender_widget = self.sender()
        if not isinstance(sender_widget, QLineEdit): return
        validator = sender_widget.validator()
        if not validator: return
        state = validator.validate(sender_widget.text(), 0)[0]
        sender_widget.setStyleSheet("background-color: #ffe0e0;" if state != QValidator.Acceptable else "")

    @Slot(str)
    def _update_field_states_wrapper(self, tx_type: str):
        self._update_field_states(tx_type, self.symbol_edit.text())

    @Slot(str)
    def _update_field_states_wrapper_symbol(self, symbol: str):
        self._update_field_states(self.type_combo.currentText(), symbol)

    def _update_field_states(self, tx_type: str = None, symbol: str = None):
        tx_type_lower = (tx_type or self.type_combo.currentText()).lower()
        symbol_upper = (symbol or self.symbol_edit.text()).upper().strip()
        is_cash_symbol = symbol_upper == self.cash_symbol_csv

        qty_enabled, price_enabled, total_enabled, commission_enabled, split_enabled = False, False, False, True, False
        price_readonly, total_readonly = False, False
        price_text_override = None

        if is_cash_symbol:
            if tx_type_lower in ["deposit", "withdrawal", "buy", "sell"]:
                qty_enabled, price_text_override, price_readonly, price_enabled, total_enabled, total_readonly = True, "1.00", True, False, True, True
            elif tx_type_lower == "dividend": total_enabled = True
        elif tx_type_lower in ["buy", "sell", "short sell", "buy to cover"]:
            qty_enabled, price_enabled, total_enabled, total_readonly = True, True, True, True
        elif tx_type_lower == "dividend":
            qty_enabled, price_enabled, total_enabled = True, True, True
        elif tx_type_lower in ["split", "stock split"]:
            split_enabled = True

        self.quantity_edit.setEnabled(qty_enabled)
        self.price_edit.setEnabled(price_enabled)
        self.price_edit.setReadOnly(price_readonly)
        if price_text_override is not None: self.price_edit.setText(price_text_override)

        self.total_amount_edit.setEnabled(total_enabled)
        self.total_amount_edit.setReadOnly(total_readonly)
        self.commission_edit.setEnabled(commission_enabled)
        self.split_ratio_edit.setEnabled(split_enabled)
        self.split_ratio_label.setEnabled(split_enabled)

        for widget in [self.quantity_edit, self.price_edit, self.total_amount_edit, self.commission_edit, self.split_ratio_edit]:
            if not widget.isEnabled() and (widget != self.price_edit or not self.price_edit.isReadOnly()):
                 widget.clear()

        if total_readonly and not self.total_amount_locked_by_user: self._auto_calculate_total()
        elif not total_readonly: self.total_amount_locked_by_user = False

    @Slot()
    def _auto_calculate_total(self):
        if not self.total_amount_edit.isReadOnly() and self.total_amount_locked_by_user: return
        if not self.quantity_edit.isEnabled(): return

        current_symbol = self.symbol_edit.text().strip().upper()
        is_cash_tx = current_symbol == self.cash_symbol_csv

        if not is_cash_tx and not self.price_edit.isEnabled():
            if self.total_amount_edit.isReadOnly(): self.total_amount_edit.clear()
            return

        qty_str = self.quantity_edit.text().strip().replace(",", "")
        price_str = "1.00" if is_cash_tx else (self.price_edit.text().strip().replace(",", "") if self.price_edit.isEnabled() else "")

        try:
            if qty_str and price_str:
                qty, price = float(qty_str), float(price_str)
                if qty >= 0 and price >= 0:
                    total = qty * price
                    # Assuming parent is PortfolioApp and has config for decimal_places
                    decimal_places = getattr(self.parent(), "config", {}).get("decimal_places", 2) if self.parent() else 2
                    self.total_amount_edit.setText(f"{total:.{decimal_places}f}")
                elif self.total_amount_edit.isReadOnly(): self.total_amount_edit.clear()
            elif self.total_amount_edit.isReadOnly(): self.total_amount_edit.clear()
        except ValueError:
            if self.total_amount_edit.isReadOnly(): self.total_amount_edit.clear()

    def _populate_fields_for_edit(self, data: Dict[str, Any]):
        # Temporarily disconnect signals
        # ... (signal disconnection logic - can be complex with lambdas, simplified here)

        date_val_str = data.get("Date (MMM DD, YYYY)")
        if date_val_str:
            try:
                parsed_dt = datetime.strptime(str(date_val_str), CSV_DATE_FORMAT)
                self.date_edit.setDate(QDate(parsed_dt.year, parsed_dt.month, parsed_dt.day))
            except Exception as e: logging.error(f"Error parsing date for edit: {e}")

        self.type_combo.setCurrentText(str(data.get("Transaction Type", "")))
        self.symbol_edit.setText(str(data.get("Stock / ETF Symbol", "")))
        self.account_combo.setCurrentText(str(data.get("Investment Account", "")))

        formatter_func = lambda val, prec: str(val if pd.notna(val) else "")
        if self.parent() and hasattr(self.parent(), "_format_for_edit_dialog"):
             formatter_func = self.parent()._format_for_edit_dialog

        self.quantity_edit.setText(formatter_func(data.get("Quantity of Units"), 8))
        self.price_edit.setText(formatter_func(data.get("Amount per unit"), 8))
        self.total_amount_edit.setText(formatter_func(data.get("Total Amount"), 2))
        self.commission_edit.setText(formatter_func(data.get("Fees"), 2))
        self.split_ratio_edit.setText(formatter_func(data.get("Split Ratio (new shares per old share)"), 8))
        self.note_edit.setText(str(data.get("Note", "")))

        if self.total_amount_edit.text():
             tx_type_lower = self.type_combo.currentText().lower()
             is_cash_edit = self.symbol_edit.text().upper().strip() == self.cash_symbol_csv
             if (not is_cash_edit and tx_type_lower in ["buy", "sell", "short sell", "buy to cover", "dividend", "fees"]) or \
                (is_cash_edit and tx_type_lower in ["dividend", "fees"]):
                 self.total_amount_locked_by_user = True

        self._update_field_states(self.type_combo.currentText(), self.symbol_edit.text())
        # Reconnect signals if they were disconnected
        # ... (signal reconnection logic)

    def get_transaction_data(self) -> Optional[Dict[str, Any]]:
        data = {}
        tx_type = self.type_combo.currentText()
        tx_type_lower = tx_type.lower()
        symbol = self.symbol_edit.text().strip().upper()
        account = self.account_combo.currentText().strip()
        date_val = self.date_edit.date().toPython()

        if not symbol: QMessageBox.warning(self, "Input Error", "Symbol cannot be empty."); self.symbol_edit.setFocus(); return None
        if not account: QMessageBox.warning(self, "Input Error", "Account cannot be empty."); self.account_combo.setFocus(); return None

        qty_str, price_str, total_str, comm_str, split_str = (
            self.quantity_edit.text().strip().replace(",", ""),
            self.price_edit.text().strip().replace(",", ""),
            self.total_amount_edit.text().strip().replace(",", ""),
            self.commission_edit.text().strip().replace(",", ""),
            self.split_ratio_edit.text().strip().replace(",", "")
        )
        note_str = self.note_edit.text().strip()

        qty, price, total, comm, split = None, None, None, 0.0, None

        if comm_str:
            try: comm = float(comm_str); assert comm >= 0
            except: QMessageBox.warning(self, "Input Error", "Commission invalid."); self.commission_edit.setFocus(); return None

        is_stock_trade = tx_type_lower in ["buy", "sell", "short sell", "buy to cover"] and symbol != self.cash_symbol_csv
        is_cash_op = symbol == self.cash_symbol_csv and tx_type_lower in ["deposit", "withdrawal", "buy", "sell"]

        if is_stock_trade:
            if not qty_str or not price_str: QMessageBox.warning(self, "Input Error", "Quantity and Price required."); return None
            try: qty = float(qty_str); price = float(price_str); assert qty > 1e-8 and price > 1e-8
            except: QMessageBox.warning(self, "Input Error", "Quantity/Price invalid."); return None
            total = qty * price if not (self.total_amount_locked_by_user and total_str) else float(total_str)
        elif is_cash_op:
            if not qty_str: QMessageBox.warning(self, "Input Error", "Amount (Quantity) required."); return None
            try: qty = float(qty_str); assert qty > 1e-8
            except: QMessageBox.warning(self, "Input Error", "Amount (Quantity) invalid."); return None
            price, total = 1.0, qty
        elif tx_type_lower == "dividend":
            if total_str:
                try: total = float(total_str); assert total >=0
                except: QMessageBox.warning(self, "Input Error", "Dividend Total Amount invalid."); return None
                if qty_str:
                    try: qty = float(qty_str); assert qty >=0
                    except: QMessageBox.warning(self, "Input Error", "Dividend Quantity invalid."); return None
                if price_str:
                    try: price = float(price_str); assert price >=0
                    except: QMessageBox.warning(self, "Input Error", "Dividend Price invalid."); return None
            elif qty_str and price_str:
                try: qty = float(qty_str); price = float(price_str); assert qty > 0 and price > 0; total = qty * price
                except: QMessageBox.warning(self, "Input Error", "Dividend Qty/Price invalid for calculating total."); return None
            else: QMessageBox.warning(self, "Input Error", "Dividend: Total or Qty & Price required."); return None
            if symbol == self.cash_symbol_csv and total is not None:
                if qty is None: qty = total
                if price is None: price = 1.0
        elif tx_type_lower in ["split", "stock split"]:
            if not split_str: QMessageBox.warning(self, "Input Error", "Split Ratio required."); return None
            try: split = float(split_str); assert split > 1e-8
            except: QMessageBox.warning(self, "Input Error", "Split Ratio invalid."); return None
            qty, price, total = None, None, None
        elif tx_type_lower == "fees":
            if not comm_str: QMessageBox.warning(self, "Input Error", "Fee amount (Commission) required."); return None
            qty, price, total, split = None, None, None, None
        else: QMessageBox.warning(self, "Input Error", f"Type '{tx_type}' not handled."); return None

        data = {
            "Date (MMM DD, YYYY)": date_val, "Transaction Type": tx_type, "Stock / ETF Symbol": symbol,
            "Quantity of Units": qty, "Amount per unit": price, "Total Amount": total,
            "Fees": comm if comm_str else None, "Investment Account": account,
            "Split Ratio (new shares per old share)": split, "Note": note_str if note_str else None,
        }
        return data

    def accept(self):
        invalid_fields = []
        fields_to_check = [
            (self.quantity_edit, "Quantity"), (self.price_edit, "Price/Unit"),
            (self.total_amount_edit, "Total Amount"), (self.commission_edit, "Commission"),
            (self.split_ratio_edit, "Split Ratio"),
        ]
        for widget, name in fields_to_check:
            if widget.isEnabled() and widget.validator() and widget.validator().validate(widget.text(),0)[0] != QValidator.Acceptable:
                 # More nuanced check for optional fields might be needed here if they can be empty but still "valid" for some types
                if not (name == "Commission" and not widget.text().strip()): # Commission can be empty
                     invalid_fields.append(name); widget.setStyleSheet("background-color: #ffe0e0;")

        if invalid_fields:
            QMessageBox.warning(self, "Invalid Input", f"Correct invalid fields: {', '.join(invalid_fields)}")
            return

        if self.get_transaction_data(): # This performs type-specific validation
            super().accept()
