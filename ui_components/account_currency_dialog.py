from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QTableWidget,
    QDialogButtonBox,
    QTableWidgetItem,
    QAbstractItemView,
    QHeaderView
)
from PySide6.QtCore import Qt
from typing import Dict, List, Optional, Tuple
import logging

# Assuming config.py is in the path and COMMON_CURRENCIES can be imported.
# If config.py is not directly importable, COMMON_CURRENCIES might need to be passed
# or accessed differently. For this refactoring step, we assume it's available.
try:
    from config import COMMON_CURRENCIES
except ImportError:
    logging.error("AccountCurrencyDialog: Could not import COMMON_CURRENCIES from config. Using fallback.")
    COMMON_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CAD", "AUD", "CHF", "CNY", "THB"]


class AccountCurrencyDialog(QDialog):
    """Dialog to manage account-to-currency mappings and default currency."""

    def __init__(
        self,
        current_map: Dict[str, str],
        current_default: str,
        all_accounts: List[str],
        parent=None,
    ):
        super().__init__(parent)
        self._parent_app = parent  # Store reference if needed
        self.setWindowTitle("Account Currency Settings")
        self.setMinimumSize(450, 400)  # Adjust size as needed

        # Store original values and all unique accounts found
        self._original_map = current_map.copy()
        self._original_default = current_default
        # Ensure all_accounts is a unique list
        self._all_accounts = sorted(list(set(all_accounts)))

        # Attributes to store results on accept
        self.updated_map = self._original_map.copy()
        self.updated_default = self._original_default

        # --- Layout ---
        main_layout = QVBoxLayout(self)

        # --- Default Currency Setting ---
        default_layout = QHBoxLayout()
        default_layout.addWidget(QLabel("Default Currency (for unmapped accounts):"))
        self.default_currency_combo = QComboBox()
        # Ensure default and common currencies are available
        default_currencies = sorted(
            list(set([self._original_default] + COMMON_CURRENCIES))
        )
        self.default_currency_combo.addItems(default_currencies)
        self.default_currency_combo.setCurrentText(self._original_default)
        self.default_currency_combo.setMinimumWidth(100)  # Adjust 100 as needed
        default_layout.addWidget(self.default_currency_combo)
        default_layout.addStretch()
        main_layout.addLayout(default_layout)

        # --- Account Mapping Table ---
        main_layout.addWidget(QLabel("Assign Currency per Account:"))
        self.table_widget = QTableWidget()
        self.table_widget.setObjectName("AccountCurrencyTable")
        self.table_widget.setColumnCount(2)
        self.table_widget.setHorizontalHeaderLabels(["Account", "Assigned Currency"])
        self.table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table_widget.verticalHeader().setVisible(False)
        self.table_widget.setSortingEnabled(True)  # Allow sorting by account name

        # Populate table
        self._populate_table()

        # Resize columns
        self.table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.Stretch
        )  # Account name stretches
        self.table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeToContents
        )  # Currency fits content
        self.table_widget.setMinimumHeight(250)  # Give table some minimum space

        main_layout.addWidget(self.table_widget)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)  # Connect to override
        self.button_box.rejected.connect(self.reject)
        main_layout.addWidget(self.button_box)

        # Apply parent's style and font
        if parent:
            if hasattr(parent, "styleSheet"):
                self.setStyleSheet(parent.styleSheet())
            if hasattr(parent, "font"):
                self.setFont(parent.font())

    def _populate_table(self):
        """Fills the table with accounts and currency combo boxes."""
        self.table_widget.setRowCount(len(self._all_accounts))
        self.table_widget.setSortingEnabled(False)  # Disable sorting during population

        for row_idx, account_name in enumerate(self._all_accounts):
            # Column 0: Account Name (Read-only)
            item_account = QTableWidgetItem(account_name)
            item_account.setFlags(
                item_account.flags() & ~Qt.ItemIsEditable
            )  # Make read-only
            self.table_widget.setItem(row_idx, 0, item_account)

            # Column 1: Currency ComboBox
            combo_currency = QComboBox()
            current_assigned_currency = self._original_map.get(
                account_name, self._original_default
            )

            # Ensure current and default currencies are in the list for this specific combo box
            available_currencies = sorted(
                list(
                    set(
                        [current_assigned_currency, self._original_default]
                        + COMMON_CURRENCIES
                    )
                )
            )
            combo_currency.addItems(available_currencies)

            # Set the initial selection
            combo_currency.setCurrentText(current_assigned_currency)

            # Add the combo box widget to the cell
            self.table_widget.setCellWidget(row_idx, 1, combo_currency)

        self.table_widget.setSortingEnabled(True)  # Re-enable sorting

    def accept(self):
        """Overrides accept to gather data before closing."""
        new_map = {}
        new_default = self.default_currency_combo.currentText()

        for row_idx in range(self.table_widget.rowCount()):
            account_item = self.table_widget.item(row_idx, 0)
            currency_combo = self.table_widget.cellWidget(row_idx, 1)

            if account_item and currency_combo:
                account_name = account_item.text()
                selected_currency = currency_combo.currentText()
                new_map[account_name] = selected_currency
            else:
                logging.warning(
                    f"Could not read data from row {row_idx} in AccountCurrencyDialog."
                )
                # Optionally show an error to the user?

        # Store the results
        self.updated_map = new_map
        self.updated_default = new_default
        logging.info(
            f"AccountCurrencyDialog accepted. New Default: {self.updated_default}, New Map: {self.updated_map}"
        )
        super().accept()  # Call the original accept to close the dialog

    # --- Static method to retrieve results cleanly ---
    @staticmethod
    def get_settings(
        parent=None, current_map=None, current_default=None, all_accounts=None
    ) -> Optional[Tuple[Dict[str, str], str]]:
        """Creates, shows dialog, and returns updated settings if saved."""
        if current_map is None:
            current_map = {}
        if current_default is None:
            current_default = "USD"
        if all_accounts is None:
            all_accounts = list(current_map.keys())

        dialog = AccountCurrencyDialog(
            current_map, current_default, all_accounts, parent
        )
        if dialog.exec():  # Returns 1 if accepted (Save clicked), 0 if rejected
            return dialog.updated_map, dialog.updated_default
        return None  # Return None if Cancel was clicked
