# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          manage_transactions_dialog_db.py
 Purpose:       Dialog for managing transactions in the database.

 Author:        Kit Matan and Google Gemini
 Author Email:  kittiwit@gmail.com

 Created:       [Date of creation, e.g., 26/04/2024]
 Copyright:     (c) Kit Matan 2024
 Licence:       MIT
-------------------------------------------------------------------------------
"""
import logging
import pandas as pd
import sqlite3 # For type hinting if db_conn is passed directly
from typing import TYPE_CHECKING, Dict, Any, List, Optional

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTableView,
    QPushButton,
    QDialogButtonBox,
    QHeaderView,
    QAbstractItemView,
    QMessageBox,
    QHBoxLayout,
)
from PySide6.QtCore import Qt, Signal, Slot

if TYPE_CHECKING:
    from main_gui import PortfolioApp, PandasModel # Import for type hinting

# It's better if PandasModel is also moved to a common module,
# but for now, we might need to pass it or expect it from the parent.

class ManageTransactionsDialogDB(QDialog):
    """Dialog for viewing, editing, and deleting transactions from the database."""

    # Signal to indicate that data has changed (transaction edited or deleted)
    data_changed = Signal()

    def __init__(
        self,
        transactions_df: pd.DataFrame,
        db_conn: sqlite3.Connection,
        parent_app: "PortfolioApp",
    ): # Added parent_app
        super().__init__(parent_app) # Use parent_app for parent
        self.setWindowTitle("Manage Transactions (Database)")
        self.transactions_df = transactions_df.copy()
        self.db_conn = db_conn
        self.parent_app = parent_app # Store reference to main app

        # --- Themed Colors (get from parent_app if possible) ---
        # These are fallbacks if the parent_app or its themed colors aren't available.
        # It's assumed parent_app will have these attributes after its theming is applied.
        fallback_text_color = "#000000"
        fallback_bg_color = "#FFFFFF"
        fallback_header_bg_color = "#F0F0F0"
        fallback_border_color = "#C0C0C0"

        self.text_color = getattr(parent_app, 'QCOLOR_TEXT_PRIMARY_THEMED', Qt.black).name() \
            if parent_app else fallback_text_color
        self.background_color = getattr(parent_app, 'QCOLOR_BACKGROUND_THEMED', Qt.white).name() \
            if parent_app else fallback_bg_color
        self.header_background_color = getattr(parent_app, 'QCOLOR_HEADER_BACKGROUND_THEMED', Qt.lightGray).name() \
            if parent_app else fallback_header_bg_color
        self.border_color = getattr(parent_app, 'QCOLOR_BORDER_THEMED', Qt.gray).name() \
            if parent_app else fallback_border_color
        # --- End Themed Colors ---

        # --- Main Layout ---
        layout = QVBoxLayout(self)
        layout.setSpacing(10) # Add some spacing
        self.setMinimumSize(1000, 600) # Set a reasonable minimum size

        # --- Table View ---
        self.table_view = QTableView()
        self.table_view.setObjectName("ManageTransactionsDBTable")
        # IMPORTANT: PandasModel is defined in main_gui.py.
        # For this dialog to use it directly, either PandasModel needs to be in a shared module,
        # or we rely on parent_app to provide it or a way to create it.
        # Assuming parent_app has PandasModel as an accessible class for now.
        if hasattr(parent_app, 'PandasModel'):
            self.table_model = parent_app.PandasModel(self.transactions_df, parent=self, log_mode=True)
        else:
            # Fallback or error if PandasModel is not accessible via parent_app
            logging.error("PandasModel not accessible via parent_app in ManageTransactionsDialogDB.")
            # As a minimal fallback, you might disable editing/sorting or use a simpler model,
            # but the expectation is that PandasModel is available.
            # For now, let's assume it will be available and proceed.
            # If not, this will raise an AttributeError later.
            from main_gui import PandasModel # Temporary direct import if not via parent
            self.table_model = PandasModel(self.transactions_df, parent=self, log_mode=True)


        self.table_view.setModel(self.table_model)
        self.table_view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table_view.setSelectionMode(QAbstractItemView.SingleSelection) # Single selection for edit/delete
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table_view.verticalHeader().setVisible(False)
        layout.addWidget(self.table_view)

        # --- Buttons ---
        button_layout = QHBoxLayout() # Horizontal layout for buttons
        self.edit_button = QPushButton("Edit Selected Transaction")
        self.edit_button.clicked.connect(self.edit_selected_transaction)
        button_layout.addWidget(self.edit_button)

        self.delete_button = QPushButton("Delete Selected Transaction")
        self.delete_button.clicked.connect(self.delete_selected_transaction)
        button_layout.addWidget(self.delete_button)

        button_layout.addStretch() # Push dialog buttons to the right

        self.dialog_buttons = QDialogButtonBox(QDialogButtonBox.Close)
        self.dialog_buttons.rejected.connect(self.reject) # Close on "Close"
        button_layout.addWidget(self.dialog_buttons)
        layout.addLayout(button_layout)

        # Apply initial sort (e.g., by Date descending if Date column exists)
        try:
            date_col_name = "Date" # Assuming this is the actual column name after potential mapping
            if date_col_name in self.transactions_df.columns:
                date_col_idx = self.transactions_df.columns.get_loc(date_col_name)
                self.table_view.sortByColumn(date_col_idx, Qt.DescendingOrder)
        except KeyError:
            logging.warning(f"Column '{date_col_name}' not found for initial sort in ManageTransactionsDialogDB.")
        except Exception as e_sort:
            logging.error(f"Error applying initial sort in ManageTransactionsDialogDB: {e_sort}")

        self.table_view.resizeColumnsToContents()
        self._apply_styles()


    def _apply_styles(self):
        """Applies basic styling, potentially using themed colors."""
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {self.background_color};
                color: {self.text_color};
            }}
            QTableView {{
                gridline-color: {self.border_color};
                background-color: {self.background_color}; /* Ensure table bg matches dialog */
                alternate-background-color: {getattr(self.parent_app, 'QCOLOR_TABLE_ALT_ROW_THEMED', '#F5F5F5').name() if self.parent_app else '#F5F5F5'};
            }}
            QHeaderView::section {{
                background-color: {self.header_background_color};
                color: {self.text_color};
                padding: 4px;
                border: 1px solid {self.border_color};
            }}
            QPushButton {{
                min-width: 100px; /* Give buttons some space */
                padding: 5px;
            }}
        """)
        # Force style update for the table view
        self.table_view.style().unpolish(self.table_view)
        self.table_view.style().polish(self.table_view)
        self.table_view.update()


    @Slot()
    def edit_selected_transaction(self):
        """Handles editing of the selected transaction."""
        selected_rows = self.table_view.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "Please select a transaction to edit.")
            return

        model_index = selected_rows[0] # QModelIndex of the first cell in the selected row
        # The 'original_index' column in self.transactions_df holds the DB transaction_id
        try:
            # Get the DataFrame index from the model index
            # This assumes the underlying model data (_data in PandasModel) has 'original_index'
            df_row_index = model_index.row() # This is the view's row, map to model's data if sorted/filtered

            # If table is sorted, map view row to source row in PandasModel
            source_row_index = self.table_model.mapToSource(model_index).row()
            transaction_id = self.table_model._data.iloc[source_row_index]['original_index']

            # Get all data for this row from the model's underlying DataFrame
            # to pre-fill the edit dialog.
            # self.table_model._data has cleaned column names (DB names).
            # AddTransactionDialog expects CSV-like headers. We need to map.

            transaction_data_for_dialog_db_names = self.table_model._data.iloc[source_row_index].to_dict()

            # Map DB names to CSV-like names for AddTransactionDialog
            # This mapping needs to be comprehensive.
            # Example: Date (YYYY-MM-DD from DB) -> Date (MMM DD, YYYY for Dialog)
            #          Type -> Transaction Type
            #          Symbol -> Stock / ETF Symbol, etc.

            # For simplicity, we'll rely on AddTransactionDialog to be flexible or
            # we'll need a more robust mapping here.
            # Let's assume AddTransactionDialog can take a dict with DB column names for pre-filling,
            # or it has a method to pre-fill from such a dict.
            # This part is complex due to header differences.

            # For now, let's assume the parent_app has a method to show the AddTransactionDialog
            # pre-filled for editing.
            if hasattr(self.parent_app, 'open_add_transaction_dialog'):
                # We need to convert `transaction_data_for_dialog_db_names` (DB col names)
                # to the format AddTransactionDialog expects (CSV col names & Python types)

                from main_gui import AddTransactionDialog # Assuming AddTransactionDialog is accessible

                # Convert DB data to the format AddTransactionDialog.get_transaction_data() would provide
                # This is a simplified conversion, a full one would be more robust.
                data_to_prefill_dialog = {}

                # Date: DB has YYYY-MM-DD, Dialog wants datetime.date
                db_date_str = transaction_data_for_dialog_db_names.get("Date")
                if db_date_str:
                    try:
                        data_to_prefill_dialog["Date (MMM DD, YYYY)"] = datetime.strptime(db_date_str, "%Y-%m-%d").date()
                    except ValueError:
                        data_to_prefill_dialog["Date (MMM DD, YYYY)"] = None # Or handle error

                data_to_prefill_dialog["Transaction Type"] = transaction_data_for_dialog_db_names.get("Type")
                data_to_prefill_dialog["Stock / ETF Symbol"] = transaction_data_for_dialog_db_names.get("Symbol")
                data_to_prefill_dialog["Quantity of Units"] = transaction_data_for_dialog_db_names.get("Quantity")
                data_to_prefill_dialog["Amount per unit"] = transaction_data_for_dialog_db_names.get("Price/Share")
                data_to_prefill_dialog["Total Amount"] = transaction_data_for_dialog_db_names.get("Total Amount")
                data_to_prefill_dialog["Fees"] = transaction_data_for_dialog_db_names.get("Commission")
                data_to_prefill_dialog["Investment Account"] = transaction_data_for_dialog_db_names.get("Account")
                data_to_prefill_dialog["Split Ratio (new shares per old share)"] = transaction_data_for_dialog_db_names.get("Split Ratio")
                data_to_prefill_dialog["Note"] = transaction_data_for_dialog_db_names.get("Note")
                # Local Currency is not directly edited in AddTransactionDialog, it's derived.

                edit_dialog = AddTransactionDialog(
                    existing_accounts=self.parent_app.available_accounts,
                    portfolio_symbols= self.parent_app.all_transactions_df_cleaned_for_logic["Symbol"].unique().tolist() if not self.parent_app.all_transactions_df_cleaned_for_logic.empty else [],
                    parent=self.parent_app,
                    edit_mode_data=data_to_prefill_dialog # Pass data for pre-filling
                )

                if edit_dialog.exec():
                    updated_data_pytypes = edit_dialog.get_transaction_data()
                    if updated_data_pytypes:
                        # Call parent_app's method to handle DB update
                        # This method needs to map CSV-like headers back to DB columns
                        if self.parent_app._edit_transaction_in_db(transaction_id, updated_data_pytypes):
                            self.data_changed.emit() # Signal that data changed
                            self.refresh_table_data() # Refresh this dialog's table
                            QMessageBox.information(self, "Success", "Transaction updated successfully.")
                        else:
                            QMessageBox.critical(self, "Update Failed", "Failed to update transaction in database.")
            else:
                QMessageBox.critical(self, "Error", "Edit functionality not fully connected in parent application.")

        except (IndexError, KeyError) as e:
            logging.error(f"Error accessing transaction data for edit: {e}")
            QMessageBox.critical(self, "Error", "Could not retrieve transaction details for editing.")


    @Slot()
    def delete_selected_transaction(self):
        """Handles deletion of the selected transaction from the database."""
        selected_rows = self.table_view.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "Selection Error", "Please select a transaction to delete.")
            return

        model_index = selected_rows[0]
        try:
            # Map view row to source row in PandasModel
            source_row_index = self.table_model.mapToSource(model_index).row()
            transaction_id_to_delete = self.table_model._data.iloc[source_row_index]['original_index']
            symbol_to_delete = self.table_model._data.iloc[source_row_index].get('Symbol', 'N/A')
            date_to_delete = self.table_model._data.iloc[source_row_index].get('Date', 'N/A')

            reply = QMessageBox.question(self, "Confirm Delete",
                                         f"Are you sure you want to delete the transaction for '{symbol_to_delete}' on '{date_to_delete}' (ID: {transaction_id_to_delete})?\nThis action cannot be undone.",
                                         QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

            if reply == QMessageBox.Yes:
                if self.parent_app._delete_transaction_from_db(transaction_id_to_delete):
                    self.data_changed.emit() # Signal that data changed
                    self.refresh_table_data() # Refresh this dialog's table
                    QMessageBox.information(self, "Success", "Transaction deleted successfully.")
                else:
                    QMessageBox.critical(self, "Delete Failed", "Failed to delete transaction from database.")
        except (IndexError, KeyError) as e:
            logging.error(f"Error accessing transaction ID for deletion: {e}")
            QMessageBox.critical(self, "Error", "Could not retrieve transaction ID for deletion.")

    def refresh_table_data(self):
        """Refreshes the table data from the database."""
        # This method is called after an edit or delete to update this dialog's view.
        logging.debug("ManageTransactionsDialogDB: Refreshing table data from DB...")
        if self.db_conn and hasattr(self.parent_app, 'config'):
            # Use account_currency_map and default_currency from parent_app's config
            acc_map_config_refresh = self.parent_app.config.get('account_currency_map', {})
            def_curr_config_refresh = self.parent_app.config.get('default_currency', 'USD')

            # Assuming load_all_transactions_from_db is globally accessible or imported
            from db_utils import load_all_transactions_from_db
            df_new, success = load_all_transactions_from_db(self.db_conn, acc_map_config_refresh, def_curr_config_refresh)

            if success and df_new is not None:
                self.transactions_df = df_new.copy()
                self.table_model.updateData(self.transactions_df)
                self.table_view.resizeColumnsToContents()
                logging.debug("ManageTransactionsDialogDB: Table data refreshed.")
            else:
                logging.error("ManageTransactionsDialogDB: Failed to reload data from DB for refresh.")
                QMessageBox.warning(self, "Refresh Error", "Could not reload transaction data from the database.")
        else:
            logging.warning("ManageTransactionsDialogDB: DB connection or parent config not available for refresh.")

# --- Helper for date conversion (if needed, though AddTransactionDialog should handle it) ---
from datetime import datetime

def robust_date_parse(date_str: Any) -> Optional[datetime.date]:
    """
    Robustly parses a date string into a datetime.date object.
    Handles common date formats and returns None if parsing fails.
    """
    if isinstance(date_str, datetime):
        return date_str.date()
    if isinstance(date_str, datetime.date):
        return date_str
    if not isinstance(date_str, str):
        return None

    formats_to_try = [
        "%Y-%m-%d",  # Standard DB format
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%b %d, %Y", # e.g., Jan 01, 2023 (CSV_DATE_FORMAT)
        "%Y%m%d"
    ]
    for fmt in formats_to_try:
        try:
            return datetime.strptime(date_str, fmt).date()
        except (ValueError, TypeError):
            continue
    return None
