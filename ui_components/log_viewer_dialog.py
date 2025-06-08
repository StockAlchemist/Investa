import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QLabel,
    QTableView,
    QHeaderView,
    QDialogButtonBox
)
from PySide6.QtCore import Qt

# Assuming PandasModel will be imported from main_gui or its own module later.
# For this step, we expect main_gui.py to still provide PandasModel.
# from main_gui import PandasModel
# (This line is commented out as PandasModel is expected to be passed via parent or exist in main_gui's scope)
import logging


class LogViewerDialog(QDialog):
    """Dialog to display ignored transactions and reasons."""

    def __init__(self, ignored_df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Ignored Transactions Log")
        self.setMinimumSize(800, 400)  # Make it reasonably sized
        self._parent_app = parent # Store parent for PandasModel access

        layout = QVBoxLayout(self)

        if ignored_df is None or ignored_df.empty:
            label = QLabel("No transactions were ignored during the last calculation.")
            label.setAlignment(Qt.AlignCenter)
            layout.addWidget(label)
        else:
            self.table_view = QTableView()
            self.table_view.setObjectName("IgnoredLogTable")

            # Critical dependency: PandasModel must be accessible.
            # This will try to get PandasModel from the parent (PortfolioApp) if it's passed.
            if self._parent_app and hasattr(self._parent_app, 'PandasModel'):
                self.table_model = self._parent_app.PandasModel(
                    ignored_df.copy(), parent=parent, log_mode=True # log_mode for sorting
                )
                self.table_view.setModel(self.table_model)
            else:
                logging.error("LogViewerDialog: PandasModel not found to create table model via parent.")
                # Fallback: show an error message in the dialog
                error_label = QLabel("Error: Table display component (PandasModel) not loaded.")
                layout.addWidget(error_label)
                # Do not add table_view if model creation failed


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

            if self.table_view.model(): # Only add if model was set
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
