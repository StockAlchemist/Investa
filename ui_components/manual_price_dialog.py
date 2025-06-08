from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
    QAbstractItemView
)
from PySide6.QtGui import QDoubleValidator, QColor, QFont
from PySide6.QtCore import Qt, Slot
from typing import Dict, Any, Optional, Set, List # Added List here
import logging
import pandas as pd


class ManualPriceDialog(QDialog):
    """Dialog to manage manual overrides, symbol mappings, and excluded symbols."""

    def __init__(
        self,
        current_overrides: Dict[str, Dict[str, Any]],
        current_symbol_map: Dict[str, str],
        current_excluded_symbols: Set[str],
        parent=None,
    ):
        super().__init__(parent)
        self._parent_app = parent
        self.setWindowTitle("Symbol Settings")
        self.setMinimumSize(800, 500)

        self._original_overrides = {
            k.upper().strip(): v for k, v in current_overrides.items()
        }
        self._original_symbol_map = {
            k.upper().strip(): v.upper().strip() for k, v in current_symbol_map.items()
        }
        self._original_excluded_symbols = {
            s.upper().strip() for s in current_excluded_symbols
        }

        self.updated_settings = {
            "manual_price_overrides": self._original_overrides.copy(),
            "user_symbol_map": self._original_symbol_map.copy(),
            "user_excluded_symbols": list(
                self._original_excluded_symbols
            ),
        }

        main_layout = QVBoxLayout(self)
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # Tab 1: Manual Overrides
        self.overrides_tab = QWidget()
        overrides_layout = QVBoxLayout(self.overrides_tab)
        overrides_layout.addWidget(QLabel("Edit manual overrides (used as fallback):"))
        self.overrides_table_widget = QTableWidget()
        self.overrides_table_widget.setObjectName("ManualOverridesTable")
        self.overrides_table_widget.setColumnCount(6)
        self.overrides_table_widget.setHorizontalHeaderLabels(
            [
                "Symbol",
                "Manual Price",
                "Manual Asset Type",
                "Manual Sector",
                "Manual Geography",
                "Manual Industry",
            ]
        )
        self.overrides_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.overrides_table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.overrides_table_widget.verticalHeader().setVisible(False)
        self.overrides_table_widget.setSortingEnabled(True)
        self.price_validator = QDoubleValidator(0.00000001, 1000000000.0, 8, self)
        self.price_validator.setNotation(QDoubleValidator.StandardNotation)
        self._populate_overrides_table()
        self.overrides_table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col_idx in range(1, 6):
            self.overrides_table_widget.horizontalHeader().setSectionResizeMode(col_idx, QHeaderView.ResizeToContents)
        overrides_layout.addWidget(self.overrides_table_widget)
        self.tab_widget.addTab(self.overrides_tab, "Manual Overrides")

        # Tab 2: Symbol Mapping
        self.symbol_map_tab = QWidget()
        symbol_map_layout = QVBoxLayout(self.symbol_map_tab)
        symbol_map_layout.addWidget(QLabel("Define custom symbol mappings to Yahoo Finance tickers:"))
        self.symbol_map_table_widget = QTableWidget()
        self.symbol_map_table_widget.setObjectName("SymbolMapTable")
        self.symbol_map_table_widget.setColumnCount(2)
        self.symbol_map_table_widget.setHorizontalHeaderLabels(["Internal Symbol", "Yahoo Finance Ticker"])
        self.symbol_map_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.symbol_map_table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.symbol_map_table_widget.verticalHeader().setVisible(False)
        self.symbol_map_table_widget.setSortingEnabled(True)
        self._populate_symbol_map_table()
        self.symbol_map_table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.symbol_map_table_widget.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        symbol_map_layout.addWidget(self.symbol_map_table_widget)
        self.tab_widget.addTab(self.symbol_map_tab, "Symbol Mapping")

        # Tab 3: Excluded Symbols
        self.excluded_symbols_tab = QWidget()
        excluded_symbols_layout = QVBoxLayout(self.excluded_symbols_tab)
        excluded_symbols_layout.addWidget(QLabel("Define symbols to exclude from Yahoo Finance fetching:"))
        self.excluded_symbols_table_widget = QTableWidget()
        self.excluded_symbols_table_widget.setObjectName("ExcludedSymbolsTable")
        self.excluded_symbols_table_widget.setColumnCount(1)
        self.excluded_symbols_table_widget.setHorizontalHeaderLabels(["Excluded Symbol"])
        self.excluded_symbols_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.excluded_symbols_table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.excluded_symbols_table_widget.verticalHeader().setVisible(False)
        self.excluded_symbols_table_widget.setSortingEnabled(True)
        self._populate_excluded_symbols_table()
        self.excluded_symbols_table_widget.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        excluded_symbols_layout.addWidget(self.excluded_symbols_table_widget)
        self.tab_widget.addTab(self.excluded_symbols_tab, "Excluded Symbols")

        table_buttons_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Row")
        self.delete_row_button = QPushButton("Delete Selected Row")
        table_buttons_layout.addWidget(self.add_row_button)
        table_buttons_layout.addWidget(self.delete_row_button)
        table_buttons_layout.addStretch()
        main_layout.addLayout(table_buttons_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        main_layout.addWidget(self.button_box)

        self.add_row_button.clicked.connect(self._add_row_to_current_tab)
        self.delete_row_button.clicked.connect(self._delete_row_from_current_tab)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.tab_widget.currentChanged.connect(self._update_button_states)

        if parent:
            if hasattr(parent, "styleSheet"):
                self.setStyleSheet(parent.styleSheet())
            if hasattr(parent, "font"):
                self.setFont(parent.font())
            else:
                self.setFont(QFont("Arial", 9)) # Fallback font

    def _populate_overrides_table(self):
        table = self.overrides_table_widget
        table.setSortingEnabled(False)
        table.setRowCount(0)
        sorted_symbols = sorted(self._original_overrides.keys())
        table.setRowCount(len(sorted_symbols))

        for row_idx, symbol in enumerate(sorted_symbols):
            override_data = self._original_overrides.get(symbol, {})
            price = override_data.get("price")
            asset_type = override_data.get("asset_type", "")
            sector = override_data.get("sector", "")
            geography = override_data.get("geography", "")
            industry = override_data.get("industry", "")

            item_symbol = QTableWidgetItem(symbol)
            table.setItem(row_idx, 0, item_symbol)
            price_str = f"{price:.4f}" if price is not None and pd.notna(price) else ""
            item_price = QTableWidgetItem(price_str)
            table.setItem(row_idx, 1, item_price)
            table.setItem(row_idx, 2, QTableWidgetItem(asset_type))
            table.setItem(row_idx, 3, QTableWidgetItem(sector))
            table.setItem(row_idx, 4, QTableWidgetItem(geography))
            table.setItem(row_idx, 5, QTableWidgetItem(industry))

        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(False)
        table.itemChanged.connect(self._validate_overrides_cell_change)

    def _populate_symbol_map_table(self):
        table = self.symbol_map_table_widget
        table.setSortingEnabled(False)
        table.setRowCount(0)
        sorted_internal_symbols = sorted(self._original_symbol_map.keys())
        table.setRowCount(len(sorted_internal_symbols))

        for row_idx, internal_symbol in enumerate(sorted_internal_symbols):
            yf_ticker = self._original_symbol_map.get(internal_symbol, "")
            table.setItem(row_idx, 0, QTableWidgetItem(internal_symbol))
            table.setItem(row_idx, 1, QTableWidgetItem(yf_ticker))

        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.itemChanged.connect(self._validate_map_cell_change)

    def _populate_excluded_symbols_table(self):
        table = self.excluded_symbols_table_widget
        table.setSortingEnabled(False)
        table.setRowCount(0)
        sorted_excluded = sorted(list(self._original_excluded_symbols))
        table.setRowCount(len(sorted_excluded))

        for row_idx, excluded_symbol in enumerate(sorted_excluded):
            table.setItem(row_idx, 0, QTableWidgetItem(excluded_symbol))

        table.setSortingEnabled(True)
        table.resizeColumnsToContents()
        table.itemChanged.connect(self._validate_excluded_cell_change)

    @Slot(QTableWidgetItem)
    def _validate_map_cell_change(self, item: QTableWidgetItem):
        text = item.text().strip().upper()
        item.setText(text)
        if not text:
            item.setBackground(QColor("salmon"))
            item.setToolTip("Symbol cannot be empty.")
        else:
            item.setBackground(QColor("white"))
            item.setToolTip("")

    @Slot(QTableWidgetItem)
    def _validate_excluded_cell_change(self, item: QTableWidgetItem):
        text = item.text().strip().upper()
        item.setText(text)
        if not text:
            item.setBackground(QColor("salmon"))
            item.setToolTip("Symbol cannot be empty.")
        else:
            item.setBackground(QColor("white"))
            item.setToolTip("")

    @Slot()
    def _update_button_states(self):
        pass

    @Slot(QTableWidgetItem)
    def _validate_overrides_cell_change(self, item: QTableWidgetItem):
        if item.column() == 1:  # Price column
            text = item.text().strip().replace(",", "")
            state = self.price_validator.validate(text, 0)[0]
            if state != QDoubleValidator.Acceptable:
                item.setBackground(QColor("salmon"))
                item.setToolTip("Invalid price: Must be a positive number.")
            else:
                item.setBackground(QColor("white"))
                item.setToolTip("")
                try:
                    item.setText(f"{float(text):.4f}")
                except ValueError:
                    pass
        elif item.column() == 0:  # Symbol column
            text = item.text().strip().upper()
            item.setText(text)
            if not text:
                item.setBackground(QColor("salmon"))
                item.setToolTip("Symbol cannot be empty.")
            else:
                item.setBackground(QColor("white"))
                item.setToolTip("")
        elif item.column() in [2, 3, 4, 5]:
            item.setText(item.text().strip())

    def _add_row_to_current_tab(self):
        current_tab_index = self.tab_widget.currentIndex()
        table_widget = None
        num_cols = 0

        if current_tab_index == 0:
            table_widget = self.overrides_table_widget
            num_cols = 6
        elif current_tab_index == 1:
            table_widget = self.symbol_map_table_widget
            num_cols = 2
        elif current_tab_index == 2:
            table_widget = self.excluded_symbols_table_widget
            num_cols = 1

        if table_widget:
            current_row_count = table_widget.rowCount()
            table_widget.insertRow(current_row_count)
            first_item = None
            for col in range(num_cols):
                item = QTableWidgetItem("")
                table_widget.setItem(current_row_count, col, item)
                if col == 0:
                    first_item = item
            if first_item:
                table_widget.scrollToItem(first_item, QAbstractItemView.PositionAtTop)
                table_widget.setCurrentItem(first_item)
                table_widget.editItem(first_item)

    def _delete_row_from_current_tab(self):
        current_tab_index = self.tab_widget.currentIndex()
        table_widget = None

        if current_tab_index == 0:
            table_widget = self.overrides_table_widget
        elif current_tab_index == 1:
            table_widget = self.symbol_map_table_widget
        elif current_tab_index == 2:
            table_widget = self.excluded_symbols_table_widget

        if not table_widget:
            return

        selected_items = table_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Selection Error", "Please select a cell in the row to delete.")
            return

        row_to_delete = selected_items[0].row()
        symbol_item = table_widget.item(row_to_delete, 0)
        symbol = symbol_item.text() if symbol_item else "this row"

        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Are you sure you want to delete the entry for '{symbol}'?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            table_widget.removeRow(row_to_delete)
            logging.info(f"Row for '{symbol}' removed from dialog.")

    def accept(self):
        new_overrides: Dict[str, Dict[str, Any]] = {}
        new_symbol_map: Dict[str, str] = {}
        new_excluded_symbols: Set[str] = set()
        has_errors = False

        # Process Manual Overrides Tab
        duplicate_symbols = set()
        seen_symbols = set()
        for row_idx in range(self.overrides_table_widget.rowCount()):
            # ... (validation logic as in the original class)
            symbol_item = self.overrides_table_widget.item(row_idx, 0)
            price_item = self.overrides_table_widget.item(row_idx, 1)
            asset_type_item = self.overrides_table_widget.item(row_idx, 2)
            sector_item = self.overrides_table_widget.item(row_idx, 3)
            geography_item = self.overrides_table_widget.item(row_idx, 4)
            industry_item = self.overrides_table_widget.item(row_idx, 5)

            if not all([symbol_item, price_item, asset_type_item, sector_item, geography_item, industry_item]):
                QMessageBox.warning(self, "Save Error", f"Error reading data from overrides row {row_idx+1}.")
                has_errors = True; break

            symbol = symbol_item.text().strip().upper()
            price_text = price_item.text().strip().replace(",", "")
            asset_type_text = asset_type_item.text().strip()
            sector_text = sector_item.text().strip()
            geography_text = geography_item.text().strip()
            industry_text = industry_item.text().strip()
            current_override_entry: Dict[str, Any] = {}

            if not symbol:
                QMessageBox.warning(self, "Save Error", f"Symbol cannot be empty in overrides row {row_idx+1}.")
                has_errors = True; self.overrides_table_widget.setCurrentItem(symbol_item); break
            if symbol in seen_symbols:
                duplicate_symbols.add(symbol); has_errors = True
            seen_symbols.add(symbol)

            if price_text:
                try:
                    price = float(price_text)
                    if price <= 0: raise ValueError("Price must be positive")
                    current_override_entry["price"] = price
                except (ValueError, TypeError):
                    QMessageBox.warning(self, "Save Error", f"Invalid price '{price_text}' for symbol '{symbol}' in overrides row {row_idx+1}.")
                    has_errors = True; self.overrides_table_widget.setCurrentItem(price_item); break

            if asset_type_text: current_override_entry["asset_type"] = asset_type_text
            if sector_text: current_override_entry["sector"] = sector_text
            if geography_text: current_override_entry["geography"] = geography_text
            if industry_text: current_override_entry["industry"] = industry_text
            if current_override_entry: new_overrides[symbol] = current_override_entry

        if duplicate_symbols:
            QMessageBox.warning(self, "Save Error", f"Duplicate symbols found in overrides: {', '.join(sorted(list(duplicate_symbols)))}.")
            has_errors = True
        if has_errors: super().reject(); return

        # Process Symbol Mapping Tab
        # ... (validation logic as in the original class)
        seen_internal_symbols_map = set()
        duplicate_internal_symbols_map = set()
        for row_idx in range(self.symbol_map_table_widget.rowCount()):
            internal_sym_item = self.symbol_map_table_widget.item(row_idx, 0)
            yf_ticker_item = self.symbol_map_table_widget.item(row_idx, 1)
            if not internal_sym_item or not yf_ticker_item:
                QMessageBox.warning(self, "Save Error", f"Error reading symbol map data from row {row_idx+1}.")
                has_errors = True; break
            internal_sym = internal_sym_item.text().strip().upper()
            yf_ticker = yf_ticker_item.text().strip().upper()
            if not internal_sym or not yf_ticker:
                QMessageBox.warning(self, "Save Error", f"Internal Symbol and YF Ticker cannot be empty in Symbol Mapping row {row_idx+1}.")
                has_errors = True; self.symbol_map_table_widget.setCurrentItem(internal_sym_item); break
            if internal_sym in seen_internal_symbols_map:
                duplicate_internal_symbols_map.add(internal_sym); has_errors = True
            seen_internal_symbols_map.add(internal_sym)
            new_symbol_map[internal_sym] = yf_ticker

        if duplicate_internal_symbols_map:
            QMessageBox.warning(self, "Save Error", f"Duplicate Internal Symbols found in Symbol Mapping: {', '.join(sorted(list(duplicate_internal_symbols_map)))}.")
            has_errors = True
        if has_errors: super().reject(); return

        # Process Excluded Symbols Tab
        # ... (validation logic as in the original class)
        for row_idx in range(self.excluded_symbols_table_widget.rowCount()):
            excluded_sym_item = self.excluded_symbols_table_widget.item(row_idx, 0)
            if not excluded_sym_item:
                QMessageBox.warning(self, "Save Error", f"Error reading excluded symbol data from row {row_idx+1}.")
                has_errors = True; break
            excluded_sym = excluded_sym_item.text().strip().upper()
            if not excluded_sym:
                QMessageBox.warning(self, "Save Error", f"Excluded Symbol cannot be empty in row {row_idx+1}.")
                has_errors = True; self.excluded_symbols_table_widget.setCurrentItem(excluded_sym_item); break
            new_excluded_symbols.add(excluded_sym)

        if has_errors: super().reject(); return

        if not has_errors:
            self.updated_settings["manual_price_overrides"] = new_overrides
            self.updated_settings["user_symbol_map"] = new_symbol_map
            self.updated_settings["user_excluded_symbols"] = sorted(list(new_excluded_symbols))
            logging.info(f"SymbolSettingsDialog accepted. Updated Settings: {self.updated_settings}")
            super().accept()

    @staticmethod
    def get_symbol_settings(
        parent=None,
        current_overrides=None,
        current_symbol_map=None,
        current_excluded_symbols=None,
    ) -> Optional[Dict[str, Any]]:
        if current_overrides is None: current_overrides = {}
        if current_symbol_map is None: current_symbol_map = {}
        if current_excluded_symbols is None: current_excluded_symbols = set()

        dialog = ManualPriceDialog(
            current_overrides, current_symbol_map, current_excluded_symbols, parent
        )
        if dialog.exec():
            return dialog.updated_settings
        return None
