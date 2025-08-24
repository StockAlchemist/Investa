# Auto-generated from main_gui.py modularization
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QFormLayout,
    QTableWidget,
    QTableWidgetItem,
    QDialogButtonBox,
    QTabWidget,
    QPushButton,
    QGroupBox,
    QTextEdit,
    QTableView,
    QAbstractItemView,
    QHeaderView,
    QMessageBox,
    QSpinBox,
    QListWidget,
    QSplitter,
    QListWidgetItem,
    QSpacerItem,
    QSizePolicy,
    QWidget,
    QScrollArea,
    QCompleter,
    QDateEdit,
)
from PySide6.QtCore import Qt, Signal, Slot, QStringListModel, QDate
from PySide6.QtGui import QDoubleValidator, QFont, QIcon, QColor, QValidator
from typing import Dict, Any, Optional, List, Set, Tuple
import pandas as pd
import json
import os
import logging
import traceback
from datetime import date, datetime
import sqlite3

from models import PandasModel
import config

from finutils import (
    format_currency_value,
    format_percentage_value,
    format_large_number_display,
    format_integer_with_commas,
    format_float_with_commas,
)

from config import (
    COMMON_CURRENCIES,
    STANDARD_SECTORS,
    STANDARD_ASSET_TYPES,
    SECTOR_INDUSTRY_MAP,
)


from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# If you use mplcursors for tooltips:
try:
    import mplcursors

    MPLCURSORS_AVAILABLE = True
except ImportError:
    MPLCURSORS_AVAILABLE = False

from config import CHART_DPI

from PySide6.QtGui import QColor

FALLBACK_QCOLOR_BG_DARK = QColor("#232629")
FALLBACK_QCOLOR_TEXT_DARK = QColor("#222")
FALLBACK_QCOLOR_TEXT_SECONDARY = QColor("#888")
FALLBACK_QCOLOR_ACCENT_TEAL = QColor("#1abc9c")
FALLBACK_QCOLOR_BORDER_LIGHT = QColor("#bbb")
FALLBACK_QCOLOR_BORDER_DARK = QColor("#555")
FALLBACK_QCOLOR_LOSS = QColor("#e74c3c")
FALLBACK_QCOLOR_GAIN = QColor("#2ecc71")

from config import CASH_SYMBOL_CSV, CSV_DATE_FORMAT


class FundamentalDataDialog(QDialog):
    """Dialog to display fundamental stock data."""

    def __init__(
        self, display_symbol: str, fundamental_data_dict: Dict[str, Any], parent=None
    ):
        super().__init__(parent)
        self.setWindowTitle(f"Fundamental Data: {display_symbol}")
        self.setMinimumSize(700, 750)  # Increased size for tabs
        self.setFont(
            parent.font()
            if parent and hasattr(parent, "font")
            else QFont("Arial", 10)  # Added hasattr check
        )  # Inherit font from parent app
        self._parent_app = parent  # Store parent reference to access methods
        self.display_symbol_for_title = (
            display_symbol  # Store for use in _dialog_format_value
        )
        main_layout = QVBoxLayout(self)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Overview (Existing Content) ---
        overview_tab = QWidget()
        overview_tab.setObjectName("OverviewTab")  # Add object name
        self._populate_overview_tab(
            overview_tab, fundamental_data_dict
        )  # display_symbol removed from call
        self.tab_widget.addTab(overview_tab, "Overview")

        # --- Tab 2: Income Statement (New) ---
        income_statement_tab = QWidget()
        income_statement_tab.setObjectName("IncomeStatementTab")
        financials_annual_data = fundamental_data_dict.get("financials_annual")
        financials_quarterly_data = fundamental_data_dict.get("financials_quarterly")
        self._setup_financial_statement_tab(
            income_statement_tab, financials_annual_data, financials_quarterly_data
        )
        self.tab_widget.addTab(income_statement_tab, "Income Statement")

        # --- Tab 3: Balance Sheet ---
        balance_sheet_tab = QWidget()
        balance_sheet_tab.setObjectName("BalanceSheetTab")
        balance_sheet_annual_data = fundamental_data_dict.get("balance_sheet_annual")
        balance_sheet_quarterly_data = fundamental_data_dict.get(
            "balance_sheet_quarterly"
        )
        self._setup_financial_statement_tab(
            balance_sheet_tab, balance_sheet_annual_data, balance_sheet_quarterly_data
        )
        self.tab_widget.addTab(balance_sheet_tab, "Balance Sheet")

        # --- Tab 4: Cash Flow ---
        cash_flow_tab = QWidget()
        cash_flow_tab.setObjectName("CashFlowTab")
        cash_flow_annual_data = fundamental_data_dict.get("cashflow_annual")
        cash_flow_quarterly_data = fundamental_data_dict.get("cashflow_quarterly")
        self._setup_financial_statement_tab(
            cash_flow_tab, cash_flow_annual_data, cash_flow_quarterly_data
        )
        self.tab_widget.addTab(cash_flow_tab, "Cash Flow")

        # --- Tab 5: Key Ratios ---
        self.key_ratios_tab = (
            QWidget()
        )  # Ensure this attribute exists if referenced elsewhere
        self.key_ratios_tab.setObjectName("KeyRatiosTab")
        self._setup_key_ratios_tab(
            self.key_ratios_tab, fundamental_data_dict.get("key_ratios_timeseries")
        )
        self.tab_widget.addTab(self.key_ratios_tab, "Key Ratios")

        # --- Dialog Buttons ---
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(self.accept)
        main_layout.addWidget(button_box)

        if parent and hasattr(parent, "styleSheet"):
            self.setStyleSheet(parent.styleSheet())

    def _get_dialog_currency_symbol(self):
        """Helper to get currency symbol based on the dialog's main symbol."""
        currency_symbol = "$"  # Default
        if self._parent_app and hasattr(self._parent_app, "_get_currency_symbol"):
            local_curr_code = self._parent_app._get_currency_for_symbol(
                self.display_symbol_for_title
            )
            if local_curr_code:
                currency_symbol = self._parent_app._get_currency_symbol(
                    currency_code=local_curr_code
                )
            else:  # Fallback to app's default display currency
                currency_symbol = self._parent_app._get_currency_symbol()
        return currency_symbol

    def _dialog_format_value(self, key, value) -> Tuple[str, Optional[QColor]]:
        if value is None or pd.isna(value):
            return "N/A", None

        currency_symbol_for_formatting = self._get_dialog_currency_symbol()
        color = None  # Default color is None (i.e., use default text color)

        # Define sets of keys for different formatting types
        LARGE_CURRENCY_KEYS = {
            "marketCap",
            "enterpriseValue",
            "totalRevenue",
            "ebitda",
            "grossProfits",
            "freeCashflow",
            "operatingCashflow",
            "totalCash",
            "totalDebt",
        }
        PERCENTAGE_KEYS_AS_FACTORS = {  # These values are factors (e.g., 0.05 for 5%)
            "payoutRatio",
            "heldPercentInsiders",
            "heldPercentInstitutions",
            "shortPercentOfFloat",
            "profitMargins",
            "grossMargins",
            "ebitdaMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
            "revenueGrowth",  # e.g., yfinance 'revenueGrowth'
            "earningsQuarterlyGrowth",  # e.g., yfinance 'earningsQuarterlyGrowth'
        }
        PERCENTAGE_KEYS_ALREADY_PERCENTAGES = {  # These values are already percentages (e.g., 5.0 for 5%)
            "dividendYield",
            "trailingAnnualDividendYield",
            "fiveYearAvgDividendYield",
            "Dividend Yield (%)",  # From current_valuation_ratios
            # Ratios from key_ratios_timeseries (though not directly used by overview tab's _dialog_format_value)
            "Gross Profit Margin (%)",
            "Net Profit Margin (%)",
            "Return on Equity (ROE) (%)",
            "Return on Assets (ROA) (%)",
        }
        CURRENCY_PER_SHARE_OR_PRICE_KEYS = {
            "dividendRate",
            "trailingAnnualDividendRate",
            "currentPrice",
            "regularMarketPrice",
            "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow",
            "targetHighPrice",
            "targetLowPrice",
            "targetMeanPrice",
            "targetMedianPrice",
            "regularMarketDayHigh",
            "regularMarketDayLow",
            "regularMarketOpen",
            "regularMarketPreviousClose",
            "bookValue",  # Book Value Per Share
            "trailingEps",
            "forwardEps",
            "revenuePerShare",
            "totalCashPerShare",
        }
        RATIO_KEYS_NO_CURRENCY = {
            # yfinance keys
            "trailingPE",
            "forwardPE",
            "pegRatio",
            "priceToBook",
            "priceToSalesTrailing12Months",
            "enterpriseToRevenue",
            "enterpriseToEbitda",
            "beta",
            "beta3Year",
            "shortRatio",
            "debtToEquity",
            "currentRatio",
            "quickRatio",
            # Friendly names from current_valuation_ratios (used in Overview tab)
            "P/E Ratio (TTM)",
            "Forward P/E Ratio",
            "Price-to-Sales (P/S) Ratio (TTM)",
            "Price-to-Book (P/B) Ratio (MRQ)",
            "Dividend Yield (%)",
            "Enterprise Value to EBITDA",
            # Ratios from key_ratios_timeseries (not directly formatted by this func for Overview, but good to list)
            "Current Ratio",
            "Quick Ratio",
            "Debt-to-Equity Ratio",
            "Interest Coverage Ratio",
            "Asset Turnover",
        }
        INTEGER_COUNT_KEYS = {
            "regularMarketVolume",
            "averageVolume",
            "averageVolume10days",
            "fullTimeEmployees",
            "sharesOutstanding",
            "floatShares",  # Can be very large
            "sharesShort",
            "sharesShortPriorMonth",
        }

        # Keys that should be colored based on their value (positive/negative)
        COLOR_CODED_KEYS = {
            "revenueGrowth",
            "earningsQuarterlyGrowth",
            "profitMargins",
            "grossMargins",
            "ebitdaMargins",
            "operatingMargins",
            "returnOnAssets",
            "returnOnEquity",
        }

        if isinstance(value, (int, float)):  # Common check for most numeric types
            # --- Start Color Logic ---
            if key in COLOR_CODED_KEYS:
                if value > 0:
                    color = (
                        self._parent_app.QCOLOR_GAIN_THEMED
                        if self._parent_app
                        else FALLBACK_QCOLOR_GAIN
                    )
                elif value < 0:
                    color = (
                        self._parent_app.QCOLOR_LOSS_THEMED
                        if self._parent_app
                        else FALLBACK_QCOLOR_LOSS
                    )
            # --- End Color Logic ---

            if key in LARGE_CURRENCY_KEYS:
                formatted_value = format_large_number_display(
                    value, currency_symbol_for_formatting
                )
            elif key in PERCENTAGE_KEYS_AS_FACTORS:
                formatted_value = format_percentage_value(value * 100, decimals=2)
            elif key in PERCENTAGE_KEYS_ALREADY_PERCENTAGES:
                formatted_value = format_percentage_value(
                    value, decimals=2
                )  # Already a percentage
            elif key in CURRENCY_PER_SHARE_OR_PRICE_KEYS:
                formatted_value = format_currency_value(
                    value, currency_symbol_for_formatting, decimals=2
                )
            elif key in RATIO_KEYS_NO_CURRENCY:
                formatted_value = format_float_with_commas(
                    value, decimals=2
                )  # No currency symbol
            elif key in INTEGER_COUNT_KEYS:
                formatted_value = format_integer_with_commas(value)
            elif key == "sharesShortPreviousMonthDate":  # Timestamp
                try:
                    formatted_value = datetime.fromtimestamp(value).strftime("%Y-%m-%d")
                except:
                    formatted_value = str(value)  # Fallback
            else:  # General fallback for other numeric values not explicitly categorized
                logging.debug(
                    f"Key '{key}' not in specific format lists, formatting as currency by default."
                )
                formatted_value = format_currency_value(
                    value, currency_symbol_for_formatting, decimals=2
                )
        elif isinstance(value, str):  # Handle strings directly
            formatted_value = value
        else:
            formatted_value = str(value)

        return formatted_value, color

    def _create_scrollable_form_tab(self) -> Tuple[QWidget, QFormLayout]:
        """Creates a scrollable tab with a QFormLayout."""
        tab_widget = QWidget()
        scroll_area = QScrollArea(tab_widget)
        scroll_area.setWidgetResizable(True)

        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        form_layout = QFormLayout(content_widget)
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        form_layout.setLabelAlignment(Qt.AlignRight)

        # Set the scroll_area as the layout for the tab_widget
        tab_main_layout = QVBoxLayout(tab_widget)
        tab_main_layout.addWidget(scroll_area)
        tab_main_layout.setContentsMargins(
            5, 5, 5, 5
        )  # Add some margins to the tab itself

        return tab_widget, form_layout

    def _setup_financial_statement_tab(
        self,
        tab_page: QWidget,
        annual_data_df: Optional[pd.DataFrame],
        quarterly_data_df: Optional[pd.DataFrame],
    ):
        """Sets up a financial statement tab with a period selector and a table view."""
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(5, 5, 5, 5)

        # Controls (ComboBox for period type)
        controls_layout = QHBoxLayout()
        period_label = QLabel("Period:")
        period_combo = QComboBox()
        period_combo.addItems(["Annual", "Quarterly"])
        period_combo.setMinimumWidth(120)  # MODIFIED: Increased width
        period_combo.setObjectName(f"{tab_page.objectName()}PeriodCombo")

        controls_layout.addWidget(period_label)
        controls_layout.addWidget(period_combo)
        controls_layout.addStretch()
        tab_layout.addLayout(controls_layout)

        # Table View
        table_view = QTableView()
        table_view.setObjectName(f"{tab_page.objectName()}Table")
        table_view.setAlternatingRowColors(True)
        table_view.setSelectionBehavior(QTableView.SelectRows)
        table_view.setWordWrap(False)
        table_view.setSortingEnabled(True)
        table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table_view.verticalHeader().setVisible(False)

        # Initial display: Annual data, fallback to Quarterly if Annual is not available
        initial_df_to_display = annual_data_df
        current_period_type = "Annual"  # Keep track of which type is being processed
        if (
            initial_df_to_display is None
            or not isinstance(initial_df_to_display, pd.DataFrame)
            or initial_df_to_display.empty
        ):
            initial_df_to_display = quarterly_data_df
            current_period_type = "Quarterly"
            if initial_df_to_display is not None and not initial_df_to_display.empty:
                period_combo.setCurrentText(
                    "Quarterly"
                )  # Reflect that quarterly is shown

        if (
            initial_df_to_display is None
            or not isinstance(initial_df_to_display, pd.DataFrame)
            or initial_df_to_display.empty
        ):
            display_df_for_model = pd.DataFrame(columns=["Metric", "No Data Available"])
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            model_df_init = initial_df_to_display.copy()
            # --- Format column headers (Dates) ---
            if current_period_type == "Annual":
                model_df_init.columns = [
                    (
                        pd.to_datetime(col).strftime("%Y")
                        if isinstance(
                            pd.to_datetime(col, errors="coerce"), pd.Timestamp
                        )
                        else str(col)
                    )
                    for col in model_df_init.columns
                ]
            elif current_period_type == "Quarterly":
                new_cols_q_init = []
                for col in model_df_init.columns:
                    try:
                        ts_init = pd.to_datetime(col)
                        new_cols_q_init.append(f"Q{ts_init.quarter} {ts_init.year}")
                    except (ValueError, TypeError):
                        new_cols_q_init.append(str(col))
                model_df_init.columns = new_cols_q_init
            # --- End Column Formatting ---
            if isinstance(model_df_init.index, pd.DatetimeIndex):
                model_df_init.index = model_df_init.index.strftime("%Y-%m-%d")
            display_df_for_model = model_df_init.reset_index()
            display_df_for_model.rename(columns={"index": "Metric"}, inplace=True)

        # Pass the financial statement flag and currency override to the model
        model = PandasModel(
            display_df_for_model,
            parent=self._parent_app,
            log_mode=True,
            is_financial_statement_model=True,
            currency_symbol_override=self._get_dialog_currency_symbol(),
        )  # Use _parent_app
        table_view.setModel(model)

        if not (initial_df_to_display is None or initial_df_to_display.empty):
            table_view.resizeColumnsToContents()
            try:
                metric_col_idx = display_df_for_model.columns.get_loc("Metric")
                table_view.setColumnWidth(metric_col_idx, 250)
            except KeyError:
                pass

        tab_layout.addWidget(table_view)

        # Connect ComboBox signal to update the table
        period_combo.currentTextChanged.connect(
            lambda text, tv=table_view, adf=annual_data_df, qdf=quarterly_data_df: self._update_financial_table_view(
                text, tv, adf, qdf
            )
        )

    def _update_financial_table_view(
        self,
        period_text: str,
        table_view: QTableView,
        annual_df: Optional[pd.DataFrame],
        quarterly_df: Optional[pd.DataFrame],
    ):
        """Updates the QTableView with the selected financial data (annual or quarterly)."""
        df_to_display = None
        current_period_type_update = ""
        if period_text == "Annual":
            df_to_display = annual_df
            current_period_type_update = "Annual"
        elif period_text == "Quarterly":
            df_to_display = quarterly_df
            current_period_type_update = "Quarterly"

        current_model = table_view.model()
        if not isinstance(current_model, PandasModel):
            logging.error("Table model is not a PandasModel instance. Cannot update.")
            return

        if (
            df_to_display is None
            or not isinstance(df_to_display, pd.DataFrame)
            or df_to_display.empty
        ):
            display_df_for_model = pd.DataFrame(
                columns=["Metric", f"{period_text} Data Not Available"]
            )
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            model_df = df_to_display.copy()
            # --- Format column headers (Dates) ---
            if current_period_type_update == "Annual":
                model_df.columns = [
                    (
                        pd.to_datetime(col).strftime("%Y")
                        if isinstance(
                            pd.to_datetime(col, errors="coerce"), pd.Timestamp
                        )
                        else str(col)
                    )
                    for col in model_df.columns
                ]
            elif current_period_type_update == "Quarterly":
                new_cols_q_update = []
                for col in model_df.columns:
                    try:
                        ts_update = pd.to_datetime(col)
                        new_cols_q_update.append(
                            f"Q{ts_update.quarter} {ts_update.year}"
                        )
                    except (ValueError, TypeError):
                        new_cols_q_update.append(str(col))
                model_df.columns = new_cols_q_update
            # --- End Column Formatting ---
            if isinstance(model_df.index, pd.DatetimeIndex):
                model_df.index = model_df.index.strftime("%Y-%m-%d")
            display_df_for_model = model_df.reset_index()
            display_df_for_model.rename(columns={"index": "Metric"}, inplace=True)
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

        current_model.updateData(display_df_for_model)
        # Ensure the model's currency symbol override is still correct if model is reused
        if hasattr(current_model, "_currency_symbol_override"):
            current_model._currency_symbol_override = self._get_dialog_currency_symbol()

        if not (df_to_display is None or df_to_display.empty):
            table_view.resizeColumnsToContents()
            try:
                metric_col_idx = display_df_for_model.columns.get_loc("Metric")
                table_view.setColumnWidth(metric_col_idx, 250)
            except KeyError:
                pass
        table_view.viewport().update()

    def _populate_overview_tab(
        self, overview_tab_page: QWidget, fundamental_data_dict: Dict[str, Any]
    ):
        """Populates the Overview tab with company details and summary."""
        scrollable_widget_container, form_layout_to_populate = (
            self._create_scrollable_form_tab()
        )

        page_main_layout = QVBoxLayout(overview_tab_page)
        page_main_layout.setContentsMargins(0, 0, 0, 0)
        page_main_layout.addWidget(scrollable_widget_container)

        # Use self._dialog_format_value which is now a class method
        # The form_layout_to_populate is where QGroupBoxes will be added.

        # --- Group 1: Company Overview ---
        group_overview = QGroupBox("Company Overview")
        layout_overview = QFormLayout(group_overview)
        layout_overview.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout_overview.setLabelAlignment(Qt.AlignRight)
        overview_fields = [
            ("shortName", "Short Name"),
            ("longName", "Long Name"),
            ("sector", "Sector"),
            ("industry", "Industry"),
            ("fullTimeEmployees", "Full Time Employees"),
        ]
        for key, friendly_name in overview_fields:
            value = fundamental_data_dict.get(key)
            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_overview.addRow(QLabel(f"<b>{friendly_name}:</b>"), value_label)
        form_layout_to_populate.addRow(group_overview)  # Add to the correct form layout

        # --- Group 2: Valuation Metrics ---
        group_valuation = QGroupBox("Valuation Metrics")
        layout_valuation = QFormLayout(group_valuation)
        layout_valuation.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout_valuation.setLabelAlignment(Qt.AlignRight)
        valuation_fields = [  # Original yfinance fields
            ("marketCap", "Market Cap"),
            ("enterpriseValue", "Enterprise Value"),
            ("trailingPE", "Trailing P/E"),
            ("forwardPE", "Forward P/E"),
            ("trailingEps", "Trailing EPS"),
            ("forwardEps", "Forward EPS"),
            ("pegRatio", "PEG Ratio"),
            ("priceToBook", "Price/Book (mrq)"),
            ("priceToSalesTrailing12Months", "Price/Sales (ttm)"),
            ("enterpriseToRevenue", "EV/Revenue (ttm)"),
            ("enterpriseToEbitda", "EV/EBITDA (ttm)"),
            ("beta", "Beta"),
        ]
        # Add current valuation ratios if available
        current_valuation_ratios_data = fundamental_data_dict.get(
            "current_valuation_ratios", {}
        )
        valuation_ratios_display_order = [
            "P/E Ratio (TTM)",
            "Forward P/E Ratio",
            "Price-to-Sales (P/S) Ratio (TTM)",
            "Price-to-Book (P/B) Ratio (MRQ)",
            "Dividend Yield (%)",
            "Enterprise Value to EBITDA",
        ]
        for ratio_key in valuation_ratios_display_order:
            if (
                ratio_key in current_valuation_ratios_data
            ):  # Check if key exists in the fetched data
                # Use ratio_key as friendly name if not in a predefined map, or map it
                friendly_name_ratio = ratio_key  # Default to key itself
                # Example mapping if needed:
                # if ratio_key == "P/E Ratio (TTM)": friendly_name_ratio = "P/E (TTM)"
                valuation_fields.append((ratio_key, friendly_name_ratio))

        for key, friendly_name in valuation_fields:
            value = fundamental_data_dict.get(key)  # Try direct yf key first
            if (
                value is None and key in current_valuation_ratios_data
            ):  # Fallback to our calculated ratio
                value = current_valuation_ratios_data.get(key)

            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_valuation.addRow(QLabel(f"<b>{friendly_name}:</b>"), value_label)
        form_layout_to_populate.addRow(group_valuation)

        # --- Group 3: Dividend Information ---
        group_dividend = QGroupBox("Dividend Information")
        layout_dividend = QFormLayout(group_dividend)
        layout_dividend.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout_dividend.setLabelAlignment(Qt.AlignRight)
        dividend_fields = [
            (
                "dividendYield",
                "Dividend Yield",
            ),  # yfinance often provides this as a factor
            ("dividendRate", "Annual Dividend Rate"),
            (
                "trailingAnnualDividendYield",
                "Trailing Ann. Div. Yield",
            ),  # yfinance factor
            ("trailingAnnualDividendRate", "Trailing Ann. Div. Rate"),
            ("fiveYearAvgDividendYield", "5Y Avg. Div. Yield"),  # yfinance factor
            ("payoutRatio", "Payout Ratio"),  # yfinance factor
        ]
        for key, friendly_name in dividend_fields:
            value = fundamental_data_dict.get(key)
            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_dividend.addRow(QLabel(f"<b>{friendly_name}:</b>"), value_label)
        form_layout_to_populate.addRow(group_dividend)

        # --- Group 4: Price Statistics ---
        group_price_stats = QGroupBox("Price Statistics")
        layout_price_stats = QFormLayout(group_price_stats)
        layout_price_stats.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout_price_stats.setLabelAlignment(Qt.AlignRight)
        price_stats_fields = [
            ("currentPrice", "Current Price"),
            ("regularMarketPrice", "Regular Market Price"),
            ("regularMarketDayHigh", "Day High"),
            ("regularMarketDayLow", "Day Low"),
            ("regularMarketOpen", "Open"),
            ("regularMarketPreviousClose", "Previous Close"),
            ("fiftyTwoWeekHigh", "52 Week High"),
            ("fiftyTwoWeekLow", "52 Week Low"),
            ("regularMarketVolume", "Volume"),
            ("averageVolume", "Avg. Volume (3 month)"),
            ("averageVolume10days", "Avg. Volume (10 day)"),
            ("targetHighPrice", "Target High Est."),
            ("targetLowPrice", "Target Low Est."),
            ("targetMeanPrice", "Target Mean Est."),
            ("targetMedianPrice", "Target Median Est."),
        ]
        for key, friendly_name in price_stats_fields:
            value = fundamental_data_dict.get(key)
            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_price_stats.addRow(QLabel(f"<b>{friendly_name}:</b>"), value_label)
        form_layout_to_populate.addRow(group_price_stats)

        # --- Group 5: Financial Highlights ---
        group_financial_highlights = QGroupBox("Financial Highlights")
        layout_financial_highlights = QFormLayout(group_financial_highlights)
        layout_financial_highlights.setFieldGrowthPolicy(
            QFormLayout.ExpandingFieldsGrow
        )
        layout_financial_highlights.setLabelAlignment(Qt.AlignRight)
        financial_fields = [
            ("totalRevenue", "Total Revenue (ttm)"),
            ("revenuePerShare", "Revenue/Share (ttm)"),
            ("revenueGrowth", "Quarterly Revenue Growth (yoy)"),
            ("grossProfits", "Gross Profit (ttm)"),
            ("ebitda", "EBITDA (ttm)"),
            ("profitMargins", "Profit Margin"),
            ("grossMargins", "Gross Margin"),
            ("ebitdaMargins", "EBITDA Margin"),
            ("operatingMargins", "Operating Margin (ttm)"),
            ("earningsQuarterlyGrowth", "Quarterly Earnings Growth (yoy)"),
            ("returnOnAssets", "Return on Assets (ttm)"),
            ("returnOnEquity", "Return on Equity (ttm)"),
            ("totalCash", "Total Cash (mrq)"),
            ("totalCashPerShare", "Cash/Share (mrq)"),
            ("totalDebt", "Total Debt (mrq)"),
            ("debtToEquity", "Debt/Equity (mrq)"),
            ("currentRatio", "Current Ratio (mrq)"),
            ("quickRatio", "Quick Ratio (mrq)"),
            ("freeCashflow", "Free Cash Flow (ttm)"),
            ("operatingCashflow", "Operating Cash Flow (ttm)"),
        ]
        for key, friendly_name in financial_fields:
            value = fundamental_data_dict.get(key)
            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_financial_highlights.addRow(
                QLabel(f"<b>{friendly_name}:</b>"), value_label
            )
        form_layout_to_populate.addRow(group_financial_highlights)

        # --- Group 6: Share Statistics ---
        group_share_stats = QGroupBox("Share Statistics")
        layout_share_stats = QFormLayout(group_share_stats)
        layout_share_stats.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        layout_share_stats.setLabelAlignment(Qt.AlignRight)
        share_stats_fields = [
            ("sharesOutstanding", "Shares Outstanding"),
            ("floatShares", "Float Shares"),
            ("heldPercentInsiders", "% Held by Insiders"),
            ("heldPercentInstitutions", "% Held by Institutions"),
            ("sharesShort", "Shares Short"),
            ("shortRatio", "Short Ratio"),
            ("shortPercentOfFloat", "Short % of Float"),
            ("sharesShortPreviousMonthDate", "Short Prev. Month Date"),
            ("sharesShortPriorMonth", "Shares Short Prev. Month"),
        ]
        for key, friendly_name in share_stats_fields:
            value = fundamental_data_dict.get(key)
            formatted_value, color = self._dialog_format_value(key, value)
            value_label = QLabel(formatted_value)
            if color:
                palette = value_label.palette()
                palette.setColor(value_label.foregroundRole(), color)
                value_label.setPalette(palette)
            layout_share_stats.addRow(QLabel(f"<b>{friendly_name}:</b>"), value_label)
        form_layout_to_populate.addRow(group_share_stats)

        # --- Group 7: Business Summary ---
        group_summary = QGroupBox("Business Summary")
        layout_summary = QVBoxLayout(group_summary)
        layout_summary.setContentsMargins(5, 5, 5, 5)
        layout_summary.setSpacing(5)

        summary_text = fundamental_data_dict.get("longBusinessSummary", "N/A")
        if summary_text and summary_text != "N/A":
            summary_text_edit = QTextEdit()
            summary_text_edit.setPlainText(summary_text)
            summary_text_edit.setReadOnly(True)
            summary_text_edit.setFixedHeight(150)
            layout_summary.addWidget(summary_text_edit)
        else:
            layout_summary.addWidget(QLabel("N/A"))

        form_layout_to_populate.addRow(group_summary)

    def _setup_key_ratios_tab(
        self, tab_page: QWidget, ratios_df: Optional[pd.DataFrame]
    ):
        """Sets up the Key Ratios tab with a table view."""
        tab_layout = QVBoxLayout(tab_page)
        tab_layout.setContentsMargins(5, 5, 5, 5)

        table_view = QTableView()
        table_view.setObjectName(f"{tab_page.objectName()}Table")
        table_view.setAlternatingRowColors(True)
        table_view.setSelectionBehavior(QTableView.SelectRows)
        table_view.setWordWrap(False)
        table_view.setSortingEnabled(True)  # Allow sorting
        table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        table_view.verticalHeader().setVisible(False)

        if (
            ratios_df is None
            or not isinstance(ratios_df, pd.DataFrame)
            or ratios_df.empty
        ):
            display_df_for_model = pd.DataFrame(columns=["Ratio", "No Data Available"])
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            # The ratios_df from calculate_key_ratios_timeseries has Period as index
            # and ratios as columns. For display, we want ratios as rows and periods as columns.
            # So, we transpose it.
            model_df_transposed = (
                ratios_df.transpose()
            )  # Ratios as index, Periods as columns

            # Format period columns (Dates)
            model_df_transposed.columns = [
                (
                    pd.to_datetime(col).strftime("%Y-%m-%d")
                    if isinstance(pd.to_datetime(col, errors="coerce"), pd.Timestamp)
                    else str(col)
                )
                for col in model_df_transposed.columns
            ]

            display_df_for_model = model_df_transposed.reset_index()
            display_df_for_model.rename(
                columns={"index": "Financial Ratio"}, inplace=True
            )
            table_view.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)

            # --- ADDED: Ensure numeric conversion for ratio value columns ---
            if not display_df_for_model.empty:
                for col_to_convert in display_df_for_model.columns:
                    # Skip the first column which contains ratio names (strings)
                    if col_to_convert != "Financial Ratio":
                        try:
                            # errors='coerce' will turn unconvertible strings into NaN
                            display_df_for_model[col_to_convert] = pd.to_numeric(
                                display_df_for_model[col_to_convert], errors="coerce"
                            )
                        except Exception as e_conv:
                            logging.warning(
                                f"Key Ratios Tab: Could not convert column '{col_to_convert}' to numeric: {e_conv}"
                            )
            # --- END ADDED ---

        # For Key Ratios, we don't want the "millions" formatting.
        model = PandasModel(
            display_df_for_model,
            parent=self._parent_app,  # parent should be PortfolioApp for themes
            log_mode=True,
            is_financial_statement_model=False,  # Key Ratios are not large currency values
        )  # log_mode for direct display
        table_view.setModel(model)

        if not (ratios_df is None or ratios_df.empty):
            table_view.resizeColumnsToContents()
            try:
                ratio_col_idx = display_df_for_model.columns.get_loc("Financial Ratio")
                table_view.setColumnWidth(
                    ratio_col_idx, 200
                )  # Make ratio name column wider
            except KeyError:
                pass

        tab_layout.addWidget(table_view)


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


class AccountCurrencyDialog(QDialog):
    """Dialog to manage account-to-currency mappings and default currency."""

    def __init__(
        self,
        current_map: Dict[str, str],
        current_default: str,
        all_accounts: List[str],
        parent=None,
        user_currencies: Optional[List[str]] = None,
    ):
        super().__init__(parent)
        self._parent_app = parent  # Store reference if needed
        self.setWindowTitle("Account Currency Settings")
        self.setMinimumSize(450, 400)  # Adjust size as needed

        # Store original values and all unique accounts found
        self._original_map = current_map.copy()
        self._original_default = current_default
        self._user_currencies = (
            user_currencies if user_currencies else COMMON_CURRENCIES.copy()
        )
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
            list(set([self._original_default] + self._user_currencies))
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
                        + self._user_currencies
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
        parent=None,
        current_map=None,
        current_default=None,
        all_accounts=None,
        user_currencies=None,
    ) -> Optional[Tuple[Dict[str, str], str]]:
        """Creates, shows dialog, and returns updated settings if saved."""
        if current_map is None:
            current_map = {}
        if current_default is None:
            current_default = "USD"
        if all_accounts is None:
            all_accounts = list(current_map.keys())
        if user_currencies is None:
            user_currencies = COMMON_CURRENCIES.copy()

        dialog = AccountCurrencyDialog(
            current_map, current_default, all_accounts, parent, user_currencies
        )
        if dialog.exec():  # Returns 1 if accepted (Save clicked), 0 if rejected
            return dialog.updated_map, dialog.updated_default
        return None  # Return None if Cancel was clicked


class SymbolChartDialog(QDialog):
    """Dialog to display a historical price chart for a single symbol."""

    def __init__(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        price_data: pd.DataFrame,
        display_currency: str,  # display_currency is the CODE (e.g., "USD") passed in
        parent=None,
    ):
        """
        Initializes the dialog and plots the historical data.

        Args:
            symbol (str): The symbol whose data is being plotted.
            start_date (date): The start date for the x-axis limit.
            end_date (date): The end date for the x-axis limit.
            price_data (pd.DataFrame): DataFrame indexed by date, with a 'price' column
                                        containing historical prices (should be adjusted).
            display_currency (str): The currency the prices *should* represent (used for axis label).
                                    Note: This dialog doesn't perform FX conversion itself;
                                    it assumes the input price_data is already effectively
                                    in the desired display context if needed, although typically
                                    stock charts show local price. Let's label with local for now.
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._symbol = symbol
        # Determine the likely local currency for labelling - requires parent access or passing it in.
        # For now, let's assume the parent (PortfolioApp) can provide it.
        self._local_currency_symbol = "$"  # Default
        if parent and hasattr(parent, "_get_currency_for_symbol"):
            local_curr_code = parent._get_currency_for_symbol(
                symbol
            )  # Need to implement this helper in PortfolioApp
            if local_curr_code and hasattr(parent, "_get_currency_symbol"):
                self._local_currency_symbol = parent._get_currency_symbol(
                    currency_code=local_curr_code
                )

        self.setWindowTitle(f"Historical Price Chart: {symbol}")
        self.setMinimumSize(700, 500)  # Good default size

        # --- Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        # --- Chart Area ---
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(1)

        self.figure = Figure(figsize=(7, 4.5), dpi=CHART_DPI)  # Slightly adjusted size
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("SymbolChartCanvas")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # --- Toolbar ---
        try:
            # Always create the standard toolbar for now
            self.toolbar = NavigationToolbar(
                self.canvas,
                chart_widget,
                coordinates=True,  # Show coordinates in this dialog's toolbar
            )
            self.toolbar.setObjectName(
                "SymbolChartToolbar"
            )  # Set object name for styling
            # Optional: Apply styles dynamically if needed, but QSS is preferred
            # if self._parent_app and hasattr(self._parent_app, 'styleSheet'):
            #      self.toolbar.setStyleSheet(self._parent_app.styleSheet()) # Might interfere? Use object name targeting in main QSS.
            logging.debug(
                f"Created standard NavigationToolbar for {self._symbol} chart."
            )
        except Exception as e_tb:
            logging.error(f"Error creating standard symbol chart toolbar: {e_tb}")
            self.toolbar = None  # Fallback

        chart_layout.addWidget(self.canvas, 1)
        if self.toolbar:  # Only add if successfully created
            chart_layout.addWidget(self.toolbar)

        main_layout.addWidget(chart_widget)
        # --- End Toolbar Modification ---

        # --- Plot Data ---
        # Pass the determined SYMBOL ($) for formatting, and the CODE for the axis label
        self._plot_data(
            symbol,
            start_date,
            end_date,
            price_data,
            self._local_currency_symbol,  # Pass the symbol $, ฿ etc.
            display_currency,
        )  # Pass the code USD, THB etc.

        # Apply parent's style and font if possible
        if parent:
            if hasattr(parent, "styleSheet"):
                self.setStyleSheet(parent.styleSheet())
            if hasattr(parent, "font"):
                self.setFont(parent.font())

    def _plot_data(
        self,
        symbol,
        start_date,
        end_date,
        price_data,
        currency_symbol_display,  # e.g. $
        currency_code_label,
    ):
        # e.g. USD
        """Helper method to perform the actual plotting."""
        ax = self.ax
        ax.clear()

        # Explicitly set background colors based on current theme
        # Both figure and axes should match the main dashboard background
        bg_color = (
            self.parent().QCOLOR_BACKGROUND_THEMED.name()
            if self.parent()
            else FALLBACK_QCOLOR_BG_DARK.name()
        )
        fig_bg_color = bg_color
        ax_bg_color = bg_color

        self.figure.patch.set_facecolor(fig_bg_color)
        ax.patch.set_facecolor(ax_bg_color)

        if price_data is None or price_data.empty or "price" not in price_data.columns:
            ax.text(
                0.5,
                0.5,
                f"No historical data available\nfor {symbol}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color=(
                    self.parent().QCOLOR_TEXT_SECONDARY_THEMED.name()
                    if self.parent()
                    and hasattr(self.parent(), "QCOLOR_TEXT_SECONDARY_THEMED")
                    else FALLBACK_QCOLOR_TEXT_SECONDARY.name()
                ),
            )
            ax.set_title(
                f"Price Chart: {symbol}",
                fontsize=10,
                weight="bold",
                color=(
                    self.parent().QCOLOR_TEXT_PRIMARY_THEMED.name()
                    if self.parent()
                    and hasattr(self.parent(), "QCOLOR_TEXT_PRIMARY_THEMED")
                    else FALLBACK_QCOLOR_TEXT_DARK.name()
                ),
            )
            self.canvas.draw()
            return

        # Ensure index is DatetimeIndex for plotting
        try:
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data.sort_index()
        except Exception as e:
            logging.error(f"Error processing index for symbol chart {symbol}: {e}")
            ax.text(
                0.5,
                0.5,
                f"Error processing data\nfor {symbol}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color=(
                    self.parent().QCOLOR_LOSS_THEMED.name()
                    if self.parent() and hasattr(self.parent(), "QCOLOR_LOSS_THEMED")
                    else FALLBACK_QCOLOR_LOSS.name()
                ),
            )
            self.canvas.draw()
            return

        # Plot the price data
        ax.plot(
            price_data.index,
            price_data["price"],
            label=f"{symbol} Price ({currency_symbol_display})",
            color=(
                self.parent().QCOLOR_ACCENT_THEMED.name()
                if self.parent() and hasattr(self.parent(), "QCOLOR_ACCENT_THEMED")
                else FALLBACK_QCOLOR_ACCENT_TEAL.name()
            ),
        )

        # Formatting
        title_color_name = (
            self.parent().QCOLOR_TEXT_PRIMARY_THEMED.name()
            if self.parent() and hasattr(self.parent(), "QCOLOR_TEXT_PRIMARY_THEMED")
            else FALLBACK_QCOLOR_TEXT_DARK.name()
        )
        label_color_name = (
            self.parent().QCOLOR_TEXT_PRIMARY_THEMED.name()
            if self.parent() and hasattr(self.parent(), "QCOLOR_TEXT_PRIMARY_THEMED")
            else FALLBACK_QCOLOR_TEXT_DARK.name()
        )
        tick_color_name = (
            self.parent().QCOLOR_TEXT_SECONDARY_THEMED.name()
            if self.parent() and hasattr(self.parent(), "QCOLOR_TEXT_SECONDARY_THEMED")
            else FALLBACK_QCOLOR_TEXT_SECONDARY.name()
        )
        grid_color_name = (
            self.parent().QCOLOR_BORDER_THEMED.name()
            if self.parent() and hasattr(self.parent(), "QCOLOR_BORDER_THEMED")
            else FALLBACK_QCOLOR_BORDER_LIGHT.name()
        )
        spine_color_name = (
            self.parent().QCOLOR_BORDER_THEMED.name()
            if self.parent() and hasattr(self.parent(), "QCOLOR_BORDER_THEMED")
            else FALLBACK_QCOLOR_BORDER_DARK.name()
        )

        ax.set_title(
            f"Historical Price: {symbol}",
            fontsize=10,
            weight="bold",
            color=title_color_name,
        )
        ax.set_ylabel(
            f"Price ({currency_symbol_display})", fontsize=9, color=label_color_name
        )
        ax.grid(
            True, which="major", linestyle="--", linewidth=0.5, color=grid_color_name
        )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(spine_color_name)
        ax.spines["left"].set_color(spine_color_name)
        ax.tick_params(axis="x", colors=tick_color_name, labelsize=8)
        ax.tick_params(axis="y", colors=tick_color_name, labelsize=8)

        # Y-axis formatting (simplified currency)
        formatter = mtick.FormatStrFormatter(f"{currency_symbol_display}%.2f")
        ax.yaxis.set_major_formatter(formatter)

        # Set X-axis limits based on provided dates
        try:
            # Convert date objects to something matplotlib understands for limits if needed
            # Often pandas Timestamps work directly
            pd_start = pd.Timestamp(start_date)
            pd_end = pd.Timestamp(end_date)
            ax.set_xlim(pd_start, pd_end)
        except Exception as e_lim:
            logging.warning(
                f"Could not set x-limits for symbol chart {symbol}: {e_lim}"
            )
            ax.autoscale(enable=True, axis="x", tight=True)  # Fallback

        # Rotate x-axis labels
        self.figure.autofmt_xdate(rotation=15)
        self.figure.tight_layout(pad=0.5)  # Add some padding

        # --- Add mplcursors Tooltip ---
        if MPLCURSORS_AVAILABLE:
            try:
                cursor = mplcursors.cursor(
                    ax.lines, hover=mplcursors.HoverMode.Transient
                )

                # Use lambda to capture the currency_symbol_display for the formatter
                @cursor.connect("add")
                def on_add(sel, sym=currency_symbol_display):  # Capture symbol
                    x_dt = mdates.num2date(sel.target[0])
                    date_str = x_dt.strftime("%Y-%m-%d")
                    price_val = sel.target[1]
                    # Use the captured symbol (sym)
                    sel.annotation.set_text(f"{date_str}\nPrice: {sym}{price_val:.2f}")
                    sel.annotation.get_bbox_patch().set(
                        facecolor="lightyellow", alpha=0.8, edgecolor="gray"
                    )
                    sel.annotation.set_fontsize(8)

            except Exception as e_cursor_sym:
                logging.error(
                    f"Error activating mplcursors for symbol chart {symbol}: {e_cursor_sym}"
                )
        # --- End mplcursors ---

        self.canvas.draw()


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
        self.setMinimumSize(800, 500)  # Increased width from 700 to 800

        # Store original values and prepare for updates
        # Ensure keys are uppercase for consistency
        self._original_overrides = {  # Renamed attribute
            k.upper().strip(): v for k, v in current_overrides.items()
        }
        self._original_symbol_map = {
            k.upper().strip(): v.upper().strip() for k, v in current_symbol_map.items()
        }
        self._original_excluded_symbols = {
            s.upper().strip() for s in current_excluded_symbols
        }

        # This will hold all updated settings
        self.updated_settings = {
            "manual_price_overrides": self._original_overrides.copy(),
            "user_symbol_map": self._original_symbol_map.copy(),
            "user_excluded_symbols": list(
                self._original_excluded_symbols
            ),  # Store as list for dialog return
        }

        # Store standard lists for comboboxes
        self.standard_asset_types = STANDARD_ASSET_TYPES
        self.standard_sectors = sorted(list(SECTOR_INDUSTRY_MAP.keys()))
        self.sector_industry_map = SECTOR_INDUSTRY_MAP

        # --- Layout ---
        main_layout = QVBoxLayout(self)

        # --- Tab Widget ---
        self.tab_widget = QTabWidget()
        main_layout.addWidget(self.tab_widget)

        # --- Tab 1: Manual Overrides (Price, Sector, etc.) ---
        self.overrides_tab = QWidget()
        overrides_layout = QVBoxLayout(self.overrides_tab)
        overrides_layout.addWidget(QLabel("Edit manual overrides (used as fallback):"))
        self.overrides_table_widget = QTableWidget()
        self.overrides_table_widget.setObjectName("ManualOverridesTable")
        self.overrides_table_widget.setColumnCount(
            6
        )  # Symbol, Price, Asset Type, Sector, Geography, Industry
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
        self.overrides_table_widget.setSelectionMode(
            QAbstractItemView.SingleSelection
        )  # Easier for delete
        self.overrides_table_widget.verticalHeader().setVisible(False)
        self.overrides_table_widget.setSortingEnabled(True)

        self.price_validator = QDoubleValidator(0.00000001, 1000000000.0, 8, self)
        self.price_validator.setNotation(QDoubleValidator.StandardNotation)

        self._populate_overrides_table()
        self.overrides_table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        for col_idx in range(1, 6):
            self.overrides_table_widget.horizontalHeader().setSectionResizeMode(
                col_idx, QHeaderView.ResizeToContents
            )
        overrides_layout.addWidget(self.overrides_table_widget)
        self.tab_widget.addTab(self.overrides_tab, "Manual Overrides")

        # --- Tab 2: Symbol Mapping ---
        self.symbol_map_tab = QWidget()
        symbol_map_layout = QVBoxLayout(self.symbol_map_tab)
        symbol_map_layout.addWidget(
            QLabel("Define custom symbol mappings to Yahoo Finance tickers:")
        )
        self.symbol_map_table_widget = QTableWidget()
        self.symbol_map_table_widget.setObjectName("SymbolMapTable")
        self.symbol_map_table_widget.setColumnCount(2)
        self.symbol_map_table_widget.setHorizontalHeaderLabels(
            ["Internal Symbol", "Yahoo Finance Ticker"]
        )
        self.symbol_map_table_widget.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.symbol_map_table_widget.setSelectionMode(QAbstractItemView.SingleSelection)
        self.symbol_map_table_widget.verticalHeader().setVisible(False)
        self.symbol_map_table_widget.setSortingEnabled(True)
        self._populate_symbol_map_table()
        self.symbol_map_table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        self.symbol_map_table_widget.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.Stretch
        )
        symbol_map_layout.addWidget(self.symbol_map_table_widget)
        self.tab_widget.addTab(self.symbol_map_tab, "Symbol Mapping")

        # --- Tab 3: Excluded Symbols ---
        self.excluded_symbols_tab = QWidget()
        excluded_symbols_layout = QVBoxLayout(self.excluded_symbols_tab)
        excluded_symbols_layout.addWidget(
            QLabel("Define symbols to exclude from Yahoo Finance fetching:")
        )
        self.excluded_symbols_table_widget = QTableWidget()
        self.excluded_symbols_table_widget.setObjectName("ExcludedSymbolsTable")
        self.excluded_symbols_table_widget.setColumnCount(1)
        self.excluded_symbols_table_widget.setHorizontalHeaderLabels(
            ["Excluded Symbol"]
        )
        self.excluded_symbols_table_widget.setSelectionBehavior(
            QAbstractItemView.SelectRows
        )
        self.excluded_symbols_table_widget.setSelectionMode(
            QAbstractItemView.SingleSelection
        )
        self.excluded_symbols_table_widget.verticalHeader().setVisible(False)
        self.excluded_symbols_table_widget.setSortingEnabled(True)
        self._populate_excluded_symbols_table()
        self.excluded_symbols_table_widget.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeToContents
        )
        excluded_symbols_layout.addWidget(self.excluded_symbols_table_widget)
        self.tab_widget.addTab(self.excluded_symbols_tab, "Excluded Symbols")

        # Common Add/Delete buttons (could be per tab or one set if context is clear)
        # For simplicity, let's assume they act on the currently visible tab's table.
        # This requires connecting them dynamically or having separate buttons per tab.
        # For now, let's add them outside the tab widget, acting on the current tab.
        # A better UX might be per-tab buttons.

        # --- Buttons ---
        table_buttons_layout = QHBoxLayout()
        self.add_row_button = QPushButton("Add Row")
        self.delete_row_button = QPushButton("Delete Selected Row")
        table_buttons_layout.addWidget(self.add_row_button)
        table_buttons_layout.addWidget(self.delete_row_button)
        table_buttons_layout.addStretch()
        main_layout.addLayout(table_buttons_layout)

        # --- Dialog Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        main_layout.addWidget(self.button_box)

        # --- Connections ---
        self.add_row_button.clicked.connect(self._add_row_to_current_tab)
        self.delete_row_button.clicked.connect(self._delete_row_from_current_tab)
        self.button_box.accepted.connect(self.accept)  # Override accept
        self.button_box.rejected.connect(self.reject)

        self.tab_widget.currentChanged.connect(
            self._update_button_states
        )  # Optional: disable buttons if table not focused

        # Apply parent's style and font
        if parent:
            if hasattr(parent, "styleSheet"):
                self.setStyleSheet(parent.styleSheet())
            if hasattr(parent, "font"):
                self.setFont(parent.font())

    def _populate_overrides_table(self):
        """Fills the manual overrides table."""
        table = self.overrides_table_widget
        table.setSortingEnabled(False)
        table.setRowCount(0)  # Clear existing rows
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

            # --- Asset Type ComboBox (Column 2) ---
            asset_type_combo = QComboBox()
            asset_type_combo.setEditable(True)
            asset_type_combo.setInsertPolicy(QComboBox.NoInsert)
            asset_type_combo.addItems(self.standard_asset_types)
            if asset_type and asset_type not in self.standard_asset_types:
                asset_type_combo.addItem(asset_type)  # Add non-standard value
            asset_type_combo.setCurrentText(str(asset_type))
            table.setCellWidget(row_idx, 2, asset_type_combo)

            # --- Sector ComboBox (Column 3) ---
            sector_combo = QComboBox()
            sector_combo.setEditable(True)
            sector_combo.setInsertPolicy(QComboBox.NoInsert)
            sector_combo.addItems(self.standard_sectors)
            if sector and sector not in self.standard_sectors:
                sector_combo.addItem(sector)  # Add non-standard value if it exists
            sector_combo.setCurrentText(str(sector))
            table.setCellWidget(row_idx, 3, sector_combo)

            table.setItem(row_idx, 4, QTableWidgetItem(str(geography)))

            # --- Industry ComboBox (Column 5) ---
            industry_combo = QComboBox()
            industry_combo.setEditable(True)
            industry_combo.setInsertPolicy(QComboBox.NoInsert)
            # Populate industries based on the current sector
            industries_for_sector = self.sector_industry_map.get(sector, [""])
            industry_combo.addItems(industries_for_sector)

            # If the saved industry is not in the standard list for that sector, add it.
            if industry and industry not in industries_for_sector:
                industry_combo.addItem(industry)  # Add non-standard value if it exists
            industry_combo.setCurrentText(str(industry))
            table.setCellWidget(row_idx, 5, industry_combo)

            # Connect sector change signal to update industry options
            sector_combo.currentTextChanged.connect(
                lambda text, r=row_idx: self._update_industry_options(r, text)
            )

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
        """Validates changes in the symbol map table."""
        text = item.text().strip().upper()
        item.setText(text)  # Force uppercase and strip
        if not text:
            item.setBackground(QColor("salmon"))
            item.setToolTip("Symbol cannot be empty.")
        else:
            item.setBackground(QColor("white"))
            item.setToolTip("")

    @Slot(QTableWidgetItem)
    def _validate_excluded_cell_change(self, item: QTableWidgetItem):
        """Validates changes in the excluded symbols table."""
        text = item.text().strip().upper()
        item.setText(text)  # Force uppercase and strip
        if not text:
            item.setBackground(QColor("salmon"))
            item.setToolTip("Symbol cannot be empty.")
        else:
            item.setBackground(QColor("white"))
            item.setToolTip("")

    @Slot()
    def _update_button_states(self):
        # This is a placeholder if you want to enable/disable add/delete
        # buttons based on which tab is active or if a table is focused.
        # For now, they will always be enabled.
        pass

    @Slot(QTableWidgetItem)
    def _validate_overrides_cell_change(self, item: QTableWidgetItem):
        """Validates changes made directly in the table cells."""
        if item.column() == 1:  # Price column
            text = item.text().strip().replace(",", "")  # Allow commas during input
            state = self.price_validator.validate(text, 0)[0]  # Get validation state
            if state != QDoubleValidator.Acceptable:
                item.setBackground(QColor("salmon"))  # Indicate error
                item.setToolTip("Invalid price: Must be a positive number.")
            else:
                item.setBackground(QColor("white"))  # Clear background on valid
                item.setToolTip("")
                # Optional: Format the valid number back into the cell
                try:
                    item.setText(f"{float(text):.4f}")  # Changed to .4f
                except ValueError:
                    pass  # Should not happen if Acceptable
        elif item.column() == 0:  # Symbol column (basic validation)
            text = item.text().strip().upper()
            item.setText(text)  # Force uppercase and strip whitespace
            if not text:
                item.setBackground(QColor("salmon"))
                item.setToolTip("Symbol cannot be empty.")
            else:
                item.setBackground(QColor("white"))
                item.setToolTip("")
        elif item.column() in [
            2,
            3,
            4,
            5,
        ]:  # Asset Type, Sector, Geography, ADDED Industry
            # Simple string validation: just strip whitespace
            item.setText(item.text().strip())
            # No background change for these for now, unless specific validation rules are added

    def _add_row_to_current_tab(self):
        current_tab_index = self.tab_widget.currentIndex()

        if current_tab_index == 0:  # Manual Overrides
            table_widget = self.overrides_table_widget
            current_row_count = table_widget.rowCount()
            table_widget.insertRow(current_row_count)

            # Set QTableWidgetItems for text-based columns
            for col_idx in [0, 1, 4]:  # Symbol, Price, Geography
                table_widget.setItem(current_row_count, col_idx, QTableWidgetItem(""))

            # --- ADDED: Set QComboBox for Asset Type column (col 2) ---
            asset_type_combo = QComboBox()
            asset_type_combo.setEditable(True)
            asset_type_combo.setInsertPolicy(QComboBox.NoInsert)
            asset_type_combo.addItems(self.standard_asset_types)
            table_widget.setCellWidget(current_row_count, 2, asset_type_combo)

            # Set QComboBox for Sector column (col 3)
            sector_combo = QComboBox()
            sector_combo.setEditable(True)
            sector_combo.setInsertPolicy(QComboBox.NoInsert)
            sector_combo.addItems(self.standard_sectors)
            table_widget.setCellWidget(current_row_count, 3, sector_combo)

            # Set QComboBox for Industry column (col 5)
            industry_combo = QComboBox()
            industry_combo.setEditable(True)
            industry_combo.setInsertPolicy(QComboBox.NoInsert)
            # Initially, populate with industries for the blank sector
            industry_combo.addItems(self.sector_industry_map.get("", [""]))
            table_widget.setCellWidget(current_row_count, 5, industry_combo)

            # Connect the new sector combo to update the new industry combo
            sector_combo.currentTextChanged.connect(
                lambda text, r=current_row_count: self._update_industry_options(r, text)
            )

            # Focus on the first cell
            first_item = table_widget.item(current_row_count, 0)
            if first_item:
                table_widget.scrollToItem(first_item, QAbstractItemView.PositionAtTop)
                table_widget.setCurrentItem(first_item)
                table_widget.editItem(first_item)

        elif current_tab_index == 1:  # Symbol Mapping
            table_widget = self.symbol_map_table_widget
            num_cols = table_widget.columnCount()
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

        elif current_tab_index == 2:  # Excluded Symbols
            table_widget = self.excluded_symbols_table_widget
            num_cols = table_widget.columnCount()
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

        if current_tab_index == 0:  # Manual Overrides
            table_widget = self.overrides_table_widget
        elif current_tab_index == 1:  # Symbol Mapping
            table_widget = self.symbol_map_table_widget
        elif current_tab_index == 2:  # Excluded Symbols
            table_widget = self.excluded_symbols_table_widget

        if not table_widget:
            return

        selected_items = table_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(
                self, "Selection Error", "Please select a cell in the row to delete."
            )
            return

        row_to_delete = selected_items[0].row()  # Get row from the first selected item
        symbol_item = table_widget.item(row_to_delete, 0)
        symbol = symbol_item.text() if symbol_item else "this row"

        reply = QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete the entry for '{symbol}'?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )

        if reply == QMessageBox.Yes:
            table_widget.removeRow(row_to_delete)
            logging.info(f"Row for '{symbol}' removed from dialog.")

    def _update_industry_options(self, row: int, sector_text: str):
        """Updates the industry dropdown options based on the selected sector for a given row."""
        try:
            industry_combo = self.overrides_table_widget.cellWidget(row, 5)
            if not isinstance(industry_combo, QComboBox):
                return

            # Get the list of standard industries for the new sector
            industries_for_sector = self.sector_industry_map.get(sector_text, [])

            # Always ensure a blank option is at the top
            final_industry_list = sorted(list(set([""] + industries_for_sector)))

            industry_combo.clear()
            industry_combo.addItems(final_industry_list)

            # After repopulating, the combobox will automatically select the first item ("").
            # This is the desired behavior when a sector changes, as the old industry is likely invalid.

        except Exception as e:
            logging.error(f"Error updating industry options for row {row}: {e}")

    def accept(self):
        """Overrides accept to validate all data and store results."""
        new_overrides: Dict[str, Dict[str, Any]] = {}
        new_symbol_map: Dict[str, str] = {}
        new_excluded_symbols: Set[str] = set()  # Use set for uniqueness check
        has_errors = False

        # --- Process Manual Overrides Tab ---
        duplicate_symbols = set()
        seen_symbols = set()

        for row_idx in range(self.overrides_table_widget.rowCount()):
            symbol_item = self.overrides_table_widget.item(row_idx, 0)
            price_item = self.overrides_table_widget.item(row_idx, 1)
            asset_type_combo_widget = self.overrides_table_widget.cellWidget(row_idx, 2)
            sector_combo_widget = self.overrides_table_widget.cellWidget(row_idx, 3)
            geography_item = self.overrides_table_widget.item(row_idx, 4)
            industry_combo_widget = self.overrides_table_widget.cellWidget(row_idx, 5)

            # Ensure all items exist for the row
            if not all(
                [
                    symbol_item,
                    price_item,
                    asset_type_combo_widget,
                    sector_combo_widget,
                    geography_item,
                    industry_combo_widget,
                ]
            ):
                QMessageBox.warning(
                    self, "Save Error", f"Error reading data from row {row_idx+1}."
                )
                has_errors = True
                break

            symbol = symbol_item.text().strip().upper()
            price_text = price_item.text().strip().replace(",", "")
            asset_type_text = asset_type_combo_widget.currentText().strip()
            sector_text = sector_combo_widget.currentText().strip()
            geography_text = geography_item.text().strip()
            industry_text = industry_combo_widget.currentText().strip()
            current_override_entry: Dict[str, Any] = {}

            # Validate Symbol
            if not symbol:
                QMessageBox.warning(
                    self, "Save Error", f"Symbol cannot be empty in row {row_idx+1}."
                )
                has_errors = True
                self.overrides_table_widget.setCurrentItem(
                    symbol_item
                )  # Highlight error
                break
            if symbol in seen_symbols:
                duplicate_symbols.add(symbol)
                has_errors = True  # Mark as error but continue checking all rows
            seen_symbols.add(symbol)

            # Validate Price (only if text is not empty)
            if price_text:  # Only validate if user entered something
                try:
                    price = float(price_text)
                    if price <= 0:
                        raise ValueError("Price must be positive")
                    current_override_entry["price"] = price
                except (ValueError, TypeError):
                    QMessageBox.warning(
                        self,
                        "Save Error",
                        f"Invalid price '{price_text}' for symbol '{symbol}' in row {row_idx+1}. Must be a positive number if entered.",
                    )
                    has_errors = True
                    self.overrides_table_widget.setCurrentItem(price_item)
                    break

            # Store other fields if they are not empty
            if asset_type_text:
                current_override_entry["asset_type"] = asset_type_text
            if sector_text:
                current_override_entry["sector"] = sector_text
            if geography_text:
                current_override_entry["geography"] = geography_text

            if industry_text:
                current_override_entry["industry"] = industry_text
            if (
                current_override_entry
            ):  # Only add if there's at least one override value
                new_overrides[symbol] = current_override_entry

        if duplicate_symbols:
            QMessageBox.warning(
                self,
                "Save Error",
                f"Duplicate symbols found: {', '.join(sorted(list(duplicate_symbols)))}. Please correct before saving.",
            )
            has_errors = True

        if has_errors:
            super().reject()  # Don't close if errors on this tab
            return

        # --- Process Symbol Mapping Tab ---
        seen_internal_symbols_map = set()
        duplicate_internal_symbols_map = set()
        for row_idx in range(self.symbol_map_table_widget.rowCount()):
            internal_sym_item = self.symbol_map_table_widget.item(row_idx, 0)
            yf_ticker_item = self.symbol_map_table_widget.item(row_idx, 1)

            if not internal_sym_item or not yf_ticker_item:
                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Error reading symbol map data from row {row_idx+1}.",
                )
                has_errors = True
                break

            internal_sym = internal_sym_item.text().strip().upper()
            yf_ticker = yf_ticker_item.text().strip().upper()

            if not internal_sym or not yf_ticker:
                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Internal Symbol and YF Ticker cannot be empty in Symbol Mapping row {row_idx+1}.",
                )
                has_errors = True
                self.symbol_map_table_widget.setCurrentItem(internal_sym_item)
                break

            if internal_sym in seen_internal_symbols_map:
                duplicate_internal_symbols_map.add(internal_sym)
                has_errors = True
            seen_internal_symbols_map.add(internal_sym)
            new_symbol_map[internal_sym] = yf_ticker

        if duplicate_internal_symbols_map:
            QMessageBox.warning(
                self,
                "Save Error",
                f"Duplicate Internal Symbols found in Symbol Mapping: {', '.join(sorted(list(duplicate_internal_symbols_map)))}. Please correct.",
            )
            has_errors = True

        if has_errors:
            super().reject()
            return

        # --- Process Excluded Symbols Tab ---
        for row_idx in range(self.excluded_symbols_table_widget.rowCount()):
            excluded_sym_item = self.excluded_symbols_table_widget.item(row_idx, 0)
            if not excluded_sym_item:
                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Error reading excluded symbol data from row {row_idx+1}.",
                )
                has_errors = True
                break

            excluded_sym = excluded_sym_item.text().strip().upper()
            if not excluded_sym:
                QMessageBox.warning(
                    self,
                    "Save Error",
                    f"Excluded Symbol cannot be empty in row {row_idx+1}.",
                )
                has_errors = True
                self.excluded_symbols_table_widget.setCurrentItem(excluded_sym_item)
                break

            new_excluded_symbols.add(
                excluded_sym
            )  # Add to set, duplicates handled automatically

        if has_errors:
            super().reject()
            return

        # --- If all validations pass ---
        if not has_errors:
            self.updated_settings["manual_price_overrides"] = new_overrides
            self.updated_settings["user_symbol_map"] = new_symbol_map
            self.updated_settings["user_excluded_symbols"] = sorted(
                list(new_excluded_symbols)
            )  # Convert set to sorted list for saving

            logging.info(
                f"SymbolSettingsDialog accepted. Updated Settings: {self.updated_settings}"
            )
            super().accept()  # Close dialog if validation passes

    # --- Static method to retrieve results cleanly ---
    @staticmethod
    def get_symbol_settings(  # Renamed method
        parent=None,
        current_overrides=None,
        current_symbol_map=None,
        current_excluded_symbols=None,
    ) -> Optional[Dict[str, Any]]:  # Return type is now a more general dict
        """Creates, shows dialog, and returns updated prices if saved."""
        if current_overrides is None:
            current_overrides = {}

        if current_symbol_map is None:
            current_symbol_map = {}
        if current_excluded_symbols is None:
            current_excluded_symbols = set()

        dialog = ManualPriceDialog(  # Class name is still ManualPriceDialog, will rename later
            current_overrides, current_symbol_map, current_excluded_symbols, parent
        )
        if dialog.exec():  # Returns 1 if accepted (Save clicked), 0 if rejected
            return dialog.updated_settings  # Return the new structure
        return None  # Return None if Cancel was clicked


class AddTransactionDialog(QDialog):
    """Dialog window for manually adding or editing a transaction entry."""

    def __init__(
        self,
        existing_accounts: List[str],
        parent=None,
        portfolio_symbols: Optional[List[str]] = None,  # <-- ADDED
        edit_data: Optional[Dict[str, Any]] = None,
    ):
        """
        Initializes the dialog with fields for transaction details.
        If edit_data is provided, pre-fills the dialog for editing.

        Args:
            existing_accounts (List[str]): A list of known account names to populate
                                           the account dropdown.
            parent (QWidget, optional): The parent widget.
            portfolio_symbols (Optional[List[str]], optional): A list of unique symbols
                                                               in the portfolio for autocompletion.
            edit_data (Optional[Dict[str, Any]], optional): Data to pre-fill for editing.
                                                            Keys should match AddTransactionDialog's expected field names
                                                            (i.e., CSV-like headers: "Date (MMM DD, YYYY)", "Quantity of Units", etc.).
        """
        super().__init__(parent)
        self.setWindowTitle(
            "Add New Transaction" if not edit_data else "Edit Transaction"
        )
        self.setMinimumWidth(350)  # Adjusted width slightly

        dialog_font = QFont("Arial", 10)
        self.setFont(dialog_font)

        self.transaction_types = [
            "Buy",
            "Sell",
            "Dividend",
            "Split",
            "Deposit",
            "Withdrawal",
            "Fees",
            "Short Sell",
            "Buy to Cover",
        ]
        self.total_amount_locked_by_user = (
            False  # Flag to prevent auto-calc from overwriting manual/edited total
        )

        main_dialog_layout = QVBoxLayout(self)
        form_layout = QFormLayout()
        form_layout.setContentsMargins(5, 5, 5, 5)  # Added small margins
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(8)
        input_min_width = 180  # Increased min width for inputs

        # --- Date ---
        self.date_edit = QDateEdit(date.today())
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")  # Consistent internal format
        self.date_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Date:", self.date_edit)

        # --- Type ---
        self.type_combo = QComboBox()
        self.type_combo.addItems(self.transaction_types)
        self.type_combo.setMinimumWidth(input_min_width)
        form_layout.addRow("Type:", self.type_combo)

        # --- Symbol ---
        self.symbol_edit = QLineEdit()
        self.symbol_edit.setPlaceholderText("e.g., AAPL, GOOG, $CASH")
        self.symbol_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Symbol:", self.symbol_edit)

        # --- ADDED: Autocompleter for Symbol ---
        if portfolio_symbols:
            symbols_for_completer = (
                portfolio_symbols  # <-- MODIFIED: No longer exclude $CASH here
            )
            self.symbol_completer_model = QStringListModel(self)
            self.symbol_completer_model.setStringList(
                sorted(list(set(symbols_for_completer)))
            )
            self.symbol_completer = QCompleter(self.symbol_completer_model, self)
            self.symbol_completer.setCaseSensitivity(Qt.CaseInsensitive)
            self.symbol_completer.setFilterMode(Qt.MatchContains)
            self.symbol_edit.setCompleter(self.symbol_completer)

        # --- Account ---
        self.account_combo = QComboBox()
        self.account_combo.addItems(
            sorted(list(set(existing_accounts)))
        )  # Ensure unique and sorted
        self.account_combo.setEditable(True)
        self.account_combo.setInsertPolicy(QComboBox.NoInsert)
        self.account_combo.setMinimumWidth(input_min_width)
        form_layout.addRow("Account:", self.account_combo)

        # --- Quantity ---
        self.quantity_edit = QLineEdit()
        self.quantity_edit.setPlaceholderText("e.g., 100.5")
        self.quantity_edit.setMinimumWidth(input_min_width)
        self.quantity_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.quantity_validator.setNotation(QDoubleValidator.StandardNotation)
        self.quantity_edit.setValidator(self.quantity_validator)
        form_layout.addRow("Quantity:", self.quantity_edit)

        # --- Price/Unit ---
        self.price_edit = QLineEdit()
        self.price_edit.setPlaceholderText("Per unit")
        self.price_edit.setMinimumWidth(input_min_width)
        self.price_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.price_validator.setNotation(QDoubleValidator.StandardNotation)
        self.price_edit.setValidator(self.price_validator)
        form_layout.addRow("Price/Unit:", self.price_edit)

        # --- Total Amount ---
        self.total_amount_edit = QLineEdit()
        self.total_amount_edit.setPlaceholderText("Auto for Buy/Sell/Short or manual")
        self.total_amount_edit.setMinimumWidth(input_min_width)
        self.total_validator = QDoubleValidator(0.0, 1e12, 2, self)  # Allow 0
        self.total_validator.setNotation(QDoubleValidator.StandardNotation)
        self.total_amount_edit.setValidator(self.total_validator)
        form_layout.addRow("Total Amount:", self.total_amount_edit)

        # --- Commission ---
        self.commission_edit = QLineEdit()
        self.commission_edit.setPlaceholderText("e.g., 6.95 (optional, default 0)")
        self.commission_edit.setMinimumWidth(input_min_width)
        self.commission_validator = QDoubleValidator(0.0, 1e12, 2, self)  # Allow 0
        self.commission_validator.setNotation(QDoubleValidator.StandardNotation)
        self.commission_edit.setValidator(self.commission_validator)
        form_layout.addRow("Commission:", self.commission_edit)

        # --- Split Ratio ---
        self.split_ratio_label = QLabel(
            "Split Ratio:"
        )  # Keep reference for enable/disable
        self.split_ratio_edit = QLineEdit()
        self.split_ratio_edit.setPlaceholderText("New shares per old (e.g., 2 for 2:1)")
        self.split_ratio_edit.setMinimumWidth(input_min_width)
        self.split_validator = QDoubleValidator(0.00000001, 1e12, 8, self)
        self.split_validator.setNotation(QDoubleValidator.StandardNotation)
        self.split_ratio_edit.setValidator(self.split_validator)
        form_layout.addRow(self.split_ratio_label, self.split_ratio_edit)

        # --- Note ---
        self.note_edit = QLineEdit()
        self.note_edit.setObjectName("NoteEdit")
        self.note_edit.setPlaceholderText("Optional note")
        self.note_edit.setMinimumWidth(input_min_width)
        form_layout.addRow("Note:", self.note_edit)

        # --- Buttons ---
        self.button_box = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)

        main_dialog_layout.addLayout(form_layout)
        main_dialog_layout.addWidget(self.button_box)

        # --- Connect Signals ---
        self.type_combo.currentTextChanged.connect(self._update_field_states_wrapper)
        self.symbol_edit.textChanged.connect(self._update_field_states_wrapper_symbol)

        # Numeric input validation feedback
        self.quantity_edit.textChanged.connect(self._validate_numeric_input)
        self.price_edit.textChanged.connect(self._validate_numeric_input)
        self.total_amount_edit.textChanged.connect(self._validate_numeric_input)
        self.commission_edit.textChanged.connect(self._validate_numeric_input)
        self.split_ratio_edit.textChanged.connect(self._validate_numeric_input)

        # Auto-calculation (ensure these are connected AFTER validators for proper behavior)
        self.quantity_edit.textChanged.connect(self._auto_calculate_total)
        self.price_edit.textChanged.connect(self._auto_calculate_total)
        # --- FIX: Store lambda as an attribute to allow disconnection ---
        self._total_amount_edited_slot = lambda text: setattr(
            self, "total_amount_locked_by_user", bool(text)
        )
        self.total_amount_edit.textEdited.connect(self._total_amount_edited_slot)
        # --- END FIX ---

        if edit_data:
            self._populate_fields_for_edit(edit_data)
        else:
            # Initial state update for a new transaction
            self._update_field_states(
                self.type_combo.currentText(), self.symbol_edit.text()
            )

    def _update_field_states(self, tx_type: str = None, symbol: str = None):
        """Enables/disables input fields based on the selected transaction type.

        Args:
            tx_type (str): The currently selected transaction type (e.g., "Buy", "Split").
            symbol (str, optional): The currently entered symbol. Defaults to None.
        """
        logging.debug(
            f"_update_field_states: Called with tx_type='{tx_type}', symbol='{symbol}'. Current Total Amount: '{self.total_amount_edit.text()}'"
        )
        # Use current values if None passed, and normalize
        tx_type_lower = (tx_type or self.type_combo.currentText()).lower()
        symbol_upper = (symbol or self.symbol_edit.text()).upper().strip()

        # or passed in if it's dynamic. For now, assuming it's an attribute.
        cash_symbol_to_use = getattr(
            self, "cash_symbol_csv", CASH_SYMBOL_CSV
        )  # Fallback to global
        is_cash_symbol = symbol_upper == cash_symbol_to_use

        # Default states
        qty_enabled, price_enabled, total_enabled, commission_enabled, split_enabled = (
            False,
            False,
            False,
            True,
            False,
        )
        price_readonly, total_readonly = False, False
        price_text_override = None

        if is_cash_symbol:
            if tx_type_lower in [
                "deposit",
                "withdrawal",
                "buy",
                "sell",
            ]:  # 'buy'/'sell' for $CASH are cash movements
                qty_enabled = True
                price_text_override = "1.00"  # Price of cash is 1
                price_readonly = True
                price_enabled = False  # Field itself is not interactive for $CASH price
                total_enabled = True  # Total is same as quantity
                total_readonly = True  # Auto-calculated from quantity
            elif tx_type_lower == "dividend":  # Cash dividend (e.g. interest)
                total_enabled = True  # User enters total dividend amount
                # Qty/Price not typically used for cash dividend entry if total is given
            elif tx_type_lower == "fees":  # Cash fees
                # Commission field is used for fee amount
                pass  # commission_enabled is already True by default
        elif tx_type_lower in [
            "buy",
            "sell",
            "short sell",
            "buy to cover",
        ]:  # Stock/ETF trades
            qty_enabled = True
            price_enabled = True
            total_enabled = True
            total_readonly = True  # Auto-calculated from Qty * Price
        elif tx_type_lower == "dividend":  # Stock/ETF dividend
            qty_enabled = True  # Optional, if per-share dividend
            price_enabled = True  # Dividend per share
            total_enabled = True  # Or total dividend amount
        elif tx_type_lower in ["split", "stock split"]:
            split_enabled = True
        elif tx_type_lower == "fees":  # Fees on a stock holding (rare)
            pass  # commission_enabled is True

        # Enable/disable and clear fields based on type
        self.quantity_edit.setEnabled(qty_enabled)
        self.price_edit.setEnabled(
            price_enabled
        )  # For stocks, this is true. For cash, false.
        self.price_edit.setReadOnly(price_readonly)
        if price_text_override is not None:
            self.price_edit.setText(price_text_override)

        self.total_amount_edit.setEnabled(total_enabled)
        self.total_amount_edit.setReadOnly(total_readonly)

        self.commission_edit.setEnabled(commission_enabled)
        self.split_ratio_edit.setEnabled(split_enabled)
        self.split_ratio_label.setEnabled(split_enabled)

        # Clear fields that are not relevant for the current transaction type
        if not self.quantity_edit.isEnabled():
            self.quantity_edit.clear()
        if not self.price_edit.isEnabled():
            logging.debug(
                f"_update_field_states: Price edit NOT enabled. Current text: '{self.price_edit.text()}' -> Clearing."
            )  # Don't clear if it's readonly and has a value (e.g. $CASH price=1.00)
            if not self.price_edit.isReadOnly():
                self.price_edit.clear()
        if not self.total_amount_edit.isEnabled():
            logging.debug(
                f"_update_field_states: Total Amount edit NOT enabled. Current text: '{self.total_amount_edit.text()}' -> Clearing."
            )
            self.total_amount_edit.clear()
        if not self.commission_edit.isEnabled():
            self.commission_edit.clear()
        if not self.split_ratio_edit.isEnabled():
            self.split_ratio_edit.clear()

        logging.debug(
            f"_update_field_states: After conditional clears. Current Total Amount: '{self.total_amount_edit.text()}'"
        )
        # Auto-calculate if total is readonly and not locked by user (e.g. not an edit where total was manually set)
        if total_readonly and not self.total_amount_locked_by_user:
            self._auto_calculate_total()
        elif not total_readonly:  # If total is editable by user, ensure lock is false
            self.total_amount_locked_by_user = False
        logging.debug(
            f"_update_field_states: After final _auto_calculate_total. Current Total Amount: '{self.total_amount_edit.text()}'"
        )

    # --- ADDED: Slot for live validation feedback ---
    @Slot()
    def _validate_numeric_input(self):
        """Provides visual feedback on numeric input fields based on validator state."""
        sender_widget = self.sender()
        if not isinstance(sender_widget, QLineEdit):
            return

        validator = sender_widget.validator()
        if not validator:
            return

        text = sender_widget.text()
        state = validator.validate(text, 0)[0]  # Get the QValidator.State

        if state == QValidator.Acceptable:
            sender_widget.setStyleSheet("")  # Reset style
        else:  # Intermediate or Invalid
            sender_widget.setStyleSheet(
                "background-color: #ffe0e0;"
            )  # Light red background

    def get_transaction_data(self) -> Optional[Dict[str, Any]]:
        """
        Validates user input and returns transaction data with Python types,
        keyed by standard CSV header names.

        Performs checks based on the transaction type. Shows warning messages
        for invalid input.

        Returns:
            Optional[Dict[str, Any]]: A dictionary where keys are CSV header names
                (e.g., "Date (MMM DD, YYYY)", "Quantity of Units") and values are
                Python types (datetime.date, float, str, or None).
                Returns None if validation fails.
        """
        data_for_processing: Dict[str, Any] = {}
        tx_type_display_case = (
            self.type_combo.currentText()
        )  # Original case for storing
        tx_type_lower = tx_type_display_case.lower()
        symbol = self.symbol_edit.text().strip().upper()
        account = (
            self.account_combo.currentText().strip()
        )  # Strip whitespace from account
        date_val_qt = self.date_edit.date()

        # Validate and get Python date object
        date_val: Optional[date] = None
        if date_val_qt.isValid():
            date_val = date_val_qt.toPython()
        else:  # Should not happen if QDateEdit is used properly
            QMessageBox.warning(self, "Input Error", "Date is invalid.")
            return None

        if not symbol:
            QMessageBox.warning(self, "Input Error", "Symbol cannot be empty.")
            self.symbol_edit.setFocus()
            return None
        if not account:
            QMessageBox.warning(self, "Input Error", "Account cannot be empty.")
            self.account_combo.setFocus()
            return None

        # Get raw string values, strip and remove commas for numeric conversion
        qty_str = self.quantity_edit.text().strip().replace(",", "")
        price_str = self.price_edit.text().strip().replace(",", "")
        total_str_from_field = self.total_amount_edit.text().strip().replace(",", "")
        comm_str = self.commission_edit.text().strip().replace(",", "")
        split_str = self.split_ratio_edit.text().strip().replace(",", "")
        note_str = self.note_edit.text().strip()

        # Initialize Python type variables
        qty: Optional[float] = None
        price: Optional[float] = None
        total: Optional[float] = None
        comm: float = 0.0  # Default to 0.0 if empty
        split: Optional[float] = None

        # Validate and convert commission first (optional field)
        if comm_str:
            try:
                comm = float(comm_str)
                if comm < 0:
                    QMessageBox.warning(
                        self, "Input Error", "Commission cannot be negative."
                    )
                    self.commission_edit.setFocus()
                    return None
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Commission must be a valid number."
                )
                self.commission_edit.setFocus()
                return None
        # If comm_str is empty, comm remains 0.0

        # Type-specific validation logic
        is_stock_trade = (
            tx_type_lower in ["buy", "sell", "short sell", "buy to cover"]
            and symbol != CASH_SYMBOL_CSV
        )
        is_cash_op = symbol == CASH_SYMBOL_CSV and tx_type_lower in [
            "deposit",
            "withdrawal",
            "buy",
            "sell",
        ]

        if is_stock_trade:
            if not self.quantity_edit.isEnabled() or not qty_str:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Quantity is required for '{tx_type_display_case}'.",
                )
                self.quantity_edit.setFocus()
                return None
            if not self.price_edit.isEnabled() or not price_str:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Price/Unit is required for '{tx_type_display_case}'.",
                )
                self.price_edit.setFocus()
                return None
            try:
                qty = float(qty_str)
                if qty <= 1e-8:  # Use a small tolerance, effectively must be > 0
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Quantity must be positive for this transaction type.",
                    )
                    self.quantity_edit.setFocus()
                    return None
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Quantity must be a valid number."
                )
                self.quantity_edit.setFocus()
                return None
            try:
                price = float(price_str)
                if price <= 1e-8:  # Price must be positive
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Price/Unit must be positive for this transaction type.",
                    )
                    self.price_edit.setFocus()
                    return None
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Price/Unit must be a valid number."
                )
                self.price_edit.setFocus()
                return None

            # Total amount for these types is typically calculated Qty * Price.
            # If total_amount_edit was locked (user entered or pre-filled), respect that value.
            # Otherwise, calculate it.
            if self.total_amount_locked_by_user and total_str_from_field:
                try:
                    total = float(total_str_from_field)
                    # Basic sanity check: calculated total vs provided total.
                    # This could be more sophisticated (e.g. tolerance).
                    # For now, if user locked it, we trust it unless it's clearly invalid (negative).
                    if total < 0:
                        QMessageBox.warning(
                            self, "Input Error", "Total Amount cannot be negative."
                        )
                        self.total_amount_edit.setFocus()
                        return None
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Manually entered Total Amount is not a valid number.",
                    )
                    self.total_amount_edit.setFocus()
                    return None
            else:  # Calculate total
                total = qty * price

        elif is_cash_op:  # $CASH Deposit/Withdrawal (or Buy/Sell aliased)
            if not self.quantity_edit.isEnabled() or not qty_str:
                QMessageBox.warning(
                    self,
                    "Input Error",
                    f"Amount (Quantity) is required for cash '{tx_type_display_case}'.",
                )
                self.quantity_edit.setFocus()
                return None
            try:
                qty = float(qty_str)
                if qty <= 1e-8:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Amount (Quantity) must be positive for cash operations.",
                    )
                    self.quantity_edit.setFocus()
                    return None
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Amount (Quantity) must be a valid number."
                )
                self.quantity_edit.setFocus()
                return None
            price = 1.0  # Price for $CASH is always 1.0
            total = qty  # For cash, total amount is the quantity itself.

        elif tx_type_lower == "dividend":
            # Initialize to None, they will be set if fields are valid and populated
            # qty, price, total are already initialized to None or 0.0 further up

            # Try to parse Total Amount first
            if self.total_amount_edit.isEnabled() and total_str_from_field:
                try:
                    parsed_total = float(total_str_from_field)
                    if parsed_total < 0:  # Dividends are typically non-negative
                        QMessageBox.warning(
                            self,
                            "Input Error",
                            "Dividend Total Amount cannot be negative.",
                        )
                        self.total_amount_edit.setFocus()
                        return None
                    total = parsed_total  # total is now set
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Dividend Total Amount is not a valid number.",
                    )
                    self.total_amount_edit.setFocus()
                    return None

            # If Total Amount was NOT provided (i.e., total is still None or was invalidly parsed before this point)
            if total is None:
                # Quantity and Price/Unit become mandatory
                if not qty_str:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Dividend: Quantity is required if Total Amount is not provided.",
                    )
                    self.quantity_edit.setFocus()
                    return None
                try:
                    qty = float(qty_str)
                    if qty <= 0:  # If calculating total, qty should be positive
                        QMessageBox.warning(
                            self,
                            "Input Error",
                            "Dividend Quantity must be positive if Total Amount is not provided.",
                        )
                        self.quantity_edit.setFocus()
                        return None
                except ValueError:
                    QMessageBox.warning(
                        self, "Input Error", "Dividend Quantity is not a valid number."
                    )
                    self.quantity_edit.setFocus()
                    return None

                if not price_str:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Dividend: Price/Unit is required if Total Amount is not provided.",
                    )
                    self.price_edit.setFocus()
                    return None
                try:
                    price = float(price_str)
                    if price <= 0:  # If calculating total, price should be positive
                        QMessageBox.warning(
                            self,
                            "Input Error",
                            "Dividend Price/Unit must be positive if Total Amount is not provided.",
                        )
                        self.price_edit.setFocus()
                        return None
                except ValueError:
                    QMessageBox.warning(
                        self,
                        "Input Error",
                        "Dividend Price/Unit is not a valid number.",
                    )
                    self.price_edit.setFocus()
                    return None

                total = qty * price  # Calculate total since it wasn't provided
            else:  # Total Amount WAS provided and is valid
                # Quantity and Price/Unit are optional. If provided, parse and validate them.
                if qty_str:  # If qty is provided, validate it
                    try:
                        qty = float(qty_str)
                        if qty < 0:  # Allow 0 quantity if total is given
                            QMessageBox.warning(
                                self,
                                "Input Error",
                                "Dividend Quantity cannot be negative.",
                            )
                            self.quantity_edit.setFocus()
                            return None
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            "Input Error",
                            "Dividend Quantity is not a valid number.",
                        )
                        self.quantity_edit.setFocus()
                        return None
                # else qty remains None (or its initial value if parsed from an empty string earlier)

                if price_str:  # If price is provided, validate it
                    try:
                        price = float(price_str)
                        if price < 0:  # Allow 0 price if total is given
                            QMessageBox.warning(
                                self,
                                "Input Error",
                                "Dividend Price/Unit cannot be negative.",
                            )
                            self.price_edit.setFocus()
                            return None
                    except ValueError:
                        QMessageBox.warning(
                            self,
                            "Input Error",
                            "Dividend Price/Unit is not a valid number.",
                        )
                        self.price_edit.setFocus()
                        return None
                # else price remains None (or its initial value)

            # Special handling for $CASH dividend (interest)
            if symbol == CASH_SYMBOL_CSV and total is not None:
                # If total is given for $CASH dividend, qty can be total and price 1.
                if qty is None:
                    qty = total  # If qty wasn't provided, set it to total
                if price is None:
                    price = 1.0  # If price wasn't provided, set it to 1.0

        elif tx_type_lower in ["split", "stock split"]:
            if not self.split_ratio_edit.isEnabled() or not split_str:
                QMessageBox.warning(
                    self, "Input Error", "Split Ratio is required for 'Split' type."
                )
                self.split_ratio_edit.setFocus()
                return None
            try:
                split = float(split_str)
                if split <= 1e-8:  # Ratio must be positive
                    QMessageBox.warning(
                        self, "Input Error", "Split Ratio must be positive."
                    )
                    self.split_ratio_edit.setFocus()
                    return None
            except ValueError:
                QMessageBox.warning(
                    self, "Input Error", "Split Ratio must be a valid number."
                )
                self.split_ratio_edit.setFocus()
                return None
            # qty, price, total are not primary inputs for split effect, can be None
            qty, price, total = None, None, None

        elif tx_type_lower == "fees":
            if (
                not self.commission_edit.isEnabled() or not comm_str
            ):  # For fees, commission IS the fee amount
                QMessageBox.warning(
                    self,
                    "Input Error",
                    "Fee amount (Commission) is required for 'Fees' type.",
                )
                self.commission_edit.setFocus()
                return None
            # For fees, other numeric fields are not applicable. Commission (comm) holds the fee.
            qty, price, total, split = None, None, None, None
            # Total amount can be considered the negative of commission for cash flow, but for CSV, often empty.
            # Let's ensure total remains None if not explicitly set for 'fees'.

        else:
            QMessageBox.warning(
                self,
                "Input Error",
                f"Transaction type '{tx_type_display_case}' not fully handled for validation or is unsupported.",
            )
            return None

        # Prepare dictionary with CSV-like headers and Python typed values
        data_for_processing = {
            "Date (MMM DD, YYYY)": date_val,  # datetime.date object
            "Transaction Type": tx_type_display_case,  # Original case string
            "Stock / ETF Symbol": symbol,
            "Quantity of Units": qty,  # float or None
            "Amount per unit": price,  # float or None
            "Total Amount": total,  # float or None
            "Fees": (
                comm if comm_str else None
            ),  # float or None (if comm_str was empty, comm is 0.0, make it None for CSV)
            "Investment Account": account,
            "Split Ratio (new shares per old share)": split,  # float or None
            "Note": note_str if note_str else None,  # string or None
            # "Local Currency" is determined by PortfolioApp based on account
        }
        logging.debug(f"Validated transaction data from dialog: {data_for_processing}")
        return data_for_processing

    def accept(self):
        """
        Overrides the default accept behavior to validate input before closing.

        Calls `get_transaction_data` for validation. If validation passes, the
        dialog is accepted; otherwise, it remains open.
        """
        tx_type_lower = self.type_combo.currentText().lower()
        total_amount_text_raw = self.total_amount_edit.text()  # Get raw text
        total_amount_text_stripped = total_amount_text_raw.strip()

        total_amount_is_filled_and_valid = False
        if self.total_amount_edit.isEnabled() and total_amount_text_stripped:
            # Validate against its own validator, considering potential commas
            total_validator_state = self.total_amount_edit.validator().validate(
                total_amount_text_raw, 0
            )[0]
            if total_validator_state == QValidator.Acceptable:
                try:  # Final check for non-negativity for relevant types
                    if (
                        float(total_amount_text_stripped.replace(",", "")) >= 0
                    ):  # Allow 0
                        total_amount_is_filled_and_valid = True
                except ValueError:
                    pass  # Validator should have caught non-numeric
        # --- ADDED: Explicit Validator Check Before Accept ---
        invalid_fields = []
        fields_to_check = [
            (self.quantity_edit, "Quantity"),
            (self.price_edit, "Price/Unit"),
            (self.total_amount_edit, "Total Amount"),
            (self.commission_edit, "Commission"),
            (self.split_ratio_edit, "Split Ratio"),
        ]

        for widget, name in fields_to_check:
            if widget.isEnabled():  # Only check enabled fields
                # logging.debug(f"Validating field: {name}, Enabled: {widget.isEnabled()}")
                validator = widget.validator()
                if validator:
                    text_raw = widget.text()  # Get raw text for validator
                    text_stripped = text_raw.strip()

                    # --- Start of new conditional validation logic ---
                    is_optional_due_to_dividend_total = False
                    if tx_type_lower == "dividend" and total_amount_is_filled_and_valid:
                        if name in ["Quantity", "Price/Unit"]:
                            is_optional_due_to_dividend_total = True

                    is_commission_field = name == "Commission"

                    if not text_stripped:  # Field is empty
                        # Commission is optional (defaults to 0) UNLESS tx_type is "Fees"
                        if is_commission_field and tx_type_lower != "fees":
                            # logging.debug(f"Field {name} is empty and optional (Commission not for Fees). Skipping strict validation.")
                            continue
                        if is_optional_due_to_dividend_total:
                            # logging.debug(f"Field {name} is empty and optional (Dividend Qty/Price with Total). Skipping strict validation.")
                            continue
                    # --- End of new conditional validation logic ---
                    state = validator.validate(text_raw, 0)[0]  # Validate raw text
                    if state != QValidator.Acceptable:
                        invalid_fields.append(name)
                        widget.setStyleSheet("background-color: #ffe0e0;")

        if invalid_fields:
            QMessageBox.warning(
                self,
                "Invalid Input",
                f"Please correct the following fields with invalid numeric input:\n- {', '.join(invalid_fields)}",
            )
            return  # Do not accept
        # --- END ADDED ---

        # Proceed with existing type-specific validation if numeric validation passes
        validated_data = self.get_transaction_data()
        if validated_data:
            # Reset background colors on successful save (optional, but good UX)
            for widget, _ in fields_to_check:
                widget.setStyleSheet("")
            super().accept()

    @Slot()
    def _auto_calculate_total(self):
        """Automatically calculates Total Amount from Quantity and Price."""
        # If the Total Amount field is itself read-only, it implies it's always auto-calculated.
        # The total_amount_locked_by_user flag is more for when Total Amount is editable by the user
        # and they have manually entered a value.
        if not self.total_amount_edit.isReadOnly() and self.total_amount_locked_by_user:
            logging.debug(
                f"_auto_calculate_total: SKIPPED. Total Amount is editable and locked by user. Lock: {self.total_amount_locked_by_user}"
            )
            return

        # Determine if price field should be used or if it's $CASH
        current_symbol = self.symbol_edit.text().strip().upper()
        is_cash_tx = (
            current_symbol == CASH_SYMBOL_CSV
        )  # CASH_SYMBOL_CSV is imported from config

        logging.debug(
            f"_auto_calculate_total: Called. Lock: {self.total_amount_locked_by_user}, Symbol: '{current_symbol}', Qty: '{self.quantity_edit.text()}', Price: '{self.price_edit.text()}'"
        )

        # If quantity cannot be entered, no auto-calculation based on it.
        if not self.quantity_edit.isEnabled():
            logging.debug(
                "_auto_calculate_total: SKIPPED because Qty field is not enabled."
            )
            return

        # If it's not a cash transaction, and price_edit is also disabled (e.g. for "Split"), then skip.
        if not is_cash_tx and not self.price_edit.isEnabled():
            logging.debug(
                "_auto_calculate_total: SKIPPED because (not $CASH and Price field is not enabled)."
            )
            # If total_amount_edit is read-only, it should be cleared as its inputs are not available.
            if self.total_amount_edit.isReadOnly():
                self.total_amount_edit.clear()
            return

        qty_str = self.quantity_edit.text().strip().replace(",", "")
        price_str = ""

        if is_cash_tx:
            price_str = "1.00"  # For $CASH, price is always 1.0
        elif (
            self.price_edit.isEnabled()
        ):  # Only read from price_edit if it's enabled for non-$CASH
            price_str = self.price_edit.text().strip().replace(",", "")
        else:  # Not $CASH and price_edit is not enabled (e.g. for Split type where price isn't used for total)
            logging.debug(
                "_auto_calculate_total: Price field not used for calculation (e.g., Split type or $CASH handled)."
            )
            # If total_amount_edit is read-only, it should be cleared as its inputs are not available.
            if self.total_amount_edit.isReadOnly():
                self.total_amount_edit.clear()
            return

        try:
            if (
                qty_str and price_str
            ):  # Both quantity and a determined price string must exist
                qty = float(qty_str)
                price = float(price_str)

                # Allow calculation for qty >= 0 and price >= 0.
                # Validators should prevent negative values where inappropriate.
                if qty >= 0 and price >= 0:
                    total = qty * price
                    decimal_places = 2  # Default
                    parent_app = self.parent()  # PortfolioApp instance
                    if (
                        parent_app
                        and hasattr(parent_app, "config")
                        and "decimal_places" in parent_app.config
                    ):
                        decimal_places = parent_app.config.get("decimal_places", 2)
                    self.total_amount_edit.setText(f"{total:.{decimal_places}f}")
                    logging.debug(
                        f"_auto_calculate_total: Calculated total {total:.{decimal_places}f}. Setting text."
                    )
                else:
                    # This case implies qty or price was negative, which should be handled by validators.
                    # If total_amount_edit is read-only (always auto-calculated), clear it. Otherwise, preserve.
                    if self.total_amount_edit.isReadOnly():
                        self.total_amount_edit.clear()
                    logging.debug(
                        f"_auto_calculate_total: Qty or Price was negative. Qty={qty}, Price={price}. Total cleared if read-only."
                    )
            elif (
                self.total_amount_edit.isReadOnly()
            ):  # If inputs are incomplete and total is auto-calc only
                self.total_amount_edit.clear()
                logging.debug(
                    "_auto_calculate_total: Qty or Price string empty and Total is read-only. Total cleared."
                )
            # If inputs are incomplete and total is editable, do nothing (preserve existing text as per original logic)

        except ValueError:
            if (
                self.total_amount_edit.isReadOnly()
            ):  # If conversion error and total is auto-calc only
                self.total_amount_edit.clear()
            logging.debug(
                f"_auto_calculate_total: ValueError during float conversion. Qty_str='{qty_str}', Price_str='{price_str}'. Total cleared if read-only."
            )
            # Pass, as validators on Qty/Price fields will show error to user.
            pass

    def _populate_fields_for_edit(self, data: Dict[str, Any]):
        """
        Pre-fills the dialog fields with data for editing.
        Assumes `data` keys are CSV-like headers (e.g., "Date (MMM DD, YYYY)").
        """
        logging.debug(f"Populating AddTransactionDialog for edit with raw data: {data}")

        # --- Temporarily disconnect signals that might interfere with pre-filling ---
        signals_to_disconnect = [
            (self.quantity_edit.textChanged, self._auto_calculate_total),
            (self.price_edit.textChanged, self._auto_calculate_total),
            (self.total_amount_edit.textEdited, self._total_amount_edited_slot),
            (self.type_combo.currentTextChanged, self._update_field_states_wrapper),
            (self.symbol_edit.textChanged, self._update_field_states_wrapper_symbol),
        ]
        for widget_signal, slot_func in signals_to_disconnect:
            try:
                widget_signal.disconnect(slot_func)
            except RuntimeError:  # Signal might not have been connected
                logging.debug(
                    f"Signal already disconnected or never connected for pre-fill: {widget_signal}"
                )

        # --- Date ---
        date_val_str = data.get(
            "Date (MMM DD, YYYY)"
        )  # Expects CSV standard format string
        if date_val_str:
            try:
                # CSV_DATE_FORMAT is "%b %d, %Y"
                parsed_dt = datetime.strptime(str(date_val_str), CSV_DATE_FORMAT)
                q_date = QDate(parsed_dt.year, parsed_dt.month, parsed_dt.day)
                if q_date.isValid():
                    self.date_edit.setDate(q_date)
                    logging.debug(f"  Set Date to: {q_date.toString('yyyy-MM-dd')}")
                else:
                    logging.warning(
                        f"  Could not parse date '{date_val_str}' to a valid QDate for edit dialog."
                    )
            except ValueError as e_date:
                logging.error(
                    f"  Error parsing date '{date_val_str}' for edit dialog: {e_date}"
                )
            except Exception as e_date_other:  # Catch any other unexpected error
                logging.error(
                    f"  Unexpected error processing date '{date_val_str}': {e_date_other}",
                    exc_info=True,
                )

        # --- Type ---
        tx_type_val = data.get("Transaction Type", "")
        type_idx = self.type_combo.findText(
            str(tx_type_val), Qt.MatchFixedString
        )  # Case-sensitive match
        if type_idx >= 0:
            self.type_combo.setCurrentIndex(type_idx)
        else:  # If not found, try case-insensitive, then set editable text as fallback
            type_idx_insensitive = self.type_combo.findText(
                str(tx_type_val), Qt.MatchContains | Qt.MatchCaseInsensitive
            )
            if type_idx_insensitive >= 0:
                self.type_combo.setCurrentIndex(type_idx_insensitive)
            else:
                self.type_combo.setEditText(str(tx_type_val))
                logging.warning(
                    f"  Transaction type '{tx_type_val}' not in standard list. Set as editable text."
                )
        logging.debug(f"  Set Type to: {self.type_combo.currentText()}")

        # --- Symbol ---
        self.symbol_edit.setText(str(data.get("Stock / ETF Symbol", "")))
        logging.debug(f"  Set Symbol to: {self.symbol_edit.text()}")

        # --- Account ---
        # AddTransactionDialog's account_combo is pre-filled with existing accounts.
        # setCurrentText will select it if it exists, or set the editable text if it's new.
        self.account_combo.setCurrentText(str(data.get("Investment Account", "")))
        logging.debug(f"  Set Account to: {self.account_combo.currentText()}")

        # --- Numeric Fields & Note ---
        # These keys match what get_transaction_data() would return and what CSVs store
        # We use the parent's formatter if available, otherwise simple string conversion
        parent = self.parent()  # QWidget.parent()
        formatter_func = (
            parent._format_for_edit_dialog
            if parent and hasattr(parent, "_format_for_edit_dialog")
            else lambda val, prec: str(val if pd.notna(val) else "")
        )

        self.quantity_edit.setText(formatter_func(data.get("Quantity of Units"), 8))
        self.price_edit.setText(formatter_func(data.get("Amount per unit"), 8))
        self.total_amount_edit.setText(formatter_func(data.get("Total Amount"), 2))
        self.commission_edit.setText(formatter_func(data.get("Fees"), 2))
        self.split_ratio_edit.setText(
            formatter_func(data.get("Split Ratio (new shares per old share)"), 8)
        )
        self.note_edit.setText(str(data.get("Note", "")))

        logging.debug(f"  Set Quantity to: {self.quantity_edit.text()}")
        logging.debug(f"  Set Price/Unit to: {self.price_edit.text()}")
        logging.debug(f"  Set Total Amount to: {self.total_amount_edit.text()}")
        logging.debug(f"  Set Commission to: {self.commission_edit.text()}")
        logging.debug(f"  Set Split Ratio to: {self.split_ratio_edit.text()}")
        logging.debug(f"  Set Note to: {self.note_edit.text()}")

        # If Total Amount was pre-filled and is relevant for the transaction type, lock it
        # to prevent auto-calculation from overwriting it during the initial _update_field_states call.
        current_tx_type_lower = self.type_combo.currentText().lower()
        current_symbol_upper = self.symbol_edit.text().upper().strip()  # Get symbol
        cash_symbol_to_use_edit = getattr(self, "cash_symbol_csv", CASH_SYMBOL_CSV)
        is_cash_symbol_edit = current_symbol_upper == cash_symbol_to_use_edit

        self.total_amount_locked_by_user = False  # Default to not locked

        if self.total_amount_edit.text():  # If there's pre-filled total amount
            if not is_cash_symbol_edit and current_tx_type_lower in [
                "buy",
                "sell",
                "short sell",
                "buy to cover",
                "dividend",
                "fees",
            ]:
                # For non-cash stock operations where total might be manually entered or important
                self.total_amount_locked_by_user = True
                logging.debug(
                    "  Locked Total Amount as it was pre-filled for a relevant non-cash transaction type."
                )
            elif is_cash_symbol_edit and current_tx_type_lower in ["dividend", "fees"]:
                # For $CASH dividend/fees, total is the primary input
                self.total_amount_locked_by_user = True
                logging.debug(
                    "  Locked Total Amount for $CASH dividend/fees as it was pre-filled."
                )
            # For $CASH buy/sell (deposit/withdrawal), total is always Qty * 1,
            # so total_amount_locked_by_user remains False to allow _auto_calculate_total to run.
            elif is_cash_symbol_edit and current_tx_type_lower in [
                "buy",
                "sell",
                "deposit",
                "withdrawal",
            ]:
                logging.debug(
                    "  Total Amount for $CASH buy/sell/deposit/withdrawal will be auto-calculated."
                )

        # --- Call _update_field_states AFTER all fields are populated ---
        # This ensures correct enabling/disabling based on the pre-filled type and symbol.
        self._update_field_states(
            self.type_combo.currentText(), self.symbol_edit.text()
        )
        logging.debug("  Called _update_field_states after populating all fields.")

        # --- Reconnect signals ---
        for widget_signal, slot_func in signals_to_disconnect:
            try:
                widget_signal.connect(slot_func)
            except Exception as e_reconnect:  # More general catch
                logging.debug(
                    f"Error reconnecting signal after pre-fill: {widget_signal} to {slot_func}. Error: {e_reconnect}"
                )

        logging.debug("Finished populating AddTransactionDialog for edit.")

    @Slot(str)  # Added Slot decorator for consistency if used elsewhere via signal
    def _update_field_states_wrapper(self, tx_type: str):
        """
        Wrapper to call _update_field_states with the current symbol text
        when the transaction type changes.
        """
        current_symbol = self.symbol_edit.text()
        logging.debug(
            f"_update_field_states_wrapper (type changed): tx_type='{tx_type}', current_symbol='{current_symbol}'"
        )
        self._update_field_states(tx_type, current_symbol)

    @Slot(str)  # Added Slot decorator
    def _update_field_states_wrapper_symbol(self, symbol: str):
        """
        Wrapper to call _update_field_states with the current transaction type
        when the symbol text changes.
        """
        current_tx_type = self.type_combo.currentText()
        logging.debug(
            f"_update_field_states_wrapper_symbol (symbol changed): symbol='{symbol}', current_tx_type='{current_tx_type}'"
        )
        self._update_field_states(current_tx_type, symbol)


# --- Main Application Window ---
