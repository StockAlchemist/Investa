import pandas as pd
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QTabWidget,
    QWidget,
    QFormLayout,
    QScrollArea,
    QGroupBox,
    QLabel,
    QTextEdit,
    QDialogButtonBox,
    QTableView,
    QComboBox,
    QHBoxLayout,
    QHeaderView
)
from PySide6.QtGui import QFont
from PySide6.QtCore import Qt
from typing import Dict, Any, Optional, Tuple

# Assuming PandasModel will be imported from main_gui or its own module later
# from main_gui import PandasModel
# Assuming finutils will be available in the path
from finutils import (
    format_large_number_display,
    format_currency_value,
    format_percentage_value,
    format_float_with_commas,
    format_integer_with_commas
)
import logging # For logging within the dialog if any
from datetime import datetime # For date formatting if needed directly

# It seems PandasModel is defined in main_gui.py.
# For now, this dialog will expect PandasModel to be passed or imported from where it's defined.
# If PandasModel is also being refactored, this import will change.
# For the purpose of this step, we are only moving FundamentalDataDialog.
# We will need to adjust main_gui.py to import this dialog,
# and ensure this dialog can still access PandasModel.

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

        # Critical: PandasModel needs to be accessible.
        # This will likely cause an error if PandasModel is not imported or passed.
        # For now, this is a placeholder to show where it's used.
        # The actual PandasModel definition is NOT moved into this file.
        if not hasattr(self, 'PandasModel'): # Check if PandasModel is somehow already available
            if parent and hasattr(parent, 'PandasModel'):
                self.PandasModel = parent.PandasModel # Try to get from parent if possible (e.g. main_gui.PandasModel)
            else:
                # This will cause a NameError if not resolved by imports/passing
                logging.error("PandasModel is not available to FundamentalDataDialog.")
                # raise NameError("PandasModel is not defined and cannot be accessed.") # Or handle gracefully

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

    def _dialog_format_value(self, key, value):
        if value is None or pd.isna(value):
            return "N/A"

        currency_symbol_for_formatting = self._get_dialog_currency_symbol()

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
            "dividendYield",
            "trailingAnnualDividendYield",
            "fiveYearAvgDividendYield",
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

        if isinstance(value, (int, float)):  # Common check for most numeric types
            if key in LARGE_CURRENCY_KEYS:
                return format_large_number_display(
                    value, currency_symbol_for_formatting
                )
            elif key in PERCENTAGE_KEYS_AS_FACTORS:
                return format_percentage_value(value, decimals=2)
            elif key in PERCENTAGE_KEYS_ALREADY_PERCENTAGES:
                return format_percentage_value(
                    value, decimals=2
                )  # Already a percentage
            elif key in CURRENCY_PER_SHARE_OR_PRICE_KEYS:
                return format_currency_value(
                    value, currency_symbol_for_formatting, decimals=2
                )
            elif key in RATIO_KEYS_NO_CURRENCY:
                return format_float_with_commas(value, decimals=2)  # No currency symbol
            elif key in INTEGER_COUNT_KEYS:
                return format_integer_with_commas(value)
            elif key == "sharesShortPreviousMonthDate":  # Timestamp
                try:
                    return datetime.fromtimestamp(value).strftime("%Y-%m-%d")
                except:
                    return str(value)  # Fallback
            else:  # General fallback for other numeric values not explicitly categorized
                # This is a catch-all. If it's a financial value from yfinance, it likely needs currency.
                # If it's a ratio we missed, it needs no currency.
                # Defaulting to currency for unknown numeric yfinance fields is often safer.
                logging.debug(
                    f"Key '{key}' not in specific format lists, formatting as currency by default."
                )
                return format_currency_value(
                    value, currency_symbol_for_formatting, decimals=2
                )
        elif isinstance(value, str):  # Handle strings directly
            return value  # Return the string as is
        else:
            return str(value)

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

        # This is where PandasModel is instantiated.
        # It must be available in the scope, either via import or passed from parent.
        if hasattr(self, 'PandasModel'):
            model = self.PandasModel(display_df_for_model, parent=self._parent_app, log_mode=True)
            table_view.setModel(model)
        elif self._parent_app and hasattr(self._parent_app, 'PandasModel'):
             model = self._parent_app.PandasModel(display_df_for_model, parent=self._parent_app, log_mode=True)
             table_view.setModel(model)
        else:
            logging.error("FundamentalDataDialog: PandasModel not found to create table model.")
            # Fallback: create a simple label or leave table empty
            error_label = QLabel("Error: Table display component (PandasModel) not loaded.")
            tab_layout.addWidget(error_label)
            # Do not add table_view if model creation failed
            period_combo.setEnabled(False) # Disable selector if table cannot be shown

        if not (initial_df_to_display is None or initial_df_to_display.empty) and table_view.model():
            table_view.resizeColumnsToContents()
            try:
                metric_col_idx = display_df_for_model.columns.get_loc("Metric")
                table_view.setColumnWidth(metric_col_idx, 250)
            except KeyError:
                pass

        if table_view.model(): # Only add if model was set
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
        current_model = table_view.model()
        # Ensure current_model is an instance of the expected PandasModel type
        # This check might need adjustment depending on how PandasModel is made available.
        # For now, assuming it's a known type if it exists.
        if not current_model or not hasattr(current_model, 'updateData'): # Basic check
            logging.error("Table model is not a valid PandasModel instance. Cannot update.")
            return

        df_to_display = None
        current_period_type_update = ""
        if period_text == "Annual":
            df_to_display = annual_df
            current_period_type_update = "Annual"
        elif period_text == "Quarterly":
            df_to_display = quarterly_df
            current_period_type_update = "Quarterly"

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
            layout_overview.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),  # Use self method
            )
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

            layout_valuation.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),
            )
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
            layout_dividend.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),
            )
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
            layout_price_stats.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),
            )
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
            layout_financial_highlights.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),
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
            layout_share_stats.addRow(
                QLabel(f"<b>{friendly_name}:</b>"),
                QLabel(self._dialog_format_value(key, value)),
            )
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

        # This is where PandasModel is instantiated.
        if hasattr(self, 'PandasModel'):
            model = self.PandasModel(display_df_for_model, parent=self, log_mode=True)
            table_view.setModel(model)
        elif self._parent_app and hasattr(self._parent_app, 'PandasModel'):
             model = self._parent_app.PandasModel(display_df_for_model, parent=self, log_mode=True)
             table_view.setModel(model)
        else:
            logging.error("FundamentalDataDialog KeyRatios: PandasModel not found.")
            # Fallback: create a simple label or leave table empty
            error_label = QLabel("Error: Table display component (PandasModel) not loaded.")
            tab_layout.addWidget(error_label)
            # Do not add table_view if model creation failed

        if not (ratios_df is None or ratios_df.empty) and table_view.model():
            table_view.resizeColumnsToContents()
            try:
                ratio_col_idx = display_df_for_model.columns.get_loc("Financial Ratio")
                table_view.setColumnWidth(
                    ratio_col_idx, 200
                )  # Make ratio name column wider
            except KeyError:
                pass

        if table_view.model(): # Only add if model was set
            tab_layout.addWidget(table_view)
