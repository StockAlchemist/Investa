import pandas as pd
import numpy as np
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QWidget,
    QSizePolicy
)
from PySide6.QtGui import QFont # QColor is used via parent's themed colors
from PySide6.QtCore import Qt, Slot # Slot might not be strictly needed here
import logging
from datetime import date

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

try:
    from config import (
        CHART_DPI,
        MPLCURSORS_AVAILABLE,
        FALLBACK_QCOLOR_BG_DARK,
        FALLBACK_QCOLOR_TEXT_DARK,
        FALLBACK_QCOLOR_TEXT_SECONDARY,
        FALLBACK_QCOLOR_ACCENT_TEAL,
        FALLBACK_QCOLOR_BORDER_LIGHT,
        FALLBACK_QCOLOR_BORDER_DARK,
        FALLBACK_QCOLOR_LOSS # For error text color
    )
except ImportError:
    logging.error("SymbolChartDialog: Could not import constants from config. Using hardcoded fallbacks.")
    CHART_DPI = 96
    MPLCURSORS_AVAILABLE = False
    FALLBACK_QCOLOR_BG_DARK = "#2E2E2E" # Example, should be QColor if used directly
    FALLBACK_QCOLOR_TEXT_DARK = "#E0E0E0"
    FALLBACK_QCOLOR_TEXT_SECONDARY = "#B0B0B0"
    FALLBACK_QCOLOR_ACCENT_TEAL = "#4DB6AC"
    FALLBACK_QCOLOR_BORDER_LIGHT = "#4E4E4E"
    FALLBACK_QCOLOR_BORDER_DARK = "#3E3E3E"
    FALLBACK_QCOLOR_LOSS = "#F44336"


try:
    import mplcursors
except ImportError:
    # This is handled by MPLCURSORS_AVAILABLE, but good to have a local try-except
    MPLCURSORS_AVAILABLE = False


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
            parent (QWidget, optional): Parent widget. Defaults to None.
        """
        super().__init__(parent)
        self._parent_app = parent # Store for theme access
        self._symbol = symbol
        self._local_currency_symbol = "$"  # Default

        if parent and hasattr(parent, "_get_currency_for_symbol") and hasattr(parent, "_get_currency_symbol"):
            local_curr_code = parent._get_currency_for_symbol(symbol)
            if local_curr_code:
                self._local_currency_symbol = parent._get_currency_symbol(currency_code=local_curr_code)
            elif hasattr(parent, 'config'): # Fallback to parent's display currency if symbol specific fails
                 self._local_currency_symbol = parent._get_currency_symbol(currency_code=parent.config.get("display_currency", "USD"))


        self.setWindowTitle(f"Historical Price Chart: {symbol}")
        self.setMinimumSize(700, 500)  # Good default size
        if parent and hasattr(parent, "font"):
            self.setFont(parent.font())
        else:
            self.setFont(QFont("Arial", 9))


        # --- Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(2)

        # --- Chart Area ---
        chart_widget = QWidget()
        chart_layout = QVBoxLayout(chart_widget)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(1)

        self.figure = Figure(figsize=(7, 4.5), dpi=CHART_DPI)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setObjectName("SymbolChartCanvas")
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        try:
            self.toolbar = NavigationToolbar(self.canvas, chart_widget, coordinates=True)
            self.toolbar.setObjectName("SymbolChartToolbar")
            logging.debug(f"Created standard NavigationToolbar for {self._symbol} chart.")
        except Exception as e_tb:
            logging.error(f"Error creating standard symbol chart toolbar: {e_tb}")
            self.toolbar = None

        chart_layout.addWidget(self.canvas, 1)
        if self.toolbar:
            chart_layout.addWidget(self.toolbar)

        main_layout.addWidget(chart_widget)

        self._plot_data(
            symbol,
            start_date,
            end_date,
            price_data,
            self._local_currency_symbol,
            display_currency, # This is the currency CODE
        )

        if parent and hasattr(parent, "styleSheet"):
            self.setStyleSheet(parent.styleSheet())

    def _get_themed_color(self, color_attr_name, fallback_hex):
        if self._parent_app and hasattr(self._parent_app, color_attr_name):
            return getattr(self._parent_app, color_attr_name).name()
        return fallback_hex

    def _plot_data(
        self,
        symbol,
        start_date,
        end_date,
        price_data,
        currency_symbol_display,
        currency_code_label, # e.g. USD (not used in current plot, but kept for signature)
    ):
        ax = self.ax
        ax.clear()

        bg_color = self._get_themed_color("QCOLOR_BACKGROUND_THEMED", FALLBACK_QCOLOR_BG_DARK)

        self.figure.patch.set_facecolor(bg_color)
        ax.patch.set_facecolor(bg_color)

        title_color_name = self._get_themed_color("QCOLOR_TEXT_PRIMARY_THEMED", FALLBACK_QCOLOR_TEXT_DARK)
        label_color_name = self._get_themed_color("QCOLOR_TEXT_PRIMARY_THEMED", FALLBACK_QCOLOR_TEXT_DARK)
        tick_color_name = self._get_themed_color("QCOLOR_TEXT_SECONDARY_THEMED", FALLBACK_QCOLOR_TEXT_SECONDARY)
        grid_color_name = self._get_themed_color("QCOLOR_BORDER_THEMED", FALLBACK_QCOLOR_BORDER_LIGHT)
        spine_color_name = self._get_themed_color("QCOLOR_BORDER_THEMED", FALLBACK_QCOLOR_BORDER_DARK)
        accent_color_name = self._get_themed_color("QCOLOR_ACCENT_THEMED", FALLBACK_QCOLOR_ACCENT_TEAL)
        error_text_color = self._get_themed_color("QCOLOR_LOSS_THEMED", FALLBACK_QCOLOR_LOSS)


        if price_data is None or price_data.empty or "price" not in price_data.columns:
            ax.text(
                0.5, 0.5, f"No historical data available\nfor {symbol}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12,
                color=tick_color_name # Using tick color for placeholder text
            )
            ax.set_title(f"Price Chart: {symbol}", fontsize=10, weight="bold", color=title_color_name)
            self.canvas.draw()
            return

        try:
            if not isinstance(price_data.index, pd.DatetimeIndex):
                price_data.index = pd.to_datetime(price_data.index)
            price_data = price_data.sort_index()
        except Exception as e:
            logging.error(f"Error processing index for symbol chart {symbol}: {e}")
            ax.text(
                0.5, 0.5, f"Error processing data\nfor {symbol}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12, color=error_text_color
            )
            self.canvas.draw()
            return

        ax.plot(
            price_data.index, price_data["price"],
            label=f"{symbol} Price ({currency_symbol_display})",
            color=accent_color_name
        )

        ax.set_title(f"Historical Price: {symbol}", fontsize=10, weight="bold", color=title_color_name)
        ax.set_ylabel(f"Price ({currency_symbol_display})", fontsize=9, color=label_color_name)
        ax.grid(True, which="major", linestyle="--", linewidth=0.5, color=grid_color_name)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_color(spine_color_name)
        ax.spines["left"].set_color(spine_color_name)
        ax.tick_params(axis="x", colors=tick_color_name, labelsize=8)
        ax.tick_params(axis="y", colors=tick_color_name, labelsize=8)

        formatter = mtick.FormatStrFormatter(f"{currency_symbol_display}%.2f")
        ax.yaxis.set_major_formatter(formatter)

        try:
            pd_start = pd.Timestamp(start_date)
            pd_end = pd.Timestamp(end_date)
            ax.set_xlim(pd_start, pd_end)
        except Exception as e_lim:
            logging.warning(f"Could not set x-limits for symbol chart {symbol}: {e_lim}")
            ax.autoscale(enable=True, axis="x", tight=True)

        self.figure.autofmt_xdate(rotation=15)
        self.figure.tight_layout(pad=0.5)

        if MPLCURSORS_AVAILABLE:
            try:
                cursor = mplcursors.cursor(ax.lines, hover=mplcursors.HoverMode.Transient)
                @cursor.connect("add")
                def on_add(sel, sym=currency_symbol_display):
                    x_dt = mdates.num2date(sel.target[0])
                    date_str = x_dt.strftime("%Y-%m-%d")
                    price_val = sel.target[1]
                    sel.annotation.set_text(f"{date_str}\nPrice: {sym}{price_val:.2f}")
                    sel.annotation.get_bbox_patch().set(facecolor="lightyellow", alpha=0.8, edgecolor="gray")
                    sel.annotation.set_fontsize(8)
            except Exception as e_cursor_sym:
                logging.error(f"Error activating mplcursors for symbol chart {symbol}: {e_cursor_sym}")

        self.canvas.draw()
