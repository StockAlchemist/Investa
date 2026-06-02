# Auto-generated from main_gui.py modularization
import os
import sys

from config import (
    COLOR_BG_DARK,
    COLOR_BG_HEADER_LIGHT,
    COLOR_TEXT_DARK,
    COLOR_TEXT_SECONDARY,
    COLOR_ACCENT_TEAL,
    COLOR_BORDER_LIGHT,
    COLOR_BORDER_DARK,
    COLOR_GAIN,
    COLOR_LOSS,
)

# Qt-free presentation defaults live in display_config; re-export them here so
# existing GUI imports (`from utils import get_column_definitions`, ...) keep working.
from display_config import (  # noqa: F401
    get_column_definitions,
    DEFAULT_GRAPH_START_DATE,
    DEFAULT_GRAPH_END_DATE,
)

from PySide6.QtGui import QColor

QCOLOR_GAIN = QColor(COLOR_GAIN)


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        # For development, __file__ is the path to the current script.
        # os.path.dirname(__file__) gives the directory of the script.
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


# For DEFAULT_CSV, it's often better to let the user select it or store its path in config,
# rather than bundling a specific "my_transactions.csv".
# If "my_transactions.csv" is an example/template, then resource_path could be used.
# DEFAULT_CSV is now imported from config.py

# --- User-specific file paths using QStandardPaths ---
# APP_NAME_FOR_QT was "Investa". config.APP_NAME ("Investa") will be used instead.

# CONFIG_FILE and MANUAL_PRICE_FILE will be initialized as instance attributes
# in PortfolioApp.__init__ after QApplication is fully configured.

# DEFAULT_API_KEY is now imported from config.py

# --- Graph Defaults ---
# DEFAULT_GRAPH_START_DATE / DEFAULT_GRAPH_END_DATE are defined in display_config
# and re-exported above (kept importable from utils for GUI back-compat).
# DEFAULT_GRAPH_INTERVAL is imported from config.py
# DEFAULT_GRAPH_BENCHMARKS is imported from config.py
# BENCHMARK_MAPPING is imported from config.py
# BENCHMARK_OPTIONS_DISPLAY is imported from config.py

# --- Theme Colors ---
# Define color palette for styling the UI (Minimal Theme)
# Hex color strings (COLOR_BG_DARK, etc.) are now imported from config.py
# QColor objects will be defined using these imported hex strings.

# Convert hex colors to QColor objects for easier use in Qt palettes
# These are the base light theme colors. Themed versions will be created in __init__.
QCOLOR_GAIN = QColor(COLOR_GAIN)
QCOLOR_LOSS = QColor(COLOR_LOSS)
QCOLOR_TEXT_DARK = QColor(COLOR_TEXT_DARK)
QCOLOR_TEXT_SECONDARY = QColor(COLOR_TEXT_SECONDARY)
# Add other base QColor objects if they are directly used and need theming
QCOLOR_BG_DARK = QColor(
    COLOR_BG_DARK
)  # Used for QWidget background and figure backgrounds
QCOLOR_BG_HEADER_LIGHT = QColor(COLOR_BG_HEADER_LIGHT)  # Used for some headers/frames
QCOLOR_BORDER_LIGHT = QColor(COLOR_BORDER_LIGHT)  # General light borders
QCOLOR_BORDER_DARK = QColor(
    COLOR_BORDER_DARK
)  # General dark borders (like table header bottom)
QCOLOR_ACCENT_TEAL = QColor(COLOR_ACCENT_TEAL)  # Primary accent for lines/etc.


# --- Fallback QColor Constants (for dialogs if themed attributes are not on parent) ---
FALLBACK_QCOLOR_GAIN = QCOLOR_GAIN
FALLBACK_QCOLOR_LOSS = QCOLOR_LOSS
FALLBACK_QCOLOR_TEXT_DARK = QCOLOR_TEXT_DARK
FALLBACK_QCOLOR_TEXT_SECONDARY = QCOLOR_TEXT_SECONDARY
FALLBACK_QCOLOR_BG_DARK = QCOLOR_BG_DARK
FALLBACK_QCOLOR_BG_HEADER_LIGHT = QCOLOR_BG_HEADER_LIGHT
FALLBACK_QCOLOR_BORDER_LIGHT = QCOLOR_BORDER_LIGHT
FALLBACK_QCOLOR_BORDER_DARK = QCOLOR_BORDER_DARK
FALLBACK_QCOLOR_ACCENT_TEAL = QCOLOR_ACCENT_TEAL
# Add any other FALLBACK_QCOLOR_ needed by dialogs, mirroring the global QCOLOR_ definitions


# --- Helper Classes for Background Processing ---

from PySide6.QtWidgets import QStyledItemDelegate  # noqa: E402
from PySide6.QtGui import QColor  # noqa: E402


class GroupHeaderDelegate(QStyledItemDelegate):
    def __init__(self, parent=None, theme="light"):
        super().__init__(parent)
        self.theme = theme

    def paint(self, painter, option, index):
        super().paint(painter, option, index)
