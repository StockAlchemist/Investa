import sys
import os

# --- Add src directory to sys.path ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest
import pandas as pd
from PySide6.QtCore import Qt, QModelIndex, QObject
from PySide6.QtGui import QColor
from models import PandasModel

# --- Mocks ---
class MockParent(QObject):
    def __init__(self):
        super().__init__()
        self.QCOLOR_TEXT_PRIMARY_THEMED = QColor("black")
        self.QCOLOR_GAIN_THEMED = QColor("green")
        self.QCOLOR_LOSS_THEMED = QColor("red")
        self.QCOLOR_ACCENT_THEMED = QColor("blue")
        self.QCOLOR_HEADER_BACKGROUND_THEMED = QColor("gray")
        self.currency_combo = type("obj", (object,), {"currentText": lambda: "USD"})
    
    def _get_currency_symbol(self, get_name=False):
        return "$"

# --- Fixtures ---
@pytest.fixture
def sample_df():
    data = {
        "Symbol": ["AAPL", "MSFT", "GOOG"],
        "Quantity": [10.0, 5.0, -2.0],
        "Price": [150.0, 200.0, 100.0],
        "Total G/L": [500.0, -100.0, 0.0],
        "Account": ["Acc1", "Acc2", "Acc1"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def model(sample_df):
    parent = MockParent()
    return PandasModel(sample_df, parent=parent)

# --- Tests ---

def test_row_column_count(model, sample_df):
    assert model.rowCount() == 3
    assert model.columnCount() == 5

def test_data_display_role(model):
    # Test string display
    index = model.index(0, 0) # AAPL
    assert model.data(index, Qt.DisplayRole) == "AAPL"
    
    # Test numeric display (Quantity)
    index = model.index(0, 1) # 10.0
    assert model.data(index, Qt.DisplayRole) == "10.0000" # Quantity format
    
    # Test numeric display (Price - generic fallback)
    index = model.index(0, 2) # 150.0
    assert model.data(index, Qt.DisplayRole) == "$150.00" # Currency format via parent

def test_data_alignment_role(model):
    # Symbol (Left)
    index = model.index(0, 0)
    assert model.data(index, Qt.TextAlignmentRole) == int(Qt.AlignLeft | Qt.AlignVCenter)
    
    # Quantity (Right)
    index = model.index(0, 1)
    assert model.data(index, Qt.TextAlignmentRole) == int(Qt.AlignRight | Qt.AlignVCenter)

def test_data_foreground_role(model):
    # Total G/L (Gain -> Green)
    index = model.index(0, 3) # 500.0
    color = model.data(index, Qt.ForegroundRole)
    assert color == model._gain_color
    
    # Total G/L (Loss -> Red)
    index = model.index(1, 3) # -100.0
    color = model.data(index, Qt.ForegroundRole)
    assert color == model._loss_color

def test_header_data(model):
    assert model.headerData(0, Qt.Horizontal, Qt.DisplayRole) == "Symbol"
    assert model.headerData(0, Qt.Vertical, Qt.DisplayRole) == "1"

def test_sort(model):
    # Sort by Symbol Ascending (col 0)
    model.sort(0, Qt.AscendingOrder)
    assert model._data.iloc[0]["Symbol"] == "AAPL"
    assert model._data.iloc[2]["Symbol"] == "MSFT"
    
    # Sort by Symbol Descending
    model.sort(0, Qt.DescendingOrder)
    assert model._data.iloc[0]["Symbol"] == "MSFT"
    
    # Sort by Quantity (Numeric) Ascending (col 1)
    model.sort(1, Qt.AscendingOrder)
    assert model._data.iloc[0]["Quantity"] == -2.0 # GOOG
