# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          historical_holdings_tab.py
 Purpose:       GUI for the Historical Holdings tab

 Author:        Jules
 Created:       30/07/2025
 Copyright:     (c) Jules 2025
 Licence:       MIT
-------------------------------------------------------------------------------
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
)
import pyqtgraph as pg


class HistoricalHoldingsTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.label = QLabel("Historical Holdings")
        self.layout.addWidget(self.label)
        self.percentage_plot = pg.PlotWidget()
        self.layout.addWidget(self.percentage_plot)
        self.value_plot = pg.PlotWidget()
        self.layout.addWidget(self.value_plot)
