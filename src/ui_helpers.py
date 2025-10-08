import logging
from PySide6.QtWidgets import QMenu, QMessageBox
from PySide6.QtCore import QPoint
from PySide6.QtGui import QAction
from config import BENCHMARK_OPTIONS_DISPLAY


class UiHelpersMixin:
    """Reusable UI helper methods for menus and positioning."""

    # --- Centralized status and message helpers ---
    def set_status(self, message: str) -> None:
        """Safely set status text if the status label exists."""
        try:
            if hasattr(self, "status_label") and getattr(self, "status_label", None):
                self.status_label.setText(message)
            else:
                logging.debug(f"STATUS: {message}")
        except Exception as e:
            logging.error(f"Failed to set status text: {e}")

    def show_error(
        self, message: str, *, popup: bool = False, title: str = "Error"
    ) -> None:
        """Log, surface in status bar, and optionally show a popup for errors."""
        logging.error(message)
        self.set_status(f"Error: {message}")
        if popup:
            try:
                QMessageBox.critical(self, title, message)
            except Exception as e:
                logging.error(f"Failed to show error popup: {e}")

    def show_warning(
        self, message: str, *, popup: bool = False, title: str = "Warning"
    ) -> None:
        """Log a warning, set status, and optionally show a warning popup."""
        logging.warning(message)
        self.set_status(message)
        if popup:
            try:
                QMessageBox.warning(self, title, message)
            except Exception as e:
                logging.error(f"Failed to show warning popup: {e}")

    def show_info(
        self, message: str, *, popup: bool = False, title: str = "Information"
    ) -> None:
        """Log info, set status, and optionally show an information popup."""
        logging.info(message)
        self.set_status(message)
        if popup:
            try:
                QMessageBox.information(self, title, message)
            except Exception as e:
                logging.error(f"Failed to show info popup: {e}")

    def _exec_menu_below_widget(self, widget, menu: QMenu) -> None:
        """Executes a QMenu aligned directly below the given widget, if available."""
        if not widget:
            logging.warning("Warn: Menu anchor widget not available.")
            return
        button_pos = widget.mapToGlobal(QPoint(0, widget.height()))
        menu.exec(button_pos)

    def _build_benchmark_menu_actions(self, menu: QMenu) -> None:
        """Populate the given menu with benchmark toggle actions."""
        for display_name in BENCHMARK_OPTIONS_DISPLAY:
            action = QAction(display_name, self)
            action.setCheckable(True)
            action.setChecked(display_name in getattr(self, "selected_benchmarks", []))
            action.triggered.connect(
                lambda checked, name=display_name: self.toggle_benchmark_selection(
                    name, checked
                )
            )
            menu.addAction(action)

    def _build_account_menu_actions(self, menu: QMenu) -> None:
        """Populate the given menu with account toggle actions, plus select/deselect all."""
        available = getattr(self, "available_accounts", [])
        selected = getattr(self, "selected_accounts", [])

        action_all = QAction("Select/Deselect All", self)
        is_all_selected = len(selected) == len(available)
        action_all.triggered.connect(
            lambda: self._toggle_all_accounts(not is_all_selected)
        )
        menu.addAction(action_all)
        menu.addSeparator()

        for account_name in available:
            action = QAction(account_name, self)
            action.setCheckable(True)
            action.setChecked(account_name in selected)
            action.triggered.connect(
                lambda checked, name=account_name: self.toggle_account_selection(
                    name, checked
                )
            )
            menu.addAction(action)
