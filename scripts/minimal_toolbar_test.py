# minimal_toolbar_test.py
import sys
import os
from PySide6.QtWidgets import QApplication, QMainWindow, QToolBar, QMessageBox
from PySide6.QtGui import QAction, QIcon, QPixmap, QColor
from PySide6.QtCore import QSize


# Helper to find resources (simplified for this test)
def resource_path(relative_path):
    base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


def run_test():
    app = QApplication(sys.argv)
    app.setOrganizationName("TestOrg")
    app.setApplicationName("TestApp")

    main_window = QMainWindow()
    main_window.setWindowTitle("Minimal Toolbar Test")

    toolbar = main_window.addToolBar("Test Toolbar")
    if not isinstance(toolbar, QToolBar):
        print("CRITICAL: Toolbar is not a QToolBar instance!")
        return

    toolbar.setIconSize(QSize(24, 24))  # Set a common icon size

    print(f"Toolbar instance: {toolbar}")

    # Test 1: QIcon.fromTheme
    action_from_theme = QAction("From Theme", main_window)
    icon_from_theme = QIcon.fromTheme("document-open")
    if icon_from_theme.isNull():
        print("ICON_FROM_THEME: QIcon.fromTheme('document-open') is NULL.")
    else:
        print("ICON_FROM_THEME: QIcon.fromTheme('document-open') is VALID.")
    action_from_theme.setIcon(icon_from_theme)

    returned_action1 = toolbar.addAction(action_from_theme)
    if returned_action1 is action_from_theme:  # Correct check
        print("ADD_ACTION (From Theme): SUCCESS")
    else:
        print(
            f"ADD_ACTION (From Theme): FAILED. addAction returned: {returned_action1}"
        )

    # Test 2: QStyle.standardIcon
    action_sp_icon = QAction("SP Icon", main_window)
    icon_sp = main_window.style().standardIcon(
        QApplication.style().StandardPixmap.SP_DialogApplyButton
    )  # Correct way to get SP
    if icon_sp.isNull():
        print("ICON_SP: Standard icon SP_DialogApplyButton is NULL.")
    else:
        print("ICON_SP: Standard icon SP_DialogApplyButton is VALID.")
    action_sp_icon.setIcon(icon_sp)

    returned_action2 = toolbar.addAction(action_sp_icon)
    if returned_action2 is action_sp_icon:
        print("ADD_ACTION (SP Icon): SUCCESS")
    else:
        print(f"ADD_ACTION (SP Icon): FAILED. addAction returned: {returned_action2}")

    # Test 3: Icon from local file (create a dummy 'test_icon.png')
    action_file_icon = QAction("File Icon", main_window)
    # Create a dummy PNG for testing
    dummy_icon_path = resource_path("test_icon.png")
    if not os.path.exists(dummy_icon_path):
        try:
            pix = QPixmap(24, 24)
            pix.fill(QColor("red"))
            pix.save(dummy_icon_path, "PNG")
            print(f"Created dummy icon: {dummy_icon_path}")
        except Exception as e:
            print(f"Could not create dummy icon: {e}")

    if os.path.exists(dummy_icon_path):
        icon_file = QIcon(dummy_icon_path)
        if icon_file.isNull():
            print(f"ICON_FILE: Icon from '{dummy_icon_path}' is NULL.")
        else:
            print(f"ICON_FILE: Icon from '{dummy_icon_path}' is VALID.")
        action_file_icon.setIcon(icon_file)
    else:
        print(f"ICON_FILE: '{dummy_icon_path}' does not exist. Skipping icon set.")

    returned_action3 = toolbar.addAction(action_file_icon)
    if returned_action3 is action_file_icon:
        print("ADD_ACTION (File Icon): SUCCESS")
    else:
        print(f"ADD_ACTION (File Icon): FAILED. addAction returned: {returned_action3}")

    # Test 4: Action with NO icon
    action_no_icon = QAction("No Icon Action", main_window)
    returned_action4 = toolbar.addAction(action_no_icon)
    if returned_action4 is action_no_icon:
        print("ADD_ACTION (No Icon): SUCCESS")
    else:
        print(f"ADD_ACTION (No Icon): FAILED. addAction returned: {returned_action4}")

    main_window.resize(400, 300)
    main_window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    run_test()
