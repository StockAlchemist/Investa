
import sys
import os
import pandas as pd
from datetime import date, datetime
from unittest.mock import MagicMock

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Mocking QDate for the test
class MockQDate:
    def __init__(self, d):
        self._date = d
    def toPython(self):
        return self._date
    def __eq__(self, other):
        return self._date == other._date
    def __repr__(self):
        return f"MockQDate({self._date})"

# Mocking the App class
class MockApp:
    def __init__(self):
        self.config = {"selected_accounts": []}
        self.all_transactions_df_cleaned_for_logic = pd.DataFrame()
        self.graph_start_date_edit = MagicMock()
        self.date_preset_combo = MagicMock()
        self.date_preset_combo.currentText.return_value = "Presets..."
        
        # Setup logging mock
        import logging
        logging.basicConfig(level=logging.DEBUG)

    def _update_graph_date_range_for_accounts(self):
        # Copy-paste the relevant logic from main_gui.py for testing
        # or import it if it was a standalone function (it's a method).
        # Since we can't easily import the class and instantiate it without GUI,
        # we will replicate the logic we want to test here.
        
        if (
            not hasattr(self, "all_transactions_df_cleaned_for_logic")
            or self.all_transactions_df_cleaned_for_logic.empty
        ):
            return

        selected_accounts = getattr(self, "selected_accounts", [])
        if not selected_accounts:
             selected_accounts = self.config.get("selected_accounts", [])
        
        df = self.all_transactions_df_cleaned_for_logic

        if selected_accounts:
            if "Account" in df.columns:
                df = df[df["Account"].isin(selected_accounts)]
            else:
                return

        if df.empty:
            return

        if "Date" in df.columns:
            try:
                min_date = df["Date"].min()
                if pd.notna(min_date):
                    if isinstance(min_date, (pd.Timestamp, datetime)):
                        min_date = min_date.date()

                    new_start_date = min_date
                    
                    # Mocking the QDate call
                    current_ui_date = self.graph_start_date_edit.date().toPython()
                    
                    print(f"Current UI Date: {current_ui_date}")
                    print(f"New Start Date: {new_start_date}")

                    # --- THE LOGIC TO TEST (UPDATED) ---
                    should_force_update = False
                    if hasattr(self, "date_preset_combo"):
                        current_preset = self.date_preset_combo.currentText()
                        if current_preset in ["Presets...", "All"]:
                            should_force_update = True

                    if current_ui_date < new_start_date or should_force_update:
                        print(f"Updating date to {new_start_date}")
                        self.graph_start_date_edit.setDate(MockQDate(new_start_date))
                        if hasattr(self, "date_preset_combo"):
                            self.date_preset_combo.setCurrentIndex(0)
                    else:
                        print(f"Kept existing graph start date {current_ui_date}")
                    # -------------------------

            except Exception as e:
                print(f"Error: {e}")

def run_test():
    app = MockApp()
    
    # Setup Data
    # Account A: Long history (2020)
    # Account B: Short history (2024)
    data = [
        {"Account": "Account A", "Date": pd.Timestamp("2020-01-01")},
        {"Account": "Account A", "Date": pd.Timestamp("2025-01-01")},
        {"Account": "Account B", "Date": pd.Timestamp("2024-01-01")},
        {"Account": "Account B", "Date": pd.Timestamp("2025-01-01")},
    ]
    app.all_transactions_df_cleaned_for_logic = pd.DataFrame(data)
    
    print("--- Test Case 1: Long -> Short ---")
    # Initial State: Long Account Selected, Date is 2020
    app.selected_accounts = ["Account B"]
    app.graph_start_date_edit.date.return_value = MockQDate(date(2020, 1, 1))
    
    app._update_graph_date_range_for_accounts()
    
    # Expectation: Update to 2024
    # 2020 < 2024 -> True. Should update.
    
    print("\n--- Test Case 2: Short -> Long (The Bug) ---")
    # Initial State: Short Account Selected (or previously selected), Date is 2024
    app.selected_accounts = ["Account A"]
    app.graph_start_date_edit.date.return_value = MockQDate(date(2024, 1, 1))
    
    app._update_graph_date_range_for_accounts()
    
    # Expectation: Update to 2020
    # 2024 < 2020 -> False. 
    # Current code will NOT update. This confirms the bug.

if __name__ == "__main__":
    run_test()
