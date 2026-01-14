import pytest
import pandas as pd
from datetime import date, timedelta
from src.config_manager import ConfigManager
import os
import json

# Mocking the helper function we plan to write
from src.server.api import calculate_mtd_average_daily_balance 
# Note: calculate_mtd_average_daily_balance doesn't exist yet, so this test will fail initially or we need to stub it here if we want to test logic isolation first.
# For TDD, I will define a local version or mock to verify logic, then move it to src.

def mock_calculate_mtd_adb(current_cash, mtd_transactions, today_date):
    """
    Reconstructs daily balances backwards from current cash.
    """
    # 1. Create a date range from 1st to Today
    start_of_month = date(today_date.year, today_date.month, 1)
    days_in_month_so_far = (today_date - start_of_month).days + 1
    
    # Map dates to net change
    daily_changes = {}
    if not mtd_transactions.empty:
        # Filter for cash-affecting types if needed, but assuming mtd_transactions are pre-filtered or we filter here
        # For simplicity, assume all relevant.
        # Simplify: Group by date
        # Invert the flow: To go BACKWARDS from current, we SUBTRACT inflows and ADD outflows.
        # But easier: Reconstruct start balance?
        # Start + Changes = Current
        # Start = Current - Changes
        
        # Actually easier to just map each day's balance.
        # balance[today] = current_cash
        # balance[yesterday] = balance[today] - (Transactions on Today)
        pass 

    # Let's write the actual logic we intend to implement to verify it passes scenarios
    
    # Group by Date
    if mtd_transactions.empty:
        changes_by_date = {}
    else:
        # Assuming 'Total Amount' is signed correctly (positive for inflow, negative for outflow)
        # Note: In Investa, Buy is negative Total Amount usually, Sell is positive. 
        # But we need to check how it's stored. 
        # Let's assume standard signed amounts: + = cash increases, - = cash decreases.
        changes_by_date = mtd_transactions.groupby("Date")["Total Amount"].sum().to_dict()
        # Convert keys to date objects if needed
        changes_by_date = {k.date() if hasattr(k, 'date') else k: v for k,v in changes_by_date.items()}

    daily_balances = []
    
    # We walk BACKWARDS from Today
    running_balance = current_cash
    
    # Range: Today down to 1st
    # inclusive of today? Yes.
    # inclusive of 1st? Yes.
    
    for d_idx in range(days_in_month_so_far):
        # Current day we are looking at
        lookback_date = today_date - timedelta(days=d_idx)
        
        # The running_balance represents the END OF DAY balance for lookback_date
        daily_balances.append(running_balance)
        
        # Before moving to yesterday (next iteration), adjust balance
        # If we had transactions ON lookback_date, we remove them to get "start of day" which is "end of yesterday"
        change_on_day = changes_by_date.get(lookback_date, 0.0)
        
        # If we had +500 inflow today, yesterday's end balance was (Current - 500)
        running_balance = running_balance - change_on_day
        
    return sum(daily_balances) / len(daily_balances)

# --- TESTS ---

def test_config_manager_settings(tmp_path):
    """Verify ConfigManager can save/load new fields."""
    d = tmp_path / "appdata"
    d.mkdir()
    cm = ConfigManager(str(d))
    
    # Should be empty defaults initially
    assert "account_interest_rates" not in cm.manual_overrides or cm.manual_overrides["account_interest_rates"] == {}
    
    # Update
    new_data = cm.manual_overrides.copy()
    new_data["account_interest_rates"] = {"TestAcc": 0.05}
    new_data["interest_free_thresholds"] = {"TestAcc": 1000.0}
    
    cm.save_manual_overrides(new_data)
    
    # Reload
    cm2 = ConfigManager(str(d))
    assert cm2.manual_overrides["account_interest_rates"]["TestAcc"] == 0.05
    assert cm2.manual_overrides["interest_free_thresholds"]["TestAcc"] == 1000.0

def test_calculate_mtd_adb():
    """Test Average Daily Balance Logic."""
    today = date(2025, 1, 10) # 10th of Jan
    current_cash = 10000.0
    
    # Case 1: No transactions. Balance was 10,000 all 10 days.
    empty_tx = pd.DataFrame(columns=["Date", "Total Amount"])
    adb = mock_calculate_mtd_adb(current_cash, empty_tx, today)
    assert adb == 10000.0
    
    # Case 2: Deposit of 5000 on Jan 5th.
    # Means:
    # Jan 1-4: 5000
    # Jan 5-10: 10000 (Current)
    # Days: 4 days @ 5000, 6 days @ 10000 = (20000 + 60000) / 10 = 8000
    
    df = pd.DataFrame([
        {"Date": pd.Timestamp("2025-01-05"), "Total Amount": 5000.0}
    ])
    adb = mock_calculate_mtd_adb(current_cash, df, today)
    assert adb == 8000.0
    
    # Case 3: Withdrawal of 2000 on Jan 9th.
    # Logic: Current is 10000.
    # Jan 9-10 balances: 10000
    # Jan 1-8 balances: 12000 (Before withdrawal)
    # Days: 8 days @ 12000, 2 days @ 10000 = (96000 + 20000) / 10 = 11600
    df2 = pd.DataFrame([
        {"Date": pd.Timestamp("2025-01-09"), "Total Amount": -2000.0}
    ])
    adb = mock_calculate_mtd_adb(current_cash, df2, today)
    assert adb == 11600.0

