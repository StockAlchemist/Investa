
import pandas as pd
from datetime import date
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from finutils import get_cash_flows_for_mwr

def test_mwr_transfer_logic():
    # Setup Data
    # Account A: Buy AAPL ($1000), Transfer Out ($500) to B.
    # Account B: Transfer In ($500).
    
    rows = [
        # Date, Account, Symbol, Type, Qty, Price, Amount, To Account
        {
            "Date": pd.Timestamp("2023-01-01"),
            "Account": "A",
            "Symbol": "AAPL",
            "Type": "Start", # Ignored usually
            "Quantity": 0, "Price/Share": 0, "Total Amount": 0,
            "Local Currency": "USD"
        },
        # A buys AAPL. Outflow 1000.
        {
            "Date": pd.Timestamp("2023-01-02"),
            "Account": "A",
            "Symbol": "AAPL",
            "Type": "Buy",
            "Quantity": 10, "Price/Share": 100, "Total Amount": 1000,
            "Local Currency": "USD"
        },
        # Transfer A -> B. $500.
        # usually represented as two rows in DB? Or one?
        # In Investa, transfers are often Single Row with "To Account".
        {
            "Date": pd.Timestamp("2023-01-05"),
            "Account": "A",
            "Symbol": "$CASH",
            "Type": "Transfer",
            "Quantity": 500, "Price/Share": 1, "Total Amount": 500,
            "Local Currency": "USD",
            "To Account": "b" # Lowercase to test case-insensitivity
        }
    ]
    
    df = pd.DataFrame(rows)
    df["original_index"] = df.index
    
    # CASE 1: MWR for Account A
    # Expect: -1000 (Buy), +500 (Transfer Out/Withdrawal)
    # End Value: Suppose AAPL is worth 1200. + Cash 0. Total 1200.
    # MWR Flows: -1000, +500. Final +1200.
    
    dates_a, flows_a = get_cash_flows_for_mwr(
        account_transactions=df[df["Account"] == "A"], # Filter for A's rows
        final_account_market_value=1200.0,
        end_date=date(2023, 12, 31),
        target_currency="USD",
        fx_rates={"USD": 1.0},
        display_currency="USD",
        include_accounts=["A"] # Scope = A
    )
    
    print("\n--- CASE 1: Account A Only ---")
    print(f"Dates: {dates_a}")
    print(f"Flows: {flows_a}")
    
    # Verify A
    # Flow 1: Buy -1000. Correct.
    # Flow 2: Transfer Out +500. Correct.
    # Final: +1200. 
    # Check values
    assert flows_a[0] == -1000.0, f"Expected -1000, got {flows_a[0]}"
    assert flows_a[1] == 500.0, f"Expected 500 (Withdrawal), got {flows_a[1]}"
    assert flows_a.pop() == 1200.0, "Expected Final Value"
    
    
    # CASE 2: MWR for Account B
    # Expect: Transfer In 500 (Deposit). -500.
    # End Value: 500 Cash. +500.
    
    # IMPORTANT: The DataFrame passed to get_cash_flows_for_mwr usually contains rows related to the account.
    # If we filter by `Account == B`, we might get nothing if the row is `Account=A, To=B`.
    # But `portfolio_analyzer` now filters by `Account == B OR To Account == B`.
    
    # Filter case-insensitively for the test setup
    df_b = df[
        (df["Account"] == "B") | 
        (df["To Account"].astype(str).str.upper() == "B")
    ]
    dates_b, flows_b = get_cash_flows_for_mwr(
        account_transactions=df_b,
        final_account_market_value=500.0,
        end_date=date(2023, 12, 31),
        target_currency="USD",
        fx_rates={"USD": 1.0},
        display_currency="USD",
        include_accounts=["B"] # Scope = B
    )
    
    print("\n--- CASE 2: Account B Only ---")
    print(f"Dates: {dates_b}")
    print(f"Flows: {flows_b}")
    
    # Verify B
    # Flow 1: Transfer In -500. Correct.
    assert flows_b[0] == -500.0, f"Expected -500 (Deposit), got {flows_b[0]}"
    
    
    # CASE 3: MWR for Portfolio (A + B)
    # Expect: Buy -1000. Transfer Internal -> 0.
    # End Value: A(1200) + B(500) = 1700.
    
    dates_ab, flows_ab = get_cash_flows_for_mwr(
        account_transactions=df, # All rows
        final_account_market_value=1700.0,
        end_date=date(2023, 12, 31),
        target_currency="USD",
        fx_rates={"USD": 1.0},
        display_currency="USD",
        include_accounts=["A", "B"] # Scope = A + B
    )
    
    print("\n--- CASE 3: Portfolio (A + B) ---")
    print(f"Dates: {dates_ab}")
    print(f"Flows: {flows_ab}")
    
    # Verify AB
    # Flow 1: Buy -1000.
    # Flow 2: Transfer -> Should be 0.
    
    buy_found = False
    transfer_zero = False
    
    for f in flows_ab:
        if f == -1000.0: buy_found = True
        
    # Transfer might not appear in flows if 0 (logic usually skips 0 flows).
    # But let's check if there is a 500 or -500.
    for f in flows_ab:
        if abs(f) == 500.0:
            print("FAILURE: Found transfer flow 500 in Portfolio View!")
            assert False, "Found internal transfer flow in portfolio view"
            
    assert buy_found, "Expected Buy -1000"
    
    print("\nSUCCESS: All Transfer MWR Logic Verified!")

if __name__ == "__main__":
    test_mwr_transfer_logic()
