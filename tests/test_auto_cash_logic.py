import sys
import os
import sqlite3
import pandas as pd
from typing import Dict, Any

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from server.api import _handle_auto_cash_generation
from db_utils import add_transaction_to_db

# Mock add_transaction_to_db to capture calls
captured_transactions = []

def mock_add_transaction_to_db(conn, tx_data):
    captured_transactions.append(tx_data)
    return True, len(captured_transactions)

# Patch the imported function
import server.api
server.api.add_transaction_to_db = mock_add_transaction_to_db

def test_auto_cash_buy():
    global captured_transactions
    captured_transactions = []
    
    tx_data = {
        "Date": "2026-03-13",
        "Type": "Buy",
        "Symbol": "AAPL",
        "Quantity": 10.0,
        "Price/Share": 150.0,
        "Commission": 5.0,
        "Account": "TestAcc",
        "Local Currency": "USD",
        "Total Amount": 1500.0,
        "Auto-add Cash": True
    }
    
    conn = None # Mocked
    _handle_auto_cash_generation(conn, tx_data)
    
    print(f"Captured {len(captured_transactions)} transactions for Buy")
    for i, tx in enumerate(captured_transactions):
        print(f"  {i+1}: {tx['Type']} {tx['Symbol']} {tx['Quantity']} - {tx['Note']}")
        
    assert len(captured_transactions) == 2
    assert captured_transactions[0]["Type"] == "Sell"
    assert captured_transactions[0]["Symbol"] == "$CASH"
    assert captured_transactions[0]["Quantity"] == 1500.0
    assert captured_transactions[1]["Type"] == "Withdrawal"
    assert captured_transactions[1]["Quantity"] == 5.0

def test_auto_cash_sell():
    global captured_transactions
    captured_transactions = []
    
    tx_data = {
        "Date": "2026-03-13",
        "Type": "Sell",
        "Symbol": "MSFT",
        "Quantity": 5.0,
        "Price/Share": 400.0,
        "Commission": 10.0,
        "Account": "TestAcc",
        "Local Currency": "USD",
        "Total Amount": 2000.0,
        "Auto-add Cash": True
    }
    
    conn = None # Mocked
    _handle_auto_cash_generation(conn, tx_data)
    
    print(f"Captured {len(captured_transactions)} transactions for Sell")
    for i, tx in enumerate(captured_transactions):
        print(f"  {i+1}: {tx['Type']} {tx['Symbol']} {tx['Quantity']} - {tx['Note']}")
        
    assert len(captured_transactions) == 2
    assert captured_transactions[0]["Type"] == "Buy"
    assert captured_transactions[0]["Symbol"] == "$CASH"
    assert captured_transactions[0]["Quantity"] == 2000.0
    assert captured_transactions[1]["Type"] == "Withdrawal"
    assert captured_transactions[1]["Quantity"] == 10.0

if __name__ == "__main__":
    try:
        test_auto_cash_buy()
        test_auto_cash_sell()
        print("\nAll backend tests passed!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
