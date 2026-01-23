import sys
import os

# --- Add src directory to sys.path ---
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import pytest
import sqlite3
import pandas as pd
from datetime import datetime, date
from db_utils import (
    initialize_database,
    add_transaction_to_db,
    update_transaction_in_db,
    delete_transaction_from_db,
    load_all_transactions_from_db,
    migrate_csv_to_db,
    check_if_db_empty_and_csv_exists,
    DB_SCHEMA_VERSION
)

# --- Fixtures ---
@pytest.fixture
def temp_db_path(tmp_path):
    """Creates a temporary database file path."""
    db_file = tmp_path / "test_investa.db"
    return str(db_file)

@pytest.fixture
def db_conn(temp_db_path):
    """Initializes a database and returns the connection."""
    conn = initialize_database(temp_db_path)
    yield conn
    if conn:
        conn.close()

@pytest.fixture
def sample_transaction_data():
    return {
        "Date": date(2023, 1, 15),
        "Type": "Buy",
        "Symbol": "AAPL",
        "Quantity": 10.0,
        "Price/Share": 150.0,
        "Total Amount": 1500.0,
        "Commission": 5.0,
        "Account": "TestAccount",
        "Split Ratio": None,
        "Note": "Test Note",
        "Local Currency": "USD",
        "To Account": None
    }

# --- Tests ---

def test_initialize_database(temp_db_path):
    conn = initialize_database(temp_db_path)
    assert conn is not None
    
    cursor = conn.cursor()
    # Check transactions table
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='transactions';")
    assert cursor.fetchone() is not None
    
    # Check schema_version table
    cursor.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1;")
    row = cursor.fetchone()
    assert row is not None
    assert row[0] == DB_SCHEMA_VERSION
    
    # Check columns in transactions table (including 'To Account')
    cursor.execute("PRAGMA table_info(transactions);")
    columns = [info[1] for info in cursor.fetchall()]
    assert "To Account" in columns
    assert "Local Currency" in columns
    conn.close()

def test_add_transaction(db_conn, sample_transaction_data):
    success, new_id = add_transaction_to_db(db_conn, sample_transaction_data)
    assert success is True
    assert new_id is not None
    
    # Verify insertion
    df, success = load_all_transactions_from_db(db_conn, {}, "USD")
    assert success is True
    assert len(df) == 1
    row = df.iloc[0]
    assert row["Symbol"] == "AAPL"
    assert row["Quantity"] == 10.0
    # Date comes back as Timestamp from pandas read_sql
    assert row["Date"].date() == sample_transaction_data["Date"] 

def test_update_transaction(db_conn, sample_transaction_data):
    _, new_id = add_transaction_to_db(db_conn, sample_transaction_data)
    
    update_data = {
        "Quantity": 20.0,
        "Note": "Updated Note"
    }
    success = update_transaction_in_db(db_conn, new_id, update_data)
    assert success is True
    
    # Verify update
    df, _ = load_all_transactions_from_db(db_conn, {}, "USD")
    row = df.iloc[0]
    assert row["Quantity"] == 20.0
    assert row["Note"] == "Updated Note"
    assert row["Symbol"] == "AAPL" # Unchanged

def test_delete_transaction(db_conn, sample_transaction_data):
    _, new_id = add_transaction_to_db(db_conn, sample_transaction_data)
    
    success = delete_transaction_from_db(db_conn, new_id)
    assert success is True
    
    # Verify deletion
    df, _ = load_all_transactions_from_db(db_conn, {}, "USD")
    assert len(df) == 0

def test_migrate_csv_to_db(db_conn, tmp_path):
    # Create a dummy CSV
    csv_path = tmp_path / "migration_test.csv"
    csv_content = {
        "Date": ["2023-01-01", "2023-01-02"],
        "Type": ["Buy", "Sell"],
        "Symbol": ["MSFT", "GOOG"],
        "Quantity": [5, 10],
        "Price/Share": [200, 100],
        "Total Amount": [1000, 1000],
        "Commission": [1, 1],
        "Account": ["MigratedAcc", "MigratedAcc"],
        "Local Currency": ["USD", "USD"]
    }
    pd.DataFrame(csv_content).to_csv(csv_path, index=False)
    
    migrated, errors = migrate_csv_to_db(str(csv_path), db_conn, {}, "USD")
    assert migrated == 2
    assert errors == 0
    
    df, _ = load_all_transactions_from_db(db_conn, {}, "USD")
    assert len(df) == 2
    assert "MSFT" in df["Symbol"].values
    assert "GOOG" in df["Symbol"].values

def test_check_if_db_empty_and_csv_exists(db_conn, tmp_path):
    csv_path = tmp_path / "exists.csv"
    csv_path.touch()
    
    # DB is empty initially
    assert check_if_db_empty_and_csv_exists(db_conn, str(csv_path)) is True
    
    # Add a transaction
    add_transaction_to_db(db_conn, {
        "Date": date(2023,1,1), "Type": "Buy", "Symbol": "A", "Account": "B", "Local Currency": "USD"
    })
    
    # DB not empty
    assert check_if_db_empty_and_csv_exists(db_conn, str(csv_path)) is False
