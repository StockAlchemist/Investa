
import sqlite3
import pytest
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
import config

def test_database_integrity():
    """
    Verifies the integrity of the local SQLite database.
    Checks for: file integrity, table existence, and basic data quality.
    """
    db_path = os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")
    
    if not os.path.exists(db_path):
        pytest.skip(f"Database file not found at {db_path}")

    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. PRAGMA integrity_check
        cursor.execute("PRAGMA integrity_check")
        result = cursor.fetchone()
        assert result and result[0] == "ok", f"SQLite integrity check failed: {result}"
            
        # 2. Check Tables
        tables = ["daily_ohlcv", "daily_fx", "sync_metadata", "intraday_ohlcv"]
        for table in tables:
            try:
                cursor.execute(f"SELECT count(*) FROM {table}")
                count = cursor.fetchone()[0]
                assert count >= 0, f"Table '{table}' count query returned invalid result"
            except sqlite3.OperationalError as e:
                pytest.fail(f"Table '{table}' check failed: {e}")

        # 3. Check Data Quality
        # Check for invalid prices
        cursor.execute("SELECT count(*) FROM intraday_ohlcv WHERE close <= 0 OR high < low")
        bad_rows = cursor.fetchone()[0]
        assert bad_rows == 0, f"Found {bad_rows} rows with invalid prices (<=0 or high < low) in intraday_ohlcv"
            
        # Check for future dates (naive check)
        now_iso = datetime.now().isoformat()
        cursor.execute("SELECT count(*) FROM intraday_ohlcv WHERE timestamp > ?", (now_iso,))
        future_rows = cursor.fetchone()[0]
        if future_rows > 0:
             # Just warn for future timestamps as timezone diffs can cause minor issues
             print(f"WARN: Found {future_rows} rows with future timestamps in intraday_ohlcv")

    except Exception as e:
        pytest.fail(f"Database check failed with error: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    # Allow running directly for manual verification
    try:
        test_database_integrity()
        print("PASS: Database integrity check passed.")
    except Exception as e:
        print(f"FAIL: {e}")
