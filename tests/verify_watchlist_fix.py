import sqlite3
import os
import sys
from datetime import datetime

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from db_utils import get_all_watchlists, add_to_watchlist, get_watchlist, create_transactions_table

def verify_watchlist_fix():
    db_path = "test_watchlist_isolated.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    
    conn = sqlite3.connect(db_path)
    # Initialize tables
    create_transactions_table(conn)
    
    # 1. Manually insert a watchlist with a specific user_id to simulate a migrated DB
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO watchlists (name, created_at, user_id) VALUES (?, ?, ?)", 
        ("Migrated Watchlist", datetime.now().isoformat(), 99) # ID 99 doesn't match our 'session' user
    )
    watchlist_id = cursor.lastrowid
    
    # Add an item
    add_to_watchlist(conn, "AAPL", "Note", watchlist_id=watchlist_id)
    conn.commit()
    
    print(f"Watchlist created with ID {watchlist_id} and user_id 99.")
    
    # 2. Try to fetch watchlists WITHOUT passing user_id (as our new API does)
    # Or passing a DIFFERENT user_id (which should now be ignored for filtering)
    all_wls = get_all_watchlists(conn)
    print(f"Found {len(all_wls)} watchlists (should be 1).")
    for wl in all_wls:
        print(f"  - {wl['name']} (ID: {wl['id']})")
    
    # 3. Verify item retrieval
    items = get_watchlist(conn, watchlist_id=watchlist_id)
    print(f"Found {len(items)} items in watchlist {watchlist_id} (should be 1).")
    for item in items:
        print(f"  - {item['Symbol']}")

    # Clean up
    conn.close()
    if os.path.exists(db_path):
        os.remove(db_path)
    
    if len(all_wls) >= 1 and any(wl['id'] == watchlist_id for wl in all_wls) and len(items) == 1:
        print("\nVERIFICATION SUCCESSFUL: Watchlists are accessible regardless of internal user_id.")
    else:
        print("\nVERIFICATION FAILED: Watchlist access issue persists.")
        sys.exit(1)

if __name__ == "__main__":
    verify_watchlist_fix()
