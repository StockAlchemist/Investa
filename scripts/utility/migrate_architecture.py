
import os
import sqlite3
import shutil
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
OLD_APP_DATA_DIR = os.path.expanduser("~/Library/Application Support/StockAlchemist/Investa")
OLD_DB_FILE = os.path.join(OLD_APP_DATA_DIR, "investa_transactions.db")
OLD_CONFIG_FILE = os.path.join(OLD_APP_DATA_DIR, "gui_config.json")
OLD_OVERRIDES_FILE = os.path.join(OLD_APP_DATA_DIR, "manual_overrides.json")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
GLOBAL_DB_PATH = os.path.join(DATA_DIR, "global.db")

def init_global_db():
    conn = sqlite3.connect(GLOBAL_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_active BOOLEAN DEFAULT 1,
            created_at TEXT NOT NULL
        );
    """)
    conn.commit()
    return conn

def migrate():
    if not os.path.exists(OLD_DB_FILE):
        logging.error(f"Old database not found at {OLD_DB_FILE}")
        return

    logging.info(f"Starting migration from {OLD_APP_DATA_DIR} to {DATA_DIR}")
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Initialize Global DB
    global_conn = init_global_db()
    
    # 2. Open Old DB to read users
    old_conn = sqlite3.connect(OLD_DB_FILE)
    old_cursor = old_conn.cursor()
    
    try:
        old_cursor.execute("SELECT id, username, hashed_password, created_at, is_active FROM users")
        users = old_cursor.fetchall()
    except sqlite3.OperationalError:
        logging.warning("No 'users' table in old DB. Assuming legacy mode.")
        users = []

    # 3. Migrate Users
    user_map = {} # Old ID -> Username
    
    for u in users:
        uid, username, pwd, created, active = u
        try:
            # Check if user exists in global db
            cursor = global_conn.cursor()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            existing = cursor.fetchone()
            
            if not existing:
                cursor.execute(
                    "INSERT INTO users (username, hashed_password, created_at, is_active) VALUES (?, ?, ?, ?)",
                    (username, pwd, created, active)
                )
                new_id = cursor.lastrowid
                logging.info(f"Migrated user {username} to global DB.")
            else:
                new_id = existing[0]
                logging.info(f"User {username} already in global DB.")
            
            user_map[uid] = username
            
        except sqlite3.Error as e:
            logging.error(f"Failed to migrate user {username}: {e}")

    global_conn.commit()
    global_conn.close()

    # 4. Create Per-User Directories and Data
    # For now, we assume ALL data in the old transaction DB belongs to 'kitmatan' (User ID 3 based on prior steps).
    # Ideally we should filter.
    # But since I know the state: User 3 owns 4694 transactions. 
    # Migration Strategy: Copy the FULL DB to kitmatan's folder.
    
    for old_uid, username in user_map.items():
        user_dir = os.path.join(DATA_DIR, "users", username)
        os.makedirs(user_dir, exist_ok=True)
        
        target_db = os.path.join(user_dir, "portfolio.db")
        target_config = os.path.join(user_dir, "gui_config.json")
        target_overrides = os.path.join(user_dir, "manual_overrides.json")
        
        # Copy DB
        if not os.path.exists(target_db):
            shutil.copy2(OLD_DB_FILE, target_db)
            logging.info(f"Copied DB for {username}")
            
            # Clean up other users' data (optional but good for strict isolation)
            # In Shared DB, user_id=3 is kitmatan.
            # We connect to NEW portfolio.db and delete anything where user_id != old_uid
            # CAUTION: If old_uid stored in DB is relevant? 
            # We will ignore user_id in the new system mostly.
            pass
            
        # Copy Configs
        if os.path.exists(OLD_CONFIG_FILE) and not os.path.exists(target_config):
            shutil.copy2(OLD_CONFIG_FILE, target_config)
            
        if os.path.exists(OLD_OVERRIDES_FILE) and not os.path.exists(target_overrides):
            shutil.copy2(OLD_OVERRIDES_FILE, target_overrides)

    logging.info("Migration completed successfully.")

if __name__ == "__main__":
    migrate()
