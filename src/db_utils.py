# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          db_utils.py
 Purpose:       Database utility functions for SQLite.
                Handles database connection, schema creation, and path management.

 Author:        Google Gemini


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import sqlite3
from datetime import datetime, date
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import traceback
import threading
import config

DB_FILENAME = "investa_transactions.db"
DB_SCHEMA_VERSION = 9


def get_database_path(db_filename: str = DB_FILENAME) -> str:
    """
    Determines the full path for the SQLite database file.
    Prioritizes centralized application data directory, falls back to CWD.
    """
    # Priority 1: Check centralized application data directory
    app_data_dir = config.get_app_data_dir()
    app_data_path = os.path.join(app_data_dir, db_filename)
    if os.path.exists(app_data_path):
        return app_data_path

    # Priority 2: Check if DB exists in current working directory
    cwd_path = os.path.join(os.getcwd(), db_filename)
    if os.path.exists(cwd_path):
        return cwd_path

    # Final Fallback: Return app data path (it will be created if it doesn't exist)
    return app_data_path


_DB_CONN_CACHE = threading.local()

def get_db_connection(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database, with thread-local caching."""
    if db_path is None:
        db_path = get_database_path()
    
    if not hasattr(_DB_CONN_CACHE, 'connections'):
        _DB_CONN_CACHE.connections = {}
    
    if db_path in _DB_CONN_CACHE.connections:
        # Verify connection is still open
        try:
            _DB_CONN_CACHE.connections[db_path].execute("SELECT 1")
            return _DB_CONN_CACHE.connections[db_path]
        except (sqlite3.ProgrammingError, sqlite3.Error):
            del _DB_CONN_CACHE.connections[db_path]

    try:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        import time
        retries = 5
        conn = None
        for i in range(retries):
            try:
                conn = sqlite3.connect(db_path, timeout=30)
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                break
            except sqlite3.OperationalError as e:
                if "unable to open database file" in str(e) or "database is locked" in str(e):
                    if i < retries - 1:
                        time.sleep(0.1 * (i + 1))
                        continue
                raise e
        
        if conn:
            _DB_CONN_CACHE.connections[db_path] = conn
            logging.info(f"Successfully connected to database (cached): {db_path}")
            return conn
        return None
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database at {db_path}: {e}", exc_info=True)
        return None
    except OSError as e_os:
        logging.error(
            f"OS error setting up database path {db_path}: {e_os}", exc_info=True
        )
        return None


def create_transactions_table(conn: sqlite3.Connection):
    """Creates the transactions table and schema_version table if they don't exist."""
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS transactions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        Date TEXT NOT NULL,
        Type TEXT NOT NULL,
        Symbol TEXT NOT NULL,
        Quantity REAL,
        "Price/Share" REAL,
        "Total Amount" REAL,
        Commission REAL,
        Account TEXT NOT NULL,
        "Split Ratio" REAL,
        Note TEXT,
        "Local Currency" TEXT NOT NULL,
        "To Account" TEXT,
        "Tags" TEXT
    );
    """
    create_version_table_sql = """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_on TEXT NOT NULL
    );
    """
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        cursor.execute(create_version_table_sql)

        cursor.execute(
            "SELECT version FROM schema_version ORDER BY version DESC LIMIT 1"
        )
        current_db_version_row = cursor.fetchone()
        current_db_version = current_db_version_row[0] if current_db_version_row else 0

        if current_db_version < 1:
            # This is a new DB or pre-versioning, set version to current.
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                (DB_SCHEMA_VERSION, datetime.now().isoformat()),
            )
            logging.info(f"Initialized database schema to version {DB_SCHEMA_VERSION}.")

        if current_db_version < 2:
            logging.info(
                "Schema version is less than 2. Applying 'To Account' column migration."
            )
            # Migration for version 2: Add "To Account" column and populate it
            try:
                cursor.execute('ALTER TABLE transactions ADD COLUMN "To Account" TEXT;')
                logging.info("Added 'To Account' column to transactions table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logging.info("'To Account' column already exists.")
                else:
                    raise  # Re-raise other operational errors

            # Populate the new column from existing notes
            # This is a best-effort migration. The regex looks for "To: Account Name"
            cursor.execute(
                "UPDATE transactions SET \"To Account\" = SUBSTR(Note, INSTR(Note, ':') + 2) WHERE Type = 'Transfer' AND Note LIKE 'To:%' AND \"To Account\" IS NULL;"
            )
            logging.info(
                f"Migrated {cursor.rowcount} existing transfer transactions to use the 'To Account' column."
            )

            # Update schema version after successful migration
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                (DB_SCHEMA_VERSION, datetime.now().isoformat()),
            )

        create_watchlist_sql = """
        CREATE TABLE IF NOT EXISTS watchlist (
            Symbol TEXT PRIMARY KEY,
            Note TEXT,
            AddedOn TEXT NOT NULL
        );
        """
        cursor.execute(create_watchlist_sql)

        if current_db_version < 3:
            logging.info("Schema version is less than 3. Watchlist table created (if not exists).")
            # In this case, CREATE TABLE already handled it, so we just update version if needed
            # but usually we want specific migrations if structure changed.
            # For a NEW table, the CREATE TABLE IF NOT EXISTS is enough.
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                (DB_SCHEMA_VERSION, datetime.now().isoformat()),
            )

        if current_db_version < 4:
            logging.info("Schema version is less than 4. Applying 'Tags' column migration.")
            try:
                cursor.execute('ALTER TABLE transactions ADD COLUMN Tags TEXT;')
                logging.info("Added 'Tags' column to transactions table.")
            except sqlite3.OperationalError as e:
                if "duplicate column name" in str(e):
                    logging.info("'Tags' column already exists.")
                else:
                    raise

        if current_db_version < 5:
            logging.info("Schema version is less than 5. Applying Multiple Watchlists migration.")
            try:
                # 1. Create watchlists table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS watchlists (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    );
                """)
                
                # 2. Create default watchlist if not exists
                cursor.execute("SELECT count(*) FROM watchlists")
                if cursor.fetchone()[0] == 0:
                     cursor.execute(
                         "INSERT INTO watchlists (name, created_at) VALUES (?, ?)", 
                         ("My Watchlist", datetime.now().isoformat())
                     )
                     default_id = cursor.lastrowid
                else:
                     cursor.execute("SELECT id FROM watchlists LIMIT 1")
                     default_id = cursor.fetchone()[0]

                # 3. Create watchlist_items table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS watchlist_items (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        watchlist_id INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        note TEXT,
                        added_on TEXT NOT NULL,
                        FOREIGN KEY(watchlist_id) REFERENCES watchlists(id) ON DELETE CASCADE
                    );
                """)
                
                # 4. Migrate existing data from old 'watchlist' table if it exists
                # Check if old table exists
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='watchlist';")
                if cursor.fetchone():
                    logging.info("Migrating existing watchlist items...")
                    cursor.execute("SELECT Symbol, Note, AddedOn FROM watchlist")
                    rows = cursor.fetchall()
                    for r in rows:
                        cursor.execute(
                            "INSERT INTO watchlist_items (watchlist_id, symbol, note, added_on) VALUES (?, ?, ?, ?)",
                            (default_id, r[0], r[1], r[2])
                        )
                    # Drop old table
                    cursor.execute("DROP TABLE watchlist")
                    logging.info(f"Migrated {len(rows)} items and dropped old watchlist table.")

            except sqlite3.Error as e:
                logging.error(f"Error during migration v5: {e}")
                raise

            logging.info("Updating schema_version to 5...")
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                (5, datetime.now().isoformat()),
            )
            logging.info("Updated schema_version to 5 (pending commit).")

        if current_db_version < 6:
            logging.info("Schema version is less than 6. Creating screener_cache table.")
            try:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS screener_cache (
                        symbol TEXT PRIMARY KEY,
                        name TEXT,
                        price REAL,
                        intrinsic_value REAL,
                        margin_of_safety REAL,
                        pe_ratio REAL,
                        market_cap REAL,
                        sector TEXT,
                        ai_moat REAL,
                        ai_financial_strength REAL,
                        ai_predictability REAL,
                        ai_growth REAL,
                        ai_summary TEXT,
                        last_fiscal_year_end INTEGER,
                        most_recent_quarter INTEGER,
                        updated_at TEXT NOT NULL
                    );
                """)
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                    (6, datetime.now().isoformat()),
                )
                logging.info("Updated schema_version to 6.")
            except sqlite3.Error as e:
                logging.error(f"Error during migration v6: {e}")
                raise

        if current_db_version < 7:
            logging.info("Schema version is less than 7. Adding 'universe' column to screener_cache.")
            try:
                cursor.execute('ALTER TABLE screener_cache ADD COLUMN universe TEXT;')
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                    (7, datetime.now().isoformat()),
                )
                logging.info("Updated schema_version to 7.")
            except sqlite3.Error as e:
                if "duplicate column name" in str(e):
                    logging.info("'universe' column already exists.")
                else:
                    logging.error(f"Error during migration v7: {e}")
                    raise

        if current_db_version < 8:
            logging.info("Schema version is less than 8. Migrating screener_cache to composite PK (symbol, universe).")
            try:
                # 1. Rename old table
                cursor.execute("ALTER TABLE screener_cache RENAME TO screener_cache_old;")
                
                # 2. Create new table with composite PK and universe NOT NULL
                cursor.execute("""
                    CREATE TABLE screener_cache (
                        symbol TEXT NOT NULL,
                        name TEXT,
                        price REAL,
                        intrinsic_value REAL,
                        margin_of_safety REAL,
                        pe_ratio REAL,
                        market_cap REAL,
                        sector TEXT,
                        ai_moat REAL,
                        ai_financial_strength REAL,
                        ai_predictability REAL,
                        ai_growth REAL,
                        ai_summary TEXT,
                        last_fiscal_year_end INTEGER,
                        most_recent_quarter INTEGER,
                        universe TEXT NOT NULL DEFAULT 'manual',
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (symbol, universe)
                    );
                """)
                
                # 3. Copy data, filling NULL universes with 'manual'
                cursor.execute("""
                    INSERT INTO screener_cache (
                        symbol, name, price, intrinsic_value, margin_of_safety,
                        pe_ratio, market_cap, sector, ai_moat, ai_financial_strength,
                        ai_predictability, ai_growth, ai_summary, last_fiscal_year_end,
                        most_recent_quarter, universe, updated_at
                    )
                    SELECT 
                        symbol, name, price, intrinsic_value, margin_of_safety,
                        pe_ratio, market_cap, sector, ai_moat, ai_financial_strength,
                        ai_predictability, ai_growth, ai_summary, last_fiscal_year_end,
                        most_recent_quarter, COALESCE(universe, 'manual'), updated_at
                    FROM screener_cache_old;
                """)
                
                # 4. Drop old table
                cursor.execute("DROP TABLE screener_cache_old;")
                
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                    (8, datetime.now().isoformat()),
                )
                logging.info("Updated schema_version to 8.")
            except sqlite3.Error as e:
                logging.error(f"Error during migration v8: {e}")
                raise

        if current_db_version < 9:
            logging.info("Schema version is less than 9. Adding 'valuation_details' column to screener_cache.")
            try:
                cursor.execute('ALTER TABLE screener_cache ADD COLUMN valuation_details TEXT;')
                cursor.execute(
                    "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                    (9, datetime.now().isoformat()),
                )
                logging.info("Updated schema_version to 9.")
            except sqlite3.Error as e:
                if "duplicate column name" in str(e):
                    logging.info("'valuation_details' column already exists.")
                else:
                    logging.error(f"Error during migration v9: {e}")
                    raise

        conn.commit()
        logging.info(
            "Transactions, watchlist, and schema_version tables checked/created/updated successfully."
        )
    except sqlite3.Error as e:
        logging.error(f"Error creating/updating tables: {e}", exc_info=True)
        conn.rollback()


def check_if_db_empty_and_csv_exists(
    db_conn: sqlite3.Connection, csv_file_path: str
) -> bool:
    if not db_conn:
        return False
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM transactions")
        count_result = cursor.fetchone()
        count = count_result[0] if count_result else 0

        csv_exists = os.path.exists(csv_file_path) and os.path.isfile(csv_file_path)
        if count == 0 and csv_exists:
            logging.info(
                f"Database is empty and CSV '{csv_file_path}' exists. Migration may be needed."
            )
            return True
        elif count > 0:
            logging.info(
                f"Database is not empty (contains {count} transactions). Migration prompt for '{csv_file_path}' might be skipped."
            )
        elif not csv_exists:
            logging.info(
                f"Migration CSV '{csv_file_path}' does not exist. No migration prompt."
            )

    except sqlite3.Error as e:
        logging.error(f"Error checking database count: {e}", exc_info=True)
    return False


def migrate_csv_to_db(
    csv_file_path: str,
    db_conn: sqlite3.Connection,
    account_currency_map: Dict[str, str],
    default_currency: str,
) -> Tuple[int, int]:
    """
    Migrates transaction data from a CSV file to the SQLite database.
    """
    # --- MODIFIED: Removed global DATA_LOADER_AVAILABLE flag and its usage ---
    try:
        from data_loader import load_and_clean_transactions
    except ImportError:
        logging.error(
            "CRITICAL: data_loader.load_and_clean_transactions could not be imported in migrate_csv_to_db. Migration failed."
        )
        return 0, 1  # Return error if import fails

    logging.info(f"Starting migration from CSV: {csv_file_path}")
    migrated_count = 0
    error_count = 0

    try:
        transactions_df, _, _, _, load_error, _, _ = load_and_clean_transactions(
            csv_file_path,
            account_currency_map,
            default_currency,
            is_db_source=False,
        )

        if load_error or transactions_df is None:
            logging.error(
                f"Failed to load or clean CSV '{csv_file_path}' for migration."
            )
            return 0, 1

        if transactions_df.empty:
            logging.info(f"CSV file '{csv_file_path}' is empty. Nothing to migrate.")
            return 0, 0

        cursor = db_conn.cursor()
        if "Date" in transactions_df.columns and pd.api.types.is_datetime64_any_dtype(
            transactions_df["Date"]
        ):
            transactions_df["Date"] = transactions_df["Date"].dt.strftime("%Y-%m-%d")

        db_columns = [
            "Date",
            "Type",
            "Symbol",
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Account",
            "Split Ratio",
            "Note",
            "Local Currency",  # "To Account" is not in the CSV standard
            "Tags",
        ]
        df_for_insert = transactions_df.reindex(columns=db_columns)
        placeholders = ", ".join(["?"] * len(db_columns))
        quoted_db_columns = [
            f'"{col}"' if "/" in col or " " in col else col for col in db_columns
        ]
        sql_insert = f"INSERT INTO transactions ({', '.join(quoted_db_columns)}) VALUES ({placeholders});"

        for _, row_series in df_for_insert.iterrows():
            try:
                data_tuple = tuple(
                    None if pd.isna(x) else x for x in row_series.tolist()
                )
                cursor.execute(sql_insert, data_tuple)
                migrated_count += 1
            except sqlite3.Error as e:
                logging.error(
                    f"Error inserting row into database: {row_series.to_dict()} - Error: {e}",
                    exc_info=True,
                )
                error_count += 1
            except Exception as ex:
                logging.error(
                    f"General error processing row for DB insert: {row_series.to_dict()} - Error: {ex}",
                    exc_info=True,
                )
                error_count += 1
        db_conn.commit()
        logging.info(
            f"Migration complete. Migrated: {migrated_count}, Errors: {error_count}"
        )
    except Exception as e:
        logging.error(
            f"Critical error during CSV migration process: {e}", exc_info=True
        )
        try:
            db_conn.rollback()
        except:
            pass
        return migrated_count, error_count + 1
    return migrated_count, error_count


def load_all_transactions_from_db(
    db_conn: sqlite3.Connection,
    account_currency_map: Dict[str, str],
    default_currency: str,
) -> Tuple[Optional[pd.DataFrame], bool]:
    logging.info("Loading all transactions from the database.")
    try:
        query = """
        SELECT id as original_index, Date, Type, Symbol, Quantity, "Price/Share",
               "Total Amount", Commission, Account, "Split Ratio", Note, "Local Currency", "To Account", Tags
        FROM transactions
        ORDER BY Date, original_index;
        """
        df = pd.read_sql_query(query, db_conn, parse_dates=["Date"])

        if "Local Currency" in df.columns and "Account" in df.columns:
            is_empty_local_currency = df["Local Currency"].isin(
                [None, np.nan, pd.NA, ""]
            )
            if pd.api.types.is_string_dtype(df["Local Currency"]):
                is_empty_local_currency |= (
                    df["Local Currency"].astype(str).str.upper().str.strip() == "<NA>"
                )

            if is_empty_local_currency.any():
                df.loc[is_empty_local_currency, "Local Currency"] = (
                    df.loc[is_empty_local_currency, "Account"]
                    .map(account_currency_map)
                    .fillna(default_currency)
                )
            df["Local Currency"] = df["Local Currency"].str.upper()

        if "original_index" in df.columns:
            df["original_index"] = pd.to_numeric(
                df["original_index"], errors="coerce"
            ).astype("Int64")
        numeric_cols = [
            "Quantity",
            "Price/Share",
            "Total Amount",
            "Commission",
            "Split Ratio",
        ]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        logging.info(f"Successfully loaded {len(df)} transactions from the database.")
        return df, True
    except Exception as e:
        logging.error(f"Error loading transactions from database: {e}", exc_info=True)
        return None, False


def add_transaction_to_db(
    db_conn: sqlite3.Connection, transaction_data: Dict[str, Any]
) -> Tuple[bool, Optional[int]]:
    db_column_order = [
        "Date",
        "Type",
        "Symbol",
        "Quantity",
        "Price/Share",
        "Total Amount",
        "Commission",
        "Account",
        "Split Ratio",
        "Note",
        "Local Currency",
        "Local Currency",
        "To Account",
        "Tags",
    ]
    quoted_db_columns = [
        f'"{col}"' if "/" in col or " " in col else col for col in db_column_order
    ]
    placeholders_list = []
    data_for_sql_ordered = []

    for col_name in db_column_order:
        placeholder_name = (
            col_name.replace("/", "_per_").replace(" ", "_").replace(".", "_")
        )
        placeholders_list.append(f":{placeholder_name}")
        value = transaction_data.get(col_name)
        if col_name == "Type" and isinstance(value, str):
            data_for_sql_ordered.append(value.strip().title())
        elif col_name == "Date":
            if isinstance(value, (datetime, date)):
                data_for_sql_ordered.append(value.strftime("%Y-%m-%d"))
            elif isinstance(value, str):
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                    data_for_sql_ordered.append(value)
                except ValueError:
                    logging.error(f"Invalid date string for DB: {value}")
                    return False, None
            else:
                data_for_sql_ordered.append(None)
        elif pd.isna(value):
            data_for_sql_ordered.append(None)
        else:
            data_for_sql_ordered.append(value)

    data_for_sql_dict = {
        ph.lstrip(":"): val for ph, val in zip(placeholders_list, data_for_sql_ordered)
    }

    sql_insert = f"INSERT INTO transactions ({', '.join(quoted_db_columns)}) VALUES ({', '.join(placeholders_list)});"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql_insert, data_for_sql_dict)
        db_conn.commit()
        new_id = cursor.lastrowid
        logging.info(f"Successfully added transaction with ID: {new_id}.")
        return True, new_id
    except sqlite3.Error as e:
        logging.error(
            f"Error adding transaction to database. Data: {transaction_data} - Error: {e}",
            exc_info=True,
        )
        try:
            db_conn.rollback()
        except:
            pass
        return False, None


def update_transaction_in_db(
    db_conn: sqlite3.Connection, transaction_id: int, new_data_dict: Dict[str, Any]
) -> bool:
    set_clauses = []
    values_for_sql: Dict[str, Any] = {}

    for key, value in new_data_dict.items():
        placeholder = key.replace("/", "_per_").replace(" ", "_").replace(".", "_")
        set_clauses.append(f'"{key}" = :{placeholder}')

        if key == "Type" and isinstance(value, str):
            values_for_sql[placeholder] = value.strip().title()
        elif key == "Date":
            if isinstance(value, (datetime, date)):
                values_for_sql[placeholder] = value.strftime("%Y-%m-%d")
            elif isinstance(value, str):
                try:
                    datetime.strptime(value, "%Y-%m-%d")
                    values_for_sql[placeholder] = value
                except ValueError:
                    logging.error(f"Invalid date string for update: {value}")
                    return False
            else:
                values_for_sql[placeholder] = None
        elif pd.isna(value):
            values_for_sql[placeholder] = None
        else:
            values_for_sql[placeholder] = value

    if not set_clauses:
        logging.warning("Update transaction: No data provided.")
        return False

    sql_update = (
        f"UPDATE transactions SET {', '.join(set_clauses)} WHERE id = :id_placeholder"
    )
    values_for_sql["id_placeholder"] = transaction_id

    # --- BUG FIX: Handle linked transfer transaction ---
    # If the transaction is a transfer, we need to update the corresponding deposit record as well.
    # This logic assumes the 'Note' field links the two parts of the transfer.
    original_note_for_transfer = None
    if new_data_dict.get("Type", "").lower() == "transfer":
        try:
            # Fetch the original note before the update to find the linked transaction.
            cursor_for_note = db_conn.cursor()
            cursor_for_note.execute(
                "SELECT Note FROM transactions WHERE id = ?", (transaction_id,)
            )
            result = cursor_for_note.fetchone()
            if result:
                original_note_for_transfer = result[0]
        except sqlite3.Error as e:
            logging.error(
                f"Could not fetch original note for transfer transaction ID {transaction_id}: {e}"
            )

    try:
        cursor = db_conn.cursor()
        cursor.execute(sql_update, values_for_sql)
        db_conn.commit()
        if cursor.rowcount == 0:
            logging.warning(
                f"Update transaction: No row found with ID: {transaction_id}"
            )
            return False
        logging.info(f"Successfully updated transaction ID: {transaction_id}.")

        # If it was a transfer and we found the original note, update the other side.
        if original_note_for_transfer:
            # The 'new_data_dict' contains the new data. We only want to update a few fields
            # on the other side of the transfer (Date, Total Amount, Note).
            linked_update_data = {
                "Date": new_data_dict.get("Date"),
                "Total Amount": new_data_dict.get("Total Amount"),
                "Note": new_data_dict.get("Note"),
            }
            # Remove None values so we only update fields that were actually changed.
            linked_update_data = {
                k: v for k, v in linked_update_data.items() if v is not None
            }

            if linked_update_data:
                # Update the 'deposit' part of the transfer, which has the same note but is not the transaction we just updated.
                set_clauses_linked = [f'"{k}" = ?' for k in linked_update_data.keys()]
                sql_update_linked = f"UPDATE transactions SET {', '.join(set_clauses_linked)} WHERE Note = ? AND id != ?"
                cursor.execute(
                    sql_update_linked,
                    (
                        *linked_update_data.values(),
                        original_note_for_transfer,
                        transaction_id,
                    ),
                )
                logging.info(
                    f"Updated linked transfer transaction for original note: '{original_note_for_transfer}'"
                )

        return True
    except sqlite3.Error as e:
        logging.error(
            f"Error updating transaction ID {transaction_id}. Data: {new_data_dict} - Error: {e}",
            exc_info=True,
        )
        try:
            db_conn.rollback()
        except:
            pass
        return False


def delete_transaction_from_db(
    db_conn: sqlite3.Connection, transaction_id: int
) -> bool:
    sql_delete = "DELETE FROM transactions WHERE id = ?"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql_delete, (transaction_id,))
        db_conn.commit()
        if cursor.rowcount == 0:
            logging.warning(
                f"Delete transaction: No row found with ID: {transaction_id}"
            )
            return False
        logging.info(f"Successfully deleted transaction ID: {transaction_id}")
        return True
    except sqlite3.Error as e:
        logging.error(
            f"Error deleting transaction ID {transaction_id}: {e}", exc_info=True
        )
        try:
            db_conn.rollback()
        except:
            pass
        return False



def get_all_watchlists(db_conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Fetches all available watchlists."""
    sql = "SELECT id, name, created_at FROM watchlists ORDER BY created_at ASC"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2]} for r in rows]
    except sqlite3.Error as e:
        logging.error(f"Error fetching watchlists: {e}")
        return []

def create_watchlist(db_conn: sqlite3.Connection, name: str) -> Optional[int]:
    """Creates a new watchlist."""
    sql = "INSERT INTO watchlists (name, created_at) VALUES (?, ?)"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, (name, datetime.now().isoformat()))
        db_conn.commit()
        logging.info(f"Created watchlist '{name}' with ID {cursor.lastrowid}")
        return cursor.lastrowid
    except sqlite3.Error as e:
        logging.error(f"Error creating watchlist '{name}': {e}")
        return None

def rename_watchlist(db_conn: sqlite3.Connection, watchlist_id: int, new_name: str) -> bool:
    """Renames an existing watchlist."""
    sql = "UPDATE watchlists SET name = ? WHERE id = ?"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, (new_name, watchlist_id))
        db_conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error renaming watchlist {watchlist_id}: {e}")
        return False

def delete_watchlist(db_conn: sqlite3.Connection, watchlist_id: int) -> bool:
    """Deletes a watchlist and all its items."""
    # Note: ON DELETE CASCADE should handle items, but we explicitly delete to be safe if foreign keys disabled
    try:
        cursor = db_conn.cursor()
        # Verify it's not the last watchlist? Or allow deleting? 
        # For now, allow deleting anything. Frontend should prevent deleting the last one if needed.
        cursor.execute("DELETE FROM watchlists WHERE id = ?", (watchlist_id,))
        db_conn.commit()
        return cursor.rowcount > 0
    except sqlite3.Error as e:
        logging.error(f"Error deleting watchlist {watchlist_id}: {e}")
        return False

def add_to_watchlist(db_conn: sqlite3.Connection, symbol: str, note: str = "", watchlist_id: int = 1) -> bool:
    """Adds a symbol to a specific watchlist."""
    # Check if symbol exists in this watchlist
    # Use INSERT OR REPLACE on (watchlist_id, symbol) if unique constraint existed.
    # Currently we don't strictly enforce unique constraint in CREATE TABLE above, 
    # but we should probably avoid duplicates.
    # Let's check first.
    try:
        cursor = db_conn.cursor()
        cursor.execute("SELECT id FROM watchlist_items WHERE watchlist_id = ? AND symbol = ?", (watchlist_id, symbol))
        if cursor.fetchone():
            # Update existing
            sql = "UPDATE watchlist_items SET note = ?, added_on = ? WHERE watchlist_id = ? AND symbol = ?"
            cursor.execute(sql, (note, datetime.now().isoformat(), watchlist_id, symbol))
        else:
            # Insert new
            sql = "INSERT INTO watchlist_items (watchlist_id, symbol, note, added_on) VALUES (?, ?, ?, ?)"
            cursor.execute(sql, (watchlist_id, symbol, note, datetime.now().isoformat()))
        
        db_conn.commit()
        logging.info(f"Successfully added/updated {symbol} in watchlist {watchlist_id}.")
        return True
    except sqlite3.Error as e:
        logging.error(f"Error adding {symbol} to watchlist {watchlist_id}: {e}")
        return False


def remove_from_watchlist(db_conn: sqlite3.Connection, symbol: str, watchlist_id: int = 1) -> bool:
    """Removes a symbol from a specific watchlist."""
    sql = "DELETE FROM watchlist_items WHERE watchlist_id = ? AND symbol = ?"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, (watchlist_id, symbol))
        db_conn.commit()
        logging.info(f"Successfully removed {symbol} from watchlist {watchlist_id}.")
        return True
    except sqlite3.Error as e:
        logging.error(f"Error removing {symbol} from watchlist {watchlist_id}: {e}")
        return False


def get_watchlist(db_conn: sqlite3.Connection, watchlist_id: int = 1) -> List[Dict[str, Any]]:
    """Fetches all items in a specific watchlist."""
    # We fallback to fetching items from old table ONLY if migration failed? 
    # No, migration should have handled it. We assume watchlist_items table exists.
    
    # Check if watchlist_items table exists (safety check during dev/migration)
    # But for prod code, we modify the query.
    sql = "SELECT symbol, note, added_on FROM watchlist_items WHERE watchlist_id = ? ORDER BY added_on DESC"
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, (watchlist_id,))
        rows = cursor.fetchall()
        return [{"Symbol": r[0], "Note": r[1], "AddedOn": r[2]} for r in rows]
    except sqlite3.Error as e:
        # Fallback for transient state or error
        logging.error(f"Error fetching watchlist {watchlist_id}: {e}")
        return []

def upsert_screener_results(db_conn: sqlite3.Connection, results: List[Dict[str, Any]], universe: Optional[str] = None):
    """Batch updates/inserts screener results into the cache."""
    if not results:
        return
        
    sql = """
    INSERT INTO screener_cache (
        symbol, name, price, intrinsic_value, margin_of_safety, 
        pe_ratio, market_cap, sector, 
        ai_moat, ai_financial_strength, ai_predictability, ai_growth, ai_summary,
        last_fiscal_year_end, most_recent_quarter, universe, updated_at, valuation_details
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(symbol, universe) DO UPDATE SET
        name=excluded.name,
        price=excluded.price,
        intrinsic_value=excluded.intrinsic_value,
        margin_of_safety=excluded.margin_of_safety,
        pe_ratio=excluded.pe_ratio,
        market_cap=excluded.market_cap,
        sector=excluded.sector,
        ai_moat=COALESCE(excluded.ai_moat, screener_cache.ai_moat),
        ai_financial_strength=COALESCE(excluded.ai_financial_strength, screener_cache.ai_financial_strength),
        ai_predictability=COALESCE(excluded.ai_predictability, screener_cache.ai_predictability),
        ai_growth=COALESCE(excluded.ai_growth, screener_cache.ai_growth),
        ai_summary=COALESCE(excluded.ai_summary, screener_cache.ai_summary),
        last_fiscal_year_end=excluded.last_fiscal_year_end,
        most_recent_quarter=excluded.most_recent_quarter,
        updated_at=excluded.updated_at,
        valuation_details=COALESCE(excluded.valuation_details, screener_cache.valuation_details)
    """
    
    now_str = datetime.now().isoformat()
    data = []
    for r in results:
        data.append((
            r.get("symbol"),
            r.get("name"),
            r.get("price"),
            r.get("intrinsic_value"),
            r.get("margin_of_safety"),
            r.get("pe_ratio"),
            r.get("market_cap"),
            r.get("sector"),
            r.get("ai_moat"),
            r.get("ai_financial_strength"),
            r.get("ai_predictability"),
            r.get("ai_growth"),
            r.get("ai_summary"),
            r.get("last_fiscal_year_end"),
            r.get("most_recent_quarter"),
            universe,
            now_str,
            r.get("valuation_details")
        ))
        
    try:
        cursor = db_conn.cursor()
        cursor.executemany(sql, data)
        db_conn.commit()
        logging.info(f"Upserted {len(results)} screener results to DB cache.")
    except sqlite3.Error as e:
        logging.error(f"Error upserting screener results: {e}")
        db_conn.rollback()

def get_cached_screener_results(db_conn: sqlite3.Connection, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieves cached screener data for a list of symbols."""
    if not symbols:
        return {}
        
    placeholders = ",".join(["?"] * len(symbols))
    sql = f"SELECT * FROM screener_cache WHERE symbol IN ({placeholders})"
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, symbols)
        rows = cursor.fetchall()
        
        # Get column names
        cols = [description[0] for description in cursor.description]
        
        results = {}
        for row in rows:
            row_dict = dict(zip(cols, row))
            
            # Calculate average AI score if available
            ai_scores = [
                row_dict.get("ai_moat"),
                row_dict.get("ai_financial_strength"),
                row_dict.get("ai_predictability"),
                row_dict.get("ai_growth")
            ]
            valid_scores = [s for s in ai_scores if s is not None]
            if valid_scores:
                row_dict["ai_score"] = sum(valid_scores) / len(valid_scores)
            else:
                row_dict["ai_score"] = None
                
            results[row_dict["symbol"]] = row_dict
        return results
    except sqlite3.Error as e:
        logging.error(f"Error fetching cached screener results: {e}")
        return {}

def get_screener_results_by_universe(db_conn: sqlite3.Connection, universe: str) -> List[Dict[str, Any]]:
    """Retrieves all cached screener results for a specific universe tag."""
    sql = "SELECT * FROM screener_cache WHERE universe = ? ORDER BY margin_of_safety DESC"
    
    try:
        cursor = db_conn.cursor()
        cursor.execute(sql, (universe,))
        rows = cursor.fetchall()
        
        cols = [description[0] for description in cursor.description]
        results = []
        for row in rows:
            row_dict = dict(zip(cols, row))
            
            # Cleanup for frontend consistency
            if "ai_summary" in row_dict:
                row_dict["has_ai_review"] = row_dict["ai_summary"] is not None
            
            # Calculate average AI score if available
            ai_scores = [
                row_dict.get("ai_moat"),
                row_dict.get("ai_financial_strength"),
                row_dict.get("ai_predictability"),
                row_dict.get("ai_growth")
            ]
            valid_scores = [s for s in ai_scores if s is not None]
            if valid_scores:
                row_dict["ai_score"] = sum(valid_scores) / len(valid_scores)
            else:
                row_dict["ai_score"] = None
                
            results.append(row_dict)
        return results
    except sqlite3.Error as e:
        logging.error(f"Error fetching screener results for universe {universe}: {e}")
        return []
def update_ai_review_in_cache(db_conn: sqlite3.Connection, symbol: str, ai_data: Dict[str, Any], info: Optional[Dict[str, Any]] = None, universe: str = 'manual'):
    """
    Specifically updates the AI review portions of the screener cache.
    Targets ALL universes for the given symbol to keep them in sync.
    If no entries exist, creates a new entry with the specified universe.
    Also syncs metadata from 'info' if provided.
    """
    now_str = datetime.now().isoformat()
    scorecard = ai_data.get("scorecard", {})
    
    # 1. Update all existing entries
    update_sql = """
    UPDATE screener_cache SET
        ai_moat = ?,
        ai_financial_strength = ?,
        ai_predictability = ?,
        ai_growth = ?,
        ai_summary = ?,
        name = COALESCE(?, name),
        price = COALESCE(?, price),
        pe_ratio = COALESCE(?, pe_ratio),
        market_cap = COALESCE(?, market_cap),
        sector = COALESCE(?, sector),
        last_fiscal_year_end = COALESCE(?, last_fiscal_year_end),
        most_recent_quarter = COALESCE(?, most_recent_quarter),
        updated_at = ?
    WHERE symbol = ?
    """
    
    name = info.get("shortName") if info else None
    price = info.get("currentPrice") if info else None
    pe = info.get("trailingPE") if info else None
    mcap = info.get("marketCap") if info else None
    sector = info.get("sector") if info else None
    fy_end = info.get("lastFiscalYearEnd") if info else None
    mrq = info.get("mostRecentQuarter") if info else None

    try:
        cursor = db_conn.cursor()
        cursor.execute(update_sql, (
            scorecard.get("moat"),
            scorecard.get("financial_strength"),
            scorecard.get("predictability"),
            scorecard.get("growth"),
            ai_data.get("summary"),
            name, price, pe, mcap, sector,
            fy_end, mrq,
            now_str,
            symbol.upper()
        ))
        
        # 2. If no rows affected, create new entry using the provided universe
        if cursor.rowcount == 0:
            insert_sql = """
            INSERT INTO screener_cache (
                symbol, universe, ai_moat, ai_financial_strength, 
                ai_predictability, ai_growth, ai_summary, 
                name, price, pe_ratio, market_cap, sector, 
                last_fiscal_year_end, most_recent_quarter, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, (
                symbol.upper(),
                universe,
                scorecard.get("moat"),
                scorecard.get("financial_strength"),
                scorecard.get("predictability"),
                scorecard.get("growth"),
                ai_data.get("summary"),
                name, price, pe, mcap, sector,
                fy_end, mrq,
                now_str
            ))
        
        db_conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error updating AI review in cache for {symbol}: {e}")
        db_conn.rollback()

def update_intrinsic_value_in_cache(
    db_conn: sqlite3.Connection, 
    symbol: str, 
    intrinsic_value: Optional[float],
    margin_of_safety: Optional[float],
    last_fiscal_year_end: Optional[int] = None,
    most_recent_quarter: Optional[int] = None,
    info: Optional[Dict[str, Any]] = None
):
    """
    Updates intrinsic value and related metrics in the screener cache.
    Targets ALL universes for the given symbol to keep them in sync.
    If no entries exist, creates a 'manual' universe entry.
    Also syncs metadata from 'info' if provided.
    """
    now_str = datetime.now().isoformat()
    
    # 1. Update all existing entries
    update_sql = """
    UPDATE screener_cache SET
        intrinsic_value = ?,
        margin_of_safety = ?,
        last_fiscal_year_end = COALESCE(?, last_fiscal_year_end),
        most_recent_quarter = COALESCE(?, most_recent_quarter),
        name = COALESCE(?, name),
        price = COALESCE(?, price),
        pe_ratio = COALESCE(?, pe_ratio),
        market_cap = COALESCE(?, market_cap),
        sector = COALESCE(?, sector),
        updated_at = ?,
        valuation_details = COALESCE(?, valuation_details)
    WHERE symbol = ?
    """
    
    name = info.get("shortName") if info else None
    price = info.get("currentPrice") if info else None
    pe = info.get("trailingPE") if info else None
    mcap = info.get("marketCap") if info else None
    sector = info.get("sector") if info else None
    
    # Check if info contains valuation_details
    valuation_details = info.get("valuation_details") if info else None

    try:
        cursor = db_conn.cursor()
        cursor.execute(update_sql, (
            intrinsic_value,
            margin_of_safety,
            last_fiscal_year_end,
            most_recent_quarter,
            name, price, pe, mcap, sector,
            now_str,
            valuation_details,
            symbol.upper()
        ))
        
        # 2. If no rows affected, create new 'manual' entry
        if cursor.rowcount == 0:
            insert_sql = """
            INSERT INTO screener_cache (
                symbol, universe, intrinsic_value, margin_of_safety, 
                last_fiscal_year_end, most_recent_quarter, 
                name, price, pe_ratio, market_cap, sector, updated_at, valuation_details
            ) VALUES (?, 'manual', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """
            cursor.execute(insert_sql, (
                symbol.upper(),
                intrinsic_value,
                margin_of_safety,
                last_fiscal_year_end,
                most_recent_quarter,
                name, price, pe, mcap, sector,
                now_str,
                valuation_details
            ))
            
        db_conn.commit()
    except sqlite3.Error as e:
        logging.error(f"Error updating intrinsic value in cache for {symbol}: {e}")
        db_conn.rollback()


def initialize_database(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    conn = get_db_connection(db_path)
    if conn:
        create_transactions_table(conn)
    return conn


# ... (test block truncated/updated for brevity)
    pass
# End of file
