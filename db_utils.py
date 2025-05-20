# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          db_utils.py
 Purpose:       Database utility functions for SQLite.
                Handles database connection, schema creation, and path management.

 Author:        Kit Matan and Google Gemini
 Author Email:  kittiwit@gmail.com

 Created:       02/05/2025
 Copyright:     (c) Kittiwit Matan 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""
import sqlite3
from datetime import datetime, date  # Added import for datetime and date
import numpy as np  # Added import for numpy
import os
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import traceback

# --- MODIFIED: Defer import of load_and_clean_transactions to the function that uses it ---
# This helps avoid direct import cycle issues at the module level.
DATA_LOADER_AVAILABLE = ()

try:
    from PySide6.QtCore import QStandardPaths

    PYSIDE_AVAILABLE = True
except ImportError:
    logging.warning(
        "PySide6.QtCore.QStandardPaths not found in db_utils.py. Database path will be relative."
    )
    QStandardPaths = None
    PYSIDE_AVAILABLE = False

DB_FILENAME = "investa_transactions.db"
DB_SCHEMA_VERSION = 1


def get_database_path(db_filename: str = DB_FILENAME) -> str:
    """
    Determines the full path for the SQLite database file.
    Uses QStandardPaths.AppDataLocation if PySide6 is available and successful,
    otherwise defaults to a subdirectory in the user's home directory,
    and finally to the current working directory as a last resort.

    Args:
        db_filename (str): The name of the database file.

    Returns:
        str: The absolute path to the database file.
    """
    preferred_path = None
    if PYSIDE_AVAILABLE and QStandardPaths:
        # Prefer AppDataLocation as it's often more appropriate for user data files like a DB.
        # AppConfigLocation is more for config files.
        app_data_dir = QStandardPaths.writableLocation(QStandardPaths.AppDataLocation)
        if not app_data_dir:  # Fallback to AppConfigLocation
            app_data_dir = QStandardPaths.writableLocation(
                QStandardPaths.AppConfigLocation
            )

        if app_data_dir:
            # QStandardPaths usually includes the app name if set via QApplication.setApplicationName()
            # Ensure the base directory exists before joining the filename
            try:
                os.makedirs(app_data_dir, exist_ok=True)
                preferred_path = os.path.join(app_data_dir, db_filename)
            except Exception as e:
                logging.error(f"Could not create directory {app_data_dir}: {e}")
        else:
            logging.warning(
                "QStandardPaths.AppDataLocation/AppConfigLocation returned empty. Trying user home directory."
            )

    if not preferred_path:
        try:
            # Fallback to a directory in user's home
            app_name_for_folder = (
                "InvestaApp"  # Consistent name, can be from config.APP_NAME
            )
            home_dir = os.path.expanduser("~")
            fallback_app_dir = os.path.join(home_dir, f".{app_name_for_folder.lower()}")
            os.makedirs(fallback_app_dir, exist_ok=True)
            preferred_path = os.path.join(fallback_app_dir, db_filename)
        except Exception as e:
            logging.warning(
                f"Could not determine or create user home directory path '{fallback_app_dir}': {e}"
            )

    if not preferred_path:
        # Last resort: current working directory
        preferred_path = os.path.join(os.getcwd(), db_filename)
        logging.warning(
            "Could not determine standard application data path or user home path. Using current working directory for DB."
        )
        # Ensure current working directory is writable for the DB file itself (os.makedirs not needed for file in CWD)

    return preferred_path


def get_db_connection(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database."""
    if db_path is None:
        db_path = get_database_path()
    try:
        # Ensure the directory for the database file exists before connecting
        db_dir = os.path.dirname(db_path)
        if db_dir:  # If db_path includes a directory component
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")  # Good practice if using foreign keys
        logging.info(f"Successfully connected to database: {db_path}")
        return conn
    except sqlite3.Error as e:
        logging.error(f"Error connecting to database at {db_path}: {e}", exc_info=True)
        return None
    except (
        OSError
    ) as e_os:  # Catch errors like permission denied for directory creation
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
        "Local Currency" TEXT NOT NULL
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

        if current_db_version < DB_SCHEMA_VERSION:
            logging.info(
                f"Database schema version is {current_db_version}, target is {DB_SCHEMA_VERSION}. Applying migrations if any."
            )
            # Placeholder for actual migration logic if schema changes
            # e.g., if DB_SCHEMA_VERSION == 2 and current_db_version == 1:
            #   cursor.execute("ALTER TABLE transactions ADD COLUMN NewColumn TEXT;")
            # After successful migration steps:
            cursor.execute(
                "INSERT OR REPLACE INTO schema_version (version, applied_on) VALUES (?, ?)",
                (DB_SCHEMA_VERSION, datetime.now().isoformat()),
            )
            logging.info(f"Database schema updated to version {DB_SCHEMA_VERSION}.")
        elif current_db_version == 0:  # Fresh database
            cursor.execute(
                "INSERT INTO schema_version (version, applied_on) VALUES (?, ?)",
                (DB_SCHEMA_VERSION, datetime.now().isoformat()),
            )
            logging.info(f"Initialized database schema to version {DB_SCHEMA_VERSION}.")

        conn.commit()
        logging.info(
            "Transactions and schema_version tables checked/created/updated successfully."
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
    global DATA_LOADER_AVAILABLE  # Check the flag
    try:
        # Local import to attempt resolving dependency only when function is called
        from data_loader import load_and_clean_transactions
    except ImportError:
        logging.error(
            "CRITICAL: data_loader.load_and_clean_transactions could not be imported in migrate_csv_to_db. Migration failed."
        )
        DATA_LOADER_AVAILABLE = False  # Update flag if import fails here

    if not DATA_LOADER_AVAILABLE:
        return 0, 1

    logging.info(f"Starting migration from CSV: {csv_file_path}")
    migrated_count = 0
    error_count = 0

    try:
        transactions_df, _, _, _, load_error, _, _ = load_and_clean_transactions(
            csv_file_path,
            account_currency_map,
            default_currency,
            is_db_source=False,  # Explicitly False for CSV
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
            "Local Currency",
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


# ... (load_all_transactions_from_db, add_transaction_to_db, update_transaction_in_db, delete_transaction_from_db, initialize_database)
# ... these functions remain as previously provided ...
def load_all_transactions_from_db(
    db_conn: sqlite3.Connection,
    account_currency_map: Dict[str, str],  # ADDED
    default_currency: str,  # ADDED
) -> Tuple[Optional[pd.DataFrame], bool]:
    logging.info("Loading all transactions from the database.")
    try:
        query = """
        SELECT id as original_index, Date, Type, Symbol, Quantity, "Price/Share",
               "Total Amount", Commission, Account, "Split Ratio", Note, "Local Currency"
        FROM transactions
        ORDER BY Date, original_index;
        """
        df = pd.read_sql_query(query, db_conn, parse_dates=["Date"])

        # --- ADDED: Clean Local Currency after loading from DB ---
        if "Local Currency" in df.columns and "Account" in df.columns:
            # Check for None, NaN, empty string, or the string "<NA>"
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
            df["Local Currency"] = df[
                "Local Currency"
            ].str.upper()  # Standardize to uppercase
        # --- END ADDED ---

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
    ]
    quoted_db_columns = [
        f'"{col}"' if "/" in col or " " in col else col for col in db_column_order
    ]
    placeholders_list = []
    data_for_sql_ordered = []

    for col_name in db_column_order:
        placeholder_name = (
            col_name.replace("/", "_per_").replace(" ", "_").replace(".", "_")
        )  # Make valid placeholder
        placeholders_list.append(f":{placeholder_name}")
        value = transaction_data.get(col_name)
        if col_name == "Date":
            if isinstance(value, (datetime, date)):
                data_for_sql_ordered.append(value.strftime("%Y-%m-%d"))
            elif isinstance(value, str):
                try:  # Validate YYYY-MM-DD format if string
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

    # Create dict for execute using placeholders and ordered values
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
    values_for_sql: Dict[str, Any] = {}  # Ensure type for values

    for key, value in new_data_dict.items():
        # Key from new_data_dict is already the DB column name
        placeholder = key.replace("/", "_per_").replace(" ", "_").replace(".", "_")
        set_clauses.append(f'"{key}" = :{placeholder}')
        if key == "Date":
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
    values_for_sql["id_placeholder"] = (
        transaction_id  # Use a different placeholder for id
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


def initialize_database(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    conn = get_db_connection(db_path)
    if conn:
        create_transactions_table(conn)
    return conn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,  # Use DEBUG for more verbose output during standalone test
        format="%(asctime)s [%(levelname)-8s] %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    )
    # Test Database Setup
    test_db_filename = (
        "test_investa_transactions_dbutils.db"  # Unique name for this test
    )
    db_file_path_test = get_database_path(test_db_filename)
    print(f"Test DB path set to: {db_file_path_test}")
    if os.path.exists(db_file_path_test):
        try:
            os.remove(db_file_path_test)
            print(f"Removed old test database: {db_file_path_test}")
        except Exception as e:
            print(f"Error removing old test database {db_file_path_test}: {e}")
            exit(1)

    conn = initialize_database(db_file_path_test)
    if not conn:
        print(f"Database initialization FAILED for {db_file_path_test}.")
        exit(1)
    print(f"Database initialized successfully at {db_file_path_test}.")

    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='transactions';"
        )
        if cursor.fetchone():
            print("Verified 'transactions' table exists.")
        else:
            print("Error: 'transactions' table does NOT exist after initialization.")
        cursor.execute("SELECT version FROM schema_version;")
        version_row = cursor.fetchone()
        if version_row and version_row[0] == DB_SCHEMA_VERSION:
            print(
                f"Verified 'schema_version' table exists and version is {DB_SCHEMA_VERSION}."
            )
        else:
            print("Error: 'schema_version' table or correct version NOT found.")

        # Test Migration
        dummy_csv_path = "dummy_transactions_for_db_migration.csv"
        csv_content_for_test = {
            # MODIFIED: Added Local Currency column, including an empty one
            "Date (MMM DD, YYYY)": [
                "Jan 01, 2023",
                "Jan 02, 2023",
                "Jan 03, 2023",
                "Jan 04, 2023",
                "Jan 01, 2023",
                "Jan 02, 2023",
                "Jan 03, 2023",
                "Jan 04, 2023",
            ],  # Added one more row
            "Transaction Type": ["Buy", "Sell", "Dividend", "Fees"],
            "Stock / ETF Symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "Quantity of Units": ["10.0", "5.0", "", "N/A"],  # N/A for fees qty
            "Amount per unit": ["150.0", "250.0", "", ""],  # Price not for fees
            "Total Amount": ["1500.0", "1250.0", "25.0", ""],  # Total not for fees
            "Fees": ["5.0", "5.0", "0.0", "1.25"],  # Fee amount for 'Fees' type
            "Investment Account": ["Brokerage", "Brokerage", "Brokerage", "IRA"],
            "Split Ratio (new shares per old share)": ["", "", "", ""],
            "Note": ["Buy Apple", "Sell Microsoft", "Apple Dividend", "Account Fee"],
        }
        csv_content_for_test["Local Currency"] = [
            "USD",
            "USD",
            "USD",
            "",
            "EUR",
            "EUR",
            "EUR",
            "",
        ]  # ADDED
        pd.DataFrame(csv_content_for_test).to_csv(dummy_csv_path, index=False)
        print(f"\nAttempting to migrate '{dummy_csv_path}'...")
        test_account_map = {"Brokerage": "USD", "IRA": "USD"}
        test_default_currency = "CAD"  # Use a different default to test mapping

        if check_if_db_empty_and_csv_exists(conn, dummy_csv_path):
            print("DB is empty and CSV exists, proceeding with migration.")
            mig_count, err_count = migrate_csv_to_db(
                dummy_csv_path, conn, test_account_map, test_default_currency
            )
            print(f"Migration test result: Migrated {mig_count}, Errors {err_count}")
            if mig_count > 0:
                # MODIFIED: Pass map and default to load_all_transactions_from_db
                df_from_db, load_success = load_all_transactions_from_db(
                    conn, test_account_map, test_default_currency
                )
                # END MODIFIED
                if load_success and df_from_db is not None:
                    print(
                        f"Successfully loaded {len(df_from_db)} rows from DB post-migration."
                    )
                    print("DB content (head):\n", df_from_db.head().to_string())
                    if "Local Currency" in df_from_db.columns:
                        print(
                            "Local Currencies in DB:",
                            df_from_db["Local Currency"].unique(),
                        )
                else:
                    print(
                        "Failed to load data from DB after migration for verification."
                    )
        else:
            print("Skipping migration test (DB not empty or CSV missing).")

        # Test Add, Update, Delete
        print("\n--- Testing Add, Update, Delete ---")
        test_tx_data_add = {
            "Date": "2024-03-15",
            "Type": "Buy",
            "Symbol": "GOOG",
            "Quantity": 5.0,
            "Price/Share": 140.0,
            "Total Amount": 700.0,
            "Commission": 2.50,
            "Account": "NewAcc",
            "Split Ratio": None,
            "Note": "Test Add GOOG",
            "Local Currency": "EUR",
        }
        add_success, new_id = add_transaction_to_db(conn, test_tx_data_add)
        if add_success and new_id is not None:
            print(f"Added transaction with ID: {new_id}")
            test_tx_data_update = {
                "Note": "GOOG Buy Updated Note",
                "Quantity": 5.5,
                "Date": "2024-03-16",
            }
            update_success = update_transaction_in_db(conn, new_id, test_tx_data_update)
            if update_success:
                cursor.execute(
                    "SELECT Note, Quantity, Date FROM transactions WHERE id = ?",
                    (new_id,),
                )
                print(f"  Verified update: {cursor.fetchone()}")
            else:
                print(f"Failed to update transaction ID: {new_id}")
            delete_success = delete_transaction_from_db(conn, new_id)
            if delete_success:
                cursor.execute(
                    "SELECT COUNT(*) FROM transactions WHERE id = ?", (new_id,)
                )
                print(f"  Count for ID {new_id} after delete: {cursor.fetchone()[0]}")
            else:
                print(f"Failed to delete transaction ID: {new_id}")
        else:
            print("Failed to add test transaction.")

        if os.path.exists(dummy_csv_path):
            try:
                os.remove(dummy_csv_path)
                print(f"Cleaned up dummy CSV: {dummy_csv_path}")
            except Exception as e:
                print(f"Error removing dummy CSV {dummy_csv_path}: {e}")
    except Exception as e_main_test:
        print(
            f"An unexpected error occurred in db_utils __main__ test block: {e_main_test}"
        )
        traceback.print_exc()
    finally:
        if conn:
            conn.close()
            print("Database connection closed.")
        if os.path.exists(db_file_path_test):
            try:
                os.remove(db_file_path_test)
                print(f"Cleaned up test database: {db_file_path_test}")
            except Exception as e_remove_final:
                print(
                    f"Error removing test database {db_file_path_test} on final cleanup: {e_remove_final}"
                )
