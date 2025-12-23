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
from datetime import datetime, date
import numpy as np
import os
import logging
from typing import Optional, Dict, Any, Tuple, Union, List
import pandas as pd
import traceback
import config

DB_FILENAME = "investa_transactions.db"
DB_SCHEMA_VERSION = 2


def get_database_path(db_filename: str = DB_FILENAME) -> str:
    """
    Determines the full path for the SQLite database file using the
    centralized application data directory.
    """
    app_data_dir = config.get_app_data_dir()
    return os.path.join(app_data_dir, db_filename)


def get_db_connection(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database."""
    if db_path is None:
        db_path = get_database_path()
    try:
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON;")
        logging.info(f"Successfully connected to database: {db_path}")
        return conn
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
        "To Account" TEXT
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
               "Total Amount", Commission, Account, "Split Ratio", Note, "Local Currency", "To Account"
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
        "To Account",
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
        if col_name == "Date":
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


def initialize_database(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    conn = get_db_connection(db_path)
    if conn:
        create_transactions_table(conn)
    return conn


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)-8s] %(module)s.%(funcName)s:%(lineno)d - %(message)s",
    )
    test_db_filename = "test_investa_transactions_dbutils.db"
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

        dummy_csv_path = "dummy_transactions_for_db_migration.csv"
        csv_content_for_test = {
            "Date (MMM DD, YYYY)": [
                "Jan 01, 2023",
                "Jan 02, 2023",
                "Jan 03, 2023",
                "Jan 04, 2023",
                "Jan 01, 2023",
                "Jan 02, 2023",
                "Jan 03, 2023",
                "Jan 04, 2023",
            ],
            "Transaction Type": ["Buy", "Sell", "Dividend", "Fees"],
            "Stock / ETF Symbol": ["AAPL", "MSFT", "AAPL", "MSFT"],
            "Quantity of Units": ["10.0", "5.0", "", "N/A"],
            "Amount per unit": ["150.0", "250.0", "", ""],
            "Total Amount": ["1500.0", "1250.0", "25.0", ""],
            "Fees": ["5.0", "5.0", "0.0", "1.25"],
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
        ]
        pd.DataFrame(csv_content_for_test).to_csv(dummy_csv_path, index=False)
        print(f"\nAttempting to migrate '{dummy_csv_path}'...")
        test_account_map = {"Brokerage": "USD", "IRA": "USD"}
        test_default_currency = "CAD"

        if check_if_db_empty_and_csv_exists(conn, dummy_csv_path):
            print("DB is empty and CSV exists, proceeding with migration.")
            mig_count, err_count = migrate_csv_to_db(
                dummy_csv_path, conn, test_account_map, test_default_currency
            )
            print(f"Migration test result: Migrated {mig_count}, Errors {err_count}")
            if mig_count > 0:
                df_from_db, load_success = load_all_transactions_from_db(
                    conn, test_account_map, test_default_currency
                )
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
