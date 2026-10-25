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
import json
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import threading
import investa_core

DB_FILENAME = "portfolio.db"
# Must match the highest migration applied in create_tables() below (currently the
# v15 migration adding the AI per-dimension analysis columns to screener_cache).
DB_SCHEMA_VERSION = 15


# --- Helper for JSON serialization with NaNs ---
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            # Convert NaN to None for JSON compatibility
            return float(obj) if np.isfinite(obj) else None
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # Let the base class default method raise the TypeError
        return super(NpEncoder, self).default(obj)


def get_database_path(db_filename: str = DB_FILENAME) -> str:
    """
    Determines the full path for the SQLite database file.
    Prioritizes centralized application data directory (db subfolder), falls back to CWD.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_database_path(db_filename)


def get_global_screener_db_path() -> str:
    """Returns the path to the global public screener cache database.

    This is a SHARED/PUBLIC database containing AI reviews and intrinsic value
    calculations for all stock universes (S&P 500, S&P 400, Russell 2000).
    It is NOT a user-specific database.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_global_screener_db_path()


_SCREENER_SCHEMA_INITED = False


def _ensure_screener_schema(conn: "sqlite3.Connection") -> None:
    """Idempotent CREATE TABLE + indexes for screener_cache on the global DB.

    Fresh installs that never went through the per-user portfolio-DB
    migration path otherwise wouldn't have the table — every prior code
    path created it as a side-effect of initialising a user DB.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    investa_core.db_ensure_screener_schema(conn)


def open_screener_db_conn() -> Optional["sqlite3.Connection"]:
    """Opens a fresh connection to the canonical (global) screener DB.

    All AI-review, intrinsic-value, and screener-row reads and writes go
    through this single store. Per-user portfolio DBs no longer participate
    in screener_cache; any legacy rows there should be merged via
    scripts/migrate_screener_cache_to_global.py.

    Caller is responsible for closing the connection.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_open_screener_db_conn()


_DB_CONN_CACHE = threading.local()


def is_path_on_cloud_drive(path: str) -> bool:
    """
    Checks if a given path is likely within a cloud-synced folder (Google Drive, Dropbox, OneDrive).
    These environments often conflict with SQLite's WAL mode and shared memory segments.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_is_path_on_cloud_drive(path)


def get_db_connection(
    db_path: Optional[str] = None,
    check_same_thread: bool = True,
    use_cache: bool = True,
    timeout: float = 30.0,
) -> Optional[sqlite3.Connection]:
    """Establishes a connection to the SQLite database, with thread-local caching."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_db_connection(
        db_path, check_same_thread, use_cache, timeout
    )


def create_transactions_table(conn: sqlite3.Connection):
    """Creates the transactions table and schema_version table if they don't exist."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    investa_core.db_create_transactions_table(conn)


def check_if_db_empty_and_csv_exists(
    db_conn: sqlite3.Connection, csv_file_path: str
) -> bool:
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_check_empty_and_csv_exists(db_conn, csv_file_path)


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
        except Exception:
            pass
        return migrated_count, error_count + 1
    return migrated_count, error_count


def load_all_transactions_from_db(
    db_conn: sqlite3.Connection,
    account_currency_map: Dict[str, str],
    default_currency: str,
) -> Tuple[Optional[pd.DataFrame], bool]:
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_load_all_transactions(
        db_conn, account_currency_map, default_currency
    )


_OUTFLOW_TYPES = frozenset(
    {
        "buy",
        "withdrawal",
        "fees",
        "fee",
        "tax",
        "withholding tax",
        "split",
        "stock split",
        "buy to cover",
    }
)


def _apply_total_amount_sign(transaction_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise the sign of 'Total Amount' to match the outflow/inflow convention.

    Outflow types (Buy, Withdrawal, Fees, Tax, Split, Buy To Cover) are stored as
    negative; inflow types (Sell, Deposit, Dividend, Interest, Transfer …) are positive.
    This is idempotent — already-signed values from the web manual-add form are unaffected
    because we always start from abs() and then re-apply the direction.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_apply_total_amount_sign(transaction_data)


def add_transaction_to_db(
    db_conn: sqlite3.Connection,
    transaction_data: Dict[str, Any],
    *,
    commit: bool = True,
) -> Tuple[bool, Optional[int]]:
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_add_transaction(db_conn, transaction_data, commit)


def update_transaction_in_db(
    db_conn: sqlite3.Connection, transaction_id: int, new_data_dict: Dict[str, Any]
) -> bool:
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_update_transaction(db_conn, transaction_id, new_data_dict)


def delete_transaction_from_db(
    db_conn: sqlite3.Connection, transaction_id: int
) -> bool:
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_delete_transaction(db_conn, transaction_id)


# The one built-in list. It is an ordinary watchlist found by name rather than a
# flagged row, so no schema change is needed and every existing watchlist
# screen shows it without knowing it is special.
FAVORITES_WATCHLIST_NAME = "Favorites"


def is_favorites_name(name: Optional[str]) -> bool:
    return (name or "").strip().casefold() == FAVORITES_WATCHLIST_NAME.casefold()


def get_or_create_favorites_watchlist(
    db_conn: sqlite3.Connection, user_id: Optional[int] = None
) -> Optional[Dict[str, Any]]:
    """
    The user's Favorites watchlist, created on first use.

    The oldest list with the reserved name wins, so a stray duplicate can never
    split one user's favourites across two lists.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_or_create_favorites(db_conn, user_id)


def get_watchlist_ids_for_symbol(db_conn: sqlite3.Connection, symbol: str) -> List[int]:
    """Ids of every watchlist that holds `symbol`."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_watchlist_ids_for_symbol(db_conn, symbol)


def get_all_watchlists(
    db_conn: sqlite3.Connection, user_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Fetches all available watchlists in the user-isolated database."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_all_watchlists(db_conn, user_id)


def create_watchlist(
    db_conn: sqlite3.Connection, name: str, user_id: Optional[int] = None
) -> Optional[int]:
    """Creates a new watchlist in the user-isolated database."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_create_watchlist(db_conn, name, user_id)


def rename_watchlist(
    db_conn: sqlite3.Connection,
    watchlist_id: int,
    new_name: str,
    user_id: Optional[int] = None,
) -> bool:
    """Renames an existing watchlist in the user-isolated database."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_rename_watchlist(db_conn, watchlist_id, new_name, user_id)


def delete_watchlist(
    db_conn: sqlite3.Connection, watchlist_id: int, user_id: Optional[int] = None
) -> bool:
    """Deletes a watchlist and all its items in the user-isolated database."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_delete_watchlist(db_conn, watchlist_id, user_id)


def add_to_watchlist(
    db_conn: sqlite3.Connection, symbol: str, note: str = "", watchlist_id: int = 1
) -> bool:
    """Adds a symbol to a specific watchlist."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_add_to_watchlist(db_conn, symbol, note, watchlist_id)


def remove_from_watchlist(
    db_conn: sqlite3.Connection, symbol: str, watchlist_id: int = 1
) -> bool:
    """Removes a symbol from a specific watchlist."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_remove_from_watchlist(db_conn, symbol, watchlist_id)


def get_watchlist(
    db_conn: sqlite3.Connection, watchlist_id: int = 1
) -> List[Dict[str, Any]]:
    """Fetches all items in a specific watchlist."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_get_watchlist(db_conn, watchlist_id)


def upsert_screener_results(
    results: List[Dict[str, Any]], universe: Optional[str] = None
):
    """Batch updates/inserts screener results into the global screener cache."""
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    investa_core.db_upsert_screener_results(results, universe)


def refresh_screener_rows_by_symbol(results: List[Dict[str, Any]]):
    """Refresh existing screener_cache rows for the given symbols (any universe).

    Performs UPDATE-by-symbol only — never INSERTs a new row. Used by the
    'All Database Stocks' screener path: those results were derived from
    existing rows in canonical universes (sp500, russell2000, manual, …),
    so price/IV refreshes should land on the canonical rows rather than
    create a redundant 'all' duplicate.
    """
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    investa_core.db_refresh_screener_rows(results)


def get_cached_screener_results(symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    """Retrieves cached screener data for a list of symbols from the global DB."""
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    return investa_core.db_get_cached_screener_results(symbols)


def get_screener_results_by_universe(universe: str) -> List[Dict[str, Any]]:
    """Retrieves all cached screener results for a specific universe tag from the global DB."""
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    return investa_core.db_get_screener_results_by_universe(universe)


def get_all_distinct_screener_results() -> List[Dict[str, Any]]:
    """
    Retrieves the latest cached screener results for all unique symbols in the
    global screener DB (one row per symbol — the one with the most recent
    updated_at across all universes).
    """
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    return investa_core.db_get_all_distinct_screener_results()


_SYNTHETIC_SYMBOLS = frozenset(
    {
        "PORTFOLIO-WIDE",
        "CASH-AGGREGATE",
        "TOTAL",
        "PORTFOLIO",
    }
)


def _is_synthetic_symbol(symbol: str) -> bool:
    """Returns True for internal/synthetic symbols that are not real tickers."""
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    return investa_core.db_is_synthetic_symbol(symbol)


def update_ai_review_in_cache(
    symbol: str,
    ai_data: Dict[str, Any],
    info: Optional[Dict[str, Any]] = None,
    universe: str = "manual",
):
    """
    Updates the AI review portions of the global screener cache.
    Targets ALL universes for the given symbol to keep them in sync.
    If no entries exist, creates a new entry with the specified universe.
    Also syncs metadata from 'info' if provided.
    """
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    investa_core.db_update_ai_review_in_cache(symbol, ai_data, info, universe)


def update_intrinsic_value_in_cache(
    symbol: str,
    intrinsic_value: Optional[float],
    margin_of_safety: Optional[float],
    last_fiscal_year_end: Optional[int] = None,
    most_recent_quarter: Optional[int] = None,
    info: Optional[Dict[str, Any]] = None,
    universe: str = "manual",
):
    """
    Updates intrinsic value and related metrics in the global screener cache.
    Targets ALL universes for the given symbol to keep them in sync.
    If no entries exist, creates a new entry with the specified universe.
    Also syncs metadata from 'info' if provided.
    """
    # Ported to Rust: rust/investa_core/src/db_screener.rs
    investa_core.db_update_intrinsic_value_in_cache(
        symbol,
        intrinsic_value,
        margin_of_safety,
        last_fiscal_year_end,
        most_recent_quarter,
        info,
        universe,
    )


def initialize_database(db_path: Optional[str] = None) -> Optional[sqlite3.Connection]:
    """Initializes the Portfolio DatabaseSchema."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_initialize_database(db_path)


def initialize_global_database() -> Optional[sqlite3.Connection]:
    """Initializes the Global Database Schema (Users)."""
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_initialize_global_database()


# Alias for backward compatibility if needed
init_db = initialize_database

# End of file


def connect_readonly(db_path: str, timeout: float = 60.0) -> sqlite3.Connection:
    """A read-only connection that works whatever journal mode the file is in.

    `file:...?mode=ro` is the obvious way to promise "I will not write", and it
    fails outright on a WAL database when no `-shm` file exists — SQLite needs
    to create that shared-memory index before it can read, and mode=ro forbids
    creating anything. Scripts using it appear to work for as long as something
    else holds the database open, then break the first time they run alone.

    So: try the strict form, and fall back to an ordinary connection with
    `query_only` set, which is the same promise enforced a different way.
    """
    # Ported to Rust: rust/investa_core/src/db_portfolio.rs
    return investa_core.db_connect_readonly(db_path, timeout)
