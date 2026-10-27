# -*- coding: utf-8 -*-
"""
The SQL the routes, the services and the AI workers used to run inline.

Each function is one statement (or one fixed group of statements) moved out
of its caller unchanged, so the SQL lives in one place. Connections,
transactions (`commit`, `rollback`, `with conn:`), retries and HTTP errors
stay with the callers, exactly where they were.
"""

import sqlite3
from datetime import date, datetime  # noqa: F401 -- read by the Rust port at call time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

import investa_core

# --- users (the global database) -----------------------------------------------


def find_user_id(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """The id row of the user called `username`, or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_find_user_id(conn, username)


def insert_user(
    conn: sqlite3.Connection, username: str, hashed_password: str, created_at: str
) -> Optional[int]:
    """Insert a user row; its rowid."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_insert_user(conn, username, hashed_password, created_at)


def user_credentials(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """(id, username, hashed_password) for a login, or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_user_credentials(conn, username)


def user_record(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """(id, username, is_active, created_at, alias), or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_user_record(conn, username)


def set_user_alias(conn: sqlite3.Connection, user_id: int, alias: str) -> None:
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_set_user_alias(conn, user_id, alias)


def delete_user(conn: sqlite3.Connection, user_id: int) -> int:
    """Delete the user row; the rows deleted."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_delete_user(conn, user_id)


def user_password_hash(conn: sqlite3.Connection, user_id: int) -> Optional[tuple]:
    """(hashed_password,), or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_user_password_hash(conn, user_id)


def set_user_password(
    conn: sqlite3.Connection, user_id: int, hashed_password: str
) -> None:
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_set_user_password(conn, user_id, hashed_password)


# --- a user's portfolio database ------------------------------------------------
def set_holding_tags(
    conn: sqlite3.Connection, tags: str, symbol: str, account: str
) -> int:
    """Tag every transaction of one holding; the rows updated."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_set_holding_tags(conn, tags, symbol, account)


# The columns a pending transaction carries, in the order approval copies them.
PENDING_COLUMNS = [
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
    "Tags",
    "ExternalID",
    "user_id",
]


def stage_pending_transaction(
    conn: sqlite3.Connection, tx_data: Dict[str, Any], user_id: int
) -> Tuple[bool, Any]:
    """Stage one synced transaction for review.

    (False, "duplicate_main" | "duplicate_pending") when its ExternalID is
    already booked or already staged; otherwise (True, the new row's id).
    """
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_stage_pending_transaction(conn, tx_data, user_id)


def pending_transaction_records(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Every pending transaction, newest first, as records."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_pending_transaction_records(conn)


def approve_pending_transactions(
    conn: sqlite3.Connection, ids: Iterable[int], accounts: Dict[int, str]
) -> int:
    """Move pending rows into the book, with an optional destination account
    each; a row already booked under its ExternalID is dropped instead. The
    rows approved. The caller commits or rolls back."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_approve_pending_transactions(conn, ids, accounts)


def reject_pending_transactions(conn: sqlite3.Connection, ids: Sequence[int]) -> int:
    """Discard pending rows; the rows deleted."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_reject_pending_transactions(conn, ids)


def begin_immediate(conn: sqlite3.Connection) -> None:
    """Take the write lock now, so concurrent submissions serialize."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_begin_immediate(conn)


def find_receipt(
    conn: sqlite3.Connection, user_id: int, endpoint: str, request_key: str
) -> Optional[tuple]:
    """(fingerprint, response) of an earlier submission under this key, or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_find_receipt(conn, user_id, endpoint, request_key)


def insert_receipt(
    conn: sqlite3.Connection,
    user_id: int,
    endpoint: str,
    request_key: str,
    fingerprint: str,
    response: str,
) -> None:
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_insert_receipt(
        conn, user_id, endpoint, request_key, fingerprint, response
    )


_SNAPSHOT_INSERT = """
                            INSERT INTO portfolio_snapshots (snapshot_date, account, total_value, total_cost, total_gain, total_return_pct, twr, irr, created_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """


def replace_portfolio_snapshots(
    conn: sqlite3.Connection, snapshot_date: str, rows: Iterable[tuple]
) -> None:
    """Replace one day's snapshot rows: (snapshot_date, account, total_value,
    total_cost, total_gain, total_return_pct, twr, irr, created_at) each. The
    caller holds the transaction."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_replace_portfolio_snapshots(conn, snapshot_date, rows)


def symbol_currency_pairs(conn: sqlite3.Connection) -> List[tuple]:
    """Each (Symbol, Local Currency) pair the book holds."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_symbol_currency_pairs(conn)


# --- the market archive ---------------------------------------------------------
def data_quality_table_present(conn: sqlite3.Connection) -> bool:
    """Whether the data-quality scan has ever created its table."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_data_quality_table_present(conn)


def touch_sync_metadata(
    conn: sqlite3.Connection, symbols: Iterable[str], synced_at: str
) -> None:
    """Stamp `last_synced` for each symbol, leaving its other columns alone.

    INSERT OR REPLACE would delete the row and re-insert it, resetting every
    column not named here — a converted symbol's price_basis among them.
    """
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_touch_sync_metadata(conn, symbols, synced_at)


# --- the EDGAR fact store ---------------------------------------------------------
def facts_for(
    conn: sqlite3.Connection, tags: Sequence[str], ciks: Sequence[str]
) -> List[tuple]:
    """(cik, tag, period_end, val) of the given tags for the given filers."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_facts_for(conn, tags, ciks)


# --- the screener cache -----------------------------------------------------------
def screener_rows(conn: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    """The rows of a validated SELECT on the screener cache, as dicts."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_screener_rows(conn, sql)


def cached_ai_review(
    conn: sqlite3.Connection, symbol: str, universe: str
) -> Optional[tuple]:
    """(ai_summary, last_fiscal_year_end, most_recent_quarter), or None."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_cached_ai_review(conn, symbol, universe)


def symbols_missing_ai_review(conn: sqlite3.Connection) -> List[tuple]:
    """(symbol, universe) of every row without a usable AI summary."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_symbols_missing_ai_review(conn)


def ai_review_frame(
    conn: sqlite3.Connection, dimensions: Sequence[str]
) -> pd.DataFrame:
    """symbol, the AI dimensions and updated_at, for every row scored on any."""
    # Ported to Rust: rust/investa_core/src/ss.rs
    return investa_core.ss_ai_review_frame(conn, dimensions)
