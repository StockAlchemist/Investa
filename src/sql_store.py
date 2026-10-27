# -*- coding: utf-8 -*-
"""
The SQL the routes, the services and the AI workers used to run inline.

Each function is one statement (or one fixed group of statements) moved out
of its caller unchanged, so the SQL lives in one place. Connections,
transactions (`commit`, `rollback`, `with conn:`), retries and HTTP errors
stay with the callers, exactly where they were.
"""

import sqlite3
from datetime import date, datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

# --- users (the global database) -----------------------------------------------


def find_user_id(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """The id row of the user called `username`, or None."""
    return conn.execute(
        "SELECT id FROM users WHERE username = ?", (username,)
    ).fetchone()


def insert_user(
    conn: sqlite3.Connection, username: str, hashed_password: str, created_at: str
) -> Optional[int]:
    """Insert a user row; its rowid."""
    return conn.execute(
        "INSERT INTO users (username, hashed_password, created_at) VALUES (?, ?, ?)",
        (username, hashed_password, created_at),
    ).lastrowid


def user_credentials(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """(id, username, hashed_password) for a login, or None."""
    return conn.execute(
        "SELECT id, username, hashed_password FROM users WHERE username = ?",
        (username,),
    ).fetchone()


def user_record(conn: sqlite3.Connection, username: str) -> Optional[tuple]:
    """(id, username, is_active, created_at, alias), or None."""
    return conn.execute(
        "SELECT id, username, is_active, created_at, alias FROM users WHERE username = ?",
        (username,),
    ).fetchone()


def set_user_alias(conn: sqlite3.Connection, user_id: int, alias: str) -> None:
    conn.execute("UPDATE users SET alias = ? WHERE id = ?", (alias, user_id))


def delete_user(conn: sqlite3.Connection, user_id: int) -> int:
    """Delete the user row; the rows deleted."""
    return conn.execute("DELETE FROM users WHERE id = ?", (user_id,)).rowcount


def user_password_hash(conn: sqlite3.Connection, user_id: int) -> Optional[tuple]:
    """(hashed_password,), or None."""
    return conn.execute(
        "SELECT hashed_password FROM users WHERE id = ?", (user_id,)
    ).fetchone()


def set_user_password(
    conn: sqlite3.Connection, user_id: int, hashed_password: str
) -> None:
    conn.execute(
        "UPDATE users SET hashed_password = ? WHERE id = ?",
        (hashed_password, user_id),
    )


# --- a user's portfolio database ------------------------------------------------


def set_holding_tags(
    conn: sqlite3.Connection, tags: str, symbol: str, account: str
) -> int:
    """Tag every transaction of one holding; the rows updated."""
    return conn.execute(
        "UPDATE transactions SET Tags = ? WHERE Symbol = ? AND Account = ?",
        (tags, symbol, account),
    ).rowcount


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
    ext_id = tx_data.get("ExternalID")
    cursor = conn.cursor()
    if ext_id:
        # Check main table
        cursor.execute("SELECT id FROM transactions WHERE ExternalID = ?", (ext_id,))
        if cursor.fetchone():
            return False, "duplicate_main"
        # Check pending table
        cursor.execute(
            "SELECT id FROM pending_transactions WHERE ExternalID = ?",
            (ext_id,),
        )
        if cursor.fetchone():
            return False, "duplicate_pending"

    db_cols = PENDING_COLUMNS
    placeholders = ", ".join(
        [f":{c.replace('/', '_').replace(' ', '_')}" for c in db_cols]
    )
    cols_str = ", ".join([f'"{c}"' for c in db_cols])

    sql = f"INSERT INTO pending_transactions ({cols_str}) VALUES ({placeholders});"
    sql_data = {}
    for col in db_cols:
        val = tx_data.get(col) if col != "user_id" else user_id

        # Fallback for Total Amount if missing (or provided as 'Amount')
        if col == "Total Amount" and val is None:
            val = tx_data.get("Amount")  # Try alternate key
            if val is None:  # Calculate
                q = tx_data.get("Quantity", 0)
                p = tx_data.get("Price/Share", 0)
                c = tx_data.get("Commission", 0)
                t = tx_data.get("Type", "").upper()
                if q and p:
                    val = (q * p) + (c if t == "BUY" else -c)

        if col == "Type" and isinstance(val, str):
            val = val.strip().title()
        if col == "Date" and isinstance(val, (datetime, date)):
            val = val.strftime("%Y-%m-%d")
        sql_data[col.replace("/", "_").replace(" ", "_")] = (
            None if pd.isna(val) else val
        )

    cursor.execute(sql, sql_data)
    return True, cursor.lastrowid


def pending_transaction_records(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Every pending transaction, newest first, as records."""
    query = "SELECT * FROM pending_transactions ORDER BY Date DESC"
    df = pd.read_sql_query(query, conn)
    return df.to_dict(orient="records")


def approve_pending_transactions(
    conn: sqlite3.Connection, ids: Iterable[int], accounts: Dict[int, str]
) -> int:
    """Move pending rows into the book, with an optional destination account
    each; a row already booked under its ExternalID is dropped instead. The
    rows approved. The caller commits or rolls back."""
    cursor = conn.cursor()
    approved_count = 0
    cols = PENDING_COLUMNS
    cols_str = ", ".join([f'"{c}"' for c in cols])

    ext_id_idx = cols.index("ExternalID")
    for p_id in ids:
        # Fetch from pending
        cursor.execute(
            f"SELECT {cols_str} FROM pending_transactions WHERE id = ?", (p_id,)
        )
        row = cursor.fetchone()
        if row:
            row = list(row)
            if p_id in accounts:
                row[cols.index("Account")] = accounts[p_id]
            ext_id = row[ext_id_idx]
            if ext_id:
                cursor.execute(
                    "SELECT 1 FROM transactions WHERE ExternalID = ? LIMIT 1",
                    (ext_id,),
                )
                if cursor.fetchone():
                    cursor.execute(
                        "DELETE FROM pending_transactions WHERE id = ?", (p_id,)
                    )
                    continue
            # Insert into main table
            placeholders = ", ".join(["?"] * len(cols))
            cursor.execute(
                f"INSERT INTO transactions ({cols_str}) VALUES ({placeholders})",
                row,
            )
            # Delete from pending
            cursor.execute("DELETE FROM pending_transactions WHERE id = ?", (p_id,))
            approved_count += 1
    return approved_count


def reject_pending_transactions(conn: sqlite3.Connection, ids: Sequence[int]) -> int:
    """Discard pending rows; the rows deleted."""
    return conn.execute(
        f"DELETE FROM pending_transactions WHERE id IN ({','.join(['?'] * len(ids))})",
        (*ids,),
    ).rowcount


def begin_immediate(conn: sqlite3.Connection) -> None:
    """Take the write lock now, so concurrent submissions serialize."""
    conn.execute("BEGIN IMMEDIATE")


def find_receipt(
    conn: sqlite3.Connection, user_id: int, endpoint: str, request_key: str
) -> Optional[tuple]:
    """(fingerprint, response) of an earlier submission under this key, or None."""
    return conn.execute(
        "SELECT fingerprint, response FROM transaction_receipts "
        "WHERE user_id=? AND endpoint=? AND request_key=?",
        (user_id, endpoint, request_key),
    ).fetchone()


def insert_receipt(
    conn: sqlite3.Connection,
    user_id: int,
    endpoint: str,
    request_key: str,
    fingerprint: str,
    response: str,
) -> None:
    conn.execute(
        "INSERT INTO transaction_receipts (user_id, endpoint, request_key, fingerprint, response) VALUES (?, ?, ?, ?, ?)",
        (user_id, endpoint, request_key, fingerprint, response),
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
    cursor = conn.cursor()
    cursor.execute(
        "DELETE FROM portfolio_snapshots WHERE snapshot_date=?",
        (snapshot_date,),
    )
    for row in rows:
        cursor.execute(_SNAPSHOT_INSERT, row)


def symbol_currency_pairs(conn: sqlite3.Connection) -> List[tuple]:
    """Each (Symbol, Local Currency) pair the book holds."""
    return conn.execute(
        "SELECT DISTINCT Symbol, \"Local Currency\" FROM transactions WHERE Symbol IS NOT NULL AND Symbol != ''"
    ).fetchall()


# --- the market archive ---------------------------------------------------------


def data_quality_table_present(conn: sqlite3.Connection) -> bool:
    """Whether the data-quality scan has ever created its table."""
    return bool(
        conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='data_quality'"
        ).fetchone()
    )


def touch_sync_metadata(
    conn: sqlite3.Connection, symbols: Iterable[str], synced_at: str
) -> None:
    """Stamp `last_synced` for each symbol, leaving its other columns alone.

    INSERT OR REPLACE would delete the row and re-insert it, resetting every
    column not named here — a converted symbol's price_basis among them.
    """
    for sym in symbols:
        conn.execute(
            """
                    INSERT INTO sync_metadata (symbol, last_synced)
                    VALUES (?, ?)
                    ON CONFLICT(symbol) DO UPDATE SET last_synced = excluded.last_synced
                """,
            (sym, synced_at),
        )


# --- the EDGAR fact store ---------------------------------------------------------


def facts_for(
    conn: sqlite3.Connection, tags: Sequence[str], ciks: Sequence[str]
) -> List[tuple]:
    """(cik, tag, period_end, val) of the given tags for the given filers."""
    return conn.execute(
        f"SELECT cik, tag, period_end, val FROM facts "
        f"WHERE tag IN ({','.join('?' * len(tags))}) "
        f"AND cik IN ({','.join('?' * len(ciks))})",
        [*tags, *ciks],
    ).fetchall()


# --- the screener cache -----------------------------------------------------------


def screener_rows(conn: sqlite3.Connection, sql: str) -> List[Dict[str, Any]]:
    """The rows of a validated SELECT on the screener cache, as dicts."""
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute(sql)
    return [dict(row) for row in cursor.fetchall()]


def cached_ai_review(
    conn: sqlite3.Connection, symbol: str, universe: str
) -> Optional[tuple]:
    """(ai_summary, last_fiscal_year_end, most_recent_quarter), or None."""
    return conn.execute(
        "SELECT ai_summary, last_fiscal_year_end, most_recent_quarter FROM screener_cache WHERE symbol=? AND universe=?",
        (symbol, universe),
    ).fetchone()


def symbols_missing_ai_review(conn: sqlite3.Connection) -> List[tuple]:
    """(symbol, universe) of every row without a usable AI summary."""
    return conn.execute(
        """
            SELECT symbol, universe FROM screener_cache
            WHERE ai_summary IS NULL
               OR length(trim(ai_summary)) <= 20
               OR lower(ai_summary) LIKE '%error%'
            """
    ).fetchall()


def ai_review_frame(
    conn: sqlite3.Connection, dimensions: Sequence[str]
) -> pd.DataFrame:
    """symbol, the AI dimensions and updated_at, for every row scored on any."""
    columns = ", ".join(dimensions)
    query = (
        f"SELECT symbol, {columns}, updated_at FROM screener_cache "
        "WHERE " + " OR ".join(f"{c} > 0" for c in dimensions)
    )
    return pd.read_sql_query(query, conn)
