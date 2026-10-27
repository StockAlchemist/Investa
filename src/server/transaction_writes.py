"""Atomic writes and durable receipts for retryable transaction submissions."""

import hashlib
import json
import sqlite3
from typing import Callable

from fastapi import HTTPException

import sql_store

RECEIPTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS transaction_receipts (
    user_id INTEGER NOT NULL,
    endpoint TEXT NOT NULL,
    request_key TEXT NOT NULL,
    fingerprint TEXT NOT NULL,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (user_id, endpoint, request_key)
)
"""


def atomic_submission(
    conn: sqlite3.Connection,
    user_id: int,
    endpoint: str,
    request_key: str | None,
    payload: dict,
    write: Callable,
):
    """Serialize writers, then commit receipt and all transaction legs together."""
    if request_key is not None and (not request_key.strip() or len(request_key) > 128):
        raise HTTPException(400, "Invalid Idempotency-Key")
    fingerprint = hashlib.sha256(
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    try:
        sql_store.begin_immediate(conn)
        if request_key:
            try:
                previous = sql_store.find_receipt(conn, user_id, endpoint, request_key)
            except sqlite3.OperationalError as exc:
                if "no such table" in str(exc):
                    raise HTTPException(
                        503,
                        "Transaction receipt migration required; run scripts/migrate_transaction_receipts.py with the backend stopped",
                    ) from exc
                raise
            if previous:
                if previous[0] != fingerprint:
                    raise HTTPException(
                        409,
                        "Idempotency-Key was already used for a different submission",
                    )
                conn.rollback()
                return json.loads(previous[1]), False
        response = write()
        if request_key:
            sql_store.insert_receipt(
                conn, user_id, endpoint, request_key, fingerprint, json.dumps(response)
            )
        conn.commit()
        return response, True
    except Exception:
        conn.rollback()
        raise
