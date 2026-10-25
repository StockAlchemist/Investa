"""
SQL through investa_core on Python's own sqlite3 connections.

The storage modules keep their `sqlite3.Connection` objects: connection
caching, PRAGMAs and the callers' `with conn:` blocks all stay as they were.
What runs in Rust is the statement, on the very `sqlite3*` the Python
connection holds, so it joins the caller's open transaction, sees its
uncommitted writes and is committed or rolled back with it. It runs through
the SQLite library Python's module itself calls (see
rust/investa_core/src/sqlite_read.rs), never a second copy.

`execute`, `executemany` and `read_frame` stand in for `conn.execute`,
`conn.executemany` and `pd.read_sql_query(sql, conn, params=...)`, with what
those do around a statement: the thread and closed-connection checks, the
implicit BEGIN before DML, the binding rules, `rowcount` and `lastrowid`, and
the exception class, message and `sqlite_errorcode` / `sqlite_errorname` for
every SQLite error. Whatever the Rust side does not bind exactly as Python
would (an adapter, a NumPy integer, named parameters, a `row_factory`,
explicit autocommit control, ...) is handed to Python's own call, before
anything has run.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sqlite3
import tempfile
from typing import Any, Iterable, Optional

import investa_core

# CPython 3.12 `pysqlite_Connection`: PyObject_HEAD, then `sqlite3 *db`.
_HEAD = object.__basicsize__
_BOUND: Optional[bool] = None


def _pointer(conn: sqlite3.Connection) -> Optional[int]:
    return ctypes.c_void_p.from_address(id(conn) + _HEAD).value


def _layout_holds() -> bool:
    """The handle read at `_HEAD` names the file the connection opened."""
    with tempfile.TemporaryDirectory() as directory:
        conn = sqlite3.connect(os.path.join(directory, "probe.db"))
        try:
            ptr = _pointer(conn)
            main = conn.execute("PRAGMA database_list").fetchone()[2]
            return bool(ptr) and investa_core.sql_db_filename(ptr) == main
        finally:
            conn.close()


def bound() -> bool:
    """investa_core reaches Python's SQLite, and connections lay out as expected."""
    global _BOUND
    if _BOUND is None:
        _BOUND = False
        try:
            import _sqlite3

            _BOUND = (
                bool(investa_core.sqlite_bind(_sqlite3.__file__)) and _layout_holds()
            )
        except Exception as exc:
            logging.info(f"rust_sql: Rust SQL unavailable ({exc})")
        if not _BOUND:
            logging.info("rust_sql: statements run through Python's sqlite3")
    return _BOUND


_BASE_TYPES = (int, float, str, bytes, bytearray)


def _adapters_reach_base_types() -> bool:
    # Python adapts every value once an adapter is registered for a base type
    # (or a subclass Rust would bind as its base).
    return any(
        isinstance(key[0], type) and issubclass(key[0], _BASE_TYPES)
        for key in sqlite3.adapters
    )


def handle(conn: Any) -> Optional[int]:
    """
    `conn`'s sqlite3* after the checks Python's `execute` makes first (wrong
    thread, closed connection), which raise the same ProgrammingError. None
    when the statement must stay on Python's own path.
    """
    if type(conn) is not sqlite3.Connection or not bound():
        return None
    conn.cursor()  # Python's own thread and closed checks
    if (
        conn.row_factory is not None
        or conn.text_factory is not str
        or conn.autocommit != sqlite3.LEGACY_TRANSACTION_CONTROL
        or _adapters_reach_base_types()
    ):
        return None
    return _pointer(conn) or None


# --- statements ----------------------------------------------------------------


def execute(conn: sqlite3.Connection, sql: str, params: Any = ()):
    """`conn.execute(sql, params)`; the rows are fetched up front. Runs in
    Rust (`pyconn.rs`) on the connection's handle, or through the
    connection's own method when Rust must not."""
    return investa_core.conn_execute(conn, sql, params)


def executemany(conn: sqlite3.Connection, sql: str, seq_of_params: Iterable[Any]):
    """`conn.executemany(sql, seq_of_params)`."""
    return investa_core.conn_executemany(conn, sql, seq_of_params)


def read_frame(conn: sqlite3.Connection, sql: str, params: Any = None):
    """
    `pd.read_sql_query(sql, conn, params=params)`: the statement runs here,
    the frame is built by pandas' own `_wrap_result` from the rows, and a
    failure rolls the connection back and raises pandas' DatabaseError, as
    pandas does.
    """
    import pandas as pd

    try:
        from pandas.io.sql import _wrap_result
    except ImportError:  # pragma: no cover - a pandas without the helper
        return pd.read_sql_query(sql, conn, params=params)
    if handle(conn) is None or type(sql) is not str:
        return pd.read_sql_query(sql, conn, params=params)
    try:
        cursor = execute(conn, sql, () if params is None else params)
    except Exception as exc:
        try:
            conn.rollback()
        except Exception as inner_exc:  # pragma: no cover - mirrors pandas
            ex = pd.errors.DatabaseError(
                f"Execution failed on sql: {sql}\n{exc}\nunable to rollback"
            )
            raise ex from inner_exc
        ex = pd.errors.DatabaseError(f"Execution failed on sql '{sql}': {exc}")
        raise ex from exc
    # pandas iterates the description as is: None (no result columns) raises.
    columns = [col_desc[0] for col_desc in cursor.description]
    data = cursor.fetchall()
    if not isinstance(data, list):
        data = list(data)
    return _wrap_result(
        data,
        columns,
        index_col=None,
        coerce_float=True,
        parse_dates=None,
        dtype=None,
        dtype_backend="numpy",
    )
