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
from typing import Any, Iterable, List, Optional, Sequence

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


def _plain(params: Any) -> bool:
    """A positional parameter list Rust binds as Python would."""
    if type(params) not in (tuple, list):
        return False
    for v in params:
        t = type(v)
        if t is not float and isinstance(v, float) and hasattr(v, "__conform__"):
            return False
    return True


def _begin(conn: sqlite3.Connection) -> Optional[str]:
    level = conn.isolation_level
    return None if level is None else "BEGIN " + level


# --- errors ------------------------------------------------------------------

_PRIMARY_NAMES = (
    "SQLITE_OK SQLITE_ERROR SQLITE_INTERNAL SQLITE_PERM SQLITE_ABORT SQLITE_BUSY "
    "SQLITE_LOCKED SQLITE_NOMEM SQLITE_READONLY SQLITE_INTERRUPT SQLITE_IOERR "
    "SQLITE_CORRUPT SQLITE_NOTFOUND SQLITE_FULL SQLITE_CANTOPEN SQLITE_PROTOCOL "
    "SQLITE_EMPTY SQLITE_SCHEMA SQLITE_TOOBIG SQLITE_CONSTRAINT SQLITE_MISMATCH "
    "SQLITE_MISUSE SQLITE_NOLFS SQLITE_AUTH SQLITE_FORMAT SQLITE_RANGE SQLITE_NOTADB "
    "SQLITE_NOTICE SQLITE_WARNING SQLITE_ROW SQLITE_DONE"
).split()
_ERROR_NAMES = {getattr(sqlite3, n): n for n in _PRIMARY_NAMES if hasattr(sqlite3, n)}
_ERROR_NAMES.update(
    {
        getattr(sqlite3, n): n
        for n in dir(sqlite3)
        if n.startswith("SQLITE_")
        and any(n.startswith(p + "_") for p in _PRIMARY_NAMES)
        and isinstance(getattr(sqlite3, n), int)
        and getattr(sqlite3, n) > 255
    }
)


def _error_class(primary: int):
    """CPython 3.12 `get_exception_class`."""
    if primary in (2, 12):  # INTERNAL, NOTFOUND
        return sqlite3.InternalError
    if primary == 7:  # NOMEM
        return MemoryError
    if primary in (1, 3, 4, 5, 6, 8, 9, 10, 13, 14, 15, 16, 17):
        return sqlite3.OperationalError
    if primary == 11:  # CORRUPT
        return sqlite3.DatabaseError
    if primary == 18:  # TOOBIG
        return sqlite3.DataError
    if primary in (19, 20):  # CONSTRAINT, MISMATCH
        return sqlite3.IntegrityError
    if primary in (21, 25):  # MISUSE, RANGE
        return sqlite3.InterfaceError
    return sqlite3.DatabaseError


def _raise(outcome) -> None:
    kind = outcome[0]
    if kind == "sqlite":
        code, message = outcome[1], outcome[2]
        cls = _error_class(code & 0xFF)
        if cls is MemoryError:
            raise MemoryError()
        exc = cls(message)
        exc.sqlite_errorcode = code
        exc.sqlite_errorname = _ERROR_NAMES.get(code, "unknown")
        raise exc
    if kind == "programming":
        raise sqlite3.ProgrammingError(outcome[1])
    if kind == "undecodable":
        raise sqlite3.OperationalError(
            f"Could not decode to UTF-8 column '{outcome[1]}' with text '{outcome[2]}'"
        )
    raise AssertionError(f"unexpected outcome {outcome!r}")


# --- results -----------------------------------------------------------------


class Rows:
    """What a caller reads from the cursor `execute` returns: the rows (all
    fetched), `description`, `rowcount` and `lastrowid`."""

    __slots__ = ("description", "rowcount", "lastrowid", "arraysize", "_rows", "_at")

    def __init__(self, columns, rows, rowcount, lastrowid):
        self.description = (
            tuple((name, None, None, None, None, None, None) for name in columns)
            if columns
            else None
        )
        self.rowcount = rowcount
        self.lastrowid = lastrowid
        self.arraysize = 1
        self._rows = rows
        self._at = 0

    def fetchone(self):
        if self._at >= len(self._rows):
            return None
        self._at += 1
        return self._rows[self._at - 1]

    def fetchmany(self, size: Optional[int] = None):
        n = self.arraysize if size is None else size
        out = self._rows[self._at : self._at + n]
        self._at += len(out)
        return out

    def fetchall(self):
        out = self._rows[self._at :]
        self._at = len(self._rows)
        return out

    def __iter__(self):
        return self

    def __next__(self):
        row = self.fetchone()
        if row is None:
            raise StopIteration
        return row

    def close(self) -> None:
        pass


# --- statements ----------------------------------------------------------------


def execute(conn: sqlite3.Connection, sql: str, params: Sequence[Any] = ()):
    """`conn.execute(sql, params)`; the rows are fetched up front."""
    h = handle(conn)
    if h is None or type(sql) is not str or not _plain(params):
        return conn.execute(sql, params)
    out = investa_core.sql_execute(h, sql, list(params), _begin(conn))
    if out[0] == "ok":
        return Rows(out[1], out[2], out[3], out[4])
    if out[0] == "decline":
        return conn.execute(sql, params)
    _raise(out)


def executemany(
    conn: sqlite3.Connection, sql: str, seq_of_params: Iterable[Sequence[Any]]
):
    """`conn.executemany(sql, seq_of_params)`."""
    items: List[Any] = list(seq_of_params)
    h = handle(conn)
    if h is None or type(sql) is not str or not all(_plain(p) for p in items):
        return conn.executemany(sql, items)
    out = investa_core.sql_executemany(h, sql, [list(p) for p in items], _begin(conn))
    if out[0] == "ok":
        # A fresh cursor's lastrowid, which executemany leaves unset.
        return Rows(out[1], out[2], out[3], None)
    if out[0] == "decline":
        return conn.executemany(sql, items)
    _raise(out)


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
    if (
        handle(conn) is None
        or type(sql) is not str
        or not (params is None or _plain(params))
    ):
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
