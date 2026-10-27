"""Phase 17a: the crate's SQLite is rusqlite linked against the system's
shared library, and must be the very image Python's `_sqlite3` calls.

- One image: `sqlite3_libversion` resolved through Python's extension and
  through the crate's extension is the same address, and the start-up check
  agrees.
- CPython's one-statement rule, not rusqlite's: every tail that only
  whitespace and comments may follow, every tail that must be refused
  (including text rusqlite would accept or fail to parse), `;` inside strings
  and comments, empty and comment-only statements.
- Failures raised while stepping (after rusqlite has reset the statement)
  keep CPython's class, message and extended code, in `execute` and in the
  middle of an `executemany`.
"""

import ctypes
import os
import sqlite3
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402
import rust_sql  # noqa: E402
from test_rust_sql_parity import _outcome, _pair, _state  # noqa: E402


def test_one_sqlite_image():
    import _sqlite3

    theirs = ctypes.cast(ctypes.CDLL(_sqlite3.__file__).sqlite3_libversion, ctypes.c_void_p).value
    ours = ctypes.cast(ctypes.CDLL(investa_core.investa_core.__file__).sqlite3_libversion, ctypes.c_void_p).value
    assert theirs == ours
    assert rust_sql.bound()


TAILS = [
    "SELECT 1", "SELECT 1;", "SELECT 1;;", "SELECT 1; ", "SELECT 1; -- c", "SELECT 1; /* c */",
    "SELECT 1; /* open", "SELECT 1; garbage", "SELECT 1; SELECT 2", "SELECT 1;\nSELECT 2;",
    "SELECT ';' AS x", "SELECT 'a;b'; ", "SELECT 1 /* ; */", "SELECT 1 -- ;", ";", " ; ", ";;",
    "-- only", "/* open", "", "   ", "SELEC 1; SELECT 2", "SELECT 1; SELEC 2",
    "INSERT INTO t (a) VALUES (1); INSERT INTO t (a) VALUES (2)", "INSERT INTO t (a) VALUES (';');",
    "CREATE TABLE z (a); ", "/* lead */ SELECT 2;  -- trail",
]  # fmt: skip


@pytest.mark.parametrize("level", ["", "DEFERRED", None])
def test_statement_tails_match_cpython(tmp_path, level):
    py, rs = _pair(tmp_path, "tails", level)
    for sql in TAILS:
        a = _outcome(lambda: py.execute(sql))
        b = _outcome(lambda: rust_sql.execute(rs, sql))
        assert a == b, sql
        assert _state(py) == _state(rs), sql


def test_failures_while_stepping(tmp_path):
    py, rs = _pair(tmp_path, "step", "")
    for conn in (py, rs):
        conn.execute("INSERT INTO t (id, u) VALUES (1, 10)")
    cases = [
        ("SELECT abs(-9223372036854775808)", ()),  # integer overflow, raised by step
        ("INSERT INTO t (id, u) VALUES (2, 10)", ()),  # UNIQUE constraint (extended code 2067)
        ("INSERT INTO t (id, u) VALUES (1, 11)", ()),  # PRIMARY KEY constraint (1555)
        ("SELECT * FROM missing", ()),
        ("UPDATE t SET u = ? WHERE id = 1", ("x",)),
    ]
    for sql, params in cases:
        a = _outcome(lambda: py.execute(sql, params))
        b = _outcome(lambda: rust_sql.execute(rs, sql, params))
        assert a == b, sql
        assert _state(py) == _state(rs), sql
    rows = [(3, 30), (4, 10), (5, 50)]  # the middle row violates UNIQUE
    a = _outcome(lambda: py.executemany("INSERT INTO t (id, u) VALUES (?, ?)", rows))
    b = _outcome(lambda: rust_sql.executemany(rs, "INSERT INTO t (id, u) VALUES (?, ?)", rows))
    assert a == b
    assert _state(py) == _state(rs)
    for conn in (py, rs):
        conn.rollback()
    assert _state(py) == _state(rs)


def test_extended_code_is_cpythons(tmp_path):
    py, rs = _pair(tmp_path, "codes", "")
    for conn in (py, rs):
        conn.execute("INSERT INTO t (id, u) VALUES (1, 10)")
    outs = []
    for fn in (lambda: py.execute("INSERT INTO t (u) VALUES (10)"), lambda: rust_sql.execute(rs, "INSERT INTO t (u) VALUES (10)")):
        try:
            fn()
        except sqlite3.IntegrityError as exc:
            outs.append((exc.sqlite_errorcode, exc.sqlite_errorname, str(exc)))
    assert outs[0] == outs[1] == (2067, "SQLITE_CONSTRAINT_UNIQUE", "UNIQUE constraint failed: t.u")
