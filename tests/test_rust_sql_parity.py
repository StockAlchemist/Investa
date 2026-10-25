"""Phase 10: SQL in Rust on Python's own connections (src/rust_sql.py,
rust/investa_core/src/sql.rs) against Python's sqlite3.

Two identical databases take the same random sequence of operations, one
through `conn.execute` / `executemany` / `pd.read_sql_query`, the other
through `rust_sql`. After every operation the two must agree on:
- the outcome: the rows, `description`, `rowcount` and `lastrowid`, or the
  exception's class, message, `sqlite_errorcode` and `sqlite_errorname`;
- `in_transaction` and `total_changes`;
- every table's contents.
The sequences mix:
- DML with leading comments and any case, so the implicit BEGIN is exercised,
  plus commits and rollbacks;
- every isolation level, autocommit included;
- RETURNING, PRAGMAs, schema changes, a second statement, syntax errors,
  binding-count errors and constraint violations;
- parameters of every type, including the ones Python must bind itself.
"""

import datetime
import math
import os
import sqlite3
import struct
import sys
import threading
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import rust_sql  # noqa: E402

SCHEMA = [
    "CREATE TABLE t (id INTEGER PRIMARY KEY, a, b TEXT, c REAL, u INTEGER UNIQUE)",
    "CREATE TABLE s (k TEXT PRIMARY KEY, v BLOB)",
]


def _canon(x):
    if isinstance(x, float):
        return ("f", "nan" if math.isnan(x) else struct.pack("<d", x))
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_canon(v) for v in x])
    return (type(x).__name__, x)


def _value(rng):
    r = int(rng.integers(0, 17))
    return [
        None, True, False, 7, -(2**63), 2**63, 1.5, float("nan"), np.float64(2.5),
        np.int64(3), "text", "日本語", "\ud800", b"\x00\x01", bytearray(b"x"),
        datetime.date(2024, 1, 2), int(rng.integers(0, 50)),
    ][r]  # fmt: skip


def _statement(rng):
    lead = str(rng.choice(["", "  ", "\n-- note\n", "/* c */ "]))
    case = str if rng.random() < 0.7 else str.lower
    pick = int(rng.integers(0, 22))
    v = lambda: _value(rng)  # noqa: E731
    if pick < 5:
        return lead + case("INSERT INTO t (a, b, c, u) VALUES (?, ?, ?, ?)"), [
            v(),
            v(),
            v(),
            v(),
        ]
    if pick < 7:
        return lead + case("INSERT OR REPLACE INTO s (k, v) VALUES (?, ?)"), [v(), v()]
    if pick < 9:
        return lead + case("UPDATE t SET a = ? WHERE u = ?"), [v(), v()]
    if pick == 9:
        return lead + case("DELETE FROM t WHERE id % 3 = ?"), [int(rng.integers(0, 3))]
    if pick == 10:
        return lead + case("REPLACE INTO t (id, a) VALUES (?, ?)"), [
            int(rng.integers(1, 9)),
            v(),
        ]
    if pick < 14:
        return lead + "SELECT id, a, b, c, u FROM t WHERE id > ? ORDER BY id", [
            int(rng.integers(0, 5))
        ]
    if pick == 14:
        return "SELECT k, v, typeof(v) FROM s ORDER BY k", []
    if pick == 15:
        return "INSERT INTO t (a) VALUES (?) RETURNING id, a", [v()]
    if pick == 16:
        return "PRAGMA table_info(t)", []
    if pick == 17:
        return "SELECT 1; SELECT 2", []
    if pick == 18:
        return "SELEC 1", []
    if pick == 19:
        return "INSERT INTO t (a, b) VALUES (?, ?)", [v()]  # wrong count
    if pick == 20:
        return "CREATE TABLE IF NOT EXISTS x (q)", []
    if pick == 21:
        return _named(rng, lead, case)
    return "SELECT count(*), sum(u), max(c) FROM t", []


def _named(rng, lead, case):
    """Named placeholders bound from a dict: missing keys, extra keys, a bare
    `?` given a dict, numbered `?1`, a dict subclass Python must bind."""
    v = lambda: _value(rng)  # noqa: E731
    pick = int(rng.integers(0, 6))
    if pick == 0:
        sql = lead + case("INSERT INTO t (a, b, u) VALUES (:a, :b, :u)")
        params = {"a": v(), "b": v(), "u": v(), "extra": 1}
    elif pick == 1:
        sql = "UPDATE t SET b = $b WHERE id > @lo"
        params = {"b": v(), "lo": int(rng.integers(0, 5))}
    elif pick == 2:
        sql = "SELECT id, a FROM t WHERE id > :lo ORDER BY id"
        params = {"lo": int(rng.integers(0, 5))}
    elif pick == 3:
        sql = "INSERT INTO t (a, b) VALUES (:a, :b)"
        params = {"a": v()}  # :b missing
    elif pick == 4:
        sql = "SELECT ?1, ?2"
        params = {"1": v(), "2": v()}
    else:
        sql = "INSERT INTO t (a) VALUES (?)"
        params = {"a": v()}  # a bare ? with a dict
    if rng.random() < 0.1:
        from collections import OrderedDict

        params = OrderedDict(params)
    if rng.random() < 0.1:
        params = list(params.values())  # a named statement bound by position
    return sql, params


def _outcome(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)  # the date adapter
        try:
            cur = fn()
        except BaseException as exc:
            return (
                "raised",
                type(exc).__name__,
                str(exc),
                getattr(exc, "sqlite_errorcode", None),
                getattr(exc, "sqlite_errorname", None),
            )
    rows = cur.fetchall()
    return ("ok", _canon(rows), cur.description, cur.rowcount, cur.lastrowid)


def _state(conn):
    tables = [
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
    ]
    return (
        conn.in_transaction,
        conn.total_changes,
        [
            (t, _canon(conn.execute(f"SELECT * FROM {t} ORDER BY rowid").fetchall()))
            for t in tables
        ],
    )


def _pair(tmp_path, name, level):
    conns = []
    for side in ("py", "rs"):
        c = sqlite3.connect(str(tmp_path / f"{name}_{side}.db"), isolation_level=level)
        for stmt in SCHEMA:
            c.execute(stmt)
        c.commit()
        conns.append(c)
    return conns


@pytest.mark.parametrize("seed", range(8))
def test_random_sequences_match(tmp_path, seed):
    assert rust_sql.bound(), "Rust SQL did not bind to Python's SQLite"
    rng = np.random.default_rng(8000 + seed)
    for trial, level in enumerate(["", "DEFERRED", "IMMEDIATE", None]):
        py, rs = _pair(tmp_path, f"{seed}_{trial}", level)
        for step in range(120):
            path = f"{seed}/{trial} step {step}"
            r = rng.random()
            if r < 0.08:
                py.commit(), rs.commit()
                continue
            if r < 0.14:
                py.rollback(), rs.rollback()
                continue
            if r < 0.22:
                sql = str(
                    rng.choice(
                        [
                            "INSERT INTO s (k, v) VALUES (?, ?)",
                            "/* x */ insert into t (a, u) values (?, ?)",
                            "SELECT ?, ?",
                            "INSERT INTO t (a, u) VALUES (:a, :u)",
                        ]
                    )
                )
                seq = []
                for _ in range(int(rng.integers(0, 5))):
                    if ":a" in sql:
                        item = {"a": _value(rng), "u": int(rng.integers(0, 40))}
                        if rng.random() < 0.15:
                            del item["u"]  # fails after the earlier items ran
                        seq.append(item)
                    elif rng.random() < 0.1:
                        seq.append([_value(rng)])  # wrong count, mid-run
                    else:
                        seq.append([_value(rng), _value(rng)])
                a = _outcome(lambda: py.executemany(sql, seq))
                b = _outcome(lambda: rust_sql.executemany(rs, sql, seq))
                assert a == b, (path, sql, seq, a, b)
            elif r < 0.30:
                sql, params = _statement(rng)
                a = _outcome(
                    lambda: _frame_cursor(pd.read_sql_query(sql, py, params=params))
                )
                b = _outcome(
                    lambda: _frame_cursor(rust_sql.read_frame(rs, sql, params))
                )
                assert a == b, (path, sql, params, a, b)
            else:
                sql, params = _statement(rng)
                a = _outcome(lambda: py.execute(sql, params))
                b = _outcome(lambda: rust_sql.execute(rs, sql, params))
                assert a == b, (path, sql, params, a, b)
            assert _state(py) == _state(rs), path
        py.close(), rs.close()


class _frame_cursor:
    """A frame seen through the cursor fields _outcome compares."""

    def __init__(self, frame):
        self.frame = frame
        self.description = (
            tuple(map(str, frame.columns)),
            tuple(map(str, frame.dtypes)),
            str(frame.index),
        )
        self.rowcount = len(frame)
        self.lastrowid = None

    def fetchall(self):
        return [tuple(r) for r in self.frame.itertuples(index=False)]


def test_a_rust_write_rolls_back_with_the_callers_block(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "block.db"))
    conn.execute(SCHEMA[0])
    conn.commit()
    with pytest.raises(RuntimeError):
        with conn:
            rust_sql.execute(conn, "INSERT INTO t (a) VALUES (?)", [1])
            assert conn.in_transaction
            raise RuntimeError("abandon")
    assert conn.execute("SELECT count(*) FROM t").fetchone() == (0,)
    with conn:
        rust_sql.execute(conn, "INSERT INTO t (a) VALUES (?)", [2])
    other = sqlite3.connect(str(tmp_path / "block.db"))
    assert other.execute("SELECT a FROM t").fetchall() == [(2,)]


def test_thread_and_closed_checks_raise_as_python(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "thread.db"))
    seen = []

    def other():
        for fn in (
            lambda: conn.execute("SELECT 1"),
            lambda: rust_sql.execute(conn, "SELECT 1"),
        ):
            try:
                fn()
            except Exception as exc:
                seen.append((type(exc).__name__, str(exc).split(" The object")[0]))

    t = threading.Thread(target=other)
    t.start()
    t.join()
    assert len(seen) == 2 and seen[0] == seen[1], seen
    conn.close()
    errs = []
    for fn in (
        lambda: conn.execute("SELECT 1"),
        lambda: rust_sql.execute(conn, "SELECT 1"),
    ):
        try:
            fn()
        except Exception as exc:
            errs.append((type(exc).__name__, str(exc)))
    assert errs[0] == errs[1]


def test_connections_rust_must_not_touch_take_pythons_path(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "rf.db"))
    conn.row_factory = sqlite3.Row
    assert rust_sql.handle(conn) is None
    row = rust_sql.execute(conn, "SELECT 1 AS one").fetchone()
    assert isinstance(row, sqlite3.Row) and row["one"] == 1
    auto = sqlite3.connect(str(tmp_path / "ac.db"), autocommit=True)
    assert rust_sql.handle(auto) is None
