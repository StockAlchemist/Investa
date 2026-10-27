"""Phase 17b: sql_store in Rust (rust/investa_core/src/ss.rs) against the
original frozen in tests/reference/sql_store_p17.py.

Two identical databases (every table the store touches) take the same
random sequence of store calls, one through each module, under every
isolation level. Inputs include duplicate usernames (UNIQUE violations),
missing and repeated ids, synced transactions with every shape of value
(None, NaN, text amounts, dates and datetimes, a None type that cannot be
upper-cased, duplicate ExternalIDs), approvals with destination accounts,
empty rejections, a second BEGIN IMMEDIATE, receipts before and after their
table exists, snapshot rows of the wrong length, and screener SELECTs with
duplicate and differently-cased column names or bad syntax. After every call
the two must agree on the result or the exception (class, message, SQLite
code and name), on `in_transaction` and `total_changes`, on the connection's
`row_factory`, and on every table's contents.
"""

import datetime as dt
import os
import sqlite3
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import sql_store as new  # noqa: E402
from db_utils import initialize_database  # noqa: E402
from reference import sql_store_p17 as ref  # noqa: E402
from test_rust_market_db_parity import canon  # noqa: E402

EXTRA = [
    """CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL,
       hashed_password TEXT NOT NULL, alias TEXT, is_active BOOLEAN DEFAULT 1, created_at TEXT NOT NULL)""",
    "CREATE TABLE IF NOT EXISTS sync_metadata (symbol TEXT PRIMARY KEY, last_synced TEXT, inception_date TEXT, info_json TEXT)",
    """CREATE TABLE IF NOT EXISTS facts (cik TEXT NOT NULL, tag TEXT NOT NULL, period_end TEXT NOT NULL, period_start TEXT,
       val REAL, unit TEXT, form TEXT, accn TEXT NOT NULL, filed TEXT, PRIMARY KEY (cik, tag, period_end, accn))""",
]  # fmt: skip


def _outcome(fn):
    try:
        return ("ok", canon(fn()))
    except BaseException as exc:
        return ("raised", type(exc).__name__, str(exc), getattr(exc, "sqlite_errorcode", None),
                getattr(exc, "sqlite_errorname", None))  # fmt: skip


def _state(conn):
    tables = [r[0] for r in sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2]).execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]  # fmt: skip
    factory = conn.row_factory
    conn.row_factory = None
    try:
        return (conn.in_transaction, conn.total_changes, factory,
                [(t, canon(conn.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall())) for t in tables])  # fmt: skip
    finally:
        conn.row_factory = factory


def _pair(tmp_path, name, level):
    conns = []
    for side in ("py", "rs"):
        path = str(tmp_path / f"{name}_{side}.db")
        initialize_database(path).close()
        c = sqlite3.connect(path, isolation_level=level)
        for ddl in EXTRA:
            c.execute(ddl)
        if name.endswith("noreceipts"):
            c.execute("DROP TABLE transaction_receipts")
        c.commit()
        conns.append(c)
    return conns


def _value(rng):
    return [None, float("nan"), 0, 3, -2.5, "12.5", "x", np.float64(4.0), dt.date(2026, 1, 2),
            dt.datetime(2026, 1, 3, 9, 30), True][int(rng.integers(0, 11))]  # fmt: skip


def _tx(rng):
    keys = ["Date", "Type", "Symbol", "Quantity", "Price/Share", "Total Amount", "Commission", "Account", "Split Ratio",
            "Note", "Local Currency", "To Account", "Tags", "ExternalID", "Amount"]  # fmt: skip
    tx = {k: _value(rng) for k in keys if rng.random() < 0.5}
    if rng.random() < 0.9:
        tx["Date"] = [dt.date(2026, 2, int(rng.integers(1, 28))), "2026-02-03", dt.datetime(2026, 2, 4, 15)][int(rng.integers(0, 3))]
        tx["Type"] = str(rng.choice(["buy", " Buy ", "SELL", "dividend", "fees"]))
        tx["Symbol"] = str(rng.choice(["AAPL", "MSFT", "$CASH"]))
        tx["Account"] = str(rng.choice(["IBKR", "SET"]))
        tx["Local Currency"] = str(rng.choice(["USD", "THB"]))
        if rng.random() < 0.6:
            tx["Quantity"], tx["Price/Share"] = int(rng.integers(0, 5)), float(rng.choice([0.0, 10.0, 25.5]))
            tx["Commission"] = float(rng.choice([0.0, 1.0]))
    if rng.random() < 0.6:
        tx["ExternalID"] = str(rng.choice(["e-1", "e-2", "e-3", "", "e-4"]))
    if rng.random() < 0.05:
        tx["Type"] = None
    return tx


def _op(rng, step):
    r = int(rng.integers(0, 26))
    user = str(rng.choice(["alice", "bob", "carol", "ALICE"]))
    uid = int(rng.integers(0, 5))
    if r == 0:
        return "insert_user", (user, f"hash-{step}", "2026-01-01T00:00:00")
    if r == 1:
        return str(rng.choice(["find_user_id", "user_credentials", "user_record"])), (user,)
    if r == 2:
        return "set_user_alias", (uid, [None, "Al", 5][int(rng.integers(0, 3))])
    if r == 3:
        return str(rng.choice(["delete_user", "user_password_hash"])), (uid,)
    if r == 4:
        return "set_user_password", (uid, f"new-{step}")
    if r == 5:
        return "set_holding_tags", (str(rng.choice(["core", "", " x "])), str(rng.choice(["AAPL", "MSFT"])), str(rng.choice(["IBKR", "SET"])))
    if r in (6, 7, 8):
        return "stage_pending_transaction", (_tx(rng), int(rng.integers(1, 3)))
    if r == 9:
        return "pending_transaction_records", ()
    if r in (10, 11):
        ids = [int(i) for i in rng.integers(0, 12, int(rng.integers(0, 4)))]
        accounts = {i: str(rng.choice(["Taxable", "Retirement"])) for i in ids if rng.random() < 0.5}
        return "approve_pending_transactions", (ids, accounts)
    if r == 12:
        return "reject_pending_transactions", ([int(i) for i in rng.integers(0, 12, int(rng.integers(0, 3)))],)
    if r == 13:
        return "begin_immediate", ()
    if r == 14:
        return "find_receipt", (1, "transactions", str(rng.choice(["k1", "k2"])))
    if r == 15:
        return "insert_receipt", (1, "transactions", str(rng.choice(["k1", "k2"])), f"fp-{step}", '{"ok": true}')
    if r == 16:
        rows = [("2026-03-01", acc, 1.0, 2.0, 3.0, 4.0, 0.1, 0.2, "now") for acc in ("ALL", "IBKR")]
        if rng.random() < 0.2:
            rows.append(("2026-03-01", "short"))
        return "replace_portfolio_snapshots", ("2026-03-01", rows)
    if r == 17:
        return "symbol_currency_pairs", ()
    if r == 18:
        return "data_quality_table_present", ()
    if r == 19:
        return "touch_sync_metadata", ([str(s) for s in rng.choice(["AAPL", "MSFT", "SPY"], int(rng.integers(0, 3)))], f"2026-03-0{step % 9 + 1}")
    if r == 20:
        tags = list(rng.choice(["EPS", "Rev", "Eq"], int(rng.integers(1, 3))))
        return "facts_for", ([str(t) for t in tags], [str(c) for c in rng.choice(["1", "2", "3"], int(rng.integers(1, 3)))])
    if r == 21:
        sql = str(rng.choice([
            "SELECT symbol, ai_moat FROM screener_cache ORDER BY symbol",
            "SELECT symbol, SYMBOL, symbol AS Symbol, ai_moat AS m FROM screener_cache ORDER BY symbol",
            "SELECT * FROM screener_cache ORDER BY symbol",
            "SELEC symbol FROM screener_cache",
            "SELECT symbol FROM missing_table",
        ]))  # fmt: skip
        return "screener_rows", (sql,)
    if r == 22:
        return "cached_ai_review", (str(rng.choice(["AAPL", "MSFT", "NONE"])), str(rng.choice(["sp500", "manual"])))
    if r == 23:
        return "symbols_missing_ai_review", ()
    if r == 24:
        return "ai_review_frame", ([str(d) for d in rng.choice(["ai_moat", "ai_growth", "ai_predictability"], int(rng.integers(1, 3)), replace=False)],)
    return "commit_or_rollback", ()


def _seed_rows(conn):
    conn.executemany('INSERT INTO transactions (Date, Type, Symbol, "Local Currency", Account, ExternalID) VALUES (?,?,?,?,?,?)',
                     [("2026-01-01", "Buy", "AAPL", "USD", "IBKR", "e-1"), ("2026-01-02", "Buy", "MSFT", "USD", "SET", None)])
    conn.executemany("INSERT INTO facts (cik, tag, period_end, val, accn) VALUES (?,?,?,?,?)",
                     [(c, t, "2024-12-31", 1.5, "a") for c in "123" for t in ("EPS", "Rev", "Eq")])
    conn.executemany("INSERT INTO screener_cache (symbol, universe, ai_summary, ai_moat, ai_growth, updated_at) VALUES (?,?,?,?,?,?)",
                     [("AAPL", "sp500", "a long enough summary of the business", 8, 7, "2026-01-01"),
                      ("MSFT", "manual", "error", 0, 0, "2025-12-31")])  # fmt: skip
    conn.commit()


@pytest.mark.parametrize("level", ["", "DEFERRED", "IMMEDIATE", None])
@pytest.mark.parametrize("seed", range(4))
def test_store_sequences_match(tmp_path, level, seed):
    py, rs = _pair(tmp_path, f"s{seed}" + ("_noreceipts" if seed == 3 else ""), level)
    for c in (py, rs):
        _seed_rows(c)
    rng = np.random.default_rng(25000 + seed)
    for step in range(160):
        name, args = _op(rng, step)
        if name == "commit_or_rollback":
            action = "commit" if rng.random() < 0.7 else "rollback"
            a = _outcome(lambda: getattr(py, action)())
            b = _outcome(lambda: getattr(rs, action)())
        else:
            a = _outcome(lambda: getattr(ref, name)(py, *args))
            b = _outcome(lambda: getattr(new, name)(rs, *args))
        assert a == b, (seed, step, name, args)
        assert _state(py) == _state(rs), (seed, step, name)
        if name == "screener_rows":
            for c in (py, rs):
                c.row_factory = None
