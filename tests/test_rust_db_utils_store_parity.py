"""Phase 11a (cont.): the rest of db_utils in Rust — schema migrations,
connections and the screener cache (db_portfolio.rs, db_screener.rs) —
against the original frozen in tests/reference/db_utils.py.

Each check runs the original and the port side by side on twin files and
compares the return value, the resulting schema (every table's SQL and
columns) and contents, and every log record and warning. Timestamps the code
stamps with `datetime.now()` are masked.

- Migrations start from random older states: any recorded schema version, the
  columns added since dropped, later tables missing, a legacy `watchlist`
  table with rows to move, a schema_version table missing entirely.
- Connections: a plain path, a cloud-drive path (journal mode kept or set to
  DELETE), the per-thread cache, a cached connection that was closed,
  `use_cache=False`, `check_same_thread=False`, an unwritable directory, and the
  read-only fallback when a WAL file has no -shm.
- The screener cache: random upserts and refreshes (lists and dicts that need
  JSON, NumPy values, missing symbols), AI-review and intrinsic-value updates
  (synthetic symbols skipped, new rows inserted when none match, `None`
  scorecards), and the three readers.
"""

import logging
import os
import re
import sqlite3
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import db_utils as new  # noqa: E402
from reference import db_utils as ref  # noqa: E402
from test_rust_db_utils_parity import _mask  # noqa: E402

STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?")


def _schema(path):
    c = sqlite3.connect(path)
    try:
        objects = c.execute(
            "SELECT type, name, sql FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' ORDER BY type, name"
        ).fetchall()
        tables = [r[1] for r in objects if r[0] == "table"]
        data = [
            (t, _mask(c.execute(f'SELECT * FROM "{t}" ORDER BY rowid').fetchall()))
            for t in tables
        ]
        cols = [(t, c.execute(f'PRAGMA table_info("{t}")').fetchall()) for t in tables]
        return objects, cols, data
    finally:
        c.close()


def _run(fn, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        try:
            out = ("ok", fn())
        except Exception as exc:
            out = ("raised", type(exc).__name__, STAMP.sub("<now>", str(exc)))
    logs = [
        (
            r.levelname,
            STAMP.sub("<now>", r.getMessage()),
            r.exc_info[0].__name__ if r.exc_info else None,
        )
        for r in caplog.records
        if r.name == "root"
    ]
    return out, logs, [(w.category.__name__, str(w.message)) for w in warned]


def _old_state(path, rng):
    """A portfolio DB as some earlier release left it."""
    c = sqlite3.connect(path)
    ref.create_transactions_table(c)
    c.execute(
        "INSERT INTO transactions (Date, Type, Symbol, Account, \"Local Currency\", Note) VALUES ('2024-01-01', 'Transfer', 'AAPL', 'A', 'USD', 'To: B')"
    )
    version = int(rng.integers(0, 16))
    c.execute("DELETE FROM schema_version WHERE version > ?", (version,))
    if version < 11 and rng.random() < 0.7:
        c.execute("DROP INDEX IF EXISTS idx_external_id")
        c.execute("ALTER TABLE transactions DROP COLUMN ExternalID")
    if version < 10 and rng.random() < 0.7:
        c.execute("ALTER TABLE transactions DROP COLUMN user_id")
    if version < 4 and rng.random() < 0.7:
        c.execute("ALTER TABLE transactions DROP COLUMN Tags")
    if version < 2 and rng.random() < 0.7:
        c.execute('ALTER TABLE transactions DROP COLUMN "To Account"')
    if version < 15 and rng.random() < 0.7:
        for col in (
            "ai_moat_analysis",
            "ai_financial_strength_analysis",
            "ai_predictability_analysis",
            "ai_growth_perspective_analysis",
        ):
            c.execute(f"ALTER TABLE screener_cache DROP COLUMN {col}")
    if version < 14 and rng.random() < 0.5:
        c.execute("ALTER TABLE screener_cache DROP COLUMN ai_catalysts")
        c.execute("ALTER TABLE screener_cache DROP COLUMN ai_sentiment")
    if version < 13 and rng.random() < 0.6:
        c.execute("DROP TABLE portfolio_snapshots")
    if version < 12 and rng.random() < 0.6:
        c.execute("DROP TABLE pending_transactions")
    if version < 8 and rng.random() < 0.5:
        c.execute("DROP TABLE screener_cache")
    if version < 5 and rng.random() < 0.6:
        c.execute("DROP TABLE watchlist_items")
        c.execute("DROP TABLE watchlists")
        c.execute(
            "CREATE TABLE IF NOT EXISTS watchlist (Symbol TEXT PRIMARY KEY, Note TEXT, AddedOn TEXT NOT NULL)"
        )
        c.executemany(
            "INSERT INTO watchlist VALUES (?, ?, ?)",
            [("AAPL", "n", "2024-01-01"), ("MSFT", None, "2024-01-02")],
        )
    if rng.random() < 0.05:
        c.execute("DROP TABLE schema_version")
    c.commit()
    c.close()


@pytest.mark.parametrize("seed", range(40))
def test_migrations_match(tmp_path, seed, caplog):
    rng = np.random.default_rng(8200 + seed)
    paths = [str(tmp_path / f"{side}.db") for side in ("ref", "new")]
    for p in paths:
        _old_state(p, np.random.default_rng(8200 + seed))
    outs = []
    for mod, p in zip((ref, new), paths):
        conn = sqlite3.connect(p)
        outs.append(_run(lambda: mod.create_transactions_table(conn), caplog))
        conn.close()
    assert outs[0] == outs[1], seed
    assert _schema(paths[0]) == _schema(paths[1]), seed
    del rng


def test_connections_match(tmp_path, caplog, monkeypatch):
    for name in ("plain", "Dropbox/sync"):
        for mode in ("wal", "delete"):
            pair = []
            for side, mod in (("ref", ref), ("new", new)):
                d = tmp_path / side / name
                d.mkdir(parents=True, exist_ok=True)
                p = str(d / f"{mode}.db")
                c = sqlite3.connect(p)
                c.execute(f"PRAGMA journal_mode={mode.upper()}")
                c.execute("CREATE TABLE t (a)")
                c.commit()
                c.close()
                mod._DB_CONN_CACHE.connections = {}
                first = _run(lambda: mod.get_db_connection(p), caplog)
                conn = mod._DB_CONN_CACHE.connections[(p, True)]
                again = mod.get_db_connection(p)
                fresh = mod.get_db_connection(p, use_cache=False)
                loose = mod.get_db_connection(p, check_same_thread=False)
                state = [
                    x.execute(q).fetchone()
                    for x in (conn, fresh, loose)
                    for q in (
                        "PRAGMA journal_mode",
                        "PRAGMA synchronous",
                        "PRAGMA foreign_keys",
                        "PRAGMA busy_timeout",
                    )
                ]
                conn.close()
                reopened = _run(lambda: mod.get_db_connection(p), caplog)
                pair.append(
                    (
                        first[0][0], first[1], again is conn, fresh is not conn, loose is not conn,
                        state, reopened[0][0], reopened[1], mod._DB_CONN_CACHE.connections[(p, True)] is not conn,
                    )
                )  # fmt: skip
            assert _aligned(pair[0], tmp_path) == _aligned(pair[1], tmp_path), (
                name,
                mode,
                pair,
            )
    # an unwritable parent directory
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    outs = [
        _run(lambda m=m: m.get_db_connection(str(blocked / "x" / "y.db")), caplog)
        for m in (ref, new)
    ]
    assert outs[0] == outs[1]


def _aligned(value, tmp_path):
    """One side's results with its own directory written as <p>, so the paths
    each side logs compare equal."""
    text = repr(_mask(value))
    return text.replace(str(tmp_path / "new"), "<p>").replace(
        str(tmp_path / "ref"), "<p>"
    )


def test_readonly_connections_match(tmp_path, caplog):
    for side, mod in (("ref", ref), ("new", new)):
        p = tmp_path / f"{side}.db"
        c = sqlite3.connect(p)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("CREATE TABLE t (a)")
        c.execute("INSERT INTO t VALUES (1)")
        c.commit()
        c.close()
        for sidecar in ("-shm", "-wal"):
            if os.path.exists(str(p) + sidecar):
                os.remove(str(p) + sidecar)
    results = []
    for side, mod in (("ref", ref), ("new", new)):
        conn = mod.connect_readonly(str(tmp_path / f"{side}.db"))
        rows = conn.execute("SELECT * FROM t").fetchall()
        try:
            conn.execute("INSERT INTO t VALUES (2)")
            wrote = "wrote"
        except sqlite3.Error as e:
            wrote = (type(e).__name__, str(e))
        results.append((rows, wrote, conn.execute("PRAGMA query_only").fetchone()))
        conn.close()
    assert results[0] == results[1]


def _screener_setup(tmp_path, monkeypatch, tag):
    paths = {}
    for side, mod in (("ref", ref), ("new", new)):
        p = str(tmp_path / f"{tag}_{side}_screener.db")
        monkeypatch.setattr(mod, "get_global_screener_db_path", lambda p=p: p)
        monkeypatch.setattr(mod, "_SCREENER_SCHEMA_INITED", False)
        paths[side] = p
    return paths


def _result(rng, symbols):
    r = {"symbol": str(rng.choice(symbols))} if rng.random() > 0.05 else {}
    for k in ("name", "price", "intrinsic_value", "margin_of_safety", "pe_ratio", "market_cap", "sector",
              "ai_moat", "ai_financial_strength", "ai_predictability", "ai_growth", "ai_summary", "ai_sentiment",
              "last_fiscal_year_end", "most_recent_quarter"):  # fmt: skip
        if rng.random() < 0.7:
            r[k] = [
                None,
                1.5,
                7,
                "x" * int(rng.integers(0, 40)),
                np.float64(2.5),
                -3.0,
            ][int(rng.integers(0, 6))]
    if rng.random() < 0.4:
        r["ai_catalysts"] = ["a", "b"] if rng.random() < 0.7 else '["c"]'
    if rng.random() < 0.4:
        r["valuation_details"] = (
            {
                "dcf": np.float64(1.5),
                "n": np.int64(3),
                "arr": np.arange(2),
                "nan": np.float64("nan"),
            }
            if rng.random() < 0.7
            else "{}"
        )
    return r


@pytest.mark.parametrize("seed", range(6))
def test_screener_cache_matches(tmp_path, monkeypatch, caplog, seed):
    rng = np.random.default_rng(8300 + seed)
    paths = _screener_setup(tmp_path, monkeypatch, f"s{seed}")
    symbols = ["AAPL", "msft", "PORTFOLIO-WIDE", "Total", "NVDA"]
    for step in range(60):
        r = rng.random()
        if r < 0.25:
            batch = [_result(rng, symbols) for _ in range(int(rng.integers(0, 4)))]
            universe = [None, "sp500", "manual"][int(rng.integers(0, 3))]
            fns = [
                lambda m=m: m.upsert_screener_results(
                    [dict(x) for x in batch], universe
                )
                for m in (ref, new)
            ]
        elif r < 0.4:
            batch = [_result(rng, symbols) for _ in range(int(rng.integers(0, 4)))]
            fns = [
                lambda m=m: m.refresh_screener_rows_by_symbol([dict(x) for x in batch])
                for m in (ref, new)
            ]
        elif r < 0.55:
            sym = str(rng.choice(symbols))
            ai = {"scorecard": {"moat": 7, "growth": 5} if rng.random() > 0.1 else None, "summary": "s" * 30,
                  "analysis": {"moat": "wide", "growth_perspective": 3}, "catalysts": ["x"] if rng.random() < 0.5 else [],
                  "valuation_details": {"a": 1} if rng.random() < 0.5 else None, "sentiment": 0.4}  # fmt: skip
            info = (
                {"shortName": "N", "currentPrice": 10.5, "sector": "Tech"}
                if rng.random() < 0.5
                else None
            )
            universe = str(rng.choice(["manual", "sp500"]))
            fns = [
                lambda m=m: m.update_ai_review_in_cache(sym, ai, info, universe)
                for m in (ref, new)
            ]
        elif r < 0.7:
            sym = str(rng.choice(symbols))
            info = (
                {
                    "shortName": "N",
                    "valuation_details": {"x": np.float64(2.0)}
                    if rng.random() < 0.5
                    else "raw",
                }
                if rng.random() < 0.6
                else None
            )
            args = (
                sym,
                [None, 12.5][int(rng.integers(0, 2))],
                0.25,
                2024,
                None,
                info,
                "manual",
            )
            fns = [
                lambda m=m: m.update_intrinsic_value_in_cache(*args) for m in (ref, new)
            ]
        elif r < 0.8:
            want = list(rng.choice(symbols, int(rng.integers(0, 4))))
            fns = [
                lambda m=m: m.get_cached_screener_results(list(want))
                for m in (ref, new)
            ]
        elif r < 0.9:
            universe = str(rng.choice(["manual", "sp500", "none"]))
            fns = [
                lambda m=m: m.get_screener_results_by_universe(universe)
                for m in (ref, new)
            ]
        else:
            fns = [
                lambda m=m: m.get_all_distinct_screener_results() for m in (ref, new)
            ]
        a = _run(fns[0], caplog)
        b = _run(fns[1], caplog)
        same_path = lambda v: repr(_mask(v)).replace("_new_screener", "_ref_screener")  # noqa: E731
        assert same_path(a) == same_path(b), (seed, step, a, b)
        assert _schema(paths["ref"]) == _schema(paths["new"]), (seed, step)
