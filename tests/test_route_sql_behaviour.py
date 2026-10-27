"""Behaviour of the routes and helpers whose SQL moves into Rust (phase 17b),
pinned before the move and held after it.

Each flow runs end to end on temporary databases: the account lifecycle
(register, duplicate, login, the current user, alias, password change,
delete), holding tags, IBKR staging with its duplicate checks and amount
fallback, the pending list, approve and reject, the settings currency map,
the archive's data-quality probe, the heatmap's EDGAR facts, narrative
search's AI-written query, and the AI workers' cache checks.
"""

import os
import sqlite3
import sys

import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
from db_utils import initialize_database, initialize_global_database  # noqa: E402
from server.auth import User  # noqa: E402


@pytest.fixture
def app_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_app_data_dir", lambda: str(tmp_path))
    os.makedirs(tmp_path / config.DB_DIR, exist_ok=True)
    return tmp_path


@pytest.fixture
def auth_client(app_dir, monkeypatch):
    from server import dependencies
    from server.routes import auth as auth_routes

    for limiter in ("register_ip_limiter", "login_ip_limiter", "failed_auth_limiter"):
        lim = getattr(auth_routes, limiter, None)
        if lim is not None and hasattr(lim, "reset_all"):
            lim.reset_all()
    initialize_global_database().close()
    global_path = os.path.join(str(app_dir), config.DB_DIR, config.GLOBAL_DB_FILENAME)

    def global_conn():
        conn = sqlite3.connect(global_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    app = FastAPI()
    app.include_router(auth_routes.router)
    app.dependency_overrides[dependencies.get_global_db_connection] = global_conn
    with TestClient(app) as client:
        yield client, global_path


def test_account_lifecycle(auth_client, app_dir):
    client, global_path = auth_client
    r = client.post("/auth/register", json={"username": "alice", "password": "pw-one-1"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["username"] == "alice" and body["id"] == 1 and body["is_active"]
    assert os.path.exists(app_dir / config.USERS_DIR / "alice" / config.PORTFOLIO_DB_FILENAME)
    assert client.post("/auth/register", json={"username": "alice", "password": "x"}).status_code == 400

    assert client.post("/auth/login", data={"username": "alice", "password": "wrong"}).status_code == 401
    r = client.post("/auth/login", data={"username": "alice", "password": "pw-one-1"})
    assert r.status_code == 200
    token = r.json()["access_token"]
    auth = {"Authorization": f"Bearer {token}"}

    me = client.get("/auth/me", headers=auth).json()
    assert (me["id"], me["username"], me["alias"]) == (1, "alice", None)
    client.cookies.clear()
    assert client.get("/auth/me").json() is None

    r = client.patch("/auth/me", json={"alias": "Al"}, headers=auth)
    assert r.status_code == 200 and r.json()["alias"] == "Al"
    with sqlite3.connect(global_path) as c:
        assert c.execute("SELECT alias FROM users WHERE id = 1").fetchone() == ("Al",)

    bad = client.post("/auth/change-password", json={"current_password": "nope", "new_password": "pw-two-2"}, headers=auth)
    assert bad.status_code == 400
    ok = client.post("/auth/change-password", json={"current_password": "pw-one-1", "new_password": "pw-two-2"}, headers=auth)
    assert ok.status_code == 200
    assert client.post("/auth/login", data={"username": "alice", "password": "pw-two-2"}).status_code == 200

    r = client.delete("/auth/me", headers=auth)
    assert r.status_code == 200 and "alice" in r.json()["message"]
    assert not os.path.exists(app_dir / config.USERS_DIR / "alice")
    client.cookies.clear()
    assert client.get("/auth/me", headers=auth).json() is None
    assert client.delete("/auth/me", headers=auth).status_code == 401
    with sqlite3.connect(global_path) as c:
        assert c.execute("SELECT COUNT(*) FROM users").fetchone() == (0,)


@pytest.fixture
def tx_client(tmp_path, monkeypatch):
    from server.routes import transactions as routes

    path = str(tmp_path / "portfolio.db")
    initialize_database(path).close()
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.executemany(
        'INSERT INTO transactions (Date, Type, Symbol, Quantity, "Price/Share", "Total Amount", Account, '
        '"Local Currency", ExternalID, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)',
        [("2026-01-02", "Buy", "AAPL", 3, 100.0, 300.0, "IBKR", "USD", "e-1"),
         ("2026-01-03", "Sell", "AAPL", 1, 110.0, 110.0, "IBKR", "USD", None),
         ("2026-01-04", "Buy", "AAPL", 2, 90.0, 180.0, "SET", "USD", None),
         ("2026-01-05", "Buy", "PTT", 5, 30.0, 150.0, "SET", "THB", None)],
    )  # fmt: skip
    conn.execute(
        'INSERT INTO pending_transactions (Date, Type, Symbol, Quantity, "Price/Share", "Total Amount", Account, '
        '"Local Currency", ExternalID, user_id) VALUES (\'2026-02-01\', \'Buy\', \'MSFT\', 1, 400.0, 400.0, \'IBKR\', \'USD\', \'p-1\', 1)'
    )
    conn.commit()
    reloads = []
    monkeypatch.setattr(routes, "reload_data_and_clear_cache", lambda u: reloads.append(u.id))
    app = FastAPI()
    app.include_router(routes.router)
    user = User(id=1, username="test", created_at="2026-01-01")
    app.dependency_overrides[routes.get_current_user] = lambda: user
    app.dependency_overrides[routes.get_user_db_connection] = lambda: conn
    app.dependency_overrides[routes.get_transaction_data] = lambda: (None, None, None, None, None, "manual", path, None)
    with TestClient(app) as client:
        yield client, conn, routes, reloads
    conn.close()


def test_holding_tags(tx_client):
    client, conn, _routes, reloads = tx_client
    r = client.post("/holdings/update_tags", json={"account": "IBKR", "symbol": "AAPL", "tags": "  core ,growth  "})
    assert r.status_code == 200 and r.json()["message"] == "Updated tags for 2 transactions"
    rows = sqlite3.connect(conn.execute("PRAGMA database_list").fetchone()[2]).execute(
        "SELECT Account, Tags FROM transactions WHERE Symbol = 'AAPL' ORDER BY id").fetchall()
    assert rows == [("IBKR", "core ,growth"), ("IBKR", "core ,growth"), ("SET", None)]
    assert client.post("/holdings/update_tags", json={"account": "X", "symbol": "Y", "tags": ""}).json()["message"] == "Updated tags for 0 transactions"
    assert reloads == [1, 1]


class FakeConnector:
    rows = []

    def __init__(self, token, query_id):
        pass

    def sync(self):
        return list(FakeConnector.rows)


def test_ibkr_staging_pending_approve_reject(tx_client, monkeypatch):
    import datetime as dt

    client, conn, routes, reloads = tx_client
    monkeypatch.setattr(config, "IBKR_TOKEN", "t")
    monkeypatch.setattr(config, "IBKR_QUERY_ID", "q")
    monkeypatch.setattr(routes, "IBKRConnector", FakeConnector)

    class CM:
        manual_overrides = {}

    app = client.app
    app.dependency_overrides[routes.get_config_manager] = lambda: CM()
    FakeConnector.rows = [
        {"ExternalID": "e-1", "Date": "2026-03-01", "Type": "buy", "Symbol": "AAPL"},  # in transactions
        {"ExternalID": "p-1", "Date": "2026-03-01", "Type": "buy", "Symbol": "MSFT"},  # already pending
        {"ExternalID": "n-1", "Date": dt.date(2026, 3, 2), "Type": " buy ", "Symbol": "NVDA", "Quantity": 2,
         "Price/Share": 10.0, "Commission": 1.0, "Account": "IBKR", "Local Currency": "USD"},
        {"ExternalID": "n-2", "Date": dt.datetime(2026, 3, 3, 9), "Type": "sell", "Symbol": "NVDA", "Quantity": 1,
         "Price/Share": 12.0, "Commission": 0.5, "Account": "IBKR", "Amount": 11.5, "Local Currency": "USD"},
        {"ExternalID": None, "Date": "2026-03-04", "Type": "dividend", "Symbol": "AAPL", "Total Amount": float("nan"),
         "Quantity": 0, "Price/Share": 0, "Account": "IBKR", "Local Currency": "USD"},
        {"Date": "2026-03-05", "Type": "fees", "Symbol": "$CASH", "Total Amount": 3.0, "Note": "fee", "Account": "IBKR",
         "Local Currency": "USD"},
    ]  # fmt: skip
    r = client.post("/sync/ibkr")
    assert r.status_code == 200, r.text
    assert (r.json()["staged_count"], r.json()["duplicate_count"]) == (4, 2)
    staged = conn.execute(
        'SELECT Date, Type, Symbol, "Total Amount", ExternalID, user_id FROM pending_transactions ORDER BY id'
    ).fetchall()
    assert staged == [
        ("2026-02-01", "Buy", "MSFT", 400.0, "p-1", 1),
        # 2 x 10 - 1: the fallback tests the unstripped " BUY ", so the commission is
        # subtracted (a known quirk, see docs/rust_core_port.md "Found while porting").
        ("2026-03-02", "Buy", "NVDA", 19.0, "n-1", 1),
        ("2026-03-03", "Sell", "NVDA", 11.5, "n-2", 1),
        ("2026-03-04", "Dividend", "AAPL", None, None, 1),
        ("2026-03-05", "Fees", "$CASH", 3.0, None, 1),
    ]
    FakeConnector.rows = []
    assert client.post("/sync/ibkr").json()["added_count"] == 0

    # A NULL number in any pending row becomes NaN in the frame, which the JSON
    # response refuses: the list fails (a known quirk, see "Found while porting").
    with pytest.raises(ValueError, match="not JSON compliant"):
        client.get("/sync/ibkr/pending")
    for col in ("Quantity", "Price/Share", "Total Amount", "Commission", "Split Ratio"):
        conn.execute(f'UPDATE pending_transactions SET "{col}" = 0 WHERE "{col}" IS NULL')
    conn.commit()
    pending = client.get("/sync/ibkr/pending")
    assert pending.status_code == 200
    got = [(p["Symbol"], p["Date"], p["ExternalID"], p["Total Amount"]) for p in pending.json()]
    assert got == [("$CASH", "2026-03-05", None, 3.0), ("AAPL", "2026-03-04", None, 0.0), ("NVDA", "2026-03-03", "n-2", 11.5),
                   ("NVDA", "2026-03-02", "n-1", 19.0), ("MSFT", "2026-02-01", "p-1", 400.0)]  # fmt: skip

    ids = [row[0] for row in conn.execute("SELECT id FROM pending_transactions ORDER BY id")]
    r = client.post("/sync/ibkr/approve", json={"ids": ids[:3], "accounts": {str(ids[1]): "Taxable"}})
    assert r.status_code == 200 and r.json()["count"] == 3
    assert conn.execute("SELECT Symbol, Account FROM transactions WHERE ExternalID IN ('p-1','n-1','n-2') ORDER BY id").fetchall() == [
        ("MSFT", "IBKR"), ("NVDA", "Taxable"), ("NVDA", "IBKR")]
    r = client.post("/sync/ibkr/reject", json=ids[3:] + [9999])
    assert r.status_code == 200 and r.json()["count"] == 2
    assert conn.execute("SELECT COUNT(*) FROM pending_transactions").fetchone() == (0,)
    assert client.get("/sync/ibkr/pending").json() == []
    assert client.post("/sync/ibkr/reject", json=[]).json()["count"] == 0
    # Last, because the failed sync leaves its write transaction open (a known
    # quirk): a staged row the schema refuses (no currency) fails the sync.
    FakeConnector.rows = [{"ExternalID": "z-1", "Date": "2026-03-09", "Type": "buy", "Symbol": "Z", "Account": "IBKR"}]
    r = client.post("/sync/ibkr")
    assert (r.status_code, r.json()) == (500, {"status": "error", "message": "IBKR sync failed"})
    assert conn.execute("SELECT COUNT(*) FROM pending_transactions WHERE ExternalID = 'z-1'").fetchone() == (0,)


def test_settings_currency_map(tmp_path):
    from server.routes.settings import _get_symbol_currency_map_sql

    path = str(tmp_path / "p.db")
    initialize_database(path).close()
    with sqlite3.connect(path) as c:
        c.executemany('INSERT INTO transactions (Date, Type, Symbol, "Local Currency", Account) VALUES (?, ?, ?, ?, ?)',
                      [("2026-01-01", "Buy", "AAPL", "USD", "A"), ("2026-01-01", "Buy", "PTT", "THB", "A"),
                       ("2026-01-01", "Buy", "", "USD", "A"), ("2026-01-01", "Buy", "AAPL", "USD", "A")])  # fmt: skip
    assert _get_symbol_currency_map_sql(path) == {"AAPL": "USD", "PTT": "THB"}
    assert _get_symbol_currency_map_sql(str(tmp_path / "missing.db")) == {}


def test_data_quality_probe(tmp_path, monkeypatch):
    from server.routes import market

    class Db:
        def __init__(self, path):
            self.path = path

        def _get_connection(self):
            return sqlite3.connect(self.path)

    class Mdp:
        def __init__(self, path):
            self.db = Db(path)

    path = str(tmp_path / "m.db")
    sqlite3.connect(path).close()
    monkeypatch.setattr(market, "get_mdp", lambda: Mdp(path))
    assert market._data_quality_table_exists() is False
    with sqlite3.connect(path) as c:
        c.execute("CREATE TABLE data_quality (symbol TEXT)")
    assert market._data_quality_table_exists() is True
    monkeypatch.setattr(market, "get_mdp", lambda: None)
    assert market._data_quality_table_exists() is False


def test_heatmap_edgar_facts(tmp_path, monkeypatch):
    import edgar_provider
    from server.routes import market

    store = edgar_provider.EdgarFactStore(str(tmp_path / "facts.db"))
    rows = []
    for cik in ("0000000001", "0000000002"):
        for tag, val in ((market._EDGAR_EPS_TAG, 2.5), (market._EDGAR_EQUITY_TAG, 1e9), ("Unrelated", 1.0)):
            rows.append((cik, tag, "2024-12-31", "2024-01-01", val, "USD", "10-K", f"a-{cik}", "2025-02-01"))
    store.upsert_facts(rows)
    monkeypatch.setattr(edgar_provider, "_shared_store", store)
    got = market._edgar_annual_facts([{"symbol": "A", "cik": "0000000001"}, {"symbol": "B", "cik": "0000000002"},
                                      {"symbol": "C"}])  # fmt: skip
    assert set(got) <= {"A", "B"} and got
    assert market._edgar_annual_facts([]) == {}


def _screener_db(path):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE screener_cache (symbol TEXT, universe TEXT, name TEXT, ai_summary TEXT,
                 last_fiscal_year_end INTEGER, most_recent_quarter INTEGER, ai_moat REAL, ai_financial_strength REAL,
                 ai_predictability REAL, ai_growth REAL, ai_catalysts TEXT, updated_at TEXT,
                 PRIMARY KEY (symbol, universe))""")  # fmt: skip
    c.executemany("INSERT INTO screener_cache VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", [
        ("AAPL", "sp500", "Apple", "A long and healthy business summary text", 1, 2, 8, 9, 7, 6, '["iphone"]', "2026-01-02"),
        ("MSFT", "sp500", "Microsoft", "short", 1, 2, 9, None, 8, 7, "not json", "2026-01-03"),
        ("TSLA", "manual", "Tesla", None, None, None, None, None, None, None, None, "2026-01-01"),
        ("IBM", None, "IBM", "an error occurred while generating", 1, 1, 0, 0, 0, 0, None, None),
    ])  # fmt: skip
    c.commit()
    c.close()


def test_screener_reads(tmp_path, monkeypatch):
    import ai_review_scores
    import ai_review_worker
    import ai_review_worker_missing
    import db_utils
    from server import screener_service

    path = str(tmp_path / "screener.db")
    _screener_db(path)
    monkeypatch.setattr(db_utils, "get_global_screener_db_path", lambda: path)
    monkeypatch.setattr(ai_review_worker, "get_global_screener_db_path", lambda: path, raising=False)
    monkeypatch.setattr(ai_review_worker, "_open_screener_conn", lambda: sqlite3.connect(path))
    monkeypatch.setattr(ai_review_worker_missing, "_open_screener_conn", lambda: sqlite3.connect(path))
    assert ai_review_worker_missing.get_missing_reviews() == {"sp500": ["MSFT"], "manual": ["TSLA", "IBM"]}
    frame = ai_review_scores.load_ai_reviews()
    assert list(frame.index) == ["MSFT", "AAPL"] and "ai_reviewed_at" in frame.columns
    assert list(ai_review_scores.load_ai_reviews(["aapl"]).index) == ["AAPL"]
    monkeypatch.setattr(screener_service, "get_global_screener_db_path", lambda: path, raising=False)
    sql = "SELECT symbol, ai_moat, ai_financial_strength, ai_predictability, ai_growth, ai_catalysts FROM screener_cache ORDER BY symbol"

    class Resp:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"candidates": [{"content": {"parts": [{"text": "```sql\n" + sql + ";\n```"}]}}]}

    calls = []
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(screener_service.requests, "post", lambda url, **kw: calls.append(url) or Resp())
    results = screener_service.run_narrative_search("anything")
    assert [r["symbol"] for r in results] == ["AAPL", "IBM", "MSFT", "TSLA"]
    assert results[0]["ai_score"] == 7.5 and results[0]["ai_catalysts"] == ["iphone"]
    assert results[2]["ai_score"] == 8.0 and results[2]["ai_catalysts"] == "not json"
    assert results[3]["ai_score"] is None
    assert len(calls) == 1  # mocked: no network
