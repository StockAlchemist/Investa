import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from db_utils import initialize_database
from server.auth import User
from server.routes import transactions as routes


@pytest.fixture
def staged(tmp_path, monkeypatch):
    path = tmp_path / "portfolio.db"
    initialize_database(str(path)).close()
    conn = sqlite3.connect(path, check_same_thread=False)
    for symbol, quantity, amount in [("ADBE", 122, 28886.23), ("MA", 50, 28337.61)]:
        conn.execute(
            """INSERT INTO pending_transactions
            (Date, Type, Symbol, Quantity, "Price/Share", "Total Amount", Commission,
             Account, "Local Currency", ExternalID, user_id)
            VALUES ('2026-09-25', 'Buy', ?, ?, 236.77, ?, 1.5, 'IBKR', 'USD', ?, 1)""",
            (symbol, quantity, amount, f"ibkr-{symbol}"),
        )
    conn.commit()
    monkeypatch.setattr(routes, "reload_data_and_clear_cache", lambda _: None)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_current_user] = lambda: User(
        id=1, username="test", created_at="2026-01-01"
    )
    app.dependency_overrides[routes.get_user_db_connection] = lambda: conn
    with TestClient(app) as client:
        yield client, conn
    conn.close()


def test_bulk_approval_applies_accounts_and_preserves_import_fields(staged):
    client, conn = staged
    response = client.post(
        "/sync/ibkr/approve",
        json={"ids": [1, 2], "accounts": {"1": " Taxable ", "2": "Retirement"}},
    )
    assert response.status_code == 200
    assert response.json()["count"] == 2
    assert conn.execute(
        "SELECT Symbol, Account, Commission, ExternalID, user_id FROM transactions ORDER BY id"
    ).fetchall() == [
        ("ADBE", "Taxable", 1.5, "ibkr-ADBE", 1),
        ("MA", "Retirement", 1.5, "ibkr-MA", 1),
    ]
    assert conn.execute("SELECT COUNT(*) FROM pending_transactions").fetchone()[0] == 0
    # Repeating an approval cannot insert the already moved rows again.
    assert (
        client.post(
            "/sync/ibkr/approve", json={"ids": [1, 2], "accounts": {"1": "Taxable"}}
        ).json()["count"]
        == 0
    )
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 2


def test_single_approval_leaves_other_pending_rows_untouched(staged):
    client, conn = staged
    assert (
        client.post(
            "/sync/ibkr/approve", json={"ids": [2], "accounts": {"2": "Retirement"}}
        ).status_code
        == 200
    )
    assert conn.execute("SELECT Symbol, Account FROM transactions").fetchall() == [
        ("MA", "Retirement")
    ]
    assert conn.execute("SELECT id, Account FROM pending_transactions").fetchall() == [
        (1, "IBKR")
    ]


@pytest.mark.parametrize("payload", [[1, 2], {"ids": [1, 2]}])
def test_legacy_and_unchanged_rows_keep_imported_accounts(staged, payload):
    client, conn = staged
    assert client.post("/sync/ibkr/approve", json=payload).status_code == 200
    assert conn.execute("SELECT DISTINCT Account FROM transactions").fetchall() == [
        ("IBKR",)
    ]


@pytest.mark.parametrize(
    "accounts", [{"1": ""}, {"1": "   "}, {"1": None}, {"2": "Retirement"}]
)
def test_invalid_account_selections_do_not_move_any_rows(staged, accounts):
    client, conn = staged
    assert (
        client.post(
            "/sync/ibkr/approve", json={"ids": [1], "accounts": accounts}
        ).status_code
        == 422
    )
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM pending_transactions").fetchone()[0] == 2


def test_failed_bulk_approval_rolls_back_earlier_rows(staged):
    client, conn = staged
    conn.execute(
        "CREATE TRIGGER reject_destination BEFORE INSERT ON transactions WHEN NEW.Account = 'Blocked' BEGIN SELECT RAISE(ABORT, 'invalid destination'); END"
    )
    conn.commit()
    assert (
        client.post(
            "/sync/ibkr/approve",
            json={"ids": [1, 2], "accounts": {"1": "Taxable", "2": "Blocked"}},
        ).status_code
        == 500
    )
    assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM pending_transactions").fetchone()[0] == 2
