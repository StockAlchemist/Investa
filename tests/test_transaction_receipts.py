import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi import HTTPException

from db_utils import initialize_database
from server.auth import User
from server.routes import transactions as routes
from server.transaction_writes import RECEIPTS_SCHEMA, atomic_submission


@pytest.fixture
def portfolio(tmp_path, monkeypatch):
    path = tmp_path / "portfolio.db"
    conn = initialize_database(str(path))
    conn.execute(RECEIPTS_SCHEMA)
    conn.commit()
    conn.close()
    monkeypatch.setattr(routes, "reload_data_and_clear_cache", lambda _: None)
    data = (None, {}, {}, set(), {}, {"Broker": "Manual"}, str(path), 0)
    user = User(id=1, username="test", created_at="2026-01-01")
    trade = routes.TransactionInput.model_validate(
        {
            "Date": "2026-10-12",
            "Type": "Buy",
            "Symbol": "AAPL",
            "Quantity": 2,
            "Price/Share": 100,
            "Commission": 5,
            "Account": "Broker",
            "Local Currency": "USD",
            "Auto-add Cash": True,
        }
    )
    return path, data, user, trade


def test_concurrent_retries_create_one_trade_with_one_set_of_cash_legs(portfolio):
    path, data, user, trade = portfolio
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda _: routes.create_transaction(trade, data, user, "request-1"),
                range(2),
            )
        )
    assert results[0] == results[1]
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 3
        assert (
            conn.execute("SELECT COUNT(*) FROM transaction_receipts").fetchone()[0] == 1
        )


def test_reused_key_with_changed_body_is_conflict(portfolio):
    _, data, user, trade = portfolio
    routes.create_transaction(trade, data, user, "request-1")
    with pytest.raises(HTTPException) as error:
        routes.create_transaction(
            trade.model_copy(update={"Quantity": 3}), data, user, "request-1"
        )
    assert error.value.status_code == 409


def test_failed_cash_leg_rolls_back_trade_and_receipt(portfolio, monkeypatch):
    path, data, user, trade = portfolio
    original = routes.add_transaction_to_db

    def fail_cash(conn, payload, **kwargs):
        if payload["Symbol"] == "$CASH":
            return False, None
        return original(conn, payload, **kwargs)

    monkeypatch.setattr(routes, "add_transaction_to_db", fail_cash)
    with pytest.raises(HTTPException):
        routes.create_transaction(trade, data, user, "request-1")
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM transaction_receipts").fetchone()[0] == 0
        )


def test_receipts_are_scoped_by_user_and_endpoint():
    conn = sqlite3.connect(":memory:")
    conn.execute(RECEIPTS_SCHEMA)
    for user_id, endpoint in [(1, "single"), (2, "single"), (1, "batch")]:
        response, changed = atomic_submission(
            conn, user_id, endpoint, "same", {}, lambda: {"id": user_id}
        )
        assert changed
        assert response["id"] == user_id
    conn.close()


def test_batch_rolls_back_every_row_when_one_row_is_invalid(portfolio):
    path, data, user, trade = portfolio
    batch = routes.TransactionBatchInput(
        transactions=[trade, trade.model_copy(update={"Date": "invalid"})],
        auto_add_cash=True,
    )
    with pytest.raises(HTTPException) as error:
        routes.add_transactions_batch(batch, data, user, "batch-1")
    assert error.value.status_code == 400
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0
        assert (
            conn.execute("SELECT COUNT(*) FROM transaction_receipts").fetchone()[0] == 0
        )


def test_existing_database_requires_offline_migration(portfolio):
    path, data, user, trade = portfolio
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE transaction_receipts")
    with pytest.raises(HTTPException) as error:
        routes.create_transaction(trade, data, user, "request-1")
    assert error.value.status_code == 503
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM transactions").fetchone()[0] == 0


def test_migration_is_repeatable_and_creates_a_recoverable_backup(portfolio):
    from scripts.migrate_transaction_receipts import migrate

    path, _, _, _ = portfolio
    with sqlite3.connect(path) as conn:
        conn.execute("DROP TABLE transaction_receipts")
    assert migrate(path, dry_run=True)
    assert not path.with_name(path.name + ".before-receipts.bak").exists()
    assert migrate(path)
    assert not migrate(path)
    backup = path.with_name(path.name + ".before-receipts.bak")
    with sqlite3.connect(backup) as conn:
        assert not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='transaction_receipts'"
        ).fetchone()
