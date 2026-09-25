"""The built-in Favorites list and the stock window's membership lookup.

Favorites is an ordinary watchlist found by name, so the cases that matter are
the ones that would split or lose it: a second list taking the name, the list
being renamed away, or a repeated "ensure" creating duplicates.
"""

import os
import sqlite3
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def client(tmp_path):
    from fastapi.testclient import TestClient

    import db_utils
    from server.dependencies import get_current_user, get_user_db_connection
    from server.main import app

    db_path = str(tmp_path / "portfolio.db")
    db_utils.initialize_database(db_path).close()

    def connection():
        conn = sqlite3.connect(db_path, check_same_thread=False)
        try:
            yield conn
        finally:
            conn.close()

    app.dependency_overrides[get_current_user] = lambda: SimpleNamespace(
        id=1, username="tester"
    )
    app.dependency_overrides[get_user_db_connection] = connection
    # No `with`: the lifespan would start the refresh workers.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_favorites_is_created_once_and_listed_first(client):
    first = client.post("/api/watchlists/favorites").json()
    second = client.post("/api/watchlists/favorites").json()

    assert first["id"] == second["id"]
    assert first["name"] == "Favorites" and first["is_favorites"] is True

    lists = client.get("/api/watchlists").json()
    assert lists[0]["id"] == first["id"] and lists[0]["is_favorites"] is True
    assert sum(w["is_favorites"] for w in lists) == 1


def test_the_reserved_name_cannot_be_taken_or_renamed_away(client):
    favorites = client.post("/api/watchlists/favorites").json()
    other = client.post("/api/watchlists", json={"name": "Tech"}).json()

    assert (
        client.post("/api/watchlists", json={"name": " favorites "}).status_code == 409
    )
    assert (
        client.put(
            f"/api/watchlists/{other['id']}", json={"name": "Favorites"}
        ).status_code
        == 409
    )
    assert (
        client.put(
            f"/api/watchlists/{favorites['id']}", json={"name": "Old"}
        ).status_code
        == 409
    )
    # Other lists rename as before.
    assert (
        client.put(f"/api/watchlists/{other['id']}", json={"name": "Chips"}).status_code
        == 200
    )


def test_membership_lists_every_watchlist_holding_the_symbol(client):
    favorites = client.post("/api/watchlists/favorites").json()
    other = client.post("/api/watchlists", json={"name": "Tech"}).json()

    client.post(
        "/api/watchlist", json={"symbol": "aapl", "watchlist_id": favorites["id"]}
    )
    client.post("/api/watchlist", json={"symbol": "AAPL", "watchlist_id": other["id"]})
    client.post("/api/watchlist", json={"symbol": "MSFT", "watchlist_id": other["id"]})

    body = client.get("/api/watchlists/membership/aapl").json()
    assert body["symbol"] == "AAPL"
    assert body["watchlist_ids"] == sorted([favorites["id"], other["id"]])

    client.delete(f"/api/watchlist/AAPL?id={favorites['id']}")
    assert client.get("/api/watchlists/membership/AAPL").json()["watchlist_ids"] == [
        other["id"]
    ]


def test_a_deleted_favorites_list_comes_back_empty_on_next_use(client):
    favorites = client.post("/api/watchlists/favorites").json()
    client.post(
        "/api/watchlist", json={"symbol": "AAPL", "watchlist_id": favorites["id"]}
    )
    assert client.delete(f"/api/watchlists/{favorites['id']}").status_code == 200

    again = client.post("/api/watchlists/favorites").json()
    assert again["id"] != favorites["id"]
    assert client.get("/api/watchlists/membership/AAPL").json()["watchlist_ids"] == []
