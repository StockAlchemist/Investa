"""The admin routes must not be callable by anyone who can reach the server."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    from server.main import app

    # No `with`: the lifespan would start the refresh workers.
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_clear_cache_requires_login(client):
    resp = client.post("/api/clear_cache")
    assert resp.status_code == 401


def test_webhook_is_off_without_a_configured_secret(client, monkeypatch):
    monkeypatch.delenv("INVESTA_WEBHOOK_SECRET", raising=False)
    # The value that used to be the built-in default must not open it.
    resp = client.post(
        "/api/webhook/refresh", json={"secret": "investa_refresh_secret_123"}
    )
    assert resp.status_code == 503


def test_webhook_rejects_a_wrong_secret(client, monkeypatch):
    monkeypatch.setenv("INVESTA_WEBHOOK_SECRET", "correct-horse")
    resp = client.post("/api/webhook/refresh", json={"secret": "wrong"})
    assert resp.status_code == 403


def test_webhook_accepts_the_configured_secret(client, monkeypatch, tmp_path):
    from server.routes import admin

    reloads = []
    monkeypatch.setenv("INVESTA_WEBHOOK_SECRET", "correct-horse")
    monkeypatch.setattr(admin.config, "get_app_data_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        admin, "reload_data_and_clear_cache", lambda: reloads.append(True)
    )

    resp = client.post("/api/webhook/refresh", json={"secret": " correct-horse "})
    assert resp.status_code == 200
    assert reloads == [True]
