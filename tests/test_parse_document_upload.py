"""Uploads are written under a name the server chooses, and always cleaned up."""

import os
import sys

import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


@pytest.fixture
def upload(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from server.auth import User
    from server.dependencies import get_current_user
    from server.main import app
    from server.routes import transactions as routes

    seen = {}

    def fake_parse(path, **_):
        seen["path"] = path
        seen["bytes"] = open(path, "rb").read()
        return [{"Symbol": "AAPL"}]

    monkeypatch.setattr(routes, "project_root", str(tmp_path))
    monkeypatch.setattr(routes, "extract_transactions_from_file", fake_parse)
    app.dependency_overrides[get_current_user] = lambda: User(
        id=7, username="t", created_at="2024-01-01"
    )
    client = TestClient(app)

    def post(filename, body=b"%PDF-1.4 statement"):
        return client.post(
            "/api/transactions/parse_document",
            files={"file": (filename, body, "application/pdf")},
        )

    yield post, seen, tmp_path / "data" / "temp_uploads"
    app.dependency_overrides.clear()


def test_a_slash_in_the_filename_is_not_a_path(upload):
    post, seen, temp_dir = upload
    resp = post("../../statements/sept.pdf")
    assert resp.status_code == 200
    assert os.path.dirname(seen["path"]) == str(temp_dir)
    assert seen["path"].endswith(".pdf")  # the parser dispatches on it
    assert seen["bytes"] == b"%PDF-1.4 statement"
    assert os.listdir(temp_dir) == []


def test_an_oversized_upload_is_refused_and_removed(upload, monkeypatch):
    from server.routes import transactions as routes

    monkeypatch.setattr(routes, "MAX_UPLOAD_BYTES", 10)
    post, seen, temp_dir = upload
    resp = post("big.pdf", b"x" * 11)
    assert resp.status_code == 413
    assert "path" not in seen
    assert os.listdir(temp_dir) == []
