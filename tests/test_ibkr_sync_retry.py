from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from ibkr_connector import IBKRConnector
from server.auth import User
from server.routes import transactions as routes


def failure(code):
    return f"<FlexStatementResponse><Status>Fail</Status><ErrorCode>{code}</ErrorCode><ErrorMessage>Please try again shortly.</ErrorMessage></FlexStatementResponse>"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("ibkr_connector.time.sleep", lambda _: None)
    monkeypatch.setattr(
        routes,
        "get_db_connection",
        lambda _: pytest.fail(
            "An unsuccessful or empty report must not open the portfolio"
        ),
    )
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_current_user] = lambda: User(
        id=1, username="test", created_at="2026-01-01"
    )
    app.dependency_overrides[routes.get_transaction_data] = lambda: (
        None,
        {},
        {},
        set(),
        {},
        {},
        "unused.db",
        0,
    )
    app.dependency_overrides[routes.get_config_manager] = lambda: SimpleNamespace(
        manual_overrides={"ibkr_token": "mock-token", "ibkr_query_id": "mock-query"}
    )
    with TestClient(app) as http:
        yield http


@pytest.mark.parametrize(
    "code,status,app_code,attempts",
    [
        (1001, 409, "IBKR_BUSY", 3),
        (1009, 409, "IBKR_BUSY", 3),
        (1015, 400, "IBKR_CONFIG", 1),
        (1003, 500, "SYNC_FAILED", 1),
    ],
)
def test_sync_classifies_provider_error_and_does_not_open_portfolio(
    client, monkeypatch, code, status, app_code, attempts
):
    calls = []

    def request(self, url, params):
        calls.append(url)
        return failure(code)

    monkeypatch.setattr(IBKRConnector, "_make_request", request)
    response = client.post("/sync/ibkr")
    assert response.status_code == status
    assert response.json()["code"] == app_code
    assert response.json()["message"] == response.json()["detail"]
    assert response.json()["message"].startswith("IBKR:")
    assert len(calls) == attempts


def test_sync_succeeds_when_1001_clears_on_retry(client, monkeypatch):
    responses = iter(
        [
            failure(1001),
            "<FlexStatementResponse><Status>Success</Status><ReferenceCode>ref</ReferenceCode><url>https://example.com/statement</url></FlexStatementResponse>",
            "<FlexQueryResponse><FlexStatements><FlexStatement/></FlexStatements></FlexQueryResponse>",
        ]
    )
    monkeypatch.setattr(
        IBKRConnector, "_make_request", lambda self, url, params: next(responses)
    )
    response = client.post("/sync/ibkr")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["added_count"] == 0
