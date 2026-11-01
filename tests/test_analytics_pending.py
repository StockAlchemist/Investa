"""A cold summary must stay retryable instead of becoming a successful empty result."""

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server.routes import analytics


@pytest.mark.parametrize(
    "endpoint",
    [
        "projection",
        "earnings_calendar",
        "dividend_calendar",
        "attribution",
        "projected_income",
    ],
)
def test_pending_summary_preserves_http_status_and_recovers(
    endpoint, monkeypatch, caplog
):
    app = FastAPI()
    app.include_router(analytics.router, prefix="/api")
    data = (pd.DataFrame({"Symbol": ["TEST"]}), {}, {}, set(), {}, {}, "unused.db", 0)
    app.dependency_overrides[analytics.get_transaction_data] = lambda: data
    app.dependency_overrides[analytics.get_current_user] = lambda: None
    app.dependency_overrides[analytics.get_config_manager] = lambda: None
    pending = HTTPException(
        status_code=503,
        detail="Portfolio calculation is still running. Please retry shortly.",
        headers={"Retry-After": "2"},
    )
    summary = {"metrics": {"market_value": 100}, "summary_df": pd.DataFrame()}
    if endpoint == "projected_income":
        fetch = AsyncMock(side_effect=[pending, ({}, pd.DataFrame(), {}, {})])
        monkeypatch.setattr(analytics, "_compute_raw_summary", fetch)
    else:
        fetch = AsyncMock(side_effect=[pending, summary])
        monkeypatch.setattr(analytics, "_calculate_portfolio_summary_internal", fetch)
    monkeypatch.setattr(
        analytics,
        "_projection_history",
        AsyncMock(return_value=pd.DataFrame({"Portfolio Accumulated Gain": [1, 1.1]})),
    )
    monkeypatch.setattr(
        analytics, "compute_projection", lambda *args, **kwargs: {"available": True}
    )
    client = TestClient(app)
    response = client.get(f"/api/{endpoint}")
    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert response.json() == {"detail": pending.detail}
    assert not any(record.levelname == "ERROR" for record in caplog.records)

    response = client.get(f"/api/{endpoint}")
    assert response.status_code == 200
    assert response.json() == (
        {"available": True, "currency": "USD"}
        if endpoint == "projection"
        else {"sectors": [], "stocks": []}
        if endpoint == "attribution"
        else []
    )
    assert fetch.await_count == 2
