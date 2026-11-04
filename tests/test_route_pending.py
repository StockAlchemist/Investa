"""A cold summary's 503 must reach the client from every route that awaits it.

The portfolio health, AI review, stock position and screener routes caught it
as a generic failure: health logged an ERROR traceback and answered 200 with a
zero score, the others turned it into a 500 the clients do not retry.
"""

from unittest.mock import AsyncMock

import pandas as pd
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from server.routes import portfolio, screener

PENDING = HTTPException(
    status_code=503,
    detail="Portfolio calculation is still running. Please retry shortly.",
    headers={"Retry-After": "2"},
)

ROUTES = {  # name: (router module, method, path, body)
    "health": (portfolio, "get", "/api/portfolio_health", None),
    "ai_review": (portfolio, "post", "/api/portfolio/ai_review", None),
    "position": (portfolio, "get", "/api/stock/TEST/position", None),
    "screener": (screener, "post", "/api/screener/run", {"universe_type": "holdings"}),
}


@pytest.mark.parametrize("name", ROUTES)
def test_pending_summary_reaches_the_client(name, monkeypatch, caplog):
    module, method, path, body = ROUTES[name]
    app = FastAPI()
    app.include_router(module.router, prefix="/api")
    df = pd.DataFrame(
        {
            "Symbol": ["TEST"],
            "Date": [pd.Timestamp("2024-01-02")],
            "original_index": [0],
        }
    )
    data = (df, {}, {}, set(), {}, {}, "unused.db", 0)
    app.dependency_overrides[module.get_transaction_data] = lambda: data
    app.dependency_overrides[module.get_current_user] = lambda: None
    if hasattr(module, "get_user_db_connection"):
        app.dependency_overrides[module.get_user_db_connection] = lambda: None
    summary = AsyncMock(side_effect=PENDING)
    monkeypatch.setattr(module, "_calculate_portfolio_summary_internal", summary)
    if module is portfolio:
        monkeypatch.setattr(
            portfolio,
            "_get_historical_performance_cached",
            AsyncMock(return_value=(pd.DataFrame(), {}, {}, "")),
        )
        monkeypatch.setattr(  # the position route builds its lots first
            portfolio, "calculate_fifo_lots_and_gains", lambda **_: (pd.DataFrame(), {})
        )

    client = TestClient(app)
    response = client.post(path, json=body) if method == "post" else client.get(path)

    assert response.status_code == 503
    assert response.headers["Retry-After"] == "2"
    assert response.json() == {"detail": PENDING.detail}
    assert summary.await_count == 1
    assert not any(record.levelname == "ERROR" for record in caplog.records)
