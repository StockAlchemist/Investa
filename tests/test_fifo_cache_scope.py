import asyncio
from unittest.mock import AsyncMock

import pandas as pd

from server.auth import User
from server.routes import portfolio as routes


def test_fifo_cache_reuses_matching_inputs_and_separates_accounts_and_settings(monkeypatch):
    df = pd.DataFrame([{
        "Date": pd.Timestamp("2026-01-01"), "Symbol": "AAPL", "Account": "A",
        "Type": "Buy", "Quantity": 1, "Price/Share": 10, "original_index": 0,
    }])
    summary = {"metrics": {"market_value": 10}, "summary_df": pd.DataFrame([
        {"Symbol": "AAPL", "Quantity": 1, "Price": 10, "Market Value": 10},
    ])}
    monkeypatch.setattr(routes, "_get_historical_performance_cached", AsyncMock(return_value=(pd.DataFrame(), {}, {}, "")))
    monkeypatch.setattr(routes, "_calculate_portfolio_summary_internal", AsyncMock(return_value=summary))
    calls = []

    def fifo(**kwargs):
        calls.append(kwargs)
        return pd.DataFrame(), {}

    monkeypatch.setattr(routes, "calculate_fifo_lots_and_gains", fifo)
    monkeypatch.setattr(routes, "_FIFO_LOTS_CACHE", routes.OrderedDict())
    user = User(id=1, username="isolated-fifo-test", created_at="2026-01-01")

    async def scenario():
        data = (df, {}, {}, set(), {}, {}, "/isolated-fifo.db", 1)
        for accounts, overrides in [(["A"], {}), (["A"], {}), (["B"], {}), (["B"], {"AAPL": 12})]:
            scoped = (df, overrides, *data[2:])
            response = await routes.get_stock_position("AAPL", currency="USD", accounts=accounts, data=scoped, current_user=user)
            assert response["symbol"] == "AAPL"
        assert len(calls) == 3

    asyncio.run(scenario())
