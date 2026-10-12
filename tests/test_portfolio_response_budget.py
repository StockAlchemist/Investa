import asyncio
import threading
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi import HTTPException

from server.auth import User
from server import portfolio_service as service
from server.routes import portfolio as routes
from server.route_utils import SWRCache, ComputationPending


def test_wait_budget_preserves_one_shared_calculation():
    async def scenario():
        cache = SWRCache()
        finish = asyncio.Event()
        calls = []

        async def compute():
            calls.append(1)
            await finish.wait()
            return "complete"

        for _ in range(2):
            with pytest.raises(ComputationPending):
                await cache.get_or_compute("user", 300, compute, wait_timeout=0.01)
        assert len(calls) == 1
        finish.set()
        assert (
            await cache.get_or_compute("user", 300, compute, wait_timeout=0.1)
            == "complete"
        )

    asyncio.run(scenario())


def test_calculation_timeout_error_is_not_misreported_as_pending():
    async def scenario():
        async def compute():
            raise TimeoutError("provider failed")

        with pytest.raises(TimeoutError, match="provider failed"):
            await SWRCache().get_or_compute("key", 300, compute, wait_timeout=1)

    asyncio.run(scenario())


@pytest.fixture
def portfolio(monkeypatch):
    service.clear_portfolio_caches()
    manager = SimpleNamespace(
        manual_overrides={}, gui_config={}, load_manual_overrides=lambda: None
    )
    monkeypatch.setattr(service, "get_config_manager", lambda _: manager)
    monkeypatch.setattr(routes, "get_config_manager", lambda _: manager)
    monkeypatch.setattr(service, "get_mdp", lambda: None)
    monkeypatch.setattr(service, "is_market_open", lambda: False)
    monkeypatch.setattr(routes, "get_cached_screener_results", lambda _: {})
    monkeypatch.setattr(service, "_SUMMARY_HISTORY_WAIT_SECONDS", 0.02)
    monkeypatch.setattr(service, "_RAW_SUMMARY_WAIT_SECONDS", 0.5)
    df = pd.DataFrame({"Date": pd.to_datetime(["2025-01-01"]), "Account": ["A"]})
    rows = pd.DataFrame(
        [
            {"Symbol": "OPEN", "Account": "A", "Quantity": 2},
            {"Symbol": "CLOSED", "Account": "A", "Quantity": 0},
        ]
    )
    lots = {("OPEN", "A"): {"lots": [{"quantity": 2}]}}
    result = (
        {"market_value": 100, "dividends": 0, "total_buy_cost": 100},
        rows,
        lots,
        {"A": {"market_value": 100}},
        None,
        None,
        None,
    )
    data = (df, {}, {}, set(), {}, {}, "/isolated-response-budget.db", 0)
    user = User(id=1, username="budget-test", created_at="2026-01-01")
    monkeypatch.setattr(service, "calculate_portfolio_summary", lambda **_: result)
    yield data, user, result
    service.clear_portfolio_caches()


def test_current_summary_and_holdings_do_not_wait_for_full_history(
    portfolio, monkeypatch
):
    data, user, _ = portfolio
    finish = threading.Event()
    calls = []

    def historical(**_):
        calls.append(1)
        assert finish.wait(2), "history was never released"
        return pd.DataFrame(), {}, {}, "|||TWR_FACTOR:1.21"

    monkeypatch.setattr(service, "calculate_historical_performance", historical)

    async def scenario():
        try:
            summary = await asyncio.wait_for(
                service._calculate_portfolio_summary_internal(
                    data=data, current_user=user
                ),
                0.5,
            )
            assert summary["metrics"]["market_value"] == 100
            assert summary["metrics"]["cumulative_twr"] is None
            assert summary["freshness"]["refreshing"] is True
            # Repeated polling and the closed/open views share the original work.
            await service._calculate_portfolio_summary_internal(
                data=data, current_user=user, show_closed_positions=False
            )
            holdings = await asyncio.wait_for(
                routes.get_holdings(
                    currency="USD",
                    accounts=None,
                    show_closed=False,
                    data=data,
                    current_user=user,
                    db_conn=None,
                ),
                0.5,
            )
            assert len(calls) == 1
            assert [row["Symbol"] for row in holdings] == ["OPEN"]
            assert holdings[0]["lots"] == [{"quantity": 2}]
            finish.set()
            await asyncio.gather(
                *list(service._PORTFOLIO_HISTORY_CACHE._inflight.values())
            )
            ready = await service._calculate_portfolio_summary_internal(
                data=data, current_user=user
            )
            assert ready["metrics"]["cumulative_twr"] == pytest.approx(21)
            assert ready["freshness"]["refreshing"] is False
            assert len(calls) == 1
        finally:
            finish.set()

    asyncio.run(scenario())


def test_cold_raw_summary_returns_retryable_response_before_proxy_deadline(
    portfolio, monkeypatch
):
    data, user, result = portfolio
    finish = threading.Event()
    calls = []

    def raw(**_):
        calls.append(1)
        assert finish.wait(2)
        return result

    monkeypatch.setattr(service, "calculate_portfolio_summary", raw)
    monkeypatch.setattr(service, "_RAW_SUMMARY_WAIT_SECONDS", 0.02)

    async def scenario():
        try:
            requests = [
                lambda: routes.get_portfolio_summary(
                    currency="USD",
                    accounts=None,
                    show_closed=False,
                    data=data,
                    current_user=user,
                ),
                lambda: routes.get_portfolio_summary_headline(
                    currency="USD", accounts=None, data=data, current_user=user
                ),
                lambda: routes.get_holdings(
                    currency="USD",
                    accounts=None,
                    show_closed=False,
                    data=data,
                    current_user=user,
                    db_conn=None,
                ),
            ]
            for request in requests:
                with pytest.raises(HTTPException) as error:
                    await asyncio.wait_for(request(), 0.5)
                assert error.value.status_code == 503
                assert error.value.headers == {"Retry-After": "2"}
            assert len(calls) == 1
            finish.set()
            await asyncio.gather(*list(service._RAW_CALC_CACHE._inflight.values()))
        finally:
            finish.set()

    asyncio.run(scenario())
