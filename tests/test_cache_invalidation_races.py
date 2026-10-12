import asyncio

from server.route_utils import SWRCache


def test_invalidation_detaches_old_work_and_rejects_its_result():
    async def scenario():
        cache = SWRCache()
        started, finish = asyncio.Event(), asyncio.Event()

        async def old():
            started.set()
            await finish.wait()
            return "old settings"

        async def new():
            return "new settings"

        old_caller = asyncio.create_task(cache.get_or_compute("user", 300, old))
        await started.wait()
        cache.invalidate(lambda key: key == "user")
        assert await cache.get_or_compute("user", 300, new) == "new settings"
        finish.set()
        await old_caller
        assert cache.peek("user") == "new settings"

    asyncio.run(scenario())


def test_clear_prevents_late_publication():
    async def scenario():
        cache = SWRCache()
        started, finish = asyncio.Event(), asyncio.Event()

        async def compute():
            started.set()
            await finish.wait()
            return "obsolete"

        caller = asyncio.create_task(cache.get_or_compute("k", 300, compute))
        await started.wait()
        cache.clear()
        finish.set()
        await caller
        assert cache.peek("k") is None

    asyncio.run(scenario())


def test_cancelled_waiter_does_not_cancel_shared_calculation():
    async def scenario():
        cache = SWRCache()
        started, finish = asyncio.Event(), asyncio.Event()

        async def compute():
            started.set()
            await finish.wait()
            return "result"

        first = asyncio.create_task(cache.get_or_compute("k", 300, compute))
        await started.wait()
        second = asyncio.create_task(cache.get_or_compute("k", 300, compute))
        first.cancel()
        try:
            await first
        except asyncio.CancelledError:
            pass
        finish.set()
        assert await second == "result"

    asyncio.run(scenario())


def test_user_invalidation_keeps_other_users_results(monkeypatch):
    from server import portfolio_service as service

    monkeypatch.setattr(service, "resolve_user_db_path", lambda user: f"/{user}.db")
    service._RAW_CALC_CACHE.put(("USD", "ALL", "/alice.db", 0), "alice")
    service._RAW_CALC_CACHE.put(("USD", "ALL", "/bob.db", 0), "bob")
    service.clear_portfolio_caches("alice")
    assert service._RAW_CALC_CACHE.peek(("USD", "ALL", "/alice.db", 0)) is None
    assert service._RAW_CALC_CACHE.peek(("USD", "ALL", "/bob.db", 0)) == "bob"
    service.clear_portfolio_caches()


def test_summary_does_not_attach_new_freshness_to_old_metrics(monkeypatch):
    import time
    import pandas as pd
    from server import portfolio_service as service

    service.clear_portfolio_caches()
    monkeypatch.setattr(service, "is_market_open", lambda: False)
    data = (
        pd.DataFrame({"Date": pd.to_datetime(["2025-01-01"]), "Account": ["A"]}),
        {},
        {},
        set(),
        {},
        {},
        "/freshness.db",
        0,
    )
    revision = service._CACHE_REVISIONS.setdefault(data[6], 0)
    raw_key = (
        "USD",
        "ALL",
        data[6],
        0,
        service.configuration_key(data[1:6], {}, {}, revision),
    )
    summary_key = (
        "USD",
        "ALL",
        data[6],
        0,
        int(time.time() / 300),
        service.configuration_key(data[1:6], {}, {}, {}, revision),
    )
    service._PORTFOLIO_SUMMARY_CACHE[summary_key] = {
        "metrics": {"market_value": 1},
        "freshness": {"calculated_at": 1},
    }
    service._RAW_CALC_CACHE.put(raw_key, "fresh result")

    async def raw(**_):
        return {"market_value": 2}, pd.DataFrame(), {}, {}

    async def history(**_):
        return pd.DataFrame(), {}, {}, ""

    monkeypatch.setattr(service, "_compute_raw_summary", raw)
    monkeypatch.setattr(service, "_get_historical_performance_cached", history)
    result = asyncio.run(service._calculate_portfolio_summary_internal(data=data))
    assert result["metrics"]["market_value"] == 2
    assert result["freshness"]["calculated_at"] > 1
    service.clear_portfolio_caches()
