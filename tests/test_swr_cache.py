"""Tests for the stale-while-revalidate cache in server.route_utils.

These pin the behaviour that makes a rolled-over cache window stop blocking
requests: a stale hit serves the previous value immediately while a fresh value
computes in the background, and concurrent callers coalesce onto one compute.
"""

import asyncio

import pytest

from server.route_utils import SWRCache


def test_cold_miss_computes_and_caches():
    cache = SWRCache()
    calls = []

    async def compute():
        calls.append(1)
        return "v1"

    async def scenario():
        first = await cache.get_or_compute("k", ttl=100, compute=compute)
        # A fresh second read must reuse the cached value, not recompute.
        second = await cache.get_or_compute("k", ttl=100, compute=compute)
        return first, second

    first, second = asyncio.run(scenario())
    assert first == "v1"
    assert second == "v1"
    assert len(calls) == 1


def test_stale_hit_serves_stale_then_refreshes_in_background():
    cache = SWRCache()
    values = iter(["v1", "v2"])

    async def compute():
        return next(values)

    async def scenario():
        # Prime the cache, then force the entry to look stale.
        await cache.get_or_compute("k", ttl=100, compute=compute)
        cache._store["k"] = (cache._store["k"][0], 0.0)  # computed_at far in the past

        # A stale read returns the OLD value immediately...
        served = await cache.get_or_compute("k", ttl=100, compute=compute)
        # ...and schedules a background refresh; let it run.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        refreshed = cache.peek("k")
        return served, refreshed

    served, refreshed = asyncio.run(scenario())
    assert served == "v1"          # caller never waited for the recompute
    assert refreshed == "v2"       # next reader will see fresh data


def test_concurrent_cold_callers_coalesce():
    cache = SWRCache()
    calls = []

    async def compute():
        calls.append(1)
        await asyncio.sleep(0.01)  # window for a second caller to pile on
        return "v1"

    async def scenario():
        return await asyncio.gather(
            cache.get_or_compute("k", ttl=100, compute=compute),
            cache.get_or_compute("k", ttl=100, compute=compute),
        )

    results = asyncio.run(scenario())
    assert results == ["v1", "v1"]
    assert len(calls) == 1  # both callers shared a single computation


def test_cold_miss_propagates_compute_error():
    cache = SWRCache()

    async def compute():
        raise ValueError("boom")

    async def scenario():
        with pytest.raises(ValueError, match="boom"):
            await cache.get_or_compute("k", ttl=100, compute=compute)
        # A failed compute must not leave a poisoned in-flight task behind.
        assert "k" not in cache._inflight

    asyncio.run(scenario())


def test_failed_background_refresh_keeps_serving_stale():
    cache = SWRCache()

    async def good():
        return "v1"

    async def bad():
        raise RuntimeError("refresh failed")

    async def scenario():
        await cache.get_or_compute("k", ttl=100, compute=good)
        cache._store["k"] = (cache._store["k"][0], 0.0)  # mark stale

        served = await cache.get_or_compute("k", ttl=100, compute=bad)
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        # Background refresh raised, but the stale value is still cached/served.
        return served, cache.peek("k")

    served, still_cached = asyncio.run(scenario())
    assert served == "v1"
    assert still_cached == "v1"
