"""Helpers shared across route modules (kept dependency-light to avoid import cycles)."""

import asyncio
import logging
import time
from collections import OrderedDict

import numpy as np

# Bounded LRU caches: oldest entries are evicted beyond this size.
_CACHE_MAX_SIZE = 20


def _lru_get(cache: OrderedDict, key):
    """Return cached value and promote to MRU position, or return sentinel."""
    if key not in cache:
        return None
    cache.move_to_end(key)
    return cache[key]


def _lru_put(cache: OrderedDict, key, value, max_size: int = _CACHE_MAX_SIZE):
    """Insert/update a cache entry, evicting the LRU entry when over capacity."""
    if key in cache:
        cache.move_to_end(key)
    cache[key] = value
    while len(cache) > max_size:
        cache.popitem(last=False)  # Remove LRU (oldest-accessed) entry


class SWRCache:
    """Stale-while-revalidate cache for expensive async computations.

    Keys must NOT embed a time bucket — freshness is tracked per entry via a
    stored timestamp. A read returns:

      * the value immediately if a (fresh or stale) entry exists;
      * on a *stale* hit (age >= ``ttl``), a background refresh is kicked off so
        the next read sees fresh data, but the current caller never blocks;
      * on a *cold* miss, the computation runs and the caller awaits it.

    Concurrent callers for the same key coalesce onto a single in-flight task,
    eliminating the thundering-herd recompute that a hard cache expiry causes.
    This is what stops window rollovers from blocking requests for 11–50s: the
    previous result is served instantly while fresh numbers compute out of band.
    """

    def __init__(self, max_size: int = _CACHE_MAX_SIZE):
        self._store: OrderedDict = OrderedDict()  # key -> (value, computed_at)
        self._inflight: dict = {}                 # key -> asyncio.Task
        self._max_size = max_size

    def put(self, key, value):
        """Store ``value`` for ``key`` with the current time as its freshness stamp."""
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (value, time.time())
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    def peek(self, key):
        """Return the stored value without a freshness check, or ``None``."""
        entry = self._store.get(key)
        if entry is None:
            return None
        self._store.move_to_end(key)
        return entry[0]

    def clear(self):
        """Drop all stored entries (in-flight refreshes are left to complete)."""
        self._store.clear()

    def invalidate(self, predicate):
        """Drop stored entries whose key matches ``predicate(key)``."""
        for k in [k for k in self._store if predicate(k)]:
            del self._store[k]

    async def get_or_compute(self, key, ttl: float, compute):
        """Return the cached value for ``key``, refreshing per stale-while-revalidate.

        ``compute`` is a zero-arg coroutine function that produces the value.
        """
        entry = self._store.get(key)
        if entry is not None:
            self._store.move_to_end(key)
            value, computed_at = entry
            if time.time() - computed_at < ttl:
                return value
            # Stale: serve the previous value now, refresh out of band.
            self._ensure_task(key, compute, background=True)
            return value
        # Cold miss: compute and await (coalescing concurrent callers).
        return await self._ensure_task(key, compute, background=False)

    def _ensure_task(self, key, compute, background: bool):
        task = self._inflight.get(key)
        if task is not None and not task.done():
            return task
        task = asyncio.get_running_loop().create_task(self._run(key, compute))
        self._inflight[key] = task
        if background:
            # Nobody awaits a background refresh — retrieve its result in a
            # callback so a failure is logged instead of warned as un-retrieved.
            task.add_done_callback(self._on_background_done)
        return task

    async def _run(self, key, compute):
        try:
            value = await compute()
            self.put(key, value)
            return value
        finally:
            if self._inflight.get(key) is asyncio.current_task():
                self._inflight.pop(key, None)

    @staticmethod
    def _on_background_done(task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logging.warning(f"SWR background refresh failed: {exc}")


def get_mdp():
    """Shared Market Data Provider (one instance per process: shared DB connections + cache)."""
    from market_data import get_shared_mdp
    return get_shared_mdp()


def clean_nans(obj):
    """Recursively replace NaN/Infinity with None for JSON serialization."""
    # bool is a subclass of int — check it first so True/False stay JSON booleans
    # instead of being coerced to 1/0 (which strict clients like Swift reject).
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, (int, np.integer)):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: clean_nans(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_nans(v) for v in obj]
    return obj
