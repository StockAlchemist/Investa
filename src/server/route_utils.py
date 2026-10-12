"""Helpers shared across route modules (kept dependency-light to avoid import cycles)."""

import asyncio
import logging
import time
import threading
from collections import OrderedDict

import numpy as np

# Bounded LRU caches: oldest entries are evicted beyond this size.
_CACHE_MAX_SIZE = 20
_LRU_LOCK = threading.RLock()


def _lru_get(cache: OrderedDict, key):
    """Return cached value and promote to MRU position, or return sentinel."""
    with _LRU_LOCK:
        if key not in cache:
            return None
        cache.move_to_end(key)
        return cache[key]


def _lru_put(cache: OrderedDict, key, value, max_size: int = _CACHE_MAX_SIZE):
    """Insert/update a cache entry, evicting the LRU entry when over capacity."""
    with _LRU_LOCK:
        if key in cache:
            cache.move_to_end(key)
        cache[key] = value
        while len(cache) > max_size:
            cache.popitem(last=False)


def _lru_invalidate(cache: OrderedDict, predicate=lambda _: True):
    with _LRU_LOCK:
        for key in list(cache):
            if predicate(key):
                del cache[key]


class ComputationPending(Exception):
    """A shared calculation is still running after the caller's wait budget."""


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
        self._inflight: dict = {}  # key -> asyncio.Task
        self._max_size = max_size
        self._lock = threading.RLock()

    def put(self, key, value):
        """Store ``value`` for ``key`` with the current time as its freshness stamp."""
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.time())
            while len(self._store) > self._max_size:
                self._store.popitem(last=False)

    def peek(self, key):
        """Return the stored value without a freshness check, or ``None``."""
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            self._store.move_to_end(key)
            return entry[0]

    def freshness(self, key):
        """Computation time and refresh state, rather than HTTP response time."""
        with self._lock:
            entry = self._store.get(key)
            return {
                "calculated_at": entry[1] if entry else None,
                "refreshing": key in self._inflight,
            }

    def clear(self):
        """Detach old computations; they may finish but cannot publish results."""
        with self._lock:
            self._store.clear()
            self._inflight.clear()

    def invalidate(self, predicate):
        """Drop stored entries whose key matches ``predicate(key)``."""
        with self._lock:
            for k in set(self._store) | set(self._inflight):
                if predicate(k):
                    self._store.pop(k, None)
                    self._inflight.pop(k, None)

    async def get_or_compute(
        self, key, ttl: float, compute, wait_timeout: float | None = None
    ):
        """Return the cached value for ``key``, refreshing per stale-while-revalidate.

        ``compute`` is a zero-arg coroutine function that produces the value.
        A wait budget bounds cold callers; the shared task continues after a
        ``ComputationPending`` response and publishes for subsequent readers.
        """
        with self._lock:
            entry = self._store.get(key)
            if entry is not None:
                self._store.move_to_end(key)
                value, computed_at = entry
                if time.time() - computed_at < ttl:
                    return value
                self._ensure_task(key, compute, background=True)
                return value
            task = self._ensure_task(key, compute, background=False)
        # Cancelling one HTTP caller must not cancel work shared by other callers.
        if wait_timeout is None:
            return await asyncio.shield(task)
        try:
            return await asyncio.wait_for(asyncio.shield(task), timeout=wait_timeout)
        except TimeoutError:
            # A timeout raised *by* the calculation remains an actual failure.
            if task.done():
                return task.result()
            raise ComputationPending from None

    def _ensure_task(self, key, compute, background: bool):
        task = self._inflight.get(key)
        if task is not None and not task.done():
            return task
        task = asyncio.get_running_loop().create_task(self._run(key, compute))
        self._inflight[key] = task
        # A shielded cold caller may disconnect too; always consume exceptions.
        task.add_done_callback(self._on_background_done)
        return task

    async def _run(self, key, compute):
        try:
            value = await compute()
            with self._lock:
                if self._inflight.get(key) is asyncio.current_task():
                    self.put(key, value)
            return value
        finally:
            with self._lock:
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
