"""Helpers shared across route modules (kept dependency-light to avoid import cycles)."""

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


def get_mdp():
    """Shared Market Data Provider (one instance per process: shared DB connections + cache)."""
    from market_data import get_shared_mdp
    return get_shared_mdp()


def clean_nans(obj):
    """Recursively replace NaN/Infinity with None for JSON serialization."""
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
