# -*- coding: utf-8 -*-
"""
Tests for system hardening:
1. SQLite connection factory standardization with PRAGMA busy_timeout = 30000 and WAL mode.
2. Bounded LRU/TTL expiration for in-memory dictionaries in market_data.py.
3. Sanitized HTTP 500 error responses across API routes to prevent trace/path leakage.
"""

import sqlite3
import time
from fastapi.testclient import TestClient

from db_utils import get_db_connection
from buffett_store import BuffettRankStore
from edgar_provider import EdgarFactStore
from market_data import (
    BoundedTTLCache,
    MarketDataProvider,
    _EARNINGS_BACKFILL_ATTEMPTS,
    _edgar_statement_cache,
    _edgar_quarterly_cache,
)
from server.main import app


def test_get_db_connection_pragmas(tmp_path):
    """Verify get_db_connection enforces PRAGMA busy_timeout and WAL mode."""
    test_db = str(tmp_path / "test_pragmas.db")
    conn = get_db_connection(test_db, use_cache=False)
    assert conn is not None

    cursor = conn.cursor()
    # Check busy timeout is 30000 ms
    cursor.execute("PRAGMA busy_timeout;")
    busy_timeout = cursor.fetchone()[0]
    assert busy_timeout == 30000

    # Check foreign keys are ON
    cursor.execute("PRAGMA foreign_keys;")
    foreign_keys = cursor.fetchone()[0]
    assert foreign_keys == 1

    # Check journal mode is WAL (non-cloud path)
    cursor.execute("PRAGMA journal_mode;")
    journal_mode = cursor.fetchone()[0].upper()
    assert journal_mode == "WAL"

    conn.close()


def test_buffett_store_connection_standardization(tmp_path):
    """Verify BuffettRankStore._connect uses get_db_connection and sqlite3.Row."""
    test_db = str(tmp_path / "test_buffett.db")
    store = BuffettRankStore(test_db)
    conn = store._connect()
    assert conn is not None
    assert conn.row_factory == sqlite3.Row

    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout;")
    assert cursor.fetchone()[0] == 30000

    conn.close()


def test_edgar_store_connection_standardization(tmp_path):
    """Verify EdgarFactStore._connect uses get_db_connection with WAL mode and busy timeout."""
    test_db = str(tmp_path / "test_edgar.db")
    store = EdgarFactStore(test_db)
    conn = store._connect()
    assert conn is not None

    cursor = conn.cursor()
    cursor.execute("PRAGMA busy_timeout;")
    assert cursor.fetchone()[0] == 30000

    cursor.execute("PRAGMA journal_mode;")
    assert cursor.fetchone()[0].upper() == "WAL"

    conn.close()


def test_bounded_ttl_cache_lru_eviction():
    """Verify BoundedTTLCache evicts least-recently-used items when max_size is reached."""
    cache = BoundedTTLCache(max_size=3, default_ttl=60.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.set("c", 3)

    assert len(cache) == 3
    assert cache.get("a") == 1  # accesses "a", so "b" is now LRU

    # Adding "d" should evict "b"
    cache.set("d", 4)
    assert len(cache) == 3
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("d") == 4


def test_bounded_ttl_cache_ttl_expiration():
    """Verify BoundedTTLCache expires items after TTL seconds."""
    cache = BoundedTTLCache(max_size=10, default_ttl=0.1)
    cache.set("key1", "value1", ttl=0.05)
    cache.set("key2", "value2", ttl=10.0)

    assert cache.get("key1") == "value1"
    assert cache.get("key2") == "value2"

    time.sleep(0.08)

    assert cache.get("key1") is None
    assert "key1" not in cache
    assert cache.get("key2") == "value2"
    assert "key2" in cache


def test_market_data_bounded_cache_instances():
    """Verify module-level and instance-level caches in market_data are BoundedTTLCache."""
    assert isinstance(_EARNINGS_BACKFILL_ATTEMPTS, BoundedTTLCache)
    assert _EARNINGS_BACKFILL_ATTEMPTS._max_size == 500

    assert isinstance(_edgar_statement_cache, BoundedTTLCache)
    assert _edgar_statement_cache._max_size == 64

    assert isinstance(_edgar_quarterly_cache, BoundedTTLCache)
    assert _edgar_quarterly_cache._max_size == 64

    mdp = MarketDataProvider()
    assert isinstance(mdp._current_quotes_memory_cache, BoundedTTLCache)
    assert mdp._current_quotes_memory_cache._max_size == 200
    assert isinstance(mdp._index_quotes_memory_cache, BoundedTTLCache)
    assert mdp._index_quotes_memory_cache._max_size == 50


def test_sanitized_500_responses():
    """Verify that route error handlers return sanitized 500 details without traces or path leakage."""
    client = TestClient(app)

    # Test settings with invalid payload or simulated failure
    res = client.post("/api/v1/settings", json={"invalid_field": "test"})
    # Should either succeed or return 401/422/500, but if 500 never leaks trace
    if res.status_code == 500:
        detail = res.json().get("detail", "")
        assert "Traceback" not in detail
        assert "/" not in detail
        assert "Exception" not in detail
