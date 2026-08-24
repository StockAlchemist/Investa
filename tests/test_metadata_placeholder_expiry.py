"""A failed metadata fetch must not classify a holding as "Unknown" for 30 days.

When yfinance returns nothing for a symbol, `_ensure_metadata_batch` persists a
placeholder entry (every field None) so a dead ticker can't stampede the API on
each request. That entry has all the keys the cache-validity check looks for, so
before this guard it was served for the full METADATA_CACHE_DURATION_DAYS — and
every holding it covered fell into the allocation views' "Unknown" bucket, which
the Rebalance Helper then priced trades against.

These tests pin the rules:

1. A placeholder older than METADATA_PLACEHOLDER_RETRY_HOURS is re-fetched.
2. A fresh placeholder is still honoured (no stampede on a dead ticker).
3. A fund with a real quoteType but no sector/country is NOT a placeholder — it
   keeps its full 30-day cache life.
4. Repeated failures back off, so a permanently dead ticker settles back to the
   old monthly cadence instead of retrying every few hours forever.
5. The background refresh worker picks up expired placeholders too, and does not
   reset their failure count.
"""

import json
import os
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

import config
import market_data
from market_data import (
    MarketDataProvider,
    is_unclassified_metadata,
    placeholder_retry_delay,
)
from server.refresh_worker import _find_stale_symbols, _refresh_batch_sync


def _entry(age_hours, **fields):
    """A v4 cache entry stamped `age_hours` in the past."""
    ts = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    base = {
        "name": "X",
        "currency": None,
        "sector": None,
        "industry": None,
        "country": None,
        "exchange": None,
        "fullExchangeName": None,
        "exchangeTimezoneName": None,
        "quoteType": None,
        "timestamp": ts.isoformat(),
        "schema_version": config.METADATA_SCHEMA_VERSION,
    }
    base.update(fields)
    return base


REAL = {
    "currency": "USD",
    "sector": "Technology",
    "industry": "Semiconductors",
    "country": "United States",
    "exchange": "NMS",
    "quoteType": "EQUITY",
}
# A fund Yahoo answers for, but with no sector or country of its own.
FUND = {"currency": "USD", "quoteType": "MUTUALFUND"}


# --- 1. the classifier itself -------------------------------------------------


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, True),  # the failed-fetch placeholder
        (REAL, False),  # a real equity
        (FUND, False),  # a fund with no sector/country
        ({"sector": "Technology"}, False),  # partially enriched (e.g. by FMP)
        ({"quoteType": ""}, True),  # empty string is not a value
    ],
)
def test_is_unclassified_metadata(fields, expected):
    assert is_unclassified_metadata(_entry(1, **fields)) is expected


def test_missing_entry_is_unclassified():
    assert is_unclassified_metadata(None) is True
    assert is_unclassified_metadata({}) is True


# --- 2. the cache-validity check ---------------------------------------------


def _batch(tmp_path, entries):
    """Run _ensure_metadata_batch over a cache dir seeded with `entries`."""
    for sym, entry in entries.items():
        with open(os.path.join(tmp_path, f"{sym}.json"), "w") as f:
            json.dump(entry, f)

    mdp = MarketDataProvider.__new__(MarketDataProvider)
    mdp._get_symbol_metadata_path = lambda s: os.path.join(tmp_path, f"{s}.json")

    # No network: a re-fetch shows up as the symbol reaching the fetch stage.
    with (
        patch.object(market_data, "_ensure_yfinance"),
        patch.object(market_data, "YFINANCE_AVAILABLE", False),
    ):
        served = mdp._ensure_metadata_batch(set(entries))
    return served


def test_stale_placeholder_is_refetched(tmp_path):
    age = config.METADATA_PLACEHOLDER_RETRY_HOURS + 1
    served = _batch(tmp_path, {"GOOG": _entry(age)})
    # Not served from cache — it fell through to the (unavailable) fetcher.
    assert "GOOG" not in served


def test_fresh_placeholder_is_honoured(tmp_path):
    age = max(config.METADATA_PLACEHOLDER_RETRY_HOURS - 1, 0)
    served = _batch(tmp_path, {"DEADTICKER": _entry(age)})
    assert "DEADTICKER" in served


def test_real_entry_keeps_full_cache_life(tmp_path):
    old_but_valid = config.METADATA_CACHE_DURATION_DAYS * 24 - 24
    served = _batch(tmp_path, {"NVDA": _entry(old_but_valid, **REAL)})
    assert served["NVDA"]["sector"] == "Technology"


def test_fund_without_sector_is_not_retried(tmp_path):
    """A fund's missing sector is Yahoo's answer, not a failed fetch."""
    old_but_valid = config.METADATA_CACHE_DURATION_DAYS * 24 - 24
    served = _batch(tmp_path, {"AAUK": _entry(old_but_valid, **FUND)})
    assert served["AAUK"]["quoteType"] == "MUTUALFUND"


# --- 3. the background worker -------------------------------------------------


def test_worker_repairs_expired_placeholders(tmp_path):
    entries = {
        "GOOG": _entry(config.METADATA_PLACEHOLDER_RETRY_HOURS + 1),  # expired
        "PLTR": _entry(0),  # fresh
        "NVDA": _entry(1, **REAL),  # healthy
        "AAUK": _entry(1, **FUND),  # healthy fund
    }
    for sym, entry in entries.items():
        with open(os.path.join(tmp_path, f"{sym}.json"), "w") as f:
            json.dump(entry, f)

    stale = _find_stale_symbols(str(tmp_path), config.METADATA_SCHEMA_VERSION, limit=50)
    assert stale == ["GOOG"]


# --- 4. backoff ---------------------------------------------------------------


def test_retry_delay_doubles_per_failure():
    base = timedelta(hours=config.METADATA_PLACEHOLDER_RETRY_HOURS)
    assert placeholder_retry_delay(1) == base
    assert placeholder_retry_delay(2) == base * 2
    assert placeholder_retry_delay(3) == base * 4


def test_retry_delay_is_capped_at_the_normal_cache_life():
    ceiling = timedelta(days=config.METADATA_CACHE_DURATION_DAYS)
    assert placeholder_retry_delay(50) == ceiling
    # A corrupt counter must not overflow the shift.
    assert placeholder_retry_delay(10**6) == ceiling


def test_repeated_failure_is_not_retried_on_the_first_window(tmp_path):
    """Attempt 3 waits 4x the base window, so the base window alone isn't enough."""
    age = config.METADATA_PLACEHOLDER_RETRY_HOURS + 1
    served = _batch(tmp_path, {"APPL": _entry(age, placeholder_attempts=3)})
    assert "APPL" in served  # still honoured — not due yet


def test_repeated_failure_is_retried_once_its_window_passes(tmp_path):
    age = config.METADATA_PLACEHOLDER_RETRY_HOURS * 4 + 1
    served = _batch(tmp_path, {"APPL": _entry(age, placeholder_attempts=3)})
    assert "APPL" not in served


def test_failure_count_survives_the_worker_refresh(tmp_path, monkeypatch):
    """The worker must not delete a placeholder — that would reset its backoff."""
    entry = _entry(
        config.METADATA_PLACEHOLDER_RETRY_HOURS * 2 + 1, placeholder_attempts=2
    )

    monkeypatch.setattr(config, "get_app_data_dir", lambda: str(tmp_path.parent))
    # _refresh_batch_sync resolves the cache dir as <app_data>/cache/metadata_cache.
    cache_dir = tmp_path.parent / "cache" / "metadata_cache"
    os.makedirs(cache_dir, exist_ok=True)
    with open(cache_dir / "APPL.json", "w") as f:
        json.dump(entry, f)

    with patch("market_data.get_shared_mdp") as mdp:
        mdp.return_value._ensure_metadata_batch.return_value = {}
        _refresh_batch_sync(["APPL"])

    assert (cache_dir / "APPL.json").exists()
    with open(cache_dir / "APPL.json") as f:
        assert json.load(f)["placeholder_attempts"] == 2


def test_consecutive_failures_increment_the_counter(tmp_path):
    """A retry that fails again must lengthen the backoff, not restart it."""
    sym = "APPL"
    path = tmp_path / f"{sym}.json"
    with open(path, "w") as f:
        json.dump(
            _entry(
                config.METADATA_PLACEHOLDER_RETRY_HOURS * 2 + 1,
                placeholder_attempts=2,
            ),
            f,
        )

    mdp = MarketDataProvider.__new__(MarketDataProvider)
    mdp._get_symbol_metadata_path = lambda s: str(tmp_path / f"{s}.json")

    with (
        patch.object(market_data, "_ensure_yfinance"),
        patch.object(market_data, "YFINANCE_AVAILABLE", True),
        patch.object(market_data, "_run_isolated_fetch", return_value={}),
        patch.object(market_data, "_maybe_enrich_with_fmp"),
    ):
        mdp._ensure_metadata_batch({sym})

    with open(path) as f:
        written = json.load(f)
    assert is_unclassified_metadata(written)
    assert written["placeholder_attempts"] == 3


def test_first_failure_starts_the_counter_at_one(tmp_path):
    sym = "NEVERSEEN"
    mdp = MarketDataProvider.__new__(MarketDataProvider)
    mdp._get_symbol_metadata_path = lambda s: str(tmp_path / f"{s}.json")

    with (
        patch.object(market_data, "_ensure_yfinance"),
        patch.object(market_data, "YFINANCE_AVAILABLE", True),
        patch.object(market_data, "_run_isolated_fetch", return_value={}),
        patch.object(market_data, "_maybe_enrich_with_fmp"),
    ):
        mdp._ensure_metadata_batch({sym})

    with open(tmp_path / f"{sym}.json") as f:
        assert json.load(f)["placeholder_attempts"] == 1
