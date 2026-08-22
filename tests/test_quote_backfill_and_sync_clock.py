"""Cached-quote backfill bounds, currency inference, and sync-throttle clocks.

Three rules are pinned here, each protecting a way the portfolio can be valued
wrong without anything looking broken:

1. A quote missing from a live fetch may be backfilled from the persistent
   cache, but only while it still describes a recent session. Copying an entry
   forward never refreshes the timestamp it was fetched with, so age stays
   honest however many times the cache is rewritten.
2. When Yahoo metadata is unavailable, a symbol's currency is inferred from its
   exchange suffix — and refused outright when the suffix is unknown. Valuing a
   foreign price as dollars is worse than having no quote.
3. Sync throttling measures a real elapsed duration. Rows written before the
   market-clock migration carry a naive, server-local timestamp; reading those
   as Eastern makes a symbol look ~11 hours fresher than it is and it never
   resyncs.
"""

import json
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import market_data
from market_data import (
    MarketDataProvider,
    _fallback_currency,
    _fallback_market_timezone,
    _quote_entry_age,
)

CACHED_PRICE = 271.55


def _seed_cache(path, entries, cache_key="A_KEY_THAT_WILL_NOT_MATCH"):
    """Write a quotes cache whose key cannot satisfy the fast-path read."""
    path.write_text(
        json.dumps(
            {
                "cache_key": cache_key,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "quotes": entries,
                "fx_rates": {"USD": 1.0},
                "fx_prev_close": {"USD": 1.0},
            }
        )
    )


def _entry(price=CACHED_PRICE, age=timedelta(hours=1)):
    return {
        "price": price,
        "change": 1.0,
        "changesPercentage": 0.4,
        "currency": "USD",
        "name": "Apple Inc.",
        "source": "yf_batch_download",
        "timestamp": (datetime.now(timezone.utc) - age).isoformat(),
    }


@pytest.fixture
def cache_file(tmp_path):
    return tmp_path / "quotes.json"


@pytest.fixture
def provider(cache_file):
    mdp = MarketDataProvider(current_cache_file=str(cache_file))
    with (
        patch.object(
            MarketDataProvider,
            "_ensure_metadata_batch",
            return_value={"AAPL": {"currency": "USD", "name": "Apple Inc."}},
        ),
        patch.object(MarketDataProvider, "get_fundamental_data_batch", return_value={}),
    ):
        yield mdp


def _quotes_with_failed_fetch(provider):
    """Run get_current_quotes with both fetch legs returning nothing."""
    with patch.object(market_data, "_run_isolated_fetch", return_value=pd.DataFrame()):
        quotes, _fx, _prev, err, _warn = provider.get_current_quotes(
            ["AAPL"], {"USD"}, {}, set()
        )
    assert not err
    return quotes


# --- 1. Backfill bounds -----------------------------------------------------


def test_recent_quote_is_backfilled_and_marked(provider, cache_file):
    """A fetch that misses a symbol falls back to its last known quote."""
    _seed_cache(cache_file, {"AAPL": _entry(age=timedelta(hours=1))})

    quote = _quotes_with_failed_fetch(provider)["AAPL"]

    assert quote["price"] == pytest.approx(CACHED_PRICE)
    # Marked, so a caller can tell this apart from a live quote.
    assert quote["source"] == "persistent_cache_backfill"
    assert quote["stale"] is True


def test_quote_past_the_age_cutoff_is_refused(provider, cache_file):
    """Beyond the cutoff the price no longer describes a recent session."""
    _seed_cache(cache_file, {"AAPL": _entry(age=timedelta(days=10))})

    assert "AAPL" not in _quotes_with_failed_fetch(provider)


def test_quote_without_a_timestamp_is_refused(provider, cache_file):
    """An unknown age is refused, never assumed to be recent."""
    undated = _entry()
    del undated["timestamp"]
    _seed_cache(cache_file, {"AAPL": undated})

    assert "AAPL" not in _quotes_with_failed_fetch(provider)


def test_backfilled_entry_does_not_launder_its_age(provider, cache_file):
    """Rewriting the cache must not reset the timestamp and buy another cycle."""
    original = _entry(age=timedelta(hours=1))
    _seed_cache(cache_file, {"AAPL": original})

    _quotes_with_failed_fetch(provider)

    written = json.loads(cache_file.read_text())["quotes"]["AAPL"]
    assert written["timestamp"] == original["timestamp"]


def test_over_age_quotes_are_pruned_from_the_written_cache(provider, cache_file):
    """Entries too old to ever be served are dropped rather than accumulated."""
    _seed_cache(
        cache_file,
        {
            "AAPL": _entry(age=timedelta(hours=1)),
            "ANCIENT": _entry(age=timedelta(days=30)),
        },
    )

    _quotes_with_failed_fetch(provider)

    written = json.loads(cache_file.read_text())["quotes"]
    assert "AAPL" in written
    assert "ANCIENT" not in written


def test_quote_entry_age_reads_a_naive_timestamp_as_utc():
    naive = (datetime.now(timezone.utc) - timedelta(hours=3)).replace(tzinfo=None)
    age = _quote_entry_age({"timestamp": naive.isoformat()}, datetime.now(timezone.utc))
    assert age is not None
    assert timedelta(hours=2, minutes=55) < age < timedelta(hours=3, minutes=5)


@pytest.mark.parametrize("entry", [{}, {"timestamp": "not a timestamp"}])
def test_quote_entry_age_is_unknown_for_unreadable_timestamps(entry):
    assert _quote_entry_age(entry, datetime.now(timezone.utc)) is None


# --- 2. Currency and timezone inference -------------------------------------


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("AAPL", "USD"),  # bare symbol: a US listing
        ("BRK-B", "USD"),  # share classes are normalized to dashes upstream
        ("PTT.BK", "THB"),
        ("1319.HK", "HKD"),
        ("7203.T", "JPY"),
        ("SHOP.TO", "CAD"),
        ("VOD.L", None),  # London quotes in pence; refuse rather than be 100x out
        ("FOO.ZZZ", None),  # exchange we cannot name
    ],
)
def test_fallback_currency(symbol, expected):
    assert _fallback_currency(symbol) == expected


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("AAPL", "America/New_York"),
        ("PTT.BK", "Asia/Bangkok"),
        ("1319.HK", "Asia/Hong_Kong"),
        ("FOO.ZZZ", "America/New_York"),
    ],
)
def test_fallback_market_timezone(symbol, expected):
    assert _fallback_market_timezone(symbol) == expected


def test_unknown_currency_is_never_persisted_as_a_guess(tmp_path):
    """A failed metadata fetch leaves currency unknown, not defaulted to USD.

    The entry is cached for METADATA_CACHE_DURATION_DAYS, so a guess written
    here would misvalue the position for a month.
    """
    mdp = MarketDataProvider(current_cache_file=str(tmp_path / "quotes.json"))
    with (
        patch.object(
            mdp,
            "_get_symbol_metadata_path",
            side_effect=lambda s: str(tmp_path / f"{s}.json"),
        ),
        patch.object(market_data, "_run_isolated_fetch", return_value={}),
        patch.object(market_data, "_maybe_enrich_with_fmp", return_value=None),
    ):
        meta = mdp._ensure_metadata_batch(["1319.HK"])

    assert meta["1319.HK"]["currency"] is None


# --- 3. Sync throttling clock -----------------------------------------------


def _run_sync(provider, last_synced_iso):
    """Drive _sync_to_db and report whether it actually fetched."""
    provider.db = MagicMock()
    provider.db.get_sync_metadata_batch.return_value = {"AAPL": last_synced_iso}
    provider.db.get_last_dates.return_value = {}
    provider.db.get_first_dates.return_value = {}

    with patch.object(
        MarketDataProvider, "_fetch_yf_historical_data", return_value={}
    ) as fetch:
        provider._sync_to_db(["AAPL"], date(2026, 1, 1), date(2026, 8, 22))
    return fetch.called


def test_legacy_naive_timestamp_is_read_on_the_server_clock(provider):
    """A naive stamp is server-local; five hours old is stale and must resync.

    Read as Eastern instead, the same stamp reports ~11 hours less elapsed than
    has actually passed, so the symbol silently stops refreshing.
    """
    five_hours_ago = (datetime.now().astimezone() - timedelta(hours=5)).replace(
        tzinfo=None
    )
    assert _run_sync(provider, five_hours_ago.isoformat()) is True


def test_recently_synced_symbol_is_still_throttled(provider):
    """The fix must not turn the throttle off — an hour old is fresh enough."""
    one_hour_ago = (datetime.now().astimezone() - timedelta(hours=1)).replace(
        tzinfo=None
    )
    with patch.object(market_data, "is_market_open", return_value=False):
        assert _run_sync(provider, one_hour_ago.isoformat()) is False


def test_timezone_aware_timestamp_is_compared_directly(provider):
    """Stamps written after the migration are aware and need no localization."""
    aware_five_hours_ago = datetime.now(timezone.utc) - timedelta(hours=5)
    assert _run_sync(provider, aware_five_hours_ago.isoformat()) is True
