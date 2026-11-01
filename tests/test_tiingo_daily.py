"""Yahoo-first daily fallback, split basis, archive provenance, and API budgets."""

from datetime import date
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pandas as pd
import pytest

import config
import market_data
import tiingo_daily as daily
from market_db import BASIS_RAW, MarketDatabase
from tiingo_provider import TiingoError

START = date(2020, 8, 27)
END = date(2020, 8, 28)
TODAY = date(2020, 9, 2)


def rows():
    # adjClose deliberately disagrees: dividends must not enter price returns.
    return [
        {
            "date": "2020-08-27T00:00:00Z",
            "close": 500,
            "open": 496,
            "high": 504,
            "low": 492,
            "volume": 100,
            "splitFactor": 1,
            "adjClose": 90,
        },
        {"date": "2020-08-28T00:00:00Z", "close": 496, "splitFactor": 1},
        {"date": "2020-08-31T00:00:00Z", "close": 129, "splitFactor": 4, "divCash": 1},
    ]


@pytest.fixture
def fallback(monkeypatch):
    monkeypatch.setattr(config, "TIINGO_API_KEY", "test-only-key")
    monkeypatch.setattr(daily, "get_est_today", lambda: TODAY)
    fetch = Mock(return_value=rows())
    monkeypatch.setattr(daily.TiingoProvider, "fetch_prices", fetch)
    instance = daily.TiingoDailyFallback()
    monkeypatch.setattr(daily, "daily_fallback", instance)
    return instance, fetch


def test_window_includes_adjustments_from_later_splits_and_caches(fallback):
    instance, fetch = fallback
    frame = instance.fetch(["AAPL"], START, END)["AAPL"]
    assert frame["price"].tolist() == [125, 124]
    assert frame.iloc[0]["Open"] == 124
    assert frame.iloc[0]["High"] == 126
    assert frame.iloc[0]["Low"] == 123
    assert frame.iloc[0]["Volume"] == 400
    assert frame.attrs["market_data_source"] == "tiingo"
    assert "Adj Close" not in frame
    fetch.assert_called_once_with("AAPL", start=date(1900, 1, 1), end=TODAY)
    frame.iloc[0, frame.columns.get_loc("price")] = -1
    # Different windows share the full series, without exposing mutable cache data.
    later = instance.fetch(["AAPL"], START, TODAY)["AAPL"]
    assert later["price"].tolist() == [125, 124, 129]
    assert fetch.call_count == 1


def test_concurrent_requests_share_one_fetch(fallback):
    instance, fetch = fallback
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(
            pool.map(lambda _: instance.fetch(["AAPL"], START, END), range(4))
        )
    assert all("AAPL" in result for result in results)
    assert fetch.call_count == 1


@pytest.mark.parametrize(
    "symbol", ["PTT.BK", "7203.T", "^GSPC", "THB=X", "BTC-USD", "GC=F", "", "AAPL/.."]
)
def test_excludes_foreign_indices_fx_and_crypto(symbol, fallback):
    instance, fetch = fallback
    assert instance.fetch([symbol], START, END) == {}
    fetch.assert_not_called()


def test_us_share_class_is_eligible(fallback):
    instance, fetch = fallback
    assert "BRK-B" in instance.fetch(["BRK-B"], START, END)
    assert fetch.call_args.args == ("BRK-B",)


def test_unconfigured_does_not_call_provider(fallback, monkeypatch):
    instance, fetch = fallback
    monkeypatch.setattr(config, "TIINGO_API_KEY", None)
    assert instance.fetch(["AAPL"], START, END) == {}
    fetch.assert_not_called()


@pytest.mark.parametrize("status", [401, 403, 429, 500, None])
def test_outage_cools_down_all_symbols_and_key_change_resets(
    status, fallback, monkeypatch, caplog
):
    instance, fetch = fallback
    fetch.side_effect = TiingoError("secret response body", status=status)
    assert instance.fetch(["AAPL", "MSFT"], START, END) == {}
    assert instance.fetch(["GOOG"], START, END) == {}
    assert fetch.call_count == 1
    assert "secret response body" not in caplog.text
    monkeypatch.setattr(config, "TIINGO_API_KEY", "replacement-test-key")
    fetch.side_effect = None
    assert "AAPL" in instance.fetch(["AAPL"], START, END)
    assert fetch.call_count == 2


def test_unknown_symbol_is_cached_without_blocking_others(fallback):
    instance, fetch = fallback
    fetch.side_effect = [TiingoError("not found", status=404), rows()]
    assert list(instance.fetch(["UNKNOWN", "AAPL"], START, END)) == ["AAPL"]
    assert instance.fetch(["UNKNOWN"], START, END) == {}
    assert fetch.call_count == 2


def test_cache_expires_and_empty_response_is_cached(fallback, monkeypatch):
    instance, fetch = fallback
    clock = [0]
    monkeypatch.setattr(daily.time, "monotonic", lambda: clock[0])
    fetch.return_value = []
    assert instance.fetch(["AAPL"], START, END) == {}
    assert instance.fetch(["AAPL"], START, END) == {}
    assert fetch.call_count == 1
    clock[0] = 901
    fetch.return_value = rows()
    assert "AAPL" in instance.fetch(["AAPL"], START, END)
    assert fetch.call_count == 2


@pytest.mark.parametrize("ratio", [0, -1, float("inf"), "garbage"])
def test_invalid_split_rejects_entire_series(ratio, fallback):
    instance, fetch = fallback
    broken = rows()
    broken[-1]["splitFactor"] = ratio
    fetch.return_value = broken
    assert instance.fetch(["AAPL"], START, END) == {}


def test_action_with_missing_close_still_adjusts_earlier_bars():
    data = rows()
    data[-1]["close"] = None
    assert daily.price_frame(data)["price"].tolist() == [125, 124]


@pytest.fixture
def provider(monkeypatch, fallback):
    provider = market_data.MarketDataProvider.__new__(market_data.MarketDataProvider)
    monkeypatch.setattr(provider, "_load_invalid_symbols_cache", lambda: {})
    monkeypatch.setattr(provider, "_save_invalid_symbols_cache", lambda _: None)
    monkeypatch.setattr(market_data, "_ensure_yfinance", lambda: None)
    monkeypatch.setattr(market_data, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(market_data, "is_market_open", lambda: True)
    monkeypatch.setattr(market_data, "get_est_today", lambda: TODAY)
    monkeypatch.setattr(
        market_data, "nyse_schedule", lambda *args: pd.DataFrame({"session": [1]})
    )
    monkeypatch.setattr(market_data.time, "sleep", lambda _: None)
    return provider


def test_yahoo_success_is_preserved_while_missing_series_uses_tiingo(
    provider, fallback, monkeypatch
):
    _, fetch = fallback
    yahoo = pd.DataFrame({("MSFT", "Close"): [200]}, index=pd.to_datetime([START]))
    monkeypatch.setattr(market_data, "_run_isolated_fetch", lambda **kwargs: yahoo)
    result = provider._fetch_yf_historical_data(["MSFT", "AAPL"], START, END)
    assert result["MSFT"]["price"].tolist() == [200]
    assert result["AAPL"]["price"].tolist() == [125, 124]
    assert fetch.call_count == 1
    assert fetch.call_args.args == ("AAPL",)


def test_invalid_yahoo_cache_does_not_block_tiingo(provider, monkeypatch):
    monkeypatch.setattr(
        provider,
        "_load_invalid_symbols_cache",
        lambda: {"AAPL": market_data.time.time()},
    )
    monkeypatch.setattr(
        market_data,
        "_run_isolated_fetch",
        lambda **_: pytest.fail("Yahoo should be skipped"),
    )
    assert "AAPL" in provider._fetch_yf_historical_data(["AAPL"], START, END)


def test_no_yahoo_installation_can_still_fetch_daily(provider, monkeypatch, fallback):
    _, fetch = fallback
    monkeypatch.setattr(market_data, "YFINANCE_AVAILABLE", False)
    assert "AAPL" in provider._fetch_yf_historical_data(["AAPL"], START, END)
    assert provider._fetch_yf_historical_data(["AAPL"], START, END, interval="5m") == {}
    assert fetch.call_count == 1


@pytest.mark.parametrize("raw", [False, True])
def test_archive_roundtrip_preserves_split_basis_and_source(provider, tmp_path, raw):
    provider.db = MarketDatabase(str(tmp_path / "archive.db"))
    # Apply migration schema only on this isolated test DB.
    with provider.db._get_connection() as conn:
        conn.execute(
            "ALTER TABLE sync_metadata ADD COLUMN price_basis TEXT NOT NULL DEFAULT 'split_adj'"
        )
        conn.execute(
            "CREATE TABLE corporate_action (symbol TEXT, date TEXT, kind TEXT, value REAL, currency TEXT, source TEXT, ingested_at TEXT, PRIMARY KEY(symbol, date, kind))"
        )
        conn.execute(
            "INSERT INTO corporate_action VALUES ('AAPL', '2020-08-31', 'split', 4, NULL, 'yahoo', 'now')"
        )
        conn.commit()
    if raw:
        provider.db.set_price_basis("AAPL", BASIS_RAW)
    frame = daily.price_frame(rows())
    provider._store_daily_history("AAPL", frame, "1d")
    stored = provider.db.get_ohlcv("AAPL", START, TODAY)
    assert stored["Close"].tolist() == [125, 124, 129]
    with provider.db._get_connection() as conn:
        assert {r[0] for r in conn.execute("SELECT source FROM daily_ohlcv")} == {
            "tiingo"
        }
        assert conn.execute("SELECT COUNT(*) FROM corporate_action").fetchone()[0] == 1
        close = conn.execute(
            "SELECT close FROM daily_ohlcv ORDER BY date LIMIT 1"
        ).fetchone()[0]
        assert close == (500 if raw else 125)


def test_sync_persists_tiingo_provenance(provider, monkeypatch):
    frame = daily.price_frame(rows())
    provider.db = Mock()
    provider.db.get_sync_metadata_batch.return_value = {}
    provider.db.get_last_dates.return_value = {}
    provider.db.get_first_dates.return_value = {}
    provider.db.check_integrity.return_value = (True, None)
    provider.db.get_price_basis.return_value = {"AAPL": "split_adj"}
    provider.db._get_connection.return_value.__enter__ = Mock(return_value=Mock())
    provider.db._get_connection.return_value.__exit__ = Mock(return_value=False)
    monkeypatch.setattr(
        provider, "_fetch_yf_historical_data", lambda *args, **kwargs: {"AAPL": frame}
    )
    monkeypatch.setattr(
        market_data.sql_store, "touch_sync_metadata", lambda *args: None
    )
    provider._sync_to_db(["AAPL"], START, TODAY)
    assert provider.db.upsert_ohlcv.call_args.kwargs["source"] == "tiingo"
