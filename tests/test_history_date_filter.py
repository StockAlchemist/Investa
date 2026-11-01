"""Daily bars retain session dates; intraday bars use the NY day boundary."""

from datetime import date

import pandas as pd
import pytest

import market_data


@pytest.mark.parametrize("tz", [None, "UTC", "Asia/Bangkok", "America/New_York"])
@pytest.mark.parametrize("recovery", [False, True])
def test_daily_bar_keeps_its_exchange_date(tz, recovery, monkeypatch):
    frame = pd.DataFrame(
        {"Close": [10.0]}, index=pd.DatetimeIndex(["2025-01-06"], tz=tz)
    )
    result, calls = fetch_history(frame, "1d", recovery, monkeypatch)
    assert result["TEST.BK"]["price"].tolist() == [10.0]
    assert result["TEST.BK"].index.equals(frame.index)
    assert calls == (2 if recovery else 1)


@pytest.mark.parametrize("recovery", [False, True])
def test_intraday_bar_uses_ny_date_including_recovery(recovery, monkeypatch):
    # UTC Jan 7 is still Jan 6 in NY and belongs in the requested window.
    frame = pd.DataFrame(
        {"Close": [10.0, 11.0]},
        index=pd.DatetimeIndex(["2025-01-07 00:30", "2025-01-07 15:00"], tz="UTC"),
    )
    result, _ = fetch_history(frame, "5m", recovery, monkeypatch)
    assert result["TEST.BK"]["price"].tolist() == [10.0]


def fetch_history(frame, interval, recovery, monkeypatch):
    provider = market_data.MarketDataProvider.__new__(market_data.MarketDataProvider)
    monkeypatch.setattr(provider, "_load_invalid_symbols_cache", lambda: {})
    monkeypatch.setattr(
        provider,
        "_save_invalid_symbols_cache",
        lambda _: pytest.fail("valid symbol marked invalid"),
    )
    monkeypatch.setattr(market_data, "_ensure_yfinance", lambda: None)
    monkeypatch.setattr(market_data, "YFINANCE_AVAILABLE", True)
    monkeypatch.setattr(market_data, "is_market_open", lambda: True)
    monkeypatch.setattr(
        market_data, "nyse_schedule", lambda *args: pd.DataFrame({"session": [1]})
    )
    monkeypatch.setattr(
        market_data, "get_latest_trading_date", lambda: date(2025, 1, 7)
    )
    monkeypatch.setattr(market_data, "get_est_today", lambda: date(2025, 1, 7))
    # Keep this test independent of the intraday archive and the live database.
    provider.db = type("DB", (), {"upsert_intraday": lambda *args: None})()
    calls = []

    def download(**kwargs):
        calls.append(kwargs)
        if interval != "1d" and kwargs["interval"] == "1d":
            # The helper separately fetches daily opens for intraday anchoring.
            return pd.DataFrame(
                {"Close": [10.0]}, index=pd.DatetimeIndex(["2025-01-06"], tz="UTC")
            )
        if recovery and len(calls) == 1:
            # Nonempty batch with no data for TEST.BK triggers isolated recovery.
            return pd.DataFrame({("OTHER", "Close"): [1.0]}, index=frame.index[:1])
        return frame

    monkeypatch.setattr(market_data, "_run_isolated_fetch", download)
    result = provider._fetch_yf_historical_data(
        ["TEST.BK"], date(2025, 1, 6), date(2025, 1, 6), interval, use_cache=False
    )
    return result, len(calls)
