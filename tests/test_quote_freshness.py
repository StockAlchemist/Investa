"""Quote freshness: the price must be the newest SESSION, not the newest UTC day.

Yahoo publishes the daily bar for a session well after that session closes, and
the 1m intraday frame is what carries the close in the meantime. These tests pin
the rules that keep the portfolio from being valued a full session behind:

1. An intraday bar from a LATER session than the daily bar wins, and the daily
   bar becomes its previous close.
2. A symbol's intraday price is read from the last minute it actually traded —
   1m frames covering several markets are NaN-padded across each other's hours.
3. An intraday bar OLDER than the daily bar is ignored.
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

import market_data
from market_data import MarketDataProvider

# Friday and Thursday closes; Monday's daily bar has not been published yet.
THU_CLOSE = 333.43
FRI_CLOSE = 308.91
# Monday's actual last trade, available only in the 1m frame.
MON_LAST = 303.27


def _daily_frame(rows):
    """MultiIndex (ticker, field) daily frame, shaped like yf.download."""
    idx = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame(
        {("AAPL", "Close"): [r[1] for r in rows]},
        index=idx,
    )


def _intraday_frame(stamps, closes, extra_symbol=None):
    """MultiIndex 1m frame, UTC-stamped like yfinance returns."""
    idx = pd.DatetimeIndex(stamps, tz="UTC", name="Datetime")
    data = {("AAPL", "Close"): closes}
    if extra_symbol:
        # The other market's rows: present in the index, NaN for this symbol.
        data[(extra_symbol, "Close")] = [float("nan")] * len(closes)
    return pd.DataFrame(data, index=idx)


@pytest.fixture
def provider(tmp_path):
    mdp = MarketDataProvider(current_cache_file=str(tmp_path / "quotes.json"))
    with (
        patch.object(
            MarketDataProvider,
            "_ensure_metadata_batch",
            return_value={"AAPL": {"currency": "USD", "name": "Apple Inc."}},
        ),
        patch.object(MarketDataProvider, "get_fundamental_data_batch", return_value={}),
    ):
        yield mdp


def _quotes(provider, daily_df, intraday_df):
    """Run get_current_quotes with both fetch legs stubbed."""

    def fake_fetch(symbols, period=None, interval=None, task=None, **kwargs):
        if interval == "1m":
            return intraday_df
        if any("=X" in s for s in symbols):  # FX leg
            return pd.DataFrame()
        return daily_df

    with patch.object(market_data, "_run_isolated_fetch", side_effect=fake_fetch):
        quotes, _fx, _fx_prev, err, _warn = provider.get_current_quotes(
            ["AAPL"], {"USD"}, {}, set()
        )
    assert not err
    return quotes


def test_intraday_beats_an_unpublished_daily_bar(provider):
    """Monday's 1m close wins; Friday's daily close becomes the previous close."""
    daily = _daily_frame(
        [
            (date(2026, 7, 30), THU_CLOSE),
            (date(2026, 7, 31), FRI_CLOSE),
            (date(2026, 8, 3), float("nan")),  # placeholder row Yahoo has not filled
        ]
    )
    intraday = _intraday_frame(
        ["2026-08-03 19:58", "2026-08-03 19:59"], [302.95, MON_LAST]
    )

    quote = _quotes(provider, daily, intraday)["AAPL"]

    assert quote["price"] == pytest.approx(MON_LAST)
    # Monday's move, measured against Friday — not Thursday, and not zero.
    assert quote["change"] == pytest.approx(MON_LAST - FRI_CLOSE)
    assert quote["changesPercentage"] == pytest.approx(
        (MON_LAST - FRI_CLOSE) / FRI_CLOSE * 100
    )


def test_intraday_read_from_the_last_minute_the_symbol_traded(provider):
    """A co-fetched market's later rows must not blank out this symbol's price."""
    daily = _daily_frame(
        [
            (date(2026, 7, 30), THU_CLOSE),
            (date(2026, 7, 31), FRI_CLOSE),
        ]
    )
    # The frame runs on past Monday's US close into Tuesday's Bangkok session,
    # where AAPL is NaN — the last ROW is not the last AAPL TRADE.
    intraday = _intraday_frame(
        ["2026-08-03 19:59", "2026-08-04 08:34", "2026-08-04 08:35"],
        [MON_LAST, float("nan"), float("nan")],
        extra_symbol="CPALL.BK",
    )

    quote = _quotes(provider, daily, intraday)["AAPL"]

    assert quote["price"] == pytest.approx(MON_LAST)
    assert quote["change"] == pytest.approx(MON_LAST - FRI_CLOSE)


def test_intraday_older_than_the_daily_bar_is_ignored(provider):
    """Monday's daily bar exists; a leftover Friday 1m frame must not overwrite it."""
    daily = _daily_frame(
        [
            (date(2026, 7, 31), FRI_CLOSE),
            (date(2026, 8, 3), MON_LAST),
        ]
    )
    intraday = _intraday_frame(["2026-07-31 19:59"], [FRI_CLOSE - 1.0])

    quote = _quotes(provider, daily, intraday)["AAPL"]

    assert quote["price"] == pytest.approx(MON_LAST)
    assert quote["change"] == pytest.approx(MON_LAST - FRI_CLOSE)


# --- Cache freshness across the opening bell -------------------------------
#
# A quote set cached while the market was shut must not survive into the open
# session. The freshness window depends on the market state at the moment the
# entry is USED, not the state when it was written — baking the closed-market
# window (an hour) into the entry's expiry froze the dashboard on the previous
# session's closes for up to an hour after the opening bell.

import json


def _age_caches(provider, seconds: float) -> None:
    """Rewind both quote caches by `seconds`, as if that long had passed."""
    with open(provider.current_cache_file) as f:
        data = json.load(f)
    stamp = pd.Timestamp(data["timestamp"]) - pd.Timedelta(seconds=seconds)
    data["timestamp"] = stamp.isoformat()
    with open(provider.current_cache_file, "w") as f:
        json.dump(data, f)

    mem = provider._current_quotes_memory_cache
    with mem._lock:
        for key, (value, expire_at) in list(mem._store.items()):
            # An entry that carries no computed-at stamp cannot be aged — that
            # shape is itself the defect these tests pin, so leave it alone and
            # let the behavioural assertion report it.
            if len(value) < 4:
                continue
            quotes, fx, fx_prev, computed_at = value
            mem._store[key] = ((quotes, fx, fx_prev, computed_at - seconds), expire_at)


def _counting_quotes(provider, daily_df, intraday_df, market_open: bool):
    """Run get_current_quotes and report how many price fetches it made."""
    calls = {"n": 0}

    def fake_fetch(symbols, period=None, interval=None, task=None, **kwargs):
        if any("=X" in s for s in symbols):  # FX leg
            return pd.DataFrame()
        calls["n"] += 1
        return intraday_df if interval == "1m" else daily_df

    with (
        patch.object(market_data, "_run_isolated_fetch", side_effect=fake_fetch),
        patch.object(market_data, "is_market_open", return_value=market_open),
    ):
        quotes, _fx, _fx_prev, err, _warn = provider.get_current_quotes(
            ["AAPL"], {"USD"}, {}, set()
        )
    assert not err
    return quotes, calls["n"]


@pytest.fixture
def two_session_frames():
    daily = _daily_frame(
        [
            (date(2026, 7, 30), THU_CLOSE),
            (date(2026, 7, 31), FRI_CLOSE),
        ]
    )
    intraday = _intraday_frame(["2026-07-31 19:59"], [FRI_CLOSE])
    return daily, intraday


def test_quotes_cached_before_the_open_are_refetched_once_trading_starts(
    provider, two_session_frames
):
    """The bell invalidates a pre-open quote set — the value must track the tape."""
    daily, intraday = two_session_frames

    _, fetches = _counting_quotes(provider, daily, intraday, market_open=False)
    assert fetches > 0  # cold: real fetch

    # Five minutes pass and the market opens. Under the closed-market window
    # this entry would still look fresh for another 55 minutes.
    _age_caches(provider, 300)

    _, fetches = _counting_quotes(provider, daily, intraday, market_open=True)
    assert fetches > 0, "pre-open quotes were served into the open session"


def test_quotes_are_reused_within_the_minute_while_trading(
    provider, two_session_frames
):
    """The open-session window is a minute, not zero — no refetch per request."""
    daily, intraday = two_session_frames

    _counting_quotes(provider, daily, intraday, market_open=True)
    _age_caches(provider, 10)

    _, fetches = _counting_quotes(provider, daily, intraday, market_open=True)
    assert fetches == 0


def test_quotes_are_reused_for_the_hour_while_the_market_is_shut(
    provider, two_session_frames
):
    """Nothing moves after the close, so the long window stays long."""
    daily, intraday = two_session_frames

    _counting_quotes(provider, daily, intraday, market_open=False)
    _age_caches(provider, 300)

    _, fetches = _counting_quotes(provider, daily, intraday, market_open=False)
    assert fetches == 0
