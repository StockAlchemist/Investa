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
