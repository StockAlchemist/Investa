"""A previous close taken from a download that skipped a session.

Yahoo's live daily window is not gap-free. On 2026-08-31 the series it served
for every US equity ran ...26, 27, 31 — Friday the 28th simply absent — and the
quote path, which took the second-to-last bar as "yesterday", priced Monday
against Thursday. Two sessions of movement were reported as one day's:

    NOW    +5.1% against a real +0.6%
    AMZN   +1.6% against a real -2.3%
    GOOG   -1.1% against a real -2.6%

The portfolio headline read +$1.5k on a day it was down $25.7k, and named AMZN
and GOOG top gainers while the index card beside it — a different fetch, one
that had the Friday bar — was correctly red.

The archive settles it. It is written from the same source but accumulated over
time, so a session missing from one window is still on disk; a bar it holds
between the download's last two dates is proof the download skipped one.
"""

from datetime import date
from unittest.mock import patch

import pandas as pd
import pytest

import market_data
from market_data import MarketDataProvider

THU = date(2026, 8, 27)
FRI = date(2026, 8, 28)
MON = date(2026, 8, 31)

THU_CLOSE = 138.43
FRI_CLOSE = 144.71
MON_PRICE = 145.33


def _daily_frame(rows):
    """MultiIndex (ticker, field) daily frame, shaped like yf.download."""
    idx = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    return pd.DataFrame({("NOW", "Close"): [r[1] for r in rows]}, index=idx)


@pytest.fixture
def provider(tmp_path):
    mdp = MarketDataProvider(
        current_cache_file=str(tmp_path / "quotes.json"),
        db_path=str(tmp_path / "market.db"),
    )
    with (
        patch.object(
            MarketDataProvider,
            "_ensure_metadata_batch",
            return_value={"NOW": {"currency": "USD", "name": "ServiceNow, Inc."}},
        ),
        patch.object(MarketDataProvider, "get_fundamental_data_batch", return_value={}),
    ):
        yield mdp


def _archive(provider, rows):
    """Put daily closes on disk, the way the refresh worker accumulates them."""
    idx = pd.DatetimeIndex([r[0] for r in rows], name="Date")
    frame = pd.DataFrame(
        {
            "Open": [r[1] for r in rows],
            "High": [r[1] for r in rows],
            "Low": [r[1] for r in rows],
            "Close": [r[1] for r in rows],
            "Adj Close": [r[1] for r in rows],
            "Volume": [0] * len(rows),
        },
        index=idx,
    )
    provider.db.upsert_ohlcv("NOW", frame, interval="1d")


def _quote(provider, daily_df):
    """Run get_current_quotes with the download stubbed and no intraday leg."""

    def fake_fetch(symbols, period=None, interval=None, task=None, **kwargs):
        if interval == "1m" or any("=X" in s for s in symbols):
            return pd.DataFrame()
        return daily_df

    with patch.object(market_data, "_run_isolated_fetch", side_effect=fake_fetch):
        quotes, _fx, _fx_prev, err, _warn = provider.get_current_quotes(
            ["NOW"], {"USD"}, {}, set()
        )
    assert not err
    return quotes["NOW"]


def test_a_session_the_download_skipped_is_taken_from_the_archive(provider):
    """Friday is missing from the window but present on disk: it is yesterday."""
    _archive(provider, [(THU, THU_CLOSE), (FRI, FRI_CLOSE), (MON, MON_PRICE)])
    quote = _quote(provider, _daily_frame([(THU, THU_CLOSE), (MON, MON_PRICE)]))

    assert quote["price"] == pytest.approx(MON_PRICE)
    assert quote["change"] == pytest.approx(MON_PRICE - FRI_CLOSE)
    assert quote["changesPercentage"] == pytest.approx(
        (MON_PRICE - FRI_CLOSE) / FRI_CLOSE * 100
    )


def test_an_ungapped_download_is_left_alone(provider):
    """The archive agrees on which session came last; nothing to repair."""
    _archive(provider, [(THU, THU_CLOSE), (FRI, FRI_CLOSE), (MON, MON_PRICE)])
    quote = _quote(
        provider, _daily_frame([(THU, THU_CLOSE), (FRI, FRI_CLOSE), (MON, MON_PRICE)])
    )

    assert quote["change"] == pytest.approx(MON_PRICE - FRI_CLOSE)


def test_a_lagging_archive_never_overrides_a_fresher_download(provider):
    """The archive stops at Thursday; the download's Friday bar still wins."""
    _archive(provider, [(THU, THU_CLOSE)])
    quote = _quote(provider, _daily_frame([(FRI, FRI_CLOSE), (MON, MON_PRICE)]))

    assert quote["change"] == pytest.approx(MON_PRICE - FRI_CLOSE)


def test_an_empty_archive_leaves_the_download_alone(provider):
    """Nothing on disk — the previous behaviour, unchanged."""
    quote = _quote(provider, _daily_frame([(THU, THU_CLOSE), (MON, MON_PRICE)]))

    assert quote["change"] == pytest.approx(MON_PRICE - THU_CLOSE)


def test_todays_own_archived_bar_is_not_its_own_previous_close(provider):
    """The archive carries Monday too, written intraday. It is not yesterday."""
    _archive(provider, [(FRI, FRI_CLOSE), (MON, MON_PRICE - 2.0)])
    quote = _quote(provider, _daily_frame([(THU, THU_CLOSE), (MON, MON_PRICE)]))

    assert quote["change"] == pytest.approx(MON_PRICE - FRI_CLOSE)
