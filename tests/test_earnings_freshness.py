"""Tests for the earnings-aware fundamentals cache expiry (`market_data`).

A report is the one moment a fundamentals blob turns from current into stale.
The cache used to park expiry 24h *after* the report, so every surface built on
the blob — the dashboard earnings calendar, the Overview tab, the consensus row —
served pre-report numbers for a full day afterwards. These pin the replacement:
the schedule may only ever *shorten* a blob's life.
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from market_data import (  # noqa: E402
    EARNINGS_PUBLISH_GRACE_MINUTES,
    POST_EARNINGS_POLL_HOURS,
    POST_EARNINGS_WATCH_DAYS,
    fundamentals_valid_until,
)

NOW = datetime(2026, 7, 21, 14, 0, tzinfo=timezone.utc)
DEFAULT_UNTIL = NOW + timedelta(hours=24)
GRACE = timedelta(minutes=EARNINGS_PUBLISH_GRACE_MINUTES)


def _ts(moment: datetime) -> int:
    return int(moment.timestamp())


def test_expiry_lands_at_the_report_not_a_day_after_it():
    report = NOW + timedelta(hours=3)
    info = {"earningsTimestamp": _ts(report)}
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == report + GRACE


def test_the_nearest_announced_date_wins_over_a_stale_timestamp():
    """`earningsTimestamp` often still points at the last report; a nearer
    announced date must not have its expiry pushed out past it."""
    info = {
        "earningsTimestamp": _ts(NOW + timedelta(hours=20)),
        "earningsTimestampStart": _ts(NOW + timedelta(hours=2)),
        "earningsTimestampEnd": _ts(NOW + timedelta(hours=6)),
    }
    expected = NOW + timedelta(hours=2) + GRACE
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == expected


def test_a_distant_report_never_extends_the_standard_ttl():
    info = {"earningsTimestamp": _ts(NOW + timedelta(days=60))}
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == DEFAULT_UNTIL


def test_a_report_with_no_figures_yet_is_re_polled():
    """Yahoo can take hours to attach the reported EPS. Until it does, checking
    back beats serving the pre-report blob for the rest of the day."""
    info = {"earningsTimestamp": _ts(NOW - timedelta(hours=2))}
    expected = NOW + timedelta(hours=POST_EARNINGS_POLL_HOURS)
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == expected


def test_polling_stops_once_the_figures_have_landed():
    report = NOW - timedelta(hours=2)
    info = {
        "exchangeTimezoneName": "America/New_York",
        "earningsTimestamp": _ts(report),
        "_earnings_history": {
            report.astimezone(ZoneInfo("America/New_York")).date().isoformat(): {
                "eps_actual": 2.1,
                "eps_estimate": 1.95,
            }
        },
    }
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == DEFAULT_UNTIL


def test_a_post_close_report_is_matched_on_the_market_clock():
    """A 16:00 ET report has already tipped into the next UTC day; reading the
    history key in UTC would miss the row and re-poll a blob that is complete."""
    report = datetime(2026, 7, 20, 16, 5, tzinfo=ZoneInfo("America/New_York"))
    now = report + timedelta(hours=3)
    info = {
        "exchangeTimezoneName": "America/New_York",
        "earningsTimestamp": _ts(report),
        "_earnings_history": {"2026-07-20": {"eps_actual": 2.1, "eps_estimate": 1.95}},
    }
    assert report.astimezone(timezone.utc).date().isoformat() == "2026-07-20"  # same UTC day here
    default_until = now + timedelta(hours=24)
    assert fundamentals_valid_until(info, now, default_until) == default_until

    # 21:00 ET — the same report, now on the *next* UTC day.
    late = datetime(2026, 7, 20, 21, 0, tzinfo=ZoneInfo("America/New_York"))
    info["earningsTimestamp"] = _ts(late)
    assert late.astimezone(timezone.utc).date().isoformat() == "2026-07-21"
    now_late = late + timedelta(hours=2)
    default_late = now_late + timedelta(hours=24)
    assert fundamentals_valid_until(info, now_late, default_late) == default_late


def test_polling_gives_up_after_the_watch_window():
    """Some names simply never get a figure attached — the poll must not run
    forever on them."""
    info = {"earningsTimestamp": _ts(NOW - timedelta(days=POST_EARNINGS_WATCH_DAYS + 1))}
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == DEFAULT_UNTIL


def test_a_recent_report_and_a_far_future_one_still_poll():
    """The usual steady state right after a print: last quarter is days old and
    next quarter is a full quarter away."""
    info = {
        "earningsTimestamp": _ts(NOW - timedelta(hours=6)),
        "earningsTimestampStart": _ts(NOW + timedelta(days=89)),
    }
    expected = NOW + timedelta(hours=POST_EARNINGS_POLL_HOURS)
    assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == expected


def test_blobs_without_a_schedule_keep_the_standard_ttl():
    for info in ({}, {"earningsTimestamp": None}, {"earningsTimestamp": 0},
                 {"earningsTimestamp": "soon"}, {"earningsTimestamp": True}, None):
        assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == DEFAULT_UNTIL


def test_expiry_is_never_pinned_in_the_past():
    """A bogus timestamp or a skewed clock must not make every request re-fetch."""
    info = {"earningsTimestamp": _ts(NOW - timedelta(minutes=5))}
    assert fundamentals_valid_until(info, NOW, NOW - timedelta(hours=5)) > NOW


def test_a_garbage_history_entry_is_treated_as_no_figures():
    report = NOW - timedelta(hours=2)
    day = report.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    for history in ({day: None}, {day: {"eps_actual": None}}, {day: "2.10"}, "nope"):
        info = {"earningsTimestamp": _ts(report), "_earnings_history": history}
        assert fundamentals_valid_until(info, NOW, DEFAULT_UNTIL) == NOW + timedelta(
            hours=POST_EARNINGS_POLL_HOURS
        )


def test_a_legacy_entry_pinned_past_its_own_report_is_stale():
    """The regression that hid every reported figure: the old writer parked
    expiry 24h *after* the next report, so a blob written in July for a company
    reporting in October was served, pre-report, until October. Expiry is now
    re-derived from the blob, so what an older build stored cannot outlive the
    standard TTL."""
    written = NOW - timedelta(days=2)
    info = {"earningsTimestamp": _ts(NOW + timedelta(days=90))}

    recomputed = fundamentals_valid_until(info, written, written + timedelta(hours=24))
    assert recomputed == written + timedelta(hours=24)
    assert recomputed < NOW  # expired, where the stored value said "October"


def test_a_blob_from_a_minimal_fetch_expires_into_a_full_one():
    """The batch path fetches with `minimal=True`, which skips the analyst and
    earnings-history extras. Such a blob must not sit on a just-reported quarter
    for a full day claiming the figures are unpublished."""
    written = NOW - timedelta(minutes=90)
    info = {"earningsTimestamp": _ts(NOW - timedelta(hours=3))}  # no _earnings_history
    valid_until = fundamentals_valid_until(info, written, written + timedelta(hours=24))
    assert valid_until == written + timedelta(hours=POST_EARNINGS_POLL_HOURS)
    assert valid_until < NOW


# ── The worker side: reported figures riding along with .info ────────────────


class _FakeTicker:
    """Stands in for `yfinance.Ticker` — only `get_earnings_dates` is used."""

    def __init__(self, frame):
        self._frame = frame

    def get_earnings_dates(self, limit=None):
        return self._frame


def test_earnings_history_rows_are_keyed_by_the_exchange_local_day():
    import pandas as pd

    from market_data_worker import _earnings_history_rows

    index = pd.DatetimeIndex(
        [
            pd.Timestamp("2026-07-20 16:05", tz="America/New_York"),
            pd.Timestamp("2026-04-21 16:05", tz="America/New_York"),
        ]
    )
    frame = pd.DataFrame(
        {
            "EPS Estimate": [1.95, 1.80],
            "Reported EPS": [2.10, float("nan")],
            "Surprise(%)": [0.0769, float("nan")],
        },
        index=index,
    )

    rows = _earnings_history_rows(_FakeTicker(frame))
    # 16:05 ET is already the next day in UTC — the key is the market's day.
    assert set(rows) == {"2026-07-20", "2026-04-21"}
    assert rows["2026-07-20"]["eps_actual"] == 2.10
    assert rows["2026-07-20"]["eps_estimate"] == 1.95
    # NaN becomes None so the cached blob stays valid JSON.
    assert rows["2026-04-21"]["eps_actual"] is None


def test_earnings_history_rows_cope_with_an_empty_table():
    import pandas as pd

    from market_data_worker import _earnings_history_rows

    assert _earnings_history_rows(_FakeTicker(None)) == {}
    assert _earnings_history_rows(_FakeTicker(pd.DataFrame())) == {}


# ── Filling the figures in when the blob has none ────────────────────────────


def test_the_history_survives_a_minimal_batch_overwrite(tmp_path):
    """A screener sweep writes minimal info over the same per-symbol file. It
    carries no `_earnings_history`, and stripping the figures off would put the
    Events panel back to saying a company reported without saying what."""
    from market_data import MarketDataProvider

    provider = MarketDataProvider(fundamentals_cache_dir=str(tmp_path))
    path = provider._get_symbol_fundamentals_path("AAPL")
    full = {"symbol": "AAPL", "_earnings_history": {"2026-07-30": {"eps_actual": 2.02}}}
    with open(path, "w") as f:
        json.dump({"timestamp": NOW.isoformat(), "data": full}, f)

    minimal = {f"k{i}": i for i in range(12)} | {"symbol": "AAPL"}
    provider._save_fundamentals_cache({"AAPL": {"ticker_info": minimal}})

    with open(path) as f:
        saved = json.load(f)
    assert saved["data"]["_earnings_history"] == {"2026-07-30": {"eps_actual": 2.02}}
    # A fetch that *does* carry the figures still wins.
    provider._save_fundamentals_cache(
        {"AAPL": {"ticker_info": minimal | {"_earnings_history": {"2026-07-30": {"eps_actual": 2.10}}}}}
    )
    with open(path) as f:
        assert json.load(f)["data"]["_earnings_history"]["2026-07-30"]["eps_actual"] == 2.10


def test_a_reported_quarter_with_no_figures_is_backfilled(tmp_path, monkeypatch):
    """The blob predates the report (or came from a batch write), so it says a
    company reported and nothing more. Go and get the print — once, then cached."""
    import market_data
    from market_data import MarketDataProvider

    provider = MarketDataProvider(fundamentals_cache_dir=str(tmp_path))
    path = provider._get_symbol_fundamentals_path("MA")
    reported_at = datetime.now(timezone.utc) - timedelta(hours=6)
    info = {"symbol": "MA", "earningsTimestamp": int(reported_at.timestamp())}
    with open(path, "w") as f:
        json.dump({"timestamp": NOW.isoformat(), "data": info}, f)

    day = reported_at.astimezone(ZoneInfo("America/New_York")).date().isoformat()
    calls = []

    def _fake_fetch(*args, **kwargs):
        calls.append(kwargs.get("task"))
        import pandas as pd

        return pd.DataFrame(
            {"EPS Estimate": [4.78], "Reported EPS": [5.04], "Surprise(%)": [5.53]},
            index=pd.DatetimeIndex([pd.Timestamp(f"{day} 08:00")], name="date"),
        )

    monkeypatch.setattr(market_data, "_run_isolated_fetch", _fake_fetch)
    market_data._EARNINGS_BACKFILL_ATTEMPTS.clear()

    filled = provider.with_reported_earnings("MA", info)
    assert filled["_earnings_history"][day]["eps_actual"] == 5.04
    # Stashed on the blob, so every later reader gets it for free.
    with open(path) as f:
        assert json.load(f)["data"]["_earnings_history"][day]["eps_actual"] == 5.04

    # A blob that already carries the figures asks Yahoo nothing.
    provider.with_reported_earnings("MA", filled)
    assert calls == ["earnings_dates"]


def test_the_backfill_does_not_re_ask_for_a_table_yahoo_has_not_filled_in(tmp_path, monkeypatch):
    import market_data
    from market_data import MarketDataProvider

    provider = MarketDataProvider(fundamentals_cache_dir=str(tmp_path))
    info = {"symbol": "X", "earningsTimestamp": int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp())}

    calls = []

    def _empty_fetch(*args, **kwargs):
        import pandas as pd

        calls.append(kwargs.get("task"))
        return pd.DataFrame()

    monkeypatch.setattr(market_data, "_run_isolated_fetch", _empty_fetch)
    market_data._EARNINGS_BACKFILL_ATTEMPTS.clear()

    assert provider.with_reported_earnings("X", info) is info
    assert provider.with_reported_earnings("X", info) is info
    assert calls == ["earnings_dates"]  # the cool-down held the second one back


def test_a_report_outside_the_backfill_window_is_left_alone(tmp_path, monkeypatch):
    """Old quarters are nobody's news: the Events panel cannot show them, so they
    are not worth a fetch."""
    import market_data
    from market_data import EARNINGS_BACKFILL_WINDOW_DAYS, MarketDataProvider

    provider = MarketDataProvider(fundamentals_cache_dir=str(tmp_path))
    old = datetime.now(timezone.utc) - timedelta(days=EARNINGS_BACKFILL_WINDOW_DAYS + 1)
    info = {"symbol": "X", "earningsTimestamp": int(old.timestamp())}

    def _boom(*args, **kwargs):
        raise AssertionError("should not fetch")

    monkeypatch.setattr(market_data, "_run_isolated_fetch", _boom)
    market_data._EARNINGS_BACKFILL_ATTEMPTS.clear()
    assert provider.with_reported_earnings("X", info) is info
