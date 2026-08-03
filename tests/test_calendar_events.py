"""Tests for server.calendar_events — the upcoming earnings/dividend derivation
shared by the dashboard Events panel and the stock-detail Overview tab.

Covers how the Yahoo timestamps are reconciled into a single "next report" and
"next payment" per symbol, and that every date is reckoned on the exchange's own
clock rather than the server's.
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import server.calendar_events as calendar_events  # noqa: E402
from server.calendar_events import (  # noqa: E402
    DEFAULT_MARKET_TIMEZONE,
    company_name,
    market_timezone,
    market_today,
    next_dividend_event,
    next_earnings_event,
    recent_earnings_event,
    upcoming_events,
)

TODAY = date(2026, 7, 21)
HORIZON = TODAY + timedelta(days=90)


def _ts(d: date) -> int:
    """Epoch seconds for midday UTC on `d` — how Yahoo reports earnings times."""
    return int(datetime(d.year, d.month, d.day, 12, 0, tzinfo=timezone.utc).timestamp())


def test_uses_announced_window_when_earnings_timestamp_is_stale():
    """`earningsTimestamp` often still points at the last report; the announced
    start date must win so an already-scheduled date isn't hidden."""
    next_date = TODAY + timedelta(days=10)
    info = {
        "earningsTimestamp": _ts(TODAY - timedelta(days=80)),
        "earningsTimestampStart": _ts(next_date),
        "earningsTimestampEnd": _ts(next_date),
        "isEarningsDateEstimate": False,
        "shortName": "Apple Inc.",
    }
    event = next_earnings_event("AAPL", info, TODAY, HORIZON)
    assert event["earnings_date"] == str(next_date)
    assert event["status"] == "confirmed"
    assert event["name"] == "Apple Inc."
    assert "earnings_date_end" not in event


def test_estimated_flag_and_date_range_are_surfaced():
    start = TODAY + timedelta(days=30)
    end = TODAY + timedelta(days=34)
    info = {
        "earningsTimestampStart": _ts(start),
        "earningsTimestampEnd": _ts(end),
        "isEarningsDateEstimate": True,
        "_earnings_estimate": {"0q": {"avg": 1.89, "yearAgoEps": 1.57}},
    }
    event = next_earnings_event("MSFT", info, TODAY, HORIZON)
    assert event["earnings_date"] == str(start)
    assert event["earnings_date_end"] == str(end)
    assert event["status"] == "estimated"
    assert event["eps_estimate"] == 1.89
    assert event["eps_year_ago"] == 1.57


def test_today_counts_as_upcoming():
    info = {"earningsTimestamp": _ts(TODAY)}
    assert next_earnings_event("NVDA", info, TODAY, HORIZON)["earnings_date"] == str(
        TODAY
    )


def test_past_only_and_beyond_horizon_yield_nothing():
    past = {"earningsTimestamp": _ts(TODAY - timedelta(days=1))}
    assert next_earnings_event("KO", past, TODAY, HORIZON) is None

    far = {"earningsTimestampStart": _ts(TODAY + timedelta(days=200))}
    assert next_earnings_event("KO", far, TODAY, HORIZON) is None


def test_company_name_falls_back_across_yahoo_name_fields():
    assert (
        company_name(
            {"shortName": "Alphabet Inc.", "longName": "Alphabet Incorporated"}
        )
        == "Alphabet Inc."
    )
    assert company_name({"longName": "Alphabet Inc."}) == "Alphabet Inc."
    assert company_name({"displayName": "Alphabet"}) == "Alphabet"
    # Blank/absent names must be dropped rather than rendered as an empty line.
    assert company_name({"shortName": "   ", "longName": None}) is None
    assert company_name({}) is None


def test_missing_or_garbage_timestamps_yield_nothing():
    for info in (
        {},
        {"earningsTimestamp": None},
        {"earningsTimestamp": 0},
        {"earningsTimestamp": "soon"},
    ):
        assert next_earnings_event("VOO", info, TODAY, HORIZON) is None


# ── Dividends ────────────────────────────────────────────────────────────────


def test_announced_dividend_is_reported_as_confirmed():
    pay = TODAY + timedelta(days=40)
    ex = TODAY + timedelta(days=25)
    info = {
        "dividendDate": _ts(pay),
        "exDividendDate": _ts(ex),
        "dividendRate": 2.12,
        "lastDividendValue": 0.53,
    }
    event = next_dividend_event("KO", info, TODAY)
    assert event["dividend_date"] == str(pay)
    assert event["ex_dividend_date"] == str(ex)
    assert event["status"] == "confirmed"
    assert event["frequency_months"] == 3
    assert event["amount_per_share"] == pytest.approx(0.53)


def test_stale_pay_date_projects_the_next_payment():
    """Yahoo leaves `dividendDate` on the last payment until the next is declared."""
    last = TODAY - timedelta(days=70)
    info = {
        "dividendDate": _ts(last),
        "exDividendDate": _ts(last - timedelta(days=3)),
        "lastDividendDate": _ts(last),
        "dividendRate": 1.08,
        "lastDividendValue": 0.27,
    }
    event = next_dividend_event("AAPL", info, TODAY)
    assert event["status"] == "estimated"
    assert date.fromisoformat(event["dividend_date"]) >= TODAY
    # One quarter on from the last payment.
    assert date.fromisoformat(event["dividend_date"]) == date(2026, 8, 12)
    assert event["ex_dividend_date"] is None


def test_non_payers_have_no_dividend_event():
    assert next_dividend_event("NVDA", {}, TODAY) is None
    assert (
        next_dividend_event("NVDA", {"dividendRate": 0, "lastDividendValue": 0}, TODAY)
        is None
    )


def test_upcoming_events_bundles_every_side():
    info = {
        "earningsTimestampStart": _ts(TODAY + timedelta(days=5)),
        "dividendDate": _ts(TODAY + timedelta(days=20)),
        "dividendRate": 2.0,
        "lastDividendValue": 0.5,
        "_earnings_history": {
            str(TODAY - timedelta(days=2)): {"eps_actual": 1.2, "eps_estimate": 1.0}
        },
    }
    both = upcoming_events("KO", info, TODAY)
    assert both["earnings"]["earnings_date"] == str(TODAY + timedelta(days=5))
    assert both["dividend"]["dividend_date"] == str(TODAY + timedelta(days=20))
    assert both["recent_earnings"]["earnings_date"] == str(TODAY - timedelta(days=2))
    assert upcoming_events("X", {}, TODAY) == {
        "earnings": None,
        "recent_earnings": None,
        "dividend": None,
    }


# ── Just-reported quarters ───────────────────────────────────────────────────


def test_reported_quarter_carries_what_was_printed():
    reported_on = TODAY - timedelta(days=1)
    info = {
        "earningsTimestamp": _ts(reported_on),
        "shortName": "Netflix, Inc.",
        "_earnings_history": {
            str(reported_on): {
                "eps_actual": 2.10,
                "eps_estimate": 1.95,
                "surprise_pct": 999,
            },
        },
    }
    event = recent_earnings_event("NFLX", info, TODAY)
    assert event["earnings_date"] == str(reported_on)
    assert event["status"] == "reported"
    assert event["name"] == "Netflix, Inc."
    assert event["eps_actual"] == 2.10
    assert event["eps_estimate"] == 1.95
    # Derived here, never read from Yahoo's ambiguous Surprise(%) column.
    assert event["surprise_pct"] == pytest.approx(7.6923, rel=1e-3)


def test_a_miss_is_reported_as_a_negative_surprise():
    info = {
        "_earnings_history": {str(TODAY): {"eps_actual": 0.80, "eps_estimate": 1.00}}
    }
    assert recent_earnings_event("X", info, TODAY)["surprise_pct"] == pytest.approx(
        -20.0
    )


def test_report_still_appears_before_the_figures_land():
    """The window between the release and Yahoo attaching the EPS: the report
    happened, so the row stays — it just has nothing to compare yet."""
    info = {
        "earningsTimestamp": _ts(TODAY),
        "isEarningsDateEstimate": False,
        "_earnings_history": {str(TODAY): {"eps_actual": None, "eps_estimate": 1.4}},
    }
    event = recent_earnings_event("MSFT", info, TODAY)
    assert event["earnings_date"] == str(TODAY)
    assert event["eps_actual"] is None
    assert event["surprise_pct"] is None


def test_a_merely_projected_past_date_is_not_a_report():
    """Yahoo guesses dates from the past cadence; a guess that has come and gone
    is no evidence anything was actually reported."""
    info = {
        "earningsTimestamp": _ts(TODAY - timedelta(days=2)),
        "isEarningsDateEstimate": True,
    }
    assert recent_earnings_event("X", info, TODAY) is None

    # …but once it shows up in the history table, it is a real report.
    info["_earnings_history"] = {
        str(TODAY - timedelta(days=2)): {"eps_actual": 3.0, "eps_estimate": 2.9}
    }
    assert recent_earnings_event("X", info, TODAY)["eps_actual"] == 3.0


def test_reports_older_than_the_lookback_are_dropped():
    old = TODAY - timedelta(days=9)
    info = {
        "earningsTimestamp": _ts(old),
        "_earnings_history": {str(old): {"eps_actual": 1.0}},
    }
    assert recent_earnings_event("X", info, TODAY) is None
    assert recent_earnings_event("X", info, TODAY, lookback_days=10)[
        "earnings_date"
    ] == str(old)


def test_the_newest_report_wins_when_history_holds_several():
    newest = TODAY - timedelta(days=3)
    info = {
        "_earnings_history": {
            str(TODAY - timedelta(days=95)): {"eps_actual": 1.0, "eps_estimate": 1.0},
            str(newest): {"eps_actual": 1.5, "eps_estimate": 1.4},
        }
    }
    assert recent_earnings_event("X", info, TODAY)["earnings_date"] == str(newest)


def test_a_report_today_is_not_also_listed_as_upcoming():
    """Yahoo leaves `earningsTimestamp` on the report it has just been through,
    so a company that reported this morning reads as both "reported today" and
    "reporting today". One event, already happened — it must not show twice."""
    info = {
        "earningsTimestamp": _ts(TODAY),
        "earningsTimestampStart": _ts(TODAY),
        "earningsTimestampEnd": _ts(TODAY),
        "_earnings_history": {str(TODAY): {"eps_actual": 2.02, "eps_estimate": 1.89}},
    }
    both = upcoming_events("AAPL", info, TODAY)
    assert both["recent_earnings"]["earnings_date"] == str(TODAY)
    assert both["recent_earnings"]["eps_actual"] == 2.02
    assert both["earnings"] is None

    # A genuinely scheduled next quarter is untouched by that suppression.
    info["earningsTimestampStart"] = _ts(TODAY + timedelta(days=91))
    assert upcoming_events("AAPL", info, TODAY)["earnings"]["earnings_date"] == str(
        TODAY + timedelta(days=91)
    )


def test_a_future_report_is_not_a_reported_one():
    info = {
        "_earnings_history": {
            str(TODAY + timedelta(days=2)): {"eps_actual": None, "eps_estimate": 1.0}
        }
    }
    assert recent_earnings_event("X", info, TODAY) is None
    assert recent_earnings_event("X", {}, TODAY) is None


def test_a_report_due_later_today_has_not_reported_yet():
    """A company reporting at 08:30 must not read as "reported" at midnight — the
    Events panel would show a result-less "Reported" row hours before the print,
    and suppress the scheduled row (with its consensus estimate) that belongs there."""
    tz = ZoneInfo("America/New_York")
    midnight = datetime(TODAY.year, TODAY.month, TODAY.day, 0, 23, tzinfo=tz)
    before_open = int(
        datetime(TODAY.year, TODAY.month, TODAY.day, 8, 30, tzinfo=tz).timestamp()
    )
    info = {
        "exchangeTimezoneName": "America/New_York",
        "earningsTimestamp": before_open,
        "earningsTimestampStart": before_open,
        "earningsTimestampEnd": before_open,
        "isEarningsDateEstimate": False,
    }
    assert recent_earnings_event("ABBV", info, TODAY, now=midnight) is None
    # It is still the next scheduled report, which is what the panel should show.
    assert next_earnings_event("ABBV", info, TODAY, HORIZON)["earnings_date"] == str(
        TODAY
    )

    # Once the announced time has gone by, it is a report — figures or not.
    after = midnight.replace(hour=9, minute=0)
    assert recent_earnings_event("ABBV", info, TODAY, now=after)[
        "earnings_date"
    ] == str(TODAY)


def test_the_day_with_figures_beats_a_projected_timestamp():
    """Yahoo's *estimated* `earningsTimestamp` can sit a day past the real print
    (ADP printed on the 28th while the blob still said the 29th). Taking the
    latest candidate date would pick the empty day and bury the result."""
    printed_on = TODAY - timedelta(days=2)
    info = {
        "earningsTimestamp": _ts(TODAY - timedelta(days=1)),
        "isEarningsDateEstimate": True,
        "_earnings_history": {
            str(printed_on): {"eps_actual": 2.64, "eps_estimate": 2.60}
        },
    }
    event = recent_earnings_event("ADP", info, TODAY)
    assert event["earnings_date"] == str(printed_on)
    assert event["eps_actual"] == 2.64


# ── Market-local reckoning ───────────────────────────────────────────────────


def _freeze(monkeypatch, moment: datetime) -> None:
    """Pin `datetime.now(tz)` inside calendar_events to one instant."""

    class _Now(datetime):
        @classmethod
        def now(cls, tz=None):
            return moment.astimezone(tz) if tz else moment

    monkeypatch.setattr(calendar_events, "datetime", _Now)


def test_market_timezone_comes_from_the_exchange():
    assert market_timezone({"exchangeTimezoneName": "Asia/Bangkok"}) == "Asia/Bangkok"
    assert (
        market_timezone({"exchangeTimezoneName": " Europe/London "}) == "Europe/London"
    )
    # Absent, blank, or unknown to the tz database → the US default.
    assert market_timezone({}) == DEFAULT_MARKET_TIMEZONE
    assert market_timezone({"exchangeTimezoneName": "  "}) == DEFAULT_MARKET_TIMEZONE
    assert (
        market_timezone({"exchangeTimezoneName": "Mars/Olympus"})
        == DEFAULT_MARKET_TIMEZONE
    )
    assert market_timezone({"exchangeTimezoneName": 1234}) == DEFAULT_MARKET_TIMEZONE


def test_market_today_follows_the_exchange_not_the_server(monkeypatch):
    """09:00 Bangkok on Jul 30 is still 22:00 New York on Jul 29."""
    _freeze(monkeypatch, datetime(2026, 7, 30, 2, 0, tzinfo=timezone.utc))
    assert market_today({}) == date(2026, 7, 29)
    assert market_today({"exchangeTimezoneName": "America/New_York"}) == date(
        2026, 7, 29
    )
    assert market_today({"exchangeTimezoneName": "Asia/Bangkok"}) == date(2026, 7, 30)


def test_us_event_today_is_not_dropped_from_a_bangkok_evening(monkeypatch):
    """The Upcoming Events regression: with the server a day ahead, a US report
    happening on the NYSE's today must still be reported, not filtered as past."""
    _freeze(monkeypatch, datetime(2026, 7, 30, 2, 0, tzinfo=timezone.utc))
    info = {
        "exchangeTimezoneName": "America/New_York",
        # 16:00 ET on Jul 29 — after the close, so already Jul 30 in UTC.
        "earningsTimestamp": int(
            datetime(
                2026, 7, 29, 16, 0, tzinfo=ZoneInfo("America/New_York")
            ).timestamp()
        ),
    }
    event = next_earnings_event("MSFT", info)
    assert event["earnings_date"] == "2026-07-29"
    assert event["market_timezone"] == "America/New_York"


def test_earnings_timestamp_is_read_on_the_market_clock():
    """A post-close report has already tipped into the next UTC day; the date the
    user cares about is the one on the exchange's wall clock."""
    ts = int(
        datetime(2026, 8, 5, 20, 30, tzinfo=ZoneInfo("America/New_York")).timestamp()
    )
    info = {"exchangeTimezoneName": "America/New_York", "earningsTimestampStart": ts}
    assert (
        next_earnings_event("X", info, date(2026, 8, 1))["earnings_date"]
        == "2026-08-05"
    )


def test_dividend_dates_stay_utc_calendar_days():
    """Yahoo's pay/ex dates are date-only values encoded as midnight UTC — reading
    them on a zone west of UTC would shift every payment a day early."""
    info = {
        "exchangeTimezoneName": "America/New_York",
        # Midnight UTC on Sep 15, exactly as Yahoo encodes a calendar day.
        "exDividendDate": int(datetime(2026, 9, 15, tzinfo=timezone.utc).timestamp()),
        "dividendDate": int(datetime(2026, 10, 1, tzinfo=timezone.utc).timestamp()),
        "dividendRate": 2.12,
        "lastDividendValue": 0.53,
    }
    event = next_dividend_event("KO", info, date(2026, 8, 1))
    assert event["dividend_date"] == "2026-10-01"
    assert event["ex_dividend_date"] == "2026-09-15"
    assert event["market_timezone"] == "America/New_York"


def test_events_carry_the_zone_they_were_reckoned_in():
    """Clients count "days from now" in this zone, so it must always be present."""
    info = {
        "exchangeTimezoneName": "Asia/Bangkok",
        "earningsTimestampStart": _ts(TODAY + timedelta(days=5)),
        "dividendDate": _ts(TODAY + timedelta(days=20)),
        "dividendRate": 2.0,
        "lastDividendValue": 0.5,
    }
    both = upcoming_events("PTT.BK", info, TODAY)
    assert both["earnings"]["market_timezone"] == "Asia/Bangkok"
    assert both["dividend"]["market_timezone"] == "Asia/Bangkok"
