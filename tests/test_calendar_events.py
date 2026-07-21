"""Tests for server.calendar_events — the upcoming earnings/dividend derivation
shared by the dashboard Events panel and the stock-detail Overview tab.

Covers how the Yahoo timestamps are reconciled into a single "next report" and
"next payment" per symbol.
"""

import os
import sys
from datetime import date, datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from server.calendar_events import (  # noqa: E402
    company_name,
    next_dividend_event,
    next_earnings_event,
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
    assert next_earnings_event("NVDA", info, TODAY, HORIZON)["earnings_date"] == str(TODAY)


def test_past_only_and_beyond_horizon_yield_nothing():
    past = {"earningsTimestamp": _ts(TODAY - timedelta(days=1))}
    assert next_earnings_event("KO", past, TODAY, HORIZON) is None

    far = {"earningsTimestampStart": _ts(TODAY + timedelta(days=200))}
    assert next_earnings_event("KO", far, TODAY, HORIZON) is None


def test_company_name_falls_back_across_yahoo_name_fields():
    assert company_name({"shortName": "Alphabet Inc.", "longName": "Alphabet Incorporated"}) == "Alphabet Inc."
    assert company_name({"longName": "Alphabet Inc."}) == "Alphabet Inc."
    assert company_name({"displayName": "Alphabet"}) == "Alphabet"
    # Blank/absent names must be dropped rather than rendered as an empty line.
    assert company_name({"shortName": "   ", "longName": None}) is None
    assert company_name({}) is None


def test_missing_or_garbage_timestamps_yield_nothing():
    for info in ({}, {"earningsTimestamp": None}, {"earningsTimestamp": 0},
                 {"earningsTimestamp": "soon"}):
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
    assert next_dividend_event("NVDA", {"dividendRate": 0, "lastDividendValue": 0}, TODAY) is None


def test_upcoming_events_bundles_both_sides():
    info = {
        "earningsTimestampStart": _ts(TODAY + timedelta(days=5)),
        "dividendDate": _ts(TODAY + timedelta(days=20)),
        "dividendRate": 2.0,
        "lastDividendValue": 0.5,
    }
    both = upcoming_events("KO", info, TODAY)
    assert both["earnings"]["earnings_date"] == str(TODAY + timedelta(days=5))
    assert both["dividend"]["dividend_date"] == str(TODAY + timedelta(days=20))
    assert upcoming_events("X", {}, TODAY) == {"earnings": None, "dividend": None}
