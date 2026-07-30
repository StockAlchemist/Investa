"""Upcoming earnings / dividend events derived from a Yahoo fundamentals blob.

Pure helpers shared by the dashboard calendars (`routes/analytics.py`) and the
stock-detail Overview tab (`routes/fundamentals` piggyback in `routes/market.py`),
so both surfaces answer "when does this company next report / pay?" identically.
"""

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Dict, Optional
from zoneinfo import ZoneInfo

import pandas as pd

# Investa reckons every market date in the market's own local time, never the
# server's or the viewer's — see CLAUDE.md. US-listed names are the default when
# a blob does not name its exchange's zone.
DEFAULT_MARKET_TIMEZONE = "America/New_York"


def market_timezone(info: Dict) -> str:
    """
    IANA zone of the exchange a symbol trades on, from Yahoo's
    `exchangeTimezoneName` (`America/New_York`, `Asia/Bangkok`, …).

    Falls back to the US zone for blobs that omit it or name a zone this
    machine's tz database does not carry.
    """
    tz = info.get("exchangeTimezoneName") if info else None
    if isinstance(tz, str) and tz.strip():
        tz = tz.strip()
        try:
            ZoneInfo(tz)
        except (KeyError, ValueError):
            logging.debug(f"Unknown exchange timezone {tz!r}; using {DEFAULT_MARKET_TIMEZONE}")
        else:
            return tz
    return DEFAULT_MARKET_TIMEZONE


def market_today(info: Dict) -> date:
    """
    Today's calendar date on the symbol's own exchange.

    Use this rather than `date.today()` for anything a user reads as "days from
    now": in Bangkok (UTC+7) the server rolls over to tomorrow while New York is
    still mid-afternoon, which turns a report happening *today* into "1d ago".
    """
    return datetime.now(ZoneInfo(market_timezone(info))).date()


def epoch_to_date(ts) -> Optional[date]:
    """
    UTC calendar date for a Yahoo epoch-seconds timestamp, or None.

    For Yahoo's *date-only* fields (`dividendDate`, `exDividendDate`,
    `lastDividendDate`), which encode a calendar day as midnight UTC — reading
    those in an exchange zone west of UTC would shift them a day early.
    """
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=timezone.utc).date()
    except (OSError, OverflowError, ValueError):
        return None


def epoch_to_market_date(ts, tz: str) -> Optional[date]:
    """
    Exchange-local calendar date for a Yahoo epoch-seconds timestamp, or None.

    For fields that carry a real wall-clock moment — the earnings timestamps sit
    at the report's actual time (08:30 or 16:00 ET), so a post-close report has
    already tipped into the next UTC day and must be read on the market's clock.
    """
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=ZoneInfo(tz)).date()
    except (OSError, OverflowError, ValueError, KeyError):
        return None


def company_name(info: Dict) -> Optional[str]:
    """
    Human-readable company name from a fundamentals blob, for calendar rows that
    show a ticker (GOOG vs GOOGL reads better as "Alphabet Inc." on both).
    """
    for key in ("shortName", "longName", "displayName"):
        value = info.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def next_earnings_event(
    symbol: str, info: Dict, today: Optional[date] = None, horizon_end: Optional[date] = None
) -> Optional[dict]:
    """
    Pull the next scheduled earnings report out of a cached fundamentals blob.

    Yahoo exposes three timestamps: `earningsTimestamp` (the most recent known
    report, which flips to the next one once announced) and the
    `earningsTimestampStart`/`End` pair (the announced window, equal when the
    date is exact). Take the earliest of those that is still in the future so a
    stale `earningsTimestamp` can't mask an already-announced next date.

    `today` defaults to today on the symbol's own exchange; pass it only to pin
    the reckoning (tests, or a caller that already resolved the market date).
    """
    tz = market_timezone(info)
    if today is None:
        today = market_today(info)
    ts_end = info.get("earningsTimestampEnd")

    upcoming = [
        d
        for d in (
            epoch_to_market_date(info.get("earningsTimestampStart"), tz),
            epoch_to_market_date(info.get("earningsTimestamp"), tz),
            epoch_to_market_date(ts_end, tz),
        )
        if d is not None and d >= today
    ]
    if not upcoming:
        return None

    event_date = min(upcoming)
    if horizon_end is not None and event_date > horizon_end:
        return None

    event = {
        "symbol": symbol,
        "name": company_name(info),
        "earnings_date": str(event_date),
        # Yahoo flags dates it has inferred from the historical reporting cadence.
        "status": "estimated" if info.get("isEarningsDateEstimate") else "confirmed",
        # The zone this date is a date *in*, so clients count "days from now" the
        # same way the backend filtered it.
        "market_timezone": tz,
    }

    # An announced window (start != end) means the company has given a range
    # rather than a day — surface the far end so the UI can say "Feb 3–7".
    window_end = epoch_to_market_date(ts_end, tz)
    if window_end and window_end > event_date:
        event["earnings_date_end"] = str(window_end)

    # Current-quarter consensus, stashed by the market-data worker.
    estimates = info.get("_earnings_estimate")
    if isinstance(estimates, dict):
        current_q = estimates.get("0q")
        if isinstance(current_q, dict):
            event["eps_estimate"] = current_q.get("avg")
            event["eps_year_ago"] = current_q.get("yearAgoEps")

    return event


def next_dividend_event(symbol: str, info: Dict, today: Optional[date] = None) -> Optional[dict]:
    """
    The next dividend for one symbol, per share (not scaled by any position).

    Prefers the announced pay/ex dates. Those go stale the moment a dividend is
    paid — Yahoo leaves `dividendDate` pointing at the last payment until the
    next is declared — so when they are in the past the next one is projected
    from the latest known payment plus the detected cadence.

    `today` defaults to today on the symbol's own exchange (see `market_today`).
    """
    from finutils import get_dividend_details

    tz = market_timezone(info)
    if today is None:
        today = market_today(info)

    details = get_dividend_details(info)
    annual_rate = details["indicated_annual_rate"]
    freq_months = details["frequency_months"]
    if not annual_rate or annual_rate <= 0:
        return None

    per_share = annual_rate / (12 // freq_months) if freq_months else annual_rate
    pay_date = epoch_to_date(info.get("dividendDate"))
    ex_date = epoch_to_date(info.get("exDividendDate"))

    def _event(d: date, ex: Optional[date], status: str) -> dict:
        return {
            "symbol": symbol,
            "name": company_name(info),
            "dividend_date": str(d),
            "ex_dividend_date": str(ex) if ex else None,
            "amount_per_share": per_share,
            "frequency_months": freq_months,
            "status": status,
            "market_timezone": tz,
        }

    # Announced: the declared pay date (or at least its ex-date) is still ahead.
    if pay_date and pay_date >= today:
        return _event(pay_date, ex_date, "confirmed")
    if ex_date and ex_date >= today:
        return _event(ex_date, ex_date, "confirmed")

    if not freq_months:
        return None

    anchor = max(
        [d for d in (pay_date, ex_date, epoch_to_date(info.get("lastDividendDate"))) if d],
        default=None,
    )
    if not anchor:
        return None

    try:
        projected = anchor
        # Guard the loop: a wildly stale anchor must not spin indefinitely.
        for _ in range(64):
            if projected >= today:
                break
            projected = (pd.Timestamp(projected) + pd.DateOffset(months=freq_months)).date()
        else:
            return None
    except (ValueError, OverflowError) as e:
        logging.debug(f"Dividend projection failed for {symbol}: {e}")
        return None

    return _event(projected, None, "estimated")


def upcoming_events(
    symbol: str, info: Dict, today: Optional[date] = None, horizon_days: int = 365
) -> dict:
    """Both event types for one symbol, as consumed by the stock-detail Overview tab."""
    if not info:
        return {"earnings": None, "dividend": None}
    if today is None:
        today = market_today(info)
    horizon_end = today + timedelta(days=horizon_days)
    return {
        "earnings": next_earnings_event(symbol, info, today, horizon_end),
        "dividend": next_dividend_event(symbol, info, today),
    }
