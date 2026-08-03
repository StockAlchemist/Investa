"""Earnings / dividend events derived from a Yahoo fundamentals blob.

Pure helpers shared by the dashboard calendars (`routes/analytics.py`) and the
stock-detail Overview tab (`routes/fundamentals` piggyback in `routes/market.py`),
so both surfaces answer "when does this company next report / pay?" identically.

Reports run forwards *and* backwards: `next_earnings_event` is the scheduled one,
`recent_earnings_event` is the quarter a company has just printed, so a report
resolves into its result on the calendar instead of silently dropping off it the
day after.
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

# How long a quarter that has just been reported stays on the calendar. Long
# enough to survive a weekend and a Monday morning, short enough that the panel
# stays a view of what is happening now.
REPORTED_LOOKBACK_DAYS = 5

# The three timestamps Yahoo names an earnings date with, earliest-first.
EARNINGS_TIMESTAMP_KEYS = (
    "earningsTimestampStart",
    "earningsTimestamp",
    "earningsTimestampEnd",
)


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
            logging.debug(
                f"Unknown exchange timezone {tz!r}; using {DEFAULT_MARKET_TIMEZONE}"
            )
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


def epoch_to_market_datetime(ts, tz: str) -> Optional[datetime]:
    """
    Exchange-local wall-clock moment for a Yahoo epoch-seconds timestamp, or None.

    The earnings timestamps carry the report's actual time (08:30 or 16:00 ET),
    which is the difference between "reports today" and "has reported".
    """
    if not isinstance(ts, (int, float)) or isinstance(ts, bool) or ts <= 0:
        return None
    try:
        return datetime.fromtimestamp(ts, tz=ZoneInfo(tz))
    except (OSError, OverflowError, ValueError, KeyError):
        return None


def epoch_to_market_date(ts, tz: str) -> Optional[date]:
    """
    Exchange-local calendar date for a Yahoo epoch-seconds timestamp, or None.

    For fields that carry a real wall-clock moment — the earnings timestamps sit
    at the report's actual time (08:30 or 16:00 ET), so a post-close report has
    already tipped into the next UTC day and must be read on the market's clock.
    """
    moment = epoch_to_market_datetime(ts, tz)
    return moment.date() if moment else None


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
    symbol: str,
    info: Dict,
    today: Optional[date] = None,
    horizon_end: Optional[date] = None,
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
            epoch_to_market_date(info.get(key), tz) for key in EARNINGS_TIMESTAMP_KEYS
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


def earnings_history(info: Dict) -> Dict[str, dict]:
    """
    Reported quarters keyed by market-local report date, as stashed on the blob
    by the market-data worker (`_earnings_history_rows`). Empty when the blob
    predates that field or the fetch failed — every caller must cope with a
    report it has no figures for yet.
    """
    history = info.get("_earnings_history") if info else None
    return history if isinstance(history, dict) else {}


def _surprise_pct(
    eps_actual: Optional[float], eps_estimate: Optional[float]
) -> Optional[float]:
    """
    Beat/miss against consensus, in percent.

    Derived here rather than taken from Yahoo's `Surprise(%)` column, which is
    inconsistently a fraction or a percentage across yfinance versions — a
    reading that would silently render a 4.7% beat as 0.05%.
    """
    if eps_actual is None or eps_estimate is None or eps_estimate == 0:
        return None
    return (eps_actual - eps_estimate) / abs(eps_estimate) * 100.0


def _report_moment_passed(info: Dict, day: date, tz: str, now: datetime) -> bool:
    """
    True when the time Yahoo announced for a report on `day` has actually gone by.

    A date alone cannot tell a report that has happened from one that is hours
    away: a company reporting at 08:30 must not read as "reported" at midnight.
    """
    for key in EARNINGS_TIMESTAMP_KEYS:
        moment = epoch_to_market_datetime(info.get(key), tz)
        if moment is not None and moment.date() == day and moment <= now:
            return True
    return False


def recent_earnings_event(
    symbol: str,
    info: Dict,
    today: Optional[date] = None,
    lookback_days: int = REPORTED_LOOKBACK_DAYS,
    now: Optional[datetime] = None,
) -> Optional[dict]:
    """
    The quarter this company has just reported, with what it actually printed.

    Carries the same shape as `next_earnings_event` under `status="reported"`, so
    a client can lay both on one timeline. `eps_actual` is None in the window
    between the release and Yahoo attaching the figure — the report still belongs
    on the calendar then, it just has nothing to compare yet.

    A merely *projected* past date is not evidence that anything was reported, so
    a date Yahoo flagged as an estimate only counts once it appears in the
    earnings history table.

    `today`/`now` default to the symbol's own exchange clock (see `market_today`).
    """
    if not info:
        return None

    tz = market_timezone(info)
    if now is None:
        now = datetime.now(ZoneInfo(tz))
    if today is None:
        today = now.date()
    earliest = today - timedelta(days=max(0, lookback_days))

    history = earnings_history(info)
    candidates = set()
    for key in history:
        try:
            candidates.add(date.fromisoformat(key))
        except (TypeError, ValueError):
            continue
    for key in EARNINGS_TIMESTAMP_KEYS:
        reported = epoch_to_market_date(info.get(key), tz)
        if reported is not None:
            candidates.add(reported)

    past = [d for d in candidates if earliest <= d <= today]
    if not past:
        return None

    def _printed(day: date) -> bool:
        row = history.get(str(day))
        return isinstance(row, dict) and row.get("eps_actual") is not None

    # A day Yahoo holds figures against is the strongest evidence of a report,
    # and it beats a bare timestamp: an *estimated* `earningsTimestamp` can sit a
    # day past the real print (ADP printed on the 28th while the blob still said
    # the 29th), so taking the latest candidate date would pick the empty day and
    # bury the result the panel exists to show.
    printed = [d for d in past if _printed(d)]
    if printed:
        report_date = max(printed)
        row = history.get(str(report_date))
    else:
        # Nothing printed yet, so the report only counts once its announced
        # moment has passed. Days already over need no such proof; today's does.
        elapsed = [
            d for d in past if d < today or _report_moment_passed(info, d, tz, now)
        ]
        if not elapsed:
            return None
        report_date = max(elapsed)
        row = history.get(str(report_date))
        if row is None and info.get("isEarningsDateEstimate"):
            return None

    row = row if isinstance(row, dict) else {}
    eps_actual = row.get("eps_actual")
    eps_estimate = row.get("eps_estimate")

    return {
        "symbol": symbol,
        "name": company_name(info),
        "earnings_date": str(report_date),
        "status": "reported",
        "eps_actual": eps_actual,
        "eps_estimate": eps_estimate,
        "surprise_pct": _surprise_pct(eps_actual, eps_estimate),
        "market_timezone": tz,
    }


def next_dividend_event(
    symbol: str, info: Dict, today: Optional[date] = None
) -> Optional[dict]:
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
        [
            d
            for d in (pay_date, ex_date, epoch_to_date(info.get("lastDividendDate")))
            if d
        ],
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
            projected = (
                pd.Timestamp(projected) + pd.DateOffset(months=freq_months)
            ).date()
        else:
            return None
    except (ValueError, OverflowError) as e:
        logging.debug(f"Dividend projection failed for {symbol}: {e}")
        return None

    return _event(projected, None, "estimated")


def upcoming_events(
    symbol: str, info: Dict, today: Optional[date] = None, horizon_days: int = 365
) -> dict:
    """Every event type for one symbol, as consumed by the stock-detail Overview tab."""
    if not info:
        return {"earnings": None, "recent_earnings": None, "dividend": None}
    if today is None:
        today = market_today(info)
    horizon_end = today + timedelta(days=horizon_days)

    reported = recent_earnings_event(symbol, info, today)
    scheduled = next_earnings_event(symbol, info, today, horizon_end)

    # Yahoo leaves `earningsTimestamp` on the report it has just been through
    # until the next quarter is scheduled, so a company that reported this
    # morning reads as both "reported today" and "reporting today". It is one
    # event, and it has happened — so look for the next one strictly *after* the
    # reported day rather than dropping the scheduled side outright, which would
    # also hide a next date the company has already announced.
    if (
        reported
        and scheduled
        and reported["earnings_date"] == scheduled["earnings_date"]
    ):
        day_after = date.fromisoformat(reported["earnings_date"]) + timedelta(days=1)
        scheduled = next_earnings_event(symbol, info, day_after, horizon_end)

    return {
        "earnings": scheduled,
        "recent_earnings": reported,
        "dividend": next_dividend_event(symbol, info, today),
    }
