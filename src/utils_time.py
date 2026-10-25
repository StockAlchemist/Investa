import pandas as pd
from datetime import date, datetime, timedelta
import time as _time
import pandas_market_calendars as mcal
import logging

# Cache for the NYSE calendar
_NYSE_CAL = None

# Per-process cache for calendar.schedule() results (key → (df, timestamp)).
# Avoids recomputing the full NYSE holiday table on every call (~0.2s).
_SCHEDULE_CACHE: dict = {}
_SCHEDULE_CACHE_TTL = 60  # seconds


def get_nyse_calendar():
    global _NYSE_CAL
    if _NYSE_CAL is None:
        _NYSE_CAL = mcal.get_calendar("NYSE")
    return _NYSE_CAL


def _cached_schedule(cal, start_date, end_date):
    """Return cal.schedule() with a short-lived per-process cache."""
    key = (start_date, end_date)
    now = _time.monotonic()
    cached = _SCHEDULE_CACHE.get(key)
    if cached is not None:
        cached_df, ts = cached
        if now - ts < _SCHEDULE_CACHE_TTL:
            return cached_df
    result = nyse_schedule(start_date, end_date)
    _SCHEDULE_CACHE[key] = (result, now)
    return result


# One NYSE schedule over a wide window, built once per process. Every
# cal.schedule() call rebuilds the holiday table (~30 ms warm, ~130 ms cold),
# and a summary or history request made a dozen of them; a slice of this frame
# is the same rows for 0.1 ms. The calendar's rules are fixed at import, so the
# frame never goes stale; only the window's end moves with the clock.
_FULL_SCHEDULE_START = date(1970, 1, 1)
_FULL_SCHEDULE_YEARS_AHEAD = 5
_FULL_SCHEDULE = None  # (frame, first day, last day, empty frame)


def _full_schedule():
    global _FULL_SCHEDULE
    today = get_est_today()
    if _FULL_SCHEDULE is None or _FULL_SCHEDULE[2] < today + timedelta(days=366):
        end = date(today.year + _FULL_SCHEDULE_YEARS_AHEAD, 12, 31)
        cal = get_nyse_calendar()
        frame = cal.schedule(start_date=_FULL_SCHEDULE_START, end_date=end)
        # A range with no session comes back with object columns, not the
        # UTC stamps a slice would carry; keep the calendar's own empty frame.
        empty = cal.schedule(start_date=date(2025, 12, 25), end_date=date(2025, 12, 25))
        _FULL_SCHEDULE = (frame, _FULL_SCHEDULE_START, end, empty)
    return _FULL_SCHEDULE


def nyse_schedule(start_date, end_date) -> pd.DataFrame:
    """
    Exactly ``get_nyse_calendar().schedule(start_date=..., end_date=...)``.

    Answered from the cached window for plain dates inside it; anything else
    (a reversed range, which raises, a Timestamp or string, a day outside the
    window) goes to the calendar itself.
    """
    if type(start_date) is date and type(end_date) is date and start_date <= end_date:
        frame, first, last, empty = _full_schedule()
        if first <= start_date and end_date <= last:
            rows = frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
            return (rows if len(rows) else empty).copy()
    return get_nyse_calendar().schedule(start_date=start_date, end_date=end_date)


def get_est_now() -> pd.Timestamp:
    """
    Returns the current timestamp in US/Eastern (America/New_York).
    Useful for ensuring all logic operates in EST regardless of server location.
    """
    return pd.Timestamp.now(tz="America/New_York")


def get_est_today() -> date:
    """
    Returns the current date in US/Eastern (America/New_York).
    Use this instead of date.today() to prevent 'tomorrow' triggering
    too early for users in eastern timezones (e.g. Asia).
    """
    return get_est_now().date()


def get_latest_trading_date() -> date:
    """
    Returns the 'effective' trading date for reporting purposes.
    - If Market is OPEN (Weekday >= 9:30 AM EST): Returns Today.
    - If Market is PRE-MARKET (Weekday < 9:30 AM EST): Returns Yesterday (or last Friday).
    - If Market is CLOSED (Weekend or Holiday): Returns last valid trading day.

    This ensures that during pre-market hours (e.g., 2 AM EST), the dashboard
    shows the 'Final Close' of the previous day rather than a blank 'Today' with 0% gain.
    """
    now_est = get_est_now()
    today = now_est.date()

    cal = get_nyse_calendar()

    # Check if market is open today or has been open today
    # We look at the last 5 days to be safe (covering long weekends)
    schedule = _cached_schedule(
        cal, start_date=today - timedelta(days=10), end_date=today
    )

    if schedule.empty:
        # Should not happen if we look back 10 days, but safety first
        return today - timedelta(days=1)

    # Market open time today if it's a trading day
    today_ts = pd.Timestamp(today)
    if today_ts in schedule.index:
        market_open = schedule.loc[today_ts].market_open
        # Convert market_open to EST for comparison
        market_open_est = market_open.tz_convert("America/New_York")

        if now_est >= market_open_est:
            # It's trading hours or post-market on a valid trading day
            return today

    # If we are here, either today is NOT a trading day (Holiday/Weekend)
    # OR it's pre-market on a valid trading day.
    # In both cases, we want the PREVIOUS trading day.

    # Get all trading days up to yesterday
    valid_days = schedule.index[schedule.index.date < today]
    if not valid_days.empty:
        return valid_days[-1].date()

    return today - timedelta(days=1)  # Ultimate fallback


def is_market_open() -> bool:
    """
    Checks if the US stock market is currently open (Monday-Friday, 09:30 - 16:00 EST).
    Accounts for NYSE holiday schedule via pandas_market_calendars.
    """
    now_est = get_est_now()
    today = now_est.date()

    cal = get_nyse_calendar()

    # schedule covers the specific times (market_open, market_close)
    try:
        schedule = _cached_schedule(cal, start_date=today, end_date=today)
        if schedule.empty:
            return False

        # Get market open/close for today
        row = schedule.iloc[0]
        market_open = row.market_open.tz_convert("America/New_York")
        market_close = row.market_close.tz_convert("America/New_York")

        return market_open <= now_est <= market_close
    except Exception as e:
        logging.warning(f"Error checking market open status in utils_time: {e}")
        # Fallback to simple weekday logic if calendar fails
        if now_est.weekday() >= 5:
            return False
        market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_est.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_est <= market_close


def is_tradable_day(dt: date | datetime | pd.Timestamp) -> bool:
    """
    Checks if a given date is a valid trading day on the NYSE.
    Skips weekends and market holidays.
    """
    if isinstance(dt, (datetime, pd.Timestamp)):
        check_date = dt.date()
    else:
        check_date = dt

    cal = get_nyse_calendar()
    # schedule() returns a DataFrame with trading days as the index
    # We check if the specific date exists in the schedule
    schedule = _cached_schedule(cal, start_date=check_date, end_date=check_date)
    return not schedule.empty
