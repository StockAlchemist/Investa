import pandas as pd
from datetime import date, datetime, timedelta
import pandas_market_calendars as mcal
import logging

# Cache for the NYSE calendar
_NYSE_CAL = None

def get_nyse_calendar():
    global _NYSE_CAL
    if _NYSE_CAL is None:
        _NYSE_CAL = mcal.get_calendar('NYSE')
    return _NYSE_CAL

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
    schedule = cal.schedule(start_date=today - timedelta(days=10), end_date=today)
    
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
        
    return today - timedelta(days=1) # Ultimate fallback

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
        schedule = cal.schedule(start_date=today, end_date=today)
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
    schedule = cal.schedule(start_date=check_date, end_date=check_date)
    return not schedule.empty
