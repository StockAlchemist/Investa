import pandas as pd
from datetime import date, datetime, timedelta

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
    - If Market is CLOSED (Weekend): Returns last Friday.
    
    This ensures that during pre-market hours (e.g., 2 AM EST), the dashboard
    shows the 'Final Close' of the previous day rather than a blank 'Today' with 0% gain.
    """
    now_est = get_est_now()
    today = now_est.date()
    
    # Check for Weekend
    # weekday(): Mon=0, Sun=6
    if today.weekday() == 5: # Saturday
        return today - timedelta(days=1)
    elif today.weekday() == 6: # Sunday
        return today - timedelta(days=2)
        
    # Weekday Logic
    # If before 9:30 AM EST, treat as "Yesterday" (or Friday if Monday)
    market_open = now_est.replace(hour=9, minute=30, second=0, microsecond=0)
    
    if now_est < market_open:
        # It's pre-market. Go back to previous trading day.
        offset = 1
        if today.weekday() == 0: # Monday -> Go back to Friday (3 days)
            offset = 3
        elif today.weekday() == 6: # Sunday (should be caught above, but safety) -> Friday
             offset = 2
        return today - timedelta(days=offset)
    
    # Otherwise, it's trading hours or post-market (same day)
    return today
