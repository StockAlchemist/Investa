
import yfinance as yf
import json
from datetime import date

def default_json(obj):
    if isinstance(obj, date):
        return obj.isoformat()
    return str(obj)

symbol = "AAPL"
ticker = yf.Ticker(symbol)

print(f"--- Fetching Info for {symbol} ---")
try:
    info = ticker.info
    # Print relevant keys
    keys_to_check = [k for k in info.keys() if 'earnings' in k.lower() or 'date' in k.lower()]
    print("Relevant Info Keys:", keys_to_check)
    for k in keys_to_check:
        print(f"{k}: {info[k]}")
except Exception as e:
    print(f"Error fetching info: {e}")

print(f"\n--- Fetching Calendar for {symbol} ---")
try:
    cal = ticker.calendar
    print(f"Calendar Type: {type(cal)}")
    print(cal)
except Exception as e:
    print(f"Error fetching calendar: {e}")
