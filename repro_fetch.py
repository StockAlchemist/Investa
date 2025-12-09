
import yfinance as yf
from datetime import date, timedelta
import pandas as pd

today = date.today()
tomorrow = today + timedelta(days=1)

print(f"Attempting to fetch data for range: {today} -> {tomorrow}")

try:
    data = yf.download(
        tickers=['AAPL'],
        start=today,
        end=tomorrow,
        progress=False,
        auto_adjust=True
    )
    print("Data fetched:")
    print(data)
except Exception as e:
    print(f"Caught exception: {e}")
