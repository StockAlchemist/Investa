import yfinance as yf
import json
import pandas as pd

def check_symbol(symbol):
    print(f"--- Checking {symbol} ---")
    t = yf.Ticker(symbol)
    print("Available Attributes:")
    print([attr for attr in dir(t) if not attr.startswith("_")])

    print("\n--- EARNINGS ESTIMATE ---")
    try:
        ee = t.earnings_estimate
        if not ee.empty:
            print(ee)
    except:
        pass

    print("\n--- REVENUE ESTIMATE ---")
    try:
        re = t.revenue_estimate
        if not re.empty:
            print(re)
    except:
        pass

    print("\n--- INFO GROWTH FIELDS ---")
    try:
        info = t.info
        for k in ["longTermGrowth", "revenueGrowth", "earningsGrowth", "earningsQuarterlyGrowth"]:
            print(f"{k}: {info.get(k)}")
    except:
        pass

if __name__ == "__main__":
    for s in ["AAPL", "MSFT", "NVDA", "SQ", "AES", "TKO"]:
        check_symbol(s)
