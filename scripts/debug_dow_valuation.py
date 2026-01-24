
import sys
import os
import logging
import yfinance as yf
import pandas as pd

# Convert to absolute path to find src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', 'src')
sys.path.append(src_path)

from financial_ratios import get_comprehensive_intrinsic_value

# Setup logging
logging.basicConfig(level=logging.INFO)

def debug_dow():
    symbol = "DOW"
    print(f"Fetching data for {symbol}...")
    stock = yf.Ticker(symbol)
    
    info = stock.info
    print(f"Info keys: {list(info.keys())}")
    print(f"Current Price: {info.get('currentPrice')}")
    print(f"EPS: {info.get('trailingEps')}")
    print(f"FCF: {info.get('freeCashflow')}")
    print(f"Revenue: {info.get('totalRevenue')}")
    print(f"Shares: {info.get('sharesOutstanding')}")
    
    financials = stock.financials
    bs = stock.balance_sheet
    cf = stock.cashflow
    
    print("\n--- Financials (Top 5 rows) ---")
    if not financials.empty:
        print(financials.head())
    else:
        print("Empty Financials")
        
    print("\n--- Balance Sheet (Top 5 rows) ---")
    if not bs.empty:
        print(bs.head())
    else:
        print("Empty BS")
        
    print("\n--- Cash Flow (Top 5 rows) ---")
    if not cf.empty:
        print(cf.head())
    else:
        print("Empty CF")

    print("\n--- Calculation Result ---")
    try:
        res = get_comprehensive_intrinsic_value(info, financials, bs, cf)
        import json
        # Helper to handle non-serializable objects if any (like numpy types)
        def default(o):
            if isinstance(o, (pd.Timestamp, pd.Period)):
                return str(o)
            import numpy as np
            if isinstance(o, (np.int_, np.intc, np.intp, np.int8,
                np.int16, np.int32, np.int64, np.uint8,
                np.uint16, np.uint32, np.uint64)):
                return int(o)
            elif isinstance(o, (np.float_, np.float16, np.float32, 
                np.float64)):
                return float(o)
            elif isinstance(o, (np.ndarray,)): 
                return o.tolist()
            raise TypeError
            
        print(json.dumps(res, indent=2, default=default))
    except Exception as e:
        print(f"Error calculating: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_dow()
