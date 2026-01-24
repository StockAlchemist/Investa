import sys
import os
import json
import logging
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

import config
from market_data import get_shared_mdp
from financial_ratios import get_comprehensive_intrinsic_value, calculate_intrinsic_value_dcf, calculate_intrinsic_value_graham

def debug_stock(symbol):
    print(f"\n--- Debugging {symbol} ---")
    mdp = get_shared_mdp()
    info = mdp.get_fundamental_data(symbol)
    financials = mdp.get_financials(symbol, "annual")
    balance_sheet = mdp.get_balance_sheet(symbol, "annual")
    cashflow = mdp.get_cashflow(symbol, "annual")
    
    if not info:
        print(f"Failed to fetch info for {symbol}")
        return

    print(f"Current Price: {info.get('currentPrice')}")
    print(f"FCF: {info.get('freeCashflow')}")
    print(f"Shares: {info.get('sharesOutstanding')}")
    print(f"Revenue: {info.get('totalRevenue')}")
    print(f"Net Income Growth: {info.get('earningsGrowth')}")
    
    # Run valuation
    results = get_comprehensive_intrinsic_value(
        info, financials, balance_sheet, cashflow
    )
    
    print("\nResults:")
    print(json.dumps(results, indent=2, default=str))

if __name__ == "__main__":
    # Test symbols from screenshot
    symbols = ["BMY", "EXE", "LUV", "DOC", "EG", "TKO", "LYV"]
    for s in symbols:
        debug_stock(s)
