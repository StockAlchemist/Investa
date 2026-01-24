import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from market_data import get_shared_mdp
from financial_ratios import estimate_growth_rate, calculate_intrinsic_value_dcf

def debug_aes():
    logging.basicConfig(level=logging.INFO)
    mdp = get_shared_mdp()
    symbol = "AES"
    
    info = mdp.get_fundamental_data(symbol, force_refresh=True)
    financials = mdp.get_financials(symbol, "annual")
    balance_sheet = mdp.get_balance_sheet(symbol, "annual")
    cashflow = mdp.get_cashflow(symbol, "annual")
    
    print(f"\n--- Debugging {symbol} ---")
    if info:
        print(f"Analyst EE present: {'_earnings_estimate' in info}")
        if '_earnings_estimate' in info:
            print(f"Analyst EE: {info['_earnings_estimate']}")
        
    growth_rate = estimate_growth_rate(financials, ticker_info=info, item_name="Net Income")
    print(f"\nCalculated CAGR Growth Rate: {growth_rate:.4%}")
    
    wacc_discount = 0.10 # dummy
    dcf = calculate_intrinsic_value_dcf(info, financials, balance_sheet, cashflow, discount_rate=wacc_discount, growth_rate=growth_rate)
    
    if "intrinsic_value" in dcf:
        print(f"DCF Intrinsic Value: ${dcf['intrinsic_value']:,.2f}")
        print(f"Parameters: {dcf['parameters']}")
    else:
        print(f"DCF Error: {dcf['error']}")

if __name__ == "__main__":
    debug_aes()
