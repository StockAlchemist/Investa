import sys
import os
import pandas as pd
import logging

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from financial_ratios import get_comprehensive_intrinsic_value

def test_intrinsic_value():
    logging.basicConfig(level=logging.INFO)
    
    # Mock ticker_info for a stable company like MSFT or AAPL
    ticker_info = {
        "symbol": "AAPL",
        "currentPrice": 180.0,
        "marketCap": 2800000000000,
        "totalCash": 60000000000,
        "totalDebt": 110000000000,
        "sharesOutstanding": 15443000000,
        "trailingEps": 6.13,
        "beta": 1.28,
        "freeCashflow": 100000000000 # 100B
    }
    
    # Simple financials (minimal for growth estimation)
    financials = pd.DataFrame({
        "2023-09-30": [96995000000],
        "2022-09-30": [99803000000],
        "2021-09-30": [94680000000]
    }, index=["Net Income"])
    
    # Call the logic
    results = get_comprehensive_intrinsic_value(ticker_info, financials)
    
    print("\n--- Intrinsic Value Test Results ---")
    print(f"Current Price: ${results['current_price']}")
    
    dcf = results['models']['dcf']
    if "intrinsic_value" in dcf:
        print(f"DCF Intrinsic Value: ${dcf['intrinsic_value']:.2f}")
        print(f"  Discount Rate: {dcf['parameters']['discount_rate']:.2%}")
        print(f"  Growth Rate: {dcf['parameters']['growth_rate']:.2%}")
    else:
        print(f"DCF Error: {dcf.get('error')}")
        
    graham = results['models']['graham']
    if "intrinsic_value" in graham:
        print(f"Graham Intrinsic Value: ${graham['intrinsic_value']:.2f}")
    else:
        print(f"Graham Error: {graham.get('error')}")
        
    if "average_intrinsic_value" in results:
        print(f"Average Intrinsic Value: ${results['average_intrinsic_value']:.2f}")
        print(f"Margin of Safety: {results.get('margin_of_safety_pct', 0):.2f}%")

if __name__ == "__main__":
    test_intrinsic_value()
