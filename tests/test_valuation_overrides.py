import sys
import os
import pandas as pd
from typing import Dict, Any, Optional

# Add src to path
sys.path.append(os.path.abspath("src"))

from financial_ratios import get_comprehensive_intrinsic_value

def test_valuation_overrides():
    # Mock data
    ticker_info = {
        "currentPrice": 150.0,
        "trailingEps": 5.0,
        "freeCashflow": 1000000000,
        "marketCap": 150000000000,
        "totalDebt": 20000000000,
        "sharesOutstanding": 1000000000,
        "shortName": "Test Stock"
    }
    
    # Simple financials
    financials = pd.DataFrame({
        "2023-12-31": [5.0, 4.0, 3.0]
    }, index=["Net Income", "Operating Income", "Gross Profit"])
    
    print("--- Test 1: No Overrides ---")
    res_no_ov = get_comprehensive_intrinsic_value(ticker_info, financials)
    dcf_val_1 = res_no_ov["models"]["dcf"]["intrinsic_value"]
    graham_val_1 = res_no_ov["models"]["graham"]["intrinsic_value"]
    print(f"DCF Intrinsic Value: ${dcf_val_1:.2f}")
    print(f"Graham Intrinsic Value: ${graham_val_1:.2f}")
    
    print("\n--- Test 2: With Overrides (Higher Growth) ---")
    overrides = {
        "dcf_growth_rate": 0.20,  # 20% growth
        "graham_growth_rate": 15.0 # 15% growth
    }
    res_with_ov = get_comprehensive_intrinsic_value(ticker_info, financials, overrides=overrides)
    dcf_val_2 = res_with_ov["models"]["dcf"]["intrinsic_value"]
    graham_val_2 = res_with_ov["models"]["graham"]["intrinsic_value"]
    print(f"DCF Intrinsic Value (20% growth): ${dcf_val_2:.2f}")
    print(f"Graham Intrinsic Value (15% growth): ${graham_val_2:.2f}")
    
    assert dcf_val_2 > dcf_val_1, "DCF value should increase with higher growth"
    assert graham_val_2 > graham_val_1, "Graham value should increase with higher growth"
    print("\nSUCCESS: Overrides are correctly applied in the logic.")

if __name__ == "__main__":
    test_valuation_overrides()
