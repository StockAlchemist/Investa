import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.abspath("src"))

from financial_ratios import get_comprehensive_intrinsic_value


def test_valuation_overrides():
    # A coherent mid-cap: $8B FCF against $20B debt and $6B cash. The previous
    # fixture paired $1B of FCF with $20B of debt, so its discounted cash flows
    # never covered net debt and the DCF returned a *negative* value per share
    # — which the old code emitted and this test silently accepted. The models
    # now refuse that case, so the fixture has to describe a company that can
    # actually be valued.
    ticker_info = {
        "currentPrice": 150.0,
        "trailingEps": 5.0,
        "freeCashflow": 8000000000,
        "marketCap": 150000000000,
        "totalCash": 6000000000,
        "totalDebt": 20000000000,
        "sharesOutstanding": 1000000000,
        "shortName": "Test Stock",
    }

    # Simple financials
    financials = pd.DataFrame(
        {"2023-12-31": [5.0, 4.0, 3.0]},
        index=["Net Income", "Operating Income", "Gross Profit"],
    )

    print("--- Test 1: No Overrides ---")
    res_no_ov = get_comprehensive_intrinsic_value(ticker_info, financials)
    dcf_val_1 = res_no_ov["models"]["dcf"]["intrinsic_value"]
    graham_val_1 = res_no_ov["models"]["graham"]["intrinsic_value"]
    print(f"DCF Intrinsic Value: ${dcf_val_1:.2f}")
    print(f"Graham Intrinsic Value: ${graham_val_1:.2f}")

    print("\n--- Test 2: With Overrides (Higher Growth) ---")
    overrides = {
        "dcf_growth_rate": 0.20,  # 20% growth
        "graham_growth_rate": 15.0,  # 15% growth
    }
    res_with_ov = get_comprehensive_intrinsic_value(
        ticker_info, financials, overrides=overrides
    )
    dcf_val_2 = res_with_ov["models"]["dcf"]["intrinsic_value"]
    graham_val_2 = res_with_ov["models"]["graham"]["intrinsic_value"]
    print(f"DCF Intrinsic Value (20% growth): ${dcf_val_2:.2f}")
    print(f"Graham Intrinsic Value (15% growth): ${graham_val_2:.2f}")

    assert dcf_val_2 > dcf_val_1, "DCF value should increase with higher growth"
    assert graham_val_2 > graham_val_1, (
        "Graham value should increase with higher growth"
    )

    # A per-share intrinsic value is a price; it is never negative. Models that
    # cannot produce one must report an error instead of a number.
    for res in (res_no_ov, res_with_ov):
        for name, model in res["models"].items():
            if "intrinsic_value" in model and model["intrinsic_value"] is not None:
                assert model["intrinsic_value"] > 0, (
                    f"{name} returned a non-positive value"
                )
        assert (res.get("average_intrinsic_value") or 1) > 0

    print("\nSUCCESS: Overrides are correctly applied in the logic.")


if __name__ == "__main__":
    test_valuation_overrides()
