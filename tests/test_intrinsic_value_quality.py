"""Guards on the intrinsic-value models' output contract.

These encode the failures found by `scripts/intrinsic_value_lab.py` over the
local fundamentals cache, so a regression shows up as a red test rather than as
a strange number in the UI three clients away.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)

import financial_ratios as fr  # noqa: E402


def _statements(fcf_by_year, revenue=1e10, ebit=1.2e9):
    """Build the three annual statements from a per-year FCF series."""
    years = [f"{2019 + i}-12-31" for i in range(len(fcf_by_year))]
    cashflow = pd.DataFrame(
        {y: [f + 5e8, -5e8] for y, f in zip(years, fcf_by_year)},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )
    financials = pd.DataFrame(
        {y: [revenue, ebit, 8e8, 2e8, 1e9] for y in years},
        index=[
            "Total Revenue",
            "Operating Income",
            "Net Income",
            "Tax Provision",
            "Pretax Income",
        ],
    )
    balance = pd.DataFrame(
        {y: [2e10] for y in years}, index=["Total Stockholder Equity"]
    )
    return financials, balance, cashflow


def _info(**over):
    base = {
        "symbol": "TEST",
        "currentPrice": 100.0,
        "regularMarketPrice": 100.0,
        "marketCap": 5e10,
        "totalCash": 3e9,
        "totalDebt": 4e9,
        "sharesOutstanding": 5e8,
        "trailingEps": 6.0,
        "totalRevenue": 1e10,
        "beta": 1.1,
        "quoteType": "EQUITY",
    }
    base.update(over)
    return base


# --- growth ---------------------------------------------------------------


def test_growth_is_shrunk_toward_base_rate():
    """A 90%-growth history must not become a 90% ten-year projection."""
    fin = pd.DataFrame(
        {f"{2019 + i}-12-31": [1e8 * (1.9**i)] for i in range(5)},
        index=["Net Income"],
    )
    res = fr.blended_growth_estimate(fin)
    assert res["growth"] <= fr.MAX_PROJECTED_GROWTH
    assert res["growth"] < res["raw_growth"], "shrinkage must pull toward the base rate"


def test_growth_band_is_enforced_for_collapsing_earnings():
    fin = pd.DataFrame(
        {f"{2019 + i}-12-31": [1e9 * (0.5**i)] for i in range(5)},
        index=["Net Income"],
    )
    res = fr.blended_growth_estimate(fin)
    assert res["growth"] >= fr.MIN_PROJECTED_GROWTH


def test_no_growth_evidence_falls_back_to_base_rate():
    res = fr.blended_growth_estimate(None, ticker_info={})
    assert res["growth"] == fr.BASE_RATE_GROWTH
    assert res["signals"] == {}


def test_estimate_growth_rate_still_measures_raw_history():
    """The raw estimator is a measurement and must stay unshrunk."""
    fin = pd.DataFrame(
        {"2022-12-31": [100.0], "2023-12-31": [120.0]}, index=["Net Income"]
    )
    assert fr.estimate_growth_rate(fin, item_name="Net Income") == pytest.approx(
        0.20, abs=0.01
    )


# --- normalized cash flow --------------------------------------------------


def test_base_fcf_uses_median_not_latest_year():
    """One exceptional year must not set a decade of projections."""
    fin, bal, cf = _statements([1e9, 1e9, 1e9, 1e9, 3e9])
    res = fr.normalized_base_fcf(_info(), fin, cf)
    assert res["normalized"] is True
    assert res["fcf"] == pytest.approx(1e9), "median should ignore the 3B outlier"


def test_one_implausible_year_does_not_discard_the_history():
    """A 90%-margin year is dropped; the four sane years still normalize."""
    fin, bal, cf = _statements([1e9, 1e9, 1e9, 1e9, 9e9])  # 9e9 / 1e10 revenue = 90%
    res = fr.normalized_base_fcf(_info(), fin, cf)
    assert res["normalized"] is True
    assert res["fcf"] == pytest.approx(1e9)


def test_loss_making_years_are_kept_in_the_margin():
    """The old filter dropped negative years, biasing the margin upward."""
    fin, bal, cf = _statements([-2e9, -1e9, 1e8, 2e8, 1e8])
    margin = fr.estimate_fcf_margin(fin, cf)
    assert margin < 0.05, "cash-burning years must drag the normalized margin down"


def test_chronic_cash_burner_is_refused_not_fabricated():
    """No revenue-based DCF for a company that never converted sales to cash."""
    fin, bal, cf = _statements([-2e9, -3e9, -1e9, -2e9, -1e9])
    res = fr.calculate_intrinsic_value_dcf(_info(), fin, bal, cf)
    assert "intrinsic_value" not in res
    assert "error" in res


# --- output contract -------------------------------------------------------


def test_intrinsic_value_is_never_negative():
    """Net debt above discounted cash flows is an error, not a negative price."""
    fin, bal, cf = _statements([2e8, 2e8, 2e8, 2e8, 2e8])
    res = fr.calculate_intrinsic_value_dcf(
        _info(totalDebt=8e10, totalCash=0), fin, bal, cf
    )
    assert "intrinsic_value" not in res
    assert "Net debt" in res["error"]


def test_penny_stock_is_ineligible():
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.get_comprehensive_intrinsic_value(
        _info(currentPrice=0.4, regularMarketPrice=0.4), fin, bal, cf, iterations=200
    )
    assert res["valuation_status"] == "ineligible"
    assert res["average_intrinsic_value"] is None


def test_micro_cap_is_ineligible():
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.get_comprehensive_intrinsic_value(
        _info(marketCap=1e7), fin, bal, cf, iterations=200
    )
    assert res["valuation_status"] == "ineligible"


def test_output_is_clamped_to_a_credible_band():
    """Nothing reaches the UI at 40x price, as it did before."""
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.get_comprehensive_intrinsic_value(
        _info(currentPrice=2.0, regularMarketPrice=2.0), fin, bal, cf, iterations=200
    )
    iv = res["average_intrinsic_value"]
    assert iv is not None
    assert iv <= fr.MAX_IV_TO_PRICE * 2.0 + 1e-6
    assert res["valuation_status"] == "clamped"


def test_valuation_is_not_anchored_to_price():
    """The blend must not change when only the quoted price moves.

    The removed rule picked whichever model sat nearer the market price, so the
    same company at a different price produced a different 'intrinsic' value.
    """
    fin, bal, cf = _statements([1e9] * 5)
    cheap = fr.get_comprehensive_intrinsic_value(
        _info(currentPrice=40.0, regularMarketPrice=40.0), fin, bal, cf, iterations=200
    )
    rich = fr.get_comprehensive_intrinsic_value(
        _info(currentPrice=90.0, regularMarketPrice=90.0), fin, bal, cf, iterations=200
    )
    assert cheap["valuation_status"] != "clamped"
    assert rich["valuation_status"] != "clamped"
    assert cheap["average_intrinsic_value"] == pytest.approx(
        rich["average_intrinsic_value"], rel=1e-6
    )
    # The margin of safety, by contrast, *must* move with price — that is the
    # whole point of comparing a fixed estimate against a changing quote.
    assert cheap["margin_of_safety_pct"] > rich["margin_of_safety_pct"]


def test_graham_does_not_masquerade_as_book_value():
    """Negative EPS must fail Graham, not silently return book value."""
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.calculate_intrinsic_value_graham(_info(trailingEps=-3.0), fin, bal)
    assert "intrinsic_value" not in res
    assert res["error"] == "Negative or missing EPS"
    assert res["diagnostics"]["book_value_per_share"] == pytest.approx(40.0)


def test_epv_is_reported_as_a_floor_not_blended_in():
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.get_comprehensive_intrinsic_value(_info(), fin, bal, cf, iterations=200)
    assert "epv" in res["models"]
    assert set(res["model_weights"]) <= {"dcf", "graham"}, (
        "EPV must stay out of the blend"
    )
    if res["models"]["epv"].get("intrinsic_value"):
        assert res["earnings_power_floor"] > 0


def test_terminal_value_share_is_reported():
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.calculate_intrinsic_value_dcf(_info(), fin, bal, cf)
    share = res["parameters"]["terminal_value_share"]
    assert 0.0 < share < 1.0


# --- Monte Carlo -----------------------------------------------------------


def test_monte_carlo_models_downside():
    """The old floor of 0% growth meant the 'bear' case had no bear in it."""
    mc = fr.run_monte_carlo_dcf(
        _info(), base_fcf=1e9, base_growth=0.05, base_discount=0.10, iterations=4000
    )
    assert mc["bear"] < mc["base"] < mc["bull"]


def test_monte_carlo_uncertainty_does_not_vanish_at_low_growth():
    """Relative sigma gave a 2%-growth firm near-certainty; absolute sigma doesn't."""
    low = fr.run_monte_carlo_dcf(
        _info(), base_fcf=1e9, base_growth=0.02, base_discount=0.10, iterations=4000
    )
    spread = (low["bull"] - low["bear"]) / low["base"]
    assert spread > 0.25, f"distribution is implausibly tight: {spread:.2%}"


def test_wacc_is_banded():
    info = _info(beta=9.0, marketCap=1e8)
    wacc = fr.calculate_wacc(info, None, None)["wacc"]
    assert fr.MIN_DISCOUNT_RATE <= wacc <= fr.MAX_DISCOUNT_RATE


def test_small_caps_carry_a_size_premium():
    big = fr.calculate_wacc(_info(marketCap=5e11), None, None)
    small = fr.calculate_wacc(_info(marketCap=2e8), None, None)
    assert small["size_premium"] > big["size_premium"]


def test_etf_still_values_at_nav():
    res = fr.get_comprehensive_intrinsic_value(
        _info(quoteType="ETF", navPrice=101.0), None, None, None, iterations=100
    )
    assert res["average_intrinsic_value"] == pytest.approx(101.0)
    assert res["valuation_status"] == "nav"


def test_no_model_reports_a_reason():
    """A refusal must say why, so the UI can explain the blank."""
    res = fr.get_comprehensive_intrinsic_value(
        _info(trailingEps=-5.0, freeCashflow=None),
        *_statements([-1e9] * 5, ebit=-5e8)[:1],
        None,
        _statements([-1e9] * 5, ebit=-5e8)[2],
        iterations=200,
    )
    assert res["average_intrinsic_value"] is None
    assert res["valuation_status"] == "no_model"
    assert res["valuation_note"]


def test_results_are_json_safe():
    """NaN/Inf must never reach the API layer."""
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.get_comprehensive_intrinsic_value(_info(), fin, bal, cf, iterations=200)
    for key in ("average_intrinsic_value", "margin_of_safety_pct", "model_spread_pct"):
        val = res.get(key)
        if val is not None:
            assert np.isfinite(val), f"{key} is not finite"


# --- DDM, Lynch, Blume Beta, Aliases & Sector Blending Tests ---------------


def test_ddm_calculation_and_sustainability():
    """Test Multi-stage DDM calculation and payout sustainability gate."""
    # Sane dividend payer
    info = _info(dividendRate=3.0, payoutRatio=0.45, currentPrice=80.0)
    fin, bal, cf = _statements([1e9] * 5)
    res = fr.calculate_intrinsic_value_ddm(info, fin, cf, bal)
    assert "intrinsic_value" in res
    assert res["intrinsic_value"] > 0
    assert res["parameters"]["base_dividend"] == 3.0
    assert res["parameters"]["mid_year_discounting"] is True

    # Unsustainable payout (>150%) must fail DDM
    bad_info = _info(dividendRate=8.0, payoutRatio=1.85, currentPrice=50.0)
    bad_res = fr.calculate_intrinsic_value_ddm(bad_info, fin, cf, bal)
    assert "intrinsic_value" not in bad_res
    assert "exceeds sustainable threshold" in bad_res["error"]


def test_ddm_monte_carlo_distribution():
    """Test DDM Monte Carlo simulation returns percentiles and smoothed histogram."""
    mc = fr.run_monte_carlo_ddm(
        base_dividend=2.5,
        base_growth=0.04,
        base_discount=0.08,
        projection_years=10,
        iterations=2000,
    )
    assert "bear" in mc and "conservative" in mc and "base" in mc and "bull" in mc
    assert mc["bear"] < mc["conservative"] <= mc["base"] <= mc["bull"]
    assert len(mc["histogram"]) > 0
    for bar in mc["histogram"]:
        assert "price" in bar and "count" in bar


def test_peter_lynch_fair_value():
    """Test Peter Lynch Fair Value with PEG=1.0 and dividend yield boost."""
    info = _info(
        trailingEps=4.0, dividendRate=2.0, currentPrice=100.0
    )  # div yield = 2%
    fin = pd.DataFrame(
        {"2022-12-31": [100.0], "2023-12-31": [110.0]}, index=["Net Income"]
    )
    # Growth = 10%, div_yield = 2% -> Multiplier = 12.0 -> Fair value = 4 * 12 = 48.0
    res = fr.calculate_intrinsic_value_lynch(info, fin, growth_rate=10.0)
    assert "intrinsic_value" in res
    assert res["intrinsic_value"] == pytest.approx(48.0)
    assert res["parameters"]["fair_pe_multiplier"] == pytest.approx(12.0)


def test_mid_year_discounting_applied():
    """Verify that mid-year discounting is reflected in DCF parameters and calculation."""
    fin, bal, cf = _statements([1e9] * 5)
    dcf = fr.calculate_intrinsic_value_dcf(_info(), fin, bal, cf)
    assert dcf["parameters"]["mid_year_discounting"] is True
    assert "mid-year discounting" in dcf["parameters"]["note"]


def test_blume_adjusted_beta_in_wacc():
    """Verify Blume adjustment: raw beta 1.6 -> 0.67*1.6 + 0.33*1.0 = 1.402."""
    info = _info(beta=1.6)
    wacc_res = fr.calculate_wacc(info)
    assert wacc_res["beta"] == pytest.approx(0.67 * 1.6 + 0.33 * 1.0, abs=0.01)


def test_statement_alias_resolution():
    """Verify alias mapping: Operating Revenue, Purchase Of PPE, Cash Dividends Paid."""
    years = ["2023-12-31", "2022-12-31", "2021-12-31"]
    fin = pd.DataFrame(
        {y: [1e10, 1.5e9, 1e9, 2e8, 1.2e9] for y in years},
        index=[
            "Operating Revenue",  # Alias for Total Revenue
            "Operating Profit",  # Alias for Operating Income
            "Net Income Common Stockholders",  # Alias for Net Income
            "Provision For Income Taxes",  # Alias for Tax Provision
            "Pretax Operating Income",  # Alias for Pretax Income
        ],
    )
    cf = pd.DataFrame(
        {y: [1.8e9, -4e8, -2e8] for y in years},
        index=[
            "Cash Flow From Continuing Operating Activities",  # Alias for OCF
            "Purchase Of PPE",  # Alias for CapEx
            "Common Stock Dividend Paid",  # Alias for Cash Dividends Paid
        ],
    )
    bal = pd.DataFrame(
        {y: [2e10, 5e9, 3e9, 5e8] for y in years},
        index=[
            "Stockholders Equity",  # Alias for Total Stockholder Equity
            "Cash Cash Equivalents And Short Term Investments",  # Alias for Total Cash
            "Long Term Debt And Capital Lease Obligation",  # Alias for Total Debt
            "Share Issued",  # Alias for Ordinary Shares Number
        ],
    )

    # _get_statement_value should resolve canonical names through aliases
    assert fr._get_statement_value(fin, "Total Revenue", "2023-12-31") == 1e10
    assert fr._get_statement_value(fin, "Operating Income", "2023-12-31") == 1.5e9
    assert fr._get_statement_value(cf, "Operating Cash Flow", "2023-12-31") == 1.8e9
    assert fr._get_statement_value(cf, "Capital Expenditure", "2023-12-31") == -4e8
    assert fr._get_statement_value(bal, "Total Cash", "2023-12-31") == 5e9

    # FCF extraction should resolve properly
    fcf = fr._extract_fcf_from_statement(cf, "2023-12-31")
    assert fcf == pytest.approx(1.4e9)


def test_sector_aware_blending_financials():
    """Financials/Banks/REITs should blend DDM and Graham with proper weights."""
    fin, bal, cf = _statements([1e9] * 5)
    info = _info(
        sector="Financial Services",
        dividendRate=4.0,
        payoutRatio=0.50,
        trailingEps=6.0,
    )
    res = fr.get_comprehensive_intrinsic_value(info, fin, bal, cf, iterations=200)
    assert "ddm" in res["models"]
    assert "dcf" in res["models"]
    assert "graham" in res["models"]
    assert "lynch" in res["models"]
    # Model weights should include ddm, graham, dcf
    assert "ddm" in res["model_weights"]
    assert res["model_weights"]["ddm"] > 0.25
