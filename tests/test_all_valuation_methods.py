import pandas as pd
import pytest
import financial_ratios as fr


def _sample_info(**kwargs):
    base = {
        "currentPrice": 100.0,
        "regularMarketPrice": 100.0,
        "sharesOutstanding": 1_000_000,
        "trailingEps": 5.0,
        "forwardPE": 15.0,
        "trailingPE": 20.0,
        "priceToBook": 2.5,
        "priceToSalesTrailing12Months": 3.0,
        "bookValue": 40.0,
        "totalRevenue": 33_333_333,
        "grossMargins": 0.60,
        "operatingCashflow": 6_000_000,
        "freeCashflow": 4_500_000,
        "marketCap": 100_000_000,
        "dividendRate": 2.0,
        "dividendYield": 0.02,
        "beta": 1.0,
        "sector": "Technology",
    }
    base.update(kwargs)
    return base


def _sample_financials(rev=30e6, ebit=6e6, net_inc=5e6, ocf=6e6, capex=-1.5e6):
    dates = pd.date_range("2024-12-31", periods=5, freq="-1YE").strftime("%Y-%m-%d")
    fin = pd.DataFrame(
        {
            d: {
                "Total Revenue": rev * (1.10**i),
                "Operating Income": ebit * (1.10**i),
                "Net Income": net_inc * (1.10**i),
                "Diluted EPS": (net_inc / 1e6) * (1.10**i),
                "Tax Provision": (net_inc * 0.21) * (1.10**i),
                "Pretax Income": (net_inc * 1.21) * (1.10**i),
            }
            for i, d in enumerate(dates)
        }
    )
    bal = pd.DataFrame(
        {
            d: {
                "Total Stockholder Equity": 40e6 * (1.05**i),
                "Total Cash": 10e6,
                "Total Debt": 5e6,
                "Ordinary Shares Number": 1_000_000,
            }
            for i, d in enumerate(dates)
        }
    )
    cf = pd.DataFrame(
        {
            d: {
                "Operating Cash Flow": ocf * (1.08**i),
                "Capital Expenditure": capex * (1.05**i),
                "Cash Dividends Paid": -2e6,
            }
            for i, d in enumerate(dates)
        }
    )
    return fin, bal, cf


def test_discounted_free_cash_flow_primary():
    fin, bal, cf = _sample_financials()
    info = _sample_info()
    res = fr.calculate_intrinsic_value_dcf(info, fin, bal, cf)
    assert "intrinsic_value" in res
    assert res["intrinsic_value"] > 0
    assert res["model"] == "DCF"
    assert res["parameters"]["mid_year_discounting"] is True


def test_discounted_cash_from_operations():
    fin, bal, cf = _sample_financials(ocf=8e6)
    info = _sample_info()
    res = fr.calculate_intrinsic_value_dcfo(info, fin, bal, cf)
    assert "intrinsic_value" in res
    assert res["intrinsic_value"] > 0
    assert res["model"] == "Discounted Cash from Operations"
    assert "when_to_use" in res
    assert "key_limitation" in res
    assert res["parameters"]["mid_year_discounting"] is True
    assert res["parameters"]["base_cfo"] == pytest.approx(8e6, rel=0.01)


def test_discounted_net_income():
    fin, bal, _ = _sample_financials(net_inc=6e6)
    info = _sample_info(trailingEps=6.0)
    res = fr.calculate_intrinsic_value_dni(info, fin, bal)
    assert "when_to_use" in res
    assert "best_suited_for" in res
    assert "key_limitation" in res
    assert "key_caveats" in res
    assert "Financial institutions" in res["best_suited_for"]


def test_mean_pe_valuation():
    """The multiple comes from what the company has traded at, not from today's.

    `trailingPE` is price/EPS, so multiplying it back by EPS returned the quote
    itself. The model now takes the median of the traded record and refuses
    when there is none.
    """
    info = _sample_info(trailingEps=4.0, trailingPE=22.0)
    history = {"value": 15.0, "n": 12, "span": "2013-2024", "low": 12.0, "high": 19.0}

    res = fr.calculate_intrinsic_value_mean_pe(info, history=history)
    assert res["intrinsic_value"] == pytest.approx(4.0 * 15.0, rel=1e-3)
    assert res["model"] == "Mean P/E Ratio"
    assert res["parameters"]["applied_pe"] == 15.0
    assert res["parameters"]["multiple_observations"] == 12
    assert "22" not in res["parameters"]["pe_source"], "must not read the trailing P/E"

    refused = fr.calculate_intrinsic_value_mean_pe(info)
    assert "intrinsic_value" not in refused
    assert "No traded P/E history" in refused["error"]


def test_mean_pe_negative_eps_rejected():
    info = _sample_info(trailingEps=-2.5)
    res = fr.calculate_intrinsic_value_mean_pe(info)
    assert "error" in res
    assert "Negative or missing EPS" in res["error"]


def test_peg_ratio_fair_value():
    info = _sample_info(trailingEps=5.0, dividendRate=1.0, currentPrice=100.0)
    res = fr.calculate_intrinsic_value_peg(info, growth_rate=15.0, target_peg=1.0)
    assert "intrinsic_value" in res
    assert res["model"] == "PEG Ratio Fair Value"
    # EPS 5.0 * (15% growth + 1% div yield) = 5.0 * 16.0 = 80.0
    assert res["parameters"]["fair_pe_multiplier"] == pytest.approx(16.0, rel=0.05)
    assert res["intrinsic_value"] == pytest.approx(80.0, rel=0.05)


def test_mean_pb_valuation_bank_benchmark():
    """The sector benchmark survives only where there is no traded record."""
    info = _sample_info(sector="Financial Services", bookValue=50.0)
    res = fr.calculate_intrinsic_value_mean_pb(info)
    assert "intrinsic_value" in res
    assert res["model"] == "Mean P/B Ratio"
    # Banks benchmark at 1.30x
    assert res["parameters"]["applied_pb"] == 1.30
    assert res["intrinsic_value"] == pytest.approx(50.0 * 1.30, rel=1e-3)

    # ...and the company's own history takes precedence over it.
    history = {"value": 0.9, "n": 11, "span": "2014-2024", "low": 0.7, "high": 1.1}
    own = fr.calculate_intrinsic_value_mean_pb(info, history=history)
    assert own["intrinsic_value"] == pytest.approx(50.0 * 0.9, rel=1e-3)


def test_mean_pb_non_financial_refuses_without_history():
    """`priceToBook` x book value is the price; there is no fallback to it."""
    res = fr.calculate_intrinsic_value_mean_pb(
        _sample_info(sector="Technology", bookValue=40.0, priceToBook=2.5)
    )
    assert "intrinsic_value" not in res
    assert "No traded P/B history" in res["error"]


def test_mean_ps_valuation():
    info = _sample_info(
        totalRevenue=50_000_000,
        sharesOutstanding=1_000_000,
        priceToSalesTrailing12Months=2.5,
    )
    history = {"value": 2.0, "n": 9, "span": "2016-2024", "low": 1.6, "high": 2.6}

    res = fr.calculate_intrinsic_value_mean_ps(info, history=history)
    assert res["model"] == "Mean P/S Ratio"
    # SPS = $50, median traded P/S = 2.0 => IV = $100
    assert res["parameters"]["sales_per_share"] == 50.0
    assert res["intrinsic_value"] == pytest.approx(100.0, rel=1e-3)

    refused = fr.calculate_intrinsic_value_mean_ps(info)
    assert "intrinsic_value" not in refused
    assert "No traded P/S history" in refused["error"]


def test_psg_valuation_unprofitable_growth():
    info = _sample_info(totalRevenue=40_000_000, sharesOutstanding=1_000_000, grossMargins=0.70)
    res = fr.calculate_intrinsic_value_psg(info, revenue_growth_pct=25.0, target_psg=1.0)
    assert "intrinsic_value" in res
    assert res["model"] == "Price-to-Sales Growth (PSG)"
    assert res["parameters"]["sales_per_share"] == 40.0
    assert res["intrinsic_value"] > 0


def test_recommend_best_valuation_method_classifier():
    # 1. Bank -> Mean P/B or D-NI
    bank_info = _sample_info(sector="Banks", trailingEps=4.0, bookValue=30.0)
    fin, bal, cf = _sample_financials()
    res_bank = fr.get_comprehensive_intrinsic_value(bank_info, fin, bal, cf, iterations=100)
    assert "recommended_method" in res_bank
    assert res_bank["recommended_method"]["method_key"] in ("mean_pb", "dni", "ddm")

    # 2. Tech cash cow with strong FCF -> DCF
    tech_info = _sample_info(sector="Technology", freeCashflow=10_000_000)
    res_tech = fr.get_comprehensive_intrinsic_value(tech_info, fin, bal, cf, iterations=100)
    assert res_tech["recommended_method"]["method_key"] == "dcf"
    assert "Primary" in res_tech["recommended_method"]["name"]

    # 3. All 8 models populated in models dictionary
    models = res_tech["models"]
    for expected_key in ("dcf", "dcfo", "dni", "mean_pe", "peg", "mean_pb", "mean_ps", "psg", "ddm", "graham", "epv", "lynch"):
        assert expected_key in models
