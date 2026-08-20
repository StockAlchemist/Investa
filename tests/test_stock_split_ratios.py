"""Tests for stock-split consistency in financial statements, ratios, and valuation bands."""

import os
import sys
import datetime
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import edgar_provider
from market_data import MarketDataProvider, _cik_for_symbol
from financial_ratios import calculate_key_ratios_timeseries
import valuation_history


class TestStockSplitStatementAdjustment:
    """Verify that get_statements restates share counts and per-share figures."""

    def test_goog_split_adjustment(self):
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        stmts = edgar_provider.get_statements(cik)
        fin = stmts["financials"]
        bs = stmts["balance_sheet"]

        assert "Diluted EPS" in fin.index
        eps = fin.loc["Diluted EPS"]

        # GOOG 20:1 split occurred in July 2022.
        # 2019 filed EPS was 49.16 -> restated should be ~2.46
        # 2018 filed EPS was 43.70 -> restated should be ~2.18
        # 2014 filed EPS was 20.57 -> restated should be ~1.03
        stamp_2019 = pd.Timestamp("2019-12-31")
        stamp_2018 = pd.Timestamp("2018-12-31")
        stamp_2014 = pd.Timestamp("2014-12-31")

        if stamp_2019 in eps.index:
            assert eps[stamp_2019] == pytest.approx(2.458, rel=0.05)
        if stamp_2018 in eps.index:
            assert eps[stamp_2018] == pytest.approx(2.185, rel=0.05)
        if stamp_2014 in eps.index:
            assert eps[stamp_2014] == pytest.approx(1.028, rel=0.05)

        assert "Ordinary Shares Number" in bs.index
        shares = bs.loc["Ordinary Shares Number"]

        # 2020 filed shares ~675M -> restated should be ~13.5B
        # 2019 filed shares ~688M -> restated should be ~13.7B
        # 2014 filed shares ~680M -> restated should be ~13.6B
        stamp_2020 = pd.Timestamp("2020-12-31")
        if stamp_2020 in shares.index:
            assert shares[stamp_2020] > 1.2e10
            assert shares[stamp_2020] < 1.5e10
        if stamp_2019 in shares.index:
            assert shares[stamp_2019] > 1.2e10
            assert shares[stamp_2019] < 1.5e10
        if stamp_2014 in shares.index:
            assert shares[stamp_2014] > 1.2e10
            assert shares[stamp_2014] < 1.5e10

    def test_split_adjustment_factors_fallback(self):
        """split_adjustment_factors should return factors for historical periods even if shares_diluted has few years."""
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        factors = edgar_provider.split_adjustment_factors(cik)
        # Should cover older years (e.g. 2014-2020) and have ~20.0 factor
        assert "2019-12-31" in factors
        assert factors["2019-12-31"] == pytest.approx(20.0, rel=0.05)
        assert factors["2022-12-31"] == pytest.approx(1.0, rel=0.05)

    def test_goog_historical_valuation_ratios_not_distorted(self):
        mdp = MarketDataProvider()
        symbol = "GOOG"
        fin = mdp.get_financials(symbol, "annual")
        bs = mdp.get_balance_sheet(symbol, "annual")
        cf = mdp.get_cashflow(symbol, "annual")

        start_date = datetime.date.today() - datetime.timedelta(days=365 * 15)
        hist, _ = mdp.get_historical_data([symbol], start_date, datetime.date.today())
        prices_df = hist.get(symbol)

        ratios = calculate_key_ratios_timeseries(fin, bs, cf, prices_df=prices_df)
        assert not ratios.empty

        # Verify historical P/E for 2016-2020 is not ~1.0-2.0, but > 15
        for year in ["2016", "2017", "2018", "2019", "2020"]:
            matching_idx = [idx for idx in ratios.index if str(idx).startswith(year)]
            if matching_idx:
                row = ratios.loc[matching_idx[0]]
                pe = row.get("P/E Ratio")
                pb = row.get("P/B Ratio")
                ps = row.get("P/S Ratio")
                ev_ebitda = row.get("EV/EBITDA")

                if pe is not None and not pd.isna(pe):
                    assert pe > 15, f"P/E for {year} is {pe}, expected > 15"
                if pb is not None and not pd.isna(pb):
                    assert pb > 2.0, f"P/B for {year} is {pb}, expected > 2.0"
                if ps is not None and not pd.isna(ps):
                    assert ps > 2.0, f"P/S for {year} is {ps}, expected > 2.0"
                if ev_ebitda is not None and not pd.isna(ev_ebitda):
                    assert ev_ebitda > 10.0, (
                        f"EV/EBITDA for {year} is {ev_ebitda}, expected > 10.0"
                    )

    def test_valuation_history_bands_for_goog(self):
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        res = valuation_history.bands("GOOG", cik)
        assert isinstance(res, list)
        assert len(res) > 0, "Valuation history bands should return data for GOOG"
        metrics = [b["metric"] for b in res]
        assert "earnings" in metrics
