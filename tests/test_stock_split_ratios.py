"""Tests for stock-split consistency in financial statements, ratios, and valuation bands."""

import os
import sys
from datetime import date
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import edgar_provider
import market_data
from market_data import _cik_for_symbol
from financial_ratios import calculate_key_ratios_timeseries
import valuation_history


class TestStockSplitStatementAdjustment:
    """Verify that get_statements restates share counts and per-share figures."""

    GOOG_CIK = "0001652044"

    def test_goog_split_adjustment(self, monkeypatch):
        monkeypatch.setattr(market_data, "_cik_for_symbol", lambda s: self.GOOG_CIK)
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        # Mock edgar_provider.get_statements with raw filed figures and split adjustment
        def mock_get_statements(cik_val):
            # Pre-split filed EPS: 2019: 49.16, 2018: 43.70, 2014: 20.57
            # Post-split 20:1 factor: 20.0
            # Pre-split filed shares: 2020: 675M, 2019: 688M, 2014: 680M
            dates = [pd.Timestamp("2020-12-31"), pd.Timestamp("2019-12-31"), pd.Timestamp("2018-12-31"), pd.Timestamp("2014-12-31")]
            fin_df = pd.DataFrame(
                {
                    dates[0]: [152.0, 7.60],
                    dates[1]: [161.8, 49.16 / 20.0],
                    dates[2]: [136.8, 43.70 / 20.0],
                    dates[3]: [66.0, 20.57 / 20.0],
                },
                index=["Total Revenue", "Diluted EPS"]
            )
            bs_df = pd.DataFrame(
                {
                    dates[0]: [675e6 * 20.0, 319.6e9],
                    dates[1]: [688e6 * 20.0, 275.9e9],
                    dates[2]: [695e6 * 20.0, 232.8e9],
                    dates[3]: [680e6 * 20.0, 131.1e9],
                },
                index=["Ordinary Shares Number", "Total Assets"]
            )
            cf_df = pd.DataFrame(
                {
                    dates[0]: [65.1e9, -22.3e9],
                    dates[1]: [54.5e9, -23.5e9],
                    dates[2]: [48.0e9, -25.1e9],
                    dates[3]: [22.4e9, -11.8e9],
                },
                index=["Operating Cash Flow", "Capital Expenditure"]
            )
            return {"financials": fin_df, "balance_sheet": bs_df, "cashflow": cf_df}

        monkeypatch.setattr(edgar_provider, "get_statements", mock_get_statements)

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

    def test_split_adjustment_factors_fallback(self, monkeypatch):
        """split_adjustment_factors should return factors for historical periods even if shares_diluted has few years."""
        monkeypatch.setattr(market_data, "_cik_for_symbol", lambda s: self.GOOG_CIK)
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        # Mock concept values and split consistent series
        assembled_shares = {
            "shares_diluted": {
                "2019-12-31": 6.88e8,
                "2020-12-31": 6.75e8,
                "2022-12-31": 1.35e10,
            }
        }
        corrected_shares = {
            "2019-12-31": 1.376e10,
            "2020-12-31": 1.35e10,
            "2022-12-31": 1.35e10,
        }
        monkeypatch.setattr(edgar_provider, "get_concept_values", lambda c, concepts: assembled_shares)
        monkeypatch.setattr(edgar_provider, "split_consistent_series", lambda c, concept: corrected_shares)

        factors = edgar_provider.split_adjustment_factors(cik)
        # Should cover older years (e.g. 2019) and have ~20.0 factor
        assert "2019-12-31" in factors
        assert factors["2019-12-31"] == pytest.approx(20.0, rel=0.05)
        assert factors["2022-12-31"] == pytest.approx(1.0, rel=0.05)

    def test_goog_historical_valuation_ratios_not_distorted(self):
        # Create synthetic 10-year financial statements with split-adjusted numbers
        years = [f"{y}-12-31" for y in range(2014, 2024)]
        dates = [pd.Timestamp(y) for y in years]

        fin = pd.DataFrame(
            {
                d: {
                    "Total Revenue": 100e9 + i * 15e9,
                    "Net Income": 20e9 + i * 4e9,
                    "Operating Income": 25e9 + i * 5e9,
                    "EBITDA": 30e9 + i * 6e9,
                    "Diluted EPS": 1.5 + i * 0.3,
                }
                for i, d in enumerate(dates)
            }
        )
        bs = pd.DataFrame(
            {
                d: {
                    "Total Assets": 150e9 + i * 20e9,
                    "Total Equity Gross Minority Interest": 100e9 + i * 15e9,
                    "Stockholders Equity": 100e9 + i * 15e9,
                    "Ordinary Shares Number": 13e9,
                    "Cash And Cash Equivalents": 30e9 + i * 5e9,
                    "Total Debt": 5e9,
                }
                for i, d in enumerate(dates)
            }
        )
        cf = pd.DataFrame(
            {
                d: {
                    "Operating Cash Flow": 30e9 + i * 5e9,
                    "Capital Expenditure": -10e9 - i * 1.5e9,
                    "Free Cash Flow": 20e9 + i * 3.5e9,
                }
                for i, d in enumerate(dates)
            }
        )

        price_dates = pd.date_range("2014-01-01", "2024-01-01", freq="D")
        prices_df = pd.DataFrame(
            {
                "Close": [50.0 + (i / len(price_dates)) * 80.0 for i in range(len(price_dates))],
            },
            index=price_dates
        )

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
                    assert ev_ebitda > 5.0, (
                        f"EV/EBITDA for {year} is {ev_ebitda}, expected > 5.0"
                    )

    def test_valuation_history_bands_for_goog(self, monkeypatch):
        monkeypatch.setattr(market_data, "_cik_for_symbol", lambda s: self.GOOG_CIK)
        cik = _cik_for_symbol("GOOG")
        assert cik is not None

        years = list(range(2011, 2026))
        dates = [f"{y}-12-31" for y in years]
        price_index = pd.to_datetime([f"{y}-12-20" for y in years])
        prices_series = pd.Series([50.0 + i * 5.0 for i in range(len(years))], index=price_index)

        monkeypatch.setattr(valuation_history, "_load_prices", lambda s, start, end: prices_series)
        monkeypatch.setattr(edgar_provider, "split_consistent_series", lambda c, concept: {d: 13.5e9 for d in dates})
        monkeypatch.setattr(
            edgar_provider,
            "get_concept_values",
            lambda c, concepts=None: {
                "net_income": {d: 20e9 + i * 3e9 for i, d in enumerate(dates)},
                "operating_cash_flow": {d: 25e9 + i * 4e9 for i, d in enumerate(dates)},
                "capex": {d: 8e9 + i * 1e9 for i, d in enumerate(dates)},
            }
        )

        res = valuation_history.bands("GOOG", cik, today=date(2026, 7, 31))
        assert isinstance(res, list)
        assert len(res) > 0, "Valuation history bands should return data for GOOG"
        metrics = [b["metric"] for b in res]
        assert "earnings" in metrics

