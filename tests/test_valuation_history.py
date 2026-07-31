"""Tests for self-relative valuation bands (`valuation_history`).

Investa's value score is cross-sectional — a stock is cheap because its earnings
yield beats other stocks'. That is right for a ranking and useless to a reader
looking at one company, because it cannot say whether 23x is this business being
expensive or this business being normal. Fifteen years of filings and prices can.

The tests pin the three things that make the arithmetic honest rather than
merely available: market cap instead of per-share (so no split basis can be
mixed), point-in-time earnings (so a 2012 multiple is not priced on 2019
restatements), and a refusal to draw a fifteen-year band from four years.
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import valuation_history as vh  # noqa: E402


def price_series(per_year: dict) -> pd.Series:
    """Daily-ish closes: one point per year end, which is all the sampler needs."""
    index = pd.to_datetime([f"{year}-12-20" for year in sorted(per_year)])
    return pd.Series([per_year[y] for y in sorted(per_year)], index=index)


class TestPercentileAndWording:
    def test_percentile_counts_the_history_at_or_below(self):
        assert vh._percentile_of(30.0, [10.0, 20.0, 30.0, 40.0]) == pytest.approx(75.0)
        assert vh._percentile_of(5.0, [10.0, 20.0]) == 0.0

    @pytest.mark.parametrize(
        "current,expected",
        [
            (100.0, "dearer than almost all of its own history"),
            (1.0, "cheaper than almost all of its own history"),
        ],
    )
    def test_extremes_are_named(self, current, expected):
        history = [float(v) for v in range(10, 30)]
        assert vh._describe(current, history) == expected

    def test_the_middle_is_not_dressed_up_as_a_signal(self):
        history = [float(v) for v in range(10, 30)]
        assert vh._describe(20.0, history) == "around its own long-run average"


class TestBands:
    CIK = "0000000001"

    def _patch(self, monkeypatch, prices, shares, income, ocf=None, capex=None):
        import edgar_provider

        monkeypatch.setattr(vh, "_load_prices", lambda symbol, start, end: prices)
        monkeypatch.setattr(
            edgar_provider, "split_consistent_series", lambda cik, concept: shares
        )

        def concept_values(cik, concepts=None):
            return {
                "net_income": income,
                "operating_cash_flow": ocf or {},
                "capex": capex or {},
            }

        monkeypatch.setattr(edgar_provider, "get_concept_values", concept_values)

    def test_a_band_is_built_from_market_cap(self, monkeypatch):
        """
        Ten shares at $10 against $10 of earnings is 10x, whatever the price
        series has been back-adjusted by — which is the reason this uses market
        cap rather than a price-to-EPS ratio.
        """
        years = range(2011, 2026)
        self._patch(
            monkeypatch,
            price_series({y: 10.0 for y in years}),
            {f"{y}-12-31": 10.0 for y in years},
            {f"{y}-12-31": 10.0 for y in years},
        )
        result = vh.bands("TEST", self.CIK, today=date(2026, 7, 31))
        earnings = next(b for b in result if b["metric"] == "earnings")
        assert earnings["current"] == pytest.approx(10.0)
        assert earnings["median"] == pytest.approx(10.0)
        assert earnings["display"] == "10.0x"

    def test_todays_position_in_its_own_range(self, monkeypatch):
        years = list(range(2011, 2026))
        prices = {y: 10.0 for y in years}
        prices[2025] = 30.0  # rerated, on unchanged earnings
        self._patch(
            monkeypatch,
            price_series(prices),
            {f"{y}-12-31": 10.0 for y in years},
            {f"{y}-12-31": 10.0 for y in years},
        )
        result = vh.bands("TEST", self.CIK, today=date(2026, 7, 31))
        earnings = next(b for b in result if b["metric"] == "earnings")
        assert earnings["current"] > earnings["median"]
        assert earnings["percentile"] == pytest.approx(100.0)
        assert "dearer" in earnings["summary"]

    def test_a_short_price_history_draws_no_band(self, monkeypatch):
        """
        JPMorgan has four years of local prices. A fifteen-year band drawn from
        them would be a fabrication, so there is none.
        """
        years = range(2023, 2026)
        self._patch(
            monkeypatch,
            price_series({y: 10.0 for y in years}),
            {f"{y}-12-31": 10.0 for y in years},
            {f"{y}-12-31": 10.0 for y in years},
        )
        assert vh.bands("TEST", self.CIK, today=date(2026, 7, 31)) == []

    def test_losses_never_become_a_multiple(self, monkeypatch):
        """A negative denominator yields a negative "multiple" that sorts as cheap."""
        years = range(2011, 2026)
        self._patch(
            monkeypatch,
            price_series({y: 10.0 for y in years}),
            {f"{y}-12-31": 10.0 for y in years},
            {f"{y}-12-31": -5.0 for y in years},
        )
        assert vh.bands("TEST", self.CIK, today=date(2026, 7, 31)) == []

    def test_no_price_history_is_not_an_error(self, monkeypatch):
        import edgar_provider

        monkeypatch.setattr(vh, "_load_prices", lambda symbol, start, end: None)
        monkeypatch.setattr(
            edgar_provider,
            "split_consistent_series",
            lambda cik, concept: {"2025-12-31": 1.0},
        )
        assert vh.bands("TEST", self.CIK, today=date(2026, 7, 31)) == []

    def test_free_cash_flow_nets_capex_off(self, monkeypatch):
        years = range(2011, 2026)
        self._patch(
            monkeypatch,
            price_series({y: 10.0 for y in years}),
            {f"{y}-12-31": 10.0 for y in years},
            {f"{y}-12-31": 10.0 for y in years},
            ocf={f"{y}-12-31": 25.0 for y in years},
            capex={f"{y}-12-31": 5.0 for y in years},
        )
        result = vh.bands("TEST", self.CIK, today=date(2026, 7, 31))
        fcf = next(b for b in result if b["metric"] == "free_cash_flow")
        # 100 of market cap over (25 - 5) of free cash flow.
        assert fcf["current"] == pytest.approx(5.0)
