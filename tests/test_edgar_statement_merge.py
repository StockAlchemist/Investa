"""Tests for the EDGAR-preferred merge behind the statement getters (`market_data`).

yfinance returns four or five annual periods; the SEC XBRL store carries ~19 for
the same filer, and until now only the Buffett ranking read it. The merge lets
the per-stock views see the filed history without losing the line items Yahoo
reports and EDGAR does not tag.

Two things have to hold or the merge does damage rather than good:
  * a fiscal year must not appear twice because the two sources stamp it
    differently (Yahoo rounds AAPL's FY2025 to 2025-09-30; the filing says
    2025-09-27), and
  * income and balance must land on the *same* period stamps, because
    `calculate_key_ratios_timeseries` intersects their columns and an
    off-by-three-days disagreement would empty the ratio history entirely.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import edgar_provider  # noqa: E402
import market_data  # noqa: E402
from market_data import (  # noqa: E402
    _canonical_periods,
    _cik_for_symbol,
    _merge_with_edgar,
    _snap_to_filed_period,
    _with_edgar_history,
)


def frame(rows: dict, periods: list) -> pd.DataFrame:
    """A statement frame shaped like yfinance's: rows are line items, newest first."""
    stamps = pd.to_datetime(periods)
    return pd.DataFrame(rows, index=stamps).T.sort_index(axis=1, ascending=False)


# Filed period ends: AAPL's 52/53-week fiscal calendar.
EDGAR_INCOME = frame(
    {
        "Total Revenue": [416.2, 391.0, 383.3, 394.3],
        "Net Income": [112.0, 93.7, 97.0, 99.8],
    },
    ["2025-09-27", "2024-09-28", "2023-09-30", "2022-09-24"],
)

# Yahoo's view of the same company: month-end stamps, fewer years, extra rows,
# and a trailing column it reports as all-NaN.
YAHOO_INCOME = frame(
    {
        "Total Revenue": [416.2, 391.0, 383.3, 394.3, float("nan")],
        "Research And Development": [34.6, 31.4, 29.9, 26.3, 21.9],
    },
    ["2025-09-30", "2024-09-30", "2023-09-30", "2022-09-30", "2021-09-30"],
)


class TestPeriodSnapping:
    def test_month_end_snaps_onto_the_filed_period(self):
        calendar = pd.DatetimeIndex(pd.to_datetime(["2025-09-27", "2024-09-28"]))
        snapped = _snap_to_filed_period(pd.Timestamp("2025-09-30"), calendar)
        assert snapped == pd.Timestamp("2025-09-27")

    def test_snapping_crosses_the_new_year(self):
        """AAP's FY2025 ended 2026-01-03; Yahoo calls it 2025-12-31."""
        calendar = pd.DatetimeIndex(pd.to_datetime(["2026-01-03", "2024-12-28"]))
        snapped = _snap_to_filed_period(pd.Timestamp("2025-12-31"), calendar)
        assert snapped == pd.Timestamp("2026-01-03")

    def test_a_genuinely_new_period_keeps_its_own_stamp(self):
        """A year Yahoo has and EDGAR has not ingested must survive as its own column."""
        calendar = pd.DatetimeIndex(pd.to_datetime(["2024-09-28", "2023-09-30"]))
        stamp = pd.Timestamp("2025-09-30")
        assert _snap_to_filed_period(stamp, calendar) == stamp

    def test_empty_calendar_is_a_no_op(self):
        stamp = pd.Timestamp("2025-09-30")
        assert _snap_to_filed_period(stamp, pd.DatetimeIndex([])) == stamp


class TestMerge:
    @pytest.fixture
    def merged(self):
        calendar = _canonical_periods({"financials": EDGAR_INCOME})
        return _merge_with_edgar(YAHOO_INCOME, EDGAR_INCOME, calendar)

    def test_no_fiscal_year_is_duplicated(self, merged):
        """Four filed years plus Yahoo's one extra — not nine near-identical columns."""
        assert len(merged.columns) == 5
        assert list(merged.columns) == list(
            pd.to_datetime(
                ["2025-09-27", "2024-09-28", "2023-09-30", "2022-09-24", "2021-09-30"]
            )
        )

    def test_columns_stay_newest_first(self, merged):
        """`financial_ratios` and all three clients read column 0 as the latest period."""
        assert merged.columns[0] == pd.Timestamp("2025-09-27")

    def test_edgar_history_reaches_past_yahoo(self, merged):
        assert merged.loc["Total Revenue", pd.Timestamp("2022-09-24")] == 394.3

    def test_yahoo_only_rows_survive(self, merged):
        """EDGAR tags no R&D line; dropping it would make the merge a downgrade."""
        assert (
            merged.loc["Research And Development", pd.Timestamp("2025-09-27")] == 34.6
        )

    def test_edgar_wins_a_contested_cell(self):
        """The filed number is preferred wherever both sources have one."""
        yahoo = frame({"Total Revenue": [169.4]}, ["2024-12-31"])
        edgar = frame({"Total Revenue": [177.6]}, ["2024-12-31"])
        merged = _merge_with_edgar(yahoo, edgar, _canonical_periods({"a": edgar}))
        assert merged.loc["Total Revenue", pd.Timestamp("2024-12-31")] == 177.6

    def test_yahoo_fills_a_gap_edgar_leaves(self):
        yahoo = frame({"Total Revenue": [416.2, 391.0]}, ["2025-09-30", "2024-09-30"])
        edgar = frame(
            {"Total Revenue": [float("nan"), 391.0]}, ["2025-09-27", "2024-09-28"]
        )
        merged = _merge_with_edgar(yahoo, edgar, _canonical_periods({"a": edgar}))
        assert merged.loc["Total Revenue", pd.Timestamp("2025-09-27")] == 416.2

    def test_statement_row_order_is_preserved(self, merged):
        """EDGAR's line items keep statement order; Yahoo's extras follow."""
        assert list(merged.index) == [
            "Total Revenue",
            "Net Income",
            "Research And Development",
        ]

    def test_missing_yahoo_data_is_not_an_error(self):
        merged = _merge_with_edgar(
            None, EDGAR_INCOME, _canonical_periods({"a": EDGAR_INCOME})
        )
        assert len(merged.columns) == 4
        merged = _merge_with_edgar(
            pd.DataFrame(), EDGAR_INCOME, _canonical_periods({"a": EDGAR_INCOME})
        )
        assert len(merged.columns) == 4


class TestSharedCalendar:
    def test_income_and_balance_land_on_the_same_periods(self):
        """
        The ratio engine intersects income and balance columns. If each statement
        snapped to its own calendar, a balance sheet EDGAR happens not to cover
        would keep Yahoo's month-end stamps and the intersection would be empty.
        """
        edgar_balance = pd.DataFrame()
        statements = {"financials": EDGAR_INCOME, "balance_sheet": edgar_balance}
        calendar = _canonical_periods(statements)

        yahoo_balance = frame(
            {"Total Assets": [364.9, 365.0]}, ["2025-09-30", "2024-09-30"]
        )
        income = _merge_with_edgar(YAHOO_INCOME, EDGAR_INCOME, calendar)
        balance = _merge_with_edgar(yahoo_balance, EDGAR_INCOME, calendar).drop(
            index=["Total Revenue", "Net Income"]
        )

        common = set(income.columns) & set(balance.columns)
        assert pd.Timestamp("2025-09-27") in common


class TestApplicability:
    def test_quarterly_requests_are_left_alone(self):
        """The store holds 10-K facts only; a quarterly merge would be a lie."""
        yahoo = frame({"Total Revenue": [102.5]}, ["2025-06-28"])
        result = _with_edgar_history("AAPL", "financials", "quarterly", yahoo)
        assert result is yahoo

    @pytest.mark.parametrize("symbol", ["PTT.BK", "0700.HK", "^GSPC", "THB=X"])
    def test_non_us_symbols_skip_the_lookup(self, symbol, monkeypatch):
        """
        Foreign listings and indices are not SEC filers. They must not reach the
        network — a Thai holding should never wait on data.sec.gov.
        """
        monkeypatch.setattr(market_data, "_cik_map", None, raising=False)
        monkeypatch.setattr(market_data, "_cik_map_loaded_at", 0.0, raising=False)

        assert _cik_for_symbol(symbol) is None
        # Still unloaded: the symbol was rejected before any map was fetched.
        assert market_data._cik_map is None

    def test_a_missing_map_does_not_block_the_request(self, monkeypatch):
        """
        The SEC map is the one part of this path that can touch the network, and
        `sec_get` retries three times before giving up. A statement request must
        never wait behind that: with no map yet, the caller gets Yahoo's five
        years and the fetch happens on its own thread.
        """
        started: list[str] = []

        monkeypatch.setattr(market_data, "_cik_map", None, raising=False)
        monkeypatch.setattr(market_data, "_cik_map_refreshing", False, raising=False)
        monkeypatch.setattr(market_data, "_load_cik_map_from_disk", lambda: (None, 0.0))
        monkeypatch.setattr(
            market_data.threading,
            "Thread",
            lambda **kwargs: type(
                "FakeThread",
                (),
                {"start": lambda _self: started.append(kwargs["name"])},
            )(),
        )

        assert _cik_for_symbol("AAPL") is None
        assert started == ["edgar-cik-map"]

    def test_a_cached_map_is_adopted_without_a_fetch(self, monkeypatch):
        """A restart must not cost a day of shallow statements."""
        monkeypatch.setattr(market_data, "_cik_map", None, raising=False)
        monkeypatch.setattr(market_data, "_cik_map_refreshing", False, raising=False)
        monkeypatch.setattr(
            market_data,
            "_load_cik_map_from_disk",
            lambda: ({"AAPL": "0000320193"}, 60.0),
        )
        monkeypatch.setattr(
            market_data.threading,
            "Thread",
            lambda **kwargs: pytest.fail("refreshed a map that was still fresh"),
        )

        assert _cik_for_symbol("AAPL") == "0000320193"

    def test_a_symbol_without_edgar_data_returns_yahoo_untouched(self, monkeypatch):
        yahoo = frame({"Total Revenue": [1.0]}, ["2025-12-31"])
        monkeypatch.setattr(market_data, "_cik_for_symbol", lambda s: "0000000001")
        monkeypatch.setattr(market_data, "_edgar_annual_statements", lambda cik: {})
        assert _with_edgar_history("XYZ", "financials", "annual", yahoo) is yahoo

    def test_a_broken_merge_falls_back_to_yahoo(self, monkeypatch):
        """A bad row in the store must degrade to Yahoo's five years, not a 500."""
        yahoo = frame({"Total Revenue": [1.0]}, ["2025-12-31"])
        monkeypatch.setattr(market_data, "_cik_for_symbol", lambda s: "0000000001")
        monkeypatch.setattr(
            market_data,
            "_edgar_annual_statements",
            lambda cik: {"financials": EDGAR_INCOME},
        )

        def boom(*args, **kwargs):
            raise ValueError("unalignable axes")

        monkeypatch.setattr(market_data, "_merge_with_edgar", boom)
        assert _with_edgar_history("XYZ", "financials", "annual", yahoo) is yahoo


class TestSplitAdjustedStatements:
    """
    Per-share rows are restated onto the latest split basis before they reach a
    statement table (`edgar_provider._apply_split_adjustment`).

    As-filed is right for one filing and wrong for nineteen side by side: Apple
    filed $9.21 of diluted EPS for FY2017 and $2.98 for FY2018, and a table that
    prints both reads as a two-thirds collapse rather than the 4:1 split it is.
    The trend sparkline draws that cliff.
    """

    @staticmethod
    def values():
        return {
            "eps_diluted": {"2017-09-30": 9.21, "2018-09-29": 2.98},
            "shares_diluted": {"2017-09-30": 5.25e9, "2018-09-29": 20.0e9},
            "shares_basic": {"2017-09-30": 5.22e9, "2018-09-29": 19.82e9},
            "net_income": {"2017-09-30": 48.35e9, "2018-09-29": 59.53e9},
            "revenue": {"2017-09-30": 229.23e9, "2018-09-29": 265.60e9},
        }

    # FY2017 was filed pre-split; FY2018 is already on today's basis.
    FACTORS = {"2017-09-30": 4.0, "2018-09-29": 1.0}

    def test_shares_scale_up_and_per_share_scales_down(self):
        values = self.values()
        edgar_provider._apply_split_adjustment(values, self.FACTORS)
        assert values["shares_diluted"]["2017-09-30"] == pytest.approx(21.0e9)
        assert values["eps_diluted"]["2017-09-30"] == pytest.approx(2.3025)
        assert values["shares_basic"]["2017-09-30"] == pytest.approx(20.88e9)

    def test_the_latest_basis_is_left_alone(self):
        values = self.values()
        edgar_provider._apply_split_adjustment(values, self.FACTORS)
        assert values["eps_diluted"]["2018-09-29"] == 2.98
        assert values["shares_diluted"]["2018-09-29"] == 20.0e9

    def test_dollar_totals_are_untouched(self):
        """A split moves no money. Revenue and net income are as filed."""
        values = self.values()
        edgar_provider._apply_split_adjustment(values, self.FACTORS)
        assert values["net_income"]["2017-09-30"] == 48.35e9
        assert values["revenue"]["2017-09-30"] == 229.23e9

    def test_eps_times_shares_still_equals_net_income(self):
        """
        The reason one factor is shared across the rows rather than each concept
        being reconstructed on its own: the identity has to survive.
        """
        values = self.values()
        edgar_provider._apply_split_adjustment(values, self.FACTORS)
        for period in ("2017-09-30", "2018-09-29"):
            product = values["eps_diluted"][period] * values["shares_diluted"][period]
            assert product == pytest.approx(values["net_income"][period], rel=0.01)

    def test_a_period_without_a_factor_keeps_its_filed_value(self):
        """No factor means nothing to restate onto — better a visible step than
        an invented one."""
        values = self.values()
        edgar_provider._apply_split_adjustment(values, {"2018-09-29": 1.0})
        assert values["eps_diluted"]["2017-09-30"] == 9.21

    def test_no_factors_at_all_is_a_no_op(self):
        values = self.values()
        edgar_provider._apply_split_adjustment(values, {})
        assert values == self.values()
