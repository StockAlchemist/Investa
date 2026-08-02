"""Tests for the quarterly ratio series behind the Ratios & Trends tab.

A ratio that divides a flow by a level — return on equity, asset turnover — is
meaningless on a raw quarter: one quarter of profit over the whole equity
balance reads as a quarter of the real return and would sit four times below the
annual series for the same company on the same date. `to_trailing_twelve_months`
sums the four quarters ending at each period first, so the quarterly view is the
same measurement sampled four times as often rather than a different one.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from financial_ratios import (  # noqa: E402
    calculate_key_ratios_timeseries,
    to_trailing_twelve_months,
)

QUARTERS = ["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31", "2024-12-31"]


def frame(rows: dict, periods=QUARTERS) -> pd.DataFrame:
    """Newest-first, the way every statement in this system is shaped."""
    return pd.DataFrame(rows, index=pd.to_datetime(periods)).T


class TestTrailingTwelveMonths:
    def test_sums_the_four_quarters_ending_at_each_period(self):
        ttm = to_trailing_twelve_months(frame({"Total Revenue": [40, 30, 20, 10, 5]}))
        assert ttm.loc["Total Revenue"].iloc[0] == 100  # 40+30+20+10
        assert ttm.loc["Total Revenue"].iloc[1] == 65  # 30+20+10+5

    def test_drops_periods_without_a_full_year_behind_them(self):
        # Five quarters of data support two trailing years, not five.
        ttm = to_trailing_twelve_months(frame({"Total Revenue": [40, 30, 20, 10, 5]}))
        assert len(ttm.columns) == 2
        assert list(ttm.columns) == list(pd.to_datetime(QUARTERS[:2]))

    def test_averages_share_counts_instead_of_summing_them(self):
        # Four quarters of a 2.5bn share count summed would report 10bn shares.
        ttm = to_trailing_twelve_months(
            frame({"Diluted Average Shares": [2.4e9, 2.5e9, 2.6e9, 2.5e9, 2.5e9]})
        )
        assert ttm.loc["Diluted Average Shares"].iloc[0] == pytest.approx(2.5e9)

    def test_a_row_no_quarter_reported_stays_missing(self):
        # Not a confident zero: nothing was filed.
        ttm = to_trailing_twelve_months(
            frame({"Gross Profit": [None, None, None, None, None]})
        )
        assert pd.isna(ttm.loc["Gross Profit"].iloc[0])

    def test_too_little_history_yields_no_periods_rather_than_a_short_year(self):
        short = frame({"Total Revenue": [10, 20]}, periods=QUARTERS[:2])
        assert to_trailing_twelve_months(short).empty

    def test_passes_empty_input_through(self):
        assert to_trailing_twelve_months(None) is None
        assert to_trailing_twelve_months(pd.DataFrame()).empty

    def test_refuses_a_window_with_a_quarter_missing_from_it(self):
        """
        Four columns are not a year unless they are four *consecutive* quarters.

        The quarterly series refuses to difference across a missing rung, so a
        hole in it is the designed-for outcome. Summing across one anyway is how
        Apple's four columns ending 2008-12-27 came to span fifteen months.
        """
        gapped = frame(
            {"Total Revenue": [40, 30, 20, 10]},
            ["2025-12-31", "2025-09-30", "2025-06-30", "2024-06-30"],
        )
        assert to_trailing_twelve_months(gapped).empty

    def test_refuses_a_window_holding_the_same_quarter_twice(self):
        """
        NVIDIA tagged Q2 FY2011 twice, a day apart. The four columns ending
        2011-01-30 spanned 183 days and reported $3.353bn of trailing revenue
        against the $3.543bn it filed — a wrong number, not a missing one.
        """
        duplicated = frame(
            {"Total Revenue": [886, 844, 811, 811]},
            ["2011-01-30", "2010-10-31", "2010-08-01", "2010-07-31"],
        )
        assert to_trailing_twelve_months(duplicated).empty

    def test_accepts_a_sixteen_week_fourth_quarter_in_the_window(self):
        # Costco's year is 12/12/12/16 weeks, so a legitimate window spans 280
        # days rather than the 273 a 13-week filer produces.
        retail = frame(
            {"Total Revenue": [86.16, 63.20, 63.72, 62.15]},
            ["2025-08-31", "2025-05-11", "2025-02-16", "2024-11-24"],
        )
        ttm = to_trailing_twelve_months(retail)
        assert list(ttm.columns) == [pd.Timestamp("2025-08-31")]
        assert ttm.loc["Total Revenue"].iloc[0] == pytest.approx(275.23)


class TestQuarterlyMatchesAnnual:
    """
    The point of the trailing window: at a fiscal year end the quarterly series
    must report the same ratio the annual series does. If the two disagree
    there, the toggle is showing two different measurements rather than two
    sampling rates of one.
    """

    def build(self):
        year_end = "2025-12-31"
        quarters = ["2025-12-31", "2025-09-30", "2025-06-30", "2025-03-31"]
        annual_income = frame(
            {"Total Revenue": [1000], "Net Income": [200]}, [year_end]
        )
        annual_balance = frame(
            {"Stockholders Equity": [800], "Total Assets": [2000]}, [year_end]
        )
        quarterly_income = frame(
            {"Total Revenue": [400, 250, 200, 150], "Net Income": [80, 50, 40, 30]},
            quarters,
        )
        quarterly_balance = frame(
            {
                "Stockholders Equity": [800, 780, 760, 740],
                "Total Assets": [2000, 1950, 1900, 1850],
            },
            quarters,
        )
        return annual_income, annual_balance, quarterly_income, quarterly_balance

    def test_return_on_equity_agrees_at_the_fiscal_year_end(self):
        a_inc, a_bs, q_inc, q_bs = self.build()
        annual = calculate_key_ratios_timeseries(a_inc, a_bs)
        quarterly = calculate_key_ratios_timeseries(
            to_trailing_twelve_months(q_inc), q_bs
        )
        stamp = pd.Timestamp("2025-12-31")
        assert quarterly.loc[stamp, "Return on Equity (ROE) (%)"] == pytest.approx(
            annual.loc[stamp, "Return on Equity (ROE) (%)"]
        )

    def test_net_margin_agrees_at_the_fiscal_year_end(self):
        a_inc, a_bs, q_inc, q_bs = self.build()
        annual = calculate_key_ratios_timeseries(a_inc, a_bs)
        quarterly = calculate_key_ratios_timeseries(
            to_trailing_twelve_months(q_inc), q_bs
        )
        stamp = pd.Timestamp("2025-12-31")
        assert quarterly.loc[stamp, "Net Profit Margin (%)"] == pytest.approx(
            annual.loc[stamp, "Net Profit Margin (%)"]
        )

    def test_a_raw_quarter_would_have_been_four_times_too_low(self):
        """What the trailing window exists to prevent."""
        _a_inc, _a_bs, q_inc, q_bs = self.build()
        raw = calculate_key_ratios_timeseries(q_inc, q_bs)
        stamp = pd.Timestamp("2025-12-31")
        # 80 / 800 = 10%, against the real 200/800 = 25%.
        assert raw.loc[stamp, "Return on Equity (ROE) (%)"] == pytest.approx(10.0)


class TestAveragedBalanceWindow:
    """
    The ratios that divide a flow by an *averaged* balance — return on assets,
    asset turnover — have to average it over the same year the flow was earned
    in. Averaging against the previous column is that year at one column a year
    and a single quarter at four, which had the quarterly view disagreeing with
    the annual view about the same fiscal year end.
    """

    QUARTERS = [
        "2025-12-31",
        "2025-09-30",
        "2025-06-30",
        "2025-03-31",
        "2024-12-31",
        "2024-09-30",
        "2024-06-30",
        "2024-03-31",
    ]
    YEARS = ["2025-12-31", "2024-12-31"]

    def build(self):
        annual_income = frame(
            {"Total Revenue": [1000, 880], "Net Income": [200, 176]}, self.YEARS
        )
        annual_balance = frame(
            {
                "Stockholders Equity": [900, 800],
                "Total Assets": [2000, 1800],
            },
            self.YEARS,
        )
        quarterly_income = frame(
            {
                "Total Revenue": [400, 250, 200, 150, 350, 220, 180, 130],
                "Net Income": [80, 50, 40, 30, 70, 44, 36, 26],
            },
            self.QUARTERS,
        )
        # The two fiscal year ends carry the same balances the annual sheet does;
        # the quarters between them drift, which is what a one-column average
        # would wrongly pick up.
        quarterly_balance = frame(
            {
                "Stockholders Equity": [900, 880, 860, 840, 800, 780, 760, 740],
                "Total Assets": [2000, 1960, 1920, 1880, 1800, 1760, 1720, 1680],
            },
            self.QUARTERS,
        )
        return annual_income, annual_balance, quarterly_income, quarterly_balance

    def quarterly(self, periods_per_year):
        _a_inc, _a_bs, q_inc, q_bs = self.build()
        return calculate_key_ratios_timeseries(
            to_trailing_twelve_months(q_inc), q_bs, periods_per_year=periods_per_year
        )

    def annual(self):
        a_inc, a_bs, _q_inc, _q_bs = self.build()
        return calculate_key_ratios_timeseries(a_inc, a_bs)

    @pytest.mark.parametrize("ratio", ["Asset Turnover", "Return on Assets (ROA) (%)"])
    def test_agrees_with_the_annual_series_at_a_fiscal_year_end(self, ratio):
        stamp = pd.Timestamp("2025-12-31")
        assert self.quarterly(4).loc[stamp, ratio] == pytest.approx(
            self.annual().loc[stamp, ratio]
        )

    def test_averaging_over_one_quarter_is_what_used_to_disagree(self):
        """
        The defect this parameter fixes, held in place so it cannot come back.

        Apple FY2025 read 1.205 on the quarterly toggle against 1.149 on the
        annual one, while the copy on both clients promised the same ratio at a
        different sampling rate.
        """
        stamp = pd.Timestamp("2025-12-31")
        one_column_back = self.quarterly(1).loc[stamp, "Asset Turnover"]
        annual = self.annual().loc[stamp, "Asset Turnover"]
        # 1000 / ((2000+1960)/2) rather than 1000 / ((2000+1800)/2).
        assert one_column_back == pytest.approx(1000 / 1980)
        assert annual == pytest.approx(1000 / 1900)
        assert one_column_back != pytest.approx(annual)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
