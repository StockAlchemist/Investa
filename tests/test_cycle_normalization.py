"""Tests for through-cycle normalization of the DCF's starting cash flow.

The five-year window was never a judgement about what "normal" means — it was
the number of annual periods yfinance returned. A window ending in 2025 begins
in 2021 and contains no recession at all, so the figure it produced was a
recent average dressed as a normal year. The SEC-filed history reaches ~19
years, so the through-cycle number Buffett's owner earnings actually describe
is computable now.

What the tests pin is *why it is a margin and not a dollar figure*: the median
of ten years of absolute free cash flow describes a company the size this one
was five years ago, which for anything growing is not conservatism but an
error. The margin is the part that mean-reverts; today's revenue puts it back
on today's scale.
"""

import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import financial_ratios as fr  # noqa: E402


def statements(revenue, ocf, capex, start_year=2016):
    """Income and cash-flow frames, newest-first, like yfinance's."""
    periods = [f"{start_year + i}-12-31" for i in range(len(revenue))]
    columns = pd.to_datetime(periods)
    income = pd.DataFrame([revenue], index=["Total Revenue"], columns=columns)
    cashflow = pd.DataFrame(
        [ocf, capex],
        index=["Operating Cash Flow", "Capital Expenditure"],
        columns=columns,
    )
    return (
        income.sort_index(axis=1, ascending=False),
        cashflow.sort_index(axis=1, ascending=False),
    )


def flat(value, n=10):
    return [value] * n


class TestThroughCycleMargin:
    def test_a_steady_business_reports_its_steady_margin(self):
        income, cashflow = statements(flat(1000.0), flat(250.0), flat(-50.0))
        result = fr.through_cycle_fcf_margin(income, cashflow)
        assert result["margin"] == pytest.approx(0.20)
        assert result["observations"] == 10

    def test_loss_years_are_kept(self):
        """
        Dropping them would score a company that burned cash in four of ten
        years on the six that worked — the most upward-biased thing a
        normalizer can do.
        """
        ocf = [250.0] * 6 + [-100.0] * 4
        income, cashflow = statements(flat(1000.0), ocf, flat(-50.0))
        result = fr.through_cycle_fcf_margin(income, cashflow)
        # Median across six years at +20% and four at -15%.
        assert result["margin"] == pytest.approx(0.20)
        assert result["observations"] == 10

        # And with the losses in the majority the margin goes negative rather
        # than quietly excluding them.
        ocf = [250.0] * 4 + [-100.0] * 6
        income, cashflow = statements(flat(1000.0), ocf, flat(-50.0))
        assert fr.through_cycle_fcf_margin(income, cashflow)["margin"] < 0

    def test_implausible_margins_are_dropped(self):
        """Above 60% is a bad statement mapping, not a spectacular business."""
        ocf = [250.0] * 9 + [5000.0]
        income, cashflow = statements(flat(1000.0), ocf, flat(-50.0))
        result = fr.through_cycle_fcf_margin(income, cashflow)
        assert result["observations"] == 9

    def test_a_short_history_is_not_a_cycle(self):
        income, cashflow = statements(flat(1000.0, 4), flat(250.0, 4), flat(-50.0, 4))
        result = fr.through_cycle_fcf_margin(income, cashflow)
        assert result["margin"] is None
        assert result["observations"] == 4

    def test_missing_capex_years_are_skipped_not_assumed_zero(self):
        """
        Free cash flow without capex is not free cash flow. NVIDIA reports capex
        under two tags across its history and neither covers eleven of its
        nineteen filed years, so this case is common rather than theoretical.
        """
        income, cashflow = statements(flat(1000.0), flat(250.0), flat(-50.0))
        cashflow.loc["Capital Expenditure", cashflow.columns[:5]] = float("nan")
        result = fr.through_cycle_fcf_margin(income, cashflow)
        assert result["observations"] == 5
        assert result["margin"] is None


class TestNormalizedBase:
    INFO = {"symbol": "TEST", "totalRevenue": 2000.0}

    def test_the_cycle_is_applied_to_current_scale(self):
        """
        The point of the whole change. Revenue doubled over the decade, so the
        median of ten years of *dollars* describes the company as it was in the
        middle of that decade; the margin applied to today's revenue describes
        it as it is.
        """
        revenue = [1000.0 + 111.0 * i for i in range(10)]
        ocf = [r * 0.25 for r in revenue]
        capex = [-r * 0.05 for r in revenue]
        income, cashflow = statements(revenue, ocf, capex)

        result = fr.normalized_base_fcf(self.INFO, income, cashflow)
        # 20% of current revenue, not the ~$300 the dollar median would give.
        assert result["fcf"] == pytest.approx(0.20 * 2000.0)
        assert result["normalized"] is True
        assert "through-cycle" in result["method"]

    def test_a_short_history_falls_back_to_the_dollar_median(self):
        """The fallback is exactly the previous behaviour, not a refusal."""
        income, cashflow = statements(flat(1000.0, 4), flat(250.0, 4), flat(-50.0, 4))
        result = fr.normalized_base_fcf(self.INFO, income, cashflow)
        assert result["fcf"] == pytest.approx(200.0)
        assert "median of" in result["method"]

    def test_a_negative_cycle_does_not_veto_a_valuable_short_window(self):
        """
        The cycle may rescale a valuation; it may not take one away. A company
        the five-year window can value is not made unvaluable by looking
        further back — that would be a new refusal smuggled in as a
        normalization.
        """
        ocf = [-100.0] * 6 + [250.0] * 4
        income, cashflow = statements(flat(1000.0), ocf, flat(-50.0))
        assert fr.through_cycle_fcf_margin(income, cashflow)["margin"] < 0

        result = fr.normalized_base_fcf(self.INFO, income, cashflow)
        assert result["fcf"] == pytest.approx(200.0)
        assert "median of" in result["method"]

    def test_no_revenue_means_no_cycle_estimate(self):
        """Without a current scale to apply it to, a margin values nothing."""
        income, cashflow = statements(flat(1000.0), flat(250.0), flat(-50.0))
        result = fr.normalized_base_fcf({"symbol": "TEST"}, income, cashflow)
        assert "median of" in result["method"]

    def test_a_business_that_never_converts_cash_is_still_refused(self):
        income, cashflow = statements(flat(1000.0), flat(-100.0), flat(-50.0))
        result = fr.normalized_base_fcf(self.INFO, income, cashflow)
        assert result["fcf"] is None
        assert result["normalized"] is False
