"""Tests for the quarterly EDGAR series behind the Financials tab.

yfinance returns five quarters. The SEC XBRL store can reach every quarter a
filer has tagged since 2009, but only after two problems are solved:

  * a 10-Q files the three-month *and* the year-to-date figure for one tag under
    one accession, so the period start has to be part of the key or they
    overwrite each other, and
  * most filers tag the cash-flow statement year-to-date only. Q2 exists solely
    as (six months - three months), and Q4 only ever as (full year - nine
    months) — no 10-Q covers it.

`_derive_quarterly_series` does that differencing. The rule it must never break
is that a missing rung produces a *missing* quarter rather than a plausible
wrong one: (full year - six months) is two quarters of cash flow, and emitting
it as Q4 would be a fabrication.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from edgar_provider import (  # noqa: E402
    _apply_canonical_ends,
    _canonical_period_ends,
    _derive_quarterly_series,
    _filing_basis_factors,
    _spans_on_latest_basis,
    parse_company_quarterly_facts,
)


def ytd(year: int, *values: float) -> dict:
    """A calendar-year filer's year-to-date ladder: Q1, H1, 9M, FY."""
    ends = [f"{year}-03-31", f"{year}-06-30", f"{year}-09-30", f"{year}-12-31"]
    start = f"{year}-01-01"
    return {(start, end): (val, "USD") for end, val in zip(ends, values)}


class TestDeriveQuarterlySeries:
    def test_differences_a_year_to_date_ladder_into_four_quarters(self):
        # 10, 25, 45, 70 cumulative -> 10, 15, 20, 25 per quarter.
        series = _derive_quarterly_series(ytd(2024, 10, 25, 45, 70))
        assert series["2024-03-31"][0] == 10
        assert series["2024-06-30"][0] == 15
        assert series["2024-09-30"][0] == 20
        # Q4 comes only from the 10-K's full year minus the nine-month figure.
        assert series["2024-12-31"][0] == 25

    def test_as_filed_quarters_win_over_derived_ones(self):
        spans = dict(ytd(2024, 10, 25, 45, 70))
        # The same filing also tags Apr-Jun directly, and rounding in the
        # cumulative figures must not override what the filer actually said.
        spans[("2024-04-01", "2024-06-30")] = (15.4, "USD")
        series = _derive_quarterly_series(spans)
        assert series["2024-06-30"][0] == 15.4

    def test_refuses_to_difference_across_a_missing_rung(self):
        spans = {
            ("2024-01-01", "2024-03-31"): (10.0, "USD"),
            ("2024-01-01", "2024-06-30"): (25.0, "USD"),
            # No nine-month figure: the full year minus six months is two
            # quarters, not Q4.
            ("2024-01-01", "2024-12-31"): (70.0, "USD"),
        }
        series = _derive_quarterly_series(spans)
        assert series["2024-03-31"][0] == 10
        assert series["2024-06-30"][0] == 15
        assert "2024-12-31" not in series
        assert "2024-09-30" not in series

    def test_an_annual_only_tag_yields_no_quarters(self):
        # A single twelve-month duration is a year. Emitting it as the quarter
        # that ends on the same day would overstate it fourfold.
        series = _derive_quarterly_series({("2024-01-01", "2024-12-31"): (70.0, "USD")})
        assert series == {}

    def test_instants_pass_through_as_the_quarter_end_balance(self):
        # Balance-sheet facts have no start; each is already a point in time.
        series = _derive_quarterly_series(
            {("", "2024-03-31"): (500.0, "USD"), ("", "2024-06-30"): (520.0, "USD")}
        )
        assert series["2024-03-31"][0] == 500
        assert series["2024-06-30"][0] == 520

    def test_fiscal_years_do_not_leak_into_each_other(self):
        # Two ladders, two starts. Q1 of the later year must not be computed as
        # (its own figure - the prior year's close).
        spans = {**ytd(2023, 8, 20, 36, 56), **ytd(2024, 10, 25, 45, 70)}
        series = _derive_quarterly_series(spans)
        assert series["2024-03-31"][0] == 10
        assert series["2023-12-31"][0] == 20

    def test_a_fifty_three_week_quarter_still_counts_as_one(self):
        # Retail calendars close a 14-week quarter to keep the year aligned;
        # rejecting it would drop a quarter from every 52/53-week filer.
        series = _derive_quarterly_series(
            {
                ("2024-01-28", "2024-04-27"): (10.0, "USD"),
                ("2024-01-28", "2024-08-03"): (25.0, "USD"),
            }
        )
        assert series["2024-08-03"][0] == 15

    def test_a_weighted_average_is_never_differenced(self):
        """
        Share counts are averages over their period, not sums across it.

        Meta's FY2025 diluted average was 2.574bn and its nine-month average
        2.578bn; differencing them produced a fourth quarter of *minus* four
        million shares, which then flowed into every per-share ratio.
        """
        spans = {
            ("2025-01-01", "2025-03-31"): (2.590e9, "shares"),
            ("2025-01-01", "2025-06-30"): (2.580e9, "shares"),
            ("2025-01-01", "2025-09-30"): (2.578e9, "shares"),
            ("2025-01-01", "2025-12-31"): (2.574e9, "shares"),
            ("2025-04-01", "2025-06-30"): (2.570e9, "shares"),
            ("2025-07-01", "2025-09-30"): (2.572e9, "shares"),
        }
        series = _derive_quarterly_series(spans, additive=False)
        # The filed three-month averages stand for the quarters that have one.
        assert series["2025-06-30"][0] == 2.570e9
        assert series["2025-09-30"][0] == 2.572e9
        # Q4 has none, so the shortest duration ending there — the year — is
        # used. What matters is that it is a real filed count, not a difference.
        assert series["2025-12-31"][0] == 2.574e9
        assert all(value > 0 for value, _unit in series.values())

    def test_flows_are_still_differenced(self):
        """The non-additive rule must not leak into the ordinary case."""
        series = _derive_quarterly_series(ytd(2024, 10, 25, 45, 70), additive=True)
        assert series["2024-12-31"][0] == 25

    def test_negative_movement_is_kept(self):
        # A loss-making quarter inside a profitable year: the difference is
        # negative and must survive as such.
        series = _derive_quarterly_series(ytd(2024, 10, 25, 20, 40))
        assert series["2024-09-30"][0] == -5

    def test_a_sixteen_week_fourth_quarter_is_derived(self):
        """
        Costco runs a 12/12/12/16-week year, and its fourth quarter is 112 days.

        With the band capped at a 14-week quarter that Q4 was neither taken as
        filed nor reachable by differencing, so it was missing from all eighteen
        fiscal years the store covers — an empty column every September.
        """
        spans = {
            ("2024-09-02", "2024-11-24"): (62.15, "USD"),  # Q1, 12 weeks
            ("2024-09-02", "2025-02-16"): (125.87, "USD"),  # H1, 24 weeks
            ("2024-09-02", "2025-05-11"): (189.07, "USD"),  # 9M, 36 weeks
            ("2024-09-02", "2025-08-31"): (275.23, "USD"),  # FY, 52 weeks
        }
        series = _derive_quarterly_series(spans)
        # 275.23 - 189.07: the quarter no 10-Q covers, on a 16-week close.
        assert series["2025-08-31"][0] == pytest.approx(86.16)
        assert series["2025-05-11"][0] == pytest.approx(63.20)

    def test_a_half_year_is_still_not_a_quarter(self):
        # The widened ceiling must not reach the shortest rung of the ladder:
        # a 24-week year-to-date figure is two quarters, not one.
        series = _derive_quarterly_series(
            {("2024-09-02", "2025-02-16"): (125.87, "USD")}
        )
        assert series == {}


class TestCanonicalPeriodEnds:
    """
    One mistyped period end must not become a whole extra quarter.

    NVIDIA's FY2012 10-K re-filed Q2 FY2011 under an end of 2010-07-31 where its
    three earlier filings — and every balance-sheet instant for the quarter — say
    2010-08-01. Keyed on the end alone that put the same $811m quarter in the
    series twice, and the four columns behind a trailing-twelve-month figure then
    covered six months.
    """

    def nvidia_spans(self):
        return {
            "Revenues": {
                ("2010-02-01", "2010-08-01"): (1813.0, "USD"),  # H1 year-to-date
                ("2010-05-03", "2010-08-01"): (811.2, "USD"),  # Q2 as filed
                ("2010-05-03", "2010-07-31"): (811.2, "USD"),  # the same Q2, typo
            },
            "Assets": {("", "2010-08-01"): (3731.0, "USD")},
        }

    def test_the_corroborated_end_wins(self):
        canonical = _canonical_period_ends(self.nvidia_spans())
        assert canonical["2010-07-31"] == "2010-08-01"
        assert canonical["2010-08-01"] == "2010-08-01"

    def test_the_duplicate_quarter_disappears(self):
        spans = self.nvidia_spans()
        canonical = _canonical_period_ends(spans)
        series = _derive_quarterly_series(
            _apply_canonical_ends(spans["Revenues"], canonical)
        )
        assert "2010-07-31" not in series
        assert series["2010-08-01"][0] == pytest.approx(811.2)

    def test_a_typo_never_overwrites_the_value_filed_under_the_real_end(self):
        # Same period, two ends, *different* values: the corroborated end keeps
        # its own figure rather than inheriting the mistaken filing's.
        spans = {
            "Revenues": {
                ("2010-05-03", "2010-08-01"): (811.2, "USD"),
                ("2010-05-03", "2010-07-31"): (999.9, "USD"),
            },
            "Assets": {("", "2010-08-01"): (3731.0, "USD")},
        }
        canonical = _canonical_period_ends(spans)
        rewritten = _apply_canonical_ends(spans["Revenues"], canonical)
        assert rewritten[("2010-05-03", "2010-08-01")][0] == pytest.approx(811.2)

    def test_real_quarter_ends_are_never_merged(self):
        # Consecutive quarters are a quarter apart, nowhere near the tolerance.
        spans = {"Revenues": dict(ytd(2024, 10, 25, 45, 70))}
        canonical = _canonical_period_ends(spans)
        assert set(canonical.values()) == {
            "2024-03-31",
            "2024-06-30",
            "2024-09-30",
            "2024-12-31",
        }

    def test_a_52_53_week_end_moving_year_on_year_is_not_a_duplicate(self):
        # A 52/53-week filer's year end shifts by up to a week between years.
        # Those are two periods a year apart, not one tagged twice.
        spans = {
            "Revenues": {
                ("2010-02-01", "2011-01-30"): (3543.0, "USD"),
                ("2009-01-26", "2010-01-31"): (3326.0, "USD"),
            }
        }
        canonical = _canonical_period_ends(spans)
        assert canonical["2011-01-30"] == "2011-01-30"
        assert canonical["2010-01-31"] == "2010-01-31"


class TestParseQuarterlyFacts:
    def payload(self, entries):
        return {
            "cik": 1326801,
            "facts": {"us-gaap": {"Revenues": {"units": {"USD": entries}}}},
        }

    def test_keeps_both_the_quarter_and_the_year_to_date_figure(self):
        rows = parse_company_quarterly_facts(
            self.payload(
                [
                    {
                        "start": "2024-04-01",
                        "end": "2024-06-30",
                        "val": 15,
                        "form": "10-Q",
                        "accn": "a-1",
                        "filed": "2024-07-25",
                    },
                    {
                        "start": "2024-01-01",
                        "end": "2024-06-30",
                        "val": 25,
                        "form": "10-Q",
                        "accn": "a-1",
                        "filed": "2024-07-25",
                    },
                ]
            ),
            wanted_tags={"Revenues"},
        )
        # One accession, one tag, one end date — two rows, because the start
        # differs. This is the collision the separate table exists to avoid.
        assert len(rows) == 2
        assert {row[2] for row in rows} == {"2024-04-01", "2024-01-01"}

    def test_ignores_forms_that_are_not_periodic_reports(self):
        rows = parse_company_quarterly_facts(
            self.payload(
                [
                    {
                        "start": "2024-04-01",
                        "end": "2024-06-30",
                        "val": 15,
                        "form": "8-K",
                        "accn": "a-1",
                        "filed": "2024-07-25",
                    }
                ]
            ),
            wanted_tags={"Revenues"},
        )
        assert rows == []

    def test_drops_durations_longer_than_a_year(self):
        # Multi-year cumulative facts (development-stage filers tag them) are
        # not a rung on any quarterly ladder.
        rows = parse_company_quarterly_facts(
            self.payload(
                [
                    {
                        "start": "2020-01-01",
                        "end": "2024-06-30",
                        "val": 900,
                        "form": "10-Q",
                        "accn": "a-1",
                        "filed": "2024-07-25",
                    }
                ]
            ),
            wanted_tags={"Revenues"},
        )
        assert rows == []

    def test_instants_are_kept_with_an_empty_start(self):
        rows = parse_company_quarterly_facts(
            self.payload(
                [
                    {
                        "end": "2024-06-30",
                        "val": 500,
                        "form": "10-Q",
                        "accn": "a-1",
                        "filed": "2024-07-25",
                    }
                ]
            ),
            wanted_tags={"Revenues"},
        )
        assert len(rows) == 1
        assert rows[0][2] == ""


class TestSplitBasis:
    """
    A quarterly series has to sit on one split basis before anything is derived.

    The default reader takes the newest filing for each span independently, and
    for a share count that mixes two bases inside one fiscal year: a later 10-K
    restates the annual span it carries as a comparative, while the three
    quarterly spans of that year are never re-filed again. NVIDIA's FY2023
    arrived as one quarter of 25.07bn shares beside three of ~2.5bn, and no
    single per-year factor can reconcile that — it scales both by the same ratio.
    """

    # NVIDIA's FY2023, as the store holds it. The 10-K's annual span was re-filed
    # in the FY2025 10-K after the 10:1 split; the quarters never were.
    # NVIDIA split 10:1 in June 2024. What bridges the pre-split quarters to
    # today's basis is the prior-year comparative every 10-Q carries: each year's
    # filing reports the same quarter the year before's filing did, so
    # consecutive filings overlap even though no single one spans the split.
    Q2_FY25 = ("2024-04-29", "2024-07-28")
    Q2_FY24 = ("2023-05-01", "2023-07-30")
    Q2_FY23 = ("2022-05-02", "2022-07-31")
    Q1_FY23 = ("2022-01-31", "2022-05-01")

    def nvidia_filings(self):
        return [
            # The Q2 FY2025 10-Q, filed after the split: today's basis.
            (
                ("2024-08-28", "acc-fy25q2"),
                {self.Q2_FY25: (24600.0, "shares"), self.Q2_FY24: (24800.0, "shares")},
            ),
            # The Q2 FY2024 10-Q, pre-split. Overlaps the one above on Q2 FY2024,
            # and carries Q2 FY2023 as its own comparative.
            (
                ("2023-08-28", "acc-fy24q2"),
                {self.Q2_FY24: (2480.0, "shares"), self.Q2_FY23: (2516.0, "shares")},
            ),
            # The Q2 FY2023 10-Q, pre-split. Overlaps nothing newer *directly* —
            # only the filing above, which by then is itself rebased.
            (
                ("2022-08-31", "acc-fy23q2"),
                {self.Q2_FY23: (2516.0, "shares"), self.Q1_FY23: (2537.0, "shares")},
            ),
        ]

    def test_an_older_filing_is_rebased_onto_the_newest(self):
        factors = _filing_basis_factors(self.nvidia_filings())
        assert factors[("2024-08-28", "acc-fy25q2")] == pytest.approx(1.0)
        # 24800 / 2480 — the 10:1 split, read off the quarter both filings report.
        assert factors[("2023-08-28", "acc-fy24q2")] == pytest.approx(10.0)

    def test_the_factor_chains_through_an_intermediate_filing(self):
        # The FY2023 10-Q shares no span with the post-split filing. It is
        # bridged through the FY2024 one, which overlaps both.
        factors = _filing_basis_factors(self.nvidia_filings())
        assert factors[("2022-08-31", "acc-fy23q2")] == pytest.approx(10.0)

    def test_every_quarter_lands_on_one_basis(self):
        filings = self.nvidia_filings()
        spans = _spans_on_latest_basis(filings, _filing_basis_factors(filings))
        # The quarters that used to read ~2.5bn beside a 25bn fourth quarter.
        assert spans[self.Q2_FY23][0] == pytest.approx(25160.0)
        assert spans[self.Q1_FY23][0] == pytest.approx(25370.0)
        # And the post-split ones are left exactly as filed.
        assert spans[self.Q2_FY25][0] == pytest.approx(24600.0)
        assert spans[self.Q2_FY24][0] == pytest.approx(24800.0)

    def test_a_filing_nothing_corroborates_keeps_its_figures(self):
        # No overlap anywhere: reshaping it would be inventing a ratio.
        lonely = [
            (("2020-01-01", "a"), {("2019-01-01", "2019-03-31"): (100.0, "shares")}),
            (("2015-01-01", "b"), {("2014-01-01", "2014-03-31"): (50.0, "shares")}),
        ]
        factors = _filing_basis_factors(lonely)
        assert factors[("2015-01-01", "b")] == pytest.approx(1.0)

    def test_a_restatement_of_one_line_does_not_move_the_filing(self):
        # Three spans agree on the basis and one has been restated; the median
        # keeps the odd one out from rescaling everything around it.
        filings = [
            (
                ("2025-01-01", "new"),
                {
                    ("2024-01-01", "2024-03-31"): (100.0, "u"),
                    ("2024-04-01", "2024-06-30"): (100.0, "u"),
                    ("2024-07-01", "2024-09-30"): (555.0, "u"),
                },
            ),
            (
                ("2024-01-01", "old"),
                {
                    ("2024-01-01", "2024-03-31"): (100.0, "u"),
                    ("2024-04-01", "2024-06-30"): (100.0, "u"),
                    ("2024-07-01", "2024-09-30"): (100.0, "u"),
                },
            ),
        ]
        assert _filing_basis_factors(filings)[("2024-01-01", "old")] == pytest.approx(
            1.0
        )

    def test_per_share_figures_move_against_the_split(self):
        """
        Shares multiply and EPS divides, sharing one factor so that
        `EPS x shares = net income` stays true down the table.
        """
        factors = {("2023-02-24", "acc"): 10.0}
        filings = [
            (("2023-02-24", "acc"), {("2022-08-01", "2022-10-30"): (0.58, "USD")})
        ]
        eps = _spans_on_latest_basis(filings, factors, inverse=True)
        assert eps[("2022-08-01", "2022-10-30")][0] == pytest.approx(0.058)

    def test_a_span_the_share_count_never_covered_is_still_rebased(self):
        """
        The factor belongs to the filing, not to the span.

        Apple files a fourth-quarter EPS with no matching three-month share
        count. Keyed by span there would be no factor for it, so it would keep
        the old basis while every sibling quarter moved — the same mixing this
        exists to end, one row further down.
        """
        factors = {("2018-11-05", "10-K"): 4.0}
        filings = [
            (
                ("2018-11-05", "10-K"),
                {
                    ("2017-10-01", "2018-09-29"): (11.91, "USD"),
                    ("2018-07-01", "2018-09-29"): (2.91, "USD"),
                },
            )
        ]
        eps = _spans_on_latest_basis(filings, factors, inverse=True)
        assert eps[("2018-07-01", "2018-09-29")][0] == pytest.approx(0.7275)
        assert eps[("2017-10-01", "2018-09-29")][0] == pytest.approx(2.9775)

    def test_the_newest_filing_wins_the_value_and_its_own_factor(self):
        # A value read from one filing and scaled by another's factor is on
        # neither basis, so selection and correction happen together.
        filings = [
            (("2025-01-01", "new"), {("2024-01-01", "2024-03-31"): (40.0, "u")}),
            (("2024-01-01", "old"), {("2024-01-01", "2024-03-31"): (10.0, "u")}),
        ]
        spans = _spans_on_latest_basis(filings, _filing_basis_factors(filings))
        assert spans[("2024-01-01", "2024-03-31")][0] == pytest.approx(40.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
