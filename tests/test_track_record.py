"""Tests for the per-stock track record (`track_record`, `edgar_provider`).

The ranking has always measured a decade of durability for every US filer and
then kept only five pillar scores; the stock window showed a language model's
guess in their place. This exposes the measurements themselves, so the tests
that matter are about not misrepresenting them:

  * the metric set must come from the ranking's own pillar spec, or the two
    surfaces drift and "why is this ranked here" stops being answerable, and
  * a rate that spans a stock split must describe the company rather than the
    split. EDGAR restates the two prior years for a split and nothing restates
    the ones before, so Apple's assembled share count reads as +11.8%/yr of
    issuance across a decade in which it retired a quarter of its shares. The
    series is rebuilt from same-filing ratios before any rate is taken.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import buffett_rank  # noqa: E402
import track_record  # noqa: E402
from edgar_concepts import all_concepts  # noqa: E402
from track_record import _format, labelled_metrics  # noqa: E402

# Taken from the concept chain rather than hardcoded: the reconstruction resolves
# the concept to its tags, so a fixture keyed by the wrong tag would silently test
# nothing — it would find no filing and leave the series untouched.
SHARES_TAG = all_concepts()["shares_diluted"][0]


GENERIC_METRICS = {
    "roe_median": 118.97,
    "roic_median": 55.23,
    "roa_median": 21.73,
    "gross_margin_median": 40.43,
    "roe_years_above_15": 1.0,
    "roe_observation_years": 10.0,
    "debt_to_equity": 3.87,
    "interest_coverage": None,
    "current_ratio": 0.89,
    "net_debt_to_owner_earnings": 0.6,
    "roe_stdev": 57.3,
    "revenue_growth_stdev": 0.1,
    "fcf_margin_stdev": 0.02,
    "negative_owner_earnings_years": 0,
    "owner_earnings_years": 10.0,
    "revenue_cagr": 0.076,
    "owner_earnings_cagr": 0.071,
    "book_value_per_share_cagr": -0.159,
    "share_count_cagr": 0.118,
    "incremental_roic": None,
}


class TestFormatting:
    @pytest.mark.parametrize(
        "value,unit,denominator,expected",
        [
            (13.41, "percent", None, "13.4%"),
            (0.076, "cagr", None, "+7.6%/yr"),
            (-0.031, "cagr", None, "-3.1%/yr"),
            (2.62, "points", None, "2.6 pts"),
            (7.42, "times", None, "7.4×"),
            (3.87, "ratio", None, "3.87"),
        ],
    )
    def test_units(self, value, unit, denominator, expected):
        assert _format(value, unit, denominator) == expected

    def test_a_share_of_years_is_shown_as_the_count_it_came_from(self):
        """ "7 of 10 years" is the claim; "70%" hides how long the record is."""
        assert _format(0.7, "share", 10.0) == "7 of 10 years"

    def test_a_share_without_its_denominator_stays_a_share(self):
        assert _format(0.7, "share", None) == "70% of years"

    def test_a_count_carries_its_span(self):
        assert _format(2, "years", 10.0) == "2 of 10 years"

    def test_nothing_measured_formats_to_nothing(self):
        assert _format(None, "percent", None) is None


class TestPillarComposition:
    def test_the_metric_set_is_the_ranking_s_own(self):
        """
        Not a second list maintained here. If a metric is added to a pillar the
        stock window shows it automatically, and it cannot show one the ranking
        does not score.
        """
        groups = labelled_metrics(GENERIC_METRICS, "generic")
        shown = {item["key"] for group in groups for item in group["items"]}
        expected = {
            key
            for entries in buffett_rank.GENERIC_PILLARS.values()
            for key, _ in entries
        }
        assert shown == expected

    def test_every_model_is_renderable(self):
        """A bank must not be described with an industrial's metrics."""
        for model in buffett_rank.PILLARS_BY_MODEL:
            groups = labelled_metrics({}, model)
            assert groups, f"{model} produced no groups"
            for group in groups:
                assert group["title"] and not group["title"].endswith("_")
                for item in group["items"]:
                    assert item["label"] and "_" not in item["label"]

    def test_an_unmeasurable_metric_is_kept_as_a_gap(self):
        """Dropping it would read as "not applicable" rather than "not known"."""
        groups = labelled_metrics(GENERIC_METRICS, "generic")
        strength = next(g for g in groups if g["key"] == "financial_strength")
        coverage = next(i for i in strength["items"] if i["key"] == "interest_coverage")
        assert coverage["value"] is None
        assert coverage["display"] is None

    def test_direction_comes_from_the_pillar_spec(self):
        """A falling share count is a return of capital, not a shrinking company."""
        groups = labelled_metrics(GENERIC_METRICS, "generic")
        allocation = next(g for g in groups if g["key"] == "capital_allocation")
        shares = next(i for i in allocation["items"] if i["key"] == "share_count_cagr")
        assert shares["higher_is_better"] is False


class TestSplitReconstruction:
    """
    `split_consistent_series` against a fabricated store, so the test does not
    depend on a 1.4 GB fact database being present.
    """

    @pytest.fixture
    def store(self, monkeypatch):
        import edgar_provider

        class FakeStore:
            def __init__(self):
                self.by_accession = {}

            def get_tag_series_by_accession(self, cik, tags, as_of=None):
                return self.by_accession

        fake = FakeStore()
        monkeypatch.setattr(edgar_provider, "get_store", lambda: fake)
        return fake

    def _patch_assembled(self, monkeypatch, values):
        import edgar_provider

        monkeypatch.setattr(
            edgar_provider,
            "get_concept_values",
            lambda cik, concepts=None: {"shares_diluted": values},
        )

    def test_a_split_step_is_undone(self, store, monkeypatch):
        """
        Apple's shape. The assembled series reads 5.25bn in FY2017 against
        20.0bn in FY2018 — the 2020 4:1 split, applied to the years still being
        restated and to no others. The FY2019 10-K reports both years on one
        basis, which is what recovers the true FY2017 count.
        """
        import edgar_provider

        self._patch_assembled(monkeypatch, {"2017-09-30": 5.25e9, "2018-09-29": 20.0e9})
        store.by_accession = {
            SHARES_TAG: {
                "0000320193-19-000119": {"2017-09-30": 21.0e9, "2018-09-29": 20.0e9}
            }
        }
        fixed = edgar_provider.split_consistent_series("0000320193", "shares_diluted")
        # Anchored on the newest value, which is already on today's basis.
        assert fixed["2018-09-29"] == 20.0e9
        assert fixed["2017-09-30"] == pytest.approx(21.0e9)

    def test_real_issuance_survives(self, store, monkeypatch):
        """
        Realty Income issues equity constantly — genuine dilution the ranking
        must keep seeing. A fix that flattened large share growth would erase
        the REIT signal it exists to catch.
        """
        import edgar_provider

        self._patch_assembled(monkeypatch, {"2024-12-31": 800e6, "2025-12-31": 920e6})
        store.by_accession = {
            SHARES_TAG: {
                "0000726728-26-000012": {"2024-12-31": 800e6, "2025-12-31": 920e6}
            }
        }
        fixed = edgar_provider.split_consistent_series("0000726728", "shares_diluted")
        assert fixed["2024-12-31"] == pytest.approx(800e6)
        assert fixed["2025-12-31"] == pytest.approx(920e6)

    def test_unverifiable_pairs_are_left_alone(self, store, monkeypatch):
        """
        No filing covers both years, so there is no evidence of a step. Rewriting
        the series on a guess would be worse than leaving it: a company whose
        filings merely do not overlap has done nothing wrong.
        """
        import edgar_provider

        self._patch_assembled(monkeypatch, {"2016-12-31": 100e6, "2017-12-31": 400e6})
        store.by_accession = {
            SHARES_TAG: {"a": {"2016-12-31": 100e6}, "b": {"2017-12-31": 400e6}}
        }
        fixed = edgar_provider.split_consistent_series("0000000001", "shares_diluted")
        assert fixed["2016-12-31"] == pytest.approx(100e6)
        assert fixed["2017-12-31"] == pytest.approx(400e6)

    def test_two_splits_chain(self, store, monkeypatch):
        """
        NVIDIA crossed a 4:1 and then a 10:1. Each step has to be undone against
        the anchor as it stands after the later ones, not against the raw value.
        """
        import edgar_provider

        self._patch_assembled(
            monkeypatch,
            {"2022-01-30": 2.5e9, "2023-01-29": 2.5e9, "2024-01-28": 25.0e9},
        )
        store.by_accession = {
            SHARES_TAG: {
                # FY2024 10-K, post 10:1: restates FY2023 but not FY2022.
                "a": {"2023-01-29": 25.0e9, "2024-01-28": 25.0e9},
                # FY2023 10-K, post 4:1 only.
                "b": {"2022-01-30": 2.5e9, "2023-01-29": 2.5e9},
            }
        }
        fixed = edgar_provider.split_consistent_series("0001045810", "shares_diluted")
        assert fixed["2024-01-28"] == 25.0e9
        assert fixed["2023-01-29"] == pytest.approx(25.0e9)
        assert fixed["2022-01-30"] == pytest.approx(25.0e9)

    def test_a_single_period_is_returned_unchanged(self, store, monkeypatch):
        import edgar_provider

        self._patch_assembled(monkeypatch, {"2025-12-31": 1.0})
        assert edgar_provider.split_consistent_series(
            "0000000001", "shares_diluted"
        ) == {"2025-12-31": 1.0}


class TestBuild:
    def test_a_company_with_no_filings_still_answers(self, monkeypatch):
        """
        An unmeasurable company explains itself rather than 500ing — the same
        contract the exclusions list honours.
        """
        import buffett_metrics

        empty = buffett_metrics.CompanyMetrics(
            cik="0000000001", symbol="XYZ", name="XYZ", model="generic"
        )
        empty.gate_failures = ["no_fundamentals"]

        monkeypatch.setattr(
            track_record.buffett_metrics, "compute_metrics", lambda *a: empty
        )
        monkeypatch.setattr(track_record, "_ranked_row", lambda symbol: None)

        record = track_record.build("xyz", "0000000001")
        assert record["symbol"] == "XYZ"
        assert record["period_count"] == 0
        assert "no_fundamentals" in record["gate_failures"]
        assert record["groups"]


class TestRevisionHistory:
    """
    `edgar_provider.revisions` — numbers a company changed after first reporting
    them, which only exists because the fact store keeps every accession instead
    of overwriting. A vendor feed serves the current view and has thrown the rest
    away.

    The tests are mostly about what must *not* be counted. Almost every naive
    version of this drowns in false positives: a tag switch, a stock split, or a
    rounding difference each look exactly like a restatement in the raw data.
    """

    REVENUE_TAG = all_concepts()["revenue"][0]

    @pytest.fixture
    def store(self, monkeypatch):
        import edgar_provider

        class FakeStore:
            def __init__(self):
                self.history = {}

            def get_tag_revisions(self, cik, tags, as_of=None):
                return {k: v for k, v in self.history.items() if k[0] in set(tags)}

        fake = FakeStore()
        monkeypatch.setattr(edgar_provider, "get_store", lambda: fake)
        return fake

    def _patch_provenance(self, monkeypatch, mapping):
        import edgar_provider

        monkeypatch.setattr(
            edgar_provider, "get_concept_provenance", lambda cik, concepts=None: mapping
        )

    def test_a_restated_figure_is_found(self, store, monkeypatch):
        """Microsoft's shape: FY2017 revenue moved on retrospective ASC 606."""
        import edgar_provider

        store.history = {
            (self.REVENUE_TAG, "2017-06-30"): [
                ("2017-08-02", 89.95e9, "10-K"),
                ("2019-08-01", 96.57e9, "10-K"),
            ]
        }
        self._patch_provenance(
            monkeypatch, {"revenue": {"2017-06-30": self.REVENUE_TAG}}
        )

        found = edgar_provider.revisions("0000789019")
        assert len(found) == 1
        assert found[0]["original"] == 89.95e9
        assert found[0]["current"] == 96.57e9
        assert found[0]["change_pct"] == pytest.approx(7.36, abs=0.05)
        assert found[0]["first_filed"] == "2017-08-02"

    def test_a_figure_reported_once_is_not_a_revision(self, store, monkeypatch):
        import edgar_provider

        store.history = {
            (self.REVENUE_TAG, "2017-06-30"): [("2017-08-02", 89.95e9, "10-K")]
        }
        self._patch_provenance(
            monkeypatch, {"revenue": {"2017-06-30": self.REVENUE_TAG}}
        )
        assert edgar_provider.revisions("0000789019") == []

    def test_rounding_is_not_a_restatement(self, store, monkeypatch):
        import edgar_provider

        store.history = {
            (self.REVENUE_TAG, "2017-06-30"): [
                ("2017-08-02", 89.95e9, "10-K"),
                ("2019-08-01", 89.99e9, "10-K"),
            ]
        }
        self._patch_provenance(
            monkeypatch, {"revenue": {"2017-06-30": self.REVENUE_TAG}}
        )
        assert edgar_provider.revisions("0000789019") == []

    def test_only_the_tag_that_answers_is_reported(self, store, monkeypatch):
        """
        Boeing's 2017 equity otherwise appears twice — once from the concept's
        preferred tag and once from a fallback holding a narrower figure — which
        reads as two restatements of one line item.
        """
        import edgar_provider

        chain = all_concepts()["revenue"]
        if len(chain) < 2:
            pytest.skip("revenue concept has no fallback tag to collide with")
        store.history = {
            (chain[0], "2017-12-31"): [
                ("2018-01-01", 100.0, "10-K"),
                ("2020-01-01", 150.0, "10-K"),
            ],
            (chain[1], "2017-12-31"): [
                ("2018-01-01", 10.0, "10-K"),
                ("2020-01-01", 90.0, "10-K"),
            ],
        }
        self._patch_provenance(monkeypatch, {"revenue": {"2017-12-31": chain[0]}})

        found = edgar_provider.revisions("0000012927")
        assert len(found) == 1
        assert found[0]["tag"] == chain[0]

    def test_splits_are_excluded_outright(self, store, monkeypatch):
        """
        A 4:1 split rescales every prior-year share count and EPS by exactly 4x.
        Counting those would bury the real revisions under events that revise
        nothing.
        """
        import edgar_provider

        shares_tag = all_concepts()["shares_diluted"][0]
        store.history = {
            (shares_tag, "2017-09-30"): [
                ("2017-11-03", 5.25e9, "10-K"),
                ("2021-10-29", 21.0e9, "10-K"),
            ]
        }
        self._patch_provenance(
            monkeypatch, {"shares_diluted": {"2017-09-30": shares_tag}}
        )
        assert edgar_provider.revisions("0000320193") == []

    def test_largest_revisions_come_first(self, store, monkeypatch):
        import edgar_provider

        income_tag = all_concepts()["net_income"][0]
        store.history = {
            (self.REVENUE_TAG, "2019-12-31"): [
                ("2020-01-01", 100.0, "10-K"),
                ("2022-01-01", 105.0, "10-K"),
            ],
            (income_tag, "2019-12-31"): [
                ("2020-01-01", 10.0, "10-K"),
                ("2022-01-01", 20.0, "10-K"),
            ],
        }
        self._patch_provenance(
            monkeypatch,
            {
                "revenue": {"2019-12-31": self.REVENUE_TAG},
                "net_income": {"2019-12-31": income_tag},
            },
        )
        found = edgar_provider.revisions("0000000001")
        assert [row["concept"] for row in found] == ["net_income", "revenue"]


class TestMoneyFormatting:
    @pytest.mark.parametrize(
        "value,expected",
        [
            (1.95e9, "$1.95bn"),
            (-244e6, "-$244.00m"),
            (1.2e12, "$1.20tn"),
            (4500.0, "$4.50k"),
            (-12.0, "-$12"),
        ],
    )
    def test_scales(self, value, expected):
        assert track_record._money(value) == expected


class TestStressResponse:
    """
    `track_record.stress_response` — what each downturn did to a business.

    The durability signal a five-year window structurally cannot express: a
    window ending in 2025 begins in 2021 and contains no recession at all. The
    tests are mostly about the ways this arithmetic misleads if written
    naively — a fall measured from a loss, a year compared with itself, or a
    company that was not listed reported as one that sailed through.
    """

    @staticmethod
    def values(revenue=None, net_income=None, ocf=None, capex=None):
        def series(mapping):
            return {f"{year}-12-31": value for year, value in (mapping or {}).items()}

        return {
            "revenue": series(revenue),
            "net_income": series(net_income),
            "operating_cash_flow": series(ocf),
            "capex": series(capex),
        }

    def test_a_fall_and_its_recovery(self):
        values = self.values(
            revenue={2018: 100.0, 2019: 110.0, 2020: 40.0, 2021: 90.0, 2022: 115.0}
        )
        covid = next(
            w for w in track_record.stress_response(values) if w["key"] == "covid"
        )
        item = next(i for i in covid["items"] if i["metric"] == "revenue")
        assert item["peak_year"] == 2019
        assert item["trough_year"] == 2020
        assert item["change_pct"] == pytest.approx(-63.6, abs=0.1)
        assert item["recovered_year"] == 2022
        assert item["recovery_display"] == "back in 2022"

    def test_a_business_that_never_fell_has_nothing_to_recover(self):
        values = self.values(
            revenue={2018: 100.0, 2019: 110.0, 2020: 120.0, 2021: 130.0}
        )
        covid = next(
            w for w in track_record.stress_response(values) if w["key"] == "covid"
        )
        item = next(i for i in covid["items"] if i["metric"] == "revenue")
        assert item["change_pct"] > 0
        assert item["recovery_display"] is None

    def test_a_fall_that_never_came_back_says_so(self):
        values = self.values(revenue={2019: 100.0, 2020: 50.0, 2021: 60.0, 2022: 70.0})
        covid = next(
            w for w in track_record.stress_response(values) if w["key"] == "covid"
        )
        item = next(i for i in covid["items"] if i["metric"] == "revenue")
        assert item["recovered_year"] is None
        assert item["recovery_display"] == "not back to its peak"

    def test_a_loss_is_not_a_peak_to_fall_from(self):
        """
        Delta lost $8.9bn in 2008 and $1.2bn in 2009. Measured against a
        negative peak that is "+86%", which reads as growth through the crisis
        to anyone looking at it.
        """
        values = self.values(net_income={2008: -8900.0, 2009: -1200.0, 2010: 500.0})
        gfc = next(w for w in track_record.stress_response(values) if w["key"] == "gfc")
        assert all(i["metric"] != "net_income" for i in gfc["items"])

    def test_one_year_is_not_a_peak_to_trough(self):
        """
        The windows overlap at 2008 on purpose — the downturn straddles it. A
        company whose only observation is 2008 must not compare that year with
        itself and report a 0% fall.
        """
        values = self.values(revenue={2008: 100.0})
        gfc = next(w for w in track_record.stress_response(values) if w["key"] == "gfc")
        assert gfc["covered"] is False
        assert gfc["items"] == []

    def test_a_company_that_was_not_filing_is_not_a_company_that_held_up(self):
        values = self.values(revenue={2019: 100.0, 2020: 80.0})
        windows = {w["key"]: w for w in track_record.stress_response(values)}
        assert windows["gfc"]["covered"] is False
        assert windows["covid"]["covered"] is True

    def test_free_cash_flow_nets_capex_off_as_a_positive_outflow(self):
        """EDGAR reports capex as a payment, the sign convention buffett_metrics uses."""
        values = self.values(
            ocf={2019: 100.0, 2020: 60.0, 2021: 120.0},
            capex={2019: 20.0, 2020: 10.0, 2021: 20.0},
        )
        covid = next(
            w for w in track_record.stress_response(values) if w["key"] == "covid"
        )
        item = next(i for i in covid["items"] if i["metric"] == "free_cash_flow")
        # Peak 80 in 2019, trough 50 in 2020.
        assert item["change_pct"] == pytest.approx(-37.5)
