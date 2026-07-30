"""Tests for the Buffett/value ranking.

The cases here concentrate on the failure modes that would quietly corrupt a
ranking rather than crash it — a missing input scoring as excellent, a gate
firing on absent data, a sector being condemned for normal behaviour. Those are
the errors that produce a confident, wrong answer, which is the only kind this
system can really do damage with.
"""

import os
import sys
import tempfile

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import buffett_metrics as bm
import buffett_rank as br
import buffett_value as bv
import edgar_sic
import universe
from buffett_metrics import CompanyMetrics
from buffett_store import BuffettRankStore
from edgar_provider import _is_annual_duration, parse_company_facts, resolve_concept


# --- concept resolution -----------------------------------------------------


def test_resolve_concept_walks_chain_per_period():
    """
    The core reason the fallback chains exist: an accounting-standard change
    moves a company's revenue to a new tag mid-history. Resolution must stitch
    the two into one continuous series, not pick a winner and lose a decade.
    """
    tag_series = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "2023-12-31": (300.0, "USD"),
            "2022-12-31": (250.0, "USD"),
        },
        "Revenues": {
            "2022-12-31": (999.0, "USD"),  # overlapping year: must NOT win
            "2018-12-31": (100.0, "USD"),
        },
    }
    chain = ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues"]
    values, provenance = resolve_concept(tag_series, chain)

    assert values["2023-12-31"] == 300.0
    assert values["2022-12-31"] == 250.0, "earlier chain entry must take priority"
    assert values["2018-12-31"] == 100.0, "later chain entry fills the gap"
    assert provenance["2018-12-31"] == "Revenues"


def test_annual_duration_filter():
    """52/53-week fiscal calendars are annual; quarters and instants are not."""
    assert _is_annual_duration("2023-01-01", "2023-12-31")
    assert _is_annual_duration("2023-01-01", "2023-12-30")  # 364-day retail year
    assert _is_annual_duration(None, "2023-12-31")  # balance-sheet instant
    assert not _is_annual_duration("2023-10-01", "2023-12-31")  # a quarter


def test_parse_company_facts_keeps_only_annual_reports():
    payload = {
        "cik": 320193,
        "facts": {
            "us-gaap": {
                "NetIncomeLoss": {
                    "units": {
                        "USD": [
                            {
                                "start": "2023-01-01",
                                "end": "2023-12-31",
                                "val": 100,
                                "form": "10-K",
                                "accn": "a-1",
                                "filed": "2024-02-01",
                            },
                            {
                                "start": "2023-07-01",
                                "end": "2023-09-30",
                                "val": 25,
                                "form": "10-Q",
                                "accn": "a-2",
                                "filed": "2023-10-01",
                            },
                        ]
                    }
                },
                "SomeTagWeDoNotTrack": {
                    "units": {
                        "USD": [
                            {"end": "2023-12-31", "val": 1, "form": "10-K", "accn": "x"}
                        ]
                    }
                },
            }
        },
    }
    rows = parse_company_facts(payload, wanted_tags={"NetIncomeLoss"})
    assert len(rows) == 1
    cik, tag, period_end, _start, val, _unit, form, _accn, _filed = rows[0]
    assert (cik, tag, period_end, val, form) == (
        "0000320193",
        "NetIncomeLoss",
        "2023-12-31",
        100,
        "10-K",
    )


# --- the absent-vs-zero rule ------------------------------------------------


def test_total_debt_is_none_when_untagged():
    """
    The most dangerous possible error: an untagged debt line read as zero hands
    the company a perfect leverage score. Absent must stay absent.
    """
    concepts = {"short_term_debt": {}, "long_term_debt": {}}
    assert bm._total_debt(concepts, "2023-12-31") is None


def test_total_debt_sums_reported_components():
    concepts = {
        "short_term_debt": {"2023-12-31": 100.0},
        "long_term_debt": {"2023-12-31": 400.0},
    }
    assert bm._total_debt(concepts, "2023-12-31") == 500.0


def test_total_debt_treats_partial_report_as_zero_for_missing_leg():
    """One leg reported and one absent is a real number, not an unknown."""
    concepts = {"short_term_debt": {}, "long_term_debt": {"2023-12-31": 400.0}}
    assert bm._total_debt(concepts, "2023-12-31") == 400.0


def test_cagr_refuses_non_positive_start():
    """Growing out of a loss has no meaningful growth *rate*."""
    assert bm._cagr([-5.0, 10.0, 20.0]) is None
    assert bm._cagr([0.0, 10.0]) is None
    assert bm._cagr([100.0, 121.0]) == pytest.approx(0.21, abs=1e-9)


# --- gates ------------------------------------------------------------------


def _company(model="generic", periods=15, **metrics):
    company = CompanyMetrics(cik="0000000001", symbol="TEST", name="Test", model=model)
    company.period_count = periods
    company.metrics = metrics
    company.coverage = 1.0
    return company


def test_gate_never_fires_on_missing_data():
    """
    Apple stopped tagging interest expense separately, so its coverage is
    genuinely unknown. Unknown must not be read as a failure — that would
    exclude some of the highest-quality businesses in the market for a
    disclosure choice.
    """
    company = _company(interest_coverage=None, debt_to_equity=None, roe_median=None)
    assert br.evaluate_gates(company) == []


def test_gate_fires_on_known_violation():
    company = _company(interest_coverage=1.1)
    assert "interest_not_covered" in br.evaluate_gates(company)


def test_debt_free_company_skips_the_coverage_gate():
    """No debt is the best possible balance sheet, not a failure to cover interest."""
    company = _company(interest_coverage=None, debt_free=1.0)
    assert br.evaluate_gates(company) == []


def test_bank_is_not_judged_on_industrial_leverage():
    """
    A bank at 8% equity/assets is normally capitalised. Running it through the
    generic solvency gate would exclude the entire sector for operating the way
    banks operate — a bank's liabilities are its raw material.
    """
    bank = _company(
        model="bank",
        equity_to_assets_latest=8.0,
        debt_to_equity=9.0,
        net_debt_to_owner_earnings=25.0,
        roe_median=13.0,
    )
    assert br.evaluate_gates(bank) == []

    industrial = _company(model="generic", net_debt_to_owner_earnings=25.0)
    assert "debt_not_serviceable" in br.evaluate_gates(industrial)


def test_undercapitalised_bank_is_excluded():
    bank = _company(model="bank", equity_to_assets_latest=2.5, roe_median=10.0)
    assert "undercapitalised" in br.evaluate_gates(bank)


def test_short_history_is_excluded():
    assert any(
        "insufficient_history" in f for f in br.evaluate_gates(_company(periods=3))
    )


def test_reit_leverage_gate_uses_debt_to_ffo():
    assert "excessive_leverage" in br.evaluate_gates(
        _company(model="reit", debt_to_ffo=20.0)
    )
    assert br.evaluate_gates(_company(model="reit", debt_to_ffo=6.0)) == []


# --- scoring ----------------------------------------------------------------


def test_confidence_only_ever_demotes():
    assert br.confidence_factor(1.0) == 1.0
    assert br.confidence_factor(0.5) < 1.0
    assert br.confidence_factor(0.0) == pytest.approx(0.5)
    # Even nonsense coverage cannot produce a bonus.
    assert br.confidence_factor(5.0) == 1.0


def test_winsorised_percentile_direction():
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=list("abcd"))
    higher = br._winsorised_percentile(series, higher_is_better=True)
    lower = br._winsorised_percentile(series, higher_is_better=False)
    assert higher["d"] > higher["a"]
    assert lower["a"] > lower["d"]


def test_outlier_cannot_dominate_the_scale():
    """A 10,000% ROE from a near-zero equity base must not flatten everyone else."""
    normal = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0, 20.0])
    with_outlier = pd.concat([normal, pd.Series([100000.0])], ignore_index=True)
    scored = br._winsorised_percentile(with_outlier, higher_is_better=True)
    # The genuine spread across the normal names must survive.
    assert scored.iloc[:6].max() - scored.iloc[:6].min() > 40


def test_missing_metric_does_not_score_as_bad():
    """
    A pillar averages the metrics that resolved. Treating a missing one as zero
    would double-count the data gap, which the confidence factor already handles.
    """
    frame = pd.DataFrame(
        {
            "roe_median": [20.0, 10.0],
            "roic_median": [None, 5.0],
            "roa_median": [None, None],
            "gross_margin_median": [None, None],
            "roe_years_above_15": [None, None],
        },
        index=["A", "B"],
    )
    scores = br._pillar_scores(
        frame, {"returns_on_capital": br.GENERIC_PILLARS["returns_on_capital"]}
    )
    assert scores.loc["A", "returns_on_capital"] > scores.loc["B", "returns_on_capital"]


def test_quality_score_renormalises_over_present_pillars():
    """A REIT has no gross-margin pillar; its missing weight must not act as zeros."""
    pillars = pd.DataFrame(
        {
            "returns_on_capital": [80.0],
            "financial_strength": [80.0],
            "predictability": [None],
            "growth": [None],
            "capital_allocation": [None],
        },
        index=["A"],
    )
    assert br._quality_score(pillars).loc["A"] == pytest.approx(80.0)


def test_financial_strength_keeps_its_weight_despite_a_negative_ic():
    """
    Guards a conclusion that is easy to reach and wrong.

    `scripts/rank_signal_lab.py` reports a clearly negative information
    coefficient for this pillar, which reads as an invitation to delete it. It
    was tried: the strategy lost two points of CAGR and nine of drawdown,
    because the pillar's real work is vetoing levered companies at the very top
    of the composite, where an average rank correlation cannot see. The weight
    is also at the peak of a 0.10-0.35 sweep. See the comment on
    `GENERIC_PILLARS["financial_strength"]` for the measurement.
    """
    assert br.PILLAR_WEIGHTS["financial_strength"] == 0.20

    # Leverage has to stay in the pillar; it is the constituent doing the work.
    generic = dict(br.GENERIC_PILLARS["financial_strength"])
    assert generic["debt_to_equity"] is False, "low debt must score better, not worse"
    assert "current_ratio" in generic
    assert "interest_coverage" in generic

    # Every model needs a solvency pillar — a bank's is equity to assets, not D/E.
    for model, spec in br.PILLARS_BY_MODEL.items():
        assert spec.get("financial_strength"), f"{model} lost its solvency pillar"


def test_combine_applies_weights_and_confidence():
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "quality_score": [80.0],
            "coverage": [1.0],
            "eligible": [True],
        },
        index=["A"],
    )
    result = br.combine(frame, value_scores=pd.Series({"A": 40.0}))
    # 0.6*80 + 0.4*40 = 64, confidence 1.0
    assert result.loc["A", "composite_score"] == pytest.approx(64.0)


def test_combine_falls_back_to_quality_without_value():
    frame = pd.DataFrame(
        {
            "symbol": ["A"],
            "quality_score": [70.0],
            "coverage": [1.0],
            "eligible": [True],
        },
        index=["A"],
    )
    result = br.combine(frame, value_scores=None)
    assert result.loc["A", "composite_score"] == pytest.approx(70.0)


# --- value ------------------------------------------------------------------


def test_no_discounted_cash_flow_reaches_the_value_score():
    """
    The ranking is DCF-free by construction, not by configuration.

    A margin of safety in the frame is the shape the old pipeline produced; if a
    future change reintroduces the column, this asserts it still cannot move a
    rank. The measurement that retired it is in the `buffett_value` docstring.
    """
    assert "margin_of_safety" not in bv.VALUE_WEIGHTS
    assert not hasattr(bv, "conservative_intrinsic_value")

    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "model": ["generic"] * 2,
            "earnings_yield": [5.0, 5.0],
            "fcf_yield": [4.0, 4.0],
            # A would look far cheaper on a DCF; the two must still tie.
            "margin_of_safety": [90.0, -400.0],
        },
        index=["A", "B"],
    )
    scores = bv.score_value(frame)
    assert scores["A"] == pytest.approx(scores["B"])


def test_only_the_weighted_multiples_are_scored():
    """EV/EBIT, P/B and P/S are computed for context and deliberately unscored."""
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B"],
            "model": ["generic"] * 2,
            "earnings_yield": [5.0, 5.0],
            "fcf_yield": [4.0, 4.0],
            "price_to_book": [0.5, 20.0],
            "ev_to_ebit": [3.0, 60.0],
            "price_to_sales": [0.2, 15.0],
        },
        index=["A", "B"],
    )
    scores = bv.score_value(frame)
    assert scores["A"] == pytest.approx(scores["B"]), (
        "unweighted diagnostics must not move the score"
    )


def test_negative_multiples_are_not_treated_as_cheap(monkeypatch):
    """
    A loss-making company has no P/E, and negative book value is not a bargain.

    Exercised through a weighted P/B because no `lower is better` metric carries
    weight today — the guard has to stay tested so that re-adding one is safe.
    """
    monkeypatch.setattr(bv, "VALUE_WEIGHTS", {"price_to_book": 1.0})
    frame = pd.DataFrame(
        {
            "symbol": ["A", "B", "C"],
            "model": ["generic"] * 3,
            "price_to_book": [-5.0, 1.0, 3.0],
        },
        index=["A", "B", "C"],
    )
    scores = bv.score_value(frame)
    assert pd.isna(scores["A"]), "negative book value must not score at all"
    assert scores["B"] > scores["C"], "cheaper positive P/B ranks better"


# --- sector routing ---------------------------------------------------------


@pytest.mark.parametrize(
    "sic,expected",
    [
        (6021, "bank"),  # national commercial bank
        (6798, "reit"),  # REIT
        (6331, "insurer"),  # property & casualty insurance
        (3571, "generic"),  # electronic computers
        (6211, "generic"),  # broker-dealer: fee business, ordinary FCF
        (None, "generic"),  # unknown must not be routed to a special model
    ],
)
def test_sic_routing(sic, expected):
    assert edgar_sic.model_for_sic(sic) == expected


# --- universe ---------------------------------------------------------------


@pytest.mark.parametrize(
    "name,symbol,etf,test,expected",
    [
        ("Apple Inc. - Common Stock", "AAPL", "N", "N", True),
        ("Berkshire Hathaway Inc. New Common Stock", "BRK.B", "N", "N", True),
        ("SPDR S&P 500 ETF Trust", "SPY", "Y", "N", False),
        (
            "InterPrivate Investment Partners V, Inc. - Warrants",
            "IPVVW",
            "N",
            "N",
            False,
        ),
        ("Arbor Realty Trust 6.375% Series D Preferred", "ABR$D", "N", "N", False),
        ("Some Test Issue", "ZZZT", "N", "Y", False),
        ("Acme Corp Units", "ACMU", "N", "N", False),
    ],
)
def test_common_stock_filter(name, symbol, etf, test, expected):
    assert universe._is_common_stock(name, symbol, etf, test) is expected


def test_symbol_normalisation_matches_yfinance():
    assert universe._normalise_symbol("BRK.B") == "BRK-B"
    assert universe._normalise_symbol("aapl") == "AAPL"


# --- persistence ------------------------------------------------------------


def test_store_roundtrip_and_history():
    with tempfile.TemporaryDirectory() as directory:
        store = BuffettRankStore(os.path.join(directory, "ranks.db"))
        run_id = store.start_run(100, {"limit": None})

        frame = pd.DataFrame(
            {
                "cik": ["0000000001"],
                "name": ["Test Co"],
                "model": ["generic"],
                "rank": [1],
                "composite_score": [88.5],
                "quality_score": [90.0],
                "value_score": [float("nan")],
                "confidence": [1.0],
                "coverage": [1.0],
            },
            index=["TEST"],
        )
        assert store.save_scores(run_id, frame) == 1
        store.save_exclusions(
            run_id,
            [
                {
                    "symbol": "BAD",
                    "cik": "2",
                    "name": "Bad Co",
                    "model": "generic",
                    "reasons": "unprofitable",
                    "period_count": 12,
                    "coverage": 0.9,
                }
            ],
        )
        store.finish_run(run_id, 1, 1)

        ranked = store.get_ranked()
        assert len(ranked) == 1
        assert ranked[0]["symbol"] == "TEST"
        assert ranked[0]["value_score"] is None, "NaN must persist as NULL, not 'nan'"

        exclusions = store.get_exclusions()
        assert exclusions[0]["reasons"] == "unprofitable"

        history = store.get_symbol_history("TEST")
        assert len(history) == 1 and history[0]["composite_score"] == pytest.approx(
            88.5
        )


def test_no_fundamentals_does_not_add_redundant_history_failure():
    """A company with no filings is fully explained by that one reason."""
    company = _company(periods=0)
    company.gate_failures = ["no_fundamentals"]
    assert br.evaluate_gates(company) == []


# --- search -----------------------------------------------------------------


def _seeded_store(directory):
    """A store with a handful of ranked rows, for search tests."""
    store = BuffettRankStore(os.path.join(directory, "search.db"))
    run_id = store.start_run(10, {})
    frame = pd.DataFrame(
        {
            "cik": ["1", "2", "3", "4"],
            "name": [
                "Microsoft Corporation",
                "Advanced Micro Devices",
                "Realty Income Corporation",
                "100% Pure Co",  # a literal '%' in the name
            ],
            "model": ["generic", "generic", "reit", "generic"],
            "rank": [1, 2, 3, 4],
            "composite_score": [90.0, 80.0, 70.0, 60.0],
        },
        index=["MSFT", "AMD", "O", "PCT"],
    )
    store.save_scores(run_id, frame)
    store.finish_run(run_id, 4, 0)
    return store


def test_search_matches_symbol_and_name_across_the_whole_run():
    """
    The point of server-side search: a company ranked past the first page must
    still be findable. Matching on name is what makes 'micro' find both
    Microsoft and Advanced Micro Devices.
    """
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        symbols = {row["symbol"] for row in store.get_ranked(search="micro")}
        assert symbols == {"MSFT", "AMD"}
        assert {r["symbol"] for r in store.get_ranked(search="MSFT")} == {"MSFT"}


def test_search_is_case_insensitive():
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        assert {r["symbol"] for r in store.get_ranked(search="rEaLtY")} == {"O"}


def test_search_treats_sql_wildcards_as_literal_text():
    """
    An unescaped '%' would match every row, and '_' would match any character —
    so an ordinary keystroke would silently return the whole table.
    """
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        assert store.count_ranked(search="%") == 1  # only the "100% Pure Co" row
        assert store.get_ranked(search="%")[0]["symbol"] == "PCT"
        assert store.count_ranked(search="_") == 0


def test_search_combines_with_the_model_filter():
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        assert store.count_ranked(model="reit", search="corporation") == 1
        assert store.count_ranked(model="generic", search="corporation") == 1


def test_search_preserves_true_rank():
    """A searched company shows where it actually placed, not its position
    within the filtered results."""
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        assert store.get_ranked(search="realty")[0]["rank"] == 3


def test_count_ranked_reflects_filters():
    with tempfile.TemporaryDirectory() as directory:
        store = _seeded_store(directory)
        assert store.count_ranked() == 4
        assert store.count_ranked(search="nothing here") == 0


# --- solvency gate ----------------------------------------------------------


def test_buyback_heavy_compounder_is_not_excluded_for_low_book_equity():
    """
    Apple's real numbers: debt/equity 3.9 because buybacks shrank book equity,
    but $99bn of owner earnings against $55bn of net debt. Gating on D/E
    excluded 253 such companies — the ratio measures accounting history, not
    the ability to pay.
    """
    company = _company(
        debt_to_equity=3.87,
        net_debt_to_owner_earnings=0.55,
        roe_median=118.9,
    )
    assert br.evaluate_gates(company) == []


def test_genuinely_overleveraged_company_is_excluded():
    """Debt that would take a decade of free cash flow to repay is disqualifying."""
    company = _company(debt_to_equity=3.87, net_debt_to_owner_earnings=11.0)
    assert "debt_not_serviceable" in br.evaluate_gates(company)


def test_solvency_gate_ignores_unknown_serviceability():
    """Unknown must not fail a company, per P7."""
    company = _company(debt_to_equity=9.0, net_debt_to_owner_earnings=None)
    assert br.evaluate_gates(company) == []


# --- display names ----------------------------------------------------------


@pytest.mark.parametrize(
    "stored,expected",
    [
        ("Analog Devices, Inc. - Common Stock", "Analog Devices, Inc."),
        ("MGIC Investment Corporation Common Stock", "MGIC Investment Corporation"),
        (
            "Lincoln Electric Holdings, Inc. - Common Shares",
            "Lincoln Electric Holdings, Inc.",
        ),
        (
            "Federal Realty Investment Trust Common Shares of Beneficial Interest",
            "Federal Realty Investment Trust",
        ),
        ("Common Stock (DE)", "Common Stock"),
        ("Chemed Corp", "Chemed Corp"),
        (None, None),
    ],
)
def test_display_name_strips_registration_boilerplate(stored, expected):
    """
    Nearly every stored name ends in the same share-class phrase, which
    identifies nothing and crowds out the part that does.
    """
    from server.routes.buffett_rank import _display_name

    assert _display_name(stored) == expected


@pytest.mark.parametrize(
    "stored,expected",
    [
        ("Meta Platforms, Inc. - Class A Common Stock", "Meta Platforms, Inc. Class A"),
        ("News Corporation - Class B Common Stock", "News Corporation Class B"),
        ("Alphabet Inc. Class C Capital Stock", "Alphabet Inc. Class C"),
    ],
)
def test_display_name_keeps_the_share_class(stored, expected):
    """The class is the only thing separating one listing from its twin."""
    from server.routes.buffett_rank import _display_name

    assert _display_name(stored) == expected
