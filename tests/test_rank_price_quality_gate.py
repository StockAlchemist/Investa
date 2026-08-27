"""The ranking must not score a company on price history known to be wrong.

The value half of the Buffett score is built from price — E/P and FCF/P divide
by it — so a series carrying an unapplied reverse split is wrong by the ratio.
That is the difference between a stock looking desperately cheap and being
ordinary, and it is how bad data turns into a decision someone acts on.

Measured when this was written: 21 flagged symbols sat inside the 1,229-symbol
rankable universe, five of them at `high` severity, and one flagged name was in
the live top 20.
"""

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

import buffett_pipeline as pipeline  # noqa: E402


class FakeCompany:
    def __init__(self, symbol):
        self.symbol = symbol
        self.gate_failures = []


@pytest.fixture
def flags(monkeypatch):
    """Stand in for MarketDatabase.get_data_quality."""
    table = {}

    class FakeDB:
        def get_data_quality(self, symbols=None):
            return {s: table[s] for s in (symbols or table) if s in table}

    import market_db

    monkeypatch.setattr(market_db, "MarketDatabase", lambda *a, **k: FakeDB())
    return table


def test_a_high_severity_symbol_is_excluded(flags):
    flags["KGEI"] = {"severity": "high"}
    companies = [FakeCompany("KGEI"), FakeCompany("AAPL")]

    gated = pipeline._apply_price_quality_gate(companies)

    assert gated == 1
    assert companies[0].gate_failures == ["price_history_unreliable"]
    assert companies[1].gate_failures == []


def test_a_merely_suspicious_symbol_still_ranks(flags):
    """An unexplained jump is not proof. It is flagged for the reader, not
    removed from consideration — over-excluding on a thin stock's real move
    would quietly shrink the universe."""
    flags["OVV"] = {"severity": "medium"}
    companies = [FakeCompany("OVV")]

    assert pipeline._apply_price_quality_gate(companies) == 0
    assert companies[0].gate_failures == []


def test_the_reason_is_distinct_from_a_fundamentals_gate(flags):
    """`evaluate_gates` excludes a company for what its filings show. This is a
    fact about our data, not the company, and the exclusion list must say so."""
    flags["MNST"] = {"severity": "high"}
    company = FakeCompany("MNST")
    company.gate_failures.append("persistent_cash_burn")

    pipeline._apply_price_quality_gate([company])

    assert company.gate_failures == ["persistent_cash_burn", "price_history_unreliable"]


def test_running_twice_does_not_duplicate_the_reason(flags):
    flags["KGEI"] = {"severity": "high"}
    company = FakeCompany("KGEI")

    pipeline._apply_price_quality_gate([company])
    pipeline._apply_price_quality_gate([company])

    assert company.gate_failures == ["price_history_unreliable"]


def test_a_missing_scan_never_blocks_a_ranking_run(monkeypatch):
    """An archive nobody has scanned has no flags. That must rank everything,
    not nothing."""
    import market_db

    def boom(*a, **k):
        raise RuntimeError("no such table: data_quality")

    monkeypatch.setattr(market_db, "MarketDatabase", boom)
    companies = [FakeCompany("AAPL")]

    assert pipeline._apply_price_quality_gate(companies) == 0
    assert companies[0].gate_failures == []
