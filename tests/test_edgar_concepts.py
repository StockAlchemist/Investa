"""Tests for the concept-to-tag chains (`edgar_concepts`).

A chain decides what a number *means*, so the failure mode is silent: add a tag
that resolves and the series gets longer and wronger at the same time. These
pin the two rules that keep the capex chain honest, both of which were arrived
at by measuring 120 ranked filers rather than by reading the taxonomy.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from edgar_concepts import CASHFLOW_CONCEPTS, all_concepts, all_tags  # noqa: E402


class TestCapexChain:
    # Tags that cover the gap years and are not capital expenditure. Each one
    # appeared in the measured sample as a tempting gap-filler.
    NOT_CAPEX = [
        "PaymentsToAcquireBusinessesNetOfCashAcquired",
        "PaymentsToAcquireBusinessesGross",
        "PaymentsToAcquireAvailableForSaleSecurities",
        "PaymentsToAcquireAvailableForSaleSecuritiesDebt",
        "PaymentsToAcquireHeldToMaturitySecurities",
        "PaymentsToAcquireEquityMethodInvestments",
        "PaymentsToAcquireMarketableSecurities",
        "PaymentsToAcquireNotesReceivable",
        "PaymentsToAcquireOtherInvestments",
    ]

    @pytest.mark.parametrize("tag", NOT_CAPEX)
    def test_investment_purchases_are_not_capex(self, tag):
        """
        Subtracting an acquisition spree from operating cash flow does not
        produce free cash flow. These cover the missing years and mean something
        else entirely, which is the trap.
        """
        assert tag not in CASHFLOW_CONCEPTS["capex"]

    @pytest.mark.parametrize(
        "tag", ["PaymentsToAcquireRealEstate", "PaymentsToDevelopRealEstateAssets"]
    )
    def test_real_estate_programmes_are_not_capex(self, tag):
        """
        For a REIT, buying buildings is the growth programme, not maintenance.
        Counting it would drive every REIT's free cash flow deeply negative and
        cost them a valuation they can legitimately have — which is why REITs are
        scored on FFO instead.
        """
        assert tag not in CASHFLOW_CONCEPTS["capex"]

    def test_the_original_tags_still_lead(self):
        """
        `resolve_concept` takes the first tag with data for each period, so the
        established tags must stay at the front: appended entries may fill gaps
        and may not change a series that already resolved.
        """
        assert CASHFLOW_CONCEPTS["capex"][:3] == [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
            "PaymentsForCapitalImprovements",
        ]

    def test_every_chained_tag_is_ingested(self):
        """
        `all_tags` is the ingest filter. A tag in a chain but not in that set is
        a series that silently never resolves.
        """
        chained = {tag for chain in all_concepts().values() for tag in chain}
        assert chained <= set(all_tags())

    def test_no_tag_is_repeated_within_a_chain(self):
        for concept, chain in all_concepts().items():
            assert len(chain) == len(set(chain)), f"{concept} repeats a tag"
