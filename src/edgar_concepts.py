# -*- coding: utf-8 -*-
"""
XBRL concept fallback chains.

This is the crux of the EDGAR data layer. Companies do not report a given line
item under a stable us-gaap tag: the tag changes with accounting standards, with
the filer's own presentation choices, and with industry. Apple reports revenue
under `Revenues` in FY2018 and `RevenueFromContractWithCustomerExcludingAssessedTax`
from FY2019 onward, because ASC 606 took effect. A naive single-tag lookup
silently loses a decade of history and — worse — loses it for *some* companies
and not others, which biases any cross-sectional ranking built on top.

So every logical concept resolves through an ordered chain of candidate tags.
The resolver walks the chain per fiscal period and takes the first tag that has
a value, recording which tag answered so coverage stays auditable (principle P7:
a missing input must never masquerade as a good score).

Chain ordering rule: most specific and most modern first, broadest last. A tag
earlier in the chain must never be a *superset* of a later one, or periods will
silently mix incompatible definitions — e.g. `Revenues` (which for a bank
includes interest income) sits after the narrower revenue tags.

Sector-specific concepts live in their own chains. Banks and REITs are ranked on
different models (residual income and FFO respectively), so they need inputs the
generic model never asks for. Their tag coverage is measurably worse than
generic concepts — JPMorgan reports `ProvisionForLoanAndLeaseLosses` for only a
few of its filed years — which is a fact about EDGAR, not a bug here, and is
surfaced by the coverage report rather than hidden.
"""
from __future__ import annotations

from typing import Dict, List

# --- Income statement -------------------------------------------------------

INCOME_CONCEPTS: Dict[str, List[str]] = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "SalesRevenueServicesNet",
        # Broadest last: for banks `Revenues` folds in interest income.
        "Revenues",
        "RevenuesNetOfInterestExpense",
    ],
    "cost_of_revenue": [
        "CostOfGoodsAndServicesSold",
        "CostOfRevenue",
        "CostOfGoodsSold",
        "CostOfServices",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "pretax_income": [
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesDomestic",
    ],
    "tax_provision": ["IncomeTaxExpenseBenefit"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestExpenseNonoperating",
        "InterestAndDebtExpense",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "shares_diluted": [
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfDilutedSharesOutstandingBasic",
    ],
    "shares_basic": [
        "WeightedAverageNumberOfSharesOutstandingBasic",
        "WeightedAverageNumberOfSharesOutstanding",
    ],
}

# --- Balance sheet ----------------------------------------------------------

BALANCE_CONCEPTS: Dict[str, List[str]] = {
    "total_assets": ["Assets"],
    "total_liabilities": ["Liabilities"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "equity_including_minority": [
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "StockholdersEquity",
    ],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "inventory": ["InventoryNet", "InventoryFinishedGoods"],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "CashAndDueFromBanks",
    ],
    "short_term_investments": [
        "ShortTermInvestments",
        "AvailableForSaleSecuritiesDebtSecuritiesCurrent",
    ],
    "short_term_debt": [
        "ShortTermBorrowings",
        "DebtCurrent",
        "LongTermDebtCurrent",
        "OtherShortTermBorrowings",
    ],
    # `NotesPayable` and the secured/unsecured pair sit late but matter a lot:
    # REITs run unclassified balance sheets and mostly do not use the
    # `LongTermDebtNoncurrent` tag at all. Realty Income reports $25bn under
    # `NotesPayable` and nothing under the first two entries, so without these
    # it would score as debt-free — the most dangerous possible error, since
    # zero leverage is the best score the strength pillar can award.
    "long_term_debt": [
        "LongTermDebtNoncurrent",
        "LongTermDebt",
        "LongTermDebtAndCapitalLeaseObligations",
        "NotesPayable",
        "SecuredDebt",
        "UnsecuredDebt",
        "SeniorNotes",
    ],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "CommonStockSharesIssued",
    ],
}

# --- Cash flow --------------------------------------------------------------

CASHFLOW_CONCEPTS: Dict[str, List[str]] = {
    "operating_cash_flow": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    # Appended entries fill gaps only — `resolve_concept` takes the first tag
    # with data for each period, so order is what protects the existing series.
    #
    # Measured over 120 ranked filers: 33 have years where operating cash flow is
    # reported and capex is not. Plenty of tags *cover* those years, and almost
    # all of them are the wrong thing — purchases of securities, held-to-maturity
    # debt, equity-method stakes, whole businesses. Subtracting an acquisition
    # spree from operating cash flow does not produce free cash flow, so only
    # tags that mean "spent on productive assets" appear here.
    #
    # Real-estate acquisition and development are excluded on the same principle
    # from the other direction: for a REIT, buying buildings is the growth
    # programme rather than maintenance, and treating it as capex would drive
    # every REIT's free cash flow deeply negative and cost them a valuation they
    # can legitimately have. REITs are scored on FFO for exactly this reason.
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
        "PaymentsForCapitalImprovements",
        "PaymentsForProceedsFromProductiveAssets",
        "PaymentsToAcquireOtherPropertyPlantAndEquipment",
        "PaymentsToAcquireMachineryAndEquipment",
        "PaymentsToAcquireBuildings",
        "PaymentsToAcquireOilAndGasProperty",
        "PaymentsToExploreAndDevelopOilAndGasProperties",
    ],
    "depreciation_amortization": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "dividends_paid": [
        "PaymentsOfDividendsCommonStock",
        "PaymentsOfDividends",
        "PaymentsOfDividendsMinorityInterest",
    ],
    "share_repurchase": [
        "PaymentsForRepurchaseOfCommonStock",
        "PaymentsForRepurchaseOfEquity",
    ],
}

# --- Sector: banks and insurers ---------------------------------------------
# Ranked on residual income rather than FCF, so the inputs differ from the
# generic model. Coverage here is materially thinner — see module docstring.

BANK_CONCEPTS: Dict[str, List[str]] = {
    "net_interest_income": [
        "InterestIncomeExpenseNet",
        "InterestIncomeExpenseAfterProvisionForLoanLoss",
    ],
    "interest_income": [
        "InterestAndDividendIncomeOperating",
        "InterestIncomeOperating",
    ],
    "noninterest_income": ["NoninterestIncome"],
    "noninterest_expense": ["NoninterestExpense"],
    # `ProvisionForDoubtfulAccounts` is deliberately absent: it is the generic
    # trade-receivables allowance, not a loan-loss provision. Measured across
    # the universe it answered 68% of this chain's periods — i.e. it was mostly
    # firing for non-lenders and would have made the credit-discipline metric
    # measure the wrong thing entirely.
    "provision_for_credit_losses": [
        "ProvisionForLoanLeaseAndOtherLosses",
        "ProvisionForLoanAndLeaseLosses",
        "ProvisionForCreditLossesExpensed",
    ],
    "deposits": ["Deposits", "InterestBearingDepositLiabilities"],
    # `NotesReceivableNet` is likewise excluded — it is any company's notes
    # receivable, not a bank loan book.
    "loans": [
        "FinancingReceivableExcludingAccruedInterestAfterAllowanceForCreditLoss",
        "LoansAndLeasesReceivableNetReportedAmount",
    ],
}

# --- Sector: REITs ----------------------------------------------------------
# FFO = net income + real-estate depreciation − gains on property sales.
# GAAP earnings understate REIT economics because depreciation is charged on
# assets that generally appreciate, so net income alone is the wrong input.

REIT_CONCEPTS: Dict[str, List[str]] = {
    "real_estate_net": [
        "RealEstateInvestmentPropertyNet",
        "RealEstateInvestmentPropertyAtCost",
    ],
    "real_estate_depreciation": [
        "RealEstateInvestmentPropertyAccumulatedDepreciation",
    ],
    "gain_on_sale_real_estate": [
        "GainLossOnSaleOfPropertiesNetOfApplicableIncomeTaxes",
        "GainsLossesOnSalesOfInvestmentRealEstate",
        "GainLossOnSaleOfProperties",
    ],
    "rental_revenue": [
        "OperatingLeasesIncomeStatementLeaseRevenue",
        "OperatingLeaseLeaseIncome",
    ],
}


def all_concepts() -> Dict[str, List[str]]:
    """Every concept chain, flattened. Used by the ingest and coverage report."""
    merged: Dict[str, List[str]] = {}
    for group in (
        INCOME_CONCEPTS,
        BALANCE_CONCEPTS,
        CASHFLOW_CONCEPTS,
        BANK_CONCEPTS,
        REIT_CONCEPTS,
    ):
        merged.update(group)
    return merged


def all_tags() -> List[str]:
    """
    Every us-gaap tag referenced by any chain.

    The bulk ingest keeps only these, which is what makes a 1.39 GB archive
    collapse into a database of workable size.
    """
    tags = set()
    for chain in all_concepts().values():
        tags.update(chain)
    return sorted(tags)


# Concepts that are point-in-time balances rather than flows over a period.
# They arrive from XBRL without a start date and must not be treated as annual
# durations when filtering facts.
INSTANT_CONCEPTS = frozenset(BALANCE_CONCEPTS) | {
    "deposits",
    "loans",
    "real_estate_net",
    "real_estate_depreciation",
}
