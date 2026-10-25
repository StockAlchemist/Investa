# Frozen reference: _add_derived_rows from src/edgar_provider.py at commit
# e19ec07a, before the bulk rewrite. tests/test_rust_ratios_parity.py holds the
# rewrite to it, frame for frame. Never edit this to follow the rewrite.
# ruff: noqa
from typing import Dict, Optional

import pandas as pd


def _add_derived_rows(
    income: pd.DataFrame,
    balance: pd.DataFrame,
    values: Dict[str, Dict[str, float]],
    cashflow: Optional[pd.DataFrame] = None,
) -> None:
    """
    Fill in line items EDGAR does not tag directly but the ratio engine expects.

    Kept separate from `_frame_from_concepts` because these are *derivations*,
    not reported facts, and the distinction matters when auditing a score.
    """
    # Total Debt is rarely tagged as such; it is short-term plus long-term.
    short_term = values.get("short_term_debt", {})
    long_term = values.get("long_term_debt", {})
    debt_periods = set(short_term) | set(long_term)
    if debt_periods and not balance.empty:
        for period_end in debt_periods:
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in balance.columns:
                continue
            total = short_term.get(period_end, 0.0) + long_term.get(period_end, 0.0)
            balance.loc["Total Debt", stamp] = total

    # EBIT: pretax income plus interest expense is closer to the definition than
    # operating income, which excludes non-operating items inconsistently.
    pretax = values.get("pretax_income", {})
    interest = values.get("interest_expense", {})
    if pretax and not income.empty:
        for period_end, pretax_value in pretax.items():
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in income.columns:
                continue
            income.loc["Ebit", stamp] = pretax_value + interest.get(period_end, 0.0)

    # Free cash flow is not a filed concept — no filer tags it — so without this
    # the row would exist only for the handful of periods Yahoo covers while
    # operating cash flow beside it ran back to 2009.
    operating = values.get("operating_cash_flow", {})
    capex = values.get("capex", {})
    if cashflow is not None and not cashflow.empty:
        for period_end, ocf in operating.items():
            if period_end not in capex:
                continue
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in cashflow.columns:
                continue
            # `capex` is filed as a payment; the frame emits it negated.
            cashflow.loc["Free Cash Flow", stamp] = ocf - capex[period_end]

    # Some filers omit Liabilities entirely; assets minus equity recovers it.
    assets = values.get("total_assets", {})
    equity = values.get("equity", {})
    if (
        not balance.empty
        and "Total Liabilities Net Minority Interest" not in balance.index
    ):
        for period_end, asset_value in assets.items():
            if period_end not in equity:
                continue
            stamp = pd.to_datetime(period_end, errors="coerce")
            if pd.isna(stamp) or stamp not in balance.columns:
                continue
            balance.loc["Total Liabilities Net Minority Interest", stamp] = (
                asset_value - equity[period_end]
            )


