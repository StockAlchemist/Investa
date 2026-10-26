# -*- coding: utf-8 -*-
"""
One company's measured track record, assembled for a reader rather than a ranker.

The ranking has always computed durability evidence for every US filer — a
decade of ROE readings, how many years the business lost money, whether the
share count fell — and then thrown all of it away except five pillar scores.
The stock window meanwhile showed a language model's *guess* at moat and
predictability. This module closes that gap: the same numbers the ranking scores
on, presented as the record they are.

Two rules shape it:

  * **The metric set is not defined here.** It is read from
    `buffett_rank.PILLARS_BY_MODEL`, so the stock window shows exactly what the
    ranking scores, per model, and the two cannot drift. Adding a metric to a
    pillar makes it appear here automatically; this module only supplies a label
    and a unit.
  * **Evidence, not a verdict.** Nothing here colours a number good or bad. The
    thresholds that do exist in this system are the hard gates, and those are
    reported verbatim as the exclusion reasons they are. A median ROE of 13% is
    excellent for a bank and mediocre for a software company, and a per-metric
    traffic light would have to pretend otherwise.

`labelled_metrics` is deliberately separable from the fetch so it can be tested
without a fact store.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import buffett_metrics  # noqa: F401 -- read by the Rust port at call time
import buffett_rank  # noqa: F401 -- read by the Rust port at call time
import investa_core

# Human titles for the five quality pillars, in the order they are scored.
PILLAR_TITLES: Dict[str, str] = {
    "returns_on_capital": "Returns on capital",
    "financial_strength": "Financial strength",
    "predictability": "Predictability",
    "growth": "Growth",
    "capital_allocation": "Capital allocation",
}

# label, unit. Units decide formatting only:
#   percent  - already in percentage points (ROE of 13.4 means 13.4%)
#   cagr     - a fraction per year (0.18 means 18%/yr)
#   ratio    - a plain multiple, shown to two places
#   times    - a coverage multiple, shown with a multiplication sign
#   points   - a standard deviation of a percentage series
#   years    - a count of years, paired with its denominator below
#   share    - a fraction of years, paired with a count below
_METRIC_LABELS: Dict[str, tuple] = {
    # Returns on capital
    "roe_median": ("Median return on equity", "percent"),
    "roic_median": ("Median return on invested capital", "percent"),
    "roa_median": ("Median return on assets", "percent"),
    "gross_margin_median": ("Median gross margin", "percent"),
    "roe_years_above_15": ("Years with ROE above 15%", "share"),
    "roe_years_above_12": ("Years with ROE above 12%", "share"),
    "net_interest_margin_median": ("Median net interest margin", "percent"),
    "ffo_on_equity_median": ("Median FFO on equity", "percent"),
    "ffo_margin_median": ("Median FFO margin", "percent"),
    # Financial strength
    "debt_to_equity": ("Debt to equity", "ratio"),
    "interest_coverage": ("Interest coverage", "times"),
    "current_ratio": ("Current ratio", "ratio"),
    "net_debt_to_owner_earnings": ("Net debt to owner earnings", "times"),
    "equity_to_assets_latest": ("Equity to assets", "percent"),
    "provision_rate_median": ("Median credit-loss provision rate", "percent"),
    "debt_to_ffo": ("Debt to FFO", "times"),
    "debt_to_real_estate": ("Debt to real estate", "ratio"),
    # Predictability
    "roe_stdev": ("ROE variability", "points"),
    "revenue_growth_stdev": ("Revenue-growth variability", "points"),
    "fcf_margin_stdev": ("Cash-margin variability", "points"),
    "negative_owner_earnings_years": ("Years burning cash", "years"),
    "loss_years": ("Loss-making years", "years"),
    "provision_rate_stdev": ("Provisioning variability", "points"),
    "net_interest_margin_stdev": ("Net-interest-margin variability", "points"),
    "ffo_margin_stdev": ("FFO-margin variability", "points"),
    "negative_ffo_years": ("Years with negative FFO", "years"),
    # Growth
    "revenue_cagr": ("Revenue growth", "cagr"),
    "owner_earnings_cagr": ("Owner-earnings growth", "cagr"),
    "book_value_per_share_cagr": ("Book value per share growth", "cagr"),
    "ffo_cagr": ("FFO growth", "cagr"),
    "ffo_per_share_cagr": ("FFO per share growth", "cagr"),
    # Capital allocation
    "share_count_cagr": ("Share count change", "cagr"),
    "incremental_roic": ("Return on incremental capital", "percent"),
    "efficiency_ratio_median": ("Median efficiency ratio", "percent"),
}

# Where a count metric can find its denominator, so "2" can be shown as
# "2 of 10 years" — without the span, a count says nothing.
_DENOMINATORS: Dict[str, str] = {
    "roe_years_above_15": "roe_observation_years",
    "roe_years_above_12": "roe_observation_years",
    "negative_owner_earnings_years": "owner_earnings_years",
    "loss_years": "net_income_years",
    "negative_ffo_years": "ffo_years",
}

# Pillar scores carried on a ranked row, in scoring order.
_PILLAR_SCORE_KEYS = list(PILLAR_TITLES)

# How many revisions to send. GE has 103; the largest handful is the story, and
# the count carries the rest.
REVISION_LIMIT = 8

# Per-share rates used to be suppressed here, because the assembled share count
# steps at a stock split and every rate spanning it was the split rather than
# the company. That is now repaired at the source — `buffett_metrics` rebuilds
# the series from same-filing ratios — so these are shown like any other metric
# and the ranking and this view agree on them again.


def _format(
    value: Optional[float], unit: str, denominator: Optional[float]
) -> Optional[str]:
    """The reader-facing form of one metric. None when there is nothing to show."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_format(value, unit, denominator)


def labelled_metrics(
    metrics: Dict[str, Optional[float]], model: str
) -> List[Dict[str, Any]]:
    """
    The pillars of `model`, each with its metrics labelled and formatted.

    A metric the filings do not support is kept with a null value rather than
    dropped: "we could not measure this" and "we did not look" are different
    statements, and only the first is honest about a data gap.
    """
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_labelled_metrics(metrics, model)


# The two downturns the filed history reaches. A five-year window contains
# neither, which is the whole reason a company's behaviour in one was not
# something Investa could show before.
#
# Peak years bracket the entry into each downturn and trough years the exit,
# read off the fiscal year end — a January year-end lands a quarter early, which
# is close enough for "what happened to this business in the crisis" and far
# more robust than trying to align fiscal calendars.
STRESS_PERIODS = (
    {
        "key": "gfc",
        "label": "2008-09 recession",
        "peak_years": (2006, 2007, 2008),
        "trough_years": (2008, 2009, 2010),
    },
    {
        "key": "covid",
        "label": "2020 shutdown",
        "peak_years": (2018, 2019),
        "trough_years": (2020, 2021),
    },
)

# What a downturn is read on. Demand, profit and cash can fall by very different
# amounts in the same year, and which one gave way is the information.
_STRESS_METRICS = (
    ("revenue", "Revenue"),
    ("net_income", "Net income"),
    ("free_cash_flow", "Free cash flow"),
)


def _by_fiscal_year(series: Dict[str, float]) -> Dict[int, float]:
    """{calendar year of the period end: value}, latest wins within a year."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_by_fiscal_year(series)


def stress_response(
    values: Dict[str, Dict[str, float]], periods=STRESS_PERIODS
) -> List[Dict[str, Any]]:
    """
    How far each metric fell peak-to-trough in each downturn, and when it recovered.

    Evidence about one company, never a score. Two things make it unfit for
    cross-sectional use and both are stated rather than smoothed over: only the
    companies that were filing then have a 2008 reading at all, and the ones that
    did not survive are not in the fact store to be compared against. Ranking on
    this would reward having existed.

    A company with no data covering a window comes back `covered: false` — "not
    listed then" and "did not fall" are opposite claims.
    """
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_stress_response(values, periods)


def _money(value: float) -> str:
    """A filing-scale figure, short enough to sit twice on one line."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_money(value)


def revisions(cik: str, limit: int = REVISION_LIMIT) -> Dict[str, Any]:
    """
    Numbers this company changed after first reporting them.

    Nothing else in the retail world shows this, and the only reason Investa can
    is that the fact store keys every (cik, tag, period_end, accession) and never
    overwrites — a vendor feed carries the current view and has thrown the rest
    away.

    Presented as history, not as an accusation. Most revisions are the
    retrospective adoption of an accounting standard (Microsoft's FY2017 tax
    provision moved 127% on ASC 606) or a discontinued operation reclassifying
    years of revenue at once. The magnitude and the dates are the information;
    what they mean is the reader's call.
    """
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_revisions(cik, limit)


def _ranked_row(symbol: str) -> Optional[Dict[str, Any]]:
    """The company's row in the most recent run, or None if it was not ranked."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_ranked_row(symbol)


def build(symbol: str, cik: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Assemble the track record for one filer.

    Never raises: a company whose filings cannot be measured comes back with the
    reasons in `gate_failures` and empty metrics, which is the same thing the
    exclusions list would say about it.
    """
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_build(symbol, cik, name)


def _bands(symbol: str, cik: str) -> List[Dict[str, Any]]:
    """Today's multiples against the company's own fifteen-year record."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_bands(symbol, cik)


def _stress(cik: str) -> List[Dict[str, Any]]:
    """The company's behaviour in the two downturns the filed history reaches."""
    # Ported to Rust: rust/investa_core/src/tr.rs
    return investa_core.tr_stress(cik)
