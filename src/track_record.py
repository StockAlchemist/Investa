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

import logging
from typing import Any, Dict, List, Optional

import buffett_metrics
import buffett_rank

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
    if value is None:
        return None
    try:
        if unit == "percent":
            return f"{value:.1f}%"
        if unit == "cagr":
            return f"{value * 100:+.1f}%/yr"
        if unit == "points":
            return f"{value:.1f} pts"
        if unit == "times":
            return f"{value:.1f}×"
        if unit == "share":
            # A share of years is only meaningful next to how many years there
            # were, so it is rendered as the count it came from when possible.
            if denominator:
                return f"{round(value * denominator):.0f} of {denominator:.0f} years"
            return f"{value * 100:.0f}% of years"
        if unit == "years":
            if denominator:
                return f"{value:.0f} of {denominator:.0f} years"
            return f"{value:.0f} years"
        return f"{value:.2f}"
    except (TypeError, ValueError):
        return None


def labelled_metrics(
    metrics: Dict[str, Optional[float]], model: str
) -> List[Dict[str, Any]]:
    """
    The pillars of `model`, each with its metrics labelled and formatted.

    A metric the filings do not support is kept with a null value rather than
    dropped: "we could not measure this" and "we did not look" are different
    statements, and only the first is honest about a data gap.
    """
    spec = (
        buffett_rank.PILLARS_BY_MODEL.get(model)
        or buffett_rank.PILLARS_BY_MODEL["generic"]
    )
    groups: List[Dict[str, Any]] = []

    for pillar, entries in spec.items():
        items: List[Dict[str, Any]] = []
        for key, higher_is_better in entries:
            label, unit = _METRIC_LABELS.get(
                key, (key.replace("_", " ").capitalize(), "ratio")
            )
            value = metrics.get(key)
            denominator = metrics.get(_DENOMINATORS.get(key, ""))
            items.append(
                {
                    "key": key,
                    "label": label,
                    "unit": unit,
                    "value": value,
                    "display": _format(value, unit, denominator),
                    "note": None,
                    "higher_is_better": higher_is_better,
                }
            )
        groups.append(
            {
                "key": pillar,
                "title": PILLAR_TITLES.get(
                    pillar, pillar.replace("_", " ").capitalize()
                ),
                "items": items,
            }
        )
    return groups


def _money(value: float) -> str:
    """A filing-scale figure, short enough to sit twice on one line."""
    magnitude = abs(value)
    for scale, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m"), (1e3, "k")):
        if magnitude >= scale:
            return f"{'-' if value < 0 else ''}${magnitude / scale:,.2f}{suffix}"
    return f"{'-' if value < 0 else ''}${magnitude:,.0f}"


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
    try:
        import edgar_provider

        found = edgar_provider.revisions(cik)
        labels = edgar_provider.concept_labels()
    except Exception as exc:
        logging.debug(f"Track record: revision history unavailable for {cik}: {exc}")
        return {"count": 0, "items": []}

    items = []
    for row in found[:limit]:
        items.append(
            {
                "concept": row["concept"],
                "label": labels.get(
                    row["concept"], row["concept"].replace("_", " ").capitalize()
                ),
                "period_end": row["period_end"],
                "original": row["original"],
                "current": row["current"],
                "change_pct": row["change_pct"],
                "display": f"{_money(row['original'])} → {_money(row['current'])}",
                "change_display": f"{row['change_pct']:+.1f}%",
                "first_filed": row["first_filed"],
                "restated_filed": row["restated_filed"],
            }
        )
    return {"count": len(found), "items": items}


def _ranked_row(symbol: str) -> Optional[Dict[str, Any]]:
    """The company's row in the most recent run, or None if it was not ranked."""
    try:
        from buffett_store import get_store

        history = get_store().get_symbol_history(symbol, limit=1)
    except Exception as exc:
        logging.debug(f"Track record: no ranking history for {symbol}: {exc}")
        return None
    return history[0] if history else None


def build(symbol: str, cik: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    Assemble the track record for one filer.

    Never raises: a company whose filings cannot be measured comes back with the
    reasons in `gate_failures` and empty metrics, which is the same thing the
    exclusions list would say about it.
    """
    import buffett_pipeline

    symbol = symbol.upper()
    try:
        model = buffett_pipeline.infer_model_from_facts(cik)
    except Exception as exc:
        logging.debug(f"Track record: model inference failed for {symbol}: {exc}")
        model = "generic"

    company = buffett_metrics.compute_metrics(cik, symbol, name or symbol, model)

    # Gates are the one place this system states an absolute threshold, so they
    # are the one judgement worth passing through — as the exclusion reasons
    # they already are, not as a score.
    try:
        gate_failures = list(company.gate_failures) + buffett_rank.evaluate_gates(
            company
        )
    except Exception as exc:
        logging.debug(f"Track record: gate evaluation failed for {symbol}: {exc}")
        gate_failures = list(company.gate_failures)

    ranked = _ranked_row(symbol)
    rank: Optional[Dict[str, Any]] = None
    if ranked:
        rank = {
            "run_id": ranked.get("run_id"),
            "rank": ranked.get("rank"),
            "composite_score": ranked.get("composite_score"),
            "quality_score": ranked.get("quality_score"),
            "value_score": ranked.get("value_score"),
            "confidence": ranked.get("confidence"),
            "pillars": {key: ranked.get(key) for key in _PILLAR_SCORE_KEYS},
        }

    return {
        "symbol": symbol,
        "name": name or (ranked.get("name") if ranked else None),
        "cik": cik,
        "model": model,
        "period_count": company.period_count,
        "first_period": company.first_period,
        "latest_period": company.latest_period,
        # The metrics span the durability window, not every filed year: ten
        # years is the span over which an advantage either holds or does not.
        "window_years": min(company.period_count, buffett_metrics.DURABILITY_WINDOW),
        "coverage": company.coverage,
        "gate_failures": gate_failures,
        "rank": rank,
        "groups": labelled_metrics(company.metrics, model),
        "revisions": revisions(cik),
    }
