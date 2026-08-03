# -*- coding: utf-8 -*-
"""
Quality pillars, gates and cross-sectional ranking.

This turns per-company metrics (`buffett_metrics`) into a comparable score. The
shape of the calculation follows directly from the design principles:

**Gates run before scoring (P1).** A business that fails a solvency or
persistent-loss test is removed from the ranked list and reported separately
with its reason, rather than being scored badly and buried. Buffett's ordering
is quality first, price second; a pure margin-of-safety sort — which is what the
existing screener does — puts melting ice cubes on top, because a business the
market expects to shrink looks cheap against any model that assumes it will not.

**A gate only fires on a known violation (P7).** Missing data never fails a
gate. Apple, for instance, stopped tagging interest expense separately, so its
interest coverage is genuinely unknown — not zero, and not infinite. Unknowns
flow into the coverage-driven confidence factor, which can only demote.

**Ranking is cross-sectional and within-model (P6).** Raw ratios have
incomparable units and long tails, so every metric becomes a winsorised
percentile. Percentiles are computed *within* a valuation model: a bank's
equity-to-assets ratio is meaningful against other banks and meaningless against
software companies. Comparing a REIT's leverage to Microsoft's would rank the
entire sector last for doing something normal.

**A pillar earns its place in the portfolio, not in a correlation table.** This
is worth stating because the obvious statistic points the wrong way for one of
them — see `financial_strength` below, which has a negative information
coefficient and is nonetheless the difference between a 17.4% strategy and a
15.3% one. A signal is judged by the book it produces, not by its average rank
correlation, and the two can disagree.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from buffett_metrics import CompanyMetrics

# Quality pillar weights. Returns on capital carries the most weight because a
# durable high return on capital *is* the moat; growth carries least because
# growth without returns destroys value.
PILLAR_WEIGHTS: Dict[str, float] = {
    "returns_on_capital": 0.30,
    "financial_strength": 0.20,
    "predictability": 0.20,
    "growth": 0.15,
    "capital_allocation": 0.15,
}

# Decision from plan review: quality 60%, value 40%.
QUALITY_WEIGHT = 0.60
VALUE_WEIGHT = 0.40

# (metric, higher_is_better) per pillar, per model.
PillarSpec = Dict[str, List[Tuple[str, bool]]]

GENERIC_PILLARS: PillarSpec = {
    "returns_on_capital": [
        ("roe_median", True),
        ("roic_median", True),
        ("roa_median", True),
        ("gross_margin_median", True),
        ("roe_years_above_15", True),
    ],
    # DO NOT remove or shrink this pillar on the strength of its information
    # coefficient. That was proposed and measured on 2026-07-29, and the
    # measurement said the opposite of what the IC did.
    #
    # The IC is genuinely bad: -3.0 (t = -1.6) for generic companies, -9.2 for
    # banks and -4.2 for REITs, negative in both the 2013-19 and 2020-25
    # windows, with every constituent neutral or negative. Removing the pillar
    # and redistributing its weight nevertheless made the strategy far worse in
    # three of three shapes tested — top-20 CAGR 17.4% -> 15.3%, Sharpe
    # 1.05 -> 0.85, worst drawdown -21.5% -> -30.9%.
    #
    # The reason is that an IC is an average over the whole distribution while
    # the strategy is a tail operation: it buys the top twenty of roughly
    # twelve hundred. Comparing the two books directly, they agree on 167 of
    # 260 slots over thirteen years; on the 93 that differ, the names this
    # pillar admits returned 19.7%/yr at a median debt-to-equity of 0.31, and
    # the names admitted without it returned 13.5%/yr at a median
    # debt-to-equity of 1.77. What the pillar does is veto companies whose
    # returns on capital are high *because* they are levered — a trap the other
    # pillars cannot see, since leverage flatters every one of them.
    #
    # The weight is also already at its optimum: sweeping it over
    # 0.10-0.35 peaks at 0.20 on both CAGR and Sharpe, and dropping any single
    # constituent (including `current_ratio`, the weakest by IC) is worse.
    #
    # Note the window contains no credit event, so the apparent reward to
    # leverage here is measured entirely inside a falling-rate expansion. That
    # is a reason to keep the brake, not to reconsider it.
    "financial_strength": [
        ("debt_to_equity", False),
        ("interest_coverage", True),
        ("current_ratio", True),
        ("net_debt_to_owner_earnings", False),
    ],
    "predictability": [
        ("roe_stdev", False),
        ("revenue_growth_stdev", False),
        ("fcf_margin_stdev", False),
        ("negative_owner_earnings_years", False),
    ],
    "growth": [
        ("revenue_cagr", True),
        ("owner_earnings_cagr", True),
        ("book_value_per_share_cagr", True),
    ],
    "capital_allocation": [
        # Falling share count is a return of capital; rising is dilution.
        ("share_count_cagr", False),
        ("incremental_roic", True),
    ],
}

BANK_PILLARS: PillarSpec = {
    "returns_on_capital": [
        ("roe_median", True),
        ("roa_median", True),
        ("roe_years_above_12", True),
        ("net_interest_margin_median", True),
    ],
    "financial_strength": [
        # Equity/assets, not debt/equity: for a deposit-taker, liabilities are
        # the raw material of the business rather than a financing choice.
        ("equity_to_assets_latest", True),
        ("provision_rate_median", False),
    ],
    "predictability": [
        ("roe_stdev", False),
        ("provision_rate_stdev", False),
        ("net_interest_margin_stdev", False),
        ("loss_years", False),
    ],
    "growth": [
        ("revenue_cagr", True),
        ("book_value_per_share_cagr", True),
    ],
    "capital_allocation": [
        ("share_count_cagr", False),
        ("efficiency_ratio_median", False),
    ],
}

INSURER_PILLARS: PillarSpec = {
    "returns_on_capital": [
        ("roe_median", True),
        ("roa_median", True),
        ("roe_years_above_12", True),
    ],
    "financial_strength": [("equity_to_assets_latest", True)],
    "predictability": [("roe_stdev", False), ("loss_years", False)],
    "growth": [("revenue_cagr", True), ("book_value_per_share_cagr", True)],
    "capital_allocation": [("share_count_cagr", False)],
}

REIT_PILLARS: PillarSpec = {
    "returns_on_capital": [
        ("ffo_on_equity_median", True),
        ("ffo_margin_median", True),
    ],
    "financial_strength": [
        ("debt_to_ffo", False),
        ("debt_to_real_estate", False),
    ],
    "predictability": [
        ("ffo_margin_stdev", False),
        ("negative_ffo_years", False),
    ],
    "growth": [
        ("ffo_per_share_cagr", True),
        ("ffo_cagr", True),
    ],
    "capital_allocation": [
        # Weighted heavily in practice because REITs fund themselves by issuing
        # equity; growing FFO while growing share count faster is not growth.
        ("share_count_cagr", False),
    ],
}

PILLARS_BY_MODEL: Dict[str, PillarSpec] = {
    "generic": GENERIC_PILLARS,
    "bank": BANK_PILLARS,
    "insurer": INSURER_PILLARS,
    "reit": REIT_PILLARS,
}


# --- gates ------------------------------------------------------------------


def evaluate_gates(company: CompanyMetrics, min_periods: int = 8) -> List[str]:
    """
    Hard quality gates. Returns the list of failures — empty means eligible.

    Every test is written so that `None` cannot trigger it. That is the whole
    discipline: a company is excluded for something its filings *show*, never
    for something they omit.
    """
    failures: List[str] = []
    metrics = company.metrics

    # A company with no filings at all is already fully explained by the
    # failure `compute_metrics` recorded; adding "insufficient_history(0y)" on
    # top just makes the exclusion list harder to read.
    if any(
        f in ("no_fundamentals", "fundamentals_unavailable")
        for f in company.gate_failures
    ):
        return failures

    if company.period_count < min_periods:
        failures.append(f"insufficient_history({company.period_count}y)")

    def below(key: str, threshold: float) -> bool:
        value = metrics.get(key)
        return value is not None and value < threshold

    def above(key: str, threshold: float) -> bool:
        value = metrics.get(key)
        return value is not None and value > threshold

    if company.model in ("bank", "insurer"):
        # A bank funded with under 4% equity has no margin for credit error.
        if below("equity_to_assets_latest", 4.0):
            failures.append("undercapitalised")
        if above("loss_years", 2):
            failures.append("persistent_losses")
        if below("roe_median", 0.0):
            failures.append("unprofitable")

    elif company.model == "reit":
        if above("debt_to_ffo", 15.0):
            failures.append("excessive_leverage")
        if above("negative_ffo_years", 2):
            failures.append("persistent_negative_ffo")

    else:
        # Interest coverage is only a fair test when the company actually
        # carries debt and reports the expense; `debt_free` companies skip it.
        if not metrics.get("debt_free") and below("interest_coverage", 2.0):
            failures.append("interest_not_covered")

        # Solvency is tested against cash flow, not against book equity.
        #
        # Debt-to-equity was the obvious gate and it was wrong: sustained
        # buybacks shrink book equity, so the ratio explodes for precisely the
        # highest-quality compounders. At a 2.0 threshold it excluded 253
        # companies on that basis alone — Apple (D/E 3.9, yet $99bn of owner
        # earnings against $55bn of net debt), ADP, Amgen, American Express.
        # Equity is an accounting residual; the question that actually matters
        # is whether the business generates enough cash to retire what it owes.
        # D/E remains a *scoring* input in the financial-strength pillar, where
        # a bad reading costs percentile points rather than removing a company.
        if above("net_debt_to_owner_earnings", 6.0):
            failures.append("debt_not_serviceable")
        if above("negative_owner_earnings_years", 2):
            failures.append("persistent_cash_burn")
        if below("roe_median", 0.0):
            failures.append("unprofitable")

    return failures


# --- percentile scoring -----------------------------------------------------


def _winsorised_percentile(series: pd.Series, higher_is_better: bool) -> pd.Series:
    """
    Convert a raw metric to a 0–100 percentile, clipped at the 1st/99th centile.

    Winsorising first stops a single absurd outlier — a 10,000% ROE from a
    near-zero equity base — from compressing every other company into a narrow
    band at the bottom of the scale.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() < 2:
        return pd.Series(np.nan, index=series.index)

    lower, upper = numeric.quantile(0.01), numeric.quantile(0.99)
    clipped = numeric.clip(lower=lower, upper=upper)
    ranked = clipped.rank(pct=True, na_option="keep")
    if not higher_is_better:
        ranked = 1.0 - ranked
    return ranked * 100.0


def _pillar_scores(frame: pd.DataFrame, spec: PillarSpec) -> pd.DataFrame:
    """
    Score each pillar as the mean of its available metric percentiles.

    Averaging only the metrics that resolved — rather than treating a missing
    one as zero — keeps a data gap from masquerading as a bad result. The gap is
    accounted for once, in the confidence factor, instead of twice.
    """
    scores = pd.DataFrame(index=frame.index)

    for pillar, metrics in spec.items():
        columns = []
        for metric, higher_is_better in metrics:
            if metric not in frame.columns:
                continue
            percentile = _winsorised_percentile(frame[metric], higher_is_better)
            if percentile.notna().any():
                columns.append(percentile.rename(metric))
        if columns:
            scores[pillar] = pd.concat(columns, axis=1).mean(axis=1, skipna=True)
        else:
            scores[pillar] = np.nan

    return scores


def _quality_score(pillar_scores: pd.DataFrame) -> pd.Series:
    """
    Weighted mean of pillar scores, renormalised over the pillars present.

    Renormalising matters: a REIT has no gross margin and therefore a thinner
    returns pillar, and it should not be scored as though the missing weight
    were zeros.
    """
    total = pd.Series(0.0, index=pillar_scores.index)
    weight_sum = pd.Series(0.0, index=pillar_scores.index)

    for pillar, weight in PILLAR_WEIGHTS.items():
        if pillar not in pillar_scores.columns:
            continue
        # An all-missing pillar arrives as object dtype; coercing keeps the
        # arithmetic below numeric instead of silently downcasting.
        values = pd.to_numeric(pillar_scores[pillar], errors="coerce")
        present = values.notna()
        total = total.add(values.fillna(0.0) * weight * present, fill_value=0.0)
        weight_sum = weight_sum.add(weight * present, fill_value=0.0)

    return (total / weight_sum.replace(0.0, np.nan)).astype(float)


def confidence_factor(coverage: float, floor: float = 0.5) -> float:
    """
    Map data coverage to a multiplier in [floor, 1.0].

    Capped at 1.0 so it can only ever demote (P7) — full coverage earns no
    bonus, it is simply the absence of a penalty. The floor stops a thinly
    reported company from collapsing to zero and disappearing silently instead
    of ranking low and visibly.
    """
    if coverage is None:
        return floor
    return float(max(floor, min(1.0, floor + (1.0 - floor) * coverage)))


# --- assembly ---------------------------------------------------------------


def metrics_to_frame(companies: Sequence[CompanyMetrics]) -> pd.DataFrame:
    """Flatten a list of CompanyMetrics into one row per company."""
    rows = []
    for company in companies:
        row = {
            "symbol": company.symbol,
            "cik": company.cik,
            "name": company.name,
            "model": company.model,
            "period_count": company.period_count,
            "latest_period": company.latest_period,
            "coverage": company.coverage,
            "gate_failures": ",".join(company.gate_failures),
            "eligible": not company.gate_failures,
        }
        row.update(company.metrics)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.set_index("symbol", drop=False)
    return frame


def score_quality(companies: Sequence[CompanyMetrics]) -> pd.DataFrame:
    """
    Rank a universe on quality alone.

    Percentiles are computed within each valuation model, then concatenated —
    so a company's quality score says "how good is this bank among banks",
    which is the only comparison that means anything.
    """
    frame = metrics_to_frame(companies)
    if frame.empty:
        return frame

    eligible = frame[frame["eligible"]].copy()
    if eligible.empty:
        logging.warning("Rank: no companies passed the quality gates")
        frame["quality_score"] = np.nan
        return frame

    scored_parts = []
    for model, group in eligible.groupby("model"):
        spec = PILLARS_BY_MODEL.get(model, GENERIC_PILLARS)
        if len(group) < 5:
            # Percentiles within a handful of companies are noise, not signal.
            logging.warning(
                f"Rank: only {len(group)} eligible '{model}' companies — "
                "percentile scores for this group are unreliable"
            )
        pillars = _pillar_scores(group, spec)
        pillars["quality_score"] = _quality_score(pillars)
        scored_parts.append(pillars)

    pillar_frame = pd.concat(scored_parts) if scored_parts else pd.DataFrame()
    result = frame.join(pillar_frame, how="left")
    return result


def combine(
    frame: pd.DataFrame,
    value_scores: Optional[pd.Series] = None,
    quality_weight: float = QUALITY_WEIGHT,
    value_weight: float = VALUE_WEIGHT,
) -> pd.DataFrame:
    """
    Blend quality and value into the final composite, then apply confidence.

    `score = (0.60 * quality + 0.40 * value) * confidence`

    When no value score is supplied the composite is quality alone, so the
    ranking degrades gracefully if market data is unavailable rather than
    failing outright.
    """
    result = frame.copy()

    if value_scores is not None:
        result["value_score"] = value_scores.reindex(result.index)
    elif "value_score" not in result.columns:
        result["value_score"] = np.nan

    quality = pd.to_numeric(result.get("quality_score"), errors="coerce")
    value = pd.to_numeric(result.get("value_score"), errors="coerce")

    has_value = value.notna()
    blended = quality.copy()
    blended[has_value] = (
        quality[has_value] * quality_weight + value[has_value] * value_weight
    )

    result["confidence"] = result["coverage"].apply(confidence_factor)
    result["composite_score"] = blended * result["confidence"]

    result = result.sort_values("composite_score", ascending=False, na_position="last")
    result["rank"] = np.where(
        result["composite_score"].notna(), range(1, len(result) + 1), np.nan
    )
    return result
