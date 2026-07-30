# -*- coding: utf-8 -*-
"""
The value half of the ranking: two reported cash-flow yields, and nothing else.

Earnings yield and free-cash-flow yield. Both are arithmetic on a filed figure
and a quote, so neither depends on a forecast. EV/EBIT, price-to-book and
price-to-sales are still computed here for context, but they are *not* scored —
see `VALUE_WEIGHTS` for the measurement that excluded them.

**There is deliberately no discounted-cash-flow term here.** There used to be:
the value score was 35% weighted to a margin of safety computed from the P25 of
a Monte Carlo DCF. It was measured against thirteen years of point-in-time
rankings (`scripts/rank_signal_lab.py`) and it does not work:

  * Its information coefficient against the next year's return is +1.7% with a
    t-statistic of 1.05 and a 54% hit rate — indistinguishable from noise — and
    the sign is unstable, near zero in 2013-19 and positive only in 2020-25.
  * The distribution is not credible. The median eligible company scored a
    margin of safety of -33% and the 10th percentile around -500%, so the model
    claimed the typical business was worth a third of its price.
  * As a standalone top-20 selector it compounded at 9.7%/yr against 12.25% for
    the eligible universe average — worse than buying everything.
  * It correlated 0.58 with earnings yield, so most of what it did contribute
    was a noisier restatement of E/P, which is right here already.

Removing it took the default strategy from 16.3% to 17.4% CAGR, lifted the
Sharpe from 0.95 to 1.05 and cut the worst drawdown from -24.3% to -21.5%,
measured point-in-time over 2013-2025.

A third-party DCF is not the fix: the same audit found FMP's published values
put Amazon at $72 against a $231 price and Alphabet at $128 against $334, and
analyst consensus targets correlate 0.972 with the current price. The failure is
in the model class, not in whose arithmetic runs it. `financial_ratios` still
holds the DCF and Graham models, which remain in use for the per-stock screener
and stock-detail views where a user is reading one company's assumptions
deliberately; what changed is that a cross-sectional *ranking* no longer rests
on them.

Both scored metrics are ranked as sector-relative percentiles (P6). An earnings
yield that is ordinary for a bank is exceptional for a software company, and
comparing the two directly would just re-rank the market by industry.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd

import edgar_provider
from buffett_metrics import CompanyMetrics, _finite, _safe_divide

# What the value score is made of, and in which direction. This mapping is the
# single definition of "scored": `score_value` reads it directly, so a metric
# cannot be weighted here and ranked differently there.
#
# Only the two cash-flow yields survive, because only they measure up. Once the
# DCF came out, the remaining multiples became measurable for the first time
# (they had never been persisted to the backtest cache) and three of the five
# turned out to be dead weight over 2013-2025:
#
#     signal            IC      t      2013-19   2020-25
#     earnings_yield   +3.49   0.87     +3.27     +3.75     <- the only strong one
#     fcf_yield        +1.30   0.31     +3.23     -0.95
#     price_to_sales   -0.12  -0.03     -1.31     +1.27
#     ev_to_ebit       -0.60  -0.15     +0.09     -1.40
#     price_to_book    -1.38  -0.26     -0.62     -2.27     <- negative both windows
#
# Price-to-book fails the same test the DCF failed, and for a related reason:
# book equity is an accounting residual that buybacks and unrecorded intangibles
# distort most for exactly the businesses this ranking is trying to find. Cutting
# all three improved Sharpe, drawdown and held-out return in three of three
# strategy shapes tested (`scripts/rank_signal_lab.py`).
#
# The 60/40 split favours earnings yield because it is the one component
# positive in both windows, and it sits inside a broad 0.50-0.70 plateau rather
# than on the grid's argmax (0.65) — the difference between a stable region and
# a fitted point.
VALUE_WEIGHTS: Dict[str, float] = {
    "earnings_yield": 0.60,
    "fcf_yield": 0.40,
}

# Direction of each metric: True where a higher raw value is better. Kept
# separate from the weights only because a yield and a multiple point opposite
# ways; anything absent from VALUE_WEIGHTS is computed for context and is
# deliberately not scored.
HIGHER_IS_BETTER: Dict[str, bool] = {
    "earnings_yield": True,
    "fcf_yield": True,
    "ev_to_ebit": False,
    "price_to_book": False,
    "price_to_sales": False,
}


def compute_market_metrics(
    company: CompanyMetrics,
    price: Optional[float],
    shares_outstanding: Optional[float],
) -> Dict[str, Optional[float]]:
    """
    Price-dependent value metrics for one company.

    Kept separate from `buffett_metrics` because everything here changes daily
    while the fundamentals change quarterly — and because a ranking that mixed
    the two would have to recompute the fundamentals on every price tick.
    """
    result: Dict[str, Optional[float]] = {
        "price": price,
        "market_cap": None,
        "earnings_yield": None,
        "fcf_yield": None,
        "ev_to_ebit": None,
        "price_to_book": None,
        "price_to_sales": None,
    }

    price = _finite(price)
    shares = _finite(shares_outstanding)
    if price is None or price <= 0 or shares is None or shares <= 0:
        return result

    market_cap = price * shares
    result["market_cap"] = market_cap

    # --- classic multiples --------------------------------------------------
    concepts = edgar_provider.get_concept_values(
        company.cik,
        ["revenue", "equity", "pretax_income", "interest_expense", "net_income", "cash"],
    )
    latest = company.latest_period
    if not latest:
        return result

    owner_earnings = company.get("owner_earnings_latest") or company.get("ffo_latest")
    net_income = _finite(concepts.get("net_income", {}).get(latest))
    equity = _finite(concepts.get("equity", {}).get(latest))
    revenue = _finite(concepts.get("revenue", {}).get(latest))
    cash = _finite(concepts.get("cash", {}).get(latest)) or 0.0
    total_debt = company.get("total_debt")

    result["earnings_yield"] = _pct_yield(net_income, market_cap)
    result["fcf_yield"] = _pct_yield(owner_earnings, market_cap)
    result["price_to_book"] = _safe_divide(market_cap, equity)
    result["price_to_sales"] = _safe_divide(market_cap, revenue)

    if total_debt is not None:
        enterprise_value = market_cap + total_debt - cash
        pretax = _finite(concepts.get("pretax_income", {}).get(latest))
        interest = _finite(concepts.get("interest_expense", {}).get(latest)) or 0.0
        if pretax is not None:
            ebit = pretax + interest
            if ebit > 0:
                result["ev_to_ebit"] = _safe_divide(enterprise_value, ebit)

    return result


def _pct_yield(numerator: Optional[float], market_cap: float) -> Optional[float]:
    value = _safe_divide(numerator, market_cap)
    return value * 100.0 if value is not None else None


def score_value(frame: pd.DataFrame) -> pd.Series:
    """
    Blend the value metrics into a 0–100 score, ranked within each model.

    Uses the same winsorised-percentile machinery as the quality pillars so the
    two halves of the composite are on genuinely the same scale — otherwise the
    quality/value weighting would not mean what it says.
    """
    from buffett_rank import _winsorised_percentile

    if frame.empty:
        return pd.Series(dtype=float)

    scores = pd.Series(np.nan, index=frame.index, dtype=float)

    for model, group in frame.groupby("model"):
        total = pd.Series(0.0, index=group.index)
        weight_sum = pd.Series(0.0, index=group.index)

        # Iterating the weights, not every computed metric, is what keeps the
        # unscored diagnostics out of the score.
        for metric in VALUE_WEIGHTS:
            higher_is_better = HIGHER_IS_BETTER[metric]
            if metric not in group.columns:
                continue
            values = pd.to_numeric(group[metric], errors="coerce")
            # A negative multiple is meaningless, not cheap: a loss-making
            # company has no P/E and a negative book value has no P/B.
            if not higher_is_better:
                values = values.where(values > 0)
            percentile = _winsorised_percentile(values, higher_is_better)
            weight = VALUE_WEIGHTS.get(metric, 0.0)
            present = percentile.notna()
            total = total.add(percentile.fillna(0.0) * weight * present, fill_value=0.0)
            weight_sum = weight_sum.add(weight * present, fill_value=0.0)

        scores.loc[group.index] = total / weight_sum.replace(0.0, np.nan)

    return scores


def build_value_frame(
    companies: Sequence[CompanyMetrics],
    market_data: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Assemble price-dependent metrics for a universe.

    `market_data` maps symbol → {"price": float, "shares": float}. It is passed
    in rather than fetched here so this stays testable without a network and so
    the caller controls the (slow) market-data batching.
    """
    rows = []

    for company in companies:
        quote = market_data.get(company.symbol) or {}
        try:
            metrics = compute_market_metrics(
                company,
                quote.get("price"),
                quote.get("shares"),
            )
        except Exception as exc:  # pragma: no cover - defensive
            logging.warning(f"Value: failed for {company.symbol}: {exc}")
            metrics = {}

        row = {"symbol": company.symbol, "model": company.model}
        row.update(metrics)
        rows.append(row)

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.set_index("symbol", drop=False)
    return frame
