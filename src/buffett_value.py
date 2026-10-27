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

from typing import Any, Dict, Optional, Sequence

import investa_core
import pandas as pd

import edgar_provider  # noqa: F401 -- read by the Rust port at call time
from buffett_metrics import CompanyMetrics, _finite, _safe_divide  # noqa: F401 -- re-exported

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
    # Ported to Rust: rust/investa_core/src/bm.rs
    return investa_core.bv_compute_market_metrics(company, price, shares_outstanding)


def _pct_yield(numerator: Optional[float], market_cap: float) -> Optional[float]:
    # Ported to Rust: rust/investa_core/src/bm.rs
    return investa_core.bv_pct_yield(numerator, market_cap)


def score_value(frame: pd.DataFrame) -> pd.Series:
    """
    Blend the value metrics into a 0–100 score, ranked within each model.

    Uses the same winsorised-percentile machinery as the quality pillars so the
    two halves of the composite are on genuinely the same scale — otherwise the
    quality/value weighting would not mean what it says. The percentiles and
    the weight-renormalised blend run in Rust.
    """
    # Ported to Rust: rust/investa_core/src/bm.rs
    return investa_core.bv_score_value(frame)


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
    # Ported to Rust: rust/investa_core/src/bm.rs
    return investa_core.bv_build_value_frame(companies, market_data)
