# -*- coding: utf-8 -*-
"""
What a company has historically been worth, measured against itself.

Investa's value score is cross-sectional: a stock is cheap because its earnings
yield beats other stocks'. That is the right question for a ranking and the
wrong one for a reader looking at one company, because it cannot say whether a
23x multiple is this business being expensive or this business being normal.
Fifteen years of filings and prices can.

Three things make the arithmetic honest rather than merely available:

  * **Market cap, not per-share.** A price-to-earnings ratio built from a
    back-adjusted price and an as-filed EPS mixes two split bases and is wrong
    by whatever splits fell between. Market cap sidesteps it: the price series
    is on today's split basis and `split_consistent_series` puts every historical
    share count on that same basis, so their product is the market value that
    actually stood at the time.
  * **Point-in-time earnings.** Each historical multiple uses the figures filed
    by then, through `edgar_provider.as_of`. Using today's restated numbers
    would price a 2012 stock on facts published in 2019 — the same look-ahead
    the backtest exists to avoid, and this system restates often enough for it
    to matter.
  * **Prices are read, never fetched.** The store is consulted directly; a
    company with no price history gets no bands rather than a request that
    blocks on a network fill.

Deliberately not a signal. Nothing here feeds the ranking: "cheap against its
own history" is a statement about a company's past valuation regime, and a
business whose economics have permanently changed will look cheap all the way
down.
"""

from __future__ import annotations

from datetime import date, timedelta  # noqa: F401 -- read by the Rust port at call time
from typing import Any, Dict, List, Optional

import investa_core

# Years of history a band spans. The filed record reaches ~19, but the earliest
# years are thin enough across the universe that the extra observations mostly
# add noise.
BAND_YEARS = 15

# A fiscal year is not public knowledge on the day it ends. Filers have 60-90
# days, so the multiple the market could actually see is priced a quarter later.
FILING_LAG_DAYS = 90

# Fewer than this and a percentile is theatre.
MIN_OBSERVATIONS = 6

_METRICS = (
    ("earnings", "Price to earnings", "net_income"),
    ("free_cash_flow", "Price to free cash flow", None),
)


def _percentile_of(value: float, population: List[float]) -> float:
    """Share of the history at or below `value`, in percent."""
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_percentile_of(value, population)


def _describe(current: float, history: List[float]) -> str:
    """
    Where today sits, said in words rather than as a bare percentile.

    Phrased as a comparison and never as advice: this is what the company's own
    record says about the price, not a claim that the price is wrong.
    """
    return _describe_percentile(_percentile_of(current, history))


def _describe_percentile(pct: float) -> str:
    """`_describe` from an already computed percentile."""
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_describe_percentile(pct)


def _price_on_or_before(prices, when: date) -> Optional[float]:
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_price_on_or_before(prices, when)


def _load_prices(symbol: str, start: date, end: date):
    """
    Split-adjusted closes from the local store, or None.

    `Close` rather than `Adj Close`: the dividend adjustment in the latter would
    quietly depress every historical price and make the past look cheaper than
    it was.
    """
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_load_prices(symbol, start, end)


def _observations(
    cik: str, prices, shares: Dict[str, float], today: date
) -> Dict[str, List[float]]:
    """One multiple per fiscal year, each priced as it could have been seen."""
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_observations(cik, prices, shares, today)


def bands(symbol: str, cik: str, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Today's multiples against the company's own fifteen-year record.

    Returns [] rather than raising for anything unmeasurable — a recent listing,
    a company with no price history in the store, a loss-maker with no positive
    denominator to divide by.
    """
    # Ported to Rust: rust/investa_core/src/vh.rs
    return investa_core.vh_bands(symbol, cik, today)
