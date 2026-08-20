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

import logging
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import numpy as np

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
    if not population:
        return float("nan")
    return 100.0 * sum(1 for v in population if v <= value) / len(population)


def _describe(current: float, history: List[float]) -> str:
    """
    Where today sits, said in words rather than as a bare percentile.

    Phrased as a comparison and never as advice: this is what the company's own
    record says about the price, not a claim that the price is wrong.
    """
    pct = _percentile_of(current, history)
    if pct >= 90:
        return "dearer than almost all of its own history"
    if pct >= 65:
        return "dearer than usual for this company"
    if pct <= 10:
        return "cheaper than almost all of its own history"
    if pct <= 35:
        return "cheaper than usual for this company"
    return "around its own long-run average"


def _price_on_or_before(prices, when: date) -> Optional[float]:
    try:
        window = prices.loc[: str(when)]
        if window.empty:
            return None
        value = float(window.iloc[-1])
        return value if value > 0 else None
    except Exception:
        return None


def _load_prices(symbol: str, start: date, end: date):
    """
    Split-adjusted closes from the local store, or None.

    `Close` rather than `Adj Close`: the dividend adjustment in the latter would
    quietly depress every historical price and make the past look cheaper than
    it was.
    """
    try:
        from market_db import MarketDatabase

        frame = MarketDatabase().get_ohlcv(symbol.upper(), start, end)
        if frame is None or frame.empty or "Close" not in frame.columns:
            return None
        series = frame["Close"].dropna()
        return series if not series.empty else None
    except Exception as exc:
        logging.debug(f"Valuation bands: no price history for {symbol}: {exc}")
        return None


def _observations(
    cik: str, prices, shares: Dict[str, float], today: date
) -> Dict[str, List[float]]:
    """One multiple per fiscal year, each priced as it could have been seen."""
    import edgar_provider

    earliest = today.replace(year=today.year - BAND_YEARS)
    history: Dict[str, List[float]] = {"earnings": [], "free_cash_flow": []}

    for period_end in sorted(shares):
        try:
            ended = date.fromisoformat(str(period_end)[:10])
        except ValueError:
            continue
        if ended < earliest:
            continue
        known_on = ended + timedelta(days=FILING_LAG_DAYS)
        if known_on >= today:
            continue

        price = _price_on_or_before(prices, known_on)
        share_count = shares.get(period_end)
        if not price or not share_count:
            continue
        market_cap = price * share_count

        with edgar_provider.as_of(known_on.isoformat()):
            values = edgar_provider.get_concept_values(
                cik, ["net_income", "operating_cash_flow", "capex"]
            )
        net_income = (values.get("net_income") or {}).get(period_end)
        operating = (values.get("operating_cash_flow") or {}).get(period_end)
        capex = (values.get("capex") or {}).get(period_end)

        if net_income and net_income > 0:
            history["earnings"].append(market_cap / net_income)
        if operating is not None:
            free_cash_flow = operating - (capex or 0.0)
            if free_cash_flow > 0:
                history["free_cash_flow"].append(market_cap / free_cash_flow)

    return history


def bands(symbol: str, cik: str, today: Optional[date] = None) -> List[Dict[str, Any]]:
    """
    Today's multiples against the company's own fifteen-year record.

    Returns [] rather than raising for anything unmeasurable — a recent listing,
    a company with no price history in the store, a loss-maker with no positive
    denominator to divide by.
    """
    today = today or date.today()
    try:
        import edgar_provider

        shares = edgar_provider.split_consistent_series(cik, "shares_diluted")
        if len(shares) < MIN_OBSERVATIONS:
            for fallback_concept in ("shares_outstanding", "shares_basic"):
                fallback_shares = edgar_provider.split_consistent_series(
                    cik, fallback_concept
                )
                if len(fallback_shares) > len(shares):
                    shares = fallback_shares
        if not shares:
            return []

        prices = _load_prices(
            symbol, today.replace(year=today.year - BAND_YEARS - 1), today
        )
        if prices is None:
            return []

        history = _observations(cik, prices, shares, today)

        latest_price = float(prices.iloc[-1])
        latest_period = max(shares)
        latest_shares = shares[latest_period]
        current_cap = latest_price * latest_shares

        values = edgar_provider.get_concept_values(
            cik, ["net_income", "operating_cash_flow", "capex"]
        )
        latest_income = (values.get("net_income") or {}).get(latest_period)
        latest_ocf = (values.get("operating_cash_flow") or {}).get(latest_period)
        latest_capex = (values.get("capex") or {}).get(latest_period)
        latest_fcf = None if latest_ocf is None else latest_ocf - (latest_capex or 0.0)
        current = {
            "earnings": (current_cap / latest_income)
            if latest_income and latest_income > 0
            else None,
            "free_cash_flow": (current_cap / latest_fcf)
            if latest_fcf and latest_fcf > 0
            else None,
        }
    except Exception as exc:
        logging.debug(f"Valuation bands unavailable for {symbol}: {exc}")
        return []

    results: List[Dict[str, Any]] = []
    for key, label, _concept in _METRICS:
        population = [v for v in history.get(key, []) if np.isfinite(v) and v > 0]
        now = current.get(key)
        if len(population) < MIN_OBSERVATIONS or not now or not np.isfinite(now):
            continue
        results.append(
            {
                "metric": key,
                "label": label,
                "current": float(now),
                "median": float(np.median(population)),
                "p25": float(np.percentile(population, 25)),
                "p75": float(np.percentile(population, 75)),
                "low": float(np.min(population)),
                "high": float(np.max(population)),
                "percentile": float(_percentile_of(now, population)),
                "observations": len(population),
                "display": f"{now:.1f}x",
                "median_display": f"{float(np.median(population)):.1f}x",
                "summary": _describe(now, population),
            }
        )
    return results
