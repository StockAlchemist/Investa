"""Forward portfolio-value projections.

Models the portfolio value as a lognormal (geometric Brownian motion) process,
parameterized by the historical annualized drift and volatility estimated from
the daily time-weighted-return series. Returns the median projected value plus
percentile bands for a set of standard horizons, so the UI can show a fan/cone
of outcomes rather than a single misleading point estimate.

The same daily-return series feeds ``risk_metrics.calculate_all_risk_metrics``,
so the volatility here is consistent with the risk card.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import pandas as pd

# Default horizons (years) shown in the UI.
HORIZONS_YEARS: List[int] = [1, 3, 5, 10, 20]

_TRADING_DAYS = 252

# Standard-normal quantiles for the percentile bands we report.
_BANDS = {
    "p10": -1.2815515594,
    "p25": -0.6744897502,
    "p75": 0.6744897502,
    "p90": 1.2815515594,
}


def compute_projection(
    twr_series: pd.Series,
    current_value: Optional[float],
    horizons: Optional[List[int]] = None,
) -> dict:
    """Project ``current_value`` forward over ``horizons`` (years).

    Args:
        twr_series: Daily time-weighted-return wealth index (e.g. the
            "Portfolio Accumulated Gain" column). ``pct_change`` yields the
            daily TWR returns used to estimate drift/volatility.
        current_value: Current total portfolio value (the projection's V0).
        horizons: Horizons in years; defaults to ``HORIZONS_YEARS``.

    Returns:
        A dict with ``available`` plus, when available, the per-horizon median
        and percentile band values, the annualized return/volatility used, and
        the starting value. All monetary values are in the summary's currency.
    """
    horizons = horizons or HORIZONS_YEARS

    if (
        twr_series is None
        or current_value is None
        or current_value <= 0
        or len(twr_series) < 30
    ):
        return {"available": False}

    # Daily TWR returns (cash-flow neutral) — same series the risk card uses.
    returns = (
        pd.to_numeric(twr_series, errors="coerce")
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    # Clip daily artifacts (missing flows, splits, …) to keep the estimates sane,
    # matching the robustness clip in risk_metrics.
    returns = returns.clip(lower=-0.90, upper=1.0)
    if len(returns) < 20:
        return {"available": False}

    log_ret = np.log1p(returns)
    mu_log = float(log_ret.mean()) * _TRADING_DAYS          # annual log drift
    sigma_log = float(log_ret.std(ddof=1)) * math.sqrt(_TRADING_DAYS)  # annual log vol

    if not math.isfinite(mu_log) or not math.isfinite(sigma_log):
        return {"available": False}

    points = []
    for t in horizons:
        drift = mu_log * t
        spread = sigma_log * math.sqrt(t)
        median = current_value * math.exp(drift)
        point = {
            "years": t,
            "median_value": median,
            "median_return_pct": (math.exp(drift) - 1.0) * 100.0,
            # Mean of a lognormal sits above the median.
            "expected_value": current_value * math.exp(drift + 0.5 * spread * spread),
        }
        for name, z in _BANDS.items():
            point[name] = current_value * math.exp(drift + z * spread)
        points.append(point)

    return {
        "available": True,
        "current_value": current_value,
        # Geometric (median) annualized return and annualized volatility.
        "annual_return_pct": (math.exp(mu_log) - 1.0) * 100.0,
        "annual_volatility_pct": sigma_log * 100.0,
        "horizons": points,
    }
