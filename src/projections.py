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

from finutils import infer_periods_per_year

# Every year out to 20 so the chart can draw a smooth cone; clients pick which
# horizons to tabulate (typically the 1/3/5/10/20y milestones).
HORIZONS_YEARS: List[int] = list(range(1, 21))

_TRADING_DAYS = 252

# Drift-shrinkage strength: the historical drift gets weight
# N_years / (N_years + _DRIFT_SHRINK_K), so a portfolio with ~10y of history is
# shrunk halfway toward the prior. Backtesting (walk-forward, S&P 500 + a stock
# basket) showed this removes the long-horizon over-extrapolation of a single
# realized path without materially hurting shorter horizons.
_DRIFT_SHRINK_K = 10.0

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
    benchmark_log_return: Optional[float] = None,
) -> dict:
    """Project ``current_value`` forward over ``horizons`` (years).

    Args:
        twr_series: Daily time-weighted-return wealth index (e.g. the
            "Portfolio Accumulated Gain" column). ``pct_change`` yields the
            daily TWR returns used to estimate drift/volatility.
        current_value: Current total portfolio value (the projection's V0).
        horizons: Horizons in years; defaults to ``HORIZONS_YEARS``.
        benchmark_log_return: Optional annual log-return of a broad benchmark
            (e.g. the S&P 500) over a comparable window. When supplied, the
            portfolio's noisy historical drift is shrunk toward it so a single
            lucky/unlucky run isn't extrapolated forever. Omit to skip shrinkage.

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

    # Observations per year, inferred from the index so annualization and the
    # history length are correct whether the series is calendar-daily (~365/yr,
    # as the portfolio TWR is) or trading-daily (~252/yr, as raw market data is).
    periods_per_year = infer_periods_per_year(returns.index, default=_TRADING_DAYS)
    n_years = len(returns) / periods_per_year

    mu_raw = float(log_ret.mean()) * periods_per_year                 # annual log drift (historical)
    sigma_log = float(log_ret.std(ddof=1)) * math.sqrt(periods_per_year)  # annual log vol

    if not math.isfinite(mu_raw) or not math.isfinite(sigma_log):
        return {"available": False}

    # --- Drift shrinkage toward the benchmark ---
    # A single realized path is a noisy drift estimate; pull it toward the broad
    # market, trusting the data more the longer the history.
    mu_log = mu_raw
    if benchmark_log_return is not None and math.isfinite(benchmark_log_return):
        w_data = n_years / (n_years + _DRIFT_SHRINK_K)
        mu_log = w_data * mu_raw + (1.0 - w_data) * benchmark_log_return

    points = []
    for t in horizons:
        drift = mu_log * t
        # --- Predictive spread = process risk + parameter (drift) uncertainty ---
        # The drift is estimated from n_years of data (SE ~ sigma/sqrt(n_years)),
        # so the cumulative drift over t years carries variance t^2*sigma^2/n_years
        # on top of the process variance t*sigma^2. Total: sigma^2 * t*(1 + t/n_years).
        # Without this, the cone is badly overconfident at long horizons (backtested).
        spread = sigma_log * math.sqrt(t * (1.0 + t / n_years)) if n_years > 0 else sigma_log * math.sqrt(t)
        median = current_value * math.exp(drift)
        point = {
            "years": t,
            "median_value": median,
            "median_return_pct": (math.exp(drift) - 1.0) * 100.0,
            # Mean of the lognormal sits above the median (uses process variance
            # only, so the central "expected" value stays stable/interpretable).
            "expected_value": current_value * math.exp(drift + 0.5 * sigma_log * sigma_log * t),
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
