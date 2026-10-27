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

import investa_core
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

    The returns are prepared here; the fit (annual log drift and volatility,
    drift shrinkage) and the cone — whose spread carries parameter
    uncertainty, sigma^2 * t * (1 + t / n_years) — run in Rust
    (investa_core.projection).
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

    # Observations per year, inferred from the index so annualization and the
    # history length are correct whether the series is calendar-daily (~365/yr,
    # as the portfolio TWR is) or trading-daily (~252/yr, as raw market data is).
    periods_per_year = infer_periods_per_year(returns.index, default=_TRADING_DAYS)
    prior = (
        float(benchmark_log_return)
        if benchmark_log_return is not None and math.isfinite(benchmark_log_return)
        else None
    )
    fitted = investa_core.projection(
        np.ascontiguousarray(returns.to_numpy(dtype=np.float64)),
        float(periods_per_year),
        float(current_value),
        [float(t) for t in horizons],
        prior,
    )
    if fitted is None:
        return {"available": False}
    annual_return_pct, annual_volatility_pct, rows = fitted

    points = []
    for t, (median, median_return_pct, expected, p10, p25, p75, p90) in zip(
        horizons, rows
    ):
        points.append(
            {
                "years": t,
                "median_value": median,
                "median_return_pct": median_return_pct,
                # Mean of the lognormal sits above the median (process variance
                # only, so the central "expected" value stays interpretable).
                "expected_value": expected,
                "p10": p10,
                "p25": p25,
                "p75": p75,
                "p90": p90,
            }
        )

    return {
        "available": True,
        "current_value": current_value,
        # Geometric (median) annualized return and annualized volatility.
        "annual_return_pct": annual_return_pct,
        "annual_volatility_pct": annual_volatility_pct,
        "horizons": points,
    }


# --------------------------------------------------------------------------- #
# Walk-forward backtest
# --------------------------------------------------------------------------- #
# "Would this model have told the truth?" — at each past date, fit the model on
# the data available *then*, project forward, and compare with what actually
# happened. What matters is calibration (do outcomes land inside the bands at
# the advertised frequency), not whether the median happened to be close.

# Horizons back-checked by default. Each needs `min_history_years` of history to
# fit on plus the horizon itself to verify against, so a short track record only
# supports the short ones.
BACKTEST_HORIZONS: List[int] = [1, 3, 5, 10]

# Data required before the first fit — below this the drift estimate is too
# noisy for the result to say anything about the model.
MIN_BACKTEST_HISTORY_YEARS = 5.0

_DAYS_PER_YEAR = 365.25

# std_z (spread of standardized errors) outside this range means the cone is
# too narrow (overconfident) or too wide (uninformative); 1.0 is perfect.
_CALIBRATED_BAND = (0.85, 1.15)


def _benchmark_prior(
    bench_window: Optional[pd.Series], periods_per_year: float
) -> Optional[float]:
    """Trailing annual log-return of the benchmark over the fitting window.

    This is the drift-shrinkage prior ``compute_projection`` takes, recomputed at
    each anchor from data that existed then (no look-ahead).
    """
    if bench_window is None:
        return None
    b = pd.to_numeric(bench_window, errors="coerce").dropna()
    if len(b) < periods_per_year or periods_per_year <= 0:
        return None
    first, last = float(b.iloc[0]), float(b.iloc[-1])
    if first <= 0 or last <= 0:
        return None
    return math.log(last / first) / (len(b) / periods_per_year)


def walk_forward_errors(
    twr_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    horizons: Optional[List[int]] = None,
    min_history_years: float = MIN_BACKTEST_HISTORY_YEARS,
    step: Optional[int] = None,
) -> pd.DataFrame:
    """One row per (anchor date, horizon) with the standardized error z.

    At each past anchor the model is fitted on the data that existed then, and
    the outcome ``horizon`` years later is scored against its cone. The loop
    runs in Rust (investa_core.walk_forward): each window's returns are a prefix
    of one array instead of being re-derived through pandas per anchor.

    Args:
        twr_series: Wealth index to back-check (the portfolio TWR series, or a
            price series when run over a ticker).
        benchmark_series: Optional benchmark wealth/price series aligned to the
            same dates; supplies the drift-shrinkage prior at each anchor.
        horizons: Horizons in years to verify.
        min_history_years: History required before the first fit.
        step: Anchor spacing in observations; defaults to roughly monthly.

    Returns:
        DataFrame with columns ``years, anchor, z, actual_log, median_log``.
    """
    cols = ["years", "anchor", "z", "actual_log", "median_log"]
    wealth = pd.to_numeric(twr_series, errors="coerce").dropna()
    wealth = wealth[wealth > 0]
    horizons = sorted(horizons or BACKTEST_HORIZONS)
    if len(wealth) < 30 or not horizons:
        return pd.DataFrame(columns=cols)

    is_dated = isinstance(wealth.index, pd.DatetimeIndex)
    # Nanosecond ticks, which the Rust walk-forward converts to years; pandas 3
    # indexes may count in s/ms/us.
    ticks = (
        np.ascontiguousarray(wealth.index.as_unit("ns").asi8, dtype=np.int64)
        if is_dated
        else None
    )
    bench = None
    if benchmark_series is not None:
        bench = np.ascontiguousarray(
            pd.to_numeric(benchmark_series, errors="coerce")
            .reindex(wealth.index)
            .ffill()
            .to_numpy(dtype=np.float64, na_value=np.nan)
        )
    rows = investa_core.walk_forward(
        np.ascontiguousarray(wealth.to_numpy(dtype=np.float64)),
        ticks,
        bench,
        [float(h) for h in horizons],
        float(min_history_years),
        step or None,
    )
    by_value = {float(h): h for h in horizons}
    return pd.DataFrame(
        [(by_value[h], wealth.index[a], z, actual, mu) for h, a, z, actual, mu in rows],
        columns=cols,
    )


def summarize_errors(
    errors: pd.DataFrame, horizons: Optional[List[int]] = None
) -> List[dict]:
    """Per-horizon calibration stats from ``walk_forward_errors`` output.

    ``std_z`` 1.0 / ``in_band_pct`` 80 / ``below_p10_pct`` 10 / ``mean_u`` 0.5 are
    the ideal values: bands the right width, outcomes inside them as often as
    advertised, and no systematic over- or under-shoot of the drift. The
    statistics run in Rust (investa_core.summarize_errors).
    """
    if errors is None or errors.empty:
        return []
    horizons = sorted(horizons or sorted(errors["years"].unique()))
    out = []
    for h in horizons:
        g = errors[errors["years"] == h]
        if len(g) < 3:
            continue
        (
            std_z,
            in_band_pct,
            below_p10_pct,
            above_p90_pct,
            mean_u,
            median_actual_return_pct,
            median_projected_return_pct,
        ) = investa_core.summarize_errors(
            g["z"].to_numpy(dtype=float).tolist(),
            g["actual_log"].to_numpy(dtype=float).tolist(),
            g["median_log"].to_numpy(dtype=float).tolist(),
        )
        verdict = "calibrated"
        if std_z > _CALIBRATED_BAND[1]:
            verdict = "narrow"
        elif std_z < _CALIBRATED_BAND[0]:
            verdict = "wide"
        out.append(
            {
                "years": int(h),
                "samples": int(len(g)),
                "std_z": std_z,
                "in_band_pct": in_band_pct,
                "below_p10_pct": below_p10_pct,
                "above_p90_pct": above_p90_pct,
                "mean_u": mean_u,
                # Median realized vs projected growth over the horizon, in the
                # units a reader actually thinks in.
                "median_actual_return_pct": median_actual_return_pct,
                "median_projected_return_pct": median_projected_return_pct,
                "verdict": verdict,
            }
        )
    return out


def _build_replay(
    wealth: pd.Series,
    bench: Optional[pd.Series],
    years: float,
    start_value: Optional[float] = None,
    min_history_years: float = MIN_BACKTEST_HISTORY_YEARS,
) -> Optional[dict]:
    """The single most legible backtest: the cone the model drew ``years`` ago,
    with the path the portfolio actually took drawn through it.

    The actual path is the TWR path (what the money already invested at the
    anchor did), because that is what the projection models — later deposits and
    withdrawals are deliberately excluded from both lines.
    """
    if not isinstance(wealth.index, pd.DatetimeIndex) or len(wealth) < 30:
        return None
    periods_per_year = infer_periods_per_year(wealth.index, default=_TRADING_DAYS)
    target = wealth.index[-1] - pd.Timedelta(days=round(_DAYS_PER_YEAR * years))
    anchor = int(wealth.index.searchsorted(target, side="right")) - 1
    if anchor < int(round(periods_per_year * min_history_years)):
        return None

    window = wealth.iloc[: anchor + 1]
    base = float(window.iloc[-1])
    if base <= 0:
        return None
    indexed = not (start_value and start_value > 0)
    v0 = 100.0 if indexed else float(start_value)

    prior = _benchmark_prior(
        None if bench is None else bench.iloc[: anchor + 1], periods_per_year
    )
    months = max(1, int(round(years * 12)))
    proj = compute_projection(
        window,
        v0,
        horizons=[k / 12.0 for k in range(1, months + 1)],
        benchmark_log_return=prior,
    )
    if not proj.get("available"):
        return None

    anchor_date = wealth.index[anchor]
    points = [
        {
            "date": anchor_date.strftime("%Y-%m-%d"),
            "years": 0.0,
            "actual": v0,
            "median": v0,
            "p10": v0,
            "p25": v0,
            "p75": v0,
            "p90": v0,
        }
    ]
    for point in proj["horizons"]:
        t = float(point["years"])
        when = anchor_date + pd.Timedelta(days=round(_DAYS_PER_YEAR * t))
        pos = int(wealth.index.searchsorted(when, side="right")) - 1
        actual = None
        if pos > anchor:
            actual = v0 * float(wealth.iloc[pos]) / base
        points.append(
            {
                "date": when.strftime("%Y-%m-%d"),
                "years": t,
                "actual": actual,
                "median": point["median_value"],
                "p10": point["p10"],
                "p25": point["p25"],
                "p75": point["p75"],
                "p90": point["p90"],
            }
        )

    last = points[-1]
    if last["actual"] is None:
        return None
    outcome = "inside"
    if last["actual"] < last["p10"]:
        outcome = "below"
    elif last["actual"] > last["p90"]:
        outcome = "above"

    return {
        "anchor_date": anchor_date.strftime("%Y-%m-%d"),
        "years": float(years),
        "start_value": v0,
        "indexed": indexed,
        "fit_years": (anchor + 1) / periods_per_year,
        "annual_return_pct": proj["annual_return_pct"],
        "annual_volatility_pct": proj["annual_volatility_pct"],
        "final_actual": last["actual"],
        "final_median": last["median"],
        "final_p10": last["p10"],
        "final_p90": last["p90"],
        "outcome": outcome,
        "points": points,
    }


def backtest_projection(
    twr_series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    value_series: Optional[pd.Series] = None,
    horizons: Optional[List[int]] = None,
    min_history_years: float = MIN_BACKTEST_HISTORY_YEARS,
) -> dict:
    """Walk-forward backtest of ``compute_projection`` on a portfolio's own history.

    Args:
        twr_series: Daily TWR wealth index — the same series the projection fits.
        benchmark_series: Benchmark wealth index (e.g. ``^GSPC Accumulated Gain``)
            for the drift-shrinkage prior, recomputed at each anchor.
        value_series: Portfolio market value, used to denominate the replay in
            real money (its value on the anchor date is the replay's V0). Omit to
            get an indexed (start = 100) replay.
        horizons: Horizons to verify; defaults to ``BACKTEST_HORIZONS``.
        min_history_years: History required before the first fit.

    Returns:
        ``{"available": False, "reason": ...}`` when the track record is too
        short, else the per-horizon calibration table plus a ``replay`` of the
        longest verifiable horizon.
    """
    horizons = sorted(horizons or BACKTEST_HORIZONS)
    wealth = (
        pd.to_numeric(twr_series, errors="coerce").dropna()
        if twr_series is not None
        else None
    )
    if wealth is not None:
        wealth = wealth[wealth > 0]
    if wealth is None or len(wealth) < 30:
        return {"available": False, "reason": "no_history"}

    periods_per_year = infer_periods_per_year(wealth.index, default=_TRADING_DAYS)
    history_years = len(wealth) / periods_per_year
    usable = [h for h in horizons if history_years >= min_history_years + h]
    if not usable:
        return {
            "available": False,
            "reason": "insufficient_history",
            "history_years": history_years,
            "required_years": min_history_years + horizons[0],
        }

    errors = walk_forward_errors(
        wealth,
        benchmark_series=benchmark_series,
        horizons=usable,
        min_history_years=min_history_years,
    )
    stats = summarize_errors(errors, usable)

    bench = None
    if benchmark_series is not None:
        bench = (
            pd.to_numeric(benchmark_series, errors="coerce")
            .reindex(wealth.index)
            .ffill()
        )

    values = (
        pd.to_numeric(value_series, errors="coerce").reindex(wealth.index).ffill()
        if value_series is not None
        else None
    )

    # Replay the longest horizon the history can actually verify.
    replay = None
    for years in sorted((s["years"] for s in stats), reverse=True):
        start_value = None
        if values is not None:
            when = wealth.index[-1] - pd.Timedelta(days=round(_DAYS_PER_YEAR * years))
            pos = int(wealth.index.searchsorted(when, side="right")) - 1
            if pos >= 0 and pd.notna(values.iloc[pos]) and float(values.iloc[pos]) > 0:
                start_value = float(values.iloc[pos])
        replay = _build_replay(
            wealth,
            bench,
            years,
            start_value=start_value,
            min_history_years=min_history_years,
        )
        if replay is not None:
            break

    return {
        "available": bool(stats),
        "reason": None if stats else "insufficient_history",
        "history_years": history_years,
        "history_start": wealth.index[0].strftime("%Y-%m-%d")
        if isinstance(wealth.index, pd.DatetimeIndex)
        else None,
        "history_end": wealth.index[-1].strftime("%Y-%m-%d")
        if isinstance(wealth.index, pd.DatetimeIndex)
        else None,
        "min_history_years": min_history_years,
        "horizons": stats,
        "replay": replay,
    }
