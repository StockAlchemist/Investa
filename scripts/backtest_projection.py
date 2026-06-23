#!/usr/bin/env python3
"""Walk-forward backtest of the portfolio-value projection model.

Back-checks ``projections.compute_projection`` against history: at each past
date it fits the model on data up to that point, projects forward, then compares
to what actually happened. It reports *calibration* — whether outcomes land
inside the predicted bands at the right frequency — which is what tells you if
the cone is honest (not just whether the median was "close").

Metrics per horizon (ideal value in parentheses):
  std_z   spread of standardized errors (1.0). >1 = bands too narrow / overconfident.
  in80    fraction inside the p10-p90 band (0.80).
  <p10    fraction below the p10 line (0.10). >0.10 = downside under-covered.
  mean_u  mean probability-integral-transform (0.50). <0.5 = drift over-extrapolated.

Usage:
  python scripts/backtest_projection.py                 # S&P 500, default
  python scripts/backtest_projection.py AAPL MSFT KO    # a basket, S&P 500 prior
  python scripts/backtest_projection.py --no-shrink     # compare without the benchmark prior

Requires network (yfinance). This is an analysis tool, not part of the app.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from projections import compute_projection  # noqa: E402

_TRADING_DAYS = 252
_P90_Z = 1.2815515594  # the z used for the p90 band, to back out the model's predictive sd


def _norm_cdf(z):
    return 0.5 * (1.0 + np.vectorize(math.erf)(np.asarray(z) / math.sqrt(2.0)))


def backtest(
    wealth: pd.Series,
    benchmark: pd.Series | None,
    horizons=(1, 3, 5, 10),
    step: int = 21,
    min_history_years: int = 5,
    use_shrink: bool = True,
) -> pd.DataFrame:
    """Return one row per (date, horizon) with the standardized error z."""
    wealth = wealth.dropna()
    rows = []
    start = _TRADING_DAYS * min_history_years
    for t in range(start, len(wealth), step):
        window = wealth.iloc[:t]
        v0 = float(window.iloc[-1])

        prior = None
        if use_shrink and benchmark is not None:
            bw = benchmark.reindex(window.index).ffill().dropna()
            if len(bw) > _TRADING_DAYS and float(bw.iloc[0]) > 0:
                prior = math.log(float(bw.iloc[-1]) / float(bw.iloc[0])) / (len(bw) / _TRADING_DAYS)

        proj = compute_projection(window, v0, benchmark_log_return=prior)
        if not proj.get("available"):
            continue
        by_year = {p["years"]: p for p in proj["horizons"]}
        for h in horizons:
            fut = t + h * _TRADING_DAYS
            if fut >= len(wealth) or h not in by_year:
                continue
            pt = by_year[h]
            mu_h = math.log(pt["median_value"] / v0)             # predictive mean log-return
            sd_h = (math.log(pt["p90"] / v0) - mu_h) / _P90_Z    # predictive sd the model used
            if sd_h <= 0:
                continue
            actual = math.log(float(wealth.iloc[fut]) / v0)
            rows.append((h, (actual - mu_h) / sd_h))
    return pd.DataFrame(rows, columns=["h", "z"])


def summarize(df: pd.DataFrame, horizons) -> None:
    print(f"\n{'h':>4} {'n':>5} {'std_z':>6} {'in80':>6} {'<p10':>6} {'mean_u':>7}   (ideal 1.0 / 0.80 / 0.10 / 0.50)")
    for h in horizons:
        g = df[df.h == h]
        if g.empty:
            continue
        u = _norm_cdf(g.z.values)
        print(f"{h:>3}y {len(g):>5} {g.z.std():>6.2f} {np.mean((u > .1) & (u < .9)):>6.2f} "
              f"{np.mean(u < .1):>6.2f} {u.mean():>7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tickers", nargs="*", default=["^GSPC"], help="ticker(s) to backtest (default ^GSPC)")
    ap.add_argument("--benchmark", default="^GSPC", help="drift-shrinkage prior ticker (default ^GSPC)")
    ap.add_argument("--start", default="1970-01-01")
    ap.add_argument("--no-shrink", action="store_true", help="disable benchmark drift shrinkage")
    args = ap.parse_args()
    tickers = args.tickers or ["^GSPC"]

    import yfinance as yf

    bench = yf.download(args.benchmark, start=args.start, auto_adjust=True, progress=False)["Close"].dropna().squeeze()
    px = yf.download(tickers, start=args.start, auto_adjust=True, progress=False)["Close"]
    horizons = (1, 3, 5, 10)

    frames = []
    for tk in tickers:
        # yf.download(list) returns a DataFrame keyed by ticker even for one ticker.
        s = (px[tk] if isinstance(px, pd.DataFrame) else px).dropna()
        # Don't shrink a ticker toward itself (degenerate).
        b = None if (tk == args.benchmark or args.no_shrink) else bench
        frames.append(backtest(s, b, horizons=horizons, use_shrink=not args.no_shrink))
    df = pd.concat(frames, ignore_index=True)

    label = "shrink OFF" if args.no_shrink else f"shrink toward {args.benchmark}"
    print(f"Backtest: {', '.join(tickers)}  ({label}, n={len(df)} samples)")
    summarize(df, horizons)


if __name__ == "__main__":
    main()
