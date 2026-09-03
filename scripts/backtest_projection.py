#!/usr/bin/env python3
"""Walk-forward backtest of the portfolio-value projection model.

Back-checks ``projections.compute_projection`` against history: at each past
date it fits the model on data up to that point, projects forward, then compares
to what actually happened. It reports *calibration* — whether outcomes land
inside the predicted bands at the right frequency — which is what tells you if
the cone is honest (not just whether the median was "close").

The walk-forward engine itself lives in ``src/projections.py``
(``walk_forward_errors`` / ``summarize_errors``), shared with the in-app
backtest at ``GET /api/projection/backtest``; this script runs it over tickers.

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
import os
import sys
import warnings

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from projections import summarize_errors, walk_forward_errors  # noqa: E402

HORIZONS = [1, 3, 5, 10]


def summarize(errors: pd.DataFrame, horizons) -> None:
    print(
        f"\n{'h':>4} {'n':>5} {'std_z':>6} {'in80':>6} {'<p10':>6} {'mean_u':>7}   (ideal 1.0 / 0.80 / 0.10 / 0.50)"
    )
    for row in summarize_errors(errors, horizons):
        print(
            f"{row['years']:>3}y {row['samples']:>5} {row['std_z']:>6.2f} "
            f"{row['in_band_pct'] / 100:>6.2f} {row['below_p10_pct'] / 100:>6.2f} "
            f"{row['mean_u']:>7.2f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "tickers",
        nargs="*",
        default=["^GSPC"],
        help="ticker(s) to backtest (default ^GSPC)",
    )
    ap.add_argument(
        "--benchmark",
        default="^GSPC",
        help="drift-shrinkage prior ticker (default ^GSPC)",
    )
    ap.add_argument("--start", default="1970-01-01")
    ap.add_argument(
        "--no-shrink", action="store_true", help="disable benchmark drift shrinkage"
    )
    args = ap.parse_args()
    tickers = args.tickers or ["^GSPC"]

    import yfinance as yf

    bench = (
        yf.download(args.benchmark, start=args.start, auto_adjust=True, progress=False)[
            "Close"
        ]
        .dropna()
        .squeeze()
    )
    px = yf.download(tickers, start=args.start, auto_adjust=True, progress=False)[
        "Close"
    ]

    frames = []
    for tk in tickers:
        # yf.download(list) returns a DataFrame keyed by ticker even for one ticker.
        s = (px[tk] if isinstance(px, pd.DataFrame) else px).dropna()
        # Don't shrink a ticker toward itself (degenerate).
        b = None if (tk == args.benchmark or args.no_shrink) else bench
        frames.append(walk_forward_errors(s, benchmark_series=b, horizons=HORIZONS))
    df = pd.concat(frames, ignore_index=True)

    label = "shrink OFF" if args.no_shrink else f"shrink toward {args.benchmark}"
    print(f"Backtest: {', '.join(tickers)}  ({label}, n={len(df)} samples)")
    summarize(df, HORIZONS)


if __name__ == "__main__":
    main()
