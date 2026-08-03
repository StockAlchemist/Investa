"""Regenerate the figures and data table for `docs/quality_strategy_report.tex`.

Kept in the repository rather than run ad hoc, because the report's numbers have
to be reproducible: a PDF asserting a 17.4% CAGR is worth nothing if the script
that drew its chart is gone. Everything here reads the same cached point-in-time
rankings the backtest uses, so the figures and `src/strategies.py` cannot drift
apart without this script being re-run and disagreeing.

    python scripts/make_strategy_figures.py

Writes `docs/figures/fig_quality_growth.pdf`, `fig_quality_annual.pdf` and
`report_data.json`. The trend-filter figure is *not* regenerated: that
measurement was made against the pre-DCF-removal ranking and is documented as
such, so redrawing it from today's data would misrepresent what was tested.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from buffett_backtest import annual_returns, simulate, statistics_for  # noqa: E402
from buffett_strategy_search import load_rank_tables  # noqa: E402
from rank_signal_lab import SPLIT, YEARS, blend, holdings_for  # noqa: E402

OUT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "docs", "figures")
)
TRAIN = [y for y in YEARS if y <= SPLIT]
TEST = [y for y in YEARS if y > SPLIT]

# id -> (quality_weight, top_n, max_per_sector), mirroring src/strategies.py.
VARIANTS = {
    "Buffett Quality 20": (0.80, 20, 3),
    "Quality 20, uncapped": (0.80, 20, None),
    "Quality 15": (0.80, 15, 3),
    "Quality, price-blind": (1.00, 20, 3),
}

# Colour-blind-safe and legible in greyscale, since the report gets printed.
COLOURS = {
    "Buffett Quality 20": "#1b4965",
    "S&P 500 (total return)": "#8d99ae",
    "NASDAQ-100 (total return)": "#bc4749",
}


def scorer_for(quality_weight: float):
    def score(table: pd.DataFrame) -> pd.Series:
        quality = pd.to_numeric(table["quality_score"], errors="coerce")
        value = pd.to_numeric(table["value_score"], errors="coerce")
        return blend(quality, value, quality_weight)

    return score


def _style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "figure.constrained_layout.use": True,
        }
    )


def main() -> int:
    tables = load_rank_tables(YEARS, 5)
    if not tables:
        print("No cached rankings — run scripts/buffett_backtest.py first")
        return 1
    panel = pd.read_pickle(
        os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "cache",
                "backtest",
                "monthly_prices.pkl",
            )
        )
    )
    adjusted = panel["Adj Close"]
    _style()

    holdings = {
        name: holdings_for(tables, scorer_for(qw), top_n=n, max_per_sector=cap)
        for name, (qw, n, cap) in VARIANTS.items()
    }

    curves: Dict[str, pd.Series] = {
        name: simulate(h, adjusted, YEARS) for name, h in holdings.items()
    }
    for symbol, label in (
        ("SPY", "S&P 500 (total return)"),
        ("QQQ", "NASDAQ-100 (total return)"),
    ):
        if symbol in adjusted.columns:
            series = (
                adjusted[symbol]
                .loc[f"{YEARS[0] - 1}-12-01" : f"{YEARS[-1]}-12-01"]
                .dropna()
            )
            curves[label] = series / series.iloc[0]

    # --- figure 1: growth of $1, log scale --------------------------------
    shown = [
        "Buffett Quality 20",
        "S&P 500 (total return)",
        "NASDAQ-100 (total return)",
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for name in shown:
        curve = curves[name]
        ax.plot(
            curve.index,
            curve.values,
            label=name,
            color=COLOURS[name],
            linewidth=2.0 if "Quality" in name else 1.3,
        )
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 3, 5, 8, 12])
    ax.get_yaxis().set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"${v:g}")
    )
    ax.set_ylabel("Growth of $1 (log scale)")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "fig_quality_growth.pdf"))
    plt.close(fig)

    # --- figure 2: calendar-year returns ----------------------------------
    annual = {name: annual_returns(curves[name], YEARS) for name in shown}
    frame = pd.DataFrame(annual) * 100
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    frame.plot(
        kind="bar",
        ax=ax,
        width=0.78,
        color=[COLOURS[c] for c in frame.columns],
        legend=True,
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Calendar-year return (%)")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=8)
    ax.tick_params(axis="x", rotation=0)
    fig.savefig(os.path.join(OUT_DIR, "fig_quality_annual.pdf"))
    plt.close(fig)

    # --- data table -------------------------------------------------------
    summary = {}
    for name, curve in curves.items():
        stats = statistics_for(curve)
        record = {
            k: stats[k]
            for k in ("cagr", "volatility", "max_drawdown", "sharpe", "total_return")
        }
        if name in holdings:
            for label, window in (("train_cagr", TRAIN), ("test_cagr", TEST)):
                sub = simulate({y: holdings[name][y] for y in window}, adjusted, window)
                record[label] = statistics_for(sub)["cagr"]
        summary[name] = record

    payload = {
        "window": f"{YEARS[0]}-{YEARS[-1]}",
        "split": SPLIT,
        "annual": {
            k: {str(y): v for y, v in annual_returns(c, YEARS).items()}
            for k, c in curves.items()
        },
        "summary": summary,
        "holdings_latest": holdings["Buffett Quality 20"][max(YEARS)],
    }
    with open(os.path.join(OUT_DIR, "report_data.json"), "w") as handle:
        json.dump(payload, handle, indent=1)

    print(f"Wrote figures and report_data.json to {OUT_DIR}")
    for name, record in summary.items():
        print(
            f"  {name:28s} cagr {record['cagr'] * 100:5.2f}  "
            f"sharpe {record['sharpe']:.2f}  dd {record['max_drawdown'] * 100:6.2f}"
            + (
                f"  train {record['train_cagr'] * 100:5.2f}"
                f"  test {record['test_cagr'] * 100:5.2f}"
                if "train_cagr" in record
                else ""
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
