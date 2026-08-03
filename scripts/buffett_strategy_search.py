"""Search for a better rule on top of the Buffett/value ranking.

The per-year rankings cached by `buffett_backtest.py` already contain the two
component scores (quality and value), the confidence factor and the market cap
each company had on the rebalance date. That means a large family of strategies
can be evaluated without re-running the expensive ranking: change how the
components are combined, filter the candidate set, take the top N, simulate.

Five knobs are swept:

  * **quality_weight** — the production blend is 0.60 quality / 0.40 value.
  * **top_n** — how concentrated the book is.
  * **min_market_cap** — a floor that removes microcaps, where a percentile
    ranking of thinly-reported companies is mostly noise and where the backtest
    is least honest about what could actually have been bought.
  * **momentum** — 12-1 month trailing return, either as a filter (refuse to buy
    falling knives) or blended into the score. Value and momentum are the
    classic complement: value says what is cheap, momentum says whether the
    market has stopped disagreeing.
  * **confidence** — whether the data-coverage penalty is applied at all.

**On data mining.** A sweep of ~700 configurations over 13 years will always
produce something that beats the index; that is arithmetic, not skill. Two
guards are applied. First, the sweep is scored on a training window and the
winner is re-measured on a held-out window it never saw. Second — and more
useful — the report shows the *marginal* effect of each knob, median across
every other setting. A knob that helps across the whole grid is a finding; a
single configuration at the top of a noisy grid is not.

Run from the repo root (requires cached rankings from buffett_backtest.py):

    python scripts/buffett_strategy_search.py --start 2013 --end 2025
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from buffett_backtest import BENCHMARKS, annual_returns, simulate, statistics_for  # noqa: E402


@dataclass(frozen=True)
class Strategy:
    quality_weight: float = 0.60
    top_n: int = 20
    min_market_cap: float = 0.0
    momentum_weight: float = 0.0
    momentum_floor: Optional[float] = None
    use_confidence: bool = True
    # Two-stage screen: keep the best `quality_gate_pct` of the universe on
    # quality, then rank *those* on value. Structurally different from a
    # weighted blend — a blend lets a spectacular price offset a mediocre
    # business, which is the ordering Buffett explicitly rejects.
    quality_gate_pct: Optional[float] = None
    exclude_models: tuple = ()
    # Industry cap. The ranking scores companies one at a time and has no notion
    # of correlation, so it will happily fill a fifth of the book with four
    # mortgage insurers that share a single risk factor. Capping walks down the
    # ranked list and skips a name once its industry is full, filling the slot
    # from further down instead.
    max_per_sector: Optional[int] = None
    sector_digits: int = 3

    def label(self) -> str:
        if self.quality_gate_pct is not None:
            parts = [f"top{self.quality_gate_pct:.0%}quality->value", f"n{self.top_n}"]
        else:
            parts = [f"q{self.quality_weight:.1f}", f"n{self.top_n}"]
        if self.min_market_cap:
            parts.append(f"mcap{self.min_market_cap / 1e9:g}B")
        if self.momentum_weight:
            parts.append(f"mom{self.momentum_weight:.2f}")
        if self.momentum_floor is not None:
            parts.append(f"momfloor{self.momentum_floor:+.0%}")
        if not self.use_confidence:
            parts.append("noconf")
        if self.exclude_models:
            parts.append("ex-" + "/".join(self.exclude_models))
        if self.max_per_sector:
            parts.append(f"max{self.max_per_sector}/sic{self.sector_digits}")
        return " ".join(parts)


def _cache_dir() -> str:
    return os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "backtest")


def load_rank_tables(
    years: Sequence[int], min_periods: int = 5
) -> Dict[int, pd.DataFrame]:
    tables = {}
    for year in years:
        path = os.path.join(_cache_dir(), f"rank_{year}_{min_periods}p.csv")
        if os.path.exists(path):
            tables[year] = pd.read_csv(path, index_col=0)
    return tables


def attach_momentum(tables: Dict[int, pd.DataFrame], adjusted: pd.DataFrame) -> None:
    """
    12-1 month trailing total return as of each rebalance.

    The most recent month is skipped, which is the standard construction: over a
    one-month horizon stocks reverse rather than continue, so including it works
    against the signal it is meant to capture.
    """
    for year, table in tables.items():
        end = pd.Timestamp(f"{year - 1}-11-01")
        start = pd.Timestamp(f"{year - 2}-11-01")
        if end not in adjusted.index or start not in adjusted.index:
            table["momentum"] = np.nan
            continue
        columns = [s for s in table.index if s in adjusted.columns]
        window = adjusted.loc[[start, end], columns]
        trailing = window.iloc[1] / window.iloc[0] - 1.0
        table["momentum"] = trailing.reindex(table.index)


def select(table: pd.DataFrame, strategy: Strategy) -> List[str]:
    """Apply one strategy to one year's ranking and return the holdings."""
    frame = table.copy()

    if strategy.exclude_models:
        frame = frame[~frame["model"].isin(strategy.exclude_models)]
    if strategy.min_market_cap > 0:
        frame = frame[
            pd.to_numeric(frame.get("market_cap"), errors="coerce")
            >= strategy.min_market_cap
        ]
    if strategy.momentum_floor is not None:
        frame = frame[
            pd.to_numeric(frame.get("momentum"), errors="coerce")
            >= strategy.momentum_floor
        ]
    if frame.empty:
        return []

    quality = pd.to_numeric(frame["quality_score"], errors="coerce")
    value = pd.to_numeric(frame["value_score"], errors="coerce")

    if strategy.quality_gate_pct is not None:
        keep = quality >= quality.quantile(1.0 - strategy.quality_gate_pct)
        frame = frame[keep.fillna(False)]
        if frame.empty:
            return []
        quality, value = quality[frame.index], value[frame.index]
        # Inside the quality cohort, price is the only thing left to decide on.
        blended = value.where(value.notna(), quality)
    else:
        # A company with no value score keeps its quality score rather than being
        # dropped, which is what `buffett_rank.combine` does.
        blended = quality.where(
            value.isna(),
            quality * strategy.quality_weight + value * (1 - strategy.quality_weight),
        )

    if strategy.momentum_weight:
        # Re-ranked within the surviving candidate set, so the percentile means
        # "relative to what this strategy would actually consider buying".
        momentum = (
            pd.to_numeric(frame["momentum"], errors="coerce").rank(pct=True) * 100.0
        )
        blended = blended.where(
            momentum.isna(),
            blended * (1 - strategy.momentum_weight)
            + momentum * strategy.momentum_weight,
        )

    if strategy.use_confidence:
        blended = blended * pd.to_numeric(frame["confidence"], errors="coerce").fillna(
            1.0
        )

    ordered = blended.sort_values(ascending=False)
    if not strategy.max_per_sector:
        return ordered.head(strategy.top_n).index.tolist()

    sectors = sector_codes(frame, strategy.sector_digits)
    counts: Dict[str, int] = {}
    picked: List[str] = []
    for symbol in ordered.index:
        # A company with no SIC code is its own industry rather than lumped with
        # every other unclassified name, which would cap them against each other.
        sector = sectors.get(symbol) or f"?{symbol}"
        if counts.get(sector, 0) >= strategy.max_per_sector:
            continue
        counts[sector] = counts.get(sector, 0) + 1
        picked.append(symbol)
        if len(picked) == strategy.top_n:
            break
    return picked


_SIC_MAP: Optional[Dict[str, int]] = None


def sector_codes(frame: pd.DataFrame, digits: int) -> Dict[str, str]:
    """
    symbol -> truncated SIC code.

    SIC rather than a vendor sector string because it is already cached, it is
    stable through time (so it cannot leak a later reclassification into an
    earlier year), and its granularity is controllable: 6351 is mortgage
    insurance, 635 is surety, 63 is every insurance carrier.
    """
    global _SIC_MAP
    if _SIC_MAP is None:
        import edgar_sic

        _SIC_MAP = edgar_sic.get_sic_map()

    result: Dict[str, str] = {}
    if "cik" not in frame.columns:
        return result
    for symbol, cik in frame["cik"].items():
        try:
            key = str(int(float(cik))).zfill(10)
        except (TypeError, ValueError):
            continue
        sic = _SIC_MAP.get(key)
        if sic is not None:
            result[symbol] = str(int(sic)).zfill(4)[:digits]
    return result


def evaluate(
    strategy: Strategy,
    tables: Dict[int, pd.DataFrame],
    adjusted: pd.DataFrame,
    years: Sequence[int],
) -> Dict[str, float]:
    holdings = {
        year: select(tables[year], strategy) for year in years if year in tables
    }
    curve = simulate(holdings, adjusted, years)
    if curve.empty or len(curve) < 12:
        return {
            "cagr": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
        }
    return statistics_for(curve)


def benchmark_curves(
    adjusted: pd.DataFrame, years: Sequence[int]
) -> Dict[str, pd.Series]:
    curves = {}
    for symbol, name in BENCHMARKS.items():
        if symbol in adjusted.columns:
            series = (
                adjusted[symbol]
                .loc[f"{years[0] - 1}-12-01" : f"{years[-1]}-12-01"]
                .dropna()
            )
            if not series.empty:
                curves[name] = series / series.iloc[0]
    return curves


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2013)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument(
        "--split", type=int, default=2019, help="last year of the training window"
    )
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--top-show", type=int, default=15)
    parser.add_argument("--rec-quality", type=float, default=0.8)
    parser.add_argument("--rec-top", type=int, default=20)
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))
    tables = load_rank_tables(years, args.min_periods)
    missing = [y for y in years if y not in tables]
    if missing:
        print(
            f"No cached ranking for {missing} — run buffett_backtest.py for those years first"
        )
        years = [y for y in years if y in tables]

    panel = pd.read_pickle(os.path.join(_cache_dir(), "monthly_prices.pkl"))
    adjusted = panel["Adj Close"]
    attach_momentum(tables, adjusted)

    train = [y for y in years if y <= args.split]
    test = [y for y in years if y > args.split]

    grid = [
        Strategy(
            quality_weight=q,
            top_n=n,
            min_market_cap=m,
            momentum_weight=mw,
            momentum_floor=mf,
        )
        for q, n, m, (mw, mf) in itertools.product(
            (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
            (5, 10, 15, 20, 30, 50),
            (0.0, 3e8, 1e9, 2e9, 1e10),
            ((0.0, None), (0.0, 0.0), (0.25, None), (0.5, None), (0.25, 0.0)),
        )
    ]
    print(f"Evaluating {len(grid)} strategies over {years[0]}-{years[-1]}")

    rows = []
    for strategy in grid:
        full = evaluate(strategy, tables, adjusted, years)
        rows.append(
            {
                "label": strategy.label(),
                "quality_weight": strategy.quality_weight,
                "top_n": strategy.top_n,
                "min_market_cap": strategy.min_market_cap,
                "momentum_weight": strategy.momentum_weight,
                "momentum_floor": strategy.momentum_floor,
                "cagr": full["cagr"],
                "sharpe": full["sharpe"],
                "max_drawdown": full["max_drawdown"],
                "train_cagr": evaluate(strategy, tables, adjusted, train)["cagr"]
                if train
                else np.nan,
                "test_cagr": evaluate(strategy, tables, adjusted, test)["cagr"]
                if test
                else np.nan,
            }
        )
    results = pd.DataFrame(rows)

    benchmarks = benchmark_curves(adjusted, years)
    print("\nBenchmarks over the full window")
    bench_stats = pd.DataFrame(
        {name: statistics_for(c) for name, c in benchmarks.items()}
    ).T
    print(
        (bench_stats[["cagr", "sharpe", "max_drawdown"]] * [100, 1, 100])
        .round(2)
        .to_string()
    )

    print("\n" + "=" * 78)
    print("MARGINAL EFFECT OF EACH KNOB (median CAGR %, across all other settings)")
    print("A knob is only interesting if it points the same way in both windows.")
    print("=" * 78)
    for knob in (
        "quality_weight",
        "top_n",
        "min_market_cap",
        "momentum_weight",
        "momentum_floor",
    ):
        grouped = results.groupby(results[knob].fillna(-99), dropna=False)
        summary = pd.DataFrame(
            {
                "full": grouped["cagr"].median() * 100,
                f"train {train[0]}-{train[-1]}": grouped["train_cagr"].median() * 100,
                f"test {test[0]}-{test[-1]}": grouped["test_cagr"].median() * 100,
            }
        ).round(2)
        print(f"\n{knob}")
        print(summary.to_string())

    print("\n" + "=" * 78)
    print(
        f"BEST ON TRAINING WINDOW {train[0]}-{train[-1]}, MEASURED ON {test[0]}-{test[-1]}"
    )
    print("=" * 78)
    ranked = results.sort_values("train_cagr", ascending=False).head(args.top_show)
    show = ranked[
        ["label", "train_cagr", "test_cagr", "cagr", "sharpe", "max_drawdown"]
    ].copy()
    for column in ("train_cagr", "test_cagr", "cagr", "max_drawdown"):
        show[column] = (show[column] * 100).round(1)
    show["sharpe"] = show["sharpe"].round(2)
    print(show.to_string(index=False))

    print(
        f"\nMedian test CAGR of the top-{args.top_show} training configs: "
        f"{ranked['test_cagr'].median() * 100:.1f}%  "
        f"(median across the whole grid: {results['test_cagr'].median() * 100:.1f}%)"
    )

    print("\n" + "=" * 78)
    print("BEST OVER THE FULL WINDOW (in-sample — read as an upper bound)")
    print("=" * 78)
    best = results.sort_values("cagr", ascending=False).head(args.top_show)
    show = best[
        ["label", "cagr", "sharpe", "max_drawdown", "train_cagr", "test_cagr"]
    ].copy()
    for column in ("cagr", "max_drawdown", "train_cagr", "test_cagr"):
        show[column] = (show[column] * 100).round(1)
    show["sharpe"] = show["sharpe"].round(2)
    print(show.to_string(index=False))

    print("\n" + "=" * 78)
    print("STRUCTURALLY DISTINCT CANDIDATES (not a sweep — each is a different rule)")
    print("=" * 78)
    candidates = [
        Strategy(quality_weight=0.6, top_n=20),  # production baseline
        Strategy(quality_weight=0.8, top_n=20),
        Strategy(quality_weight=1.0, top_n=20),
        Strategy(quality_weight=0.0, top_n=20),
        Strategy(quality_weight=0.6, top_n=20, use_confidence=False),
        Strategy(quality_gate_pct=0.10, top_n=20),
        Strategy(quality_gate_pct=0.20, top_n=20),
        Strategy(quality_gate_pct=0.30, top_n=20),
        Strategy(quality_gate_pct=0.20, top_n=20, min_market_cap=1e9),
        Strategy(quality_gate_pct=0.20, top_n=20, momentum_floor=0.0),
        Strategy(quality_weight=0.6, top_n=20, exclude_models=("reit",)),
        Strategy(
            quality_weight=0.6, top_n=20, exclude_models=("reit", "bank", "insurer")
        ),
        Strategy(quality_weight=0.8, top_n=20, min_market_cap=1e9),
        Strategy(quality_weight=0.8, top_n=20, momentum_weight=0.25),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=1, sector_digits=3),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=2, sector_digits=3),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=3, sector_digits=3),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=2, sector_digits=2),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=3, sector_digits=2),
        Strategy(quality_weight=0.8, top_n=20, max_per_sector=4, sector_digits=2),
        Strategy(quality_weight=0.6, top_n=20, max_per_sector=3, sector_digits=2),
    ]
    rows = []
    for strategy in candidates:
        full = evaluate(strategy, tables, adjusted, years)
        rows.append(
            {
                "strategy": strategy.label(),
                "cagr": full["cagr"] * 100,
                "sharpe": full["sharpe"],
                "max_dd": full["max_drawdown"] * 100,
                "train": evaluate(strategy, tables, adjusted, train)["cagr"] * 100,
                "test": evaluate(strategy, tables, adjusted, test)["cagr"] * 100,
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\n" + "=" * 78)
    print(
        f"RECOMMENDED RULE, YEAR BY YEAR (quality {args.rec_quality:.0%} / value "
        f"{1 - args.rec_quality:.0%}, top {args.rec_top})"
    )
    print("=" * 78)
    recommended = Strategy(quality_weight=args.rec_quality, top_n=args.rec_top)
    holdings = {year: select(tables[year], recommended) for year in years}
    curves = {"Recommended": simulate(holdings, adjusted, years)}
    baseline = Strategy(quality_weight=0.60, top_n=args.rec_top)
    curves["Current rule (60/40)"] = simulate(
        {year: select(tables[year], baseline) for year in years}, adjusted, years
    )
    curves.update(benchmarks)

    table = pd.DataFrame(
        {name: annual_returns(curve, years) for name, curve in curves.items()}
    )
    print((table * 100).round(1).to_string())
    print()
    summary = pd.DataFrame(
        {name: statistics_for(curve) for name, curve in curves.items()}
    ).T
    formatted = summary.copy()
    for column in ("total_return", "cagr", "volatility", "max_drawdown"):
        formatted[column] = (summary[column] * 100).round(1)
    formatted["sharpe"] = summary["sharpe"].round(2)
    print(formatted.to_string())

    print("\nSensitivity of the recommendation to concentration")
    sizes = []
    for size in (5, 10, 15, 20, 30, 50):
        variant = Strategy(quality_weight=args.rec_quality, top_n=size)
        stats = evaluate(variant, tables, adjusted, years)
        sizes.append(
            {
                "top_n": size,
                "cagr": stats["cagr"] * 100,
                "sharpe": stats["sharpe"],
                "max_dd": stats["max_drawdown"] * 100,
                "train": evaluate(variant, tables, adjusted, train)["cagr"] * 100,
                "test": evaluate(variant, tables, adjusted, test)["cagr"] * 100,
            }
        )
    print(pd.DataFrame(sizes).round(2).to_string(index=False))

    print("\nHoldings under the recommended rule")
    for year in years:
        print(f"  {year}: {', '.join(holdings[year])}")

    out = os.path.join(_cache_dir(), "strategy_search.csv")
    results.to_csv(out, index=False)
    print(f"\nFull grid written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
