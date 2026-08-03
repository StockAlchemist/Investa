"""Quality-gate + momentum, rebalanced monthly, on the point-in-time rankings.

The annual sweep in `buffett_strategy_search.py` established that momentum does
not help a portfolio that is only touched once a year — which says less about
momentum than about the holding period, since the 12-1 signal decays over a few
months. This script asks the structurally different question: keep the Buffett
quality screen as an annual *gate* (it moves slowly), but let a monthly
momentum rank decide which of the gated names to hold.

Mechanics per month M (entered at the close of month M-1):

  * Universe: the cached point-in-time ranking for M's calendar year — built
    from filings available on 31 Dec of the prior year, so nothing the gate
    knows postdates the holdings.
  * Gate: the top `quality_pct` of that table by quality score, optionally
    with a market-cap floor.
  * Signal: 12-1 momentum measured at the close of M-1 (skip the most recent
    month, the standard construction).
  * Selection with a hold buffer: buy from the top `top_n`, but an incumbent
    is kept until it falls below rank `buffer_n` (> top_n). The buffer exists
    because a monthly top-N portfolio without one churns most of the book on
    rank jitter alone.
  * Costs: `cost_bps` one-way on actual turnover, charged monthly. Backtests
    that rebalance monthly and charge nothing are advertising.
  * Optional crash filter: when SPY sits below its `sma_months` moving
    average at the close of M-1, the book moves to cash for month M.

Same honesty caveats as `buffett_backtest.py`: the universe is survivorship
biased (today's listing files), which flatters every long-only variant here
relative to the indices. Train/test discipline is kept: the split year is a
parameter and every table shows both windows.

Run from the repo root (requires cached rankings + price panel):

    python scripts/buffett_momentum_lab.py --start 2013 --end 2025
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from buffett_backtest import BENCHMARKS, statistics_for  # noqa: E402
from buffett_strategy_search import Strategy, load_rank_tables, select  # noqa: E402


def _cache_dir() -> str:
    return os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "backtest")


@dataclass(frozen=True)
class MomentumStrategy:
    quality_pct: float = 0.30          # keep this share of the year's table, by quality
    top_n: int = 20
    buffer_n: int = 40                 # incumbent survives until it drops below this rank
    min_market_cap: float = 0.0
    cost_bps: float = 15.0             # one-way, charged on turnover
    sma_months: Optional[int] = None   # crash filter on SPY; None disables
    momentum_blend: float = 1.0        # 1.0 = rank purely on momentum inside the gate;
                                       # less blends the quality percentile back in
    label_extra: str = ""

    def label(self) -> str:
        parts = [f"q{self.quality_pct:.0%}", f"n{self.top_n}/{self.buffer_n}"]
        if self.min_market_cap:
            parts.append(f"mcap{self.min_market_cap / 1e9:g}B")
        if self.momentum_blend < 1.0:
            parts.append(f"blend{self.momentum_blend:.2f}")
        if self.sma_months:
            parts.append(f"sma{self.sma_months}")
        if self.label_extra:
            parts.append(self.label_extra)
        return " ".join(parts)


def month_index(adjusted: pd.DataFrame, start_year: int, end_year: int) -> List[pd.Timestamp]:
    """Month rows to be *held* — entry is the close of the previous row."""
    stamps = [s for s in adjusted.index if start_year <= s.year <= end_year]
    return sorted(stamps)


def gated_universe(
    table: pd.DataFrame, strategy: MomentumStrategy
) -> Tuple[List[str], Dict[str, float]]:
    """Symbols passing the quality gate for one calendar year, plus quality pct."""
    frame = table.copy()
    if strategy.min_market_cap > 0:
        caps = pd.to_numeric(frame.get("market_cap"), errors="coerce")
        frame = frame[caps >= strategy.min_market_cap]
    quality = pd.to_numeric(frame["quality_score"], errors="coerce").dropna()
    if quality.empty:
        return [], {}
    keep = quality[quality >= quality.quantile(1.0 - strategy.quality_pct)]
    percentile = keep.rank(pct=True) * 100.0
    return keep.index.tolist(), percentile.to_dict()


def run_monthly(
    strategy: MomentumStrategy,
    tables: Dict[int, pd.DataFrame],
    adjusted: pd.DataFrame,
    months: Sequence[pd.Timestamp],
) -> Tuple[pd.Series, float]:
    """
    Equity curve (net of costs) and average annual one-way turnover.

    Month rows are month closes: entering month M uses information up to the
    close of row M-1 and earns adjusted[M] / adjusted[M-1].
    """
    position = adjusted.index.get_indexer([months[0]])[0]
    spy = adjusted["SPY"]

    capital = 1.0
    curve: Dict[pd.Timestamp, float] = {}
    holdings: List[str] = []
    turnover_total = 0.0

    # Pre-compute per-year gates once.
    gates: Dict[int, Tuple[List[str], Dict[str, float]]] = {}
    for year, table in tables.items():
        gates[year] = gated_universe(table, strategy)

    for offset, month in enumerate(months):
        row = position + offset
        formation = adjusted.index[row - 1]          # close of M-1: information edge
        lookback = adjusted.index[row - 13]          # close of M-13
        year_gate, quality_pct = gates.get(month.year, ([], {}))

        in_cash = False
        if strategy.sma_months:
            window = spy.iloc[max(0, row - strategy.sma_months) : row]
            if len(window.dropna()) >= strategy.sma_months and spy.loc[formation] < window.mean():
                in_cash = True

        if in_cash:
            target: List[str] = []
        else:
            symbols = [s for s in year_gate if s in adjusted.columns]
            past = adjusted.loc[lookback, symbols]
            recent = adjusted.loc[formation, symbols]
            momentum = (recent / past - 1.0).dropna()
            if momentum.empty:
                target = []
            else:
                score = momentum.rank(pct=True) * 100.0
                if strategy.momentum_blend < 1.0:
                    quality_series = pd.Series(quality_pct).reindex(score.index)
                    score = (
                        score * strategy.momentum_blend
                        + quality_series.fillna(0.0) * (1.0 - strategy.momentum_blend)
                    )
                ordered = score.sort_values(ascending=False)
                ranks = {s: i + 1 for i, s in enumerate(ordered.index)}
                # Incumbents first: keep anything still inside the buffer.
                target = [s for s in holdings if ranks.get(s, 10**9) <= strategy.buffer_n]
                for symbol in ordered.index:
                    if len(target) >= strategy.top_n:
                        break
                    if symbol not in target:
                        target.append(symbol)
                target = target[: strategy.top_n]

        # One-way turnover as a fraction of the book (equal weights).
        if holdings or target:
            previous, current = set(holdings), set(target)
            denominator = max(len(previous), len(current), 1)
            changed = len(previous.symmetric_difference(current)) / (2.0 * denominator)
            # Entering from all-cash or exiting to all-cash turns the whole book.
            if not previous or not current:
                changed = 1.0 if previous != current else 0.0
            turnover_total += changed
            capital *= 1.0 - changed * 2.0 * strategy.cost_bps / 10000.0

        if target:
            entry = adjusted.loc[formation, target]
            exit_ = adjusted.loc[month, target]
            returns = (exit_ / entry - 1.0).fillna(0.0)   # NaN exit: frozen at last quote
            capital *= 1.0 + float(returns.mean())
        holdings = target
        curve[month] = capital

    years = len(months) / 12.0
    annual_turnover = turnover_total / years if years else float("nan")
    series = pd.Series(curve).sort_index()
    # Prepend the entry point so statistics_for sees the true starting value.
    entry_stamp = adjusted.index[position - 1]
    return pd.concat([pd.Series({entry_stamp: 1.0}), series]), annual_turnover


def run_annual_baseline(
    strategy: Strategy,
    tables: Dict[int, pd.DataFrame],
    adjusted: pd.DataFrame,
    years: Sequence[int],
) -> pd.Series:
    """The existing annual-rebalance rule, for comparison, with annual costs."""
    from buffett_backtest import simulate

    holdings = {year: select(tables[year], strategy) for year in years if year in tables}
    return simulate(holdings, adjusted, years)


def index_trend_reference(
    adjusted: pd.DataFrame, months: Sequence[pd.Timestamp], symbol: str, sma: int
) -> pd.Series:
    """Buy-and-hold `symbol` gated by its own SMA — the classic trend overlay."""
    series = adjusted[symbol]
    position = adjusted.index.get_indexer([months[0]])[0]
    capital, curve = 1.0, {}
    for offset, month in enumerate(months):
        row = position + offset
        formation = adjusted.index[row - 1]
        window = series.iloc[max(0, row - sma) : row]
        invested = len(window.dropna()) >= sma and series.loc[formation] >= window.mean()
        if invested:
            capital *= float(series.loc[month] / series.loc[formation])
        curve[month] = capital
    entry = adjusted.index[position - 1]
    return pd.concat([pd.Series({entry: 1.0}), pd.Series(curve).sort_index()])


def window_stats(curve: pd.Series, start_year: int, end_year: int) -> Dict[str, float]:
    window = curve.loc[f"{start_year - 1}-12-01" : f"{end_year}-12-01"]
    if len(window) < 12:
        return {k: float("nan") for k in ("total_return", "cagr", "volatility", "max_drawdown", "sharpe")}
    return statistics_for(window / window.iloc[0])


def annual_returns_from_curve(curve: pd.Series, years: Sequence[int]) -> Dict[int, float]:
    out = {}
    for year in years:
        try:
            start = curve.loc[f"{year - 1}-12-01"]
            end = curve.loc[f"{year}-12-01"]
        except KeyError:
            continue
        out[year] = float(end / start - 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2013)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--split", type=int, default=2019, help="last year of the training window")
    parser.add_argument("--min-periods", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=15.0)
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args()

    years = list(range(args.start, args.end + 1))
    tables = load_rank_tables(years, args.min_periods)
    missing = [y for y in years if y not in tables]
    if missing:
        print(f"No cached ranking for {missing} — run buffett_backtest.py first")
        return 1

    panel = pd.read_pickle(os.path.join(_cache_dir(), "monthly_prices.pkl"))
    adjusted = panel["Adj Close"]
    months = month_index(adjusted, args.start, args.end)

    train_years = [y for y in years if y <= args.split]
    test_years = [y for y in years if y > args.split]

    candidates: List[MomentumStrategy] = [
        # The gate axis.
        MomentumStrategy(quality_pct=0.10, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.20, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.30, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.50, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=1.00, cost_bps=args.cost_bps, label_extra="nogate"),
        # Concentration axis.
        MomentumStrategy(quality_pct=0.30, top_n=10, buffer_n=20, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.30, top_n=15, buffer_n=30, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.30, top_n=30, buffer_n=60, cost_bps=args.cost_bps),
        # Market-cap floor.
        MomentumStrategy(quality_pct=0.30, min_market_cap=1e9, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.30, min_market_cap=1e10, cost_bps=args.cost_bps),
        # Blend quality back into the monthly rank.
        MomentumStrategy(quality_pct=0.30, momentum_blend=0.5, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.30, momentum_blend=0.75, cost_bps=args.cost_bps),
        # Crash filter.
        MomentumStrategy(quality_pct=0.30, sma_months=10, cost_bps=args.cost_bps),
        MomentumStrategy(quality_pct=0.20, sma_months=10, cost_bps=args.cost_bps),
        # No buffer, as a control for how much the buffer is worth after costs.
        MomentumStrategy(quality_pct=0.30, buffer_n=20, cost_bps=args.cost_bps, label_extra="nobuffer"),
    ]

    rows = []
    curves: Dict[str, pd.Series] = {}
    turnovers: Dict[str, float] = {}
    for strategy in candidates:
        curve, turnover = run_monthly(strategy, tables, adjusted, months)
        curves[strategy.label()] = curve
        turnovers[strategy.label()] = turnover
        full = window_stats(curve, args.start, args.end)
        train = window_stats(curve, train_years[0], train_years[-1])
        test = window_stats(curve, test_years[0], test_years[-1])
        rows.append(
            {
                "strategy": strategy.label(),
                "cagr": full["cagr"] * 100,
                "sharpe": full["sharpe"],
                "max_dd": full["max_drawdown"] * 100,
                "turnover x/yr": turnover,
                f"train {train_years[0]}-{train_years[-1]}": train["cagr"] * 100,
                f"test {test_years[0]}-{test_years[-1]}": test["cagr"] * 100,
            }
        )

    print(f"\nQuality gate + monthly momentum, {args.start}-{args.end}, costs {args.cost_bps:g} bps one-way")
    print("=" * 100)
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    # --- references ---------------------------------------------------------
    print("\nReferences")
    print("=" * 100)
    reference_rows = []

    annual_curve = run_annual_baseline(Strategy(quality_weight=0.8, top_n=20), tables, adjusted, years)
    curves["annual 80/20 top20 (prior best)"] = annual_curve
    reference_rows.append(("annual 80/20 top20 (prior best)", annual_curve))

    qqq_trend = index_trend_reference(adjusted, months, "QQQ", 10)
    curves["QQQ + 10m SMA"] = qqq_trend
    reference_rows.append(("QQQ + 10m SMA", qqq_trend))

    for symbol, name in BENCHMARKS.items():
        if symbol in adjusted.columns:
            series = adjusted[symbol].loc[f"{args.start - 1}-12-01" : f"{args.end}-12-01"].dropna()
            curves[name] = series / series.iloc[0]
            reference_rows.append((name, curves[name]))

    rows = []
    for name, curve in reference_rows:
        full = window_stats(curve, args.start, args.end)
        train = window_stats(curve, train_years[0], train_years[-1])
        test = window_stats(curve, test_years[0], test_years[-1])
        rows.append(
            {
                "series": name,
                "cagr": full["cagr"] * 100,
                "sharpe": full["sharpe"],
                "max_dd": full["max_drawdown"] * 100,
                f"train {train_years[0]}-{train_years[-1]}": train["cagr"] * 100,
                f"test {test_years[0]}-{test_years[-1]}": test["cagr"] * 100,
            }
        )
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    if args.out:
        payload = {
            "months": [str(m.date()) for m in months],
            "curves": {k: {str(i.date()): float(v) for i, v in c.items()} for k, c in curves.items()},
            "turnover": turnovers,
            "annual_returns": {
                k: {str(y): v for y, v in annual_returns_from_curve(c, years).items()}
                for k, c in curves.items()
            },
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
