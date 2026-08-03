"""Does each ranking input actually predict returns, and what should replace it?

`buffett_strategy_search.py` asks which *blend* of the two finished scores works
best. This script asks the prior question: which of the underlying signals earns
its place at all. That distinction matters, because a blend can only reweight
what it is given — if a component is noise, no weight on it is the right weight,
and the search will keep reporting that "less of it is better" without ever
saying why.

Two measurements, in order:

**1. Information coefficient.** For each signal, the Spearman rank correlation
between the signal and the *next* year's total return, computed across the
universe within each rebalance year and then averaged over the thirteen years.
The t-statistic is taken over those thirteen yearly ICs, so it answers "is this
signal reliably positive", not "is it large" — with 13 observations the second
question cannot be answered honestly. Each signal is sign-corrected on the way
in, so a positive IC always means the ranking's stated direction is the helpful
one, and a *negative* IC means the ranking is reading that input backwards.

**2. Rebuilt rankings.** The cached per-year tables carry every raw metric, so a
whole ranking can be reconstructed under a different design and backtested
through exactly the same simulator as production. Percentiles are rebuilt with
the production machinery (`buffett_rank._winsorised_percentile`) inside the
production per-model pillar specs, so a difference between two designs comes
from the design and not from the scaling.

**On reading the output.** The reconstruction of the shipped rule from stored
scores reproduces its published figures, which is the control: if that line
drifts, nothing below it is trustworthy. Beyond that, treat a single winning
cell as noise. What counts is a difference that holds its sign across
neighbouring settings (concentration, industry cap) *and* across both the
2013-2019 and 2020-2025 windows — the standard `buffett_strategy_search.py`
already argues for, applied one level deeper.

**A bad IC is not grounds to remove a signal.** The two sections below answer
different questions and can disagree, so section 1 is a hypothesis generator and
section 2 is the verdict. An IC averages a rank relationship over the entire
distribution; the strategy buys the top twenty of about twelve hundred, which is
a tail operation the average cannot describe. The `financial_strength` pillar is
the standing example: IC -3.0 with a t of -1.6 and negative in both windows, yet
deleting it costs two points of CAGR and nine points of drawdown, because its
real job is to veto a handful of names at the very top whose returns on capital
are high only because they are levered. Anything proposed here on the strength
of an IC alone must be run through the backtest below before it is believed —
and if the two disagree, the backtest wins.

Requires the cached rankings and price panel from `buffett_backtest.py`:

    python scripts/rank_signal_lab.py                 # both sections
    python scripts/rank_signal_lab.py --section ic    # just the ICs
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from buffett_backtest import BENCHMARKS, simulate, statistics_for  # noqa: E402
from buffett_rank import PILLARS_BY_MODEL, _winsorised_percentile  # noqa: E402
from buffett_strategy_search import load_rank_tables, sector_codes  # noqa: E402

YEARS = list(range(2013, 2026))
# The same split `buffett_strategy_search.py` uses, so results are comparable.
SPLIT = 2019

# Production pillar specs, restated as (metric, sign) so a design can edit one.
SPECS: Dict[str, Dict[str, List[Tuple[str, int]]]] = {
    model: {
        pillar: [(metric, 1 if higher else -1) for metric, higher in metrics]
        for pillar, metrics in spec.items()
    }
    for model, spec in PILLARS_BY_MODEL.items()
}
GENERIC = SPECS["generic"]

PILLAR_WEIGHTS = {
    "returns_on_capital": 0.30,
    "financial_strength": 0.20,
    "predictability": 0.20,
    "growth": 0.15,
    "capital_allocation": 0.15,
}

# The production value blend, restricted to the three components the cached
# tables carry. EV/EBIT, P/B and P/S are computed at ranking time but not
# persisted, so a rebuilt value score cannot include them — which is precisely
# why the DCF comparison below is run *within* the rebuilt family rather than
# against the stored score.
VALUE_WEIGHTS = {"margin_of_safety": 0.35, "earnings_yield": 0.20, "fcf_yield": 0.20}


def _cache_dir() -> str:
    return os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "backtest")


# --- section 1: information coefficients -------------------------------------

# (column, higher_is_better). The direction is the ranking's own claim about the
# signal, so a negative IC below is a statement that the claim is wrong.
SIGNALS: Sequence[Tuple[str, bool]] = [
    ("composite_score", True),
    ("quality_score", True),
    ("value_score", True),
    ("confidence", True),
    ("returns_on_capital", True),
    ("financial_strength", True),
    ("predictability", True),
    ("growth", True),
    ("capital_allocation", True),
    ("margin_of_safety", True),
    ("earnings_yield", True),
    ("fcf_yield", True),
    ("ev_to_ebit", False),
    ("price_to_book", False),
    ("price_to_sales", False),
    ("roe_median", True),
    ("roic_median", True),
    ("roa_median", True),
    ("gross_margin_median", True),
    ("net_margin_median", True),
    ("fcf_margin_median", True),
    ("incremental_roic", True),
    ("roe_years_above_15", True),
    ("debt_to_equity", False),
    ("net_debt_to_owner_earnings", False),
    ("interest_coverage", True),
    ("current_ratio", True),
    ("roe_stdev", False),
    ("revenue_growth_stdev", False),
    ("fcf_margin_stdev", False),
    ("revenue_cagr", True),
    ("owner_earnings_cagr", True),
    ("book_value_per_share_cagr", True),
    ("share_count_cagr", False),
    ("payout_ratio_median", True),
    ("momentum", True),
    ("volatility", False),
    ("log_mcap", True),
]


def build_panel(
    tables: Dict[int, pd.DataFrame], adjusted: pd.DataFrame
) -> pd.DataFrame:
    """Stack the per-year tables and attach each company's forward-year return.

    The forward return is measured on the adjusted series (dividends included)
    over exactly the holding period the backtest simulates — December to
    December — so an IC here and a backtest result below describe the same bet.
    """
    frames = []
    for year, table in tables.items():
        start, end = pd.Timestamp(f"{year - 1}-12-01"), pd.Timestamp(f"{year}-12-01")
        if start not in adjusted.index or end not in adjusted.index:
            continue
        table = table[~table.index.duplicated()]
        columns = [s for s in table.index if s in adjusted.columns]
        forward = adjusted.loc[end, columns] / adjusted.loc[start, columns] - 1.0

        # 12-1 momentum, skipping the most recent month (stocks reverse over one
        # month, so including it works against the signal it is meant to carry).
        m0, m1 = pd.Timestamp(f"{year - 2}-11-01"), pd.Timestamp(f"{year - 1}-11-01")
        momentum = (
            adjusted.loc[m1, columns] / adjusted.loc[m0, columns] - 1.0
            if m0 in adjusted.index and m1 in adjusted.index
            else pd.Series(dtype=float)
        )
        window = adjusted.loc[
            f"{year - 4}-01-01" : f"{year - 1}-12-01", columns
        ].pct_change()
        market_cap = pd.to_numeric(table.get("market_cap"), errors="coerce")

        frames.append(
            table.assign(
                year=year,
                fwd_ret=forward.reindex(table.index),
                momentum=momentum.reindex(table.index) if len(momentum) else np.nan,
                volatility=(window.std() * np.sqrt(12)).reindex(table.index),
                log_mcap=np.log(market_cap.where(market_cap > 0)),
            )
        )

    panel = pd.concat(frames)
    panel.index.name = "symbol"
    return panel.reset_index()


def information_coefficients(panel: pd.DataFrame, min_obs: int = 50) -> pd.DataFrame:
    from scipy import stats

    rows = []
    for column, higher in SIGNALS:
        if column not in panel.columns:
            continue
        per_year: Dict[int, float] = {}
        for year, group in panel.groupby("year"):
            x = pd.to_numeric(group[column], errors="coerce")
            y = pd.to_numeric(group["fwd_ret"], errors="coerce")
            usable = x.notna() & y.notna()
            if usable.sum() < min_obs:
                continue
            rho = stats.spearmanr(x[usable], y[usable]).statistic
            per_year[year] = float(rho if higher else -rho)
        if len(per_year) < 5:
            continue
        series = pd.Series(per_year)
        spread = series.std(ddof=1)
        rows.append(
            {
                "signal": column,
                "mean_ic": series.mean() * 100,
                "t_stat": series.mean() / (spread / np.sqrt(len(series)))
                if spread
                else np.nan,
                "hit_rate": (series > 0).mean() * 100,
                "coverage": pd.to_numeric(panel[column], errors="coerce").notna().mean()
                * 100,
                f"ic_{YEARS[0]}_{SPLIT}": series[series.index <= SPLIT].mean() * 100,
                f"ic_{SPLIT + 1}_{YEARS[-1]}": series[series.index > SPLIT].mean()
                * 100,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_ic", ascending=False)


# --- section 2: rebuilt rankings ---------------------------------------------


def percentile(frame: pd.DataFrame, column: str, sign: int) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index)
    return _winsorised_percentile(
        pd.to_numeric(frame[column], errors="coerce"), sign > 0
    )


def weighted(
    parts: Dict[str, pd.Series], weights: Dict[str, float], index
) -> pd.Series:
    """Weighted mean over whichever components resolved, renormalised.

    Identical in behaviour to `buffett_rank._quality_score`: a missing component
    costs its weight rather than scoring zero, so a data gap cannot masquerade
    as a bad reading.
    """
    total = pd.Series(0.0, index=index)
    weight_sum = pd.Series(0.0, index=index)
    for key, series in parts.items():
        weight = weights.get(key, 0.0)
        if not weight or series is None:
            continue
        present = series.notna()
        total = total.add(series.fillna(0.0) * weight * present, fill_value=0.0)
        weight_sum = weight_sum.add(weight * present, fill_value=0.0)
    return total / weight_sum.replace(0.0, np.nan)


def pillar_scores(
    frame: pd.DataFrame, spec: Dict[str, List[Tuple[str, int]]]
) -> Dict[str, pd.Series]:
    scores = {}
    for pillar, metrics in spec.items():
        columns = [percentile(frame, c, s) for c, s in metrics]
        columns = [c for c in columns if c.notna().any()]
        scores[pillar] = (
            pd.concat(columns, axis=1).mean(axis=1, skipna=True)
            if columns
            else pd.Series(np.nan, index=frame.index)
        )
    return scores


def by_model(
    table: pd.DataFrame, fn: Callable[[pd.DataFrame, str], pd.Series]
) -> pd.Series:
    """Score within each valuation model, then concatenate.

    Grouping is not optional: the generic pillar spec applied to a bank leaves
    most of its metrics unresolved, collapsing whole pillars onto a single ratio
    and handing that cohort extreme scores.
    """
    scores = pd.Series(np.nan, index=table.index, dtype=float)
    for model, group in table.groupby("model"):
        scores.loc[group.index] = fn(group, str(model))
    return scores


def reported_yields(frame: pd.DataFrame) -> pd.Series:
    """DCF-free value: the two reported yields, equally weighted."""
    return weighted(
        {
            "earnings_yield": percentile(frame, "earnings_yield", 1),
            "fcf_yield": percentile(frame, "fcf_yield", 1),
        },
        {"earnings_yield": 0.5, "fcf_yield": 0.5},
        frame.index,
    )


def blend(
    quality: pd.Series, value: pd.Series, quality_weight: float = 0.80
) -> pd.Series:
    """A company with no value score keeps its quality score, as production does."""
    return quality.where(
        value.isna(), quality * quality_weight + value * (1 - quality_weight)
    )


def design_shipped(table: pd.DataFrame) -> pd.Series:
    """The live rule, from the scores stored at ranking time. The control."""
    quality = pd.to_numeric(table["quality_score"], errors="coerce")
    return blend(quality, pd.to_numeric(table["value_score"], errors="coerce"))


def design_rebuilt_with_dcf(table: pd.DataFrame) -> pd.Series:
    """The shipped design, rebuilt from raw metrics. Baseline for the DCF test."""

    def score(group: pd.DataFrame, model: str) -> pd.Series:
        quality = weighted(
            pillar_scores(group, SPECS.get(model, GENERIC)), PILLAR_WEIGHTS, group.index
        )
        value = weighted(
            {k: percentile(group, k, 1) for k in VALUE_WEIGHTS},
            VALUE_WEIGHTS,
            group.index,
        )
        return blend(quality, value)

    return by_model(table, score)


def design_no_dcf(table: pd.DataFrame) -> pd.Series:
    """Identical, with the margin of safety removed from the value half."""

    def score(group: pd.DataFrame, model: str) -> pd.Series:
        quality = weighted(
            pillar_scores(group, SPECS.get(model, GENERIC)), PILLAR_WEIGHTS, group.index
        )
        return blend(quality, reported_yields(group))

    return by_model(table, score)


def design_no_dcf_no_book_leverage(table: pd.DataFrame) -> pd.Series:
    """No DCF, and debt-to-equity dropped from the strength pillar.

    Sustained buybacks shrink book equity, so D/E rises fastest for exactly the
    companies the ranking is trying to find; the gate already stopped using it
    for that reason. This asks whether it should score either.
    """

    def score(group: pd.DataFrame, model: str) -> pd.Series:
        spec = {k: list(v) for k, v in SPECS.get(model, GENERIC).items()}
        spec["financial_strength"] = [
            (m, s)
            for m, s in spec.get("financial_strength", [])
            if m != "debt_to_equity"
        ]
        quality = weighted(pillar_scores(group, spec), PILLAR_WEIGHTS, group.index)
        return blend(quality, reported_yields(group))

    return by_model(table, score)


def design_quality_only(table: pd.DataFrame) -> pd.Series:
    def score(group: pd.DataFrame, model: str) -> pd.Series:
        return weighted(
            pillar_scores(group, SPECS.get(model, GENERIC)), PILLAR_WEIGHTS, group.index
        )

    return by_model(table, score)


DESIGNS: Dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "shipped 80/20 (stored scores)": design_shipped,
    "rebuilt 80/20 (with DCF)": design_rebuilt_with_dcf,
    "rebuilt 80/20 (no DCF)": design_no_dcf,
    "no DCF, no book D/E": design_no_dcf_no_book_leverage,
    "quality only": design_quality_only,
}


# --- selection and simulation ------------------------------------------------


def holdings_for(
    tables: Dict[int, pd.DataFrame],
    scorer: Callable[[pd.DataFrame], pd.Series],
    top_n: int = 20,
    max_per_sector: Optional[int] = 3,
    sector_digits: int = 2,
) -> Dict[int, List[str]]:
    holdings: Dict[int, List[str]] = {}
    for year, table in tables.items():
        scores = scorer(table) * pd.to_numeric(
            table.get("confidence"), errors="coerce"
        ).fillna(1.0)
        ordered = scores.dropna().sort_values(ascending=False)
        if not max_per_sector:
            holdings[year] = ordered.head(top_n).index.tolist()
            continue
        sectors = sector_codes(table, sector_digits)
        counts: Dict[str, int] = {}
        picked: List[str] = []
        for symbol in ordered.index:
            group = sectors.get(symbol) or f"?{symbol}"
            if counts.get(group, 0) >= max_per_sector:
                continue
            counts[group] = counts.get(group, 0) + 1
            picked.append(symbol)
            if len(picked) == top_n:
                break
        holdings[year] = picked
    return holdings


def measure(holdings: Dict[int, List[str]], adjusted: pd.DataFrame) -> Dict[str, float]:
    train = [y for y in YEARS if y <= SPLIT]
    test = [y for y in YEARS if y > SPLIT]
    result: Dict[str, float] = {}
    for label, window in (("", YEARS), ("train", train), ("test", test)):
        curve = simulate({y: holdings.get(y, []) for y in window}, adjusted, window)
        stats = statistics_for(curve) if len(curve) >= 12 else {}
        if label:
            result[label] = stats.get("cagr", np.nan) * 100
        else:
            result["cagr"] = stats.get("cagr", np.nan) * 100
            result["sharpe"] = stats.get("sharpe", np.nan)
            result["max_dd"] = stats.get("max_drawdown", np.nan) * 100
    return result


# --- reporting ---------------------------------------------------------------


def report_ic(panel: pd.DataFrame) -> None:
    print("=" * 100)
    print("INFORMATION COEFFICIENT — rank correlation with the next year's return")
    print("Sign-corrected: a negative IC means the ranking reads that input backwards.")
    print("IC and coverage in %, t over the yearly ICs.")
    print("=" * 100)
    table = information_coefficients(panel)
    formatted = table.copy()
    for column in formatted.columns:
        if column != "signal":
            formatted[column] = formatted[column].round(2)
    print(formatted.to_string(index=False))


def report_designs(
    tables: Dict[int, pd.DataFrame],
    adjusted: pd.DataFrame,
    top_n: int,
    cap: Optional[int],
) -> None:
    print("\n" + "=" * 100)
    print(
        f"REBUILT RANKINGS — top {top_n}, max {cap or 'unlimited'} per 2-digit SIC, "
        "rebalanced each January"
    )
    print("=" * 100)
    rows = []
    for name, scorer in DESIGNS.items():
        stats = measure(holdings_for(tables, scorer, top_n, cap), adjusted)
        rows.append({"design": name, **stats})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print(
        "\nStability across neighbouring settings — a real difference holds its sign."
    )
    rows = []
    for name in (
        "shipped 80/20 (stored scores)",
        "rebuilt 80/20 (with DCF)",
        "rebuilt 80/20 (no DCF)",
    ):
        for setting_cap in (2, 3, 4, None):
            stats = measure(
                holdings_for(tables, DESIGNS[name], top_n, setting_cap), adjusted
            )
            rows.append({"design": name, "cap": setting_cap or "none", **stats})
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    print("\nBenchmarks")
    for symbol, label in BENCHMARKS.items():
        if symbol not in adjusted.columns:
            continue
        series = (
            adjusted[symbol]
            .loc[f"{YEARS[0] - 1}-12-01" : f"{YEARS[-1]}-12-01"]
            .dropna()
        )
        if series.empty:
            continue
        stats = statistics_for(series / series.iloc[0])
        print(
            f"  {label:32s} cagr {stats['cagr'] * 100:5.1f}  "
            f"sharpe {stats['sharpe']:.2f}  max_dd {stats['max_drawdown'] * 100:6.1f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--section", choices=("ic", "designs", "both"), default="both")
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--max-per-sector", type=int, default=3)
    parser.add_argument("--min-periods", type=int, default=5)
    args = parser.parse_args()

    tables = load_rank_tables(YEARS, args.min_periods)
    if not tables:
        print("No cached rankings — run scripts/buffett_backtest.py first")
        return 1

    panel_path = os.path.join(_cache_dir(), "monthly_prices.pkl")
    if not os.path.exists(panel_path):
        print(f"No cached price panel at {panel_path} — run buffett_backtest.py first")
        return 1
    adjusted = pd.read_pickle(panel_path)["Adj Close"]

    if args.section in ("ic", "both"):
        panel = build_panel(tables, adjusted)
        panel = panel[panel["fwd_ret"].notna()]
        print(
            f"Panel: {len(panel)} company-years over {panel['year'].nunique()} rebalances\n"
        )
        report_ic(panel)

    if args.section in ("designs", "both"):
        report_designs(tables, adjusted, args.top, args.max_per_sector)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
