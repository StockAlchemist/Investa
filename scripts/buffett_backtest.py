"""Point-in-time backtest of the Buffett/value ranking.

Re-runs the whole ranking once per year as it would have run on the first
trading day of that year, buys the top N names equally weighted, holds for a
calendar year, then re-ranks and rebalances. The result is compared against the
S&P 500 and the NASDAQ Composite.

Three things make this a backtest rather than a hindsight exercise:

  1. **Fundamentals are filed-date filtered.** Every fact in the EDGAR store
     carries the date it was filed, so the ranking for January 2015 sees only
     what had actually been filed by 31 December 2014 — the originally reported
     numbers, not the restatements that followed, and not fiscal years that had
     not yet been reported. `edgar_provider.as_of` enforces this for every read.

  2. **Prices are reconstructed in the money of the day.** Yahoo's close is
     retroactively split-adjusted, so multiplying it by the share count a
     company reported at the time understates the market cap of anything that
     has since split (Apple by 4x, NVIDIA by 40x). The split history is undone
     to recover the price actually quoted, which is then consistent with
     as-reported shares and as-reported EPS. Returns are measured separately off
     the adjusted series, so dividends are included.

  3. **No forward information reaches the ranking.** Every metric is computed
     from the point-in-time statements only; nothing is passed from a current
     quote except the price on the rebalance date itself.

**The known bias that remains** is survivorship: the universe is built from
today's listing files, so companies that were delisted, acquired or went
bankrupt between 2015 and now are not candidates in any year. That flatters the
strategy — and it flatters it more than it flatters the index benchmarks, which
are survivorship-free. Read the outperformance with that discount in mind.

Run from the repo root (first run downloads ~12 years of monthly prices):

    python scripts/buffett_backtest.py --start 2015 --end 2025 --top 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import buffett_rank  # noqa: E402
import buffett_value  # noqa: E402
import config  # noqa: E402
import edgar_provider  # noqa: E402
import edgar_sic  # noqa: E402
import universe  # noqa: E402
from buffett_metrics import CompanyMetrics, compute_metrics  # noqa: E402
from buffett_pipeline import infer_model_from_facts  # noqa: E402

# Total-return proxies (dividends reinvested) alongside the price indices
# themselves. The strategy's returns include dividends, so comparing them to a
# price-only index would hand the strategy roughly 2%/yr it did not earn.
BENCHMARKS = {
    "SPY": "S&P 500 (total return)",
    "QQQ": "NASDAQ-100 (total return)",
    "^GSPC": "S&P 500 (price index)",
    "^IXIC": "NASDAQ Composite (price index)",
}

# XBRL only became mandatory in 2009, and a 10-K carries two years of
# comparatives, so by the end of 2014 the median filer has 5-7 annual periods on
# file — not the 8 the production gate demands. Requiring 8 in 2015 would leave
# 68 eligible companies out of 4,000 and make the early years meaningless.
# `MIN_PERIODS_FOR_CONSISTENCY` (5) is the same floor `buffett_metrics` uses
# before it will compute a consistency measure at all.
BACKTEST_MIN_PERIODS = 5


def _cache_dir() -> str:
    path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "backtest")
    os.makedirs(path, exist_ok=True)
    return path


# --- prices -----------------------------------------------------------------


def _download_prices(symbols: Sequence[str], start: str, end: str) -> Dict[str, pd.DataFrame]:
    """Monthly OHLC + actions for the whole candidate list, in batches."""
    import yfinance as yf

    frames: Dict[str, List[pd.DataFrame]] = {"Close": [], "Adj Close": [], "Stock Splits": []}
    symbols = sorted(set(symbols))
    batch_size = 200

    for start_index in range(0, len(symbols), batch_size):
        batch = symbols[start_index : start_index + batch_size]
        for attempt in range(3):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    interval="1mo",
                    auto_adjust=False,
                    actions=True,
                    progress=False,
                    threads=True,
                )
                break
            except Exception as exc:
                logging.warning(f"Prices: batch {start_index} attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        else:
            continue

        if data is None or data.empty:
            continue
        for field in frames:
            if field in data.columns.get_level_values(0):
                frames[field].append(data[field])
        print(
            f"  prices {min(start_index + batch_size, len(symbols))}/{len(symbols)}",
            flush=True,
        )

    return {
        field: pd.concat(parts, axis=1).sort_index() if parts else pd.DataFrame()
        for field, parts in frames.items()
    }


def load_prices(symbols: Sequence[str], start: str, end: str, refresh: bool = False) -> Dict[str, pd.DataFrame]:
    """Cached monthly price panel. Columns are symbols, index is month start."""
    path = os.path.join(_cache_dir(), "monthly_prices.pkl")
    if not refresh and os.path.exists(path):
        panel = pd.read_pickle(path)
        missing = set(symbols) - set(panel["Close"].columns)
        if len(missing) < 50:
            print(f"Prices: reusing cache ({panel['Close'].shape[1]} symbols)")
            return panel
        print(f"Prices: cache is missing {len(missing)} symbols, refetching")

    panel = _download_prices(list(symbols) + list(BENCHMARKS), start, end)
    pd.to_pickle(panel, path)
    return panel


def split_factors(splits: pd.DataFrame) -> pd.DataFrame:
    """
    Cumulative split factor still *ahead* of each month.

    Yahoo's close is adjusted for every split up to today, so a price in month M
    must be multiplied by the product of all splits that happened after M to
    recover what the stock actually traded at. Computed as a reverse cumulative
    product, which also makes the factor 1.0 for the most recent months.
    """
    ratios = splits.replace(0.0, np.nan).fillna(1.0)
    # Reverse-cumulative product, shifted so a split in month M is *not* applied
    # to M itself: that month's close already reflects it.
    reversed_cumprod = ratios[::-1].cumprod()[::-1]
    return reversed_cumprod.shift(-1).bfill().fillna(1.0)


# --- one year's ranking -----------------------------------------------------


def rank_universe(
    filers: Sequence[Tuple[str, Any]],
    sic_map: Dict[str, int],
    as_of: str,
    prices: Dict[str, float],
    min_periods: int,
) -> pd.DataFrame:
    """
    The full pipeline — metrics, gates, quality, value, composite — restricted
    to what had been filed by `as_of`.

    Mirrors `buffett_pipeline.run` rather than calling it because the pipeline
    fetches live quotes; here the prices come from the historical panel.
    """
    with edgar_provider.as_of(as_of):
        companies: List[CompanyMetrics] = []
        for cik, entry in filers:
            sic = sic_map.get(cik)
            model = edgar_sic.model_for_sic(sic) if sic is not None else infer_model_from_facts(cik)
            company = compute_metrics(cik, entry.symbol, entry.name, model)
            company.gate_failures.extend(
                buffett_rank.evaluate_gates(company, min_periods=min_periods)
            )
            companies.append(company)

        # A company with no quote on the rebalance date could not have been
        # bought, so it is dropped rather than ranked on quality alone.
        eligible = [
            c for c in companies if not c.gate_failures and prices.get(c.symbol) is not None
        ]

        market: Dict[str, Dict[str, Any]] = {}
        for company in eligible:
            shares = _shares_as_of(company)
            if shares:
                market[company.symbol] = {"price": prices[company.symbol], "shares": shares}

        quality_frame = buffett_rank.score_quality(companies)
        value_frame = buffett_value.build_value_frame(eligible, market)
        value_scores = buffett_value.score_value(value_frame) if not value_frame.empty else None

    if value_frame is not None and not value_frame.empty:
        carry = [
            c
            for c in (
                "price",
                "market_cap",
                "earnings_yield",
                "fcf_yield",
                "ev_to_ebit",
                "price_to_book",
                "price_to_sales",
            )
            if c in value_frame.columns
        ]
        quality_frame = quality_frame.join(value_frame[carry], how="left")

    ranked = buffett_rank.combine(quality_frame, value_scores)
    # Only companies that were both eligible *and* buyable can be held.
    buyable = {c.symbol for c in eligible}
    ranked = ranked[ranked.index.isin(buyable)].copy()
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def _shares_as_of(company: CompanyMetrics) -> Optional[float]:
    """
    Share count from the latest filing available at the as-of date.

    Taken from EDGAR rather than a current quote for the obvious reason: today's
    share count applied to a 2015 price would value Apple at a third of what it
    was worth, and would reward every company that has since bought back stock.
    """
    concepts = edgar_provider.get_concept_values(
        company.cik, ["shares_outstanding", "shares_diluted"]
    )
    for name in ("shares_outstanding", "shares_diluted"):
        series = concepts.get(name) or {}
        if not series:
            continue
        latest = max(series)
        value = series[latest]
        if value and value > 0:
            return float(value)
    return None


# --- portfolio simulation ---------------------------------------------------


def simulate(
    holdings_by_year: Dict[int, List[str]],
    adjusted: pd.DataFrame,
    years: Sequence[int],
) -> pd.Series:
    """
    Monthly equity curve for an equal-weight portfolio rebalanced each January.

    Weights are set at the start of the year and then left to drift, which is
    what "hold for a year" actually means. A position whose price series ends
    mid-year (delisting, acquisition) is frozen at its last quote and carried as
    cash for the rest of the year.
    """
    curve: Dict[pd.Timestamp, float] = {}
    capital = 1.0

    for year in years:
        symbols = holdings_by_year.get(year) or []
        if not symbols:
            continue
        window = adjusted.loc[f"{year - 1}-12-01" : f"{year}-12-01", symbols]
        if window.empty:
            continue

        entry = window.iloc[0]
        # Forward-fill so a delisted name holds its last price instead of
        # dropping out of the sum and silently inflating the other weights.
        held = window.ffill()
        weights = capital / len(symbols)
        values = held.divide(entry, axis=1) * weights
        portfolio = values.sum(axis=1, min_count=1)

        for stamp, value in portfolio.items():
            curve[stamp] = value
        capital = float(portfolio.iloc[-1])

    return pd.Series(curve).sort_index()


def annual_returns(curve: pd.Series, years: Sequence[int]) -> Dict[int, float]:
    result = {}
    for year in years:
        try:
            start = curve.loc[f"{year - 1}-12-01"]
            end = curve.loc[f"{year}-12-01"]
        except KeyError:
            continue
        result[year] = float(end / start - 1.0)
    return result


def statistics_for(curve: pd.Series) -> Dict[str, float]:
    monthly = curve.pct_change().dropna()
    years = len(curve) / 12.0
    total = float(curve.iloc[-1] / curve.iloc[0] - 1.0)
    cagr = (1.0 + total) ** (1.0 / years) - 1.0 if years > 0 else float("nan")
    volatility = float(monthly.std() * np.sqrt(12))
    drawdown = float((curve / curve.cummax() - 1.0).min())
    return {
        "total_return": total,
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": drawdown,
        "sharpe": float(cagr / volatility) if volatility else float("nan"),
    }


# --- driver -----------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=2015)
    parser.add_argument("--end", type=int, default=2025)
    parser.add_argument("--top", type=int, default=20)
    parser.add_argument("--min-periods", type=int, default=BACKTEST_MIN_PERIODS)
    parser.add_argument("--limit", type=int, default=None, help="cap the universe (for a smoke run)")
    parser.add_argument("--refresh-prices", action="store_true")
    parser.add_argument("--rerank", action="store_true", help="ignore cached per-year rankings")
    parser.add_argument("--out", default=None, help="write results as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.ERROR, format="%(message)s")
    years = list(range(args.start, args.end + 1))

    # --- universe -----------------------------------------------------------
    entries = universe.get_rankable_universe()
    grouped = universe.group_by_cik(entries)
    with_facts = edgar_provider.get_store().ingested_ciks()
    filers = []
    for cik, listings in sorted(grouped.items()):
        if cik not in with_facts:
            continue
        filers.append((cik, sorted(listings, key=lambda e: (len(e.symbol), e.symbol))[0]))
    if args.limit:
        filers = filers[: args.limit]
    print(f"Universe: {len(filers)} filers with EDGAR fundamentals")

    symbols = [entry.symbol for _cik, entry in filers]
    panel = load_prices(
        symbols, start=f"{args.start - 2}-01-01", end=f"{args.end + 1}-12-31", refresh=args.refresh_prices
    )
    close, adjusted = panel["Close"], panel["Adj Close"]
    factors = split_factors(panel["Stock Splits"].reindex(columns=close.columns).fillna(0.0))
    # The price actually quoted at the time: today's split-adjusted close scaled
    # back up by every split that has happened since.
    actual = close * factors.reindex_like(close).fillna(1.0)

    sic_map = edgar_sic.get_sic_map()

    # --- rank each year -----------------------------------------------------
    holdings: Dict[int, List[str]] = {}
    rank_tables: Dict[int, pd.DataFrame] = {}

    for year in years:
        cache_path = os.path.join(_cache_dir(), f"rank_{year}_{args.min_periods}p.csv")
        if not args.rerank and os.path.exists(cache_path):
            ranked = pd.read_csv(cache_path, index_col=0)
        else:
            as_of = f"{year - 1}-12-31"
            month = pd.Timestamp(f"{year - 1}-12-01")
            if month not in actual.index:
                print(f"{year}: no price row for {month.date()}, skipping")
                continue
            row = actual.loc[month]
            prices = {s: float(v) for s, v in row.items() if pd.notna(v) and v > 0}

            started = time.time()
            ranked = rank_universe(
                filers, sic_map, as_of, prices, args.min_periods
            )
            ranked.to_csv(cache_path)
            print(
                f"{year}: ranked {len(ranked)} companies as of {as_of} "
                f"({time.time() - started:.0f}s)"
            )

        rank_tables[year] = ranked
        # The index carries the symbol; the duplicate `symbol` column survives a
        # CSV round-trip under a mangled name, so it cannot be read back by name.
        holdings[year] = ranked.head(args.top).index.tolist()

    # --- simulate -----------------------------------------------------------
    strategy = simulate(holdings, adjusted, years)
    curves = {"Strategy": strategy}
    for symbol in BENCHMARKS:
        if symbol in adjusted.columns:
            series = adjusted[symbol].loc[f"{years[0] - 1}-12-01" : f"{years[-1]}-12-01"].dropna()
            curves[BENCHMARKS[symbol]] = series / series.iloc[0]

    # --- report -------------------------------------------------------------
    print(f"\n{'=' * 78}")
    print(f"Top-{args.top} Buffett/value ranking, rebalanced each January, {years[0]}-{years[-1]}")
    print("=" * 78)

    table = pd.DataFrame({name: annual_returns(curve, years) for name, curve in curves.items()})
    print("\nCalendar-year returns (%)")
    print((table * 100).round(1).to_string())

    print("\nSummary")
    summary = pd.DataFrame({name: statistics_for(curve) for name, curve in curves.items()}).T
    formatted = summary.copy()
    for column in ("total_return", "cagr", "volatility", "max_drawdown"):
        formatted[column] = (summary[column] * 100).round(1)
    formatted["sharpe"] = summary["sharpe"].round(2)
    print(formatted.to_string())

    print("\nHoldings")
    for year in years:
        if year in holdings:
            print(f"  {year}: {', '.join(holdings[year])}")

    if args.out:
        payload = {
            "years": years,
            "top": args.top,
            "holdings": holdings,
            "annual_returns": {k: {str(y): v for y, v in annual_returns(c, years).items()} for k, c in curves.items()},
            "summary": {k: statistics_for(c) for k, c in curves.items()},
            "curves": {k: {str(i.date()): float(v) for i, v in c.items()} for k, c in curves.items()},
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
