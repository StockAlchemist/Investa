"""Trend-filtered (optionally leveraged) index exposure, backtested to the 1970s.

The stock-picking work hit a ceiling: the annual 80/20 Buffett rule beats the
S&P but nothing selection-based beats the NASDAQ-100 over 2013-2025
(`buffett_strategy_search.py`, `buffett_momentum_lab.py`). This script tests
the other lever: not *which* equities, but *when* and *how much*.

The rule (Gayed's "Leverage for the Long Run" construction):

  * At each month end, compare the index close to its 10-month moving average.
  * Above: hold the index at leverage L (daily-rebalanced, as a leveraged ETF
    actually behaves — so volatility decay is real, not assumed away).
  * Below: hold T-bills.

Why the trend filter makes leverage viable instead of ruinous: leverage's
failure mode is compounding through a deep drawdown (2x QQQ lost ~80% in
2022 peak-to-trough). Deep drawdowns happen in extended high-volatility
downtrends, which is precisely the state a 10-month filter steps out of. The
cost is whipsaw in sideways markets, which is charged here via trading in and
out at real T-bill + spread financing.

Honesty notes:

  * Financing: leveraged exposure pays (L-1) x (T-bill + `--spread`) daily,
    plus `--etf-er` expense ratio on the whole position — matching how a 2x
    fund is priced. Cash earns the T-bill rate.
  * Dividends: QQQ/SPY adjusted closes are total-return from 1999. Before
    that the price-only index (^NDX from 1985, ^IXIC from 1971, ^GSPC) is
    used with a flat dividend yield added (`--pre-etf-div`, default 1.5%/yr
    which is conservative for the 1970s-80s when yields were 3-4%).
  * Signals use the month-end close and trade at that close; using the same
    bar for signal and fill is standard for monthly systems and moves results
    by basis points, not the story.
  * No survivorship issue: the index series includes every crash it lived.

Run:

    python scripts/trend_leverage_backtest.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402


def _cache_dir() -> str:
    path = os.path.join(config.get_app_data_dir(), config.CACHE_DIR, "backtest")
    os.makedirs(path, exist_ok=True)
    return path


def load_daily(refresh: bool = False) -> pd.DataFrame:
    """Daily adjusted closes for the ETFs, price indices and the T-bill yield."""
    path = os.path.join(_cache_dir(), "daily_index_panel.pkl")
    if not refresh and os.path.exists(path):
        return pd.read_pickle(path)

    import yfinance as yf

    symbols = ["QQQ", "SPY", "^NDX", "^IXIC", "^GSPC", "^IRX"]
    data = yf.download(symbols, start="1965-01-01", auto_adjust=False, progress=False)
    frame = data["Adj Close"].copy()
    frame.to_pickle(path)
    return frame


def total_return_series(panel: pd.DataFrame, etf: str, index: str, pre_etf_div: float) -> pd.Series:
    """
    ETF adjusted close (dividends in) spliced onto the pre-ETF price index.

    The index segment gets a flat dividend yield added per trading day, so the
    early decades are total-return-ish rather than systematically understated.
    """
    etf_series = panel[etf].dropna()
    index_series = panel[index].dropna()
    start = etf_series.index[0]
    early = index_series.loc[: start - pd.Timedelta(days=1)]
    if early.empty:
        return etf_series

    daily_div = (1.0 + pre_etf_div) ** (1 / 252) - 1.0
    early_returns = early.pct_change().fillna(0.0) + daily_div
    early_tr = (1.0 + early_returns).cumprod()
    # Scale so the spliced series is continuous at the ETF's first close.
    early_tr = early_tr / early_tr.iloc[-1] * etf_series.iloc[0]
    # One overlapping point keeps pct_change continuous across the seam.
    return pd.concat([early_tr.iloc[:-1], etf_series])


def simulate(
    total_return: pd.Series,
    tbill_annual: pd.Series,
    leverage: float,
    sma_months: int = 10,
    spread: float = 0.01,
    etf_er: float = 0.0095,
    always_in: bool = False,
) -> Tuple[pd.Series, Dict[str, float]]:
    """
    Daily equity curve for the monthly-signal trend rule at a given leverage.

    Returns the curve and bookkeeping (time in market, number of round trips).
    """
    prices = total_return.dropna()
    returns = prices.pct_change().fillna(0.0)
    tbill_daily = (tbill_annual.reindex(prices.index).ffill().fillna(0.0) / 100.0) / 252.0

    month_ends = prices.groupby(prices.index.to_period("M")).tail(1).index
    sma = prices.loc[month_ends].rolling(sma_months).mean()
    signal_by_month = (prices.loc[month_ends] >= sma).astype(float)
    if always_in:
        signal_by_month[:] = 1.0
    # The signal decided at month end governs the following month.
    signal = signal_by_month.reindex(prices.index).shift(1).ffill().fillna(0.0)

    financing = (leverage - 1.0) * (tbill_daily + spread / 252.0) if leverage > 1 else 0.0
    er_daily = etf_er / 252.0 if leverage > 1 else 0.0
    strategy_returns = np.where(
        signal > 0,
        leverage * returns - financing - er_daily,
        tbill_daily,
    )
    curve = pd.Series((1.0 + strategy_returns).cumprod(), index=prices.index)

    switches = int(signal_by_month.diff().abs().sum() / 2)
    info = {
        "time_in_market": float(signal.mean()),
        "round_trips": switches,
        "start": str(prices.index[sma_months * 21].date()),
    }
    # Drop the SMA warm-up so early months without a signal don't count as cash.
    warm = month_ends[sma_months]
    return curve.loc[warm:] / curve.loc[warm:].iloc[0], info


def stats(curve: pd.Series, start: Optional[str] = None, end: Optional[str] = None) -> Dict[str, float]:
    window = curve.loc[start:end].dropna()
    if len(window) < 252:
        return {k: float("nan") for k in ("cagr", "volatility", "max_drawdown", "sharpe", "years")}
    window = window / window.iloc[0]
    years = (window.index[-1] - window.index[0]).days / 365.25
    cagr = float(window.iloc[-1] ** (1.0 / years) - 1.0)
    daily = window.pct_change().dropna()
    volatility = float(daily.std() * np.sqrt(252))
    drawdown = float((window / window.cummax() - 1.0).min())
    return {
        "cagr": cagr,
        "volatility": volatility,
        "max_drawdown": drawdown,
        "sharpe": cagr / volatility if volatility else float("nan"),
        "years": years,
    }


def annual_returns(curve: pd.Series) -> Dict[int, float]:
    yearly = curve.groupby(curve.index.year).last()
    shifted = yearly.shift(1)
    out = (yearly / shifted - 1.0).dropna()
    return {int(k): float(v) for k, v in out.items()}


def report_window(name: str, curves: Dict[str, pd.Series], start: str, end: Optional[str]) -> pd.DataFrame:
    rows = {}
    for label, curve in curves.items():
        s = stats(curve, start, end)
        rows[label] = {
            "cagr %": s["cagr"] * 100,
            "vol %": s["volatility"] * 100,
            "max_dd %": s["max_drawdown"] * 100,
            "sharpe": s["sharpe"],
            "years": s["years"],
        }
    frame = pd.DataFrame(rows).T.round(2)
    print(f"\n{name}")
    print(frame.to_string())
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sma", type=int, default=10, help="signal moving average, months")
    parser.add_argument("--spread", type=float, default=0.01, help="financing spread over T-bills")
    parser.add_argument("--etf-er", type=float, default=0.0095, help="leveraged ETF expense ratio")
    parser.add_argument("--pre-etf-div", type=float, default=0.015, help="dividend yield added to pre-ETF index years")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    panel = load_daily(refresh=args.refresh)
    tbill = panel["^IRX"]

    nasdaq = total_return_series(panel, "QQQ", "^IXIC", args.pre_etf_div)
    sp500 = total_return_series(panel, "SPY", "^GSPC", args.pre_etf_div)

    configurations = [
        ("NASDAQ buy&hold", nasdaq, 1.0, True),
        ("NASDAQ 1x trend", nasdaq, 1.0, False),
        ("NASDAQ 1.5x trend", nasdaq, 1.5, False),
        ("NASDAQ 2x trend", nasdaq, 2.0, False),
        ("NASDAQ 2x buy&hold", nasdaq, 2.0, True),
        ("S&P buy&hold", sp500, 1.0, True),
        ("S&P 1x trend", sp500, 1.0, False),
        ("S&P 2x trend", sp500, 2.0, False),
    ]

    curves: Dict[str, pd.Series] = {}
    infos: Dict[str, Dict[str, float]] = {}
    for label, series, leverage, always_in in configurations:
        curve, info = simulate(
            series,
            tbill,
            leverage,
            sma_months=args.sma,
            spread=args.spread,
            etf_er=args.etf_er,
            always_in=always_in,
        )
        curves[label] = curve
        infos[label] = info

    print(f"Signal: {args.sma}-month SMA, monthly close. Financing: T-bill + "
          f"{args.spread:.0%}, ER {args.etf_er:.2%} on levered configs.")
    print(f"NASDAQ series: ^IXIC from {nasdaq.index[0].date()} (+{args.pre_etf_div:.1%}/yr dividends), QQQ TR from 1999.")

    report_window("FULL HISTORY (1972-2025)", curves, "1972-01-01", "2025-12-31")
    report_window("MODERN / QQQ-ERA (1999-2025) — includes dot-com crash", curves, "1999-03-10", "2025-12-31")
    report_window("BACKTEST WINDOW SHARED WITH THE STOCK STRATEGY (2013-2025)", curves, "2012-12-31", "2025-12-31")
    report_window("TRAIN (2013-2019)", curves, "2012-12-31", "2019-12-31")
    report_window("TEST (2020-2025)", curves, "2019-12-31", "2025-12-31")
    report_window("USER HISTORY WINDOW (2003-2025)", curves, "2002-12-31", "2025-12-31")

    print("\nTrade accounting (full history)")
    print(pd.DataFrame(infos).T.to_string())

    print("\nWorst calendar years, NASDAQ 2x trend vs buy&hold")
    trend_annual = pd.Series(annual_returns(curves["NASDAQ 2x trend"]))
    hold_annual = pd.Series(annual_returns(curves["NASDAQ buy&hold"]))
    both = pd.DataFrame({"2x trend": trend_annual, "1x buy&hold": hold_annual}).dropna()
    print((both.sort_values("1x buy&hold").head(12) * 100).round(1).to_string())

    if args.out:
        payload = {
            "curves": {
                label: {str(i.date()): float(v) for i, v in curve.resample("ME").last().items()}
                for label, curve in curves.items()
            },
            "annual_returns": {label: annual_returns(curve) for label, curve in curves.items()},
            "info": infos,
        }
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nWrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
