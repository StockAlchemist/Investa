"""Rule-based strategies, and the market-trend indicator shown beside them.

Everything here is a *rule*, not a recommendation the app invents on the fly:
each strategy is a fixed recipe whose historical behaviour was measured in
`scripts/buffett_strategy_search.py` and documented in
`docs/quality_strategy_report.pdf`. The app's job is to say what the rule calls
for today and to keep saying it consistently — the returns in the report assume
the rule is followed mechanically, so a UI that invites improvisation would be
reporting one thing and enabling another.

**Every strategy holds individual common stock and nothing else.** Two
constraints set by the account owner, both structural rather than configurable:

  * **No leverage.** There is no leverage parameter anywhere, not even one
    defaulting to 1.0 — that would keep the borrowing path alive and one edit
    from being used.
  * **No index funds or ETFs.** A strategy's positions are the ranked companies
    themselves. There is no fund wrapper, no cash-proxy ETF, and no sleeve that
    could hold one.

`tests/test_strategies.py` asserts both against every registered strategy, so a
future addition cannot reintroduce either by accident.

**On the trend indicator.** The moving-average signal in this module is now
*informational only*. It was previously used to move a NASDAQ position in and
out of the market; with index funds excluded, holding the index is no longer
possible, and gating the *stock* book with it was measured and rejected —
13.0%/yr against 16.3% for simply holding the stocks, with a deeper drawdown
(-27.9% against -24.3%) rather than a shallower one — both figures measured on
the ranking as it stood before the DCF was removed, and not re-run since, since
the conclusion was not close. It helped in three of
thirteen years and hurt in six. The reason is a mismatch: the signal describes
the NASDAQ, while the book is twenty value-weighted businesses that correlate
with it at 0.46. Timing asset A with a signal about asset B is unreliable, and
the measurement says so. The indicator is therefore market context on the
dashboard, and no strategy acts on it.

**Its timing is still easy to misread**, which is why the payload keeps two
fields apart. The rule compares *month-end* closes with the average of the last
*N* month-ends, so on any day inside a month there are two different facts:

  * the **active** reading — decided at the last completed month-end;
  * the **provisional** reading — what the comparison would say if the current
    month ended today. It previews the *next* month's reading and is never the
    current one.

Both are returned, labelled, and the API never collapses them into one field.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

# How much history to pull for the signal. A 10-month average needs 10 completed
# months plus the running one; the extra year makes the returned history chart
# useful and costs nothing (the market-data layer caches by symbol and range).
_SIGNAL_LOOKBACK_DAYS = 365 * 3

# The market-trend indicator's default parameters. Informational only — no
# strategy acts on it (see the module docstring).
MARKET_SIGNAL_SYMBOL = "QQQ"
MARKET_SIGNAL_SMA_MONTHS = 10


@dataclass(frozen=True)
class RankingSleeve:
    """
    Top-N of the Buffett ranking at a given quality/value blend.

    The only sleeve kind there is. Its positions are individual companies, so a
    strategy built from it cannot hold a fund or borrow — the constraints are
    enforced by there being no other sleeve type to reach for.
    """

    quality_weight: float
    top_n: int
    # Industry cap, applied greedily down the ranked list. The uncapped rule
    # scores companies one at a time with no notion of correlation, so it will
    # fill a fifth of the book with mortgage insurers — the exact cohort the
    # ranking's survivorship bias flatters most.
    max_per_sector: Optional[int] = 3
    sector_digits: int = 2

    @property
    def kind(self) -> str:
        return "ranking"


@dataclass(frozen=True)
class Strategy:
    id: str
    name: str
    summary: str
    ranking: RankingSleeve
    # Measured on the 2013-2025 window. Carried here so a client can show the
    # rule's record beside what it is asking the user to do, rather than the
    # number living only in a PDF.
    backtest: Dict[str, Any] = field(default_factory=dict)
    risks: Sequence[str] = ()

    @property
    def sleeves(self) -> Dict[str, float]:
        """One sleeve, all of the capital. Kept as a mapping so the allocation
        payload has a stable shape if a second stock sleeve is ever added."""
        return {"ranking": 1.0}


# Every figure below is measured, not asserted: each comes from
# `buffett_strategy_search.py` run point-in-time over 2013-2025 with the same
# parameters the app applies, split into a 2013-2019 training window and a
# 2020-2025 window held out from the search.
_STRATEGIES: Dict[str, Strategy] = {
    "quality_20": Strategy(
        id="quality_20",
        name="Buffett Quality 20",
        summary=(
            "The top 20 companies by the 80/20 quality-value composite, equally "
            "weighted, capped at three names per industry, rebalanced each "
            "January. The default: it gives up a little return against the "
            "uncapped list to avoid putting a fifth of the book on one "
            "balance-sheet risk."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=20, max_per_sector=3),
        backtest={
            "window": "2013-2025", "cagr": 17.4, "volatility": 16.5,
            "max_drawdown": -21.5, "sharpe": 1.05,
            "train_cagr": 21.9, "test_cagr": 12.1,
        },
        risks=(
            "Fully invested at all times — there is no defensive mode, so a "
            "market-wide fall is taken in full.",
            "Much weaker in the held-out window (12.1%) than in the training "
            "one (21.9%).",
            "The universe is built from today's listing files, so companies "
            "that were delisted never appear: the backtest is "
            "survivorship-flattered by an unknown amount.",
        ),
    ),
    "quality_20_uncapped": Strategy(
        id="quality_20_uncapped",
        name="Buffett Quality 20 (uncapped)",
        summary=(
            "The same twenty-name rule with no industry cap — the ranking's raw "
            "top 20. Slightly higher backtested return, at the cost of "
            "concentration: today's list would hold six banks and three "
            "mortgage insurers, over half the book in lending risk."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=20, max_per_sector=None),
        backtest={
            "window": "2013-2025", "cagr": 17.6, "volatility": 16.3,
            "max_drawdown": -20.0, "sharpe": 1.08,
            "train_cagr": 21.4, "test_cagr": 13.0,
        },
        risks=(
            "Concentrated in whatever the ranking currently favours — often "
            "40% of the book in two correlated financial industries.",
            "That cohort is also where the survivorship bias bites hardest: "
            "post-2009 mortgage insurers are the survivors of a sector that "
            "was largely wiped out.",
            "Fully invested at all times.",
        ),
    ),
    "quality_15": Strategy(
        id="quality_15",
        name="Buffett Quality 15",
        summary=(
            "A tighter book: the top 15 rather than 20, capped at three per "
            "industry. On the measured window it led the four on every "
            "statistic — return, Sharpe, worst case and held-out return — "
            "which is a strong result from a small sample and fifteen names."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=15, max_per_sector=3),
        backtest={
            "window": "2013-2025", "cagr": 18.9, "volatility": 17.0,
            "max_drawdown": -18.8, "sharpe": 1.11,
            "train_cagr": 21.3, "test_cagr": 15.9,
        },
        risks=(
            "Single-name risk is a third higher than the twenty-name book: one "
            "position is 6.7% of capital rather than 5%.",
            "It leads the four on the measured window, but fifteen names over "
            "thirteen years is a thin sample — the gap over the twenty-name "
            "book is well inside what luck can produce.",
            "Fully invested at all times.",
        ),
    ),
    "quality_pure": Strategy(
        id="quality_pure",
        name="Buffett Quality (price-blind)",
        summary=(
            "Ranks on business quality alone, ignoring valuation entirely. "
            "The lowest return of the four and the calmest ride — the least "
            "volatile by some way, though no longer the shallowest drawdown."
        ),
        ranking=RankingSleeve(quality_weight=1.00, top_n=20, max_per_sector=3),
        backtest={
            "window": "2013-2025", "cagr": 15.9, "volatility": 14.8,
            "max_drawdown": -20.3, "sharpe": 1.07,
            "train_cagr": 20.0, "test_cagr": 11.0,
        },
        risks=(
            "Ignoring price is the weakest link: this is the only variant whose "
            "quality weight sits outside the range that improved results "
            "consistently across both windows, and its held-out return (11.0%) "
            "is the lowest of the four.",
            "Buying excellent businesses at any multiple works until it "
            "abruptly does not.",
            "Fully invested at all times.",
        ),
    ),
}

DEFAULT_STRATEGY_ID = "quality_20"


def list_strategies() -> List[Strategy]:
    return list(_STRATEGIES.values())


def get_strategy(strategy_id: str) -> Optional[Strategy]:
    return _STRATEGIES.get(strategy_id)


# --- the trend signal -------------------------------------------------------


def month_end_closes(prices: pd.Series) -> pd.Series:
    """
    Last available close of each calendar month.

    Resampling to a calendar month-end would invent a row for a month with no
    trading and stamp every value on a date the market may have been shut.
    Grouping by period and taking the last actual observation keeps both the
    value and its real date.
    """
    clean = prices.dropna()
    if clean.empty:
        return clean
    # Period arithmetic drops any timezone, and pandas warns about it. Nothing
    # here needs the offset — a close belongs to the month its exchange dated
    # it — so drop it explicitly rather than let the conversion do it noisily.
    index = clean.index
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        clean = clean.copy()
        clean.index = index.tz_localize(None)
    return clean.groupby(clean.index.to_period("M")).tail(1)


def evaluate_trend_signal(
    prices: pd.Series, sma_months: int = MARKET_SIGNAL_SMA_MONTHS, today: Optional[date] = None
) -> Optional[Dict[str, Any]]:
    """
    The active and provisional readings of the moving-average rule.

    `prices` is a daily close series indexed by date. Returns None when there is
    not enough history to form the average — a partial average silently biased
    by its own short window is worse than an honest refusal.
    """
    monthly = month_end_closes(prices)
    if monthly.empty:
        return None

    today = today or date.today()
    current_period = pd.Period(today, freq="M")

    # Split the running month off: its last close is *not* a month-end yet, and
    # treating it as one would let a mid-month price set the active signal.
    periods = monthly.index.to_period("M")
    completed = monthly[periods < current_period]
    if len(completed) < sma_months:
        return None

    window = completed.iloc[-sma_months:]
    sma = float(window.mean())
    decision_close = float(completed.iloc[-1])
    decision_stamp = completed.index[-1]
    state = "in" if decision_close >= sma else "out"

    # Provisional: replace the oldest month of the window with the running
    # month's latest close, which is what the comparison will use if the month
    # ended right now.
    latest_close = float(prices.dropna().iloc[-1])
    latest_stamp = prices.dropna().index[-1]
    provisional_window = list(completed.iloc[-(sma_months - 1):]) + [latest_close]
    provisional_sma = float(sum(provisional_window) / len(provisional_window))
    provisional_state = "in" if latest_close >= provisional_sma else "out"

    # The close at which the next decision flips. Solving
    #   p = (sum(previous sma_months-1 month-ends) + p) / sma_months
    # for p gives the mean of the *other* months in the window — the price at
    # which today's close equals the average that includes it.
    previous = list(completed.iloc[-(sma_months - 1):])
    flip_close = float(sum(previous) / len(previous)) if previous else sma
    distance_pct = (latest_close / flip_close - 1.0) * 100.0 if flip_close else None

    history = [
        {
            "date": str(stamp.date() if hasattr(stamp, "date") else stamp),
            "close": float(close),
            "sma": (
                float(completed.iloc[max(0, i - sma_months + 1): i + 1].mean())
                if i + 1 >= sma_months
                else None
            ),
        }
        for i, (stamp, close) in enumerate(completed.items())
    ][-36:]

    return {
        "state": state,
        "sma_months": sma_months,
        # The month-end that set the active signal, and the month it governs.
        "decision_date": str(decision_stamp.date() if hasattr(decision_stamp, "date") else decision_stamp),
        "decision_close": decision_close,
        "sma": sma,
        "governs_month": str(current_period),
        "provisional_state": provisional_state,
        "provisional_sma": provisional_sma,
        "latest_close": latest_close,
        "latest_date": str(latest_stamp.date() if hasattr(latest_stamp, "date") else latest_stamp),
        # What the next month-end close has to do to change the answer.
        "flip_close": flip_close,
        "distance_pct": distance_pct,
        "would_flip": provisional_state != state,
        "next_decision_date": str(_next_month_end(today)),
        "history": history,
    }


def _next_month_end(today: date) -> date:
    """Calendar month-end for `today`'s month (the next decision point)."""
    if today.month == 12:
        return date(today.year, 12, 31)
    return date(today.year, today.month + 1, 1) - timedelta(days=1)


def load_signal_prices(symbol: str, today: Optional[date] = None) -> Optional[pd.Series]:
    """Daily closes for the signal symbol, via the shared market-data provider."""
    from server.route_utils import get_mdp

    today = today or date.today()
    start = today - timedelta(days=_SIGNAL_LOOKBACK_DAYS)
    try:
        history, _ = get_mdp().get_historical_data([symbol], start, today)
    except Exception as exc:
        logging.error(f"Strategies: price fetch failed for {symbol}: {exc}")
        return None

    frame = history.get(symbol)
    if frame is None or frame.empty or "price" not in frame.columns:
        return None
    series = frame["price"].dropna()
    return series if not series.empty else None


def latest_closes(symbols: Sequence[str], today: Optional[date] = None) -> Dict[str, float]:
    """
    Most recent close per symbol, for sizing the tradeable sleeve.

    Goes through the same daily-history path as the signal rather than the
    quote API: the quote path needs a user's symbol map and exclusion set,
    which an ETF ticker in a fixed rule has no business depending on.
    """
    from server.route_utils import get_mdp

    today = today or date.today()
    wanted = sorted({s for s in symbols if s})
    if not wanted:
        return {}
    try:
        history, _ = get_mdp().get_historical_data([*wanted], today - timedelta(days=14), today)
    except Exception as exc:
        logging.warning(f"Strategies: quote fetch failed for {wanted}: {exc}")
        return {}

    prices: Dict[str, float] = {}
    for symbol in wanted:
        frame = history.get(symbol)
        if frame is None or frame.empty or "price" not in frame.columns:
            continue
        series = frame["price"].dropna()
        if not series.empty:
            prices[symbol] = float(series.iloc[-1])
    return prices


def market_trend_signal(
    symbol: str = MARKET_SIGNAL_SYMBOL,
    sma_months: int = MARKET_SIGNAL_SMA_MONTHS,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """
    The market-trend indicator: one index's close against its own moving average.

    Informational context, not an instruction — no strategy in this module acts
    on it (see the module docstring for the measurement that retired it). The
    symbol is tagged onto the result rather than added by the route, so the
    object is self-describing wherever it is read.
    """
    prices = load_signal_prices(symbol, today)
    if prices is None:
        return None
    signal = evaluate_trend_signal(prices, sma_months, today)
    if signal is None:
        return None
    signal["signal_symbol"] = symbol
    # Stated in the payload so a client cannot present the indicator as a
    # trading rule the backend is standing behind.
    signal["advisory_only"] = True
    return signal


# --- turning a strategy into positions --------------------------------------


def _ranking_positions(
    sleeve: RankingSleeve, capital: float, today: Optional[date] = None
) -> Dict[str, Any]:
    """
    The top-N names from the stored ranking snapshot, equally weighted.

    Re-blends the stored quality and value scores rather than re-running the
    pipeline: the snapshot keeps the two components separately, so changing the
    blend is arithmetic on data already computed. This is the same path
    `scripts/buffett_live_picks.py` takes, so the app and the script cannot
    disagree.

    **Membership comes from the snapshot; prices do not.** Which twenty
    companies to hold is the rule's output and must not drift with the tape.
    The share counts are arithmetic on today's price, so quoting them off a
    snapshot close means handing over an order that was wrong before it was
    placed — and progressively more wrong the older the snapshot is. Live
    quotes are fetched for the chosen names only (twenty symbols through the
    cached market-data path), with the snapshot close as the fallback so a
    quote outage degrades to the old behaviour rather than to no answer.
    """
    from buffett_store import get_store

    run = get_store().get_run()
    if not run:
        return {"positions": [], "run": None, "error": "No completed ranking run yet"}

    frame = get_store().get_scores_frame(run["run_id"])
    if frame is None or frame.empty:
        return {"positions": [], "run": run, "error": "Ranking run has no scores"}

    quality = pd.to_numeric(frame["quality_score"], errors="coerce")
    value = pd.to_numeric(frame["value_score"], errors="coerce")
    confidence = pd.to_numeric(frame.get("confidence"), errors="coerce").fillna(1.0)
    # A company with no value score keeps its quality score rather than being
    # dropped — the same fallback `buffett_rank.combine` applies.
    blended = quality.where(
        value.isna(), quality * sleeve.quality_weight + value * (1.0 - sleeve.quality_weight)
    ) * confidence

    picked = _apply_sector_cap(frame, blended, sleeve)
    per_position = capital / sleeve.top_n if sleeve.top_n else 0.0

    symbols = [str(row["symbol"]) for _, row in picked.iterrows()]
    live = latest_closes(symbols, today)

    positions = []
    live_count = 0
    for _, row in picked.iterrows():
        symbol = str(row["symbol"])
        snapshot_price = row.get("price")
        snapshot_price = (
            float(snapshot_price)
            if snapshot_price is not None and not pd.isna(snapshot_price)
            else None
        )
        quoted = live.get(symbol)
        price = quoted if (quoted and quoted > 0) else snapshot_price
        if quoted and quoted > 0:
            live_count += 1
        shares = math.floor(per_position / price) if price and price > 0 else None
        positions.append({
            "symbol": symbol,
            "name": row.get("name"),
            "role": "stock",
            "weight": 1.0 / sleeve.top_n if sleeve.top_n else 0.0,
            "amount": per_position,
            "price": price,
            "shares": shares,
            "cost": (shares * price) if (shares and price) else None,
            "score": float(row["_score"]),
            "industry": row.get("_industry"),
        })

    # Stated rather than implied: a client showing share counts should be able
    # to say whether they came from today's market or from a stored close.
    if not positions or live_count == 0:
        price_source = "snapshot"
    elif live_count == len(positions):
        price_source = "live"
    else:
        price_source = "mixed"

    return {
        "positions": positions,
        "run": run,
        "error": None,
        "price_source": price_source,
    }


def _apply_sector_cap(frame: pd.DataFrame, scores: pd.Series, sleeve: RankingSleeve) -> pd.DataFrame:
    """Walk the ranked list, skipping a name once its industry is full."""
    import edgar_sic

    order = scores.sort_values(ascending=False).index
    try:
        sic_map = edgar_sic.get_sic_map()
    except Exception as exc:
        logging.warning(f"Strategies: SIC map unavailable, industry cap disabled: {exc}")
        sic_map = {}

    codes: Dict[Any, str] = {}
    for index, cik in frame["cik"].items():
        try:
            key = str(int(float(cik))).zfill(10)
        except (TypeError, ValueError):
            continue
        sic = sic_map.get(key)
        if sic is not None:
            codes[index] = str(int(sic)).zfill(4)

    counts: Dict[str, int] = {}
    picked: List[Any] = []
    for index in order:
        if pd.isna(scores.get(index)):
            continue
        full = codes.get(index, "")
        # An unclassified company is its own industry rather than lumped with
        # every other unclassified name, which would cap them against each other.
        group = full[: sleeve.sector_digits] if full else f"?{index}"
        if sleeve.max_per_sector and counts.get(group, 0) >= sleeve.max_per_sector:
            continue
        counts[group] = counts.get(group, 0) + 1
        picked.append(index)
        if len(picked) == sleeve.top_n:
            break

    result = frame.loc[picked].copy()
    result["_score"] = scores.loc[picked]
    result["_industry"] = [_industry_label(codes.get(i, "")) for i in picked]
    return result


# Major-group labels for the industries the ranking actually surfaces. Display
# only — the cap itself works on the numeric code.
_SIC_GROUPS = {
    "15": "Homebuilders", "20": "Food products", "23": "Apparel",
    "28": "Pharma / chemicals", "30": "Rubber & plastics", "34": "Fabricated metal",
    "35": "Industrial machinery", "36": "Electronics", "37": "Transport equipment",
    "38": "Instruments", "45": "Air transport", "48": "Communications",
    "50": "Wholesale (durables)", "51": "Wholesale (nondurables)",
    "52": "Building materials", "53": "General merchandise", "54": "Food retail",
    "55": "Auto dealers", "56": "Apparel retail", "57": "Furniture retail",
    "58": "Restaurants", "59": "Miscellaneous retail", "60": "Banks",
    "61": "Non-bank credit", "62": "Brokers & exchanges", "63": "Insurance carriers",
    "64": "Insurance agents", "65": "Real estate", "67": "Holding & investment",
    "70": "Hotels", "73": "Business services / software", "79": "Recreation",
    "80": "Health services", "87": "Consulting & engineering",
}

# Codes whose major group misleads on its own: 6798 is a REIT rather than a
# generic holding company, 6351 is mortgage insurance rather than insurance at
# large — exactly the distinction that matters for concentration.
_SIC_EXACT = {
    "6798": "REITs", "6351": "Mortgage insurance", "1531": "Homebuilders",
    "5500": "Auto dealers & services",
}


def _industry_label(sic: str) -> str:
    if not sic:
        return "Unclassified"
    return _SIC_EXACT.get(sic) or _SIC_GROUPS.get(sic[:2], "Other")


# How old the ranking snapshot may get before the allocation says so. The
# batch worker is meant to run daily, so a week means several runs in a row
# did not happen — the interesting case is not "slightly stale" but "the
# scheduler is dead and nobody noticed".
STALE_RANKING_DAYS = 7


def ranking_age_days(finished_at: Optional[str], today: Optional[date] = None) -> Optional[int]:
    """Whole days between a run's completion and `today`, or None if unparseable."""
    if not finished_at:
        return None
    try:
        finished = datetime.fromisoformat(str(finished_at)).date()
    except (TypeError, ValueError):
        return None
    return max((today or date.today()) - finished, timedelta(0)).days


def build_allocation(
    strategy: Strategy, capital: float, today: Optional[date] = None
) -> Dict[str, Any]:
    """
    What the strategy says to hold today, sized to `capital`.

    Weights are the rule and are exact; share counts are arithmetic on a live
    quote (see `_ranking_positions`) and move with the market. One sleeve, all
    of the capital — the payload keeps the sleeve list so its shape is stable,
    not because a second kind exists.

    **A stale snapshot is reported, not hidden.** The ranking is produced by a
    batch worker; if that worker stops, every endpoint here keeps serving the
    last good run and looks entirely healthy. The one moment that matters is
    the January rebalance, which is precisely when trading a months-old list
    would do real damage — so the age travels in the payload and, past a
    threshold, as a warning the clients already surface.

    **A short book is reported too, and is not silently repriced.** If the
    ranking yields fewer names than the rule asks for, the shortfall is stated
    and the remaining capital is left visibly unallocated. The two alternatives
    are both worse: spreading the capital over the survivors changes the
    position weights away from the rule that was actually backtested, and
    saying nothing hands over a list that looks complete while a quarter of the
    money has nowhere to go. This has happened in practice — a truncated
    smoke-test run became the newest snapshot and the sleeve served five names
    against a twenty-name rule without a word.
    """
    warnings: List[str] = []
    built = _ranking_positions(strategy.ranking, capital, today)
    if built["error"]:
        warnings.append(built["error"])

    run = built["run"] or {}
    ranked_at = run.get("finished_at")
    age_days = ranking_age_days(ranked_at, today)
    if age_days is not None and age_days >= STALE_RANKING_DAYS:
        warnings.append(
            f"Ranking snapshot is {age_days} days old (last run {str(ranked_at)[:10]}). "
            "The daily worker may not be running — refresh with "
            "`python src/buffett_rank_worker.py` before acting on this list."
        )

    positions = built["positions"]
    wanted = strategy.ranking.top_n
    allocated = sum(p["amount"] for p in positions)
    if positions and len(positions) < wanted:
        warnings.append(
            f"The ranking produced only {len(positions)} of the {wanted} names this "
            f"rule calls for, so ${capital - allocated:,.0f} of ${capital:,.0f} is "
            "unallocated. Weights are deliberately left at the rule's "
            f"{1.0 / wanted:.1%} rather than spread over a shorter list. Check that "
            "the last ranking run covered the whole universe."
        )

    sleeves = [{
        "key": "ranking",
        "label": (
            f"Quality sleeve (top {strategy.ranking.top_n}, "
            f"{strategy.ranking.quality_weight:.0%} quality)"
        ),
        "weight": 1.0,
        "amount": capital,
        # What the rule asked for against what the ranking could supply, so a
        # client can show the gap without recounting the positions itself.
        "positions_requested": wanted,
        "positions_filled": len(positions),
        "amount_allocated": allocated,
        "positions": positions,
        "run_id": run.get("run_id"),
        "ranked_at": ranked_at,
        "price_source": built.get("price_source", "snapshot"),
    }]

    return {
        "strategy_id": strategy.id,
        "name": strategy.name,
        "capital": capital,
        "as_of": str(today or date.today()),
        "ranking_age_days": age_days,
        "ranking_is_stale": bool(age_days is not None and age_days >= STALE_RANKING_DAYS),
        "is_short": bool(positions and len(positions) < wanted),
        "sleeves": sleeves,
        "warnings": warnings,
    }


def strategy_payload(strategy: Strategy) -> Dict[str, Any]:
    """Serialisable description of a strategy, without touching market data."""
    return {
        "id": strategy.id,
        "name": strategy.name,
        "summary": strategy.summary,
        "sleeves": strategy.sleeves,
        "backtest": strategy.backtest,
        "risks": list(strategy.risks),
        "is_default": strategy.id == DEFAULT_STRATEGY_ID,
        "ranking": {
            "quality_weight": strategy.ranking.quality_weight,
            "top_n": strategy.ranking.top_n,
            "max_per_sector": strategy.ranking.max_per_sector,
            "sector_digits": strategy.ranking.sector_digits,
            "rebalance": "Each January",
        },
    }
