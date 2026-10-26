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

**It is read for more than one market.** `MARKET_SIGNAL_INDICES` lists what the
dashboard panel shows — the S&P 500 and the NASDAQ 100, broadest first. One
index alone reads like a verdict on "the market"; two that regularly disagree
show the reader how much of the answer depends on which index was chosen. Each
reading is computed independently by the same rule and carries its own symbol
and display name, so the set can grow without a client changing.

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


import investa_core
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta  # noqa: F401 -- read by the Rust port at call time
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ai_review_scores import attach_ai_scores  # noqa: F401 -- read by the Rust port at call time
from buffett_rank import DEFAULT_AI_WEIGHT, blend_scores  # noqa: F401 -- read by the Rust port at call time
from utils_time import get_est_today  # noqa: F401 -- read by the Rust port at call time

# How much history to pull for the signal. A 10-month average needs 10 completed
# months plus the running one; the extra year makes the returned history chart
# useful and costs nothing (the market-data layer caches by symbol and range).
_SIGNAL_LOOKBACK_DAYS = 365 * 3

# The market-trend indicator's default parameters. Informational only — no
# strategy acts on it (see the module docstring).
MARKET_SIGNAL_SYMBOL = "SPY"
MARKET_SIGNAL_SMA_MONTHS = 10

# The markets the trend panel reads, broadest first. The panel shows both
# because one index is not "the market": the S&P and the NASDAQ disagree often
# enough that a single reading invites the viewer to generalise from whichever
# one happened to be wired up. Two readings side by side make the
# disagreement — and therefore the limits of the indicator — visible.
#
# Both legs are ETFs rather than the raw indices (^GSPC/^IXIC) on purpose: the
# two averages are then built from the same kind of series, on the same data
# path, so a crossing in one is comparable to a crossing in the other. This
# list is the panel's contract; the clients read the pair from here (mirrored in
# `web_app/components/dashboard/MarketTrendPanel.tsx` and
# `macos_app/Investa/Features/Strategies/MarketTrendPanel.swift`).
MARKET_SIGNAL_INDICES: tuple[tuple[str, str], ...] = (
    ("SPY", "S&P 500"),
    ("QQQ", "NASDAQ 100"),
)

# Display names for the symbols a signal can be asked for. Shipped in the
# payload so every client names an index identically instead of each keeping its
# own map; an unknown symbol falls back to the ticker rather than guessing.
_SIGNAL_DISPLAY_NAMES = {
    **{symbol: name for symbol, name in MARKET_SIGNAL_INDICES},
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ Composite",
    "^NDX": "NASDAQ 100",
    "DIA": "Dow Jones",
    "^DJI": "Dow Jones",
    "IWM": "Russell 2000",
    "^RUT": "Russell 2000",
}

# The signal's dates — the month-end that set it, the next check — are US market
# dates, so they are reckoned on the US market clock and the zone ships with
# them (see the market-local convention in CLAUDE.md). Investa's server runs on
# a Bangkok clock, up to a day ahead of New York: with `date.today()` the panel
# would call a month "completed" while its final session had yet to trade.
MARKET_SIGNAL_TIMEZONE = "America/New_York"


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
    # Optional minimum market cap filter (e.g. 10_000_000_000 for $10B+ large-cap).
    min_market_cap: Optional[float] = None
    # Share of the final score given to the AI review, on top of the
    # quality/value blend. A request may override it (`build_allocation`). Not
    # part of any backtest below: the reviews are not point-in-time.
    ai_weight: float = DEFAULT_AI_WEIGHT

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
            "January. The default: balances durable quality compounders and margin "
            "of safety while preventing single-industry concentration."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=20, max_per_sector=3),
        backtest={
            "window": "2013-2025",
            "cagr": 16.9,
            "volatility": 15.8,
            "max_drawdown": -20.2,
            "sharpe": 1.07,
            "train_cagr": 20.7,
            "test_cagr": 12.4,
        },
        risks=(
            "Fully invested at all times — there is no defensive mode, so a "
            "market-wide fall is taken in full.",
            "Lower return in the held-out test window (12.4%) than in the "
            "training window (20.7%).",
            "The universe is built from today's listing files, so companies "
            "that were delisted never appear: the backtest is "
            "survivorship-flattered by an unknown amount.",
        ),
    ),
    "quality_15": Strategy(
        id="quality_15",
        name="Buffett Quality 15",
        summary=(
            "A tighter, higher-conviction book: the top 15 rather than 20, capped "
            "at three per industry. Concentrates capital in the highest-scoring "
            "quality compounders with lower drawdown."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=15, max_per_sector=3),
        backtest={
            "window": "2013-2025",
            "cagr": 17.1,
            "volatility": 16.2,
            "max_drawdown": -19.1,
            "sharpe": 1.05,
            "train_cagr": 20.1,
            "test_cagr": 13.7,
        },
        risks=(
            "Single-name risk is a third higher than the twenty-name book: one "
            "position is 6.7% of capital rather than 5.0%.",
            "Concentrated holdings can experience sharper idiosyncratic volatility "
            "during single-company earnings misses.",
            "Fully invested at all times.",
        ),
    ),
    "quality_20_uncapped": Strategy(
        id="quality_20_uncapped",
        name="Buffett Quality 20 (uncapped)",
        summary=(
            "The same twenty-name rule with no industry cap — the ranking's raw "
            "top 20. Highest overall backtested CAGR, at the expense of "
            "concentration in whichever industries score highest."
        ),
        ranking=RankingSleeve(quality_weight=0.80, top_n=20, max_per_sector=None),
        backtest={
            "window": "2013-2025",
            "cagr": 17.9,
            "volatility": 16.5,
            "max_drawdown": -19.8,
            "sharpe": 1.08,
            "train_cagr": 21.0,
            "test_cagr": 13.9,
        },
        risks=(
            "Concentrated in whatever the ranking currently favours — often "
            "over a third of the book in two correlated financial industries.",
            "That cohort is also where the survivorship bias bites hardest.",
            "Fully invested at all times.",
        ),
    ),
    "quality_largecap_20": Strategy(
        id="quality_largecap_20",
        name="Buffett Large-Cap Leaders ($10B+)",
        summary=(
            "The top 20 large-cap companies with a minimum market cap of $10B, "
            "capped at three per industry. Prioritises liquid, established market "
            "leaders with institutional-grade balance sheets and low transaction friction."
        ),
        ranking=RankingSleeve(
            quality_weight=0.80,
            top_n=20,
            max_per_sector=3,
            min_market_cap=10_000_000_000.0,
        ),
        backtest={
            "window": "2013-2025",
            "cagr": 14.1,
            "volatility": 15.2,
            "max_drawdown": -19.2,
            "sharpe": 0.93,
            "train_cagr": 15.3,
            "test_cagr": 12.1,
        },
        risks=(
            "Excludes small- and mid-cap compounders, giving up some potential "
            "growth for liquidity and size.",
            "Large caps may have lower long-term ceiling growth rates as market "
            "saturation increases.",
            "Fully invested at all times.",
        ),
    ),
    "quality_value_balanced": Strategy(
        id="quality_value_balanced",
        name="Buffett Quality-Value Balanced (60/40)",
        summary=(
            "A 60/40 quality-to-value blend: places greater emphasis on free "
            "cash flow and earnings yield multiples while maintaining "
            "fundamental quality filters."
        ),
        ranking=RankingSleeve(quality_weight=0.60, top_n=20, max_per_sector=3),
        backtest={
            "window": "2013-2025",
            "cagr": 15.2,
            "volatility": 16.4,
            "max_drawdown": -23.5,
            "sharpe": 0.91,
            "train_cagr": 19.0,
            "test_cagr": 10.8,
        },
        risks=(
            "Higher value weight increases exposure to cyclical businesses "
            "undergoing temporary or secular challenges.",
            "Value multiples can remain depressed for extended periods before "
            "mean-reverting.",
            "Fully invested at all times.",
        ),
    ),
    "quality_pure": Strategy(
        id="quality_pure",
        name="Buffett Quality (price-blind)",
        summary=(
            "Ranks on business quality alone, ignoring valuation multiples "
            "entirely. Buys extraordinary businesses regardless of their current "
            "market price."
        ),
        ranking=RankingSleeve(quality_weight=1.00, top_n=20, max_per_sector=3),
        backtest={
            "window": "2013-2025",
            "cagr": 15.6,
            "volatility": 15.3,
            "max_drawdown": -21.3,
            "sharpe": 1.02,
            "train_cagr": 20.4,
            "test_cagr": 10.0,
        },
        risks=(
            "Ignoring price means the strategy can buy great companies at peak "
            "valuations.",
            "Lowest return in the out-of-sample test window (10.0%) as valuation "
            "multiples expanded and compressed.",
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
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_month_end_closes(prices)


def evaluate_trend_signal(
    prices: pd.Series,
    sma_months: int = MARKET_SIGNAL_SMA_MONTHS,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """
    The active and provisional readings of the moving-average rule.

    `prices` is a daily close series indexed by date. Returns None when there is
    not enough history to form the average — a partial average silently biased
    by its own short window is worse than an honest refusal.

    `today` decides which month is still running, so it is a *market* date, not
    the server's (see `MARKET_SIGNAL_TIMEZONE`).
    """
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_evaluate_trend_signal(prices, sma_months, today)


def _next_month_end(today: date) -> date:
    """Calendar month-end for `today`'s month (the next decision point)."""
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_next_month_end(today)


def load_signal_prices(
    symbol: str, today: Optional[date] = None
) -> Optional[pd.Series]:
    """Daily closes for the signal symbol, via the shared market-data provider."""
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_load_signal_prices(symbol, today)


def latest_closes(
    symbols: Sequence[str], today: Optional[date] = None
) -> Dict[str, float]:
    """
    Most recent close per symbol, for sizing the tradeable sleeve.

    Goes through the same daily-history path as the signal rather than the
    quote API: the quote path needs a user's symbol map and exclusion set,
    which an ETF ticker in a fixed rule has no business depending on.
    """
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_latest_closes(symbols, today)


def market_trend_signal(
    symbol: str = MARKET_SIGNAL_SYMBOL,
    sma_months: int = MARKET_SIGNAL_SMA_MONTHS,
    today: Optional[date] = None,
) -> Optional[Dict[str, Any]]:
    """
    The market-trend indicator: one index's close against its own moving average.

    Informational context, not an instruction — no strategy in this module acts
    on it (see the module docstring for the measurement that retired it). The
    symbol, its display name and the zone its dates were reckoned in are tagged
    onto the result rather than added by the route, so the object is
    self-describing wherever it is read — and so the panel that shows several of
    these side by side names each market the same way in every client.
    """
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_market_trend_signal(symbol, sma_months, today)


# --- turning a strategy into positions --------------------------------------
def _ranking_positions(
    sleeve: RankingSleeve,
    capital: float,
    today: Optional[date] = None,
    ai_weight: Optional[float] = None,
) -> Dict[str, Any]:
    """
    The top-N names from the stored ranking snapshot, equally weighted.

    Re-blends the stored quality and value scores rather than re-running the
    pipeline: the snapshot keeps the two components separately, so changing the
    blend is arithmetic on data already computed. This is the same path
    `scripts/buffett_live_picks.py` takes, so the app and the script cannot
    disagree — until the AI review is weighted in, which the script does not
    do (it reproduces the backtest, and the review cannot be backtested).

    The AI review is joined over the *whole* run before any filter, so its
    percentile means the same here as on the ranking page.

    **Membership comes from the snapshot; prices do not.** Which twenty
    companies to hold is the rule's output and must not drift with the tape.
    The share counts are arithmetic on today's price, so quoting them off a
    snapshot close means handing over an order that was wrong before it was
    placed — and progressively more wrong the older the snapshot is. Live
    quotes are fetched for the chosen names only (twenty symbols through the
    cached market-data path), with the snapshot close as the fallback so a
    quote outage degrades to the old behaviour rather than to no answer.
    """
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_ranking_positions(sleeve, capital, today, ai_weight)


def _optional_float(value: Any) -> Optional[float]:
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_optional_float(value)


def _apply_sector_cap(
    frame: pd.DataFrame, scores: pd.Series, sleeve: RankingSleeve
) -> pd.DataFrame:
    """Walk the ranked list, skipping a name once its industry is full."""
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_apply_sector_cap(frame, scores, sleeve)


# Major-group labels for the industries the ranking actually surfaces. Display
# only — the cap itself works on the numeric code.
_SIC_GROUPS = {
    "15": "Homebuilders",
    "20": "Food products",
    "23": "Apparel",
    "28": "Pharma / chemicals",
    "30": "Rubber & plastics",
    "34": "Fabricated metal",
    "35": "Industrial machinery",
    "36": "Electronics",
    "37": "Transport equipment",
    "38": "Instruments",
    "45": "Air transport",
    "48": "Communications",
    "50": "Wholesale (durables)",
    "51": "Wholesale (nondurables)",
    "52": "Building materials",
    "53": "General merchandise",
    "54": "Food retail",
    "55": "Auto dealers",
    "56": "Apparel retail",
    "57": "Furniture retail",
    "58": "Restaurants",
    "59": "Miscellaneous retail",
    "60": "Banks",
    "61": "Non-bank credit",
    "62": "Brokers & exchanges",
    "63": "Insurance carriers",
    "64": "Insurance agents",
    "65": "Real estate",
    "67": "Holding & investment",
    "70": "Hotels",
    "73": "Business services / software",
    "79": "Recreation",
    "80": "Health services",
    "87": "Consulting & engineering",
}

# Codes whose major group misleads on its own: 6798 is a REIT rather than a
# generic holding company, 6351 is mortgage insurance rather than insurance at
# large — exactly the distinction that matters for concentration.
_SIC_EXACT = {
    "6798": "REITs",
    "6351": "Mortgage insurance",
    "1531": "Homebuilders",
    "5500": "Auto dealers & services",
}


def _industry_label(sic: str) -> str:
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_industry_label(sic)


# How old the ranking snapshot may get before the allocation says so. The
# batch worker is meant to run daily, so a week means several runs in a row
# did not happen — the interesting case is not "slightly stale" but "the
# scheduler is dead and nobody noticed".
STALE_RANKING_DAYS = 7


def ranking_age_days(
    finished_at: Optional[str], today: Optional[date] = None
) -> Optional[int]:
    """Whole days between a run's completion and `today`, or None if unparseable."""
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_ranking_age_days(finished_at, today)


def build_allocation(
    strategy: Strategy,
    capital: float,
    today: Optional[date] = None,
    ai_weight: Optional[float] = None,
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

    `ai_weight` overrides the strategy's AI-review weight for this request;
    None keeps the strategy's own.
    """
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_build_allocation(strategy, capital, today, ai_weight)


def _sleeve_label(sleeve: RankingSleeve, ai_weight: float) -> str:
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_sleeve_label(sleeve, ai_weight)


# Shown beside a strategy while the AI review carries weight: the backtest
# figures beside it were measured without that term.
AI_BACKTEST_CAVEAT = (
    "The AI review's share of the score is not in the backtest: the reviews are "
    "written today, so there is no point-in-time version of them to test. The "
    "historical figures are for the quality/value ranking alone."
)


def strategy_payload(strategy: Strategy) -> Dict[str, Any]:
    """Serialisable description of a strategy, without touching market data."""
    # Ported to Rust: rust/investa_core/src/st.rs
    return investa_core.st_strategy_payload(strategy)
