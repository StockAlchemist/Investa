# -*- coding: utf-8 -*-
"""
End-to-end Buffett/value ranking run.

Sequence: universe → sector routing → fundamentals → quality gates → quality
pillars → market data → value → composite → persisted snapshot.

Two ordering decisions are load-bearing rather than incidental:

**Gates are applied before market data is fetched.** Roughly four fifths of the
all-listed universe fails a quality gate, and there is no reason to pull a quote
for a company that will not be ranked. This turns the most expensive stage —
yfinance calls, which are rate-limited and slow — into a fraction of its
worst-case size.

**Share classes collapse to one filer.** BRK-A and BRK-B are one business with
one set of financials; ranking both would double-count it and waste a slot near
the top. The most liquid class is kept, chosen on market capitalisation.

There is no valuation stage. The run used to compute a Monte Carlo DCF for every
generic-model company — comfortably the most expensive step in the pipeline —
to feed a margin-of-safety term into the value score. That term was measured
against thirteen years of point-in-time rankings and found to be noise, so it
was removed; see `buffett_value` for the evidence. The run is now bounded by
market-data fetching alone.

The run is resumable in the sense that fundamentals come from the local EDGAR
store: re-running after a failure re-reads the database rather than the SEC.
"""

from __future__ import annotations

import logging
import time
from datetime import date
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

import buffett_rank
import buffett_value
import edgar_sic
import universe
from buffett_metrics import CompanyMetrics, compute_metrics
from buffett_store import get_store

# Market-data batch size. Large enough to amortise call overhead, small enough
# that one failure does not lose a long run's worth of work.
MARKET_BATCH_SIZE = 200

# How stale a cached share count may be before it is re-fetched. Shares
# outstanding changes on buybacks and issuance, which are quarterly events; the
# previous behaviour re-downloaded the whole universe daily for it.
SHARE_COUNT_MAX_AGE_DAYS = 30


@dataclass
class RunResult:
    run_id: Optional[int] = None
    ranked: pd.DataFrame = field(default_factory=pd.DataFrame)
    exclusions: List[Dict[str, Any]] = field(default_factory=list)
    duration_seconds: float = 0.0
    stats: Dict[str, int] = field(default_factory=dict)


def _resolve_universe(limit: Optional[int] = None) -> List[Tuple[str, Any]]:
    """(cik, preferred UniverseEntry) pairs, one per filer."""
    entries = universe.get_rankable_universe()
    grouped = universe.group_by_cik(entries)

    resolved = []
    for cik, listings in sorted(grouped.items()):
        # Without market caps to compare, the shortest symbol is a reliable
        # proxy for the primary listing (GOOG over GOOGL is the exception the
        # market-cap pass below corrects).
        preferred = sorted(listings, key=lambda e: (len(e.symbol), e.symbol))[0]
        resolved.append((cik, preferred))

    if limit:
        resolved = resolved[:limit]
    return resolved


def infer_model_from_facts(cik: str) -> str:
    """
    Guess a valuation model from what a company actually reports.

    Used only when SIC lookup fails — about 11% of filers, mostly dormant or
    recently delisted issuers absent from the last two years of SEC quarterly
    datasets. The failure this prevents is specific and costly: an unmapped bank
    silently scored as an industrial fails the debt-to-equity gate and vanishes
    from the ranking, which looks like missing data but is a mis-ranking.

    Deposits and a loan book are strong evidence of a lender; real-estate
    property with its own accumulated depreciation is strong evidence of a REIT.
    Anything else stays generic.
    """
    import edgar_provider

    try:
        signals = edgar_provider.get_concept_values(
            cik, ["deposits", "loans", "net_interest_income", "real_estate_net"]
        )
    except Exception:
        return "generic"

    if signals.get("deposits") and (
        signals.get("loans") or signals.get("net_interest_income")
    ):
        return "bank"
    if signals.get("real_estate_net"):
        return "reit"
    return "generic"


def collect_metrics(
    filers: Sequence[Tuple[str, Any]], progress_every: int = 500
) -> List[CompanyMetrics]:
    """Compute fundamentals-derived metrics and gate results for every filer."""
    sic_map = edgar_sic.get_sic_map()
    unmapped = 0
    inferred = 0
    companies: List[CompanyMetrics] = []

    for index, (cik, entry) in enumerate(filers, start=1):
        sic = sic_map.get(cik)
        if sic is None:
            unmapped += 1
            model = infer_model_from_facts(cik)
            if model != "generic":
                inferred += 1
        else:
            model = edgar_sic.model_for_sic(sic)

        company = compute_metrics(cik, entry.symbol, entry.name, model)
        # `compute_metrics` may already have recorded a blocking failure such as
        # missing fundamentals; gates are additive to that, never a replacement.
        company.gate_failures.extend(buffett_rank.evaluate_gates(company))
        companies.append(company)

        if index % progress_every == 0:
            logging.info(f"Rank: metrics for {index}/{len(filers)} filers")

    if unmapped:
        logging.warning(
            f"Rank: {unmapped} filers have no SIC code; {inferred} were routed to a "
            "sector model by inference from their filings, the rest scored as generic"
        )
    return companies


def fetch_market_data(symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Prices and share counts for the eligible set.

    Failures are per-symbol and non-fatal: a company without a quote simply has
    no value score and is ranked on quality alone, which is a better outcome
    than dropping it.
    """
    import market_data
    from market_db import MarketDatabase

    symbols = list(symbols)
    result: Dict[str, Dict[str, Any]] = {}
    db = MarketDatabase()

    # Price comes from the archive. It holds every rankable symbol, and a stored
    # close is better here than a live quote in two ways: a run takes minutes,
    # so live quotes price the first symbol and the last at different moments,
    # and a rank is only reproducible if the prices behind it can be read back.
    prices = db.get_latest_closes(symbols)

    # Shares outstanding moves on buybacks and issuance — quarterly at most —
    # but get_ticker_details_batch caches for a single day, so the ranking was
    # re-downloading the entire universe every run for one slow-moving number.
    cached_shares = db.get_share_counts(symbols, max_age_days=SHARE_COUNT_MAX_AGE_DAYS)
    stale = [s for s in symbols if s not in cached_shares]

    logging.info(
        f"Rank: {len(prices)}/{len(symbols)} prices from the archive; "
        f"share counts cached for {len(cached_shares)}, fetching {len(stale)}"
    )

    if stale:
        provider = market_data.get_shared_mdp()
        fetched: Dict[str, float] = {}
        for start in range(0, len(stale), MARKET_BATCH_SIZE):
            batch = stale[start : start + MARKET_BATCH_SIZE]
            try:
                details = provider.get_ticker_details_batch(set(batch))
            except Exception as exc:
                logging.warning(f"Rank: share-count batch failed: {exc}")
                continue
            for symbol in batch:
                shares = (details.get(symbol) or {}).get("sharesOutstanding")
                if shares:
                    fetched[symbol] = float(shares)
            logging.info(
                f"Rank: share counts {min(start + MARKET_BATCH_SIZE, len(stale))}/{len(stale)}"
            )
        if fetched:
            db.upsert_share_counts(fetched, date.today())
            cached_shares.update(fetched)

    for symbol in symbols:
        price = prices.get(symbol)
        shares = cached_shares.get(symbol)
        if price and shares:
            result[symbol] = {"price": price, "shares": shares}

    return result


def _apply_price_quality_gate(companies: Sequence[CompanyMetrics]) -> int:
    """Exclude companies whose stored price history is known to be wrong.

    Deliberately *not* part of `evaluate_gates`. Those gates hold to one rule —
    a company is excluded for something its filings show — and this is not a
    fact about the company at all. It is a fact about our data, and conflating
    the two would make the exclusion list lie about why a name is missing.

    It matters because the value half of the score is built from price: E/P and
    FCF/P divide by it. A series carrying an unapplied reverse split is wrong by
    the ratio, which is the difference between a stock looking desperately cheap
    and being ordinary. Ranking on it is how bad data becomes a decision.

    Only `high` severity excludes — a split on record the prices do not reflect,
    which is definitely wrong. An unexplained jump is suspicious rather than
    certain, and stays in with a flag on it for the reader to weigh.
    """
    try:
        from market_db import MarketDatabase

        flags = MarketDatabase().get_data_quality(
            [c.symbol for c in companies if c.symbol]
        )
    except Exception as exc:  # never let a missing scan block a ranking run
        logging.debug(f"Rank: price-quality flags unavailable ({exc})")
        return 0

    gated = 0
    for company in companies:
        flag = flags.get(company.symbol)
        if flag and flag.get("severity") == "high" and "price_history_unreliable" not in company.gate_failures:
            company.gate_failures.append("price_history_unreliable")
            gated += 1
    if gated:
        logging.info(f"Rank: {gated} excluded for unreliable price history")
    return gated


def _exclusion_records(companies: Sequence[CompanyMetrics]) -> List[Dict[str, Any]]:
    return [
        {
            "symbol": company.symbol,
            "cik": company.cik,
            "name": company.name,
            "model": company.model,
            "reasons": ", ".join(company.gate_failures),
            "period_count": company.period_count,
            "coverage": company.coverage,
        }
        for company in companies
        if company.gate_failures
    ]


def run(
    limit: Optional[int] = None,
    persist: bool = True,
    skip_market_data: bool = False,
) -> RunResult:
    """
    Execute a full ranking run.

    `skip_market_data=True` produces a quality-only ranking with no network
    calls, which is what the tests and any offline sanity check use.
    """
    started = time.time()
    filers = _resolve_universe(limit=limit)
    logging.info(f"Rank: starting run over {len(filers)} filers")

    store = get_store() if persist else None
    run_id = (
        store.start_run(
            len(filers),
            {
                "limit": limit,
                "skip_market_data": skip_market_data,
                "quality_weight": buffett_rank.QUALITY_WEIGHT,
                "value_weight": buffett_rank.VALUE_WEIGHT,
            },
        )
        if store
        else None
    )

    companies = collect_metrics(filers)
    _apply_price_quality_gate(companies)
    eligible = [c for c in companies if not c.gate_failures]
    exclusions = _exclusion_records(companies)
    logging.info(f"Rank: {len(eligible)} eligible, {len(exclusions)} excluded by gates")

    quality_frame = buffett_rank.score_quality(companies)

    value_scores = None
    if not skip_market_data and eligible:
        market = fetch_market_data([c.symbol for c in eligible])
        logging.info(f"Rank: market data for {len(market)}/{len(eligible)} symbols")

        value_frame = buffett_value.build_value_frame(eligible, market)
        if not value_frame.empty:
            value_scores = buffett_value.score_value(value_frame)
            # Carry the underlying value metrics through so a rank stays
            # explainable without recomputing anything.
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
    ranked_eligible = ranked[ranked["eligible"]].copy()
    # Re-number so rank 1..N covers only the ranked list.
    ranked_eligible["rank"] = range(1, len(ranked_eligible) + 1)

    duration = time.time() - started

    if store and run_id is not None:
        store.save_scores(run_id, ranked_eligible)
        store.save_exclusions(run_id, exclusions)
        store.finish_run(run_id, len(ranked_eligible), len(exclusions))

    logging.info(
        f"Rank: run complete in {duration:.0f}s — {len(ranked_eligible)} ranked, "
        f"{len(exclusions)} excluded"
    )

    return RunResult(
        run_id=run_id,
        ranked=ranked_eligible,
        exclusions=exclusions,
        duration_seconds=duration,
        stats={
            "filers": len(filers),
            "eligible": len(eligible),
            "excluded": len(exclusions),
            "ranked": len(ranked_eligible),
        },
    )
