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

import time  # noqa: F401 -- read by the Rust port at call time
from datetime import date  # noqa: F401 -- read by the Rust port at call time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import investa_core
import pandas as pd

import buffett_rank  # noqa: F401 -- read by the Rust port at call time
import buffett_value  # noqa: F401 -- read by the Rust port at call time
import edgar_sic  # noqa: F401 -- read by the Rust port at call time
import universe  # noqa: F401 -- read by the Rust port at call time
from buffett_metrics import CompanyMetrics, compute_metrics  # noqa: F401 -- read by the Rust port at call time
from buffett_store import get_store  # noqa: F401 -- read by the Rust port at call time

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
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_resolve_universe(limit)


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
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_infer_model_from_facts(cik)


def collect_metrics(
    filers: Sequence[Tuple[str, Any]], progress_every: int = 500
) -> List[CompanyMetrics]:
    """Compute fundamentals-derived metrics and gate results for every filer."""
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_collect_metrics(filers, progress_every)


def fetch_market_data(symbols: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """
    Prices and share counts for the eligible set.

    Failures are per-symbol and non-fatal: a company without a quote simply has
    no value score and is ranked on quality alone, which is a better outcome
    than dropping it.
    """
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_fetch_market_data(symbols)


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
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_apply_price_quality_gate(companies)


def _exclusion_records(companies: Sequence[CompanyMetrics]) -> List[Dict[str, Any]]:
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_exclusion_records(companies)


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
    # Ported to Rust: rust/investa_core/src/bp.rs
    return investa_core.bp_run(limit, persist, skip_market_data)
