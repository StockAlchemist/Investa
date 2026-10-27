# -*- coding: utf-8 -*-
"""
Dated snapshots of the Buffett/value ranking.

Kept in its own database rather than folded into `screener_cache` because the
two have different lifecycles. The screener cache is a rolling "latest known
state" keyed by symbol; this is an append-only history of *runs*, and principle
P8 depends on that history surviving. Being able to ask "what did the ranking
say in March, and on what inputs" is the difference between a model you can
audit and one you can only trust.

Each row therefore stores not just the score but the provenance needed to
reconstruct it: the fiscal period the fundamentals came from, the number of
annual periods available, the coverage-derived confidence, and the pillar
breakdown. "Why is this company ranked fourth" must be answerable from the row
alone.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from typing import Any, Dict, List, Optional

import pandas as pd

import config
import investa_core
from db_utils import get_db_connection

_DB_FILENAME = "buffett_ranks.db"

# How many finished runs to keep. Every run writes the whole universe — ~1,240
# ranked rows and ~4,300 exclusions, about 0.79 MB — and nothing ever deleted
# them, so the store grew without bound: 51 MB in its first week, on course for
# GitHub's 100 MB hard limit inside another one.
#
# Thirty is set by the one reader that wants more than the latest run,
# `get_symbol_history`, which asks for 24. Keeping a margin above it means the
# rank trajectory a stock's page draws is never shortened by pruning, and the
# file settles around 24 MB instead of climbing.
#
# A ranking's inputs move quarterly, so this is a lot of history, not a little:
# at the worker's intended one run a day it is a month of it.
_KEEP_RUNS = 30

# How long an unfinished run is presumed to still be in flight. Rows are written
# before `finish_run` marks a run complete, so a run killed in between holds a
# full universe — 0.79 MB — that the retention rule would never reach, since it
# counts and evicts only finished runs. A worker that dies mid-write is not
# hypothetical: four orphaned ones were killed at once here.
#
# Six hours against a run that takes about eighty seconds. The margin is what
# makes this safe: a genuinely running job is never mistaken for an abandoned
# one, and an abandoned one is reclaimed on the next day's run at the latest.
_STALE_RUN_HOURS = 6

# Columns persisted from the ranking frame. Anything not listed here is
# reconstructible from these plus the fact store.
_SCORE_COLUMNS = [
    "cik",
    "name",
    "model",
    "rank",
    "composite_score",
    "quality_score",
    "value_score",
    "confidence",
    "coverage",
    "returns_on_capital",
    "financial_strength",
    "predictability",
    "growth",
    "capital_allocation",
    "price",
    "market_cap",
    "earnings_yield",
    "fcf_yield",
    "period_count",
    "latest_period",
]


def _db_path() -> str:
    directory = os.path.join(config.get_app_data_dir(), config.DB_DIR)
    os.makedirs(directory, exist_ok=True)
    return os.path.join(directory, _DB_FILENAME)


class BuffettRankStore:
    """Append-only store of ranking runs."""

    _write_lock = threading.Lock()

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or _db_path()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = get_db_connection(self.db_path, use_cache=False, timeout=30.0)
        if conn is None:
            raise sqlite3.OperationalError(
                f"Could not open database connection to {self.db_path}"
            )
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        investa_core.rs_init_db(self)

    def start_run(self, universe_size: int, parameters: Dict[str, Any]) -> int:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_start_run(self, universe_size, parameters)

    def finish_run(self, run_id: int, ranked: int, excluded: int) -> None:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        investa_core.rs_finish_run(self, run_id, ranked, excluded)

    def prune_runs(
        self, keep: int = _KEEP_RUNS, stale_hours: float = _STALE_RUN_HOURS
    ) -> int:
        """
        Drop all but the newest `keep` finished runs. Returns how many runs went.

        A run still in flight is never touched, however many there are: its rows
        land before `finish_run` marks it, so counting it against the budget could
        delete a run mid-write. It is also not counted *towards* the budget, so an
        in-flight run cannot evict a good one.

        An unfinished run older than `stale_hours` is a different thing — a worker
        that died — and is dropped. Without that it would hold whatever it had
        already written for good, since the budget only ever evicts finished runs.
        """
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_prune_runs(self, keep, stale_hours)

    def save_scores(self, run_id: int, frame: pd.DataFrame) -> int:
        """Persist the ranked rows of one run."""
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_save_scores(self, run_id, frame)

    def save_exclusions(self, run_id: int, records: List[Dict[str, Any]]) -> int:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_save_exclusions(self, run_id, records)

    # --- reading ----------------------------------------------------------

    def latest_run_id(self) -> Optional[int]:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_latest_run_id(self)

    def get_ranked(
        self,
        run_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        model: Optional[str] = None,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        One page of the ranked list.

        `search` matches symbol or company name across the *whole* run, not just
        the page being displayed — filtering client-side would only ever find
        the 100 rows already loaded, which is useless for a 1,100-name list.
        Rows keep their true rank, so a searched company shows where it actually
        placed rather than its position within the filtered results.
        """
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_get_ranked(self, run_id, limit, offset, model, search)

    def count_ranked(
        self,
        run_id: Optional[int] = None,
        model: Optional[str] = None,
        search: Optional[str] = None,
    ) -> int:
        """
        How many rows match the current filters.

        Needed so a client can tell "no more pages" from "no matches", which it
        cannot infer from a short page alone.
        """
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_count_ranked(self, run_id, model, search)

    def get_exclusions(
        self,
        run_id: Optional[int] = None,
        limit: int = 100,
        offset: int = 0,
        search: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        One page of the unranked bucket.

        Searchable for the same reason the ranked list is: when a company is
        absent from the ranking, "why" is the immediate next question, and
        paging through 4,000 rows alphabetically to find it is not an answer.
        """
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_get_exclusions(self, run_id, limit, offset, search)

    def count_exclusions(
        self, run_id: Optional[int] = None, search: Optional[str] = None
    ) -> int:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_count_exclusions(self, run_id, search)

    def get_run(self, run_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_get_run(self, run_id)

    def get_scores_frame(self, run_id: Optional[int] = None) -> pd.DataFrame:
        """
        Every scored row of a run as a frame.

        Paged reads serve a list the user scrolls; this serves a *re-blend*,
        which reorders the whole run — a strategy weighting quality at 80%
        rather than the stored 60% can promote a company from rank 400, so
        anything less than the full set would silently truncate the answer.
        """
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_get_scores_frame(self, run_id)

    def get_symbol_history(self, symbol: str, limit: int = 24) -> List[Dict[str, Any]]:
        """One company's rank across runs — the point of keeping snapshots."""
        # Ported to Rust: rust/investa_core/src/rank_store.rs
        return investa_core.rs_get_symbol_history(self, symbol, limit)


def _escape_like(term: str) -> str:
    """
    Neutralise SQL LIKE wildcards in user input.

    Without this a search for "%" matches everything and "_" matches any single
    character, so a user typing a perfectly ordinary character gets baffling
    results. The backslash must be escaped first or it would corrupt the escape
    sequences added after it.
    """
    # Ported to Rust: rust/investa_core/src/rank_store.rs
    return investa_core.rs_escape_like(term)


def _coerce(value: Any) -> Any:
    """Normalise numpy/pandas scalars and NaN into SQLite-friendly values."""
    # Ported to Rust: rust/investa_core/src/rank_store.rs
    return investa_core.rs_coerce(value)


_shared_store: Optional[BuffettRankStore] = None


def get_store() -> BuffettRankStore:
    global _shared_store
    if _shared_store is None:
        _shared_store = BuffettRankStore()
    return _shared_store
