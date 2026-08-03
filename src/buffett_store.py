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

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

import config

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
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rank_runs (
                    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    started_at TEXT NOT NULL,
                    finished_at TEXT,
                    universe_size INTEGER,
                    ranked_count INTEGER,
                    excluded_count INTEGER,
                    parameters TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rank_scores (
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    cik TEXT,
                    name TEXT,
                    model TEXT,
                    rank INTEGER,
                    composite_score REAL,
                    quality_score REAL,
                    value_score REAL,
                    confidence REAL,
                    coverage REAL,
                    returns_on_capital REAL,
                    financial_strength REAL,
                    predictability REAL,
                    growth REAL,
                    capital_allocation REAL,
                    price REAL,
                    market_cap REAL,
                    earnings_yield REAL,
                    fcf_yield REAL,
                    period_count INTEGER,
                    latest_period TEXT,
                    PRIMARY KEY (run_id, symbol)
                )
            """)
            # The unranked bucket is a first-class output, not a discard pile:
            # a ranking that silently drops a fifth of the market is not honest
            # about what it covers.
            conn.execute("""
                CREATE TABLE IF NOT EXISTS rank_exclusions (
                    run_id INTEGER NOT NULL,
                    symbol TEXT NOT NULL,
                    cik TEXT,
                    name TEXT,
                    model TEXT,
                    reasons TEXT,
                    period_count INTEGER,
                    coverage REAL,
                    PRIMARY KEY (run_id, symbol)
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_scores_run_rank ON rank_scores (run_id, rank)"
            )
            conn.commit()

    def start_run(self, universe_size: int, parameters: Dict[str, Any]) -> int:
        with self._write_lock, self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO rank_runs (started_at, universe_size, parameters) VALUES (?, ?, ?)",
                (datetime.now().isoformat(), universe_size, json.dumps(parameters)),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def finish_run(self, run_id: int, ranked: int, excluded: int) -> None:
        with self._write_lock, self._connect() as conn:
            conn.execute(
                """UPDATE rank_runs SET finished_at = ?, ranked_count = ?, excluded_count = ?
                   WHERE run_id = ?""",
                (datetime.now().isoformat(), ranked, excluded, run_id),
            )
            conn.commit()
        # Pruned here rather than at start_run so a run is only ever counted
        # against the retention budget once it is complete — an interrupted run
        # cannot evict a good one.
        self.prune_runs()

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
        cutoff = (datetime.now() - timedelta(hours=stale_hours)).isoformat()
        with self._write_lock, self._connect() as conn:
            doomed = [
                row[0]
                for row in conn.execute(
                    """SELECT run_id FROM rank_runs
                       WHERE (
                           finished_at IS NOT NULL AND run_id NOT IN (
                               SELECT run_id FROM rank_runs WHERE finished_at IS NOT NULL
                               ORDER BY run_id DESC LIMIT ?
                           )
                       ) OR (
                           finished_at IS NULL AND started_at < ?
                       )""",
                    (keep, cutoff),
                )
            ]
            if not doomed:
                return 0
            marks = ",".join("?" * len(doomed))
            for table in ("rank_scores", "rank_exclusions", "rank_runs"):
                conn.execute(f"DELETE FROM {table} WHERE run_id IN ({marks})", doomed)
            conn.commit()

        # VACUUM reclaims the freed pages to the filesystem; without it the file
        # never shrinks, which is the whole point here. It cannot run inside a
        # transaction, hence its own connection outside the block above.
        try:
            with self._connect() as conn:
                conn.isolation_level = None
                conn.execute("VACUUM")
        except sqlite3.Error as exc:  # pragma: no cover - never worth failing a run
            logging.warning(f"Rank store: VACUUM after pruning failed: {exc}")

        logging.info(f"Rank store: pruned {len(doomed)} run(s), keeping {keep}")
        return len(doomed)

    def save_scores(self, run_id: int, frame: pd.DataFrame) -> int:
        """Persist the ranked rows of one run."""
        if frame.empty:
            return 0

        columns = [c for c in _SCORE_COLUMNS if c in frame.columns]
        rows = []
        for symbol, record in frame.iterrows():
            values = [_coerce(record.get(column)) for column in columns]
            rows.append((run_id, str(symbol), *values))

        placeholders = ", ".join("?" * (len(columns) + 2))
        statement = (
            f"INSERT OR REPLACE INTO rank_scores (run_id, symbol, {', '.join(columns)}) "
            f"VALUES ({placeholders})"
        )
        with self._write_lock, self._connect() as conn:
            conn.executemany(statement, rows)
            conn.commit()
        return len(rows)

    def save_exclusions(self, run_id: int, records: List[Dict[str, Any]]) -> int:
        if not records:
            return 0
        rows = [
            (
                run_id,
                record.get("symbol"),
                record.get("cik"),
                record.get("name"),
                record.get("model"),
                record.get("reasons"),
                record.get("period_count"),
                record.get("coverage"),
            )
            for record in records
        ]
        with self._write_lock, self._connect() as conn:
            conn.executemany(
                """INSERT OR REPLACE INTO rank_exclusions
                   (run_id, symbol, cik, name, model, reasons, period_count, coverage)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
        return len(rows)

    # --- reading ----------------------------------------------------------

    def latest_run_id(self) -> Optional[int]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT run_id FROM rank_runs WHERE finished_at IS NOT NULL "
                "ORDER BY run_id DESC LIMIT 1"
            ).fetchone()
        return int(row["run_id"]) if row else None

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
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return []

        query = "SELECT * FROM rank_scores WHERE run_id = ?"
        params: List[Any] = [run_id]
        if model:
            query += " AND model = ?"
            params.append(model)
        if search and search.strip():
            query += " AND (symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')"
            pattern = f"%{_escape_like(search.strip())}%"
            params.extend([pattern, pattern])
        query += " ORDER BY rank ASC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params)]

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
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return 0

        query = "SELECT COUNT(*) FROM rank_scores WHERE run_id = ?"
        params: List[Any] = [run_id]
        if model:
            query += " AND model = ?"
            params.append(model)
        if search and search.strip():
            query += " AND (symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')"
            pattern = f"%{_escape_like(search.strip())}%"
            params.extend([pattern, pattern])

        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])

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
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return []

        query = "SELECT * FROM rank_exclusions WHERE run_id = ?"
        params: List[Any] = [run_id]
        if search and search.strip():
            query += " AND (symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')"
            pattern = f"%{_escape_like(search.strip())}%"
            params.extend([pattern, pattern])
        query += " ORDER BY symbol LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            return [dict(row) for row in conn.execute(query, params)]

    def count_exclusions(
        self, run_id: Optional[int] = None, search: Optional[str] = None
    ) -> int:
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return 0

        query = "SELECT COUNT(*) FROM rank_exclusions WHERE run_id = ?"
        params: List[Any] = [run_id]
        if search and search.strip():
            query += " AND (symbol LIKE ? ESCAPE '\\' OR name LIKE ? ESCAPE '\\')"
            pattern = f"%{_escape_like(search.strip())}%"
            params.extend([pattern, pattern])

        with self._connect() as conn:
            return int(conn.execute(query, params).fetchone()[0])

    def get_run(self, run_id: Optional[int] = None) -> Optional[Dict[str, Any]]:
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM rank_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None

    def get_scores_frame(self, run_id: Optional[int] = None) -> pd.DataFrame:
        """
        Every scored row of a run as a frame.

        Paged reads serve a list the user scrolls; this serves a *re-blend*,
        which reorders the whole run — a strategy weighting quality at 80%
        rather than the stored 60% can promote a company from rank 400, so
        anything less than the full set would silently truncate the answer.
        """
        run_id = run_id or self.latest_run_id()
        if run_id is None:
            return pd.DataFrame()
        with self._connect() as conn:
            rows = [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM rank_scores WHERE run_id = ? ORDER BY rank ASC",
                    (run_id,),
                )
            ]
        return pd.DataFrame(rows)

    def get_symbol_history(self, symbol: str, limit: int = 24) -> List[Dict[str, Any]]:
        """One company's rank across runs — the point of keeping snapshots."""
        with self._connect() as conn:
            return [
                dict(row)
                for row in conn.execute(
                    """SELECT s.*, r.started_at FROM rank_scores s
                       JOIN rank_runs r ON r.run_id = s.run_id
                       WHERE s.symbol = ? ORDER BY s.run_id DESC LIMIT ?""",
                    (symbol.upper(), limit),
                )
            ]


def _escape_like(term: str) -> str:
    """
    Neutralise SQL LIKE wildcards in user input.

    Without this a search for "%" matches everything and "_" matches any single
    character, so a user typing a perfectly ordinary character gets baffling
    results. The backslash must be escaped first or it would corrupt the escape
    sequences added after it.
    """
    return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _coerce(value: Any) -> Any:
    """Normalise numpy/pandas scalars and NaN into SQLite-friendly values."""
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, (int, float, str)):
        return value
    return str(value)


_shared_store: Optional[BuffettRankStore] = None


def get_store() -> BuffettRankStore:
    global _shared_store
    if _shared_store is None:
        _shared_store = BuffettRankStore()
    return _shared_store
