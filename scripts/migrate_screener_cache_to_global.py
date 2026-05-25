"""
One-shot migration: merge legacy per-user `screener_cache` rows into the
global screener DB.

Why this exists
---------------
Historically screener results were upserted into each user's portfolio DB
(`data/users/<username>/portfolio.db`) AND opportunistically mirrored to the
global screener DB (`data/screener/screener_cache.db`). The mirror was
unreliable: some rows landed only in the user DB, and the "All Database
Stocks" view in the screener merged both at read-time, while background
workers only saw the global side. The mismatch produced the bug where the
screener showed ~200 unreviewed stocks that `ai_review_worker_missing`
could never see.

The refactor consolidates AI/IV/screener data into the global DB only. This
script copies any rows from each user's `screener_cache` into the global DB
on a last-write-wins basis (by `updated_at`), so no signal is lost.

It does NOT drop the per-user table — that's left as a no-op artifact to
keep the migration reversible. A follow-up can drop it once we've verified
no regression.

Usage
-----
    python scripts/migrate_screener_cache_to_global.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from typing import List, Tuple

# Make `src/` importable
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "src"))

import config  # noqa: E402
from db_utils import (  # noqa: E402
    get_global_screener_db_path,
    open_screener_db_conn,
)


COLUMNS = [
    "symbol", "name", "price", "intrinsic_value", "margin_of_safety",
    "pe_ratio", "market_cap", "sector",
    "ai_moat", "ai_financial_strength", "ai_predictability", "ai_growth",
    "ai_summary", "ai_sentiment", "ai_catalysts",
    "last_fiscal_year_end", "most_recent_quarter",
    "universe", "updated_at", "valuation_details",
]


def discover_user_dbs() -> List[Tuple[str, str]]:
    """Returns list of (username, db_path) tuples."""
    users_root = os.path.join(config.get_app_data_dir(), config.USERS_DIR)
    if not os.path.isdir(users_root):
        return []
    out: List[Tuple[str, str]] = []
    for entry in sorted(os.listdir(users_root)):
        user_dir = os.path.join(users_root, entry)
        if not os.path.isdir(user_dir):
            continue
        db_path = os.path.join(user_dir, config.PORTFOLIO_DB_FILENAME)
        if os.path.isfile(db_path):
            out.append((entry, db_path))
    return out


def fetch_user_rows(db_path: str) -> List[dict]:
    """Returns all screener_cache rows from a user DB."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as e:
        logging.warning(f"Could not open {db_path}: {e}")
        return []
    try:
        cursor = conn.cursor()
        try:
            cursor.execute(
                f"SELECT {', '.join(COLUMNS)} FROM screener_cache"
            )
        except sqlite3.OperationalError:
            return []
        rows = [dict(r) for r in cursor.fetchall()]
        return rows
    finally:
        conn.close()


def merge_into_global(rows: List[dict], dry_run: bool) -> Tuple[int, int]:
    """Inserts/updates rows in the global DB if they are missing or newer.

    Returns (rows_written, rows_skipped_older_or_equal).
    """
    if not rows:
        return 0, 0

    global_conn = open_screener_db_conn()
    if global_conn is None:
        logging.error("Could not open global screener DB; aborting merge.")
        return 0, 0

    written = 0
    skipped = 0
    try:
        cur = global_conn.cursor()
        for row in rows:
            sym = row.get("symbol")
            univ = row.get("universe") or "manual"
            if not sym:
                continue

            cur.execute(
                "SELECT updated_at FROM screener_cache WHERE symbol = ? AND universe = ?",
                (sym, univ),
            )
            existing = cur.fetchone()
            row_ts = row.get("updated_at") or ""
            if existing is not None:
                existing_ts = existing[0] or ""
                if existing_ts >= row_ts:
                    skipped += 1
                    continue

            if dry_run:
                written += 1
                continue

            placeholders = ", ".join(["?"] * len(COLUMNS))
            cur.execute(
                f"""
                INSERT INTO screener_cache ({', '.join(COLUMNS)})
                VALUES ({placeholders})
                ON CONFLICT(symbol, universe) DO UPDATE SET
                    name = excluded.name,
                    price = excluded.price,
                    intrinsic_value = excluded.intrinsic_value,
                    margin_of_safety = excluded.margin_of_safety,
                    pe_ratio = excluded.pe_ratio,
                    market_cap = excluded.market_cap,
                    sector = excluded.sector,
                    ai_moat = excluded.ai_moat,
                    ai_financial_strength = excluded.ai_financial_strength,
                    ai_predictability = excluded.ai_predictability,
                    ai_growth = excluded.ai_growth,
                    ai_summary = excluded.ai_summary,
                    ai_sentiment = excluded.ai_sentiment,
                    ai_catalysts = excluded.ai_catalysts,
                    last_fiscal_year_end = excluded.last_fiscal_year_end,
                    most_recent_quarter = excluded.most_recent_quarter,
                    updated_at = excluded.updated_at,
                    valuation_details = excluded.valuation_details
                """,
                tuple(row.get(c) for c in COLUMNS),
            )
            written += 1
        if not dry_run:
            global_conn.commit()
    finally:
        global_conn.close()

    return written, skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would be merged without writing.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    logging.info("Global screener DB: %s", get_global_screener_db_path())

    user_dbs = discover_user_dbs()
    if not user_dbs:
        logging.info("No user DBs found.")
        return 0

    total_written = 0
    total_skipped = 0
    for username, db_path in user_dbs:
        rows = fetch_user_rows(db_path)
        if not rows:
            logging.info("[%s] no screener_cache rows.", username)
            continue
        w, s = merge_into_global(rows, dry_run=args.dry_run)
        total_written += w
        total_skipped += s
        logging.info(
            "[%s] %d rows: %d %s, %d skipped (older or equal).",
            username,
            len(rows),
            w,
            "would-write" if args.dry_run else "written",
            s,
        )

    logging.info(
        "Done. Total %s: %d. Total skipped: %d.",
        "would-write" if args.dry_run else "written",
        total_written,
        total_skipped,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
