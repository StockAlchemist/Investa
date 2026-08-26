"""Schema for the local market archive (plan Phase 1.1 / 1.3).

`market_data.db` grew as a cache: it stores whatever Yahoo last returned for
whatever range a page happened to ask for. Turning it into an archive — data
kept because it will not be re-fetchable later — needs three things the cache
never had.

**A corporate-actions table.** Yahoo's `Close` is split-adjusted *as of the
download date*. That is harmless for a cache that refetches everything and
poisonous for a store that accretes: when a symbol splits, only the last five
days are refetched and every older row keeps the pre-split basis, leaving one
series that jumps 10x in the middle for no reason. The fix is to stop storing an
adjusted price at all — keep the raw traded close plus the events, and apply the
adjustment at read time, which is the convention the transaction ledger already
uses (see migrate_unadjust_dividend_splits.py). Yahoo already returns the events:
the worker fetches with `actions=True`, and `upsert_ohlcv` has simply been
dropping the `Dividends` and `Stock Splits` columns on the floor.

**A per-symbol price basis.** Conversion has to be all-or-nothing per symbol. A
per-row flag would permit one symbol to hold both bases at once, which is
precisely the seam this exists to prevent, so the basis lives on `sync_metadata`
and moves only inside the transaction that rewrites the symbol's rows.
`delisted_at` joins it so the nightly delta can stop retrying names that will
never return.

**Somewhere to put the data that has no provider.** `fund_nav` holds Thai
SSF/RMF NAVs, which no commercial feed carries and which are currently a single
hand-entered number per fund — so every historical valuation of those positions
is flat-lined at today's price. `backfill_progress` makes a multi-hour universe
fill resumable.

Nothing here rewrites a single existing row: it adds tables and columns, and
every existing symbol keeps the `split_adj` basis it already had. The conversion
itself is a separate, per-symbol operation.

    python scripts/migrate_market_archive_phase1.py [--dry-run] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

NEW_TABLES: dict[str, str] = {
    "corporate_action": """
        CREATE TABLE corporate_action (
            symbol      TEXT NOT NULL,
            date        TEXT NOT NULL,   -- yyyy-MM-dd ex-date, market-local
            kind        TEXT NOT NULL,   -- 'split' | 'dividend'
            value       REAL NOT NULL,   -- split: ratio (4.0 = 4:1); dividend: cash/share
            currency    TEXT,            -- dividends only
            source      TEXT NOT NULL,   -- 'yahoo' | 'tiingo' | 'edgar' | 'manual'
            ingested_at TEXT NOT NULL,
            PRIMARY KEY (symbol, date, kind)
        )
    """,
    "fund_nav": """
        CREATE TABLE fund_nav (
            fund_code TEXT NOT NULL,
            date      TEXT NOT NULL,
            nav       REAL NOT NULL,
            currency  TEXT,
            source    TEXT NOT NULL,
            PRIMARY KEY (fund_code, date)
        )
    """,
    # Shares outstanding, kept because the ranking needs it and the only other
    # source re-downloads the whole universe from Yahoo every day for a number
    # that moves at most quarterly.
    "share_count": """
        CREATE TABLE share_count (
            symbol   TEXT NOT NULL,
            shares   REAL NOT NULL,
            as_of    TEXT NOT NULL,   -- yyyy-MM-dd the figure was observed
            source   TEXT NOT NULL,
            PRIMARY KEY (symbol)
        )
    """,
    # An independent provider's close for a given day, kept only where it is
    # needed to adjudicate a disagreement. Not a second price history: a handful
    # of bars either side of a suspected seam is enough to say which of two
    # sources is on the right basis.
    "reference_price": """
        CREATE TABLE reference_price (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            close  REAL NOT NULL,
            source TEXT NOT NULL,
            fetched_at TEXT NOT NULL,
            PRIMARY KEY (symbol, date, source)
        )
    """,
    "backfill_progress": """
        CREATE TABLE backfill_progress (
            tier         TEXT NOT NULL,
            symbol       TEXT NOT NULL,
            done_through TEXT,
            attempts     INTEGER NOT NULL DEFAULT 0,
            last_error   TEXT,
            updated_at   TEXT,
            PRIMARY KEY (tier, symbol)
        )
    """,
}

NEW_INDEXES: dict[str, str] = {
    "idx_action_symbol": "CREATE INDEX idx_action_symbol ON corporate_action (symbol, date)",
}

# sync_metadata gains the basis flag and a delisting marker. 'split_adj' is the
# correct default for every row already present: that is what Yahoo gave us.
NEW_COLUMNS: dict[str, str] = {
    "price_basis": "ALTER TABLE sync_metadata ADD COLUMN price_basis TEXT NOT NULL DEFAULT 'split_adj'",
    "delisted_at": "ALTER TABLE sync_metadata ADD COLUMN delisted_at TEXT",
}


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def plan_changes(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return the (description, sql) pairs still outstanding. Idempotent."""
    changes: list[tuple[str, str]] = []

    existing_tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    for name, sql in NEW_TABLES.items():
        if name not in existing_tables:
            changes.append((f"create table {name}", sql))

    existing_indexes = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }
    for name, sql in NEW_INDEXES.items():
        if name not in existing_indexes:
            changes.append((f"create index {name}", sql))

    if "sync_metadata" in existing_tables:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sync_metadata)")}
        for name, sql in NEW_COLUMNS.items():
            if name not in cols:
                changes.append((f"add sync_metadata.{name}", sql))

    return changes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="path to market_data.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.db or default_db_path()
    if not os.path.exists(path):
        print(f"No market database at {path} — nothing to migrate.")
        return 0

    conn = sqlite3.connect(path, timeout=60.0)
    try:
        changes = plan_changes(conn)
        if not changes:
            print("Archive schema already current — nothing to do.")
            return 0

        print(f"{path}\n{len(changes)} change(s) outstanding:")
        for description, _ in changes:
            print(f"  - {description}")

        if args.dry_run:
            print("\nDry run — nothing written. Re-run without --dry-run to apply.")
            return 0

        # One transaction: either the archive gains the whole shape or none of
        # it, so a half-migrated database can never be observed.
        with conn:
            for _, sql in changes:
                conn.execute(sql)

        print("\nApplied.")

        remaining = plan_changes(conn)
        if remaining:
            print(f"WARNING: {len(remaining)} change(s) still outstanding after apply.")
            return 1
        print("Verified: schema current.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
