#!/usr/bin/env python3
"""Index the archive for cross-sectional reads (plan Phase 6.1).

`daily_ohlcv` has exactly one index — the implicit primary key
`(symbol, date, interval)`. That serves "one symbol's whole history" perfectly
and every question asked the other way round not at all: "every close on date D"
is a full scan.

Measured on the live archive at 20,617,199 rows, before this migration:

    all closes on one date          2,011 ms   (SCAN daily_ohlcv)
    a 12-date month-end panel       1,849 ms
    one symbol's history                8 ms   (uses the PK)

Those are the shapes the heatmap, the ranking and every rebalance-date backtest
are built from, so the cost is paid per date, repeatedly. The plan predicted
this would "stop being acceptable" at 35M rows; it is already unacceptable at
20M.

**The cost is disk, and it is not small.** An index over 20.6M rows of
`(date, symbol)` adds several hundred MB to a 2.3 GB file — the same order as
the primary key's own 586 MB. That is the trade: a few hundred MB of a cheap
local disk against two seconds of every cross-sectional query. Worth knowing
before running it on a machine that is tight on space.

`interval` is deliberately left out of the index. Daily rows are ~99% of the
table, so a query that also filters `interval='1d'` discards almost nothing
after the index lookup, and including it would widen every entry for no gain.

    python scripts/migrate_archive_indexes.py [--dry-run] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

NEW_INDEXES: dict[str, tuple[str, str]] = {
    # (table, SQL) — table is checked so a missing table is skipped, not an error.
    "idx_daily_date": (
        "daily_ohlcv",
        "CREATE INDEX idx_daily_date ON daily_ohlcv (date, symbol)",
    ),
    # The same question asked of intraday bars, for the same reason.
    "idx_intraday_ts": (
        "intraday_ohlcv",
        "CREATE INDEX idx_intraday_ts ON intraday_ohlcv (timestamp, symbol)",
    ),
    # fund_nav and daily_fx are keyed (code, date) / (pair, date, interval), so a
    # by-date sweep across funds or pairs scans them too. They are small enough
    # that it has never mattered, and small enough that the index is nearly free.
    "idx_fund_nav_date": (
        "fund_nav",
        "CREATE INDEX idx_fund_nav_date ON fund_nav (date, fund_code)",
    ),
}


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def plan_changes(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """The (description, sql) pairs still outstanding. Idempotent."""
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    existing = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
    }
    changes = []
    for name, (table, sql) in NEW_INDEXES.items():
        if name not in existing and table in tables:
            changes.append((f"create {name} on {table}", sql))
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

    before = os.path.getsize(path)
    conn = sqlite3.connect(path, timeout=120.0)
    try:
        changes = plan_changes(conn)
        if not changes:
            print("Archive already indexed — nothing to do.")
            return 0

        print(f"{path} ({before / 1048576:.0f} MB)\n{len(changes)} change(s) outstanding:")
        for description, _ in changes:
            print(f"  - {description}")

        if args.dry_run:
            print("\nDry run — nothing written. Re-run without --dry-run to apply.")
            return 0

        # Not one transaction, unlike the schema migrations: each index is
        # independent, they take minutes on 20M rows, and a failure partway
        # through leaves the ones already built usable rather than rolling back
        # the lot.
        for description, sql in changes:
            print(f"\n  {description} ...", end="", flush=True)
            started = time.perf_counter()
            conn.execute(sql)
            conn.commit()
            print(f" {time.perf_counter() - started:.1f}s")

        after = os.path.getsize(path)
        print(
            f"\nApplied. {before / 1048576:.0f} MB -> {after / 1048576:.0f} MB "
            f"(+{(after - before) / 1048576:.0f} MB)"
        )

        if plan_changes(conn):
            print("WARNING: an index is still outstanding after apply.")
            return 1
        print("Verified: indexes present.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
