#!/usr/bin/env python3
"""Record where an FX rate came from (plan Phase 5.1).

`daily_fx` has no `source` column, so every rate in it is anonymous. That was
tolerable while Yahoo was the only writer and is not once the ECB's official
reference rates start filling the gaps Yahoo leaves: without provenance there is
no way to ask which days are Yahoo's close, which are the ECB's 14:15 CET fix,
and — after a provider outage — which days would need re-fetching if the two
were ever reconciled.

Existing rows are labelled `yahoo`, which is what they are. No rate is changed.

    python scripts/migrate_fx_provenance.py [--dry-run] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def plan_changes(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """The (description, sql) pairs still outstanding. Idempotent."""
    tables = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "daily_fx" not in tables:
        return []

    columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_fx)")}
    if "source" in columns:
        return []

    return [
        (
            "add daily_fx.source (existing rows -> 'yahoo')",
            "ALTER TABLE daily_fx ADD COLUMN source TEXT DEFAULT 'yahoo'",
        ),
        (
            "label existing rows",
            "UPDATE daily_fx SET source = 'yahoo' WHERE source IS NULL",
        ),
    ]


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
            print("daily_fx already records provenance — nothing to do.")
            return 0

        print(f"{path}\n{len(changes)} change(s) outstanding:")
        for description, _ in changes:
            print(f"  - {description}")

        if args.dry_run:
            print("\nDry run — nothing written. Re-run without --dry-run to apply.")
            return 0

        with conn:
            for _, sql in changes:
                conn.execute(sql)

        print("\nApplied.")
        if plan_changes(conn):
            print("WARNING: change still outstanding after apply.")
            return 1
        print("Verified: schema current.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
