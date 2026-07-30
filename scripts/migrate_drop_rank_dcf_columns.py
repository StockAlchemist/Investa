"""Drop the DCF columns from an existing ranking database.

`rank_scores` carried `margin_of_safety` and `conservative_iv`, both derived from
the Monte Carlo DCF that `buffett_value` no longer computes (see that module for
the measurement that retired it). `CREATE TABLE IF NOT EXISTS` cannot remove a
column from a database that already exists, so without this migration the two
columns survive in place — and that is worse than untidy:

  * every new run writes NULL into them, while the runs already stored keep
    their old non-NULL values, so a client reading the table sees a margin of
    safety for some rows and not others with nothing to distinguish "not
    computed any more" from "could not be computed";
  * `get_scores_frame` selects `*`, so those stale values reach the API payload
    and the clients would happily render a number the ranking no longer uses.

Dropping the columns makes the absence explicit and unambiguous.

The ranking database is a derived snapshot — the daily worker rebuilds it — so
this is safe to run and cheap to recover from. It is still written to be
idempotent and to leave the database untouched when there is nothing to do.

    python scripts/migrate_drop_rank_dcf_columns.py [--dry-run] [--db PATH]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

DEAD_COLUMNS = ("margin_of_safety", "conservative_iv")


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "buffett_ranks.db")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None, help="path to buffett_ranks.db")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.db or default_db_path()
    if not os.path.exists(path):
        print(f"No ranking database at {path} — nothing to migrate.")
        return 0

    conn = sqlite3.connect(path, timeout=30.0)
    try:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(rank_scores)")}
        present = [c for c in DEAD_COLUMNS if c in existing]
        if not present:
            print("rank_scores already has no DCF columns — nothing to do.")
            return 0

        # Report what is actually being discarded, so the run is auditable
        # rather than a silent schema edit.
        for column in present:
            filled = conn.execute(
                f"SELECT COUNT(*) FROM rank_scores WHERE {column} IS NOT NULL"
            ).fetchone()[0]
            print(f"  {column}: {filled} non-NULL values will be dropped")

        if args.dry_run:
            print("Dry run — no changes written.")
            return 0

        for column in present:
            conn.execute(f"ALTER TABLE rank_scores DROP COLUMN {column}")
        conn.commit()
        print(f"Dropped {', '.join(present)} from {path}")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
