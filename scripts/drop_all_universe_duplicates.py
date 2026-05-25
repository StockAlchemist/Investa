"""
Drop the redundant `universe='all'` rows from the global screener cache.

Background
----------
Historically, screening with universe_type='all' upserted a duplicate row
tagged universe='all' for every symbol, on top of its canonical universe
row (sp500, russell2000, sp400, manual, watchlist_*). The "All Database
Stocks" UI does NOT actually filter by universe='all' — it dedupes by
symbol via `get_all_distinct_screener_results`. So the 'all' rows are
pure duplication.

This script:
  1. Promotes any symbol that ONLY has an 'all' row to universe='manual',
     so we don't drop the only copy of that symbol.
  2. Deletes the remaining 'all' rows.

Usage
-----
    python scripts/drop_all_universe_duplicates.py [--dry-run]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "src"))

from db_utils import (  # noqa: E402
    get_global_screener_db_path,
    open_screener_db_conn,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    logging.info("Global screener DB: %s", get_global_screener_db_path())

    conn = open_screener_db_conn()
    if conn is None:
        logging.error("Could not open global screener DB.")
        return 1

    try:
        cur = conn.cursor()

        # 1. Symbols that ONLY have universe='all' (no canonical row).
        cur.execute(
            """
            SELECT symbol FROM screener_cache
            WHERE universe = 'all'
              AND symbol NOT IN (
                  SELECT symbol FROM screener_cache WHERE universe != 'all'
              )
            """
        )
        orphans = [r[0] for r in cur.fetchall()]
        logging.info("Symbols whose only row is 'all' (will be promoted to 'manual'): %d", len(orphans))
        if orphans:
            for sym in orphans[:20]:
                logging.info("  orphan: %s", sym)
            if len(orphans) > 20:
                logging.info("  ... (%d more)", len(orphans) - 20)

        # 2. Count total 'all' rows.
        cur.execute("SELECT count(*) FROM screener_cache WHERE universe = 'all'")
        all_count = cur.fetchone()[0]
        to_delete = all_count - len(orphans)
        logging.info("'all' rows: %d total, %d will be deleted, %d promoted.", all_count, to_delete, len(orphans))

        if args.dry_run:
            logging.info("Dry-run: no changes made.")
            return 0

        # 3. Promote orphans: rewrite their universe from 'all' to 'manual'.
        # Composite PK is (symbol, universe), so we can just UPDATE.
        for sym in orphans:
            cur.execute(
                "UPDATE screener_cache SET universe = 'manual' WHERE symbol = ? AND universe = 'all'",
                (sym,),
            )

        # 4. Delete the remaining 'all' rows.
        cur.execute("DELETE FROM screener_cache WHERE universe = 'all'")
        deleted = cur.rowcount

        conn.commit()
        logging.info("Promoted %d orphan symbols to 'manual'.", len(orphans))
        logging.info("Deleted %d 'all' rows.", deleted)

        # 5. Final state.
        cur.execute("SELECT count(*) FROM screener_cache")
        total = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT symbol) FROM screener_cache")
        distinct = cur.fetchone()[0]
        logging.info("Final: %d rows, %d distinct symbols.", total, distinct)
        return 0
    finally:
        try:
            conn.close()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
