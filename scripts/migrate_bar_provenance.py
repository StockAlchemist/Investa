#!/usr/bin/env python3
"""Record where a price bar came from (plan Phase 5.4, the part worth doing).

`daily_fx` has carried a `source` column since the ECB became a second rate
feed. `daily_ohlcv` never has, and until recently the argument against it was
sound: with one provider it would write the constant `'yahoo'` 20.6 million
times.

That argument expired. **34,120 bars have been rewritten from a second
provider** — Tiingo, via `ingest_tiingo_reference.py` and
`repair_bars_against_reference.py` — and nothing in the archive records which
ones. A bar corrected against an independent reference and a bar Yahoo served
are now indistinguishable, so "which prices did we change, and on whose word?"
has no answer.

Adding the column is cheap: SQLite stores a constant default as table metadata
rather than rewriting every row, so this is fast even at 20 million rows.

    python scripts/migrate_bar_provenance.py [--dry-run] [--db PATH]

**Labelling the repairs already applied.** Provenance arrived after the fact, so
the bars that most need it are the ones already changed. `--label-from` recovers
that by diffing against a pre-repair backup: any bar whose close differs is one
the repair rewrote. This is exact rather than inferred, and only possible while
those backups still exist — `repair_bars_against_reference.py` takes one before
every run, and they rotate.

    python scripts/migrate_bar_provenance.py \\
        --label-from data/backups/market_data_pre_barfix_20260826_223941.db \\
        --label tiingo
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

ADD_COLUMN = "ALTER TABLE daily_ohlcv ADD COLUMN source TEXT DEFAULT 'yahoo'"


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def needs_column(conn: sqlite3.Connection) -> bool:
    tables = {
        r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    if "daily_ohlcv" not in tables:
        return False
    return "source" not in {r[1] for r in conn.execute("PRAGMA table_info(daily_ohlcv)")}


def label_repaired(conn: sqlite3.Connection, backup: str, label: str, apply: bool) -> int:
    """Mark every bar that differs from `backup` — i.e. every bar a repair wrote.

    Restricted to symbols that actually have a reference on record, so an
    unrelated difference (a nightly delta that ran between the backup and now)
    cannot be mislabelled as an adjudicated repair.
    """
    symbols = [
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT symbol FROM reference_price WHERE source = ?", (label,)
        )
    ]
    if not symbols:
        print(f"  No symbols carry a '{label}' reference — nothing to label.")
        return 0

    conn.execute("ATTACH DATABASE ? AS pre", (backup,))
    try:
        changed = 0
        for symbol in symbols:
            rows = conn.execute(
                """
                SELECT d.date FROM daily_ohlcv d
                JOIN pre.daily_ohlcv p
                  ON p.symbol = d.symbol AND p.date = d.date AND p.interval = d.interval
                WHERE d.symbol = ? AND d.interval = '1d'
                  AND d.close IS NOT NULL AND p.close IS NOT NULL
                  AND abs(d.close - p.close) > 1e-9
                """,
                (symbol,),
            ).fetchall()
            if not rows:
                continue
            changed += len(rows)
            if apply:
                conn.executemany(
                    "UPDATE daily_ohlcv SET source = ? WHERE symbol = ? AND date = ? "
                    "AND interval = '1d'",
                    [(label, symbol, r[0]) for r in rows],
                )
        if apply:
            conn.commit()
        return changed
    finally:
        conn.execute("DETACH DATABASE pre")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--label-from",
        default=None,
        help="a pre-repair backup; bars differing from it are labelled as repaired",
    )
    parser.add_argument(
        "--label", default="tiingo", help="source to record for those bars"
    )
    args = parser.parse_args()

    path = args.db or default_db_path()
    if not os.path.exists(path):
        print(f"No market database at {path} — nothing to migrate.")
        return 0

    conn = sqlite3.connect(path, timeout=120.0)
    try:
        if needs_column(conn):
            print(f"{path}\n  - add daily_ohlcv.source (existing rows -> 'yahoo')")
            if args.dry_run:
                print("\nDry run — nothing written.")
                return 0
            conn.execute(ADD_COLUMN)
            conn.commit()
            print("  applied.")
        else:
            print("daily_ohlcv already records provenance.")

        if args.label_from:
            if not os.path.exists(args.label_from):
                print(f"No backup at {args.label_from}.")
                return 1
            print(f"\nLabelling bars that differ from {os.path.basename(args.label_from)}:")
            n = label_repaired(conn, args.label_from, args.label, not args.dry_run)
            verb = "would label" if args.dry_run else "labelled"
            print(f"  {verb} {n} bar(s) as source='{args.label}'.")

        counts = dict(
            conn.execute(
                "SELECT COALESCE(source, 'yahoo'), COUNT(*) FROM daily_ohlcv GROUP BY 1"
            )
        )
        print("\nProvenance now:")
        for src, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {src:12} {n:,}")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    sys.exit(main())
