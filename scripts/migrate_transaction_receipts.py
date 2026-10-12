#!/usr/bin/env python3
"""Add transaction receipts offline. Stop the backend before applying.

python scripts/migrate_transaction_receipts.py --dry-run
python scripts/migrate_transaction_receipts.py --db /path/to/portfolio.db
With no --db, migrates all existing per-user portfolio databases.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import config  # noqa: E402
from server.transaction_writes import RECEIPTS_SCHEMA  # noqa: E402


def migrate(path: Path, dry_run: bool = False):
    if not path.is_file():
        raise FileNotFoundError(path)
    with sqlite3.connect(f"file:{path}?mode=rw", uri=True) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name='transaction_receipts'"
        ).fetchone()
        if exists:
            return False
        if not dry_run:
            # A recoverable SQLite snapshot, including any committed WAL pages.
            backup = path.with_name(path.name + ".before-receipts.bak")
            if backup.exists():
                raise FileExistsError(backup)
            with sqlite3.connect(backup) as destination:
                conn.backup(destination)
            conn.execute(RECEIPTS_SCHEMA)
        return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    paths = (
        [args.db]
        if args.db
        else sorted(
            (Path(config.get_app_data_dir()) / config.USERS_DIR).glob(
                f"*/{config.PORTFOLIO_DB_FILENAME}"
            )
        )
    )
    for path in paths:
        changed = migrate(path, args.dry_run)
        print(
            f"{path}: {'would migrate' if args.dry_run else 'migrated'}"
            if changed
            else f"{path}: current"
        )


if __name__ == "__main__":
    main()
