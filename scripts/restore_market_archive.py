#!/usr/bin/env python3
"""Restore the market archive from a snapshot (plan Phase 2.4).

The other half of a backup. An untested restore path is not a backup, it is a
belief about one, so this ships alongside `backup_market_archive.py` and is
meant to be exercised.

    python scripts/restore_market_archive.py --list
    python scripts/restore_market_archive.py --latest
    python scripts/restore_market_archive.py --from <file.tar.gz>

Restoring onto an existing database is refused unless --force, and the current
database is always copied aside first — a restore that silently overwrites live
data is a worse failure than the one it is recovering from.

Snapshots are not all equivalent, and restoring the wrong one over a populated
database throws away what it never carried:

  full         everything.
  core         no intraday_ohlcv. Harmless on a cold start — intraday
               regenerates on demand — but it would drop existing intraday.
  incremental  the small tables in full plus a short window of bars. A
               supplement to a core/full restore, never a substitute: restoring
               one alone leaves an archive with two weeks of price history.

Each case is called out rather than silently performed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def default_src() -> str:
    return os.environ.get("INVESTA_BACKUP_DIR") or os.path.join(
        config.get_app_data_dir(), "backups"
    )


def list_snapshots(src: str) -> list[str]:
    if not os.path.isdir(src):
        return []
    return sorted(
        (
            os.path.join(src, f)
            for f in os.listdir(src)
            if f.startswith("market_archive_") and f.endswith(".tar.gz")
        ),
        reverse=True,
    )


def read_manifest(path: str) -> dict:
    with tarfile.open(path, "r:gz") as tar:
        member = tar.extractfile("manifest.json")
        if member is None:
            return {}
        return json.load(member)


def describe(path: str) -> str:
    try:
        m = read_manifest(path)
    except Exception as exc:  # noqa: BLE001
        return f"{os.path.basename(path):48} (unreadable: {exc})"
    tables = m.get("tables", {})
    rows = sum(tables.values()) if tables else 0
    size = os.path.getsize(path) / 1e6
    return (
        f"{os.path.basename(path):48} {m.get('mode','?'):5} "
        f"{size:6.1f} MB  {rows:>10,} rows  {str(m.get('created_at'))[:19]}"
    )


def restore(archive: str, db_path: str, force: bool) -> int:
    manifest = read_manifest(archive)
    mode = manifest.get("mode", "?")
    print(f"snapshot : {os.path.basename(archive)} ({mode})")
    for table, count in sorted((manifest.get("tables") or {}).items()):
        print(f"             {table:20} {count:>10,}")

    exists = os.path.exists(db_path)
    if exists and not force:
        print(
            f"\n{db_path} already exists.\n"
            "Refusing to overwrite a live database without --force."
        )
        if mode == "core":
            print(
                "Note: this is a CORE snapshot and carries no intraday_ohlcv — "
                "restoring it would drop whatever intraday history the live "
                "database holds."
            )
        elif mode == "incremental":
            window = manifest.get("bar_window_days")
            print(
                f"Note: this is an INCREMENTAL snapshot. It carries the small "
                f"tables in full but only the last {window} days of bars, so it "
                "is a supplement to a core/full restore, never a substitute for "
                "one."
            )
        return 1

    workdir = tempfile.mkdtemp(prefix="investa_restore_")
    try:
        with tarfile.open(archive, "r:gz") as tar:
            # Only take what we put there; never trust paths from an archive.
            for name in ("market_data.db", "manifest.json"):
                try:
                    tar.extract(name, path=workdir, filter="data")
                except KeyError:
                    if name == "market_data.db":
                        print(f"\n{archive} contains no market_data.db")
                        return 1

        candidate = os.path.join(workdir, "market_data.db")
        conn = sqlite3.connect(f"file:{candidate}?mode=ro", uri=True)
        try:
            result = conn.execute("PRAGMA integrity_check").fetchone()
        finally:
            conn.close()
        if not result or result[0] != "ok":
            print(f"\nintegrity_check FAILED on the snapshot: {result}")
            print("Refusing to install a corrupt database.")
            return 1
        print("\nintegrity_check ok")

        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        if exists:
            aside = f"{db_path}.replaced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            shutil.copy2(db_path, aside)
            print(f"existing database copied aside -> {aside}")
            # -wal / -shm belong to the old file; leaving them beside a
            # different database is how you get a confusing corruption report.
            for suffix in ("-wal", "-shm"):
                stale = db_path + suffix
                if os.path.exists(stale):
                    os.remove(stale)

        shutil.move(candidate, db_path)

        # VACUUM INTO writes a `delete`-journal database, so a restored file
        # arrives without WAL however the original was configured. That is not
        # cosmetic: under DELETE journalling a single writer blocks every
        # reader, and the first restore here left the ranking worker locking the
        # whole archive out of every other process. Put it back.
        installed = sqlite3.connect(db_path, timeout=60.0)
        try:
            mode = installed.execute("PRAGMA journal_mode=WAL").fetchone()[0]
            print(f"restored -> {db_path} (journal_mode={mode})")
            if mode != "wal":
                print(
                    "  WARNING: could not enable WAL. On a cloud-synced path this "
                    "is expected and correct; on local disk it means concurrent "
                    "readers will block behind any writer."
                )
        finally:
            installed.close()
        return 0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--src", default=None, help="snapshot directory (env INVESTA_BACKUP_DIR)")
    parser.add_argument("--db", default=None, help="database to restore to")
    parser.add_argument("--list", action="store_true", help="show available snapshots")
    parser.add_argument("--latest", action="store_true", help="restore the newest snapshot")
    parser.add_argument("--from", dest="archive", default=None, help="restore this file")
    parser.add_argument("--force", action="store_true", help="overwrite an existing database")
    args = parser.parse_args()

    src = args.src or default_src()
    db_path = args.db or default_db_path()

    if args.list or not (args.latest or args.archive):
        snapshots = list_snapshots(src)
        if not snapshots:
            print(f"No snapshots in {src}.")
            print("Create one: python scripts/backup_market_archive.py")
            return 1
        print(f"{len(snapshots)} snapshot(s) in {src}:\n")
        for path in snapshots:
            print("  " + describe(path))
        if not args.list:
            print("\nPass --latest or --from <file> to restore.")
        return 0

    archive = args.archive
    if args.latest:
        snapshots = list_snapshots(src)
        if not snapshots:
            print(f"No snapshots in {src}.")
            return 1
        archive = snapshots[0]

    if not archive or not os.path.exists(archive):
        print(f"No such snapshot: {archive}")
        return 1

    return restore(archive, db_path, args.force)


if __name__ == "__main__":
    raise SystemExit(main())
