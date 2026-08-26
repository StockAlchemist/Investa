#!/usr/bin/env python3
"""Snapshot the market archive (plan Phase 2).

The archive is `.gitignore`d, lives on one disk, and has no backup. That was
survivable while everything in it could be re-downloaded. It no longer can:
Yahoo now serves BANPU.BK only from 2016-10-28 while the archive holds it from
2000, so roughly sixteen years of that series exist here and nowhere upstream.
"Rebuild it from Yahoo" is not a recovery plan any more — a rebuild would
silently truncate.

Three tiers, because most of the archive is disposable, the part that is not is
small, and what changes in a night is smaller still:

  incremental  the small tables in full, plus the last N days of bars. This is
               the daily shape. It carries exactly what a nightly delta writes,
               so it stays small no matter how large the archive grows.
  core         everything except intraday_ohlcv, plus the static price tables.
               Intraday is most of the file and regenerates on demand.
  full         the whole database, intraday included.

Sizes move with the archive, and that is the point of having three. Measured as
the archive grew:

                 Tier A          Tier B           Tier C
    core         20 MB / 13 s    218 MB / 2 min   ~700 MB / several min
    full         68 MB / 39 s    —                ~2.5 GB

`core` was introduced as the tier "small enough to push daily". At 700 MB that
is no longer true, and a shorter retention window does not fix it — hence
`incremental`, which is bounded by the delta rather than by the archive.

What incremental does NOT carry: a repair to an old bar. Those are rare and the
weekly `full` covers them, but a restore chain of incrementals alone is not a
complete archive. Restore from the newest full or core, then apply what came
after.

`VACUUM INTO` is used rather than a file copy: it takes a consistent snapshot
without stopping writers, and compacts as it goes. Every snapshot is then
verified with PRAGMA integrity_check *before* it is allowed to displace an
older one, so a corrupt copy can never rotate out a good one.

    python scripts/backup_market_archive.py --mode incremental   # daily
    python scripts/backup_market_archive.py                      # core, weekly
    python scripts/backup_market_archive.py --mode full          # monthly
    python scripts/backup_market_archive.py --dest /cloud/folder

The live database must stay on local disk. `db_utils.is_path_on_cloud_drive`
degrades a synced path to journal_mode=DELETE, and two processes writing a
SQLite file through a sync client is how databases get corrupted. Snapshots
sync; the live file does not. Point --dest (or INVESTA_BACKUP_DIR) at the
synced folder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import tarfile
import tempfile
from datetime import date, datetime, timedelta, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

# Dropped from a core snapshot. Regenerated on demand and 77% of the file.
DISPOSABLE_TABLES = ("intraday_ohlcv",)

# Small directories worth carrying alongside the database. static_prices holds
# the frozen tables for delisted tickers (RIMM, KFT, AAUKY, BECL.BK, BML.BK) —
# no feed will ever serve those again, so they are the most irreplaceable bytes
# in the whole backup despite being under a megabyte.
EXTRA_DIRS = ("static_prices",)

# Retention per mode, sized against what each snapshot actually costs. A month
# of dailies is cheap because incremental is bounded by the delta; core and full
# are kept shallow because they are not.

DEFAULT_RETAIN = {"incremental": 30, "core": 4, "full": 3}

# Days of daily bars an incremental snapshot carries. Wider than the nightly
# delta's window so a missed night is still covered by the next snapshot.
INCREMENTAL_DAYS = 14


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def default_dest() -> str:
    return os.environ.get("INVESTA_BACKUP_DIR") or os.path.join(
        config.get_app_data_dir(), "backups"
    )


def table_counts(db_path: str) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' "
                "AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {n: conn.execute(f'SELECT COUNT(*) FROM "{n}"').fetchone()[0] for n in names}
    finally:
        conn.close()


def snapshot(db_path: str, target: str, mode: str) -> dict:
    """VACUUM INTO a temp file, strip disposables for core, verify, report."""
    conn = sqlite3.connect(db_path, timeout=120.0)
    try:
        conn.execute("VACUUM INTO ?", (target,))
    finally:
        conn.close()

    if mode in ("core", "incremental"):
        stripped = sqlite3.connect(target, timeout=120.0)
        try:
            for table in DISPOSABLE_TABLES:
                stripped.execute(f'DROP TABLE IF EXISTS "{table}"')
            if mode == "incremental":
                # Keep every small table whole — corporate_action, fund_nav,
                # share_count and sync_metadata are the irreplaceable part and
                # are tiny. Only the bar tables are trimmed to the window.
                cutoff = (date.today() - timedelta(days=INCREMENTAL_DAYS)).isoformat()
                stripped.execute("DELETE FROM daily_ohlcv WHERE date < ?", (cutoff,))
                try:
                    stripped.execute("DELETE FROM daily_fx WHERE date < ?", (cutoff,))
                except sqlite3.OperationalError:
                    pass
            stripped.commit()
            # Reclaim the pages the dropped table was using, or the "core"
            # snapshot is the same size as the full one.
            stripped.execute("VACUUM")
        finally:
            stripped.close()

    # Verify before this copy is allowed to displace anything.
    check = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    try:
        result = check.execute("PRAGMA integrity_check").fetchone()
    finally:
        check.close()
    if not result or result[0] != "ok":
        raise RuntimeError(f"integrity_check failed on the snapshot: {result}")

    return {"tables": table_counts(target), "bytes": os.path.getsize(target)}


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_archive(db_path: str, dest: str, mode: str, stamp: str) -> tuple[str, dict]:
    os.makedirs(dest, exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="investa_backup_")
    try:
        snap_path = os.path.join(workdir, "market_data.db")
        info = snapshot(db_path, snap_path, mode)

        data_dir = config.get_app_data_dir()
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "mode": mode,
            "source": db_path,
            "snapshot_bytes": info["bytes"],
            "tables": info["tables"],
            "excluded_tables": list(DISPOSABLE_TABLES) if mode in ("core", "incremental") else [],
            "bar_window_days": INCREMENTAL_DAYS if mode == "incremental" else None,
            "sha256_snapshot": sha256_of(snap_path),
        }
        with open(os.path.join(workdir, "manifest.json"), "w") as fh:
            json.dump(manifest, fh, indent=1, sort_keys=True)

        out_path = os.path.join(dest, f"market_archive_{mode}_{stamp}.tar.gz")
        with tarfile.open(out_path, "w:gz") as tar:
            tar.add(snap_path, arcname="market_data.db")
            tar.add(os.path.join(workdir, "manifest.json"), arcname="manifest.json")
            for extra in EXTRA_DIRS:
                path = os.path.join(data_dir, extra)
                if os.path.isdir(path):
                    tar.add(path, arcname=extra)
        return out_path, manifest
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def rotate(dest: str, mode: str, retain: int, dry_run: bool) -> list[str]:
    """Drop the oldest snapshots beyond `retain`. Never removes the last one."""
    prefix = f"market_archive_{mode}_"
    existing = sorted(
        f for f in os.listdir(dest) if f.startswith(prefix) and f.endswith(".tar.gz")
    )
    if len(existing) <= max(retain, 1):
        return []

    doomed = existing[: len(existing) - retain]
    for name in doomed:
        if not dry_run:
            os.remove(os.path.join(dest, name))
    return doomed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--mode",
        choices=("incremental", "core", "full"),
        default="core",
        help="incremental = small tables + the last %d days of bars (daily); "
        "core = everything but intraday (weekly); full = the lot (monthly)"
        % INCREMENTAL_DAYS,
    )
    parser.add_argument("--dest", default=None, help="output directory (env INVESTA_BACKUP_DIR)")
    parser.add_argument("--retain", type=int, default=None, help="snapshots to keep for this mode")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    db_path = args.db or default_db_path()
    if not os.path.exists(db_path):
        print(f"No market database at {db_path}.")
        return 1

    dest = args.dest or default_dest()
    retain = args.retain if args.retain is not None else DEFAULT_RETAIN[args.mode]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"source : {db_path} ({os.path.getsize(db_path) / 1e6:.0f} MB)")
    print(f"dest   : {dest}")
    print(f"mode   : {args.mode} (retain {retain})")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    try:
        out_path, manifest = build_archive(db_path, dest, args.mode, stamp)
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        return 1

    size = os.path.getsize(out_path)
    print(f"\nwrote {out_path}")
    print(f"      {size / 1e6:.1f} MB compressed, from {manifest['snapshot_bytes'] / 1e6:.0f} MB")
    print("      integrity_check ok")
    for table, count in sorted(manifest["tables"].items()):
        print(f"        {table:20} {count:>10,}")
    if manifest["excluded_tables"]:
        print(f"      excluded: {', '.join(manifest['excluded_tables'])}")

    dropped = rotate(dest, args.mode, retain, args.dry_run)
    if dropped:
        print(f"\nrotated out {len(dropped)}: {', '.join(dropped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
