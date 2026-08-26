#!/usr/bin/env python3
"""One entry point for keeping the archive current, on a machine that sleeps.

A cron line assumes the computer is awake at 02:30. This one runs on a laptop,
so the schedule is the wrong thing to reason about — what matters is how far
behind each job has fallen, whenever it happens to get a chance. Every step here
decides for itself whether it is due, which makes the whole thing safe to run
after an hour or after three weeks, by hand or on wake, twice in a row.

Steps, each skipped when it is not due:

  1. price delta      when the newest bar is older than the last trading day
  2. split check      whenever prices moved (a new bar can carry a new split)
  3. official FX      when the newest ECB-sourced rate is behind the last trading day
  4. incremental snap when the newest one is older than SNAPSHOT_MAX_AGE_HOURS
  5. core snapshot    when the newest one is older than CORE_MAX_AGE_DAYS

The delta window widens with the gap: away for a fortnight, it asks for a
fortnight plus an overlap rather than the usual five days. That overlap is what
`check_integrity` compares against, so it is never dropped entirely.

Nothing here repairs anything. Step 2 reports and exits non-zero on a *new*
finding; repair needs an independent reference and stays deliberate.

    python scripts/archive_maintenance.py            # do what is due
    python scripts/archive_maintenance.py --status   # say what is due, do nothing
    python scripts/archive_maintenance.py --force    # do everything regardless

Wire it to launchd rather than cron: a LaunchAgent with StartCalendarInterval
runs a missed job when the machine next wakes, coalescing several missed firings
into one, which is precisely the behaviour a laptop needs. See
`docs/plans/local_market_data_archive.md` for the plist.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(REPO, "src"))

import config  # noqa: E402
from db_utils import connect_readonly  # noqa: E402

# An incremental snapshot is ~7 MB and takes half a minute, so it can be taken
# often; the point is that the irreplaceable tables are never far behind.
SNAPSHOT_MAX_AGE_HOURS = 20
CORE_MAX_AGE_DAYS = 7

# Overlap added to whatever gap is being closed. check_integrity compares the
# overlap, so this is how a provider re-basing a series gets noticed.
DELTA_OVERLAP_DAYS = 5
DELTA_MAX_DAYS = 90

# The ECB's 90-day file is 70 KB against ~640 KB for the whole history since
# 1999, so a routine run reads the short one. Past this gap the short file would
# leave a hole behind it, and the full history is worth the download.
FX_RECENT_WINDOW_DAYS = 60


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def backup_dir() -> str:
    return os.environ.get("INVESTA_BACKUP_DIR") or os.path.join(
        config.get_app_data_dir(), "backups"
    )


def offsite_dir() -> Optional[str]:
    """Where the off-machine copy goes, if one is configured.

    Only the incremental snapshot is sent there, and that is the whole point of
    having three tiers. Incremental carries every small table whole — the
    corporate actions, fund NAVs and share counts that cannot be re-downloaded —
    plus a fortnight of bars, in about 7 MB. Core and full are 575 MB and ~700 MB
    and exist for a fast local restore; pushing them too would cost 4.6 GB of
    someone's Drive quota to duplicate history that Yahoo will still serve.
    """
    return os.environ.get("INVESTA_OFFSITE_DIR") or None


def newest_bar() -> Optional[str]:
    if not os.path.exists(db_path()):
        return None
    conn = connect_readonly(db_path())
    try:
        row = conn.execute(
            "SELECT MAX(date) FROM daily_ohlcv WHERE interval = '1d'"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def newest_official_fx() -> Optional[str]:
    """The newest ECB-sourced rate in the archive, or None if there is none.

    Keyed on the source and not on `MAX(date)`: the nightly price delta writes
    Yahoo rates into the same table, so the table as a whole is always current
    and would say the official feed is up to date when it has never run.
    """
    if not os.path.exists(db_path()):
        return None
    conn = connect_readonly(db_path())
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(daily_fx)")}
        if "source" not in columns:
            return None  # pre-provenance archive: treat as never filled
        row = conn.execute(
            "SELECT MAX(date) FROM daily_fx WHERE source = 'ecb'"
        ).fetchone()
    finally:
        conn.close()
    return row[0] if row else None


def last_trading_day() -> date:
    """Yesterday's market date, or the Friday before a weekend."""
    try:
        from utils_time import get_est_today

        today = get_est_today()
    except Exception:
        today = date.today()
    day = today - timedelta(days=1)
    while day.weekday() >= 5:  # Sat/Sun
        day -= timedelta(days=1)
    return day


def newest_snapshot_age(mode: str, directory: Optional[str] = None) -> Optional[float]:
    """Hours since the newest snapshot of this mode, or None if there is none."""
    directory = directory or backup_dir()
    if not os.path.isdir(directory):
        return None
    prefix = f"market_archive_{mode}_"
    stamps = [
        os.path.getmtime(os.path.join(directory, f))
        for f in os.listdir(directory)
        if f.startswith(prefix) and f.endswith(".tar.gz")
    ]
    if not stamps:
        return None
    return (datetime.now().timestamp() - max(stamps)) / 3600.0


def run(label: str, args: List[str], dry: bool) -> int:
    print(f"\n--- {label}")
    if dry:
        print(f"    would run: {' '.join(args)}")
        return 0
    result = subprocess.run(args, cwd=REPO)
    print(f"    exit {result.returncode}")
    return result.returncode


def plan(force: bool) -> Tuple[List[Tuple[str, List[str]]], List[str]]:
    """(jobs to run, reasons for what is skipped)."""
    jobs: List[Tuple[str, List[str]]] = []
    skipped: List[str] = []
    python = sys.executable

    newest = newest_bar()
    target = last_trading_day().isoformat()
    prices_due = force or not newest or newest < target

    if prices_due:
        if newest:
            gap = (date.fromisoformat(target) - date.fromisoformat(newest)).days
        else:
            gap = DELTA_OVERLAP_DAYS
        days = min(max(gap + DELTA_OVERLAP_DAYS, DELTA_OVERLAP_DAYS), DELTA_MAX_DAYS)
        jobs.append(
            (
                f"price delta ({days}-day window; newest bar {newest}, want {target})",
                [python, "scripts/backfill_market_history.py", "--tier", "A",
                 "--days", str(days), "--apply"],
            )
        )
        jobs.append(
            ("split-consistency check", [python, "scripts/check_split_consistency.py"])
        )
    else:
        skipped.append(f"prices are current (newest bar {newest})")
        skipped.append("split check — runs only when prices moved")

    # Official FX. Cheap enough to be due most days: the 90-day file is 70 KB,
    # and it fills only days Yahoo left empty, so a run that finds nothing to do
    # costs one request. The point is that the currency conversion behind every
    # portfolio figure stops depending on the price feed staying up — EUR, GBP
    # and CNY sat frozen for two months the last time it did not.
    newest_fx = newest_official_fx()
    if force or not newest_fx or newest_fx < target:
        argv = [python, "scripts/backfill_fx_rates.py", "--apply"]
        gap = (
            DELTA_MAX_DAYS
            if not newest_fx
            else (date.fromisoformat(target) - date.fromisoformat(newest_fx)).days
        )
        if gap <= FX_RECENT_WINDOW_DAYS:
            argv.append("--recent")
        jobs.append(
            (
                f"official FX from the ECB (newest ECB rate {newest_fx or 'none'}, "
                f"want {target})",
                argv,
            )
        )
    else:
        skipped.append(f"official FX is current (newest ECB rate {newest_fx})")

    offsite = offsite_dir()
    inc_age = newest_snapshot_age("incremental", offsite)
    if force or inc_age is None or inc_age > SNAPSHOT_MAX_AGE_HOURS:
        age = "none yet" if inc_age is None else f"{inc_age:.0f}h old"
        where = f" -> {offsite}" if offsite else ""
        argv = [python, "scripts/backup_market_archive.py", "--mode", "incremental"]
        if offsite:
            argv += ["--dest", offsite]
        jobs.append((f"incremental snapshot ({age}){where}", argv))
    else:
        skipped.append(f"incremental snapshot is {inc_age:.0f}h old")

    core_age = newest_snapshot_age("core")
    if force or core_age is None or core_age > CORE_MAX_AGE_DAYS * 24:
        age = "none yet" if core_age is None else f"{core_age / 24:.1f}d old"
        jobs.append(
            (f"core snapshot ({age})",
             [python, "scripts/backup_market_archive.py", "--mode", "core"])
        )
    else:
        skipped.append(f"core snapshot is {core_age / 24:.1f}d old")

    return jobs, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--status", action="store_true", help="report what is due, run nothing")
    parser.add_argument("--force", action="store_true", help="run every step regardless")
    args = parser.parse_args()

    print(f"Archive maintenance — {datetime.now().astimezone():%Y-%m-%d %H:%M %Z}")
    jobs, skipped = plan(args.force)

    for reason in skipped:
        print(f"  skip: {reason}")
    if not jobs:
        print("\nNothing due.")
        return 0

    print(f"\n{len(jobs)} job(s) due:")
    for label, _ in jobs:
        print(f"  - {label}")

    if args.status:
        return 0

    # A new split finding is worth surfacing but must not stop the snapshot that
    # would preserve the very data it is complaining about.
    findings = False
    failed = []
    for label, argv in jobs:
        code = run(label, argv, dry=False)
        if code and "split-consistency" in label:
            findings = True
        elif code:
            failed.append(label)

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        return 1
    if findings:
        print("New split-consistency findings above — see the check's output.")
        return 2
    print("All due jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
