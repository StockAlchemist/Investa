#!/usr/bin/env python3
"""Detect — and, carefully, rebase — price series a provider never back-adjusted.

**Status: diagnostic only. The repair was run against the live archive on
26 Aug 2026, measured, and reverted. Do not --apply without reading this.**

A split should be invisible in a historical series: the provider divides every
earlier price by the ratio so the line stays continuous. Sometimes it does not.
WLFC is the clean example — Yahoo serves 191.65 on 2026-07-17 and 63.12 on
07-20, records the 3:1 on 07-21, and leaves everything before it at three times
the current basis. It had been correctly adjusted as recently as 28 July, so
this is the provider revising its own history.

Two rounds of narrowing, both worth keeping:

  * A first pass flagged 586 of 5,022 splits. 515 of those had ratios inside
    1.1x — ordinary volatility that happens to match a small ratio. Repairing
    them would have rebased five held symbols (IBM, VZ, SPGI, BHP, TRUE.BK) on
    the strength of one normal session. Hence MIN_DEVIATION.
  * At 1.4x the list falls to 24 genuine-looking seams, 40,651 bars.

Those 24 were repaired and the result compared against the pre-repair database,
symbol by symbol, using the verifier as the only objective measure:

    improved   3   (WLFC 2->1, MNST 6->5, BESS 16->15)
    unchanged  2
    WORSE     19   (SRXH 6->73, WFCF 1->12, LINK 5->21, MYSZ 0->9, ...)

So the repair was reverted. The arithmetic is right — WLFC became continuous at
63.88 -> 63.12 -> 65.35 — but it is the wrong tool for most of this list. These
are reverse-split micro-caps with ratios like 0.0125 and 0.001, whose pre-split
history is sub-cent quotes with enormous relative tick noise. Multiplying that
stretch by 80 or 1000 to put it on today's basis is *correct* and still makes
things worse to read: the noise crosses the verifier's price floor and the
symbol looks more broken, not less.

Two further traps found while doing it, both now guarded:

  * SMXT had a single bar inside its pre-split stretch that the provider had
    already adjusted. Rebasing the whole stretch pushed that one bar twelve
    times too high. `_run_is_mixed` now refuses a symbol whose pre-seam run
    already contains post-seam bars.
  * The seam and the recorded ex-date disagree by one to three days for about
    5% of splits, so the repair uses the seam and moves the action to match.

What would make this shippable: judge success per symbol rather than in
aggregate, keep only the symbols that measurably improve, and leave the
penny-stock reverse splits alone entirely. The detection half is sound and
useful on its own — run it without --apply.

    python scripts/repair_unadjusted_splits.py            # report only
    python scripts/repair_unadjusted_splits.py --symbol WLFC
"""

from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from datetime import datetime
from typing import List, NamedTuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

# How close the observed seam must be to the split ratio to call it unadjusted.
# Generous, because a split day also carries ordinary trading: the price moves
# by the ratio *and* by whatever the market did that session.
RATIO_TOLERANCE = 0.06

# How far from 1 a ratio must be before a matching price move can be attributed
# to the split rather than to the market.
#
# This is the whole safety of the repair. With a 6% tolerance, a ratio of 1.046
# matches any day the stock moved about 4.6% — which is an unremarkable session.
# A first pass at 0.5% "detected" 586 seams, of which 515 had ratios inside
# 1.1x: ordinary volatility, and repairing them would have rebased five held
# symbols (IBM, VZ, SPGI, BHP, TRUE.BK) on the strength of one normal day's
# trading. Every one of those ratios is a spin-off or stock dividend between
# 1.015 and 1.125.
#
# 1.4 is the same line the verifier draws: above ordinary volatility, below the
# smallest split anyone actually runs (1.5:1). Real stock dividends below it do
# exist and are left alone — a missed repair is recoverable, a wrong one
# silently rescales a history.
MIN_DEVIATION = 1.4

# Days either side of the recorded ex-date to look for the price seam.
SEARCH_BEFORE, SEARCH_AFTER = 6, 4


class Seam(NamedTuple):
    symbol: str
    recorded_date: str
    seam_date: str
    ratio: float
    observed: float
    rows_before: int


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def detect(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    min_deviation: float = MIN_DEVIATION,
) -> List[Seam]:
    """Recorded splits whose price series still shows the pre-split basis."""
    query = "SELECT symbol, date, value FROM corporate_action WHERE kind = 'split' AND value > 0"
    params: list = []
    if symbol:
        query += " AND symbol = ?"
        params.append(symbol)

    found: List[Seam] = []
    mixed: List[str] = []
    for sym, recorded, ratio in conn.execute(query, params):
        if not ratio or max(ratio, 1.0 / ratio) < min_deviation:
            continue
        window = conn.execute(
            """
            SELECT date, close FROM daily_ohlcv
            WHERE symbol = ? AND interval = '1d' AND close > 0
              AND date BETWEEN date(?, ?) AND date(?, ?)
            ORDER BY date
            """,
            (sym, recorded, f"-{SEARCH_BEFORE} day", recorded, f"+{SEARCH_AFTER} day"),
        ).fetchall()

        for (_, c1), (d2, c2) in zip(window, window[1:]):
            observed = c1 / c2
            if abs(observed - ratio) / ratio < RATIO_TOLERANCE:
                if _run_is_mixed(conn, sym, d2, ratio):
                    mixed.append(sym)
                    break
                rows_before = conn.execute(
                    "SELECT COUNT(*) FROM daily_ohlcv WHERE symbol = ? "
                    "AND interval = '1d' AND date < ?",
                    (sym, d2),
                ).fetchone()[0]
                found.append(Seam(sym, recorded, d2, float(ratio), observed, rows_before))
                break

    if mixed:
        logging.warning(
            f"Skipped {len(mixed)} symbol(s) whose pre-seam run already contains "
            f"post-seam bars: {', '.join(sorted(set(mixed))[:10])}"
        )
    return found


def _run_is_mixed(
    conn: sqlite3.Connection, symbol: str, seam_date: str, ratio: float
) -> bool:
    """
    True when the stretch before `seam_date` is not uniformly on the old basis.

    Compares the last 20 pre-seam bars against the first post-seam close: any of
    them already within a few percent of it has been adjusted, and rebasing the
    run would move it the wrong way by the full ratio.
    """
    after = conn.execute(
        "SELECT close FROM daily_ohlcv WHERE symbol = ? AND interval = '1d' "
        "AND date >= ? AND close > 0 ORDER BY date LIMIT 1",
        (symbol, seam_date),
    ).fetchone()
    if not after:
        return False
    target = after[0]

    before = conn.execute(
        "SELECT close FROM daily_ohlcv WHERE symbol = ? AND interval = '1d' "
        "AND date < ? AND close > 0 ORDER BY date DESC LIMIT 20",
        (symbol, seam_date),
    ).fetchall()
    return any(abs(c - target) / target < 0.10 for (c,) in before if target)


def repair(conn: sqlite3.Connection, seams: List[Seam]) -> int:
    """Divide every bar before each seam by that seam's ratio."""
    touched = 0
    for seam in seams:
        conn.execute(
            """
            UPDATE daily_ohlcv
               SET open = open / :r, high = high / :r, low = low / :r,
                   close = close / :r,
                   adj_close = CASE WHEN adj_close IS NULL THEN NULL ELSE adj_close / :r END,
                   volume = CAST(volume * :r AS INTEGER)
             WHERE symbol = :s AND interval = '1d' AND date < :d
            """,
            {"r": seam.ratio, "s": seam.symbol, "d": seam.seam_date},
        )
        touched += seam.rows_before

        # Move the action to where the basis actually changes, so read-time
        # un-adjustment lines up with the repaired series.
        if seam.seam_date != seam.recorded_date:
            conn.execute(
                "UPDATE OR REPLACE corporate_action SET date = ?, source = ? "
                "WHERE symbol = ? AND date = ? AND kind = 'split'",
                (seam.seam_date, "yahoo+seam", seam.symbol, seam.recorded_date),
            )
    conn.commit()
    return touched


def backup(path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(
        os.path.dirname(os.path.dirname(path)), "backups", f"market_data_pre_split_repair_{stamp}.db"
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("VACUUM INTO ?", (target,))
    finally:
        conn.close()
    return target


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--symbol", default=None, help="check one symbol only")
    parser.add_argument(
        "--min-deviation",
        type=float,
        default=MIN_DEVIATION,
        help="how far from 1 a ratio must be to be repairable (default 1.4); "
        "below this a matching move cannot be told from ordinary trading",
    )
    parser.add_argument("--limit", type=int, default=25, help="rows to print")
    args = parser.parse_args()
    if args.dry_run:
        args.apply = False

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 1

    conn = sqlite3.connect(path, timeout=120.0)
    try:
        seams = detect(conn, args.symbol, args.min_deviation)
        if not seams:
            print("No unadjusted split seams found.")
            return 0

        seams.sort(key=lambda s: -max(s.ratio, 1.0 / s.ratio))
        rows = sum(s.rows_before for s in seams)
        symbols = len({s.symbol for s in seams})
        print(
            f"{len(seams)} unadjusted split(s) across {symbols} symbol(s); "
            f"{rows:,} bars sit on the wrong basis.\n"
        )
        print(f"  {'symbol':8} {'ratio':>10}  {'ex-date':11} {'seam':11} {'bars':>8}")
        for s in seams[: args.limit]:
            shift = "" if s.seam_date == s.recorded_date else "  (date moved)"
            print(
                f"  {s.symbol:8} {s.ratio:10.4f}  {s.recorded_date:11} "
                f"{s.seam_date:11} {s.rows_before:8,}{shift}"
            )
        if len(seams) > args.limit:
            print(f"  ... and {len(seams) - args.limit} more")

        if not args.apply:
            print("\nDry run — nothing written. Re-run with --apply to repair.")
            return 0

        saved = backup(path)
        print(f"\nBacked up to {saved}")

        touched = repair(conn, seams)
        print(f"Rebased {touched:,} bars across {symbols} symbol(s).")

        remaining = detect(conn, args.symbol, args.min_deviation)
        if remaining:
            print(f"\n{len(remaining)} seam(s) still detected:")
            for s in remaining[:10]:
                print(f"  {s.symbol} {s.seam_date} ratio {s.ratio}")
            return 1
        print("Verified: no unadjusted split seams remain.")
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
