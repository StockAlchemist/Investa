#!/usr/bin/env python3
"""Repair individual bars that a second provider says are on the wrong basis.

This replaces the seam-rebasing approach, which was tried, measured and
reverted. That approach assumed a split leaves a clean boundary — old basis
before the ex-date, new basis after — so rebasing everything before it would
make the series whole. Checking against IBKR bar by bar showed otherwise:

    MNST  08-05 UNADJUSTED(2.00x)  08-06 correct  08-07 UNADJUSTED(2.00x)
    SXTC  08-05 UNADJUSTED(0.013x) 08-06 correct  08-07 UNADJUSTED(0.013x)

Individual bars alternate between bases. There is no boundary to rebase around,
which is why rebasing before one made 19 of 24 symbols worse.

Worse for any price-only method: for MNST the *majority* of pre-split bars are
unadjusted and only a handful are correct, so a local "this bar disagrees with
both its neighbours" test flags the correct bars as the strays. Price data alone
cannot say which basis is right. It is not noisy, it is ambiguous — which is
precisely why this reads from `reference_price` and repairs nothing without it.

The rule is deliberately narrow. A bar is repaired only when it disagrees with
the reference by *exactly* a ratio the symbol actually has on record, within a
few percent. A bar that merely disagrees is left alone: that is a data question
this tool has no standing to answer, and quietly overwriting one provider's
prices with another's is not a repair, it is a migration.

Defaults to --dry-run. Pass --apply to write. Backs the database up first.

    python scripts/repair_bars_against_reference.py
    python scripts/repair_bars_against_reference.py --apply
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime
from typing import Dict, List, NamedTuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402
from db_utils import connect_readonly  # noqa: E402

# How close archive/reference must sit to a known ratio to call the bar
# mis-based rather than merely different.
RATIO_TOLERANCE = 0.04

# Below this the two providers are quoting the same thing; ordinary rounding.
AGREEMENT_TOLERANCE = 0.03

# How far the repaired value may still sit from the reference before the repair
# is refused as unconvincing. A good repair lands *on* the reference: dividing by
# the right ratio reproduces the other provider's price. Landing merely near it
# means the ratio was accepted on RATIO_TOLERANCE's generosity rather than
# because it explains the bar — BOTJ 2005-03-07 wants a 0.9091 factor for a
# disagreement that is really 0.9445, and would be written 3.9% away from the
# only evidence there is.
#
# The other thing this catches is precision, not logic: KGEI's archive bars are
# stored rounded to a cent (1.60), so dividing by 10 gives 0.16 where the truth
# is 0.155. Directionally right, and still not something to write as if it were
# measured.
DEFAULT_MAX_ERROR = 0.02


class BadBar(NamedTuple):
    symbol: str
    date: str
    archive: float
    reference: float
    factor: float
    repaired: float


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def find_bad_bars(conn: sqlite3.Connection, source: str) -> List[BadBar]:
    """Bars off the reference by exactly one of the symbol's own split ratios."""
    refs = conn.execute(
        "SELECT symbol, date, close FROM reference_price WHERE source = ? ORDER BY symbol, date",
        (source,),
    ).fetchall()
    if not refs:
        return []

    ratios: Dict[str, List[float]] = {}
    for symbol, value in conn.execute(
        "SELECT symbol, value FROM corporate_action WHERE kind = 'split' AND value > 0"
    ):
        ratios.setdefault(symbol, []).append(float(value))

    found: List[BadBar] = []
    for symbol, day, reference in refs:
        row = conn.execute(
            "SELECT close FROM daily_ohlcv WHERE symbol = ? AND date = ? AND interval = '1d'",
            (symbol, day),
        ).fetchone()
        if not row or not row[0] or not reference:
            continue
        archive = float(row[0])
        rel = archive / reference
        if abs(rel - 1.0) < AGREEMENT_TOLERANCE:
            continue

        # Candidate factors: every split this symbol has on record, and each
        # one's inverse. Nothing else is allowed to explain a difference.
        candidates: List[float] = []
        for r in ratios.get(symbol, []):
            candidates += [r, 1.0 / r] if r else []
        match = next(
            (f for f in candidates if f and abs(rel - f) / f < RATIO_TOLERANCE), None
        )
        if match is None:
            continue
        found.append(BadBar(symbol, day, archive, reference, match, archive / match))
    return found


def repair(db: str, bars: List[BadBar]) -> int:
    conn = sqlite3.connect(db, timeout=300.0)
    try:
        for bar in bars:
            conn.execute(
                """
                UPDATE daily_ohlcv
                   SET open = open / :f, high = high / :f, low = low / :f,
                       close = close / :f,
                       adj_close = CASE WHEN adj_close IS NULL THEN NULL
                                        ELSE adj_close / :f END,
                       volume = CAST(volume * :f AS INTEGER)
                 WHERE symbol = :s AND date = :d AND interval = '1d'
                """,
                {"f": bar.factor, "s": bar.symbol, "d": bar.date},
            )
        conn.commit()
    finally:
        conn.close()
    return len(bars)


def backup(path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(
        os.path.dirname(os.path.dirname(path)), "backups", f"market_data_pre_barfix_{stamp}.db"
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
    parser.add_argument("--source", default="ibkr", help="reference_price source to trust")
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR,
        help="refuse a repair that would land further than this from the "
        f"reference (default {DEFAULT_MAX_ERROR:.0%}); 0 disables the check",
    )
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        args.apply = False

    path = args.db or db_path()
    conn = connect_readonly(path)
    try:
        bars = find_bad_bars(conn, args.source)
        if args.max_error:
            unconvincing = [
                b
                for b in bars
                if b.reference
                and abs(b.repaired - b.reference) / abs(b.reference) > args.max_error
            ]
            bars = [b for b in bars if b not in unconvincing]
        else:
            unconvincing = []
        total_refs = conn.execute(
            "SELECT COUNT(*) FROM reference_price WHERE source = ?", (args.source,)
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"{total_refs} reference close(s) from '{args.source}'")
    if unconvincing:
        print(
            f"{len(unconvincing)} repair(s) refused: the result would not land on "
            f"the reference (>{args.max_error:.0%} away)"
        )
        for b in unconvincing:
            off = abs(b.repaired - b.reference) / abs(b.reference) * 100
            print(
                f"  {b.symbol:8}{b.date:12}{b.archive:12.4f}{b.reference:12.4f}"
                f"{b.factor:10.4f}{b.repaired:12.4f}  {off:5.1f}% off"
            )
        print()
    if not bars:
        print("No bar disagrees with the reference by a known split ratio.")
        return 0

    print(f"{len(bars)} bar(s) on the wrong basis, across {len({b.symbol for b in bars})} symbol(s):\n")
    print(f"  {'symbol':8}{'date':12}{'archive':>12}{'reference':>12}{'factor':>10}{'becomes':>12}")
    for b in bars:
        print(
            f"  {b.symbol:8}{b.date:12}{b.archive:12.4f}{b.reference:12.4f}"
            f"{b.factor:10.4f}{b.repaired:12.4f}"
        )

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to repair.")
        return 0

    saved = backup(path)
    print(f"\nBacked up to {saved}")
    print(f"Repaired {repair(path, bars)} bar(s).")

    conn = connect_readonly(path)
    try:
        remaining = find_bad_bars(conn, args.source)
    finally:
        conn.close()
    if remaining:
        print(f"{len(remaining)} bar(s) still disagree:")
        for b in remaining[:10]:
            print(f"  {b.symbol} {b.date} {b.archive:.4f} vs {b.reference:.4f}")
        return 1
    print("Verified: every referenced bar now agrees with the reference.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
