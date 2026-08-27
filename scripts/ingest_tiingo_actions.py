#!/usr/bin/env python3
"""Import corporate actions the archive's own feed never reported.

`corporate_action` is Yahoo's account of what happened, and Yahoo's account has
holes. PPCB is the worked example: Tiingo reports a 1:1000 reverse split on
2023-05-23 and the archive has no row for it, though its *prices* are adjusted
for it perfectly well. So the series is right and the event log is incomplete —
which matters in two places. `get_ohlcv(adjust='none')` un-adjusts by the splits
on record, so a missing one leaves the raw reconstruction wrong by that factor;
and both the verifier and the split checker use the log to *explain* a jump, so
an unrecorded event makes an explained move look unexplained forever.

**The dangerous failure here is a duplicate, not an omission.** Providers date
the same split a day apart routinely — BOTJ is 2005-03-07 at Yahoo and
2005-03-08 at Tiingo, one event with two dates. Import that blindly and the
archive holds two 1.5x splits for one corporate action, and every price before
it un-adjusts by 2.25x instead of 1.5x. So a candidate is refused when the
archive already carries the same ratio within DUPLICATE_WINDOW_DAYS: near in
time and equal in ratio means the same event, differently dated, and the
archive's own date is left alone.

Nothing here rewrites a price. Adding an action changes what `adjust='none'` and
`adjust='total_return'` reconstruct; the default `adjust='split'` returns stored
values untouched, which is why this is separable from a bar repair.

    python scripts/ingest_tiingo_actions.py --symbol PPCB
    python scripts/ingest_tiingo_actions.py --symbol PPCB --apply
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import date, datetime, timezone
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import check_split_consistency as checker  # noqa: E402
import config  # noqa: E402
from db_utils import connect_readonly  # noqa: E402
from tiingo_provider import (  # noqa: E402
    SOURCE,
    TiingoError,
    TiingoProvider,
    TiingoSymbolUnknown,
)

# Two providers dating one event a day or two apart is ordinary. Beyond this
# they are describing different events.
DUPLICATE_WINDOW_DAYS = 5

# Same tolerance the repair tool uses to call two ratios the same ratio.
RATIO_TOLERANCE = 0.04


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def archive_splits(conn: sqlite3.Connection, symbol: str) -> Dict[str, float]:
    return {
        r[0]: float(r[1])
        for r in conn.execute(
            "SELECT date, value FROM corporate_action "
            "WHERE symbol = ? AND kind = 'split' AND value > 0",
            (symbol,),
        )
    }


def tiingo_splits(rows: List[dict]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in rows:
        ratio = row.get("splitFactor")
        try:
            ratio = float(ratio) if ratio is not None else 1.0
        except (TypeError, ValueError):
            continue
        if ratio and ratio != 1.0:
            out[str(row.get("date", ""))[:10]] = ratio
    return out


def duplicate_of(
    day: str, ratio: float, existing: Dict[str, float]
) -> Tuple[str, float] | None:
    """The archive row this candidate is really the same event as, if any."""
    when = date.fromisoformat(day)
    for other_day, other_ratio in existing.items():
        try:
            gap = abs((date.fromisoformat(other_day) - when).days)
        except ValueError:
            continue
        if gap > DUPLICATE_WINDOW_DAYS:
            continue
        if other_ratio and abs(ratio - other_ratio) / other_ratio <= RATIO_TOLERANCE:
            return other_day, other_ratio
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="symbols to check (default: everything the split checker flags)",
    )
    parser.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = parser.parse_args()

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 1

    conn = connect_readonly(path)
    try:
        if args.symbol:
            symbols = sorted({s.upper() for s in args.symbol})
        else:
            symbols = sorted({f.symbol for f in checker.check(conn, since=None)})
        if not symbols:
            print("Nothing to check.")
            return 0

        provider = TiingoProvider()
        if not provider.is_configured():
            print("No TIINGO_API_KEY in .env.")
            return 1

        additions: List[Tuple[str, str, float]] = []
        for symbol in symbols:
            try:
                rows = provider.fetch_prices(symbol)
            except TiingoSymbolUnknown:
                print(f"  {symbol:8} not carried by Tiingo — skipped")
                continue
            except TiingoError as exc:
                print(f"  {symbol:8} FAILED: {exc}")
                if getattr(exc, "status", None) == 429:
                    break
                continue

            theirs = tiingo_splits(rows)
            ours = archive_splits(conn, symbol)
            new, dupes, missing_there = [], [], []
            for day, ratio in sorted(theirs.items()):
                dup = duplicate_of(day, ratio, ours)
                if dup:
                    if dup[0] != day:
                        dupes.append((day, ratio, dup[0]))
                else:
                    new.append((day, ratio))
            for day, ratio in sorted(ours.items()):
                if not duplicate_of(day, ratio, theirs):
                    missing_there.append((day, ratio))

            print(f"\n  {symbol}: tiingo {len(theirs)} split(s), archive {len(ours)}")
            for day, ratio, ours_day in dupes:
                print(
                    f"     same event, dated differently: tiingo {day} vs archive "
                    f"{ours_day} (x{ratio:g}) — archive kept"
                )
            for day, ratio in new:
                print(f"     ADD  {day}  x{ratio:g}")
                additions.append((symbol, day, ratio))
            for day, ratio in missing_there:
                print(
                    f"     note: archive has {day} x{ratio:g} and tiingo does not — "
                    "its series cannot adjudicate this symbol"
                )

        print(f"\n{len(additions)} action(s) to add.")
        if not additions:
            return 0
        if not args.apply:
            print("Dry run — nothing written. Re-run with --apply.")
            return 0
    finally:
        conn.close()

    writer = sqlite3.connect(path, timeout=120.0)
    try:
        stamp = datetime.now(timezone.utc).isoformat()
        writer.executemany(
            """
            INSERT OR IGNORE INTO corporate_action
                (symbol, date, kind, value, currency, source, ingested_at)
            VALUES (?, ?, 'split', ?, NULL, ?, ?)
            """,
            [(s, d, r, SOURCE, stamp) for s, d, r in additions],
        )
        writer.commit()
        print(f"Wrote {len(additions)} action(s) with source='{SOURCE}'.")
    finally:
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
