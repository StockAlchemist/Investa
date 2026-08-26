#!/usr/bin/env python3
"""Adjudicate split findings against Tiingo (plan Phase 5.3).

`check_split_consistency.py` runs nightly and reports splits the price series
does not reflect. It deliberately repairs nothing: price data alone cannot say
which of two disagreeing bases is right — for MNST the *majority* of pre-split
bars are unadjusted, so a "this bar disagrees with its neighbours" test flags
the correct ones — and the only reference that could settle it was IBKR, which
retail accounts cannot automate. So the findings queued up and waited for
someone to collect broker bars by hand.

This closes that loop. It reads the checker's own findings, fetches those
symbols from Tiingo, and writes reference closes into `reference_price` with
source `tiingo`. From there nothing new is needed:

    python scripts/ingest_tiingo_reference.py --apply
    python scripts/repair_bars_against_reference.py --source tiingo
    python scripts/repair_bars_against_reference.py --source tiingo --apply

**The reference is stored on the archive's basis, not Tiingo's.** This is the
trap the whole script is arranged around. `repair_bars_against_reference.py`
acts when a stored bar differs from the reference by exactly one of the symbol's
own split ratios. The archive holds split-adjusted prices; Tiingo's `close` is
raw. Hand over the raw series and *every* pre-split bar differs by exactly the
split ratio — so the repair would rewrite an entire history onto the raw basis
and call it a repair. It is a migration, and that script exists to refuse it. So
each symbol's whole series is fetched and re-adjusted locally with the same
arithmetic the archive uses.

Fetching the whole series is not laziness: a split *after* a narrow window would
be missed, and the reference would then sit on a basis of its own — worse than
having no reference, because it looks like evidence.

What is stored is every bar actually **in dispute**, wherever it falls, rather
than a window around the flagged date. An unapplied split leaves every earlier
bar wrong, not just the neighbourhood — see `disputed_days` for the run of BYND
bars a fifteen-day window left behind.

Only symbols the checker actually flagged are fetched, so the free tier's
per-hour and per-month meters are spent on the queue and nothing else. Non-US
tickers 404 and are reported, not retried — Tiingo carries no SET listings.
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Dict, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import check_split_consistency as checker  # noqa: E402
import config  # noqa: E402
from db_utils import connect_readonly  # noqa: E402
from ingest_tiingo_actions import archive_splits, duplicate_of, tiingo_splits  # noqa: E402
from tiingo_provider import (  # noqa: E402
    SOURCE,
    TiingoError,
    TiingoNotConfiguredError,
    TiingoProvider,
    TiingoSymbolUnknown,
    split_adjusted,
)

# A bar counts as disputed when the two providers differ by more than this.
# Matches repair_bars_against_reference.AGREEMENT_TOLERANCE: below it they are
# quoting the same thing and rounding differently.
DISPUTE_TOLERANCE = 0.03

# A ceiling on how much evidence one symbol may contribute, newest first. An
# unapplied split on a long history legitimately disputes thousands of bars, but
# `reference_price` is not meant to become a second price history and a run that
# wants more than this is worth looking at before it is trusted.
MAX_REFERENCE_BARS = 5000

# How far two series' daily moves may differ before they are treated as
# different instruments. The same stock moves by the same percentage on both
# feeds whatever basis each is on; two different companies share nothing. 1% is
# loose enough for a rounded close and far tighter than unrelated prices.
IDENTITY_TOLERANCE = 0.01


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def findings_by_symbol(path: str) -> Dict[str, List[str]]:
    """{symbol: [ex_date, ...]} from the checker's own detector."""
    conn = connect_readonly(path)
    try:
        found = checker.check(conn, since=None)
    finally:
        conn.close()
    out: Dict[str, List[str]] = {}
    for f in found:
        out.setdefault(f.symbol, [])
        if f.ex_date not in out[f.symbol]:
            out[f.symbol].append(f.ex_date)
    return out


def disputed_days(
    conn: sqlite3.Connection, symbol: str, closes: Dict[str, float]
) -> List[str]:
    """Every archive day that actually disagrees with the reference.

    This replaced a fixed ±15-day window around each flagged ex-date, which was
    inherited from the IBKR workflow where bars were collected by hand and a
    window was all anyone could reasonably gather. Against an API that returns
    the whole series for one request, it is an arbitrary limit — and the wrong
    shape for the commonest defect.

    BYND is the worked example. Its 1:30 reverse split of 2026-08-14 was never
    applied to a run of earlier bars, and the archive held:

        07-20  17.85   07-23  0.56   07-30  16.68   (after the first repair)
        07-21  18.30   07-24  0.559  07-31  16.98
        07-22  17.91   07-29  0.531  08-03  18.36

    The window reached back fifteen days from the ex-date, so it repaired from
    07-30 and left 07-23..07-29 exactly as wrong as before — bracketed by
    correct bars on both sides, and still enough of a discontinuity for the
    checker to keep flagging the symbol. Nineteen of twenty-five repaired
    symbols stayed flagged for this reason.

    So the reference is now every bar in dispute, wherever it falls. That keeps
    `reference_price` to its purpose — evidence about a disagreement, not a
    second price history — because a bar the two providers agree on is not
    evidence about anything.
    """
    rows = conn.execute(
        """
        SELECT date, close FROM daily_ohlcv
        WHERE symbol = ? AND interval = '1d' AND close > 0
        ORDER BY date
        """,
        (symbol,),
    ).fetchall()
    disputed = [
        day
        for day, close in rows
        if closes.get(day) and abs(float(close) / closes[day] - 1.0) > DISPUTE_TOLERANCE
    ]
    return disputed[-MAX_REFERENCE_BARS:]


def split_coverage_check(
    conn: sqlite3.Connection, symbol: str, rows: List[dict]
) -> tuple:
    """Does the reference reflect every split the archive already records?

    **The guard that was missing, and it cost 15,588 bars.** A reference is only
    comparable if it is adjusted for the same events. When Tiingo does not carry
    a split the archive does, its series sits a whole ratio away from the
    archive's on every earlier bar — and that ratio is, by construction, exactly
    one of the symbol's own recorded splits. So
    `repair_bars_against_reference.py` sees a difference it can "explain",
    divides by it, and pulls bars onto the reference's *unadjusted* basis.

    KGEI is the worked example. The archive records a 1:10 reverse split on
    2022-05-19; Tiingo has no split for KGEI at all. Every pre-2022 bar
    therefore disagreed by 10x, the repair accepted 10.0 as a known ratio, and
    the bars it rewrote landed on Tiingo's basis while the ones the max-error
    guard refused stayed on Yahoo's. The result was a series alternating between
    0.07 and 0.7 hundreds of times — a mixed basis, which is the exact
    pathology this whole effort exists to remove.

    Not caught by anything downstream: the ratio matches, the repaired value
    lands on the reference, and the identity check passes because the two series
    move together. Only the event logs disagree.
    """
    theirs = tiingo_splits(rows)
    ours = archive_splits(conn, symbol)
    unreflected = [
        (day, ratio)
        for day, ratio in sorted(ours.items())
        if not duplicate_of(day, ratio, theirs)
    ]
    if unreflected:
        day, ratio = unreflected[0]
        return False, (
            f"reference does not carry the archive's {day} x{ratio:g} split"
            + (f" (+{len(unreflected) - 1} more)" if len(unreflected) > 1 else "")
            + " — its series is on a different basis"
        )
    return True, f"reflects all {len(ours)} recorded split(s)"


def identity_check(
    conn: sqlite3.Connection, symbol: str, closes: Dict[str, float]
) -> tuple:
    """Is Tiingo's series the same instrument the archive holds? (ok, reason).

    **A ticker is not an identity**, and two different guards are needed because
    it fails in two different directions.

    *The range guard, upstream of this one,* catches the common case: AYA. The
    archive carries a 2018 split for it; Tiingo's AYA is 79 bars beginning
    2026-05-04, a different company handed the ticker after the first one left.
    Note that this check would **pass** AYA at 0.00% — both feeds agree about
    the *new* company's recent bars, which is exactly why it cannot be the test
    for whether the old ones are comparable. Only "does the history reach the
    finding at all" catches that.

    *This guard* catches the mirror image: a ticker where both feeds have deep
    history but of different companies, so the series diverge where it matters.
    `repair_bars_against_reference.py` cannot catch that itself — its guard is
    that the disagreement matches one of the symbol's own split ratios, which a
    coincidence on someone else's prices can satisfy.

    **Compared on returns, not on price levels, and that distinction is the
    whole test.** Comparing levels looks obviously right and rejects exactly the
    wrong symbols: BYND and CURX both failed it at 96% before this was fixed,
    because each had a split days earlier (0.0333 on 2026-08-14, 0.05 on
    2026-08-20) that the archive had not applied — which is the very defect
    being adjudicated. Their most recent bars agree to the cent; the older ones
    differ by exactly the ratio. A level test therefore refuses to look at any
    symbol whose problem is recent, which is most of them.

    Day-over-day returns survive a basis difference: two series of the same
    instrument move by the same percentage whether or not one of them has been
    rebased. Two different companies do not. Pairs that straddle a recorded
    ex-date are dropped, since that is the one day where an unapplied series
    legitimately jumps.
    """
    rows = conn.execute(
        """
        SELECT date, close FROM daily_ohlcv
        WHERE symbol = ? AND interval = '1d' AND close > 0
        ORDER BY date DESC LIMIT 90
        """,
        (symbol,),
    ).fetchall()
    common = sorted((d, float(c), closes[d]) for d, c in rows if closes.get(d))
    if len(common) < 12:
        return False, f"only {len(common)} recent day(s) in common"

    ex_dates = {
        r[0]
        for r in conn.execute(
            "SELECT date FROM corporate_action WHERE symbol = ? AND kind = 'split'",
            (symbol,),
        )
    }

    diffs = []
    for (d0, a0, b0), (d1, a1, b1) in zip(common, common[1:]):
        if d1 in ex_dates or not a0 or not b0:
            continue
        diffs.append(abs((a1 / a0) - (b1 / b0)))
    if len(diffs) < 10:
        return False, f"only {len(diffs)} comparable day(s)"
    diffs.sort()
    median = diffs[len(diffs) // 2]
    if median > IDENTITY_TOLERANCE:
        return False, (
            f"daily moves differ by {median * 100:.1f}% — a different listing, "
            "not a basis difference"
        )
    return True, f"{len(diffs)} daily move(s) match to {median * 100:.2f}%"


def store(path: str, symbol: str, rows: List[tuple]) -> int:
    conn = sqlite3.connect(path, timeout=120.0)
    try:
        fetched = datetime.now(timezone.utc).isoformat()
        conn.executemany(
            """
            INSERT OR REPLACE INTO reference_price
                (symbol, date, close, source, fetched_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            [(symbol, day, close, SOURCE, fetched) for day, close in rows],
        )
        conn.commit()
        return len(rows)
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument(
        "--symbol",
        action="append",
        default=None,
        help="adjudicate only these symbols (default: everything the checker flags)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="stop after this many symbols, to stay inside the free tier's meters",
    )
    parser.add_argument("--apply", action="store_true", help="write; default is a dry run")
    args = parser.parse_args()

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 1

    queue = findings_by_symbol(path)
    if args.symbol:
        wanted = {s.upper() for s in args.symbol}
        queue = {s: d for s, d in queue.items() if s.upper() in wanted}
    if not queue:
        print("Nothing flagged — no reference needed.")
        return 0

    symbols = sorted(queue)
    if args.limit:
        symbols = symbols[: args.limit]
    print(
        f"{len(queue)} symbol(s) flagged by the split check; "
        f"fetching {len(symbols)}.\n"
    )

    provider = TiingoProvider()
    if not provider.is_configured():
        print(
            "No TIINGO_API_KEY in .env — nothing to adjudicate with.\n"
            "Register at https://www.tiingo.com and add TIINGO_API_KEY=..."
        )
        return 1

    conn = connect_readonly(path)
    written = unknown = failed = out_of_range = mismatched = 0
    try:
        for symbol in symbols:
            ex_dates = queue[symbol]
            try:
                raw_rows = provider.fetch_prices(symbol)
                closes = split_adjusted(raw_rows)
            except TiingoSymbolUnknown:
                print(f"  {symbol:8} not carried by Tiingo (non-US?) — skipped")
                unknown += 1
                continue
            except TiingoNotConfiguredError as exc:
                print(f"  {exc}")
                return 1
            except TiingoError as exc:
                print(f"  {symbol:8} FAILED: {exc}")
                failed += 1
                if getattr(exc, "status", None) == 429:
                    print("  Stopping: the meter is spent. Re-run later.")
                    break
                continue

            # Two guards, both learned by measuring: Tiingo's history may not
            # reach the finding, and the ticker may not be the same company.
            earliest = min(closes) if closes else None
            if not earliest or earliest > min(ex_dates):
                print(
                    f"  {symbol:8} history starts {earliest} — after the finding "
                    f"({min(ex_dates)}); cannot testify"
                )
                out_of_range += 1
                continue
            ok, why = split_coverage_check(conn, symbol, raw_rows)
            if not ok:
                print(f"  {symbol:8} REFUSED: {why}")
                mismatched += 1
                continue
            ok, why = identity_check(conn, symbol, closes)
            if not ok:
                print(f"  {symbol:8} REFUSED: {why}")
                mismatched += 1
                continue

            days = disputed_days(conn, symbol, closes)
            rows = [(d, closes[d]) for d in days]
            if not rows:
                print(
                    f"  {symbol:8} agrees with the reference everywhere — "
                    f"nothing to adjudicate"
                )
                continue
            print(
                f"  {symbol:8} {len(rows):4d} disputed bar(s), "
                f"{rows[0][0]} → {rows[-1][0]}  (flagged {', '.join(ex_dates[:2])})"
            )
            if rows and args.apply:
                written += store(path, symbol, rows)
    finally:
        conn.close()

    print(
        f"\n{provider.calls_made} Tiingo request(s). "
        f"{unknown} not carried, {out_of_range} out of range, "
        f"{mismatched} refused as a different listing, {failed} failed."
    )
    if not args.apply:
        print("Dry run — nothing written. Re-run with --apply.")
        return 0
    print(f"Wrote {written} reference bar(s) with source='{SOURCE}'.")
    print(
        "\nNext: python scripts/repair_bars_against_reference.py --source tiingo\n"
        "      (dry-run by default; it repairs only bars off the reference by "
        "exactly one of the symbol's own split ratios)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
