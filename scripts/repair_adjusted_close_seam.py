#!/usr/bin/env python3
"""Repair the dividend-adjusted price seam in the archive (plan defect D7).

An early era of the fetch path used `auto_adjust=True`, which returns a single
already-adjusted `Close` and no `Adj Close` column at all. `upsert_ohlcv` stored
that adjusted number in `close` and copied it into `adj_close` as well. Later
fetches used `auto_adjust=False` and stored the real quoted price. The result is
one column holding two different definitions of "close", spliced at whatever date
the newer backfill happened to start — for most affected symbols, 2002-07-01.

The seam is largest for high-dividend names, because decades of distributions
compound into the adjustment. `NLY` stores 4.68 for 28 Jun 2002; it actually
closed at 77.60 that day, a 16x understatement, and the series then jumps 15.5x
into the correctly-stored rows.

The repair is a full-history refetch through the current path, which writes
proper quoted closes and captures the symbol's corporate actions on the way.

Detection, rather than a hardcoded symbol list, so this stays honest if run
again: a symbol qualifies when every row before its first `adj_close <> close`
row is a copy (the auto_adjust signature) *and* the close jumps across that
boundary by more than --min-jump.

Defaults to --dry-run. Pass --apply to write. Backs the database up first.

    python scripts/repair_adjusted_close_seam.py            # report only
    python scripts/repair_adjusted_close_seam.py --apply
"""

from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import sys
from datetime import date, datetime, timedelta

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def find_affected(db_path: str, min_jump: float) -> list[dict]:
    """Symbols carrying an adjusted-close block, worst jump first."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        symbols = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT symbol FROM daily_ohlcv WHERE interval = '1d'"
            )
        ]

        affected: list[dict] = []
        for symbol in symbols:
            boundary = conn.execute(
                "SELECT MIN(date) FROM daily_ohlcv "
                "WHERE symbol = ? AND interval = '1d' AND adj_close <> close",
                (symbol,),
            ).fetchone()[0]
            if not boundary:
                continue

            total, copied = conn.execute(
                "SELECT COUNT(*), SUM(CASE WHEN adj_close = close THEN 1 ELSE 0 END) "
                "FROM daily_ohlcv WHERE symbol = ? AND interval = '1d' AND date < ?",
                (symbol, boundary),
            ).fetchone()
            # Every row before the boundary must be a copy: that is the
            # auto_adjust-era signature. A mixed block is something else and is
            # left alone rather than guessed at.
            if not total or copied != total:
                continue

            before = conn.execute(
                "SELECT date, close FROM daily_ohlcv "
                "WHERE symbol = ? AND interval = '1d' AND date < ? "
                "ORDER BY date DESC LIMIT 1",
                (symbol, boundary),
            ).fetchone()
            after = conn.execute(
                "SELECT date, close FROM daily_ohlcv "
                "WHERE symbol = ? AND interval = '1d' AND date = ?",
                (symbol, boundary),
            ).fetchone()
            if not (before and after and before[1] and after[1]):
                continue

            ratio = after[1] / before[1]
            if abs(ratio - 1.0) < min_jump:
                continue

            earliest = conn.execute(
                "SELECT MIN(date) FROM daily_ohlcv WHERE symbol = ? AND interval = '1d'",
                (symbol,),
            ).fetchone()[0]

            affected.append(
                {
                    "symbol": symbol,
                    "suspect_rows": total,
                    "earliest": earliest,
                    "boundary": boundary,
                    "before_close": before[1],
                    "after_close": after[1],
                    "ratio": ratio,
                }
            )

        affected.sort(key=lambda r: -abs(r["ratio"] - 1.0))
        return affected
    finally:
        conn.close()


def backup(db_path: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = os.path.join(
        os.path.dirname(os.path.dirname(db_path)), "backups", f"market_data_pre_d7_{stamp}.db"
    )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    # VACUUM INTO gives a consistent copy without stopping writers; fall back to
    # a plain file copy if the source is mid-transaction.
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("VACUUM INTO ?", (target,))
        conn.close()
    except Exception:
        shutil.copy2(db_path, target)
    return target


# The refetch starts this far before the symbol's earliest stored row. Asking
# for exactly MIN(date) left that first row uncovered — the range edge is not
# reliably inclusive once yfinance has applied its own timezone handling — so
# the oldest bar kept its pre-repair value and became a fresh one-day seam.
# The margin also picks up whatever earlier history the provider has, which
# arrives on the same basis as the rest and is pure gain.
_START_MARGIN_DAYS = 30


def repair(rows: list[dict], end_date: date) -> tuple[int, list[str]]:
    """Refetch each symbol's full history through the current path."""
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import market_data
    from market_db import MarketDatabase

    provider = market_data.get_shared_mdp()
    db = MarketDatabase()

    # Symbols cached as "invalid" are skipped before any request is made. Every
    # symbol here has years of stored history, so a cache entry is stale — most
    # of the Thai listings were parked there by some earlier transient failure
    # and would silently return nothing.
    try:
        invalid = provider._load_invalid_symbols_cache()
        stale = [r["symbol"] for r in rows if r["symbol"] in invalid]
        if stale:
            for symbol in stale:
                del invalid[symbol]
            provider._save_invalid_symbols_cache(invalid)
            print(f"  cleared {len(stale)} stale invalid-symbol entries: {', '.join(stale)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not clear invalid-symbol cache: {exc}")

    repaired = 0
    failed: list[str] = []

    for index, row in enumerate(rows, start=1):
        symbol = row["symbol"]
        start = datetime.strptime(row["earliest"], "%Y-%m-%d").date() - timedelta(
            days=_START_MARGIN_DAYS
        )
        print(f"  [{index}/{len(rows)}] {symbol} from {start} ...", end="", flush=True)
        try:
            fetched = provider._fetch_yf_historical_data(
                [symbol], start, end_date, interval="1d"
            )
            frame = fetched.get(symbol)
            if frame is None or frame.empty:
                print(" no data returned")
                failed.append(symbol)
                continue

            # upsert_ohlcv overwrites by (symbol, date, interval) and now also
            # persists the Dividends / Stock Splits columns that came with it.
            db.upsert_ohlcv(symbol, frame, interval="1d")
            print(f" {len(frame)} bars")
            repaired += 1
        except Exception as exc:  # noqa: BLE001 - one bad symbol must not stop the run
            print(f" FAILED: {type(exc).__name__}: {exc}")
            failed.append(symbol)

    return repaired, failed


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=None)
    parser.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    parser.add_argument(
        "--min-jump",
        type=float,
        default=0.10,
        help="minimum relative jump at the boundary to count as a seam (default 0.10)",
    )
    parser.add_argument("--limit", type=int, default=None, help="repair at most N symbols")
    args = parser.parse_args()

    path = args.db or default_db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 0

    rows = find_affected(path, args.min_jump)
    if not rows:
        print("No adjusted-close seam found — nothing to repair.")
        return 0

    if args.limit:
        rows = rows[: args.limit]

    total_rows = sum(r["suspect_rows"] for r in rows)
    print(f"{len(rows)} symbol(s) carrying an adjusted-close seam, {total_rows:,} suspect rows:\n")
    print(f"  {'symbol':12}{'rows':>7}  {'boundary':12} {'before':>12} {'after':>12}   jump")
    for r in rows:
        print(
            f"  {r['symbol']:12}{r['suspect_rows']:7}  {r['boundary']:12} "
            f"{r['before_close']:12.4f} {r['after_close']:12.4f}   x{r['ratio']:.3f}"
        )

    if not args.apply:
        print("\nDry run — nothing written. Re-run with --apply to repair.")
        return 0

    saved = backup(path)
    print(f"\nBacked up to {saved}")

    print(f"\nRefetching {len(rows)} symbol(s) in full:")
    repaired, failed = repair(rows, date.today())

    print(f"\nRepaired {repaired}/{len(rows)} symbol(s).")
    if failed:
        print(f"Failed: {', '.join(failed)}")

    remaining = find_affected(path, args.min_jump)
    if remaining:
        print(f"\n{len(remaining)} symbol(s) still show a seam: {', '.join(r['symbol'] for r in remaining)}")
        return 1
    print("\nVerified: no adjusted-close seam remains.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
