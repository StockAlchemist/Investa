#!/usr/bin/env python3
"""Resumable backfill / refresh for the market archive (plan Phase 4.1).

The archive only ever stored what a page happened to ask for, so its coverage
is whatever the app's traffic drew. That leaves two visible defects:

  * FX is frozen. EUR=X, GBP=X and CNY=X stop at 2026-06-15, JPY=X at
    2026-06-23, THBUSD=X at 2026-01-23 — not because a fetch failed but because
    only THB is in the portfolio, so nothing asked for the rest.
  * Symbols go stale silently once nothing looks at them.

An archive kept "for future use" cannot be demand-driven. This walks a tier and
brings every member current, recording progress per symbol so a run interrupted
after ninety minutes resumes rather than restarts.

Tiers:

  A  everything already in the archive, plus every FX pair. What you actually
     look at; minutes to run.
  B  A plus the symbols the ranking scores (~1,200). ~0.4 GB, ~30 min.
  C  the full US common-stock universe (~5,600). Overnight, and its payoff is
     preventing *future* survivorship bias — it cannot fix the existing kind,
     because the names that already delisted are no longer served.

Two lessons from the D7 repair are baked in, and both were silent failures:

  * the fetch starts 30 days before the earliest stored row. Asking for exactly
    MIN(date) left that first bar uncovered, so it kept its old value and became
    a fresh one-day seam;
  * stale "invalid symbol" cache entries are cleared for symbols the archive
    already holds. `_fetch_yf_historical_data` drops cached-invalid symbols
    before making any request, and ten Thai listings sat there reporting "no
    data returned" while never being asked.

    python scripts/backfill_market_history.py --tier A --dry-run
    python scripts/backfill_market_history.py --tier A --apply
    python scripts/backfill_market_history.py --tier B --apply --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

# See the D7 repair: the range edge is not reliably inclusive once yfinance has
# applied its own timezone handling, so the oldest bar is missed and keeps its
# pre-refresh value.
START_MARGIN_DAYS = 30

# Floor for symbols the archive does not already hold. Existing history is never
# truncated — a symbol already reaching 1980 keeps reaching 1980.
NEW_SYMBOL_FLOOR = date(2000, 1, 1)

FETCH_BATCH = 25


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def tier_members(tier: str) -> Tuple[List[str], List[str]]:
    """(symbols, fx_pairs) for a tier."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path()}?mode=ro", uri=True)
    try:
        symbols = [
            r[0]
            for r in conn.execute(
                "SELECT DISTINCT symbol FROM daily_ohlcv WHERE interval = '1d' "
                "ORDER BY symbol"
            )
        ]
        pairs = [
            r[0] for r in conn.execute("SELECT DISTINCT pair FROM daily_fx ORDER BY pair")
        ]
        # Symbols already known to be gone stay out of the rotation.
        delisted = {
            r[0]
            for r in conn.execute(
                "SELECT symbol FROM sync_metadata WHERE delisted_at IS NOT NULL"
            )
        }
    finally:
        conn.close()

    symbols = [s for s in symbols if s not in delisted]

    if tier == "A":
        return symbols, pairs

    extra: List[str] = []
    if tier in ("B", "C"):
        ranks = os.path.join(config.get_app_data_dir(), config.DB_DIR, "buffett_ranks.db")
        if os.path.exists(ranks):
            rconn = sqlite3.connect(f"file:{ranks}?mode=ro", uri=True)
            try:
                run = rconn.execute(
                    "SELECT MAX(run_id) FROM rank_runs WHERE finished_at IS NOT NULL"
                ).fetchone()[0]
                if run:
                    extra += [
                        r[0]
                        for r in rconn.execute(
                            "SELECT symbol FROM rank_scores WHERE run_id = ?", (run,)
                        )
                    ]
            finally:
                rconn.close()

    if tier == "C":
        universe = os.path.join(
            config.get_app_data_dir(), config.CACHE_DIR, "universe", "us_common_stock.json"
        )
        if os.path.exists(universe):
            with open(universe) as fh:
                payload = json.load(fh)
            extra += [
                e.get("symbol")
                for e in (payload.get("entries") or [])
                if isinstance(e, dict) and e.get("symbol")
            ]

    seen = set(symbols) | delisted
    for sym in extra:
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)
    return symbols, pairs


def load_progress(tier: str) -> Dict[str, str]:
    import sqlite3

    conn = sqlite3.connect(db_path(), timeout=60.0)
    try:
        return {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT symbol, done_through FROM backfill_progress WHERE tier = ?",
                (tier,),
            )
            if r[1]
        }
    except sqlite3.OperationalError:
        return {}
    finally:
        conn.close()


def record(tier: str, symbol: str, done_through: Optional[str], error: Optional[str]) -> None:
    import sqlite3

    conn = sqlite3.connect(db_path(), timeout=60.0)
    try:
        conn.execute(
            """
            INSERT INTO backfill_progress (tier, symbol, done_through, attempts, last_error, updated_at)
            VALUES (?, ?, ?, 1, ?, ?)
            ON CONFLICT(tier, symbol) DO UPDATE SET
                done_through = excluded.done_through,
                attempts = backfill_progress.attempts + 1,
                last_error = excluded.last_error,
                updated_at = excluded.updated_at
            """,
            (tier, symbol, done_through, error, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def mark_delisted(symbol: str) -> None:
    """A symbol the provider no longer serves is recorded, not retried nightly."""
    import sqlite3

    conn = sqlite3.connect(db_path(), timeout=60.0)
    try:
        conn.execute(
            """
            INSERT INTO sync_metadata (symbol, delisted_at) VALUES (?, ?)
            ON CONFLICT(symbol) DO UPDATE SET delisted_at = excluded.delisted_at
            """,
            (symbol, date.today().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tier", choices=("A", "B", "C"), default="A")
    parser.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="skip symbols already done today")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-fx", action="store_true")
    parser.add_argument(
        "--mark-delisted",
        action="store_true",
        help="record symbols that returned nothing as delisted so later runs skip "
        "them; their stored history is left untouched",
    )
    args = parser.parse_args()
    if args.dry_run:
        args.apply = False

    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
    import market_data
    from market_db import MarketDatabase

    symbols, pairs = tier_members(args.tier)
    today = date.today()

    done = load_progress(args.tier) if args.resume else {}
    pending = [s for s in symbols if done.get(s) != today.isoformat()]
    if args.limit:
        pending = pending[: args.limit]

    print(f"tier {args.tier}: {len(symbols)} symbols, {len(pairs)} FX pairs")
    if args.resume:
        print(f"  {len(symbols) - len(pending)} already done today, {len(pending)} pending")
    if not args.apply:
        print("\nDry run — nothing fetched. Re-run with --apply.")
        return 0

    db = MarketDatabase()
    provider = market_data.get_shared_mdp()

    # Stale invalid-symbol entries suppress the request entirely — see the
    # module docstring. Anything the archive already holds is real.
    try:
        invalid = provider._load_invalid_symbols_cache()
        stale = [s for s in pending if s in invalid]
        if stale:
            for s in stale:
                del invalid[s]
            provider._save_invalid_symbols_cache(invalid)
            print(f"  cleared {len(stale)} stale invalid-symbol entries")
    except Exception as exc:  # noqa: BLE001
        print(f"  could not clear invalid-symbol cache: {exc}")

    first_dates = db.get_first_dates(pending) if pending else {}

    refreshed = 0
    empty: List[str] = []
    failed: List[str] = []

    for start in range(0, len(pending), FETCH_BATCH):
        batch = pending[start : start + FETCH_BATCH]
        # One fetch window per batch: the earliest start any member needs.
        starts = [
            (first_dates[s] - timedelta(days=START_MARGIN_DAYS))
            if s in first_dates
            else NEW_SYMBOL_FLOOR
            for s in batch
        ]
        window_start = min(starts)
        print(
            f"  [{start + len(batch)}/{len(pending)}] fetching {len(batch)} "
            f"from {window_start} ...",
            end="",
            flush=True,
        )
        try:
            fetched = provider._fetch_yf_historical_data(
                batch, window_start, today, interval="1d"
            )
        except Exception as exc:  # noqa: BLE001
            print(f" FAILED: {type(exc).__name__}: {exc}")
            for s in batch:
                record(args.tier, s, None, str(exc)[:200])
                failed.append(s)
            continue

        wrote = 0
        for sym in batch:
            frame = fetched.get(sym)
            if frame is None or frame.empty:
                empty.append(sym)
                record(args.tier, sym, None, "no data returned")
                continue
            db.upsert_ohlcv(sym, frame, interval="1d")
            record(args.tier, sym, today.isoformat(), None)
            wrote += 1
        refreshed += wrote
        print(f" {wrote} refreshed")

    if pairs and not args.skip_fx:
        print(f"\nFX ({len(pairs)} pairs):")
        fx_first = db.get_first_dates(pairs, table="daily_fx")
        for pair in pairs:
            begin = (
                fx_first[pair] - timedelta(days=START_MARGIN_DAYS)
                if pair in fx_first
                else NEW_SYMBOL_FLOOR
            )
            print(f"  {pair:12} from {begin} ...", end="", flush=True)
            try:
                fetched = provider._fetch_yf_historical_data(
                    [pair], begin, today, interval="1d"
                )
                frame = fetched.get(pair)
                if frame is None or frame.empty:
                    print(" no data")
                    empty.append(pair)
                    continue
                db.upsert_fx(pair, frame, interval="1d")
                print(f" {len(frame)} rows")
            except Exception as exc:  # noqa: BLE001
                print(f" FAILED: {type(exc).__name__}: {exc}")
                failed.append(pair)

    print(f"\nRefreshed {refreshed}/{len(pending)} symbols.")
    if empty:
        print(f"No data for {len(empty)}: {', '.join(empty[:20])}")
        if args.mark_delisted:
            # Only symbols, never FX pairs: a pair that returns nothing is a
            # transient provider gap, not a delisting.
            retired = [s for s in empty if s not in pairs]
            for sym in retired:
                mark_delisted(sym)
            print(f"  marked {len(retired)} as delisted; stored history untouched")
        else:
            print(
                "  Re-run with --mark-delisted to retire the ones the provider no "
                "longer serves, keeping them out of future runs while leaving "
                "their stored history untouched."
            )
    if failed:
        print(f"Failed for {len(failed)}: {', '.join(failed[:20])}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
