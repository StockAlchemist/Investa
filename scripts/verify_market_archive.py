#!/usr/bin/env python3
"""Scan the local market archive for price discontinuities (plan Phase 1.5).

A split that is not recorded as a corporate action leaves a seam: every row
before it sits on one price basis and every row after it on another, so the
series jumps by the split ratio for no reason. Nothing downstream notices —
cost basis, TWR, drawdown and every backtest simply consume the bad series — so
the archive needs its own check.

The rule: flag any adjacent-session close ratio outside [1/threshold, threshold]
that no `corporate_action` row explains.

Two classes of false positive are excluded rather than reported, because both
are ordinary and would otherwise bury the real findings:

  * sub-dollar tick noise. A stock oscillating between 0.0001 and 0.0002 doubles
    without anything happening; --min-price drops those rows.
  * genuine one-day moves. Leveraged ETFs and small caps really do move 50%+,
    so the default threshold sits at 1.4x, above ordinary volatility and well
    below the smallest split anyone runs (1.5:1).

Exit status is 1 when anything is flagged, so this can gate a backfill.

    python scripts/verify_market_archive.py [--symbol AAPL] [--threshold 1.4]
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402


def default_db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def load_actions(conn: sqlite3.Connection) -> dict[str, set[str]]:
    """{symbol: {dates on which a split is recorded}}."""
    out: dict[str, set[str]] = defaultdict(set)
    try:
        rows = conn.execute(
            "SELECT symbol, date FROM corporate_action WHERE kind = 'split'"
        )
    except sqlite3.OperationalError:
        return out  # pre-migration database: no actions table yet
    for symbol, day in rows:
        out[symbol].add(day)
    return out


def load_split_ratios(conn: sqlite3.Connection) -> dict[str, list[tuple[str, float]]]:
    """{symbol: [(ex-date, ratio), ...]} sorted by date."""
    out: dict[str, list[tuple[str, float]]] = defaultdict(list)
    try:
        rows = conn.execute(
            "SELECT symbol, date, value FROM corporate_action "
            "WHERE kind = 'split' AND value > 0 ORDER BY symbol, date"
        )
    except sqlite3.OperationalError:
        return out
    for symbol, day, ratio in rows:
        out[symbol].append((day, float(ratio)))
    return out


def traded_price(stored: float, day: str, splits: list[tuple[str, float]]) -> float:
    """
    Undo the back-adjustment to recover roughly what the share actually traded at.

    The price floor exists to skip tick noise, and tick noise is a property of
    the *traded* price, not the adjusted one. A stock that changed hands at
    $0.05 before a 1:68 and a 1:7 reverse split is stored at ~$24, so a
    one-tick wiggle reads as a 40% move and floods the report — LFVN and TRAK
    alone produced 117 such lines. Scaling back by the splits still ahead of
    the date puts the floor back on the number that was actually quoted.
    """
    if not splits:
        return stored
    factor = 1.0
    for ex_date, ratio in splits:
        if ex_date > day:
            factor *= ratio
    # adjusted = raw / factor, so raw = adjusted * factor. AAPL on 2020-08-27
    # stores 125.01 with a 4:1 split still ahead of it: 125.01 * 4 = 500.04,
    # which is what it actually closed at. A 1:68 reverse split carries a ratio
    # of 1/68, so the same multiplication correctly scales a sub-penny stock
    # back *down*.
    return stored * factor


def scan(
    db_path: str,
    threshold: float,
    min_price: float,
    symbol: str | None,
    as_of: str | None,
) -> list[tuple]:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        splits = load_actions(conn)
        ratios = load_split_ratios(conn)

        query = (
            "SELECT symbol, date, close FROM daily_ohlcv "
            "WHERE interval = '1d' AND close IS NOT NULL"
        )
        params: list = []
        if symbol:
            query += " AND symbol = ?"
            params.append(symbol)
        if as_of:
            query += " AND date < ?"
            params.append(as_of)
        query += " ORDER BY symbol, date"

        findings: list[tuple] = []
        prev_symbol = None
        prev_day = prev_close = None

        for sym, day, close in conn.execute(query, params):
            if sym != prev_symbol:
                prev_symbol, prev_day, prev_close = sym, day, close
                continue

            # The floor applies to what the share actually traded at, not to the
            # back-adjusted figure — see traded_price.
            sym_splits = ratios.get(sym, [])
            if (
                prev_close
                and close
                and traded_price(prev_close, prev_day, sym_splits) >= min_price
                and traded_price(close, day, sym_splits) >= min_price
            ):
                ratio = prev_close / close
                if ratio > threshold or ratio < 1.0 / threshold:
                    # A split recorded on either side of the gap explains it:
                    # the ex-date is the first session on the new basis.
                    explained = day in splits.get(sym, ()) or prev_day in splits.get(sym, ())
                    if not explained:
                        findings.append((sym, prev_day, day, prev_close, close, ratio))

            prev_day, prev_close = day, close

        return findings
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=None)
    parser.add_argument("--symbol", default=None, help="check one symbol only")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.4,
        help="adjacent-close ratio treated as suspicious (default 1.4)",
    )
    parser.add_argument(
        "--min-price",
        type=float,
        default=1.0,
        help="ignore sessions below this price — tick noise, not corporate actions (default 1.0)",
    )
    parser.add_argument("--as-of", dest="as_of", default=None, help="ignore rows on/after this date")
    parser.add_argument("--limit", type=int, default=40, help="max findings to print")
    args = parser.parse_args()

    path = args.db or default_db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 0

    findings = scan(path, args.threshold, args.min_price, args.symbol, args.as_of)

    if not findings:
        scope = args.symbol or "the archive"
        print(f"No unexplained price discontinuities in {scope}.")
        return 0

    by_symbol: dict[str, int] = defaultdict(int)
    for sym, *_ in findings:
        by_symbol[sym] += 1

    print(f"{len(findings)} unexplained discontinuity(ies) across {len(by_symbol)} symbol(s):\n")
    for sym, prev_day, day, prev_close, close, ratio in findings[: args.limit]:
        print(f"  {sym:10} {prev_day} -> {day}  {prev_close:12.4f} -> {close:<12.4f} ratio {ratio:.3f}")
    if len(findings) > args.limit:
        print(f"  ... and {len(findings) - args.limit} more")

    print("\nWorst offenders:")
    for sym, count in sorted(by_symbol.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {sym:10} {count}")

    print(
        "\nEach line is either a missing corporate action or a bad ingest. "
        "A real split needs its corporate_action row; a bad bar needs a refetch."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
