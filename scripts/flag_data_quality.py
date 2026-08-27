#!/usr/bin/env python3
"""Record which symbols have untrustworthy price history, for the app to show.

The archive knows a great deal about its own defects and has kept every bit of
it in scripts nobody runs from the UI. `check_split_consistency.py` reports
splits the price series does not reflect; `verify_market_archive.py` reports
jumps no corporate action explains. Both print to a terminal. A user opening a
chart has no way to know the line they are looking at steps 30x in the middle
for no reason.

This turns those two checks into a per-symbol flag the clients can read.

**Two sources, deliberately different in severity.**

`unapplied` / `mixed` split findings are HIGH. They mean a split is on record
and the prices do not reflect it, so the series is definitely wrong somewhere —
this is the class that had BYND reading 0.5560 where it should have read 16.68.

Unexplained discontinuities are MEDIUM, and only those at
`SEVERE_RATIO` or worse are recorded at all. The verifier reports ~15,000
findings across 3,000 symbols, most of them mild moves on thin stocks where a
1.4x day is just a 1.4x day. Flagging all of those would put a warning on half
the market and teach everyone to ignore it.

**The table is derived, not primary.** It is rebuilt wholesale on every run and
can be dropped without losing anything, which is why it is created here rather
than in a schema migration.

    python scripts/flag_data_quality.py            # rebuild the flags
    python scripts/flag_data_quality.py --dry-run  # report, write nothing
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import check_split_consistency as splits  # noqa: E402
import config  # noqa: E402
import verify_market_archive as verifier  # noqa: E402
from db_utils import connect_readonly  # noqa: E402

# Only discontinuities at least this severe are worth a user's attention. Below
# it the verifier is mostly reporting thin stocks being thin.
SEVERE_RATIO = 3.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS data_quality (
    symbol      TEXT NOT NULL,
    kind        TEXT NOT NULL,   -- 'unapplied' | 'mixed' | 'discontinuity' | ...
    severity    TEXT NOT NULL,   -- 'high' | 'medium'
    occurred_on TEXT,            -- the date the problem sits at
    detail      TEXT,
    detected_at TEXT NOT NULL,
    PRIMARY KEY (symbol, kind, occurred_on)
);
CREATE INDEX IF NOT EXISTS idx_data_quality_symbol ON data_quality (symbol);
"""


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def collect(path: str) -> list[tuple]:
    """Every flag worth showing, as (symbol, kind, severity, date, detail)."""
    rows: list[tuple] = []

    conn = connect_readonly(path)
    try:
        for f in splits.check(conn, since=None):
            rows.append(
                (
                    f.symbol,
                    f.shape,
                    "high",
                    f.ex_date,
                    # No date in the text: `occurred_on` carries it, so each
                    # client formats it in the notation it is required to use
                    # (DD MMM YYYY) rather than rendering an ISO string the
                    # backend happened to embed.
                    f"A {f.ratio:g} split is on record, but the stored prices "
                    f"do not reflect it.",
                )
            )
    finally:
        conn.close()

    for symbol, day, prev_day, prev_close, close, ratio in _severe(path):
        rows.append(
            (
                symbol,
                "discontinuity",
                "medium",
                day,
                f"The close moves {ratio:.1f}x from {prev_close:g} to {close:g} "
                f"with no corporate action to explain it.",
            )
        )
    return rows


def _severe(path: str):
    """Verifier findings at SEVERE_RATIO or worse, normalised."""
    for row in verifier.scan(path, threshold=SEVERE_RATIO, min_price=1.0,
                             symbol=None, as_of=None):
        # The verifier's tuple shape is (symbol, prev_day, day, prev_close,
        # close, ratio); it reports the ratio in whichever direction is > 1.
        symbol, prev_day, day, prev_close, close, ratio = row[:6]
        ratio = float(ratio)
        if ratio < 1:
            ratio = 1 / ratio if ratio else 0
        if ratio >= SEVERE_RATIO:
            yield symbol, day, prev_day, float(prev_close), float(close), ratio


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    path = args.db or db_path()
    if not os.path.exists(path):
        print(f"No market database at {path}.")
        return 1

    print("Scanning the archive for defects a user should know about ...")
    rows = collect(path)

    by_symbol: dict[str, list] = defaultdict(list)
    for r in rows:
        by_symbol[r[0]].append(r)
    high = {r[0] for r in rows if r[2] == "high"}
    print(
        f"  {len(rows)} finding(s) across {len(by_symbol)} symbol(s); "
        f"{len(high)} with a definite defect, "
        f"{len(by_symbol) - len(high)} with an unexplained jump only."
    )
    worst = sorted(by_symbol.items(), key=lambda kv: -len(kv[1]))[:8]
    for symbol, items in worst:
        print(f"     {symbol:8} {len(items):4d} finding(s)")

    if args.dry_run:
        print("\nDry run — nothing written.")
        return 0

    conn = sqlite3.connect(path, timeout=120.0)
    try:
        conn.executescript(SCHEMA)
        stamp = datetime.now(timezone.utc).isoformat()
        # Rebuilt wholesale: a defect that has been repaired must stop being
        # reported, and the cheapest way to guarantee that is to not carry
        # yesterday's answer forward.
        conn.execute("DELETE FROM data_quality")
        conn.executemany(
            "INSERT OR REPLACE INTO data_quality "
            "(symbol, kind, severity, occurred_on, detail, detected_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [(s, k, sev, day, detail, stamp) for s, k, sev, day, detail in rows],
        )
        conn.commit()
        print(f"\nWrote {len(rows)} flag(s) for {len(by_symbol)} symbol(s).")
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
