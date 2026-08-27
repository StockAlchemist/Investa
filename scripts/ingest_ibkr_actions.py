#!/usr/bin/env python3
"""Load broker-sourced corporate actions and reference closes into the archive.

Yahoo's split feed is the archive's only witness to a corporate action, and it
is not a reliable one: it re-bases series after the fact, dates some splits a
day off the price move, and reports nothing at all for others. A second,
independent source turns "the prices look wrong" into "these two sources
disagree and here is which one matches the broker's own books".

IBKR is that source. Its actions are typed and carry real dates:

    {"type": "Splits", "date": "20260721", "announce_date": "20260623",
     "value": "3.0"}

and its prices are adjusted where Yahoo's are not — WLFC on 2026-07-17 is
63.883 at IBKR and 191.65 at Yahoo, exactly three times higher and unadjusted
for the split IBKR dates to 07-21.

**Why a file rather than a live client.** `ibkr_connector.py` speaks the Flex
Web Service, which serves account activity, not the general corporate-action
feed; reaching the latter needs a Client Portal session the app does not hold.
So this takes a JSON payload someone else collected and is responsible only for
loading it correctly, with provenance, and never overwriting a better source
with a worse one.

Payload shape:

    {"WLFC": {"actions": [{"type": "Splits", "date": "20260721",
                           "announce_date": "20260623", "value": "3.0"}],
              "prices":  {"2026-07-17": 63.883, "2026-07-20": 63.12}}}

Splits and cash dividends are stored in `corporate_action` with source
`ibkr`; the closes go to `reference_price`, which exists to settle a
disagreement rather than to be a second price history.

    python scripts/ingest_ibkr_actions.py payload.json --dry-run
    python scripts/ingest_ibkr_actions.py payload.json --apply
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import config  # noqa: E402

SOURCE = "ibkr"

# IBKR's action types mapped onto the two kinds the archive stores. Anything
# else — tenders, rights, spin-offs, redemptions — is deliberately ignored:
# recording a rights subscription as a split would rescale a history.
KIND_BY_TYPE = {"Splits": "split", "CashDividends": "dividend"}


def db_path() -> str:
    return os.path.join(config.get_app_data_dir(), config.DB_DIR, "market_data.db")


def _iso(stamp: str) -> str:
    """IBKR dates are yyyyMMdd; the archive stores yyyy-MM-dd."""
    s = str(stamp).strip()
    if len(s) == 8 and s.isdigit():
        return f"{s[:4]}-{s[4:6]}-{s[6:]}"
    return s[:10]


def parse(payload: Dict[str, Any]) -> Tuple[List[tuple], List[tuple], List[str]]:
    """(action rows, reference-price rows, notes) from a raw payload."""
    now = datetime.now().isoformat()
    actions: List[tuple] = []
    prices: List[tuple] = []
    notes: List[str] = []

    for symbol, block in payload.items():
        # A payload is a record of an adjudication someone has to be able to
        # reproduce later, so it is worth letting it carry a note about where it
        # came from. Keys starting with an underscore are documentation, not
        # instruments.
        if symbol.startswith("_") or not isinstance(block, dict):
            continue
        for raw in block.get("actions") or []:
            kind = KIND_BY_TYPE.get(raw.get("type"))
            if not kind:
                notes.append(f"{symbol}: ignored action type {raw.get('type')!r}")
                continue
            try:
                value = float(raw.get("value"))
            except (TypeError, ValueError):
                notes.append(f"{symbol}: unparseable value {raw.get('value')!r}")
                continue
            if value <= 0:
                continue
            actions.append(
                (symbol, _iso(raw.get("date")), kind, value, None, SOURCE, now)
            )

        for day, close in (block.get("prices") or {}).items():
            try:
                prices.append((symbol, _iso(day), float(close), SOURCE, now))
            except (TypeError, ValueError):
                notes.append(f"{symbol}: unparseable close {close!r} on {day}")

    return actions, prices, notes


def load(conn: sqlite3.Connection, actions: List[tuple], prices: List[tuple]) -> Dict[str, int]:
    """
    Write both sets. An IBKR action replaces a same-day one from another source
    — it is the better witness — but the primary key is (symbol, date, kind), so
    an action IBKR dates differently lands as a *second* row rather than
    silently overwriting Yahoo's. That disagreement is the useful part and the
    adjudicator wants to see both.
    """
    conn.executemany(
        """
        INSERT INTO corporate_action
            (symbol, date, kind, value, currency, source, ingested_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date, kind) DO UPDATE SET
            value = excluded.value, source = excluded.source,
            ingested_at = excluded.ingested_at
        """,
        actions,
    )
    conn.executemany(
        """
        INSERT INTO reference_price (symbol, date, close, source, fetched_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(symbol, date, source) DO UPDATE SET
            close = excluded.close, fetched_at = excluded.fetched_at
        """,
        prices,
    )
    conn.commit()
    return {"actions": len(actions), "prices": len(prices)}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("payload", help="JSON file collected from IBKR")
    parser.add_argument("--db", default=None)
    parser.add_argument("--apply", action="store_true", help="write (default is a dry run)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.dry_run:
        args.apply = False

    with open(args.payload) as fh:
        payload = json.load(fh)

    actions, prices, notes = parse(payload)
    instruments = [k for k in payload if not k.startswith("_")]
    print(f"{len(instruments)} symbol(s): {len(actions)} action(s), {len(prices)} reference close(s)")
    for symbol in sorted(payload):
        mine = [a for a in actions if a[0] == symbol and a[2] == "split"]
        for a in mine:
            print(f"  {symbol:8} split {a[1]} ratio {a[3]}")
    for note in notes:
        print(f"  note: {note}")

    if not args.apply:
        print("\nDry run — nothing written.")
        return 0

    conn = sqlite3.connect(args.db or db_path(), timeout=300.0)
    try:
        counts = load(conn, actions, prices)
    finally:
        conn.close()
    print(f"\nStored {counts['actions']} action(s) and {counts['prices']} reference close(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
