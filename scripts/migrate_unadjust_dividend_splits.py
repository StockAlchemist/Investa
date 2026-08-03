"""
Un-adjust Dividend Quantity / Price/Share back to the actual as-paid values.

Every other row type in the ledger (Buy, Sell, Short Sell, …) is stored at the
RAW price and share count that changed hands on the trade date — e.g. a 2005
AAPL buy is stored at $88.67, a 2013 sell at $446.26 (both pre-split). The
valuation engine applies stock splits *in memory* from the Split rows, so it
expects raw, as-paid values.

Dividends are the lone exception: the legacy E*TRADE dividend import stored
them split-ADJUSTED to the current share basis. A 2016 AAPL dividend on 2,107
shares at $0.57/share (actual) is stored as 8,428 shares at $0.1425 — i.e. the
share count multiplied, and the per-share divided, by the 4-for-1 2020 split.
The Total Amount (actual cash) is identical either way, and the engine reads
Total Amount for cash flow and Quantity*Price (an invariant of the adjustment)
for dividend income, so this rewrite does not change any computed figure — it
only makes the ledger show what actually happened, matching the row's Note
("Cash Div on 2107 Shs") and the raw convention used by every trade row.

For each dividend row we compute the cumulative split factor from Split rows of
the SAME symbol dated strictly AFTER the dividend, then:

    new_price = old_price * factor
    new_qty   = old_qty   / factor      (Total Amount left untouched)

A row is only rewritten when we can PROVE it was stored adjusted: the E*TRADE
import records the actual share count in the Note ("Cash Div on 2107 Shs"), and
we require that count to match the un-adjusted quantity. Dividends without such
a note (e.g. EPP/XLE, whose quantities already track raw holdings straight
through their splits) are left untouched — dividing those would corrupt real,
as-paid data. Rows that are internally inconsistent (Qty*Price != Total) or
whose later splits carry an out-of-range ratio are likewise skipped.

Defaults to --dry-run. Pass --apply to commit. Auto-backs up the DB.

Targets dheematan's portfolio.db in the repo data submodule by default.
Pass --db to override.
"""

import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = REPO_ROOT / "data" / "users" / "dheematan" / "portfolio.db"

# Product tolerance: Qty*Price must match Total within this to be "consistent".
PRODUCT_TOL = 0.02  # dollars
# Sane multiplicative split ratio bounds. Forward splits up to 100:1, reverse
# splits down to 1:20. Excludes the legacy "target quantity" encoding (e.g.
# -446) that some rows store in the Split Ratio column.
RATIO_MIN, RATIO_MAX = 0.05, 100.0
# How closely the un-adjusted qty must match the share count stated in the row's
# Note (whole-share rounded) to confirm the row really is split-adjusted.
SHARE_TOL = 1.0

_NOTE_SHARES_RE = re.compile(r"on\s+([\d,]+(?:\.\d+)?)\s*sh", re.IGNORECASE)


def note_share_count(note: Optional[str]) -> Optional[float]:
    """Actual share count from a 'Cash Div on 2107 Shs' style note, if present."""
    if not note:
        return None
    m = _NOTE_SHARES_RE.search(note)
    if not m:
        return None
    try:
        return float(m.group(1).replace(",", ""))
    except ValueError:
        return None


def _date10(v) -> str:
    return str(v or "")[:10]


def load_split_factors(conn: sqlite3.Connection) -> Dict[str, List[Tuple[str, float]]]:
    """Map symbol -> sorted list of (date10, ratio) for clean forward/reverse splits."""
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    cur.execute(
        """SELECT "Symbol", "Date", "Split Ratio", "Quantity"
           FROM transactions WHERE lower(Type) IN ('split', 'stock split')"""
    )
    out: Dict[str, List[Tuple[str, float]]] = {}
    for r in cur.fetchall():
        sym = r["Symbol"]
        if not sym:
            continue
        ratio = r["Split Ratio"]
        # Fall back to Quantity when Split Ratio is empty and Quantity holds a
        # sane ratio (mirrors the analyzer's Sharebuilder/E*TRADE fallback).
        if ratio is None or abs(ratio) <= 1e-9:
            ratio = r["Quantity"]
        try:
            ratio = float(ratio)
        except (TypeError, ValueError):
            ratio = 0.0
        out.setdefault(sym, []).append((_date10(r["Date"]), ratio))
    for sym in out:
        out[sym].sort()
    return out


def cumulative_factor(
    splits: List[Tuple[str, float]], after_date: str
) -> Tuple[float, bool]:
    """Product of split ratios dated strictly after `after_date`.

    Returns (factor, ok). ok is False if any qualifying split has a ratio
    outside the sane range, meaning we can't safely un-adjust this row.
    """
    factor = 1.0
    for sdate, ratio in splits:
        if sdate > after_date:
            if not (RATIO_MIN <= ratio <= RATIO_MAX):
                return factor, False
            factor *= ratio
    return factor, True


def migrate(db_path: Path, symbol_filter: str | None, apply: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        return 1

    if apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = db_path.parent / "backups" / f"portfolio_pre_div_unadjust_{stamp}.db"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, backup)
        print(f"Backup written: {backup}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    splits = load_split_factors(conn)

    cur = conn.cursor()
    sql = """SELECT * FROM transactions
             WHERE Symbol != '$CASH' AND lower(Type) = 'dividend'"""
    params: list = []
    if symbol_filter:
        sql += " AND Symbol = ?"
        params.append(symbol_filter)
    sql += " ORDER BY Symbol, Date, id"
    cur.execute(sql, params)
    rows = cur.fetchall()

    scanned = fixed = skipped_inconsistent = skipped_ratio = skipped_unconfirmed = 0
    samples: list[tuple] = []
    warnings: list[str] = []

    for row in rows:
        scanned += 1
        sym = row["Symbol"]
        qty = row["Quantity"]
        price = row["Price/Share"]
        total = row["Total Amount"]
        if qty is None or price is None or total is None:
            continue
        qty, price, total = float(qty), float(price), float(total)
        if abs(qty) < 1e-9 or abs(price) < 1e-9:
            continue  # cash-style dividend (amount in Total only) — nothing to un-adjust

        factor, ok = cumulative_factor(splits.get(sym, []), _date10(row["Date"]))
        if not ok:
            skipped_ratio += 1
            warnings.append(
                f"  id={row['id']:>6}  {row['Date'][:10]}  {sym:6}  "
                f"later split ratio out of range — skipped"
            )
            continue
        if abs(factor - 1.0) < 1e-6:
            continue  # no later split — already as-paid

        # Only touch rows whose stored Qty*Price already reconciles to Total;
        # anything else is pre-existing bad data we shouldn't silently rewrite.
        if abs(qty * price - total) > PRODUCT_TOL:
            skipped_inconsistent += 1
            warnings.append(
                f"  id={row['id']:>6}  {row['Date'][:10]}  {sym:6}  "
                f"Qty*Price={qty * price:.2f} != Total={total:.2f} — skipped"
            )
            continue

        unadj_qty = qty / factor

        # Confirmation gate: a row is only safe to un-adjust if we can PROVE it
        # was stored split-adjusted. The E*TRADE dividend import records the
        # actual share count in the Note ("Cash Div on 2107 Shs"); if that count
        # matches the un-adjusted qty (and NOT the stored qty), the stored value
        # is adjusted. Rows without such a note (e.g. EPP/XLE, whose dividend
        # quantities already track raw holdings across their splits) are left
        # untouched — dividing those would corrupt real, as-paid data.
        stated = note_share_count(row["Note"])
        if stated is None or abs(stated - unadj_qty) > SHARE_TOL:
            skipped_unconfirmed += 1
            warnings.append(
                f"  id={row['id']:>6}  {row['Date'][:10]}  {sym:6}  "
                f"note doesn't confirm adjustment (stated={stated}, "
                f"unadj_qty={unadj_qty:.2f}) — skipped"
            )
            continue

        # Use the Note's actual share count as the quantity and derive the rate
        # from the (untouched) Total. This recovers clean, exact values — e.g.
        # 401 @ $2.65 rather than the split-math's 400.9994 @ 2.650004 — and
        # makes Qty*Price reconcile to Total to the cent.
        new_qty = stated
        new_price = round(total / stated, 6)
        fixed += 1
        samples.append(
            (row["id"], row["Date"][:10], sym, qty, price, new_qty, new_price, factor)
        )

        if apply:
            cur.execute(
                'UPDATE transactions SET "Quantity" = ?, "Price/Share" = ? WHERE id = ?',
                (new_qty, new_price, row["id"]),
            )

    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== {mode} — un-adjust dividend Qty/Price to actual as-paid ===")
    print(f"db: {db_path}")
    if symbol_filter:
        print(f"symbol filter: {symbol_filter}")
    print()
    print(f"  dividend rows scanned      : {scanned}")
    print(f"  rows to un-adjust          : {fixed}")
    print(f"  skipped (note unconfirmed) : {skipped_unconfirmed}")
    print(f"  skipped (Qty*Price!=Total) : {skipped_inconsistent}")
    print(f"  skipped (bad split ratio)  : {skipped_ratio}")

    if samples:
        print()
        print("Sample changes (first 15):")
        print(
            f"  {'id':>6}  {'date':10}  {'sym':6}  {'oldQty':>10} @ {'oldPx':>8}"
            f"  ->  {'newQty':>10} @ {'newPx':>8}   x{'f':<4}"
        )
        for s in samples[:15]:
            sid, sdate, ssym, oq, op, nq, npx, f = s
            print(
                f"  {sid:>6}  {sdate:10}  {ssym:6}  {oq:>10.4f} @ {op:>8.4f}"
                f"  ->  {nq:>10.4f} @ {npx:>8.4f}   x{f:<.0f}"
            )
        if len(samples) > 15:
            print(f"  ... and {len(samples) - 15} more")

    if warnings:
        print()
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(w)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    if apply:
        conn.commit()
        print("\nCommitted.")
    else:
        print("\nDry run — no changes written. Pass --apply to commit.")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="path to portfolio.db")
    ap.add_argument(
        "--symbol", default=None, help="restrict to one symbol (default: all)"
    )
    ap.add_argument(
        "--apply", action="store_true", help="commit changes (default: dry run)"
    )
    args = ap.parse_args()
    return migrate(args.db, args.symbol, args.apply)


if __name__ == "__main__":
    sys.exit(main())
