"""
Migrate SET dividends whose Note encodes a withholding-tax amount into:
  1. A gross dividend (Total Amount += tax)
  2. A separate Tax row with negative Total Amount

Cash-neutral — the SET account ends with the same balance.

The original Note has the form `Tax 1,234.56` (with optional comma thousands and
optional trailing label). After migration the dividend Note becomes a marker
that does NOT match the parse pattern, so a second run is a no-op.

Defaults to --dry-run. Pass --apply to actually mutate the database. Auto-backs
up the DB file alongside before applying.
"""
import argparse
import re
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_DB = Path(
    "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/"
    "My Drive/Finance/Investa/data/users/dheematan/portfolio.db"
)
ACCOUNT = "SET"

# Parses `Tax 1,234.56` (with optional comma thousands and optional trailing
# label such as "BEM"). Anchored at start so we never accidentally parse a
# previously-migrated note.
TAX_NOTE_RE = re.compile(r"^Tax\s+([\d,]+(?:\.\d+)?)(?:\s|$)")

MIGRATED_NOTE = (
    "[Migrated] Gross dividend; withholding tax {tax_amount:.2f} "
    "split into a separate Tax entry"
)
NEW_TAX_NOTE = (
    "Withholding tax extracted from {symbol} dividend on {date}"
)


def parse_tax(note: str | None) -> float | None:
    if not note:
        return None
    m = TAX_NOTE_RE.match(note.strip())
    if not m:
        return None
    raw = m.group(1).replace(",", "")
    try:
        return float(raw)
    except ValueError:
        return None


def find_candidates(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    cur.execute(
        """
        SELECT * FROM transactions
        WHERE Account = ?
          AND lower(Type) = 'dividend'
          AND Note IS NOT NULL
          AND Note != ''
        ORDER BY Date
        """,
        (ACCOUNT,),
    )
    return cur.fetchall()


def migrate(db_path: Path, apply: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        return 1

    if apply:
        # Mirror the existing backup naming convention in the user dir.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = db_path.parent / "backups" / f"portfolio_pre_set_tax_split_{stamp}.db"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, backup)
        print(f"Backup written: {backup}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = find_candidates(conn)

    processed = 0
    skipped_unparseable = 0
    skipped_zero = 0
    tax_total = 0.0
    cash_delta = 0.0  # sum of (gross - old) + (-tax) — should always be 0
    preview: list[tuple[int, str, str, float, float, float]] = []  # (id, date, symbol, old, new, tax)

    for row in rows:
        tax = parse_tax(row["Note"])
        if tax is None:
            skipped_unparseable += 1
            continue
        if tax <= 0:
            skipped_zero += 1
            continue

        old_amount = float(row["Total Amount"] or 0.0)
        new_amount = old_amount + tax
        tax_total += tax
        cash_delta += (new_amount - old_amount) + (-tax)  # should be 0

        preview.append((row["id"], row["Date"], row["Symbol"], old_amount, new_amount, tax))

        if apply:
            cur = conn.cursor()
            # Update the dividend to be the gross amount and replace the Note so
            # subsequent runs of this script skip it.
            cur.execute(
                """
                UPDATE transactions
                SET "Total Amount" = ?, Note = ?
                WHERE id = ?
                """,
                (new_amount, MIGRATED_NOTE.format(tax_amount=tax), row["id"]),
            )
            # Insert the matching tax row. user_id is copied so multi-user
            # databases stay scoped to the same owner.
            cur.execute(
                """
                INSERT INTO transactions
                    (Date, Type, Symbol, Quantity, "Price/Share", "Total Amount",
                     Commission, Account, "Split Ratio", Note, "Local Currency",
                     "To Account", "Tags", "ExternalID", user_id)
                VALUES (?, 'Tax', ?, 0, 0, ?, 0, ?, NULL, ?, ?, NULL, NULL, NULL, ?)
                """,
                (
                    row["Date"],
                    row["Symbol"],
                    -tax,
                    ACCOUNT,
                    NEW_TAX_NOTE.format(symbol=row["Symbol"], date=row["Date"]),
                    row["Local Currency"],
                    row["user_id"],
                ),
            )
        processed += 1

    # Header
    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== {mode} — {ACCOUNT} dividend → gross + tax split ===")
    print(f"db: {db_path}")
    print(f"dividends scanned          : {len(rows)}")
    print(f"  to process (tax > 0)     : {processed}")
    print(f"  skipped (Tax 0 or absent): {skipped_zero}")
    print(f"  skipped (unparseable)    : {skipped_unparseable}")
    print(f"total withholding tax (THB): {tax_total:,.2f}")
    print(f"net cash delta             : {cash_delta:,.4f} (must be 0)")
    print()
    print("Sample changes (first 10):")
    print(f"  {'id':>6}  {'date':10}  {'symbol':10}  {'old':>10}  -> {'new':>10}    {'tax':>10}")
    for sample in preview[:10]:
        sid, sdate, sym, old, new, t = sample
        print(f"  {sid:>6}  {sdate:10}  {sym:10}  {old:>10,.2f}  -> {new:>10,.2f}    {t:>10,.2f}")
    if len(preview) > 10:
        print(f"  ... and {len(preview) - 10} more")

    if apply:
        if abs(cash_delta) > 0.001:
            print("ERROR: cash delta non-zero, rolling back", file=sys.stderr)
            conn.rollback()
            conn.close()
            return 2
        conn.commit()
        print("\nCommitted.")
    else:
        print("\nDry run — no changes written. Pass --apply to commit.")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="path to portfolio.db")
    ap.add_argument("--apply", action="store_true", help="commit changes (default: dry run)")
    args = ap.parse_args()
    return migrate(args.db, args.apply)


if __name__ == "__main__":
    sys.exit(main())
