#!/usr/bin/env python3
"""
Merge dheematan (statement-backed brokerages) and kitmatan (historical accounts).

Rules:
  KEEP from dheematan: TD Ameritrade, E*Trade, Morgan Stanley, IBKR Dhee,
                        IBKR Atcha, ING Direct
  REPLACE with kitmatan: Sharebuilder, Penson, SET, Dime!, Webull
    - Exclude Type IN ('buy','sell') WHERE Symbol='$CASH'  (manual-cash artifacts)
  MERGE All Accounts splits: add kitmatan splits not already in dheematan

The result is written into dheematan/portfolio.db after a timestamped backup.
"""

import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DHEE_DB = BASE / "data/users/dheematan/portfolio.db"
KIT_DB  = BASE / "data/users/kitmatan/portfolio.db"

DHEE_USER_ID = 7   # confirmed via SELECT DISTINCT user_id FROM dheematan transactions

# kitmatan account name → canonical dheematan account name
KIT_SOURCE = {
    "Sharebuilder": "Sharebuilder",
    "Penson":       "Penson",
    "SET":          "SET",
    "Dime!":        "Dime!",
    "WeBull":       "Webull",
}

INSERT_SQL = """
INSERT INTO transactions (
    Date, Type, Symbol, Quantity, "Price/Share", "Total Amount",
    Commission, Account, "Split Ratio", Note, "Local Currency",
    "To Account", Tags, ExternalID, user_id
) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
"""


def backup(db_path: Path) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = Path(str(db_path) + f".backup_{ts}")
    shutil.copy2(db_path, dst)
    print(f"Backed up → {dst}")
    return dst


def is_cash_artifact(row_type: str, row_symbol: str) -> bool:
    """True for buy $CASH / sell $CASH manual-cash double-entries."""
    return row_symbol == "$CASH" and row_type.lower() in ("buy", "sell")


def fetch_kit_account(kit_cur, kit_acct: str, dhee_acct: str) -> list[tuple]:
    kit_cur.execute(
        'SELECT Date, Type, Symbol, Quantity, "Price/Share", "Total Amount", '
        'Commission, "Split Ratio", Note, "Local Currency", "To Account", '
        'Tags, ExternalID FROM transactions WHERE Account=?',
        (kit_acct,),
    )
    rows = kit_cur.fetchall()

    kept, skipped = [], 0
    for r in rows:
        date, typ, symbol = r[0], r[1], r[2]
        if is_cash_artifact(typ, symbol):
            skipped += 1
            continue
        kept.append((
            date, typ, symbol, r[3], r[4], r[5],
            r[6], dhee_acct, r[7], r[8], r[9],
            r[10], r[11], r[12], DHEE_USER_ID,
        ))

    print(f"  {kit_acct:15s}: {len(rows):4d} rows → keeping {len(kept):4d} "
          f"(dropped {skipped} buy/sell $CASH)")
    return kept


def merge_splits(dhee_cur, kit_cur):
    """Add kitmatan All Accounts splits that are not already in dheematan."""
    dhee_cur.execute(
        "SELECT Date, Symbol FROM transactions WHERE Account='All Accounts'"
    )
    existing = {(r[0], r[1]) for r in dhee_cur.fetchall()}

    kit_cur.execute(
        'SELECT Date, Type, Symbol, Quantity, "Price/Share", "Total Amount", '
        'Commission, "Split Ratio", Note, "Local Currency", "To Account", '
        'Tags, ExternalID FROM transactions WHERE Account=\'All Accounts\''
    )
    to_add = []
    for r in kit_cur.fetchall():
        date, typ, symbol = r[0], r[1], r[2]
        key = (date, symbol)
        if key in existing:
            continue
        # Normalise type capitalisation to match dheematan convention
        normalised_type = "Split" if typ.lower() == "split" else typ
        to_add.append((
            date, normalised_type, symbol, r[3], r[4], r[5],
            r[6], "All Accounts", r[7], r[8], r[9],
            r[10], r[11], r[12], DHEE_USER_ID,
        ))

    print(f"  All Accounts: adding {len(to_add)} missing splits "
          f"(skipping {kit_cur.rowcount if kit_cur.rowcount >= 0 else '?'} duplicates)")
    return to_add


def main(dry_run: bool = False):
    if not DHEE_DB.exists():
        sys.exit(f"ERROR: dheematan DB not found at {DHEE_DB}")
    if not KIT_DB.exists():
        sys.exit(f"ERROR: kitmatan DB not found at {KIT_DB}")

    if not dry_run:
        backup(DHEE_DB)

    dhee = sqlite3.connect(DHEE_DB)
    kit  = sqlite3.connect(KIT_DB)
    dhee_cur = dhee.cursor()
    kit_cur  = kit.cursor()

    # ── Collect all data to insert ─────────────────────────────────────────
    replacements: dict[str, list[tuple]] = {}
    for kit_acct, dhee_acct in KIT_SOURCE.items():
        replacements[dhee_acct] = fetch_kit_account(kit_cur, kit_acct, dhee_acct)

    splits_to_add = merge_splits(dhee_cur, kit_cur)

    # ── Summary ────────────────────────────────────────────────────────────
    total_in  = sum(len(v) for v in replacements.values()) + len(splits_to_add)
    print(f"\nTotal rows to insert: {total_in}")

    if dry_run:
        print("\n[DRY RUN] No changes written.")
        dhee.close(); kit.close()
        return

    # ── Apply changes atomically ───────────────────────────────────────────
    with dhee:
        for dhee_acct, rows in replacements.items():
            dhee_cur.execute(
                "DELETE FROM transactions WHERE Account=?", (dhee_acct,)
            )
            print(f"  Deleted existing {dhee_acct} rows, inserting {len(rows)}")
            dhee_cur.executemany(INSERT_SQL, rows)

        if splits_to_add:
            dhee_cur.executemany(INSERT_SQL, splits_to_add)
            print(f"  Inserted {len(splits_to_add)} All Accounts splits")

    dhee.close()
    kit.close()

    # ── Verify ─────────────────────────────────────────────────────────────
    print("\n── Post-merge account counts ──────────────────────────────────")
    conn = sqlite3.connect(DHEE_DB)
    for row in conn.execute(
        "SELECT Account, COUNT(*) FROM transactions GROUP BY Account ORDER BY Account"
    ):
        print(f"  {row[0]:25s}: {row[1]}")
    conn.close()
    print("\nMerge complete.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("=== DRY RUN (pass no flags to apply) ===\n")
    main(dry_run=dry_run)
