"""
Re-derive Total Amount for stock Buy / Sell / Short Sell / Buy To Cover rows
whose fee sign is reversed, to match the codebase's canonical convention:

  Buy           : Total = Qty * Price + Commission   (cash out, fee on top)
  Sell          : Total = Qty * Price - Commission   (cash in, fee deducted)
  Short Sell    : Total = Qty * Price - Commission   (proceeds, fee deducted)
  Buy To Cover  : Total = Qty * Price + Commission   (cost, fee on top)

The Sharebuilder / Penson legacy CSV import flipped the fee sign for these
rows (Buy stored gross - fee, Sell stored gross + fee). The engine's auto-cash
branch and IBKR importer both assume the canonical convention, so the legacy
values quietly inflate cumulative returns by 2 x commission per trade.

Also updates any paired Auto-Generated $CASH rows so the account's external
cash flow stays consistent with the stock row's new Total Amount. The matching
key is (Date, Account, $CASH symbol, Note contains stock symbol, original
Total Amount). If the script can't find a paired cash row, it still updates
the stock row but warns.

Defaults to --dry-run. Pass --apply to commit. Auto-backs up the DB.

Targets dheematan's portfolio.db by default. Pass --db to override.
"""
import argparse
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List

DEFAULT_DB = Path(
    "/Users/kmatan/Library/CloudStorage/GoogleDrive-kittiwit@gmail.com/"
    "My Drive/Finance/Investa/data/users/dheematan/portfolio.db"
)
TOLERANCE = 0.02  # dollars — covers float rounding in the CSV source

BUY_LIKE = {"buy", "buy to cover"}
SELL_LIKE = {"sell", "short sell"}


def canonical_total(tx_type: str, qty: float, price: float, commission: float) -> float:
    """Return the codebase-canonical Total Amount for a stock trade row."""
    gross = qty * price
    if tx_type in BUY_LIKE:
        return gross + commission
    if tx_type in SELL_LIKE:
        return gross - commission
    raise ValueError(f"unsupported type: {tx_type}")


def detect_reversed(tx_type: str, qty: float | None, price: float | None,
                    commission: float | None, total: float | None) -> bool:
    """True iff Total Amount has the fee sign reversed vs canonical."""
    if commission is None or commission <= 0:
        return False
    if qty is None or price is None or total is None:
        return False
    gross = qty * price
    if tx_type in BUY_LIKE:
        # Canonical Buy: total = gross + commission. Reversed: total = gross - commission.
        return abs((total + commission) - gross) < TOLERANCE
    if tx_type in SELL_LIKE:
        # Canonical Sell: total = gross - commission. Reversed: total = gross + commission.
        return abs((total - commission) - gross) < TOLERANCE
    return False


def find_paired_cash_rows(
    conn: sqlite3.Connection,
    date: str,
    account: str,
    symbol: str,
    old_total: float,
    user_id,
) -> List[sqlite3.Row]:
    """Find $CASH rows paired to a stock trade on this date.

    Matches on (Date, Account, Symbol=$CASH, |Total Amount| ≈ old stock total).
    Note is intentionally NOT part of the match: kitmatan-style Penson uses
    Auto-generated notes mentioning the symbol; dheematan-style short trades
    use unannotated rows with possibly negative Total Amount. Any $CASH row on
    the same day for the same account with a matching magnitude is treated as
    a settlement leg of this trade. Rows whose Note clearly references a
    DIFFERENT symbol are filtered out so we don't cross over to a same-day
    same-amount trade in another symbol.
    """
    cur = conn.cursor()
    cur.row_factory = sqlite3.Row
    user_clause = "user_id IS NULL" if user_id is None else "user_id = ?"
    params: list = [date, account, old_total, TOLERANCE]
    if user_id is not None:
        params.append(user_id)
    cur.execute(
        f"""
        SELECT * FROM transactions
        WHERE Date = ?
          AND Account = ?
          AND Symbol = '$CASH'
          AND ABS(ABS("Total Amount") - ?) < ?
          AND {user_clause}
        """,
        params,
    )
    rows = cur.fetchall()

    # Filter out rows whose Note explicitly names a different stock symbol —
    # protects against same-day same-amount coincidences between two stocks.
    filtered: List[sqlite3.Row] = []
    for r in rows:
        note = (r["Note"] or "").lower()
        if note and "auto-generated" in note:
            # Auto-generated notes always embed the related stock symbol.
            if symbol.lower() not in note:
                continue
        filtered.append(r)
    return filtered


def migrate(db_path: Path, account_filter: str | None, apply: bool) -> int:
    if not db_path.exists():
        print(f"ERROR: database not found at {db_path}", file=sys.stderr)
        return 1

    if apply:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = db_path.parent / "backups" / f"portfolio_pre_fee_sign_{stamp}.db"
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(db_path, backup)
        print(f"Backup written: {backup}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cur = conn.cursor()
    base_sql = """
        SELECT * FROM transactions
        WHERE Symbol != '$CASH'
          AND lower(Type) IN ('buy', 'sell', 'short sell', 'buy to cover')
    """
    if account_filter:
        cur.execute(base_sql + " AND Account = ? ORDER BY Date, id",
                    (account_filter,))
    else:
        cur.execute(base_sql + " ORDER BY Account, Date, id")
    rows = cur.fetchall()

    stats: dict[str, dict] = {}
    samples: list[tuple] = []
    warnings: list[str] = []

    for row in rows:
        tx_type = str(row["Type"] or "").strip().lower()
        qty = row["Quantity"]
        price = row["Price/Share"]
        comm = row["Commission"]
        total = row["Total Amount"]

        bucket = stats.setdefault(
            row["Account"],
            {"scanned": 0, "reversed": 0, "fixed_stock": 0, "fixed_cash": 0,
             "no_pair": 0, "many_pair": 0},
        )
        bucket["scanned"] += 1

        if not detect_reversed(tx_type, qty, price, comm, total):
            continue
        bucket["reversed"] += 1

        new_total = canonical_total(tx_type, float(qty), float(price), float(comm))

        paired = find_paired_cash_rows(
            conn, row["Date"], row["Account"], row["Symbol"],
            float(total), row["user_id"],
        )

        # Expected pair count: 2 for Buy/Sell (Deposit+Sell or Buy+Withdrawal).
        # Short Sell / Buy To Cover historically have no auto-cash pair (verified
        # for Penson) — that's expected, not a warning.
        if tx_type in {"buy", "sell"} and len(paired) == 0:
            bucket["no_pair"] += 1
            warnings.append(
                f"  id={row['id']:>6}  {row['Date']}  {row['Account']:14}  "
                f"{row['Symbol']:6}  {tx_type:4}  no Auto-Generated $CASH pair"
            )
        if len(paired) > 2:
            bucket["many_pair"] += 1
            warnings.append(
                f"  id={row['id']:>6}  {row['Date']}  {row['Account']:14}  "
                f"{row['Symbol']:6}  {tx_type:4}  found {len(paired)} paired rows (expected ≤2)"
            )

        samples.append((
            row["id"], row["Date"], row["Account"], row["Symbol"],
            tx_type, float(total), new_total, len(paired),
        ))

        if apply:
            cur.execute(
                """UPDATE transactions SET "Total Amount" = ? WHERE id = ?""",
                (new_total, row["id"]),
            )
            bucket["fixed_stock"] += 1
            for p in paired:
                # Preserve sign of the existing cash-row Total Amount. Most are
                # positive, but dheematan's short-trade Withdrawals store it as
                # a negative number; magnitude is what the engine reads.
                old_cash = float(p["Total Amount"] or 0.0)
                sign = -1.0 if old_cash < 0 else 1.0
                new_cash = sign * abs(new_total)
                cur.execute(
                    """UPDATE transactions SET "Total Amount" = ? WHERE id = ?""",
                    (new_cash, p["id"]),
                )
                bucket["fixed_cash"] += 1

    # --- Report ---
    mode = "APPLY" if apply else "DRY RUN"
    print(f"=== {mode} — re-derive canonical Total Amount (fee-sign fix) ===")
    print(f"db: {db_path}")
    if account_filter:
        print(f"account filter: {account_filter}")
    print()

    if not any(s["reversed"] for s in stats.values()):
        print("Nothing to do — no reversed-fee rows detected.")
        conn.close()
        return 0

    hdr = (f"  {'account':16} {'scanned':>8} {'reversed':>9} "
           f"{'fix-stock':>10} {'fix-cash':>9} {'no-pair':>8} {'many-pair':>10}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    total_rev = total_nopair = total_manypair = 0
    for acct, s in sorted(stats.items()):
        if s["reversed"] == 0:
            continue
        print(f"  {acct:16} {s['scanned']:>8} {s['reversed']:>9} "
              f"{s['fixed_stock']:>10} {s['fixed_cash']:>9} "
              f"{s['no_pair']:>8} {s['many_pair']:>10}")
        total_rev += s["reversed"]
        total_nopair += s["no_pair"]
        total_manypair += s["many_pair"]
    print()
    print(f"  reversed rows total : {total_rev}")
    print(f"  missing cash pair   : {total_nopair}  (Buy/Sell only; short trades excluded)")
    print(f"  ambiguous cash pair : {total_manypair}")

    print()
    print("Sample changes (first 12):")
    print(f"  {'id':>6}  {'date':10}  {'account':14}  {'sym':6}  {'type':12}  "
          f"{'old':>10} -> {'new':>10}   paired")
    for s in samples[:12]:
        sid, sdate, sacc, ssym, stype, sold, snew, npaired = s
        print(f"  {sid:>6}  {sdate:10}  {sacc:14}  {ssym:6}  {stype:12}  "
              f"{sold:>10,.2f} -> {snew:>10,.2f}   {npaired}")
    if len(samples) > 12:
        print(f"  ... and {len(samples) - 12} more")

    if warnings:
        print()
        print(f"WARNINGS ({len(warnings)}):")
        for w in warnings[:20]:
            print(w)
        if len(warnings) > 20:
            print(f"  ... and {len(warnings) - 20} more")

    if apply:
        if total_manypair > 0:
            print()
            print("ERROR: ambiguous cash pairs detected — rolling back.", file=sys.stderr)
            conn.rollback()
            conn.close()
            return 2
        conn.commit()
        print()
        print("Committed.")
    else:
        print()
        print("Dry run — no changes written. Pass --apply to commit.")

    conn.close()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DEFAULT_DB, help="path to portfolio.db")
    ap.add_argument("--account", default=None,
                    help="restrict to one account (default: scan all)")
    ap.add_argument("--apply", action="store_true",
                    help="commit changes (default: dry run)")
    args = ap.parse_args()
    return migrate(args.db, args.account, args.apply)


if __name__ == "__main__":
    sys.exit(main())
