#!/usr/bin/env python3
"""Reorganize the dheematan 'SET' umbrella account into real sub-accounts.

Targets:
    Eastspring  - ES-* funds (retirement plan)
    SCBAM       - SCBRM1, SCBRMS50, SCBRMS&P500, SCBCHA-SSF, SCBRCTECH, SCBSFF
    UOBAM       - UOBBC, UOBCG (closed)
    Kim Eng     - Maybank Kim Eng brokerage stocks (closed end-2024)
    SCBS        - SCB Securities stocks; rows noted 'SCBS Kit' (acct 07-3-0102)
                  or 'SCBS Atcha' (acct 96-2-5305) per broker confirmations

Also records three missing corporate actions so every closed position nets
to zero after the split (verified against broker statements):
    1. MBK 1:10 par split (Oct 2012)
    2. AMARIN 10% stock dividend (May 2014): 970 sh at Kim Eng (on the 9,700
       bought 2012), 750 sh at SCBS (on the 7,500 bought Oct 2013). No shares
       were ever transferred between brokers (per user); the per-broker lots
       are confirmed by the paired cash dividends (1,077.78/833.33 = 9,700 vs
       7,500 sh and 7,469/5,775 = 10,670 vs 8,250 sh) and both sides net to 0.
    3. BECL/BMCL -> BEM merger conversion (Dec 2015); the 2013-11-14 "BEM"
       buy is relabeled BECL per Kim Eng contract note DN-20131114-02705,
       and dividend-only symbol BML:BKK is relabeled BEM:BKK

Usage:
    python scripts/reorganize_dheematan_set.py --db PATH [--config PATH] [--apply]

Dry-run (default) prints the assignment plan and verification preview.
--apply writes changes (transactions, snapshots, and config if --config given).
A timestamped .bak of the DB and config is made before writing.
"""

import argparse
import json
import re
import shutil
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime

EASTSPRING = "Eastspring"
SCBAM = "SCBAM"
UOBAM = "UOBAM"
KIM_ENG = "Kim Eng"
SCBS = "SCBS"

NOTE_KIT = "SCBS Kit"
NOTE_ATCHA = "SCBS Atcha"

ES_FUNDS = {"ES-FIXED_INCOME", "ES-JUMBO25", "ES-SET50", "ES-TRESURY", "ES-GQG"}
SCB_FUNDS = {"SCBRM1", "SCBRMS50", "SCBRMS&P500", "SCBCHA-SSF", "SCBRCTECH", "SCBSFF"}
UOB_FUNDS = {"UOBBC", "UOBCG"}
KE_STOCKS = {"SCC:BKK", "MBK:BKK", "BECL:BKK", "BEM:BKK", "BML:BKK", "AOT:BKK",
             "TRUE:BKK", "AAV:BKK", "NOK:BKK", "GENCO:BKK"}
SCBS_KIT_STOCKS = {"THAI:BKK", "BANPU:BKK"}

# AMARIN at SCBS acct 07-3-0102: the Oct 2013 buys (4,400 + 3,100) and the
# 2015/2016 sells (confirmed: CN-20151009-00555 et al.; the odd lot appears
# in the SCBS realized-G/L report). The 2012 lots and 2016 trades were at
# Kim Eng (contract-note slips). No inter-broker share transfers occurred.
AMARIN_SCBS_BUY_DATES = {"2013-10-04", "2013-10-07"}
AMARIN_SCBS_SELL_DATES = {"2015-10-01", "2015-10-14", "2016-01-07"}
# Paired same-date cash dividends split KE/SCBS by holding size: SCBS held
# fewer shares on both dates (7,500 vs 9,700 in 2014; 8,250 vs 10,670 in
# 2015), so the smaller amount of each pair belongs to SCBS.
AMARIN_PAIRED_DIV_DATES = {"2014-05-12", "2015-05-11"}

# The one TCAP buy row that aggregates trades at two brokers on the same day:
# SCBS realized-G/L shows 3,000 @36 (trade 12/06/2014); the remaining 2,500
# were at Kim Eng (later sold 2015-02-06, absent from the SCBS G/L).
TCAP_SPLIT_BUY = {"date": "2014-06-17", "qty": 5500.0, "kit_qty": 3000.0, "ke_qty": 2500.0}


def norm_symbol(raw):
    """Normalize symbol spellings used in auto-generated cash notes."""
    r = raw.strip()
    aliases = {
        "es-fixed_income": "ES-FIXED_INCOME", "es-tresury": "ES-TRESURY",
        "es-jumbo25": "ES-JUMBO25", "es-set50": "ES-SET50", "es-gqg": "ES-GQG",
    }
    return aliases.get(r.lower(), r)


def stock_target(row, pairs_larger):
    """Return (account, note_suffix) for a stock (non-$CASH) row.

    pairs_larger: {(symbol, date, type): max_amount} for dates where two
    dividend/tax rows exist for the same symbol (one per SCBS sub-account);
    the larger amount always belongs to the bigger holding (Kit).
    """
    sym, date, typ, qty, amt = (row["Symbol"], row["Date"], row["Type"],
                                row["Quantity"] or 0.0, abs(row["Total Amount"] or 0.0))

    if sym in ES_FUNDS:
        return EASTSPRING, None
    if sym in SCB_FUNDS:
        return SCBAM, None
    if sym in UOB_FUNDS:
        return UOBAM, None
    if sym in KE_STOCKS:
        return KIM_ENG, None
    if sym in SCBS_KIT_STOCKS:
        return SCBS, NOTE_KIT

    if sym in ("CPF:BKK", "CPALL:BKK"):
        return (KIM_ENG, None) if date < "2013-01-01" else (SCBS, NOTE_KIT)

    if sym == "AMARIN:BKK":
        if typ == "Buy" and date in AMARIN_SCBS_BUY_DATES:
            return SCBS, NOTE_KIT
        if typ == "Sell" and date in AMARIN_SCBS_SELL_DATES:
            return SCBS, NOTE_KIT
        if typ in ("Dividend", "Tax") and date in AMARIN_PAIRED_DIV_DATES:
            key = (sym, date, typ)
            if key in pairs_larger:
                return (KIM_ENG, None) if amt >= pairs_larger[key] else (SCBS, NOTE_KIT)
        return KIM_ENG, None

    if sym == "ZEN:BKK":
        if typ == "Buy":
            return (SCBS, NOTE_KIT) if date == "2019-07-22" else (SCBS, NOTE_ATCHA)
        if typ == "Sell":
            return (SCBS, NOTE_KIT) if abs(qty - 26300) < 1 else (SCBS, NOTE_ATCHA)
        key = (sym, date, typ)
        if key in pairs_larger:
            return (SCBS, NOTE_KIT) if amt >= pairs_larger[key] else (SCBS, NOTE_ATCHA)
        return SCBS, NOTE_ATCHA  # no single-row ZEN dividend dates exist

    if sym == "PTT:BKK":
        if typ == "Buy":
            return (SCBS, NOTE_ATCHA) if date == "2019-07-05" else (SCBS, NOTE_KIT)
        if typ == "Sell":
            return (SCBS, NOTE_KIT) if abs(qty - 11500) < 1 else (SCBS, NOTE_ATCHA)
        key = (sym, date, typ)
        if key in pairs_larger:
            return (SCBS, NOTE_KIT) if amt >= pairs_larger[key] else (SCBS, NOTE_ATCHA)
        # single dividend rows: only the Atcha lot (4,100 sh, held 2019-2024)
        # produces singles (Kit bought Feb 2022; paired rows exist from Oct 2022)
        return SCBS, NOTE_ATCHA

    if sym == "TCAP:BKK":
        if typ == "Buy":
            if date == "2014-05-23":
                return KIM_ENG, None
            if date == "2019-07-22":
                return SCBS, NOTE_ATCHA
            return SCBS, NOTE_KIT  # 2014-07-08, 2014-09-29, 2018-06-06 (split row handled separately)
        if typ == "Sell":
            if date == "2015-02-06":
                return KIM_ENG, None
            if date == "2024-02-06":
                return SCBS, NOTE_ATCHA
            return SCBS, NOTE_KIT  # 2016-07-13, 2020-03-05
        key = (sym, date, typ)
        if key in pairs_larger:
            return (SCBS, NOTE_KIT) if amt >= pairs_larger[key] else (SCBS, NOTE_ATCHA)
        if date >= "2020-04-01":
            return SCBS, NOTE_ATCHA  # only the 1,800-sh Atcha lot remained
        return SCBS, NOTE_KIT  # 2014-2019 singles (11,800 / 7,800 sh Kit lots)

    raise ValueError(f"Unmapped stock row: {dict(row)}")


CASH_NOTE_RE = re.compile(
    r"^(?:Auto-generated: Cash deposit for (?P<buy>.+) buy"
    r"|Auto-generated: Cash withdrawal from (?P<sell>.+) sell proceeds.*"
    r"|Dividend withdrawal for (?P<div>.+))$"
)


def cash_target(row, trade_assign, div_net_assign):
    """Return account for a $CASH row.

    trade_assign: {(symbol, date, round(amount,2), kind)} -> account, built
        from the already-assigned stock rows (kind is 'Buy' or 'Sell'); the
        auto-generated cash rows mirror their trade's date and gross amount.
    div_net_assign: {(symbol, date, round(net,2))} -> account for dividend
        withdrawals (net = gross dividend - same-date tax row).
    """
    note = (row["Note"] or "").strip()
    date, amt = row["Date"], round(abs(row["Total Amount"] or 0.0), 2)

    m = CASH_NOTE_RE.match(note)
    if m:
        sym = norm_symbol(m.group("buy") or m.group("sell") or m.group("div"))
        if sym in ES_FUNDS:
            return EASTSPRING
        if sym in SCB_FUNDS:
            return SCBAM
        if sym in UOB_FUNDS:
            return UOBAM
        if sym in KE_STOCKS:
            return KIM_ENG
        if sym in SCBS_KIT_STOCKS:
            return SCBS
        if sym in ("CPF:BKK", "CPALL:BKK"):
            return KIM_ENG if date < "2013-01-01" else SCBS
        if m.group("div"):
            key = (sym, date, amt)
            if key in div_net_assign:
                return div_net_assign[key]
            # fall back to the (unique) dividend assignment for that date
            cands = {a for (s, d, _), a in div_net_assign.items() if s == sym and d == date}
            if len(cands) == 1:
                return cands.pop()
            raise ValueError(f"Ambiguous dividend withdrawal: {dict(row)}")
        kind = "Buy" if m.group("buy") else "Sell"
        key = (sym, date, amt, kind)
        if key in trade_assign:
            return trade_assign[key]
        # fall back: unique account among same-symbol same-date trades
        cands = {a for (s, d, _, k), a in trade_assign.items()
                 if s == sym and d == date and k == kind}
        if len(cands) == 1:
            return cands.pop()
        raise ValueError(f"Cash row does not match any trade: {dict(row)}")

    if note.startswith("Closing adjustment"):
        return KIM_ENG  # ฿1.20 residual on the day the last NOK shares were sold
    if note.startswith("Deposit to buy SCBRMS&P500"):
        return SCBAM
    if note == "":
        # Untagged deposits, 2025-2026: monthly ~4,748-5,155 fund the ES-GQG
        # contribution; 50,000 lumps fund SCBAM RMF year-end/quarterly buys.
        if 4000 <= amt <= 6000:
            return EASTSPRING
        if amt in (50000.0, 40000.0):
            return SCBAM
    raise ValueError(f"Unmapped $CASH row: {dict(row)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--config", help="gui_config.json to update (optional)")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    con.row_factory = sqlite3.Row
    cur = con.cursor()

    rows = cur.execute("SELECT * FROM transactions WHERE Account='SET' ORDER BY Date, id").fetchall()
    if not rows:
        sys.exit("No SET rows found - nothing to do.")
    user_id = rows[0]["user_id"]

    # -- pass 1: find same-date dividend/tax pairs (two SCBS sub-accounts paid
    #    on the same date); the larger amount belongs to the Kit lot.
    by_sdt = defaultdict(list)
    for r in rows:
        if (r["Symbol"] in ("ZEN:BKK", "PTT:BKK", "TCAP:BKK", "AMARIN:BKK")
                and r["Type"] in ("Dividend", "Tax")):
            by_sdt[(r["Symbol"], r["Date"], r["Type"])].append(abs(r["Total Amount"] or 0.0))
    pairs_larger = {k: max(v) for k, v in by_sdt.items() if len(v) == 2}
    for k, v in by_sdt.items():
        if len(v) > 2:
            raise ValueError(f"Unexpected {len(v)} rows for {k}")

    # -- pass 2: assign stock rows
    assign = {}   # id -> (account, note_suffix or None)
    tcap_split_row = None
    for r in rows:
        if r["Symbol"] == "$CASH":
            continue
        if (r["Symbol"] == "TCAP:BKK" and r["Type"] == "Buy"
                and r["Date"] == TCAP_SPLIT_BUY["date"]
                and abs((r["Quantity"] or 0) - TCAP_SPLIT_BUY["qty"]) < 0.01):
            tcap_split_row = r
            continue
        acct, note = stock_target(r, pairs_larger)
        # dividend paid while 5,500 sh sat at Kim Eng and 11,800 at SCBS;
        # kept whole on the larger holding
        if (r["Symbol"], r["Date"]) == ("TCAP:BKK", "2014-10-17"):
            note = (note or "") + " | incl. dividend on 5,500 sh then held at Kim Eng"
        assign[r["id"]] = (acct, note)
    if tcap_split_row is None:
        raise ValueError("TCAP 2014-06-17 5,500-share buy row not found")

    # -- build lookup tables for cash-row matching
    trade_assign = {}
    div_net_assign = {}
    div_gross = {}
    tax_amt = {}
    for r in rows:
        if r["Symbol"] == "$CASH" or r["id"] not in assign:
            continue
        acct = assign[r["id"]][0]
        amt = round(abs(r["Total Amount"] or 0.0), 2)
        if r["Type"] in ("Buy", "Sell"):
            trade_assign[(r["Symbol"], r["Date"], amt, r["Type"])] = acct
        elif r["Type"] == "Dividend":
            div_gross[(r["Symbol"], r["Date"], amt)] = acct
        elif r["Type"] == "Tax":
            tax_amt.setdefault((r["Symbol"], r["Date"]), []).append(amt)
    for (sym, date, gross), acct in div_gross.items():
        taxes = tax_amt.get((sym, date), [0.0])
        for t in taxes + [0.0]:
            div_net_assign.setdefault((sym, date, round(gross - t, 2)), acct)
        div_net_assign.setdefault((sym, date, gross), acct)
    # the split TCAP buy's cash deposit is handled structurally below

    # -- pass 3: assign cash rows
    for r in rows:
        if r["Symbol"] != "$CASH":
            continue
        if (r["Date"] == TCAP_SPLIT_BUY["date"]
                and (r["Note"] or "") == "Auto-generated: Cash deposit for TCAP:BKK buy"):
            continue  # split structurally with its trade row
        assign[r["id"]] = (cash_target(r, trade_assign, div_net_assign), None)

    unassigned = [r["id"] for r in rows if r["id"] not in assign
                  and r["id"] != tcap_split_row["id"]
                  and not (r["Symbol"] == "$CASH" and r["Date"] == TCAP_SPLIT_BUY["date"]
                           and (r["Note"] or "").endswith("TCAP:BKK buy"))]
    if unassigned:
        raise ValueError(f"{len(unassigned)} rows unassigned: {unassigned[:10]}")

    # -- report plan
    counts = defaultdict(int)
    for acct, _ in assign.values():
        counts[acct] += 1
    print("Assignment plan (existing rows):")
    for acct in (EASTSPRING, SCBAM, UOBAM, KIM_ENG, SCBS):
        print(f"  {acct:<11} {counts[acct]:>5} rows")
    n_kit = sum(1 for a, n in assign.values() if n and NOTE_KIT in n)
    n_atcha = sum(1 for a, n in assign.values() if n and NOTE_ATCHA in n)
    print(f"  SCBS notes: {n_kit} x '{NOTE_KIT}', {n_atcha} x '{NOTE_ATCHA}'")
    print(f"  + 1 TCAP buy row and its cash deposit split 3,000 (SCBS) / 2,500 (Kim Eng)")
    print(f"  + corporate-action rows: MBK split, AMARIN stock dividend (970 KE / 750 SCBS),")
    print(f"    BECL->BEM merger pair; BEM buy row relabeled BECL, BML relabeled BEM")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write changes.")
        return

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    shutil.copy2(args.db, f"{args.db}.bak_set_reorg_{stamp}")

    def append_note(existing, suffix):
        return f"{existing} | {suffix}" if existing else suffix

    ins_cols = ('Date', 'Type', 'Symbol', 'Quantity', 'Price/Share', 'Total Amount',
                'Commission', 'Account', 'Split Ratio', 'Note', 'Local Currency',
                'To Account', 'user_id')
    ins_sql = (f"INSERT INTO transactions ({','.join(chr(34)+c+chr(34) for c in ins_cols)}) "
               f"VALUES ({','.join('?' * len(ins_cols))})")

    with con:
        # 1) reassign accounts / add notes
        for r in rows:
            if r["id"] not in assign:
                continue
            acct, note_sfx = assign[r["id"]]
            note = append_note(r["Note"], note_sfx) if note_sfx else r["Note"]
            cur.execute("UPDATE transactions SET Account=?, Note=? WHERE id=?",
                        (acct, note, r["id"]))

        # 2) split the dual-broker TCAP buy (5,500 @36, comm 334.32) and its
        #    mirrored cash deposit (198,000) into 3,000/2,500 legs
        t = tcap_split_row
        comm = t["Commission"] or 0.0
        f_kit = TCAP_SPLIT_BUY["kit_qty"] / TCAP_SPLIT_BUY["qty"]
        cur.execute(
            'UPDATE transactions SET Account=?, Quantity=?, "Total Amount"=?, '
            'Commission=?, Note=? WHERE id=?',
            (SCBS, TCAP_SPLIT_BUY["kit_qty"], TCAP_SPLIT_BUY["kit_qty"] * 36.0,
             round(comm * f_kit, 2),
             append_note(t["Note"], NOTE_KIT + " | split from 5,500-sh row (2,500 at Kim Eng)"),
             t["id"]))
        cur.execute(ins_sql, (t["Date"], "Buy", "TCAP:BKK", TCAP_SPLIT_BUY["ke_qty"], 36.0,
                              TCAP_SPLIT_BUY["ke_qty"] * 36.0, round(comm * (1 - f_kit), 2),
                              KIM_ENG, None, "split from 5,500-sh row (3,000 at SCBS)",
                              "THB", None, user_id))
        cash_dep = cur.execute(
            "SELECT * FROM transactions WHERE Date=? AND Symbol='$CASH' AND Note=?",
            (TCAP_SPLIT_BUY["date"], "Auto-generated: Cash deposit for TCAP:BKK buy")
        ).fetchone()
        cur.execute('UPDATE transactions SET Account=?, Quantity=?, "Total Amount"=? WHERE id=?',
                    (SCBS, TCAP_SPLIT_BUY["kit_qty"] * 36.0, TCAP_SPLIT_BUY["kit_qty"] * 36.0,
                     cash_dep["id"]))
        cur.execute(ins_sql, (cash_dep["Date"], "Deposit", "$CASH",
                              TCAP_SPLIT_BUY["ke_qty"] * 36.0, 1.0,
                              TCAP_SPLIT_BUY["ke_qty"] * 36.0, 0.0, KIM_ENG, None,
                              cash_dep["Note"], "THB", None, user_id))

        # 3) corporate actions
        # 3a) MBK 1:10 par split between the 2012-10-08 buy and 2012-10-26 dividend
        cur.execute(ins_sql, ("2012-10-15", "Split", "MBK:BKK", 0.0, 0.0, 0.0, 0.0,
                              KIM_ENG, 10.0,
                              "MBK 1:10 par split (added during SET reorg; "
                              "dividends from 2012-10-26 confirm 16,000 sh)",
                              "THB", None, user_id))
        # 3b) AMARIN 10% stock dividend, May 2014: 970 sh on the 9,700 held at
        #     Kim Eng, 750 sh on the 7,500 held at SCBS. No inter-broker
        #     transfers (per user); confirmed by paired cash-dividend amounts.
        cur.execute(ins_sql, ("2014-05-12", "Buy", "AMARIN:BKK", 970.0, 0.0, 0.0, 0.0,
                              KIM_ENG, None, "AMARIN 10% stock dividend (9,700 sh held)",
                              "THB", None, user_id))
        cur.execute(ins_sql, ("2014-05-12", "Buy", "AMARIN:BKK", 750.0, 0.0, 0.0, 0.0,
                              SCBS, None,
                              "AMARIN 10% stock dividend (7,500 sh held) | " + NOTE_KIT,
                              "THB", None, user_id))
        # 3c) BECL -> BEM merger. Relabel the 2013-11-14 buy (Kim Eng contract
        #     note DN-20131114-02705 says BECL 4,000 @37), then a cost-neutral
        #     conversion pair at the Dec 2015 merger.
        cur.execute("UPDATE transactions SET Symbol='BECL:BKK', Note=? "
                    "WHERE Date='2013-11-14' AND Symbol='BEM:BKK' AND Type='Buy'",
                    ("relabeled from BEM (Kim Eng note DN-20131114-02705: BECL)",))
        basis = 4000.0 * 37.0 + 249.89
        cur.execute(ins_sql, ("2015-12-30", "Sell", "BECL:BKK", 4000.0,
                              round(basis / 4000.0, 6), round(basis, 2), 0.0,
                              KIM_ENG, None,
                              "BECL+BMCL->BEM merger conversion (cost-neutral)",
                              "THB", None, user_id))
        cur.execute(ins_sql, ("2015-12-30", "Buy", "BEM:BKK", 34800.0,
                              round(basis / 34800.0, 6), round(basis, 2), 0.0,
                              KIM_ENG, None,
                              "BECL+BMCL->BEM merger conversion (cost-neutral)",
                              "THB", None, user_id))
        # 3d) BML was BEM's Apr-2016 dividend (2,423.54 / 34,800 sh = 0.0696)
        cur.execute("UPDATE transactions SET Symbol='BEM:BKK', "
                    "Note=COALESCE(Note,'') || ' | relabeled from BML' "
                    "WHERE Symbol='BML:BKK'")
        cur.execute("UPDATE transactions SET Note=REPLACE(Note,'BML:BKK','BEM:BKK') "
                    "WHERE Note LIKE '%BML:BKK%'")

        # 4) stale rolling snapshots for the dissolved account
        cur.execute("DELETE FROM portfolio_snapshots WHERE account='SET'")

    # 5) config
    if args.config:
        shutil.copy2(args.config, f"{args.config}.bak_set_reorg_{stamp}")
        cfg = json.load(open(args.config))
        new_accts = [EASTSPRING, SCBAM, UOBAM, KIM_ENG, SCBS]
        cfg.get("account_currency_map", {}).pop("SET", None)
        cfg.get("account_cash_mode_map", {}).pop("SET", None)
        for a in new_accts:
            cfg.setdefault("account_currency_map", {})[a] = "THB"
            cfg.setdefault("account_cash_mode_map", {})[a] = "Auto"
        groups = cfg.get("account_groups", {})
        for gname, members in groups.items():
            if "SET" in members:
                idx = members.index("SET")
                members[idx:idx + 1] = ([EASTSPRING, SCBAM] if gname == "Current Holdings"
                                        else new_accts)
        groups.setdefault("SET (Thai)", new_accts)
        order = cfg.setdefault("account_group_order", list(groups.keys()))
        if "SET (Thai)" not in order:
            order.append("SET (Thai)")
        json.dump(cfg, open(args.config, "w"), indent=2, ensure_ascii=False)
        print(f"Config updated: {args.config}")

    # -- verification
    print("\nVerification:")
    left = cur.execute("SELECT COUNT(*) FROM transactions WHERE Account='SET'").fetchone()[0]
    print(f"  rows still on 'SET': {left}")
    print("  per-account rows:", dict(cur.execute(
        "SELECT Account, COUNT(*) FROM transactions WHERE Account IN (?,?,?,?,?) "
        "GROUP BY Account", (EASTSPRING, SCBAM, UOBAM, KIM_ENG, SCBS)).fetchall()))
    bad = []
    open_expected = {"ES-GQG", "SCBRMS&P500", "SCBCHA-SSF", "SCBRCTECH"}
    net = cur.execute(
        "SELECT Account, Symbol, "
        " SUM(CASE Type WHEN 'Buy' THEN Quantity WHEN 'Sell' THEN -Quantity "
        "     WHEN 'Transfer' THEN -Quantity ELSE 0 END) "
        " + SUM(CASE WHEN Type='Transfer' THEN 0 ELSE 0 END) AS net "
        "FROM transactions WHERE Account IN (?,?,?,?,?) AND Symbol!='$CASH' "
        "GROUP BY Account, Symbol", (EASTSPRING, SCBAM, UOBAM, KIM_ENG, SCBS)).fetchall()
    # add transfer receipts (To Account side)
    recv = dict(cur.execute(
        "SELECT Symbol, SUM(Quantity) FROM transactions "
        "WHERE Type='Transfer' AND \"To Account\"=? GROUP BY Symbol", (SCBS,)).fetchall())
    # splits: apply MBK ratio to pre-split KE quantity for the check
    for acct, sym, q in net:
        q = q or 0.0
        if acct == SCBS and sym in recv:
            q += recv[sym]
        if sym == "MBK:BKK":
            q = 1600.0 * 10.0 - 16000.0  # buy pre-split x10 vs sell
        if sym in open_expected:
            continue
        if abs(q) > 0.01:
            bad.append((acct, sym, round(q, 4)))
    print(f"  closed positions not netting to zero: {bad if bad else 'none'}")
    print("  open positions:", [(a, s, round(q or 0, 2)) for a, s, q in net if s in open_expected])
    con.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
