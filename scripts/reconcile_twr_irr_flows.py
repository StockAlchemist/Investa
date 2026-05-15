"""
Reconciliation diagnostic for TWR vs IRR cash-flow streams.

For each user, replays the classification rules of BOTH engines on every
transaction in scope ("All Accounts") and computes:

  - TWR_flow  : what _calculate_daily_net_cash_flow_vectorized would emit
                (external cash deposits/withdrawals + scope-crossing transfers)
  - IRR_flow  : what get_cash_flows_for_mwr would emit
                (per-transaction flows including buys/sells/divs at full scope)
  - Delta     : IRR_flow - TWR_flow

Reports:
  - Year-by-year delta totals (USD, ignoring FX conversion for now)
  - Top 20 rows where the two engines disagree by absolute amount
  - Aggregate delta broken down by transaction type

Run: python scripts/reconcile_twr_irr_flows.py dheematan
"""
import sys
import sqlite3
import pandas as pd
from collections import defaultdict

USER = sys.argv[1] if len(sys.argv) > 1 else "dheematan"
DB = f"data/users/{USER}/portfolio.db"

con = sqlite3.connect(DB)
df = pd.read_sql_query(
    """
    SELECT id, Date, Account, Symbol, Type, Quantity, "Price/Share" AS price,
           Commission, "Total Amount" AS total, "Local Currency" AS curr,
           "To Account" AS to_acct
    FROM transactions
    """,
    con,
)
con.close()

df["Date"] = pd.to_datetime(df["Date"])
df["Type"] = df["Type"].astype(str).str.lower().str.strip()
df["Account"] = df["Account"].astype(str).str.upper().str.strip()
df["to_acct"] = df["to_acct"].fillna("").astype(str).str.upper().str.strip()
df["Symbol"] = df["Symbol"].astype(str).str.strip()
for c in ("Quantity", "price", "Commission", "total"):
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Scope = ALL accounts (the "All Accounts" dashboard view)
included = set(df["Account"].unique()) | set(df.loc[df["to_acct"] != "", "to_acct"].unique())
included.discard("")


def twr_flow(row):
    """_calculate_daily_net_cash_flow_vectorized for All-Accounts scope."""
    t = row["Type"]
    sym = row["Symbol"]
    qty = row["Quantity"] or 0.0
    comm = row["Commission"] or 0.0
    # 1. Cash deposit / withdrawal
    if sym == "$CASH" and t in ("deposit", "withdrawal"):
        if row["Account"] not in included:
            return 0.0
        if t == "deposit":
            return abs(qty) - comm
        else:
            return -abs(qty) - comm
    # 2. Transfer crossing scope boundary (none at All-Accounts scope, both sides included)
    if t == "transfer":
        src_in = row["Account"] in included
        dst_in = row["to_acct"] in included if row["to_acct"] else False
        if src_in and not dst_in:
            # outbound to external
            return -(abs(qty) * (row["price"] or 1.0))
        if dst_in and not src_in:
            return +(abs(qty) * (row["price"] or 1.0))
        return 0.0
    return 0.0


def irr_flow(row):
    """get_cash_flows_for_mwr classification for All-Accounts scope."""
    t = row["Type"]
    sym = row["Symbol"]
    qty = row["Quantity"]
    price = row["price"]
    comm = 0.0 if pd.isna(row["Commission"]) else row["Commission"]
    total = row["total"]
    qty_abs = abs(qty) if pd.notna(qty) else 0.0

    if sym != "$CASH":
        if t in ("buy", "deposit"):
            if pd.notna(qty) and pd.notna(price):
                return -((qty_abs * price) + comm)
        elif t in ("sell", "withdrawal"):
            if pd.notna(qty) and pd.notna(price):
                return (qty_abs * price) - comm
        elif t == "dividend":
            if pd.notna(total):
                amt = total
            elif pd.notna(price):
                amt = (qty_abs * price) if (pd.notna(qty) and qty_abs > 0) else price
            else:
                return 0.0
            return amt - comm
        elif t == "short sell":
            if pd.notna(price) and pd.notna(qty) and qty_abs > 0:
                return (qty_abs * price) - comm
        elif t == "buy to cover":
            if pd.notna(price) and pd.notna(qty) and qty_abs > 0:
                return -((qty_abs * price) + comm)
        elif t == "fees":
            return -abs(comm)
        elif t == "transfer":
            src_in = row["Account"] in included
            dst_in = row["to_acct"] in included if row["to_acct"] else False
            if src_in and not dst_in and pd.notna(qty) and pd.notna(price):
                return (abs(qty) * price) - abs(comm)
            if dst_in and not src_in and pd.notna(qty) and pd.notna(price):
                return -(abs(qty) * price)
            return 0.0
        return 0.0
    else:  # $CASH symbol
        if t in ("deposit", "buy"):
            if pd.notna(qty):
                return -(abs(qty) + comm)
        elif t in ("withdrawal", "sell"):
            if pd.notna(qty):
                return abs(qty) - comm
        elif t in ("dividend", "interest"):
            return 0.0
        elif t == "transfer":
            src_in = row["Account"] in included
            dst_in = row["to_acct"] in included if row["to_acct"] else False
            if src_in and not dst_in and pd.notna(qty):
                return abs(qty) - comm
            if dst_in and not src_in and pd.notna(qty):
                return -(abs(qty) + comm)
        return 0.0


# Note: ignoring FX (USD-only diagnostic; THB rows in dheematan are negligible
# for this comparison since they're mostly the same in both DBs).
df = df[df["curr"] == "USD"].copy()
df["twr"] = df.apply(twr_flow, axis=1)
df["irr"] = df.apply(irr_flow, axis=1)
df["delta"] = df["irr"] - df["twr"]
df["year"] = df["Date"].dt.year

print(f"\n=== {USER}: TWR vs IRR cash-flow reconciliation (USD only) ===\n")

# Top-line summary
twr_total = df["twr"].sum()
irr_total = df["irr"].sum()
print(f"Total TWR flows (sum): ${twr_total:>14,.2f}")
print(f"Total IRR flows (sum): ${irr_total:>14,.2f}")
print(f"Total Delta (IRR-TWR): ${df['delta'].sum():>14,.2f}\n")

# Delta by year
yr = df.groupby("year").agg(twr_sum=("twr", "sum"), irr_sum=("irr", "sum"), delta=("delta", "sum"))
yr = yr[(yr["delta"].abs() > 1.0)].sort_index()
print(f"Year-by-year delta (rows with |delta|>$1):\n{yr.to_string(float_format='%.0f')}\n")

# Delta by transaction type
by_type = df.groupby("Type").agg(
    n=("delta", "count"),
    twr_sum=("twr", "sum"),
    irr_sum=("irr", "sum"),
    delta_sum=("delta", "sum"),
).sort_values("delta_sum", key=abs, ascending=False)
print("Delta by transaction type:\n", by_type.to_string(float_format="%.0f"), "\n")

# Top 20 single-row deltas
print("Top 20 single-row deltas:\n")
top = df.reindex(df["delta"].abs().sort_values(ascending=False).index).head(20)
cols = ["Date", "Account", "Symbol", "Type", "Quantity", "price", "total", "twr", "irr", "delta"]
print(top[cols].to_string(index=False, float_format="%.2f"))
