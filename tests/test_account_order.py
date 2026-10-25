"""The summary's account order is the same in every process.

Account ids were assigned in set order, and a set of strings iterates
differently under each hash seed, so the summary's rows and its per-account
metrics came out in a different order every time the server started. Each
seed here runs in its own interpreter; all must agree, with accounts in name
order.
"""

import json
import os
import subprocess
import sys

SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))

SCRIPT = r"""
import json, logging, sys
logging.disable(logging.WARNING)
import pandas as pd
from datetime import date
from portfolio_analyzer import _process_transactions_to_holdings

accounts = ["Zeta", "IBKR", "E*Trade", "Kim Eng", "alpha", "Morgan Stanley", "SCBS"]
rows = []
for i, acc in enumerate(accounts):
    for sym in ("AAPL", "MSFT", "PTT.BK"):
        rows.append({"Type": "Buy", "Symbol": sym, "Quantity": 10.0 + i, "Price/Share": 100.0,
                     "Total Amount": 1000.0, "Account": acc, "To Account": ""})
    rows.append({"Type": "Dividend", "Symbol": "AAPL", "Quantity": 0.0, "Price/Share": 0.0,
                 "Total Amount": 5.0 + i, "Account": acc, "To Account": ""})
rows.append({"Type": "Transfer", "Symbol": "AAPL", "Quantity": 5.0, "Price/Share": 0.0,
             "Total Amount": 0.0, "Account": "IBKR", "To Account": "Brand New"})
for i, r in enumerate(rows):
    r.update({"Commission": 0.0, "Local Currency": "USD", "Split Ratio": 0.0,
              "original_index": i, "Date": f"2025-01-{1 + i % 28:02d}"})
df = pd.DataFrame(rows)
df["Date"] = pd.to_datetime(df["Date"])
holdings, realized, dividends, *_ = _process_transactions_to_holdings(
    df, default_currency="USD", shortable_symbols=set(), historical_fx_lookup={},
    display_currency_for_hist_fx="USD", report_date=date(2026, 7, 1))
print(json.dumps({"holdings": [list(k) for k in holdings],
                  "dividends": [str(k) for k in dividends]}))
"""


def _run(seed):
    env = {**os.environ, "PYTHONHASHSEED": str(seed), "PYTHONPATH": SRC}
    out = subprocess.run(
        [sys.executable, "-c", SCRIPT],
        env=env,
        cwd=SRC,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_account_order_does_not_depend_on_the_hash_seed():
    runs = [_run(seed) for seed in (1, 2, 3, 4)]
    assert all(r == runs[0] for r in runs[1:]), runs
    accounts = list(dict.fromkeys(acc for _sym, acc in runs[0]["holdings"]))
    assert accounts == sorted(accounts), accounts
