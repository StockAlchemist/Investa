"""IRR and MWR cash flows (rust/investa_core/src/cashflows.rs plus the ledger
and shim in finutils) against the originals frozen in
tests/reference/cash_flows.py.

Bitwise and type for type, exceptions included, on random ledgers:
- account names that are regular expressions (`E*Trade`, `IBKR (1)`), and
  transfers named only in the Note;
- every row type, in any case and with stray spaces;
- NaN, zero and negative amounts;
- same-day rows with shuffled ids;
- integer and text-typed numeric columns.
Holdings are compared through the one-shot function and through one shared
CashFlowLedger; MWR under both flow bases, with and without a scope, with
historical and current FX.
"""

import math
import os
import struct
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils as new  # noqa: E402
from reference import cash_flows as ref  # noqa: E402

ACCOUNTS = ["IBKR", "E*Trade", "IBKR (1)", "SET", "Kim Eng"]
SYMBOLS = ["AAPL", "XYZ", "$CASH", "MSFT"]  # AAPL is shortable
TYPES = [
    "buy", "Buy ", "sell", "deposit", "withdrawal", "transfer", "Transfer", "dividend", "fees",
    "split", "Stock Split", "short sell", "buy to cover", "interest", "tax", "spin-off",
]  # fmt: skip


def _same(a, b, path="result"):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _outcome(fn, *args, **kwargs):
    try:
        return ("ok", fn(*args, **kwargs))
    except Exception as exc:  # the original's failures must be the port's
        return ("raised", type(exc))


def _ledger(rng):
    n = int(rng.integers(0, 70))
    days = np.sort(rng.integers(0, 900, n))
    if rng.random() < 0.3:
        days = days // 40 * 40  # same-day rows
    rows = []
    for i in range(n):
        typ = str(rng.choice(TYPES))
        sym = str(rng.choice(SYMBOLS))
        acc = str(rng.choice(ACCOUNTS))
        r = rng.random()
        rows.append(
            {
                "Date": pd.Timestamp("2021-01-04") + pd.Timedelta(days=int(days[i])),
                "Symbol": sym,
                "Account": acc,
                "To Account": (str(rng.choice(ACCOUNTS)) if r < 0.6 else (None if r < 0.8 else np.nan)) if "ransfer" in typ else np.nan,
                "Type": typ,
                "Quantity": float(rng.choice([np.nan, 0.0, -5.0, 3.0, 10.0, 250.0])),
                "Price/Share": float(rng.choice([np.nan, 0.0, 1.0, 12.5, 180.0])),
                "Commission": float(rng.choice([np.nan, 0.0, 1.0, -2.0, 4.95])),
                "Total Amount": float(rng.choice([np.nan, 0.0, 55.0, -120.0, 1500.0])),
                "Local Currency": str(rng.choice(["USD", "USD", "THB", "EUR"])),
                "Note": rng.choice([None, np.nan, "", f"Transfer from {rng.choice(ACCOUNTS)}", f"ACATS to {str(rng.choice(ACCOUNTS)).lower()}", "x"]),
            }
        )  # fmt: skip
    df = pd.DataFrame(rows, columns=["Date", "Symbol", "Account", "To Account", "Type", "Quantity", "Price/Share",
                                     "Commission", "Total Amount", "Local Currency", "Note"])  # fmt: skip
    df["Date"] = pd.to_datetime(df["Date"])
    ids = np.arange(n)
    if rng.random() < 0.5:
        rng.shuffle(ids)
    df["original_index"] = ids
    r = rng.random()
    if n and r < 0.1:
        df["Quantity"] = df["Quantity"].fillna(0).astype(np.int64)  # an integer column
    elif n and r < 0.2:
        df["Price/Share"] = [
            rng.choice(["12.5", "abc", None, 7.0]) for _ in range(n)
        ]  # text cells
    elif r < 0.25:
        df = df.drop(columns=["Note"])
    elif r < 0.3:
        df = df.drop(columns=["To Account"])
    return df


@pytest.mark.parametrize("seed", range(6))
def test_holding_flows_match(seed):
    rng = np.random.default_rng(6900 + seed)
    for trial in range(80):
        df = _ledger(rng)
        ledger = new.CashFlowLedger(df)
        for _ in range(12):
            sym = str(rng.choice(SYMBOLS + ["NONE"]))
            acc = None if rng.random() < 0.2 else str(rng.choice(ACCOUNTS))
            mv = float(rng.choice([0.0, 950.0, -30.0, np.nan]))
            flag = bool(rng.random() < 0.7)
            report = rng.choice([date(2023, 10, 1), date(2021, 1, 1), None])
            want = _outcome(
                ref.get_cash_flows_for_symbol_account,
                sym,
                acc,
                df.copy(),
                mv,
                flag,
                report,
            )
            got = _outcome(
                new.get_cash_flows_for_symbol_account,
                sym,
                acc,
                df.copy(),
                mv,
                flag,
                report,
            )
            shared = _outcome(ledger.flows_for, sym, acc, mv, flag, report)
            path = f"{seed}/{trial} {sym}/{acc}"
            assert got[0] == want[0] == shared[0], (path, want, got, shared)
            if want[0] == "ok":
                _same(got[1], want[1], path)
                _same(shared[1], want[1], path + " (ledger)")
            else:
                assert got[1] is want[1] is shared[1], path


@pytest.mark.parametrize("seed", range(6))
def test_mwr_flows_match(seed):
    rng = np.random.default_rng(7000 + seed)
    for trial in range(120):
        df = _ledger(rng)
        if len(df) and rng.random() < 0.6:
            acc = str(rng.choice(ACCOUNTS))
            df = df[(df["Account"].astype(str).str.strip().str.upper() == acc.upper())
                    | (df.get("To Account", pd.Series(index=df.index, dtype=object)).astype(str).str.strip().str.upper() == acc.upper())]  # fmt: skip
            scope = [acc]
        else:
            scope = (
                None
                if rng.random() < 0.5
                else list(rng.choice(ACCOUNTS, 2, replace=False))
            )
        hist = None
        if rng.random() < 0.6 and len(df):
            hist = {
                (d.date(), c): float(rng.uniform(0.02, 40))
                for d, c in zip(df["Date"], df["Local Currency"])
                if rng.random() < 0.7
            }
        fx = None if rng.random() < 0.3 else {"USD": 1.0, "THB": 35.0, "EUR": 0.9}
        args = dict(
            final_account_market_value=float(rng.choice([0.0, 5000.0, np.nan])),
            end_date=date(2023, int(rng.integers(1, 13)), 1),
            target_currency=str(rng.choice(["USD", "THB"])),
            fx_rates=fx,
            display_currency="USD",
            historical_fx_rates=hist,
            include_accounts=scope,
            flow_basis=str(rng.choice(["portfolio", "per_trade"])),
        )
        want = _outcome(ref.get_cash_flows_for_mwr, df.copy(), **args)
        got = _outcome(new.get_cash_flows_for_mwr, df.copy(), **args)
        path = f"{seed}/{trial} {args['flow_basis']}"
        assert got[0] == want[0], (path, want, got)
        if want[0] == "ok":
            _same(got[1], want[1], path)
        else:
            assert got[1] is want[1], path
