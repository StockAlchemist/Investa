"""Phase 13c: portfolio_analyzer._process_transactions_to_holdings and the
FIFO shim calculate_fifo_lots_and_gains in Rust
(rust/investa_core/src/pa_holdings.rs, pa_fifo.rs) against the originals
frozen in tests/reference/portfolio_analyzer.py.

Random ledgers carry every transaction type and spelling, cash in its several
spellings, splits with and without a ratio (the quantity read as the ratio,
or as a target quantity above 20), "All Accounts" split duplicates in one
month, transfers with and without a To Account, Int64 and plain ids, tags,
currencies outside the ledger's set, and Auto and Manual cash modes. All nine
outputs, the caller's frame as the function leaves it, the logs and the
warnings must agree.

The FIFO shim gets the same ledgers made rougher: object columns holding
numeric strings and junk, NaT dates, None symbols, NaN and Int64 ids, a
missing required column, string dates the row parse rejects, and the real USD-bridge FX lookup over a
historical table, with and without current rates for the fallback. The
original's own bookkeeping is covered bitwise by test_rust_fifo_parity.py.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import portfolio_analyzer as new  # noqa: E402
from reference import portfolio_analyzer as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

TYPES = ["buy", "Sell", "deposit", "withdrawal", "dividend", "fees", "split", "Stock Split", "transfer", "short sell",
         "buy to cover", "tax", "interest", "spin-off", "merger", "fee"]  # fmt: skip


def ledger(rng):
    n = int(rng.integers(1, 40))
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime("2023-01-02")
            + pd.to_timedelta(np.sort(rng.integers(0, 400, n)), unit="D"),
            "Type": rng.choice(TYPES, n),
            "Symbol": rng.choice(["AAPL", " MSFT ", "$CASH", "Cash ($)", "KO"], n),
            "Account": rng.choice(["IBKR", "All Accounts", " SCBS"], n),
            "Quantity": rng.choice([0.0, 1.0, 5.0, 10.0, 25.0, 4.0, np.nan], n),
            "Price/Share": rng.choice([0.0, 10.0, 55.5, np.nan], n),
            "Total Amount": rng.choice([0.0, 100.0, 50.0, -20.0, np.nan], n),
            "Commission": rng.choice([0.0, 1.0, np.nan], n),
            "Split Ratio": rng.choice([0.0, 2.0, np.nan], n),
            "Local Currency": rng.choice(["USD", "thb", "", "EURO"], n),
            "To Account": rng.choice(["SCBS", None, " IBKR "], n),
            "Tags": rng.choice(["", "core, growth", None, "x"], n),
            "original_index": np.arange(n),
        }
    )
    if rng.random() < 0.3:
        df["original_index"] = df["original_index"].astype("Int64")
    if rng.random() < 0.2:
        df = df.drop(columns=["To Account"])
    if rng.random() < 0.2:
        df = df.drop(columns=["Tags"])
    return df


@pytest.mark.parametrize("seed", range(8))
def test_holdings_match(caplog, seed):
    rng = np.random.default_rng(9400 + seed)
    for trial in range(30):
        df = ledger(rng)
        fx = {
            (d.date(), c): float(rng.uniform(0.02, 40))
            for d, c in zip(df["Date"], ["USD", "THB"] * len(df))
            if rng.random() < 0.6
        }
        modes = (
            {"IBKR": "Auto", " SCBS ": "Manual", "All Accounts": "Auto"}
            if rng.random() < 0.6
            else None
        )
        a, b = df.copy(), df.copy()
        args = ("USD", {"AAPL"}, fx, "USD", dt.date(2024, 6, 1), modes)
        ga = call(lambda: ref._process_transactions_to_holdings(a, *args), caplog)
        gb = call(lambda: new._process_transactions_to_holdings(b, *args), caplog)
        assert ga == gb, (seed, trial)
        assert canon(a) == canon(b), (seed, trial, "frame left behind")


def rough(rng):
    df = ledger(rng)
    n = len(df)
    df["Type"] = rng.choice(TYPES + ["Buy", "spin off", "spinoff", "SELL "], n)
    if rng.random() < 0.3:
        df["Quantity"] = df["Quantity"].astype(object)
        df.loc[rng.random(n) < 0.2, "Quantity"] = rng.choice(["5", "abc", "", None])
    if rng.random() < 0.2:
        df["Price/Share"] = df["Price/Share"].astype(object)
        df.loc[rng.random(n) < 0.3, "Price/Share"] = "12.5"
    if rng.random() < 0.2:
        df.loc[rng.random(n) < 0.2, "Date"] = pd.NaT
    if rng.random() < 0.15:
        df["Date"] = df["Date"].astype(object)  # a string date fails .date()
        df.loc[rng.random(n) < 0.2, "Date"] = "2023-05-01"
    if rng.random() < 0.2:
        df["Symbol"] = df["Symbol"].astype(object)
        df.loc[rng.random(n) < 0.2, "Symbol"] = None
    if rng.random() < 0.2:
        df["original_index"] = df["original_index"].astype(float)
        df.loc[rng.random(n) < 0.2, "original_index"] = np.nan
    if rng.random() < 0.2:
        df["Local Currency"] = rng.choice(["USD", "THB", "EUR", None, np.nan, " thb "], n)
    if rng.random() < 0.05:
        df = df.drop(columns=[str(rng.choice(["Date", "Symbol", "Account", "Type"]))])
    return df


@pytest.mark.parametrize("seed", range(8))
def test_fifo_shim_matches(caplog, seed):
    rng = np.random.default_rng(9500 + seed)
    days = pd.date_range("2022-12-01", "2024-03-01", freq="D").date
    for trial in range(40):
        df = rough(rng)
        hist = {
            "THB=X": pd.DataFrame(
                {"price": rng.uniform(30, 38, len(days))}, index=days
            ).iloc[rng.random(len(days)) < 0.7],
            "EUR=X": pd.DataFrame({"price": rng.uniform(0.8, 1.0, len(days))}, index=days),
        }
        if rng.random() < 0.3:
            hist = {}
        rates = None if rng.random() < 0.4 else {"USD": 1.0, "THB": 35.5}
        display = str(rng.choice(["USD", "THB"]))
        tol = float(rng.choice([1e-6, 0.01]))
        a, b = df.copy(), df.copy()
        args = (display, hist, "USD", {"AAPL"}, tol, rates)
        ga = call(lambda: ref.calculate_fifo_lots_and_gains(a, *args), caplog)
        gb = call(lambda: new.calculate_fifo_lots_and_gains(b, *args), caplog)
        assert ga == gb, (seed, trial)
        assert canon(a) == canon(b), (seed, trial, "frame left behind")
