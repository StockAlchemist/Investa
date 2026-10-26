"""Phase 14a: portfolio_cashflows in Rust (rust/investa_core/src/pc_flows.rs)
against the original frozen in tests/reference/portfolio_cashflows.py.

Random ledgers of cash deposits and withdrawals (with and without a Total
Amount), stock deposits, transfers into, out of and within a selection,
ledger clean-ups named in the Note, flows before the range, and flows in
currencies with and without FX history, over UTC and naive ranges. Transfers
with no price fall back to the price history (date or Timestamp index,
sorted or not, UTC or naive) and then to another trade; a transfer nothing prices appends
a line to the original's scratch file, which is captured here and compared
instead of being written. Results, logs, warnings and the inputs left behind
(both functions write to the FX and price tables they are handed) must agree.
"""

import builtins
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import portfolio_cashflows as new  # noqa: E402
from reference import portfolio_cashflows as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

ACCOUNTS = ["IBKR", "SCBS", "Kim Eng", "E*Trade"]
SYMBOLS = ["$CASH", "AAPL", "PTT.BK", "NOPRICE", "VOD.L"]
CCY = {"$CASH": "USD", "AAPL": "USD", "PTT.BK": "THB", "NOPRICE": "USD", "VOD.L": "GBP"}
NOTES = [None, "", "Dust cleanup", "ghost lot ADJUSTMENT", "transfer in", "Auto-generated: settle"]


def _ledger(rng):
    n = int(rng.integers(1, 50))
    rows = []
    for i in range(n):
        sym = str(rng.choice(SYMBOLS))
        typ = str(rng.choice(["Deposit", "withdrawal", "transfer", "Transfer ", "buy", "sell", "dividend"]))
        rows.append(
            {
                "Date": pd.Timestamp("2023-01-02") + pd.Timedelta(days=int(rng.integers(-40, 500))),
                "Symbol": sym,
                "Account": str(rng.choice(ACCOUNTS)),
                "To Account": str(rng.choice(ACCOUNTS + ["Outside"])) if "ransfer" in typ else None,
                "Type": typ,
                "Quantity": float(rng.choice([1.0, 10.0, 250.0, -5.0, np.nan])),
                "Price/Share": np.nan if sym == "NOPRICE" else float(rng.choice([0.0, 12.5, 101.0, np.nan])),
                "Commission": float(rng.choice([0.0, 1.0, np.nan])),
                "Total Amount": float(rng.choice([np.nan, 0.0, 1250.0, -300.0])),
                "Local Currency": CCY[sym] if rng.random() < 0.9 else "EUR",
                "Note": NOTES[int(rng.integers(0, len(NOTES)))],
            }
        )  # fmt: skip
    df = pd.DataFrame(rows)
    if rng.random() < 0.15:
        df = df.drop(columns=["Total Amount"])
    if rng.random() < 0.15:
        df = df.drop(columns=["Note"])
    if rng.random() < 0.1:
        df = df.drop(columns=["To Account"])
    return df


def _fx(rng):
    days = pd.date_range("2022-11-01", "2024-06-30", freq="B")
    out = {
        "THB=X": pd.DataFrame({"price": 35 + rng.normal(0, 0.3, len(days))}, index=days),
        "GBP=X": pd.DataFrame({"price": 0.8 + rng.normal(0, 0.01, len(days))}, index=days.date),
    }
    if rng.random() < 0.3:
        out.pop("GBP=X")
    if rng.random() < 0.2:
        out["EUR=X"] = pd.DataFrame({"Close": [0.9]}, index=days[:1])
    return out


def _prices(rng):
    days = pd.date_range("2022-11-01", "2024-06-30", freq="B")
    aapl = pd.DataFrame(
        {"price": 150 + np.arange(len(days)) * 0.1},
        index=days.tz_localize("UTC") if rng.random() < 0.6 else days,
    )
    if rng.random() < 0.3:
        aapl = aapl.iloc[::-1]  # unsorted: the fallback sorts it
    ptt = pd.DataFrame({"price": 34.0, "Close": 34.5}, index=days.date)
    return {"AAPL": aapl, "PTT.BK": ptt}


def _copy(d):
    return {k: v.copy(deep=True) for k, v in d.items()}


@pytest.fixture
def scratch_writes(monkeypatch):
    """The original appends to another tool's scratch file; capture it."""
    writes = []
    real_open = builtins.open

    class Capture:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, text):
            writes.append(text)

    def fake_open(path, *args, **kwargs):
        if str(path).endswith("missing_transfer_prices.txt"):
            return Capture()
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return writes


@pytest.mark.parametrize("seed", range(8))
def test_vectorized_flows_match(caplog, scratch_writes, seed):
    rng = np.random.default_rng(14100 + seed)
    for trial in range(40):
        df = _ledger(rng)
        tz = "UTC" if rng.random() < 0.9 else None  # a naive range raises in both
        start = pd.Timestamp("2023-01-02") + pd.Timedelta(days=int(rng.integers(0, 60)))
        date_range = pd.date_range(start, "2024-06-14", freq="D", tz=tz)
        fx, prices = _fx(rng), _prices(rng)
        target = str(rng.choice(["USD", "THB", "GBP"]))
        included = [None, [], ["IBKR"], [" ibkr", "SCBS"], ACCOUNTS][int(rng.integers(0, 5))]
        yf_map = {"AAPL": "AAPL", "PTT.BK": "PTT.BK", "NOPRICE": "NOPRICE"} if rng.random() < 0.7 else None

        sides = []
        for mod in (ref, new):
            a, f, p = df.copy(), _copy(fx), _copy(prices)
            scratch_writes.clear()
            out = call(
                lambda: mod._calculate_daily_net_cash_flow_vectorized(
                    date_range, a, target, f, "USD", included, p, yf_map
                ),
                caplog,
            )
            sides.append((out, list(scratch_writes), canon(a), canon(f), canon(p)))
        assert sides[0] == sides[1], (seed, trial)


@pytest.mark.parametrize("seed", range(6))
def test_daily_flow_matches(caplog, seed):
    rng = np.random.default_rng(14200 + seed)
    for trial in range(40):
        df = _ledger(rng)
        if rng.random() < 0.1:
            df["Type"] = df["Type"].astype(object)
            df.loc[df.index[:1], "Type"] = np.nan  # .lower() on a float raises
        fx, prices = _fx(rng), _prices(rng)
        day = df["Date"].iloc[int(rng.integers(0, len(df)))].date() if rng.random() < 0.9 else date(2020, 1, 1)
        target = str(rng.choice(["USD", "THB"]))
        included = [None, ["IBKR"], ["SCBS", "Kim Eng"]][int(rng.integers(0, 3))]
        yf_map = {"AAPL": "AAPL", "PTT.BK": "PTT.BK"} if rng.random() < 0.7 else None
        sides = []
        for mod in (ref, new):
            a, f, p = df.copy(), _copy(fx), _copy(prices)
            out = call(
                lambda: mod._calculate_daily_net_cash_flow(
                    day, a, target, f, {}, "USD", set(), included, p, yf_map
                ),
                caplog,
            )
            sides.append((out, canon(a), canon(f), canon(p)))
        assert sides[0] == sides[1], (seed, trial)
