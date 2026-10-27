"""The summary-rows speedup: `_build_summary_rows` and
`_calculate_aggregate_metrics` against the originals frozen in
tests/reference/summary_rows.py.

The rows, the account currency map, the error and warning flags, the
account-level and overall metrics, and every warning or error logged must
match, bitwise and type for type. The random books exercise what the hoisted
filters serve:
- holdings with no market price, which take the last-trade fallback, priced
  or not and with or without a row on the report date;
- trades and transfers on the report date, into and out of the holding's
  account;
- text and zero prices;
- a Date column of text, which makes the shared filters fail and must fail
  every holding's handler as before;
- no To Account column;
- an account filter.
"""

import logging
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

import portfolio_analyzer as new  # noqa: E402
from reference import summary_rows as ref  # noqa: E402

ACCOUNTS = ["IBKR", "E*Trade", "Kim Eng", "SCBS"]
SYMBOLS = ["AAPL", "MSFT", "PTT.BK", "NOFEED", "GONE", "$CASH"]
CCY = {
    "AAPL": "USD",
    "MSFT": "USD",
    "PTT.BK": "THB",
    "NOFEED": "USD",
    "GONE": "EUR",
    "$CASH": "USD",
}
REPORT = date(2024, 6, 14)


def _canon(x):
    if isinstance(x, dict):
        return ("dict", [(_canon(k), _canon(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [_canon(v) for v in x])
    if isinstance(x, set):
        return ("set", sorted(map(repr, x)))
    if isinstance(x, float):
        return (
            "float",
            type(x).__name__,
            "nan" if math.isnan(x) else struct.pack("<d", x),
        )
    if isinstance(x, pd.DataFrame):
        return ("df", _canon(x.to_dict("split")), list(map(str, x.dtypes)))
    return (type(x).__name__, repr(x))


def _ledger(rng):
    rows = []
    days = pd.bdate_range("2023-01-02", "2024-06-14")
    quiet = rng.random() < 0.3  # nothing traded on the report date
    if quiet:
        days = days[:-1]
    for i in range(int(rng.integers(20, 120))):
        sym = str(rng.choice(SYMBOLS))
        acc = str(rng.choice(ACCOUNTS))
        typ = str(
            rng.choice(
                ["buy", "buy", "buy", "sell", "dividend", "fees", "transfer", "deposit"]
            )
        )
        if sym == "$CASH":
            typ = str(rng.choice(["deposit", "withdrawal"]))
        day = (
            days[-1]
            if not quiet and rng.random() < 0.15
            else days[int(rng.integers(0, len(days)))]
        )
        price = float(rng.choice([0.0, 12.5, 101.0, 250.0, np.nan]))
        rows.append(
            {
                "Date": day,
                "Symbol": sym,
                "Account": acc,
                "To Account": str(rng.choice(ACCOUNTS))
                if typ == "transfer"
                else np.nan,
                "Type": typ,
                "Quantity": float(rng.choice([1.0, 5.0, 10.0, 40.0])),
                "Price/Share": price,
                "Commission": float(rng.choice([0.0, 1.0, np.nan])),
                "Total Amount": float(rng.choice([np.nan, 100.0, 1250.0])),
                "Local Currency": CCY[sym],
                "Note": rng.choice([None, "", f"Transfer from {rng.choice(ACCOUNTS)}"]),
                "Split Ratio": 0.0,
                "original_index": i,
            }
        )
    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _market(rng):
    data = {}
    for sym in SYMBOLS:
        r = rng.random()
        if sym in ("NOFEED", "GONE") or r < 0.2:
            continue  # no price: the last-trade fallback
        data[sym] = {
            "price": float(rng.choice([0.0, 99.0, 180.0]))
            if r < 0.3
            else float(rng.uniform(10, 300)),
            "change": float(rng.normal(0, 2)) if rng.random() < 0.8 else None,
            "changesPercentage": float(rng.normal(0, 1)),
            "dividendRate": float(rng.choice([0.0, 0.96, 2.4])),
            "dividendYield": float(rng.choice([0.0, 0.005, 0.03])),
        }
    return data


def _run(mod, args, kwargs, caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        try:
            out = ("ok", mod._build_summary_rows(*args, **kwargs))
        except Exception as exc:
            out = ("raised", type(exc).__name__, str(exc))
    logs = [(r.levelname, r.getMessage()) for r in caplog.records]
    return out, logs


def _aggregate(mod, rows, txs, include, caplog):
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        try:
            out = (
                "ok",
                mod._calculate_aggregate_metrics(
                    pd.DataFrame(rows),
                    "USD",
                    REPORT,
                    fx_rates={"USD": 1.0, "THB": 35.0, "EUR": 0.9},
                    include_accounts=include,
                    all_available_accounts=ACCOUNTS,
                    transactions_df=txs,
                    historical_fx_rates=None,
                ),
            )
        except Exception as exc:
            out = ("raised", type(exc).__name__, str(exc))
    return out, [(r.levelname, r.getMessage()) for r in caplog.records]


@pytest.mark.parametrize("seed", range(6))
def test_summary_rows_and_metrics_match(seed, caplog):
    rng = np.random.default_rng(7800 + seed)
    for trial in range(12):
        df = _ledger(rng)
        try:
            holdings, *_ = new._process_transactions_to_holdings(
                df.copy(), "USD", set(), {}, "USD", REPORT
            )
        except Exception:
            continue
        txs = df.copy()
        r = rng.random()
        if r < 0.12:
            txs["Date"] = txs["Date"].dt.strftime("%Y-%m-%d")  # the shared filters fail
        elif r < 0.25:
            txs = txs.drop(columns=["To Account"])
        elif r < 0.35:
            txs["Price/Share"] = txs["Price/Share"].map(
                lambda v: "n/a" if v == 0 else v
            )
        include = (
            None if rng.random() < 0.7 else list(rng.choice(ACCOUNTS, 2, replace=False))
        )
        args = (
            holdings,
            _market(rng),
            {"USD": 1.0, "THB": 35.0, "EUR": 0.9},
            {"USD": 1.0, "THB": 35.4, "EUR": 0.91},
            str(rng.choice(["USD", "THB"])),
            "USD",
            txs,
            REPORT,
            set(),
            {"GONE"} if rng.random() < 0.3 else set(),
            {},
            {"NOFEED": 42.0} if rng.random() < 0.3 else {},
        )
        kwargs = {"include_accounts": include}
        got, got_logs = _run(new, args, kwargs, caplog)
        want, want_logs = _run(ref, args, kwargs, caplog)
        path = f"{seed}/{trial}"
        assert _canon(got) == _canon(want), path
        assert got_logs == want_logs, path
        if got[0] == "ok":
            rows = got[1][0]
            a, a_logs = _aggregate(new, rows, txs, include, caplog)
            b, b_logs = _aggregate(ref, rows, txs, include, caplog)
            assert _canon(a) == _canon(b), path + " metrics"
            assert a_logs == b_logs, path + " metrics logs"
