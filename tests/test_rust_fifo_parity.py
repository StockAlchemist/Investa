"""Phase 6a: FIFO lots and realized gains (rust/investa_core/src/fifo.rs plus
the row-reading shim in portfolio_analyzer) against the original frozen in
tests/reference/fifo.py.

Bitwise and type for type: the gains frame (columns, dtypes, index order and
every value) and the open-lot dicts (key order, including holdings a read left
empty, and every field), on random ledgers with buys, sells, transfers,
repeated and global splits, both spin-off legs, skipped row types, cash rows,
NaN and missing prices, quantities and FX rates, and same-day lots.
"""

import math
import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import portfolio_analyzer as new  # noqa: E402
from reference import fifo as ref  # noqa: E402


def _same(a, b, path="result"):
    if isinstance(a, pd.DataFrame):
        assert isinstance(b, pd.DataFrame), path
        assert list(a.columns) == list(b.columns), f"{path}: columns"
        assert list(a.dtypes) == list(b.dtypes), (
            f"{path}: dtypes {list(a.dtypes)} vs {list(b.dtypes)}"
        )
        assert list(a.index) == list(b.index), f"{path}: index"
        for col in a.columns:
            for i, (x, y) in enumerate(zip(a[col].tolist(), b[col].tolist())):
                _same(x, y, f"{path}[{col}][{i}]")
        return
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{path}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            _same(x, y, f"{path}[{i}]")
    elif isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _fx(local, display, tx_date, historical):
    """A deterministic stand-in for the historical FX lookup, covering every
    kind of answer the real one gives: a float, a NumPy float, NaN, None."""
    if local == display:
        return 1.0
    k = (tx_date.toordinal() * 7 + len(local)) % 11
    if k == 0:
        return np.nan
    if k == 1:
        return None
    if k == 2:
        return np.float64(0.0283 + k / 1000)
    return 0.0283 + k / 997


def _rate(local, display, rates):
    if local == "EUR":
        return np.nan
    return rates.get(display, 1.0) / rates.get(local, 1.0)


@pytest.fixture(autouse=True)
def _stub_fx(monkeypatch):
    for mod in (new, ref):
        monkeypatch.setattr(mod, "get_historical_rate_via_usd_bridge", _fx)
        monkeypatch.setattr(mod, "get_conversion_rate", _rate)


def _ledger(rng):
    n = int(rng.integers(0, 120))
    symbols = ["AAA", "bbb ", "CCC", "$CASH"]
    accounts = ["IBKR", "ibkr", "SET", " Kim Eng "]
    types = (
        ["buy"] * 8 + ["Sell"] * 5 + ["transfer"] * 2 + ["split", "Stock Split", "spin-off", "Spin Off"]
        + ["dividend", "deposit", "fees", "withdrawal", "short sell"]
    )  # fmt: skip
    start = pd.Timestamp("2015-01-02")
    offsets = np.sort(rng.integers(0, 2000, n))
    if rng.random() < 0.3:
        offsets = offsets // 50 * 50  # many same-day rows
    rows = []
    for i in range(n):
        typ = str(rng.choice(types))
        r = rng.random()
        qty = (
            float("nan")
            if r < 0.04
            else 0.0
            if r < 0.07
            else -3.0
            if r < 0.09
            else float(rng.integers(1, 200)) * float(rng.choice([1, 0.5, 0.001]))
        )
        r = rng.random()
        price = (
            float("nan") if r < 0.1 else 0.0 if r < 0.14 else float(rng.lognormal(3, 1))
        )
        r = rng.random()
        rows.append(
            {
                "Date": pd.NaT if rng.random() < 0.02 else start + pd.Timedelta(days=int(offsets[i])),
                "Symbol": str(rng.choice(symbols)),
                "Account": str(rng.choice(accounts)),
                "Type": typ,
                "Local Currency": rng.choice(["USD", "THB", "EUR", "", "us", None]),
                "Quantity": qty,
                "Price/Share": price,
                "Commission": float("nan") if rng.random() < 0.3 else float(rng.uniform(0, 10)),
                "Split Ratio": float(rng.choice([2.0, 0.5, 3.0, 0.0, -1.0, np.nan])) if "split" in typ.lower() else np.nan,
                "To Account": rng.choice(["SET", "IBKR", "", None, np.nan, "new acct"]) if typ == "transfer" else np.nan,
                "Total Amount": float("nan") if r < 0.3 else float(rng.normal(500, 800)),
            }
        )  # fmt: skip
    df = pd.DataFrame(
        rows,
        columns=[
            "Date", "Symbol", "Account", "Type", "Local Currency", "Quantity", "Price/Share",
            "Commission", "Split Ratio", "To Account", "Total Amount",
        ],
    )  # fmt: skip
    df["Date"] = pd.to_datetime(df["Date"])
    ids = np.arange(n)
    if rng.random() < 0.5:
        rng.shuffle(ids)  # ids need not follow the row order
    df["original_index"] = ids
    if n and rng.random() < 0.1:
        # A repeated split row: applied once.
        df = pd.concat(
            [df, df[df["Type"].str.lower().str.contains("split")]]
        ).sort_values("Date", kind="stable")
    return df


def _run(mod, df, rng_seed):
    rng = np.random.default_rng(rng_seed)
    rates = None if rng.random() < 0.4 else {"USD": 1.0, "THB": 35.5, "EUR": 0.9}
    tol = float(rng.choice([1e-6, 1e-9, 0.01]))
    return mod.calculate_fifo_lots_and_gains(
        df.copy(), "USD", {}, "USD", set(), tol, rates
    )


@pytest.mark.parametrize("seed", range(6))
def test_fifo_matches(seed):
    rng = np.random.default_rng(6400 + seed)
    for trial in range(250):
        df = _ledger(rng)
        _same(_run(new, df, trial), _run(ref, df, trial), f"seed{seed}/trial{trial}")


def test_missing_columns_match():
    df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2020-01-02")],
            "Symbol": ["AAA"],
            "Account": ["X"],
            "Type": ["buy"],
        }
    )
    _same(_run(new, df, 0), _run(ref, df, 0))
    _same(_run(new, df.drop(columns="Type"), 0), _run(ref, df.drop(columns="Type"), 0))
