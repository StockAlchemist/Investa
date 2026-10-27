"""Phase 16b: CashFlowLedger and the cash-flow builders of finutils in Rust
(rust/investa_core/src/fu_flows.rs) against the original frozen in
tests/reference/finutils_p16.py.

The random ledgers of the phase 13 test (regex account names, transfers named
in the Note, every row type, NaN/zero/negative amounts, same-day rows,
integer and text columns, missing columns), plus frames the original handles
specially: no `original_index`, a `Type` column the `.str` accessor rejects
(the error raised again for every account), NaT dates and mixed id types (the
pandas sort), NumPy market values and datetime report dates. Holdings go
through a shared ledger and the one-shot builder, the ledger's internals are
compared directly, and MWR runs under both bases. Results, logs and warnings
agree.
"""

import os
import sys
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils as new  # noqa: E402
from reference import finutils_p16 as ref  # noqa: E402
from test_rust_cashflow_parity import ACCOUNTS, SYMBOLS, _ledger  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402


def _odd(rng, df):
    r = rng.random()
    if len(df) and r < 0.08:
        df = df.drop(columns=["original_index"])
    elif len(df) and r < 0.16:
        df = df.assign(Type=[1] * len(df))  # .str rejects a non-text column
    elif len(df) > 2 and r < 0.24:
        df = df.copy()
        df.loc[df.index[1], "Date"] = pd.NaT
    elif len(df) > 2 and r < 0.32:
        df = df.copy()
        df["original_index"] = df["original_index"].astype(object)
        df.loc[df.index[0], "original_index"] = 1.5
    return df


@pytest.mark.parametrize("seed", range(6))
def test_ledger_matches(caplog, seed):
    rng = np.random.default_rng(23000 + seed)
    for trial in range(60):
        df = _odd(rng, _ledger(rng))
        ledgers = (ref.CashFlowLedger(df.copy()), new.CashFlowLedger(df.copy()))
        for _ in range(10):
            sym = [str(rng.choice(SYMBOLS + ["NONE"])), 5, None][int(rng.choice(3, p=[0.9, 0.05, 0.05]))]
            acc = None if rng.random() < 0.2 else str(rng.choice(ACCOUNTS))
            mv = [0.0, 950.0, -30.0, np.nan, np.float64(400.0)][int(rng.integers(0, 5))]
            flag = bool(rng.random() < 0.7)
            report = [date(2023, 10, 1), date(2021, 1, 1), None, datetime(2023, 10, 1), date(2021, 6, 30)][int(rng.integers(0, 5))]
            outs = [call(lambda lg=lg: lg.flows_for(sym, acc, mv, flag, report), caplog) for lg in ledgers]
            assert outs[0] == outs[1], (seed, trial, sym, acc)
            outs = [call(lambda mod=mod: mod.get_cash_flows_for_symbol_account(sym, acc, df.copy(), mv, flag, report), caplog)
                    for mod in (ref, new)]  # fmt: skip
            assert outs[0] == outs[1], (seed, trial, "one-shot")
        for name, args in (("_rows", ()), ("_symbol_positions", ("AAPL",)), ("_symbol_positions", (3,)),
                           ("_account_mask", ("IBKR",)), ("_account_mask", ("E*Trade",)), ("_transfer_rows", ()),
                           ("_sorted", (list(range(min(len(df), 5))),))):  # fmt: skip
            outs = [call(lambda lg=lg: getattr(lg, name)(*args), caplog) for lg in ledgers]
            assert outs[0] == outs[1], (seed, trial, name)


@pytest.mark.parametrize("seed", range(6))
def test_mwr_matches(caplog, seed):
    rng = np.random.default_rng(23100 + seed)
    for trial in range(80):
        df = _ledger(rng)
        scope = [None, [], ["IBKR"], list(rng.choice(ACCOUNTS, 2, replace=False)), ["", None, "ibkr "]][int(rng.integers(0, 5))]
        hist = None
        if rng.random() < 0.6 and len(df):
            hist = {(d.date(), c): float(rng.uniform(0.02, 40)) for d, c in zip(df["Date"], df["Local Currency"]) if rng.random() < 0.7}
        fx = [None, {"USD": 1.0, "THB": 35.0, "EUR": 0.9}, {}][int(rng.integers(0, 3))]
        mv = [0.0, 5000.0, np.nan, np.float64(12.0)][int(rng.integers(0, 4))]
        end = [date(2023, int(rng.integers(1, 13)), 1), date(2020, 1, 1), datetime(2022, 5, 5)][int(rng.integers(0, 3))]
        basis = str(rng.choice(["portfolio", "per_trade"]))
        target = str(rng.choice(["USD", "THB"]))
        outs = [call(lambda mod=mod: mod.get_cash_flows_for_mwr(df.copy(), mv, end, target, fx, "USD", hist, scope, basis), caplog)
                for mod in (ref, new)]  # fmt: skip
        assert outs[0] == outs[1], (seed, trial, basis)
        outs = [call(lambda mod=mod: mod.get_cash_flows_for_mwr(df.copy(), mv, end, "USD", fx, "USD"), caplog) for mod in (ref, new)]
        assert outs[0] == outs[1], (seed, trial, "defaults")
    assert canon(new.get_cash_flows_for_mwr(pd.DataFrame(), 1.0, date(2023, 1, 1), "USD", None, "USD")) == canon(([], []))
