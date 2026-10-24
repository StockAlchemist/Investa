"""The Rust kernels (rust/investa_core) against the kernels they replaced.

tests/reference/numba_kernels.py is the Numba source exactly as it ran in
production, minus the decorators. Run as plain Python it is the IEEE-exact
meaning of that source, and the Rust port must reproduce it bit for bit — so
these tests demand equality, not closeness. (Production Numba ran with
fastmath, which reorders arithmetic: it agreed with both to ~1e-14 relative.)

The ledgers are random but seeded, built to reach every branch: each
transaction type, absent types (id -1), cash and non-cash symbols, transfers
with and without a destination, ratio and target-quantity splits, shortable
symbols, Auto and Manual accounts, and zero amounts that send every cash rule
down its fallback chain.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402
import portfolio_analyzer  # noqa: E402
import portfolio_valuation_kernels as kernels  # noqa: E402
from reference import numba_kernels as ref  # noqa: E402

CASES = 400


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    # Bitwise, so -0.0 vs 0.0 and NaN payloads count, and NaN == NaN.
    assert a.tobytes() == b.tobytes()


def _amounts(rng, n, scale):
    v = rng.normal(0, scale, n)
    v[rng.random(n) < 0.3] = 0.0
    v[rng.random(n) < 0.1] *= -1
    return v


def _ledger(rng):
    ns, na = int(rng.integers(1, 7)), int(rng.integers(1, 5))
    n = int(rng.integers(0, 60))
    ids = rng.permutation(15).astype(np.int64)
    ids[rng.random(15) < 0.15] = -1
    present = [int(x) for x in ids if x != -1] or [0]
    qty = np.abs(_amounts(rng, n, 50))
    qty[rng.random(n) < 0.1] *= -1
    splits = np.zeros(n)
    m = rng.random(n) < 0.3
    splits[m] = rng.choice([0.5, 2.0, 3.0, 10.0], m.sum())
    m2 = rng.random(n) < 0.15
    splits[m2] = -rng.integers(1, 500, m2.sum()).astype(float)
    return dict(
        ns=ns,
        na=na,
        n=n,
        cash=int(rng.integers(-1, ns)),
        type_ids=[int(x) for x in ids[:14]],
        types=rng.choice(present + [-1], n).astype(np.int64),
        dates=np.sort(rng.integers(738000, 738060, n)).astype(np.int64),
        syms=rng.integers(0, ns, n).astype(np.int64),
        accs=rng.integers(0, na, n).astype(np.int64),
        to_accs=np.where(rng.random(n) < 0.5, -1, rng.integers(0, na, n)).astype(
            np.int64
        ),
        qty=qty,
        price=np.abs(_amounts(rng, n, 100)),
        totals=_amounts(rng, n, 3000),
        comms=np.abs(_amounts(rng, n, 5)),
        splits=splits,
        ccys=rng.integers(-1, 3, n).astype(np.int64),
        shortable=rng.choice(ns, int(rng.integers(0, ns + 1)), replace=False).astype(
            np.int64
        ),
        modes=rng.integers(0, 2, na).astype(np.int64),
        tol=float(rng.choice([1e-6, 1e-4, 0.01])),
        atypes=rng.integers(-1, 13, n).astype(np.int64),
        fx=np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0.02, 40, n)),
        lcur=rng.integers(0, 3, n).astype(np.int64),
    )


def _holdings_args(rng, L):
    (split, ssplit, buy, dep, sell, wd, ss, btc, tr, so, fees, div, intr, tax) = L[
        "type_ids"
    ]
    return (
        int(rng.integers(737990, 738070)),
        L["dates"], L["syms"], L["accs"], L["to_accs"], L["types"],
        L["qty"], L["price"], L["totals"], L["comms"], L["splits"], L["ccys"],
        L["ns"], L["na"], 3,
        split, ssplit, buy, dep, sell, wd, ss, btc, tr, so, fees, div, intr, tax,
        L["cash"], L["tol"], L["shortable"], L["modes"],
    )  # fmt: skip


def _daily_args(rng, L):
    (split, ssplit, buy, dep, sell, wd, ss, btc, tr, so, fees, div, intr, tax) = L[
        "type_ids"
    ]
    days = np.arange(737995, 738065, int(rng.integers(1, 4)), dtype=np.int64)
    return (
        days, L["dates"], L["syms"], L["to_accs"], L["accs"], L["types"],
        L["qty"], L["comms"], L["splits"], L["price"], L["totals"],
        L["ns"], L["na"],
        split, ssplit, buy, dep, sell, wd, ss, btc, tr, so, div, intr, fees, tax,
        L["cash"], L["tol"], L["shortable"], L["modes"],
    )  # fmt: skip


def _core_args(L):
    return (
        L["syms"], L["accs"], L["atypes"], L["qty"], L["price"], L["comms"],
        L["totals"], L["splits"], L["to_accs"], L["lcur"], L["fx"],
        L["shortable"], L["tol"], L["cash"], L["n"], L["ns"], L["na"], L["modes"],
    )  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_single_date_holdings_match_reference(seed):
    rng = np.random.default_rng(1000 + seed)
    for _ in range(CASES // 4):
        args = _holdings_args(rng, _ledger(rng))
        for got, want in zip(
            kernels._calculate_holdings_numba(*args),
            ref._calculate_holdings_numba(*args),
        ):
            _same(got, want)


@pytest.mark.parametrize("seed", range(4))
def test_daily_holdings_match_reference(seed):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(CASES // 4):
        args = _daily_args(rng, _ledger(rng))
        for got, want in zip(
            kernels._calculate_daily_holdings_chronological_numba(*args),
            ref._calculate_daily_holdings_chronological_numba(*args),
        ):
            _same(got, want)


@pytest.mark.parametrize("seed", range(4))
def test_analyzer_core_matches_reference(seed):
    rng = np.random.default_rng(3000 + seed)
    for _ in range(CASES // 4):
        args = _core_args(_ledger(rng))
        for got, want in zip(
            portfolio_analyzer._process_numba_core(*args),
            ref._process_numba_core(*args),
        ):
            _same(got, want)


def test_dividend_amount_matches_reference():
    rng = np.random.default_rng(4000)
    for _ in range(2000):
        q, p, t = (float(x) for x in rng.choice([0.0, 1e-10, 3.5, -2.0, 17.25], 3))
        assert kernels._dividend_amount(q, p, t) == ref._dividend_amount(q, p, t)


def test_analyzer_type_numbers_agree_with_rust():
    rust = {
        k: v
        for k, v in vars(investa_core.analyzer_types).items()
        if k.startswith("TYPE_")
    }
    assert len(rust) == 14
    for name, value in rust.items():
        assert getattr(portfolio_analyzer, name) == value == getattr(ref, name)


def test_a_bad_index_is_an_error_not_a_stray_read():
    """Numba with bounds checking off read whatever lay past the array."""
    rng = np.random.default_rng(5000)
    L = _ledger(rng)
    while L["n"] == 0:
        L = _ledger(rng)
    L["syms"] = L["syms"].copy()
    L["syms"][0] = L["ns"] + 5
    L["cash"] = -1
    args = list(_core_args(L))
    with pytest.raises(IndexError):
        portfolio_analyzer._process_numba_core(*args)


def test_mismatched_column_lengths_are_refused():
    rng = np.random.default_rng(6000)
    L = _ledger(rng)
    while L["n"] == 0:
        L = _ledger(rng)
    args = list(_holdings_args(rng, L))
    args[6] = args[6][:-1]  # tx_quantities one short
    with pytest.raises(ValueError):
        kernels._calculate_holdings_numba(*args)


def test_negative_indices_wrap_like_numpy():
    """An id of -1 addresses the last row, as it did under NumPy/Numba."""
    rng = np.random.default_rng(7000)
    L = _ledger(rng)
    while L["n"] == 0:
        L = _ledger(rng)
    L["syms"] = L["syms"].copy()
    L["syms"][0] = -1
    args = _core_args(L)
    for got, want in zip(
        portfolio_analyzer._process_numba_core(*args), ref._process_numba_core(*args)
    ):
        _same(got, want)
