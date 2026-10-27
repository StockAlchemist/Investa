"""Buffett quality metrics (rust/investa_core/src/buffett.rs) against the original.

tests/reference/buffett_metrics.py is the module before the port. All three
models must return the same metrics dict — same keys in the same order, same
types (the counts are ints), bitwise-equal floats — on random concept series
that mix None, NaN, infinities, negative zero and missing periods, because
several metrics read raw values without the `_finite` filter.
"""

import os
import statistics
import struct
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_metrics as new  # noqa: E402
import investa_core  # noqa: E402
from reference import buffett_metrics as ref  # noqa: E402

CONCEPTS = [
    "net_income", "revenue", "pretax_income", "interest_expense", "tax_provision",
    "equity", "short_term_debt", "long_term_debt", "cash", "operating_cash_flow",
    "capex", "shares_diluted", "dividends_paid", "total_assets",
    "depreciation_amortization",
]  # fmt: skip
SECTOR = [
    "provision_for_credit_losses", "loans", "net_interest_income",
    "noninterest_income", "noninterest_expense", "gain_on_sale_real_estate",
    "real_estate_net",
]  # fmt: skip
RATIOS = [
    "Return on Equity (ROE) (%)", "Return on Assets (ROA) (%)",
    "Gross Profit Margin (%)", "Net Profit Margin (%)", "Debt-to-Equity Ratio",
    "Current Ratio", "Some Other Ratio",
]  # fmt: skip


def _same(a, b, path="metrics"):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, dict):
        assert list(a) == list(b), f"{path}: keys {list(a)} vs {list(b)}"
        for k in a:
            _same(a[k], b[k], f"{path}[{k!r}]")
    elif isinstance(a, float):
        assert (np.isnan(a) and np.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _value(rng, scale):
    r = rng.random()
    if r < 0.05:
        return None
    if r < 0.08:
        return float("nan")
    if r < 0.09:
        return float(rng.choice([np.inf, -np.inf]))
    if r < 0.11:
        return float(rng.choice([0.0, -0.0]))
    return float(rng.normal(scale, abs(scale) * 0.6))


def _series(rng, periods, scale, sign=1.0):
    present = [p for p in periods if rng.random() < 0.85]
    return {p: (None if (v := _value(rng, scale)) is None else sign * v) for p in present}


def _company(rng):
    n = int(rng.integers(0, 18))
    periods = [f"{2025 - i}-12-31" for i in range(n)]
    rng.shuffle(periods)  # dict order must not matter
    scale = float(rng.choice([1e6, 1e9]))
    concepts = {}
    for name in CONCEPTS:
        if rng.random() < 0.1:
            continue
        mult = {"shares_diluted": 1e-3, "net_income": 0.1, "capex": 0.05}.get(name, 1.0)
        concepts[name] = _series(rng, periods, scale * mult)
    sector = {name: _series(rng, periods, scale * 0.3) for name in SECTOR if rng.random() < 0.85}
    ratios = {
        p: {label: _value(rng, 15.0) for label in RATIOS if rng.random() < 0.9}
        for p in periods
        if rng.random() < 0.9
    }
    return concepts, sector, ratios


@pytest.mark.parametrize("seed", range(6))
def test_all_three_models_match(seed):
    rng = np.random.default_rng(3100 + seed)
    for _ in range(250):
        concepts, sector, ratios = _company(rng)
        _same(new._compute_generic_metrics(concepts, ratios), ref._compute_generic_metrics(concepts, ratios))
        _same(new._compute_bank_metrics(concepts, sector, ratios), ref._compute_bank_metrics(concepts, sector, ratios))
        _same(new._compute_reit_metrics(concepts, sector, ratios), ref._compute_reit_metrics(concepts, sector, ratios))


def test_helpers_match():
    rng = np.random.default_rng(3200)
    for _ in range(2000):
        concepts, _, _ = _company(rng)
        for period in list(concepts.get("long_term_debt", {}))[:3] + ["1999-12-31"]:
            _same(new._total_debt(concepts, period), ref._total_debt(concepts, period), "total_debt")
        series = [_value(rng, 100.0) for _ in range(int(rng.integers(0, 12)))]
        _same(new._cagr(series), ref._cagr(series), "cagr")


def test_pstdev_is_cpython_exact():
    rng = np.random.default_rng(3300)
    for _ in range(5000):
        n = int(rng.integers(1, 40))
        mag = 10.0 ** float(rng.integers(-300, 300))
        data = [float(x) for x in rng.normal(0, 1, n) * mag]
        if rng.random() < 0.1:
            data = [float(rng.choice([1.0, 2.0, -0.0, 0.0])) for _ in range(n)]
        want = statistics.pstdev(data)
        got = investa_core.py_pstdev(data)
        assert struct.pack("<d", got) == struct.pack("<d", want), (data, got, want)
