"""Phase 7b: the ratio history (rust/investa_core/src/ratios.rs plus the
resolved lookups in financial_ratios) against the original frozen in
tests/reference/financial_ratios.py.

Bitwise and type for type, the frame's columns, dtypes and index included. The
random statements carry:
- every label and alias the ratios read, with filed zeros (Python's `or`
  falls through them), negatives, NaN and integers;
- object columns holding None;
- income, balance-sheet and cash-flow periods that overlap only partly, and a
  cash flow that is missing or empty;
- annual and quarterly spacing;
- prices as a frame, a series or none, some too stale to count.
Text cells and duplicate labels must take the original code and still match.
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

import financial_ratios as new  # noqa: E402
from reference import financial_ratios as ref  # noqa: E402

LABELS = sorted(
    {
        label
        for _s, item in new._RATIO_LOOKUPS
        for label in [item] + new.STATEMENT_ALIASES.get(item, [])
    }
)


def _same_value(a, b, path):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _same(a, b, path):
    assert list(a.columns) == list(b.columns), f"{path}: columns"
    assert a.index.equals(b.index), f"{path}: index"
    assert list(a.dtypes) == list(b.dtypes), (
        f"{path}: dtypes {list(a.dtypes)} vs {list(b.dtypes)}"
    )
    for col in a.columns:
        for i, (x, y) in enumerate(zip(a[col].tolist(), b[col].tolist())):
            _same_value(x, y, f"{path}[{col}][{i}]")


def _value(rng):
    r = rng.random()
    if r < 0.12:
        return np.nan
    if r < 0.2:
        return 0.0
    if r < 0.27:
        return float(-abs(rng.normal(5e8, 3e8)))
    return float(abs(rng.lognormal(19, 2)))


def _statement(rng, periods, quarterly):
    labels = list(rng.choice(LABELS, int(rng.integers(3, len(LABELS))), replace=False))
    data = {p: [_value(rng) for _ in labels] for p in periods}
    df = pd.DataFrame(data, index=labels)
    r = rng.random()
    if r < 0.1:
        df = (
            df.round(0).astype(object).where(df.notna(), None)
        )  # object cells with None
    elif r < 0.15:
        df = df.fillna(0).astype(np.int64)
    return df


def _periods(rng, quarterly):
    step = pd.DateOffset(months=3) if quarterly else pd.DateOffset(years=1)
    n = int(rng.integers(1, 16))
    start = pd.Timestamp("2008-12-31") + pd.DateOffset(months=int(rng.integers(0, 12)))
    return [start + step * k for k in range(n)]


def _case(rng):
    quarterly = rng.random() < 0.3
    periods = _periods(rng, quarterly)
    pick = lambda: [p for p in periods if rng.random() < 0.85] or periods[:1]  # noqa: E731
    fin = _statement(rng, pick(), quarterly)
    bs = _statement(rng, pick(), quarterly)
    r = rng.random()
    cf = (
        None
        if r < 0.2
        else (pd.DataFrame() if r < 0.25 else _statement(rng, pick(), quarterly))
    )
    for df in (fin, bs, cf):
        if df is not None and not df.empty and rng.random() < 0.15:
            df.columns = [
                c.strftime("%Y-%m-%d") for c in df.columns
            ]  # text period columns
    prices = None
    r = rng.random()
    if r < 0.7:
        idx = pd.bdate_range(
            periods[0] - pd.Timedelta(days=400), periods[-1] + pd.Timedelta(days=30)
        )
        series = pd.Series(
            np.cumprod(1 + rng.normal(0, 0.01, len(idx))) * 50, index=idx
        )
        if rng.random() < 0.3:
            series = series[
                rng.random(len(series)) < 0.02
            ]  # sparse: some ends too stale to price
        prices = series.to_frame("Close") if r < 0.4 else series
    ppy = 4 if quarterly else 1
    return fin, bs, cf, ppy, prices


def _run(mod, case):
    fin, bs, cf, ppy, prices = case
    copy = lambda df: None if df is None else df.copy()  # noqa: E731
    return mod.calculate_key_ratios_timeseries(
        copy(fin), copy(bs), copy(cf), ppy, None if prices is None else prices.copy()
    )


def test_aliases_are_the_originals():
    assert new.STATEMENT_ALIASES == ref.STATEMENT_ALIASES


@pytest.mark.parametrize("seed", range(6))
def test_ratio_history_matches(seed):
    rng = np.random.default_rng(7100 + seed)
    for trial in range(150):
        case = _case(rng)
        _same(_run(new, case), _run(ref, case), f"{seed}/{trial}")


def test_frames_the_resolver_cannot_read_take_the_original_code():
    rng = np.random.default_rng(7200)
    for trial in range(60):
        fin, bs, cf, ppy, prices = _case(rng)
        if trial % 2 == 0:
            fin = fin.astype(object)
            fin.iloc[0, 0] = "n/a"  # a text cell
        else:
            bs = pd.concat([bs, bs.iloc[:1]])  # a duplicated label
        case = (fin, bs, cf, ppy, prices)
        assert not (new._numeric_statement(fin) and new._numeric_statement(bs))
        outcomes = []
        for mod in (new, ref):
            try:
                outcomes.append(("ok", _run(mod, case)))
            except Exception as exc:
                outcomes.append(("raised", type(exc)))
        assert outcomes[0][0] == outcomes[1][0]
        if outcomes[0][0] == "ok":
            _same(outcomes[0][1], outcomes[1][1], f"fallback {trial}")
        else:
            assert outcomes[0][1] is outcomes[1][1]


# --- the derived statement rows ---------------------------------------------

import copy as _copy  # noqa: E402

import edgar_provider  # noqa: E402
from reference import edgar_derived as ref_derived  # noqa: E402

_DERIVED_CONCEPTS = ["short_term_debt", "long_term_debt", "pretax_income", "interest_expense",
                     "operating_cash_flow", "capex", "total_assets", "equity", "revenue", "net_income"]  # fmt: skip


def _concepts(rng):
    ends = [f"{y}-12-31" for y in range(2009, 2009 + int(rng.integers(1, 15)))]
    if rng.random() < 0.3:
        ends += [
            str(e)
            for e in rng.choice(
                [
                    "not-a-date",
                    "2015-02-30",
                    "1600-12-31",
                    "2300-06-30",
                    "2016-06-30T00:00:00",
                ],
                2,
            )
        ]
    values = {}
    for name in _DERIVED_CONCEPTS:
        if rng.random() < 0.8:
            values[name] = {
                e: float(rng.normal(1e9, 5e8)) for e in ends if rng.random() < 0.8
            }
    return values


def _frames(rng, values):
    labels = {
        "revenue": "Total Revenue",
        "net_income": "Net Income",
        "total_assets": "Total Assets",
    }
    if rng.random() < 0.3:
        labels["equity"] = (
            "Total Liabilities Net Minority Interest"  # the row already exists
        )
    if rng.random() < 0.3:
        labels["pretax_income"] = "Ebit"
    income = edgar_provider._frame_from_concepts(
        values,
        {
            k: v
            for k, v in labels.items()
            if k in ("revenue", "net_income", "pretax_income")
        },
    )
    balance = edgar_provider._frame_from_concepts(
        values, {k: v for k, v in labels.items() if k in ("total_assets", "equity")}
    )
    cashflow = edgar_provider._frame_from_concepts(
        values,
        {"operating_cash_flow": "Operating Cash Flow"},
        negate={"capex": "Capital Expenditure"},
    )
    if rng.random() < 0.1 and not balance.empty:
        balance = balance.astype(object)  # not float64: written cell by cell
    return income, balance, (None if rng.random() < 0.15 else cashflow)


def _same_frame(a, b, path):
    if b is None:
        assert a is None, path
        return
    _same(a, b, path)


@pytest.mark.parametrize("seed", range(4))
def test_derived_rows_match(seed):
    rng = np.random.default_rng(7300 + seed)
    for trial in range(200):
        values = _concepts(rng)
        frames = _frames(rng, values)
        got, want = _copy.deepcopy(frames), _copy.deepcopy(frames)
        edgar_provider._add_derived_rows(got[0], got[1], values, got[2])
        ref_derived._add_derived_rows(want[0], want[1], values, want[2])
        for name, a, b in zip(("income", "balance", "cashflow"), got, want):
            _same_frame(a, b, f"{seed}/{trial} {name}")
