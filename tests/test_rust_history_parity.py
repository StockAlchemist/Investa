"""Phase 6b: the daily holdings valuation (rust/investa_core/src/history.rs plus
the shim in portfolio_history) against the original frozen in
tests/reference/history_valuation.py.

Bitwise: the value series (index and every value), the error flag and the
status message, on random grids. They have symbols with and without a price
feed, published NAVs and manual overrides, gaps in the market prices with and
without last-trade fallbacks, empty, negative and NaN quantities, excluded
accounts, cash in several currencies, a missing FX pair, and a grid wide
enough that one day's cells run past NumPy's 8,192-element reduction buffer.
"""

import math
from datetime import date
import os
import struct
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import portfolio_history as new  # noqa: E402
from reference import history_valuation as ref  # noqa: E402


def _same_float(a, b, path):
    assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
        "<d", b
    ), f"{path}: {a!r} vs {b!r}"


def _same(a, b):
    (s_new, err_new, msg_new), (s_ref, err_ref, msg_ref) = a, b
    assert s_new.index.equals(s_ref.index)
    assert s_new.dtype == s_ref.dtype
    for i, (x, y) in enumerate(zip(s_new.tolist(), s_ref.tolist())):
        _same_float(x, y, f"day {i}")
    assert (err_new, msg_new) == (err_ref, msg_ref)


_NAV = {}


def _fund_nav(symbol_ids, date_range):
    return {i: _NAV[s] for i, s in symbol_ids.items() if s in _NAV}


@pytest.fixture(autouse=True)
def _stub_nav(monkeypatch):
    monkeypatch.setattr(new, "_fund_nav_series", _fund_nav)
    monkeypatch.setattr(ref, "_fund_nav_series", _fund_nav)


def _prices(rng, start, n, col):
    idx = pd.date_range(start, periods=n, freq="B", tz="UTC") + pd.Timedelta(hours=21)
    values = 50 * np.cumprod(1 + rng.normal(0, 0.02, n))
    df = pd.DataFrame({col: values}, index=idx)
    if rng.random() < 0.3:
        df = df.iloc[rng.random(n) > 0.2]  # holes
    return df


def _case(rng, wide=False):
    days = int(rng.integers(1, 400))
    date_range = pd.date_range(
        "2018-01-01", periods=days, freq="D", tz="UTC"
    ) + pd.Timedelta(hours=21)
    n_sym = int(rng.integers(600, 700)) if wide else int(rng.integers(1, 12))
    n_acc = int(rng.integers(13, 16)) if wide else int(rng.integers(1, 5))
    id_to_symbol = {i: f"S{i}" for i in range(n_sym)}
    if n_sym > 1 and rng.random() < 0.5:
        id_to_symbol[n_sym - 1] = "$CASH"
    id_to_account = {
        i: ["IBKR", "KIM ENG", "SET", "E*TRADE", "X"][i % 5] + str(i)
        for i in range(n_acc)
    }
    yf_map, hist = {}, {}
    _NAV.clear()
    for i, s in id_to_symbol.items():
        r = rng.random()
        if r < 0.6:
            yf_map[s] = s + ".YF"
            col = str(rng.choice(["price", "Close", "Adj Close"]))
            start = date_range[0] - pd.Timedelta(days=int(rng.integers(-60, 60)))
            hist[s + ".YF"] = _prices(
                rng, start.tz_localize(None).normalize(), int(rng.integers(5, 450)), col
            )
        elif r < 0.7:
            _NAV[s] = pd.Series(
                10 + rng.normal(0, 0.1, days).cumsum(),
                index=date_range - pd.Timedelta(hours=int(rng.integers(0, 30))),
            )
    fx = {}
    for pair in ("USDTHB=X", "EUR=X", "JPY=X"):
        if rng.random() < 0.8:
            df = _prices(rng, "2017-11-01", 600, "price")
            fx[pair] = df * {"USDTHB=X": 0.7, "EUR=X": 0.018, "JPY=X": 2.9}[pair]
    currencies = ["USD", "THB", "EUR", "JPY", "CHF"]
    account_currency_map = {
        (name.title() if rng.random() < 0.5 else name): str(rng.choice(currencies))
        for name in id_to_account.values()
        if rng.random() < 0.8
    }
    shape = (days, n_sym, n_acc)
    qty = rng.choice([0.0, 0.0, 10.0, 3.5, -2.0, 120.0], shape)
    if rng.random() < 0.1:
        qty[rng.random(shape) < 0.01] = np.nan
    qty[:, :, rng.random(n_acc) < 0.2] = 0.0  # accounts that hold nothing
    last = None
    if rng.random() < 0.8:
        last = rng.lognormal(3, 1, shape)
        last[rng.random(shape) < 0.3] = np.nan
    cash = rng.choice([0.0, 0.0, 1000.0, -250.0, 5e5], (days, n_acc))
    included = {int(a) for a in range(n_acc) if rng.random() < 0.8}
    overrides = None
    if rng.random() < 0.5:
        overrides = {
            f"s{i}": {"price": float(rng.choice([12.5, 0.0, np.nan]))}
            if rng.random() < 0.7
            else 7.25
            for i in rng.choice(n_sym, min(n_sym, 3), replace=False)
        }
    return dict(
        date_range=date_range,
        daily_holdings_qty_np=qty,
        daily_last_prices_np=last,
        daily_cash_balances_np=cash,
        historical_prices_yf_unadjusted=hist,
        historical_fx_yf=fx,
        target_currency=str(rng.choice(["USD", "THB", "EUR", "CHF"])),
        account_currency_map=account_currency_map,
        id_to_symbol=id_to_symbol,
        id_to_account=id_to_account,
        internal_to_yf_map=yf_map,
        account_ids_to_include_set=included,
        default_currency=str(rng.choice(["USD", "THB"])),
        interval="1d",
        manual_overrides_dict=overrides,
    )


def _both(case):
    import copy

    return (
        new._value_daily_holdings_vectorized(**copy.deepcopy(case)),
        ref._value_daily_holdings_vectorized(**copy.deepcopy(case)),
    )


@pytest.mark.parametrize("seed", range(6))
def test_valuation_matches(seed):
    rng = np.random.default_rng(6500 + seed)
    for _ in range(60):
        _same(*_both(_case(rng)))


def test_wide_grid_matches():
    rng = np.random.default_rng(6600)
    for _ in range(3):
        _same(*_both(_case(rng, wide=True)))


# --- Phase 6c: the accumulation scans --------------------------------------

from reference import history_accumulation as ref_acc  # noqa: E402


def _same_frame(a, b, path):
    assert list(a.columns) == list(b.columns), (
        f"{path}: columns {list(a.columns)} vs {list(b.columns)}"
    )
    assert a.index.equals(b.index), f"{path}: index"
    assert list(a.dtypes) == list(b.dtypes), f"{path}: dtypes"
    for col in a.columns:
        for i, (x, y) in enumerate(zip(a[col].tolist(), b[col].tolist())):
            _same_float(x, y, f"{path}[{col}][{i}]")


def _daily_df(rng):
    n = int(rng.integers(1, 900))
    start = pd.Timestamp("2015-01-01", tz="UTC") + pd.Timedelta(
        days=int(rng.integers(0, 400))
    )
    idx = pd.date_range(start, periods=n, freq="D")
    r = rng.normal(0.0004, 0.012, n)
    if rng.random() < 0.7:
        r[0] = np.nan  # the first day has no return
    for v in (-1.0, -0.0, -0.9999999, np.nan, 2.0):
        if rng.random() < 0.3:
            r[rng.integers(0, n)] = v
    value = 1e5 * np.cumprod(1 + np.nan_to_num(r))
    flow = rng.choice([0.0, 0.0, 0.0, 5e3, -2e4, np.nan], n)
    if rng.random() < 0.3:
        flow[0] = -1e3  # a cost basis that starts negative
    df = pd.DataFrame(
        {
            "value": value,
            "net_flow": flow,
            "daily_gain": rng.normal(0, 500, n),
            "daily_return": r,
        },
        index=idx,
    )
    benches = []
    for sym in ("SPY", "QQQ"):
        if rng.random() < 0.7:
            p = 300 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
            p[rng.random(n) < float(rng.choice([0.0, 0.3, 1.0]))] = np.nan
            df[f"{sym} Price"] = p
        benches.append(sym)
    if rng.random() < 0.2:
        df = df.drop(columns=str(rng.choice(["value", "net_flow"])))
    first, last = idx[0].date(), idx[-1].date()
    span = (last - first).days
    lo = first + pd.Timedelta(days=int(rng.integers(-5, max(1, span))))
    hi = lo + pd.Timedelta(days=int(rng.integers(0, span + 10)))
    interval = str(rng.choice(["D", "1d", "W", "M", "ME", "Y"]))
    return df, benches, interval, lo, hi


@pytest.mark.parametrize("seed", range(6))
def test_accumulation_matches(seed):
    rng = np.random.default_rng(6700 + seed)
    for trial in range(80):
        df, benches, interval, lo, hi = _daily_df(rng)
        out_new = new._calculate_accumulated_gains_and_resample(
            df.copy(), benches, interval, lo, hi
        )
        out_ref = ref_acc._calculate_accumulated_gains_and_resample(
            df.copy(), benches, interval, lo, hi
        )
        path = f"seed{seed}/trial{trial}/{interval}"
        _same_frame(out_new[0], out_ref[0], path)
        assert type(out_new[1]) is type(out_ref[1]), path
        _same_float(float(out_new[1]), float(out_ref[1]), f"{path}: twr")
        assert out_new[2] == out_ref[2], path


def test_duplicate_dates_behave_like_the_original():
    # pandas skips a reindex onto an identical index, duplicates or not, and
    # refuses one from duplicate dates onto any other: with a price gap the
    # original's whole calculation fails, and the port must fail with it.
    idx = pd.DatetimeIndex(["2020-01-01", "2020-01-01", "2020-01-02"], tz="UTC")
    args = (["SPY"], "D", date(2020, 1, 1), date(2020, 1, 2))
    for prices in ([1.0, 2.0, 3.0], [1.0, 2.0, np.nan], [np.nan, 2.0, 3.0]):
        df = pd.DataFrame(
            {"daily_return": [np.nan, 0.01, 0.02], "SPY Price": prices}, index=idx
        )
        out_new = new._calculate_accumulated_gains_and_resample(df.copy(), *args)
        out_ref = ref_acc._calculate_accumulated_gains_and_resample(df.copy(), *args)
        _same_frame(out_new[0], out_ref[0], str(prices))
        _same_float(float(out_new[1]), float(out_ref[1]), str(prices))
        assert out_new[2] == out_ref[2]
