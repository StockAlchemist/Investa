"""Phase 16c: the FX and historical-price lookups of finutils in Rust
(rust/investa_core/src/fu_fx.rs) against the original frozen in
tests/reference/finutils_p16.py.

Current rates take codes of every case and type, rate tables keyed by code or
Yahoo pair with NaN, zero and text values, with and without the static
fallback. Historical prices take frames on naive, UTC and New York indexes,
unsorted and duplicated dates, date-object and text indexes, every price
column (and none), NaN rows, boolean and object columns, against target days
as dates, datetimes, Timestamps and aware Timestamps, before, inside and
after the data. The price-table cache is exercised by repeating lookups on
one frame and by mutating a frame between lookups. The USD bridge combines
them. Results, logs and warnings agree.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
import finutils as new  # noqa: E402
from reference import finutils_p16 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402

CODES = ["USD", "usd", "THB", "EUR", "JPY", "XXX", "", None, 5, "Thb"]


def _both(caplog, name, *args):
    outs = [call(lambda mod=mod: getattr(mod, name)(*args), caplog) for mod in (ref, new)]
    assert outs[0] == outs[1], (name, args[:2])
    return outs[0]


@pytest.mark.parametrize("fallback", [True, False])
def test_conversion_rate(caplog, monkeypatch, fallback):
    if fallback:
        monkeypatch.setattr(config, "STATIC_FX_FALLBACK", {"EUR": 0.92, "JPY": float("nan"), "THB": 36.0}, raising=False)
    else:
        monkeypatch.delattr(config, "STATIC_FX_FALLBACK", raising=False)
    rng = np.random.default_rng(24000)
    values = [1.0, 35.5, 0.0, float("nan"), None, "x", 1e-10, np.float64(150.0), -2.0]
    keys = ["USD", "THB", "USDTHB=X", "EUR", "EUR=X", "JPY", "JPY=X", "XXX"]
    for trial in range(300):
        rates = {k: values[int(rng.integers(0, len(values)))] for k in keys if rng.random() < 0.6}
        if trial % 17 == 0:
            rates = None if trial % 2 else [("USD", 1.0)]
        for a in CODES:
            for b in rng.choice(np.array(CODES, dtype=object), 3):
                _both(caplog, "get_conversion_rate", a, b, rates)


def _frame(rng):
    n = int(rng.integers(1, 30))
    start = pd.Timestamp("2021-01-01")
    days = np.sort(rng.integers(0, 400, n))
    idx = pd.DatetimeIndex([start + pd.Timedelta(days=int(d)) for d in days])
    kind = int(rng.integers(0, 7))
    if kind == 1:
        idx = idx.tz_localize("UTC")
    elif kind == 2:
        idx = idx.tz_localize("America/New_York")
    values = rng.uniform(1, 100, n)
    col = ["price", "Adj Close", "Close", "Weird"][int(rng.choice(4, p=[0.5, 0.2, 0.2, 0.1]))]
    df = pd.DataFrame({col: values}, index=idx)
    if rng.random() < 0.3:
        df["volume"] = np.where(rng.random(n) < 0.2, np.nan, 1.0)
    if rng.random() < 0.15:
        df.iloc[int(rng.integers(0, n)), 0] = np.nan
    if kind == 3:
        df = df.iloc[::-1]  # unsorted
    elif kind == 4 and n > 2:
        df = pd.concat([df, df.iloc[:1]]).sort_index()  # a duplicated day
    elif kind == 5:
        df.index = [d.date() for d in df.index]  # date objects
    elif kind == 6 and rng.random() < 0.5:
        df[col] = df[col] > 50  # a boolean column
    if rng.random() < 0.05:
        df[col] = df[col].astype(object)
    return df


def _targets(rng):
    out = [dt.date(2020, 6, 1), dt.date(2021, 3, 15), dt.date(2022, 12, 31), None, "2021-03-15", 20210315,
           dt.datetime(2021, 3, 15, 13, 0), pd.Timestamp("2021-03-15"), pd.Timestamp("2021-03-15", tz="UTC"),
           dt.date(1, 1, 1), dt.date(9999, 12, 31)]  # fmt: skip
    return out + [dt.date(2021, 1, 1) + dt.timedelta(days=int(d)) for d in rng.integers(0, 420, 5)]


@pytest.mark.parametrize("seed", range(6))
def test_historical_price(caplog, seed):
    rng = np.random.default_rng(24100 + seed)
    for trial in range(60):
        df = _frame(rng)
        prices = {"AAPL": df, "EMPTY": df.iloc[:0], "THB=X": df}
        for t in _targets(rng):
            for key in ("AAPL", "EMPTY", "MISSING"):
                _both(caplog, "get_historical_price", key, t, prices)
        tables = [mod._build_price_table(df.copy()) for mod in (ref, new)]
        assert (tables[0] is None) == (tables[1] is None)
        if tables[0] is not None:
            for ns in rng.integers(1.5e18, 1.7e18, 50).tolist() + [0, 2**62]:
                assert repr(tables[0].asof(int(ns))) == repr(tables[1].asof(int(ns)))
        if isinstance(df.index, pd.DatetimeIndex) and len(df) > 1 and df.iloc[:, 0].dtype.kind == "f":
            df.iloc[-1, 0] = 999.0  # the cache keys on shape and ends, not values
            _both(caplog, "get_historical_price", "AAPL", dt.date(2022, 12, 31), prices)
            grown = pd.concat([df, df.iloc[-1:].set_axis(df.index[-1:] + pd.Timedelta(days=1))])
            prices["AAPL"] = grown
            _both(caplog, "get_historical_price", "AAPL", dt.date(2022, 12, 31), prices)


@pytest.mark.parametrize("seed", range(3))
def test_usd_bridge(caplog, seed):
    rng = np.random.default_rng(24200 + seed)
    for trial in range(40):
        data = {k: _frame(rng) for k in ("EUR=X", "USDTHB=X", "JPY=X") if rng.random() < 0.8}
        if trial % 9 == 0:
            data = None
        for a in CODES[:6]:
            for b in CODES[:6]:
                for t in (dt.date(2021, 3, 15), dt.date(2019, 1, 1), dt.datetime(2021, 3, 15), "2021-03-15"):
                    _both(caplog, "get_historical_rate_via_usd_bridge", a, b, t, data)
