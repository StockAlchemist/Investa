"""The historical price lookup (investa_core.PriceTable behind
finutils.get_historical_price) against the original pandas asof path frozen in
tests/reference/fx_lookup.py.

Bitwise and type for type, on random frames. They cover:
- naive, UTC and New York indexes, in ns, us, ms and s units;
- sorted, unsorted and duplicated dates;
- rows with a gap in another column, and a NaN price;
- each price-column name, integer prices, and no price column at all.
Each frame is asked about days before, inside and after its range, plus
datetime, Timestamp, out-of-range and invalid targets. A frame that grows after
its first lookup must get a fresh table.
"""

import math
import os
import struct
import sys
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils as new  # noqa: E402
from reference import fx_lookup as ref  # noqa: E402


def _same(a, b, path):
    assert type(a) is type(b), f"{path}: {type(a).__name__} vs {type(b).__name__}"
    if isinstance(a, float):
        assert (math.isnan(a) and math.isnan(b)) or struct.pack("<d", a) == struct.pack(
            "<d", b
        ), f"{path}: {a!r} vs {b!r}"
    else:
        assert a == b, f"{path}: {a!r} vs {b!r}"


def _frame(rng):
    n = int(rng.integers(1, 60))
    start = pd.Timestamp("2020-01-01") + pd.Timedelta(days=int(rng.integers(0, 300)))
    days = np.sort(rng.choice(np.arange(0, 400), n, replace=bool(rng.random() < 0.2)))
    idx = pd.DatetimeIndex(start + pd.to_timedelta(days, unit="D"))
    if rng.random() < 0.3:
        idx = idx + pd.Timedelta(hours=int(rng.choice([5, 21])))  # intraday stamps
    tz = rng.choice(["naive", "UTC", "America/New_York"], p=[0.5, 0.35, 0.15])
    if tz != "naive":
        idx = idx.tz_localize(str(tz))
    idx = idx.as_unit(str(rng.choice(["ns", "us", "ms", "s"])))
    col = str(
        rng.choice(
            ["price", "Close", "Adj Close", "both", "none"],
            p=[0.5, 0.15, 0.1, 0.15, 0.1],
        )
    )
    values = rng.lognormal(3.5, 0.1, n)
    if rng.random() < 0.15:
        values = np.round(values).astype(np.int64)
    data = {}
    if col == "both":
        data["Close"] = values * 1.01
        data["price"] = values
    elif col != "none":
        data[col] = values
    else:
        data["Open"] = values
    if rng.random() < 0.4:
        other = rng.normal(0, 1, n)
        other[rng.random(n) < 0.3] = np.nan  # rows asof must skip
        data["Volume"] = other
    df = pd.DataFrame(data, index=idx)
    if (
        col in ("price", "both")
        and rng.random() < 0.2
        and df["price"].dtype.kind == "f"
    ):
        df.loc[df.index[rng.integers(0, n)], "price"] = np.nan
    if rng.random() < 0.15:
        df = df.iloc[rng.permutation(n)]  # unsorted
    return df


def _targets(rng, df):
    lo = (
        df.index.min().tz_localize(None).date()
        if df.index.tz is not None
        else df.index.min().date()
    )
    out = [lo + timedelta(days=int(d)) for d in rng.integers(-20, 450, 12)]
    out += [lo, lo - timedelta(days=1), date(1600, 1, 1), date(2300, 1, 1)]
    out += [
        datetime(2020, 6, 1, 12),
        pd.Timestamp("2020-06-01"),
        pd.Timestamp("2020-06-01", tz="UTC"),
        None,
        "2020-06-01",
    ]
    return out


@pytest.mark.parametrize("seed", range(6))
def test_lookup_matches(seed):
    rng = np.random.default_rng(6800 + seed)
    for trial in range(250):
        df = _frame(rng)
        prices = {"K": df}
        for t in _targets(rng, df):
            _same(
                new.get_historical_price("K", t, prices),
                ref.get_historical_price("K", t, prices),
                f"{seed}/{trial} {t!r}",
            )


def test_missing_and_empty_frames():
    for prices in ({}, {"K": pd.DataFrame({"price": []}, index=pd.DatetimeIndex([]))}):
        _same(
            new.get_historical_price("K", date(2020, 1, 1), prices),
            ref.get_historical_price("K", date(2020, 1, 1), prices),
            "empty",
        )


def test_a_frame_that_changes_gets_a_new_table():
    idx = pd.date_range("2020-01-01", periods=5, freq="D")
    df = pd.DataFrame({"price": [1.0, 2.0, 3.0, 4.0, 5.0]}, index=idx)
    prices = {"K": df}
    assert new.get_historical_price("K", date(2020, 1, 9), prices) == 5.0
    df.loc[pd.Timestamp("2020-01-08")] = 8.0  # grows in place
    for t in (date(2020, 1, 7), date(2020, 1, 9)):
        _same(
            new.get_historical_price("K", t, prices),
            ref.get_historical_price("K", t, prices),
            str(t),
        )
