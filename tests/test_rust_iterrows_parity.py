"""The `iterrows` row views (rust/investa_core/src/pyhelp.rs) against
`DataFrame.iterrows` itself.

The ported loops read a row only with `row[col]` and `row.get(col, default)`.
For every frame here, each row the helper yields must answer those exactly as
the Series `iterrows` yields: the same label, and for every column, a missing
column and a non-text key, the same value (type, repr and NaN-ness), the same
default or the same exception and warnings. Frames cover every column dtype a
ledger or report carries (float, int, nullable Int64 with NA, naive and
zone-aware datetimes, bool, text with None and NaN, category, string), the
index types the loops see, and the shapes the helper must hand to pandas:
homogeneous numeric or datetime values, duplicate or MultiIndex columns, a
DataFrame subclass, and an empty frame.
"""

import math
import os
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import investa_core  # noqa: E402


class SubFrame(pd.DataFrame):
    @property
    def _constructor(self):
        return SubFrame


def _ledger(rng, n):
    dates = pd.Timestamp("2024-01-02") + pd.to_timedelta(rng.integers(0, 300, n), unit="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Type": rng.choice(["Buy", "Sell", "Dividend", "Split", None], n),
            "Symbol": rng.choice(["AAPL", "$CASH", "MSFT"], n),
            "Quantity": rng.choice([1.0, 2.5, np.nan, 0.0], n),
            "Price/Share": rng.normal(100, 10, n),
            "Total Amount": rng.choice([10.0, np.nan, -3.0], n),
            "Commission": rng.choice([0.0, 1.0, np.nan], n),
            "Split Ratio": rng.choice([np.nan, 2.0, 0.5], n),
            "Account": rng.choice(["IBKR", "SET", "Kim Eng"], n),
            "Local Currency": rng.choice(["USD", "THB", None], n),
            "Note": rng.choice(["", "x", None, np.nan], n).astype(object),
            "original_index": pd.array(rng.integers(0, 50, n), dtype="Int64"),
            "user_id": rng.integers(1, 3, n),
        }
    )


def _frames():
    rng = np.random.default_rng(32000)
    out = []
    for n in (0, 1, 7, 40):
        df = _ledger(rng, n)
        out.append(df)
        if n:
            df = df.copy()
            df.loc[df.index[0], "original_index"] = pd.NA
            out.append(df)
            out.append(df.assign(Date=df["Date"].dt.tz_localize("Asia/Bangkok")))
            out.append(df.assign(flag=rng.random(n) < 0.5, cat=pd.Categorical(rng.choice(["a", "b"], n))))
            out.append(df.assign(Note=df["Note"].astype("string")))
            out.append(df.set_index("Date"))
            out.append(df.set_index(pd.Index([f"r{i}" for i in range(n)])))
            out.append(df.set_axis(pd.RangeIndex(n) * 3 + 5))
            out.append(df.iloc[::-1])
            out.append(df[["Quantity", "Price/Share"]])  # homogeneous float: pandas
            out.append(df[["Date"]])  # homogeneous datetime: pandas
            out.append(df[["Symbol", "Account"]])  # all object
            dup = df[["Symbol", "Quantity", "Symbol"]]  # duplicate columns: pandas
            out.append(dup)
            multi = df[["Symbol", "Quantity"]].copy()
            multi.columns = pd.MultiIndex.from_tuples([("a", "Symbol"), ("a", "Quantity")])
            out.append(multi)
            out.append(SubFrame(df))
            nums = df[["Symbol", "Quantity"]].copy()
            nums.columns = [0, 1]  # non-text labels
            out.append(nums)
    return out


KEYS = ["Date", "Type", "Symbol", "Quantity", "Price/Share", "Total Amount", "Commission", "Split Ratio",
        "Account", "Local Currency", "Note", "original_index", "user_id", "flag", "cat", "missing", 0, 1]  # fmt: skip


def _fingerprint(v):
    if isinstance(v, pd.Series):
        return ("series", tuple(map(_fingerprint, v.tolist())), tuple(v.index), str(v.dtype), v.name)
    nan = isinstance(v, float) and math.isnan(v)
    return (type(v).__name__, "nan" if nan else repr(v))


def _ask(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            out = ("ok", _fingerprint(fn()))
        except Exception as exc:  # noqa: BLE001 -- the exception is the answer
            out = ("raised", type(exc).__name__, str(exc))
    return out, [(w.category.__name__, str(w.message)) for w in caught]


@pytest.mark.parametrize("which", range(len(_frames())))
def test_rows_answer_as_iterrows(which):
    df = _frames()[which]
    expected = list(df.iterrows())
    got = list(investa_core.iterrows(df))
    assert len(expected) == len(got)
    for (k1, r1), (k2, r2) in zip(expected, got):
        assert type(k1) is type(k2) and repr(k1) == repr(k2)
        for key in KEYS:
            assert _ask(lambda r=r1: r[key]) == _ask(lambda r=r2: r[key]), (which, key)
            for default in ((), (None,), ("N/A",)):
                assert _ask(lambda r=r1: r.get(key, *default)) == _ask(lambda r=r2: r.get(key, *default)), (which, key)


def test_views_are_used_for_a_ledger():
    rows = list(investa_core.iterrows(_ledger(np.random.default_rng(1), 5)))
    assert all(type(r).__name__ == "RowView" for _, r in rows)
    rows = list(investa_core.iterrows(_ledger(np.random.default_rng(1), 5)[["Quantity"]]))
    assert all(isinstance(r, pd.Series) for _, r in rows)
