"""Phase 15h: the statement and ratio glue of financial_ratios in Rust
(rust/investa_core/src/fr.rs) against the original frozen in
tests/reference/financial_ratios_p15.py.

Statement lookups take random frames with aliases, filed zeros, duplicate
labels, text cells and missing columns; free cash flow takes every OCF/capex
sign; trailing twelve months take quarterly frames with gaps, typo'd ends,
share rows and too few columns. The point-in-time price takes frames, series
(whose index the original rewrites in place), dicts, aware indexes and dates,
stale and non-positive quotes. The ratio history runs both paths — the
resolved lookups and the cell-by-cell original, which is also run directly on
numeric frames — and the multiples, current ratios, WACC and price-history
loader take random profiles and a fake provider. Results, logs, warnings and
the inputs left behind agree.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import financial_ratios as new  # noqa: E402
from reference import financial_ratios_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402
from test_rust_ratios_parity import LABELS, _case  # noqa: E402


def _cp(x):
    return x.copy() if isinstance(x, (pd.DataFrame, pd.Series)) else (dict(x) if isinstance(x, dict) else x)


def _both(caplog, name, *args, **kwargs):
    """Both modules on copies of the inputs; results, logs and inputs left behind agree."""
    outs = []
    for mod in (ref, new):
        a = [_cp(x) for x in args]
        k = {key: _cp(v) for key, v in kwargs.items()}
        got = call(lambda: getattr(mod, name)(*a, **k), caplog)
        outs.append((got, [canon(x) for x in a], {key: canon(v) for key, v in k.items()}))
    assert outs[0] == outs[1], name
    return outs[0][0]


def _frames(rng):
    for _ in range(40):
        fin, bs, cf, ppy, prices = _case(rng)
        yield fin, bs, cf, ppy, prices


@pytest.mark.parametrize("seed", range(4))
def test_statement_lookups(caplog, seed):
    rng = np.random.default_rng(18000 + seed)
    for fin, bs, cf, _ppy, _prices in _frames(rng):
        for df in (fin, bs, cf):
            if df is None or df.empty:
                cols = [None, "2020-12-31"]
            else:
                cols = list(df.columns[:2]) + ["missing"]
            if df is not None and not df.empty and rng.random() < 0.2:
                df = pd.concat([df, df.iloc[:1]])  # a duplicated label: .loc gives a Series
            if df is not None and not df.empty and rng.random() < 0.2:
                df = df.astype(object)
                df.iloc[0, 0] = "n/a"
            for col in cols:
                for item in rng.choice(list(new.STATEMENT_ALIASES) + LABELS[:5] + ["Nope"], 6):
                    _both(caplog, "_get_statement_value", df, str(item), col)
                _both(caplog, "_extract_fcf_from_statement", df, col)
    _both(caplog, "_get_statement_value", None, "Total Revenue", "x")
    _both(caplog, "_get_statement_value", pd.DataFrame(), "Total Revenue", "x")
    for capex in (-5.0, 5.0, 0.0, None):
        cf = pd.DataFrame({"c": [10.0, capex]}, index=["Operating Cash Flow", "Capital Expenditure"])
        _both(caplog, "_extract_fcf_from_statement", cf, "c")
    for label in ["Diluted Average Shares", "Share Issued", "Revenue", 5, None, ("Shares",)]:
        _both(caplog, "_is_level_row", label)


def test_windows_and_ttm(caplog):
    t = pd.Timestamp
    for newest, oldest in [(t("2021-12-31"), t("2021-03-31")), (t("2021-12-31"), t("2021-06-30")), (pd.NaT, t("2021-03-31")), (t("2021-12-31"), None)]:
        for q in (4, 3, 2, 1, np.int64(4)):
            _both(caplog, "_is_consecutive_window", newest, oldest, q)
    rng = np.random.default_rng(18100)
    for trial in range(120):
        n = int(rng.integers(0, 14))
        ends = list(pd.date_range("2018-03-31", periods=n, freq="QE")[::-1])
        if ends and rng.random() < 0.3:
            del ends[int(rng.integers(0, len(ends)))]  # a missing quarter
        if ends and rng.random() < 0.2:
            ends[0] = ends[0] + pd.Timedelta(days=-1)  # a typo'd end
        cols = [e.strftime("%Y-%m-%d") for e in ends] if rng.random() < 0.3 else ends
        if cols and rng.random() < 0.1:
            cols[-1] = "bad"
        rows = ["Total Revenue", "Net Income", "Diluted Average Shares", "Share Issued", "Operating Cash Flow"]
        data = np.where(rng.random((len(rows), len(cols))) < 0.1, np.nan, rng.normal(1e8, 3e7, (len(rows), len(cols))))
        df = pd.DataFrame(data, index=rows, columns=cols)
        for mq in (4, 2, 5, 1):
            _both(caplog, "to_trailing_twelve_months", df, mq)
    _both(caplog, "to_trailing_twelve_months", None)
    _both(caplog, "to_trailing_twelve_months", pd.DataFrame())


def test_price_on_or_before(caplog):
    rng = np.random.default_rng(18200)
    idx = pd.bdate_range("2019-01-01", "2021-12-31")
    base = pd.Series(np.cumprod(1 + rng.normal(0, 0.01, len(idx))) * 50, index=idx)
    shapes = [
        base,
        base.to_frame("Close"),
        base.to_frame("Adj Close").assign(Close=base * 2),
        base.to_frame("Weird"),
        pd.DataFrame(),
        base.iloc[::40],
        base.tz_localize("America/New_York"),
        pd.Series(base.values, index=[d.strftime("%Y-%m-%d") for d in idx]),
        pd.Series([1.0, -2.0, np.nan, np.inf], index=pd.to_datetime(["2020-01-01", "2020-06-01", "2020-07-01", "2020-08-01"])),
        {"2020-01-01": 10.0, "2020-03-01": 12.0},
        {},
        [1, 2, 3],
        pd.Series([], dtype=float),
    ]
    whens = [pd.Timestamp("2020-06-30"), "2020-03-15", "2018-01-01", pd.Timestamp("2021-06-30", tz="UTC"), None, "garbage", [1, 2], pd.NaT, "2020-08-02"]
    for prices in shapes:
        for when in whens:
            _both(caplog, "_price_on_or_before", prices, when)


def test_scalars(caplog):
    for v in (None, 1, 2.5, True, np.float64(3), np.int32(2), "4", [1], 10**400):
        _both(caplog, "_opt_number", v)
    for args in [(100.0, 0.05, 0.1, 10, 0.02), (100, np.float64(0.05), 0.1, 10, 0.02), (100.0, 0.05, 0.1, np.int64(5), 0.02),
                 (100.0, 0.05, 0.1, 10.0, 0.02), (-50.0, -0.02, 0.09, 0, 0.0), ("1", 0.05, 0.1, 10, 0.02)]:
        _both(caplog, "_scalar_projection", *args)


@pytest.mark.parametrize("seed", range(4))
def test_ratio_history_both_paths(caplog, seed):
    rng = np.random.default_rng(18300 + seed)
    for trial, (fin, bs, cf, ppy, prices) in enumerate(_frames(rng)):
        if trial % 3 == 1:
            fin = fin.astype(object)
            fin.iloc[0, 0] = "n/a"
        elif trial % 3 == 2:
            bs = pd.concat([bs, bs.iloc[:1]])
        _both(caplog, "calculate_key_ratios_timeseries", fin, bs, cf, ppy, prices)
        _both(caplog, "_key_ratios_python", fin, bs, cf, ppy, prices)
        _both(caplog, "historical_mean_multiples", fin, bs, cf, prices)
        for df in (fin, bs, cf):
            _both(caplog, "_numeric_statement", df)
    for args in [(None, None), (pd.DataFrame(), pd.DataFrame()), (pd.DataFrame({"x": [1.0]}, index=["Total Revenue"]), pd.DataFrame({"y": [1.0]}, index=["Total Assets"]))]:
        _both(caplog, "calculate_key_ratios_timeseries", *args)
        _both(caplog, "_key_ratios_python", *args)
        _both(caplog, "historical_mean_multiples", *args, None, pd.Series([1.0]))


def test_statement_lookup_class():
    df = pd.DataFrame({"a": [1.0, None, 3], "b": [np.nan, 2, 0]}, index=["Revenue", "Total Revenue", "Net Income"]).astype(object)
    a, b = ref._StatementLookup(df), new._StatementLookup(df)
    for item in ("Total Revenue", "Net Income", "Nope", "Diluted EPS"):
        assert canon(a.row(item)) == canon(b.row(item))
        for col in ("a", "b"):
            assert canon(a.get(item, col)) == canon(b.get(item, col))
    with pytest.raises(KeyError):
        b.get("Net Income", "c")
    assert canon(new._period_columns(df)) == canon(ref._period_columns(df))


class FakeMdp:
    def __init__(self, mode):
        self.mode = mode
        self.calls = []

    def get_historical_data(self, symbols, start, end):
        self.calls.append((tuple(symbols), str(start), str(end)))
        if self.mode == "raise":
            raise ConnectionError("offline")
        if self.mode == "missing":
            return {}, None
        frame = pd.DataFrame({"Close": [1.0, 2.0]}) if self.mode == "ok" else pd.DataFrame()
        return {symbols[0]: frame}, {}


def test_profiles(caplog):
    for mode in ("ok", "raise", "missing", "empty"):
        for years in (15, 3):
            outs = []
            for mod in (ref, new):
                mdp = FakeMdp(mode)
                outs.append((call(lambda: mod.load_price_history_for_multiples(mdp, "AAPL", years), caplog), mdp.calls))
            assert outs[0] == outs[1], mode

    rng = np.random.default_rng(18400)
    keys = ["currentPrice", "regularMarketPrice", "marketCap", "trailingPE", "forwardPE", "dividendYield",
            "totalRevenue", "bookValue", "enterpriseToEbitda", "beta", "totalDebt", "symbol"]  # fmt: skip
    choices = [None, 0, 0.0, 12.5, -3.0, 5e9, np.float64(1.2), float("nan"), "x", 7]
    for trial in range(400):
        info = {k: choices[int(rng.integers(0, len(choices)))] for k in keys if rng.random() < 0.7}
        if trial % 50 == 0:
            info["beta"] = [1, 2]
        fin, bs, cf, _ppy, _prices = _case(rng)
        bs_arg = [None, pd.DataFrame(), bs][int(rng.integers(0, 3))]
        fin_arg = [None, pd.DataFrame(), fin][int(rng.integers(0, 3))]
        _both(caplog, "calculate_current_valuation_ratios", info if trial % 9 else {}, fin_arg, bs_arg)
        _both(caplog, "calculate_wacc", info, fin_arg, bs_arg)
        _both(caplog, "calculate_wacc", info, fin_arg, bs_arg, 0.03, 0.08, 0.25)
