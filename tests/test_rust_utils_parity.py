"""Phase 16a: risk_metrics and the first block of finutils in Rust
(rust/investa_core/src/rm.rs, fu.rs) against the originals frozen in
tests/reference/risk_metrics_p16.py and finutils_p16.py.

Risk metrics take value series with NaN, zeros, negatives, constants and
spikes on calendar-daily, trading-daily, monthly and non-date indexes, with
and without benchmarks. File hashes take plain files, databases with and
without WAL and SHM sidecars, missing paths, directories and unreadable files.
The dividend, yield, cash and external-flow helpers take profiles and symbols
of every type; NPV and IRR take dates (date, datetime, Timestamp, aware, mixed,
unsorted) and flows (int, float, NumPy, bool, text, NaN), long failing series
included; the column readers take numeric, object and missing columns.
Results, logs, warnings and the inputs left behind agree.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import finutils as new_fu  # noqa: E402
import risk_metrics as new_rm  # noqa: E402
from reference import finutils_p16 as ref_fu  # noqa: E402
from reference import risk_metrics_p16 as ref_rm  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402


def _cp(x):
    if isinstance(x, (pd.DataFrame, pd.Series, pd.Index)):
        return x.copy()
    if isinstance(x, (dict, list)):
        return type(x)(x)
    return x


def _pair(ref, new, caplog, name, *args):
    outs = []
    for mod in (ref, new):
        a = [_cp(x) for x in args]
        outs.append((call(lambda: getattr(mod, name)(*a), caplog), [canon(x) for x in a]))
    assert outs[0] == outs[1], name
    return outs[0][0]


def rm_both(caplog, name, *args):
    return _pair(ref_rm, new_rm, caplog, name, *args)


def fu_both(caplog, name, *args):
    return _pair(ref_fu, new_fu, caplog, name, *args)


def _series(rng, n):
    kind = int(rng.integers(0, 4))
    if kind == 0:
        idx = pd.date_range("2020-01-01", periods=n, freq="D")
    elif kind == 1:
        idx = pd.bdate_range("2020-01-01", periods=n)
    elif kind == 2:
        idx = pd.date_range("2015-01-31", periods=n, freq="ME")
    else:
        idx = pd.RangeIndex(n)
    v = 100 * np.cumprod(1 + rng.normal(0.0005, 0.02, n))
    r = rng.random()
    if r < 0.1 and n:
        v[int(rng.integers(0, n))] = np.nan
    elif r < 0.15 and n:
        v[int(rng.integers(0, n))] = 0.0
    elif r < 0.2:
        v = np.full(n, 50.0)
    elif r < 0.25 and n > 3:
        v[n // 2] *= 5  # a spike the clip catches
    return pd.Series(v, index=idx, name="v")


@pytest.mark.parametrize("seed", range(5))
def test_risk_metrics(caplog, seed):
    rng = np.random.default_rng(22000 + seed)
    for trial in range(60):
        n = int(rng.choice([0, 1, 2, 4, 6, 25, 300]))
        s = _series(rng, n)
        b = _series(rng, n) if rng.random() < 0.7 else pd.Series(dtype=float)
        r = s.pct_change(fill_method=None).dropna()
        for name in ("calculate_drawdown_series", "calculate_max_drawdown", "_returns", "_values"):
            rm_both(caplog, name, s)
        for ppy in (252, 365, 12.0):
            rm_both(caplog, "calculate_volatility", r, ppy)
            rm_both(caplog, "calculate_sharpe_ratio", r, 0.02, ppy)
            rm_both(caplog, "calculate_sortino_ratio", r, 0.03, ppy, 0.001)
        rm_both(caplog, "calculate_beta", r, b.pct_change(fill_method=None).dropna())
        rm_both(caplog, "calculate_alpha", 0.1, 0.08, 1.2, 0.02)
        rm_both(caplog, "calculate_alpha", np.float64(0.1), 0.08, np.float64(0.9))
        for ppy in (None, 252):
            rm_both(caplog, "calculate_all_risk_metrics", s, 0.02, ppy, b)
            rm_both(caplog, "calculate_all_risk_metrics", s, 0.02, ppy, None)
        rm_both(caplog, "calculate_benchmark_scoreboard", s, {"SPY": b, "QQQ": _series(rng, n), "none": None, "empty": pd.Series(dtype=float)})
        rm_both(caplog, "calculate_benchmark_scoreboard", None, {})
        fu_both(caplog, "infer_periods_per_year", s.index)
        fu_both(caplog, "infer_periods_per_year", s.index, 12)


def test_file_hash(caplog, tmp_path):
    plain = tmp_path / "plain.csv"
    plain.write_bytes(os.urandom(20000))
    db = tmp_path / "book.DB"
    db.write_bytes(b"main")
    (tmp_path / "book.DB-wal").write_bytes(b"wal")
    (tmp_path / "book.DB-shm").write_bytes(b"shm")
    lonely = tmp_path / "lonely.sqlite3"
    lonely.write_bytes(b"x" * 9000)
    locked = tmp_path / "locked.db"
    locked.write_bytes(b"secret")
    os.chmod(locked, 0)
    try:
        for p in (plain, db, lonely, tmp_path / "missing.db", tmp_path, locked):
            fu_both(caplog, "_get_file_hash", str(p))
    finally:
        os.chmod(locked, 0o600)
    fu_both(caplog, "_get_file_hash", None)  # an int would be opened as a file descriptor


def test_dividends_and_symbols(caplog):
    rng = np.random.default_rng(22100)
    vals = [None, 0, 0.0, 0.25, 1.0, 4.0, 0.06, "1.5", "x", float("nan"), -1.0, np.float64(0.5), 120.0]
    keys = ["lastDividendValue", "dividendRate", "trailingAnnualDividendRate", "yield", "currentPrice",
            "regularMarketPrice", "trailingAnnualDividendYield", "dividendYield"]  # fmt: skip
    for trial in range(800):
        info = {k: vals[int(rng.integers(0, len(vals)))] for k in keys if rng.random() < 0.6}
        for name in ("get_dividend_details", "calculate_indicated_dividend", "robust_dividend_yield"):
            fu_both(caplog, name, info)
    for name in ("get_dividend_details", "robust_dividend_yield"):
        fu_both(caplog, name, {})
        fu_both(caplog, name, None)
    for code in ("usd", "THB", "", None, 5, "ZZZ", "eur"):
        fu_both(caplog, "get_currency_symbol_from_code", code)
    symbols = ["$CASH", "$cash", "Cash ($)", "cash (thb)", "AAPL", "", None, 5, "$CASHX", " $CASH", "İ$CASH"]
    types = ["Deposit", " withdrawal ", "DEPOSIT", "Buy", "", None, 3, "deposit\n"]
    for s in symbols:
        fu_both(caplog, "is_cash_symbol", s)
        for t in types:
            fu_both(caplog, "is_external_flow_row", s, t)
            fu_both(caplog, "is_external_flow_row", s, t, "note")
    df = pd.DataFrame({"Symbol": symbols * 3, "Type": (types * 5)[: len(symbols) * 3]})
    fu_both(caplog, "compute_external_flow_mask", df)
    fu_both(caplog, "compute_external_flow_mask", df.iloc[:0])


def _dates(rng, n):
    kind = int(rng.integers(0, 6))
    base = dt.date(2015, 1, 1)
    days = sorted(int(x) for x in rng.integers(0, 3000, n))
    if rng.random() < 0.1 and n > 2:
        days[0], days[-1] = days[-1], days[0]  # unsorted
    if kind == 0:
        return [base + dt.timedelta(days=d) for d in days]
    if kind == 1:
        return [dt.datetime(2015, 1, 1, 9, 30) + dt.timedelta(days=d, hours=int(rng.integers(0, 12))) for d in days]
    if kind == 2:
        return [pd.Timestamp("2015-01-01") + pd.Timedelta(days=d) for d in days]
    if kind == 3:
        return [pd.Timestamp("2015-01-01", tz="UTC") + pd.Timedelta(days=d) for d in days]
    if kind == 4 and n > 1:
        return [base + dt.timedelta(days=d) for d in days[:-1]] + [dt.datetime(2020, 1, 1)]
    return [base] * n


def _flows(rng, n):
    flows = []
    for i in range(n):
        r = rng.random()
        v = float(rng.normal(-1000 if i == 0 else 200, 500))
        flows.append([v, int(v), np.float64(v), True, float("nan"), "12"][int(rng.choice(6, p=[0.8, 0.08, 0.06, 0.02, 0.02, 0.02]))] if r < 1 else v)
    return flows


@pytest.mark.parametrize("seed", range(5))
def test_npv_and_irr(caplog, seed):
    rng = np.random.default_rng(22200 + seed)
    for trial in range(80):
        n = int(rng.choice([0, 1, 2, 5, 30, 130]))
        dates, flows = _dates(rng, n), _flows(rng, n)
        fu_both(caplog, "_irr_date_ticks", dates)
        fu_both(caplog, "calculate_irr", dates, flows)
        fu_both(caplog, "calculate_irr", dates, flows[:-1])
        for rate in (0.1, -0.5, -1.0, 0.0, float("nan"), "x", 2, np.float64(0.07), True):
            fu_both(caplog, "calculate_npv", rate, dates, flows)
        fu_both(caplog, "calculate_npv", 0.1, dates, flows + [1.0])
    # A solver failure over many flows, and an extreme IRR
    d = [dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(150)]
    fu_both(caplog, "calculate_irr", d, [1.0] * 150)
    fu_both(caplog, "calculate_irr", d[:3], [-1.0, 0.0, 1e9])
    fu_both(caplog, "calculate_irr", d[:3], [-1e-12, 5e-13, 2e-12])


def test_column_readers(caplog):
    rng = np.random.default_rng(22300)
    for trial in range(60):
        n = int(rng.integers(0, 20))
        cols = {
            "Commission": rng.choice([np.nan, 1.5, 0.0], n),
            "Qty": rng.integers(-5, 5, n),
            "Text": rng.choice(np.array(["1.5", "x", "", None, "1e3", 2.0], dtype=object), n),
            "Flag": rng.random(n) < 0.5,
        }
        df = pd.DataFrame(cols)
        if trial % 4 == 0:
            df["Commission"] = df["Commission"].astype(object)
        for name in ("Commission", "Qty", "Text", "Flag", "Missing"):
            fu_both(caplog, "_numeric_column", df, name)
            fu_both(caplog, "_numeric_column", df, name, True)
        fu_both(caplog, "_commission_column", df)
        fu_both(caplog, "_commission_column", df.drop(columns=["Commission"]))
    for values in (None, ["a", 1, None]):
        for i in (0, 2):
            fu_both(caplog, "_cell_text", values, i)
            fu_both(caplog, "_cell_text", values, i, "dflt")
