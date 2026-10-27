"""Phase 11b: market_db.MarketDatabase in Rust (rust/investa_core/src/mdb.rs)
against the original frozen in tests/reference/market_db.py.

Twin archives take the same random sequence of method calls, one through the
original class and one through the port. After every call the two must agree
on:
- the return value: frames bitwise with their index type, dtype, name and
  freq, columns and dtypes; dicts, tuples, dates;
- the exception, by class and message;
- every table's rows;
- every log record and every warning.
Timestamps stamped with `datetime.now()` are masked.

The frames carry what the cleaning and adjustment must handle:
- indexes: naive, tz-aware (UTC, New York), unsorted, with NaT, or not datetimes;
- cells: text, NaN, integers, infinite and 1e20 volumes, negative values;
- split ratios outside the accepted range, negative dividends.
The calls cover every method, with tuples, empty and missing symbols, invalid
bases and adjustments, FX rows of every shape, legacy tables without the
provenance column, a data_quality table to summarise, and integrity checks on
overlapping, disagreeing and adjudicated bars.
"""

import datetime as dt
import logging
import math
import os
import re
import sqlite3
import struct
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import market_db as new  # noqa: E402
from reference import market_db as ref  # noqa: E402

STAMP = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d+)?")
SYMBOLS = ["AAPL", "MSFT", "BRK-B", "THB=X", "NEW"]

ARCHIVE = """
CREATE TABLE IF NOT EXISTS corporate_action (
    symbol TEXT NOT NULL, date TEXT NOT NULL, kind TEXT NOT NULL,
    value REAL NOT NULL, currency TEXT, source TEXT NOT NULL,
    ingested_at TEXT NOT NULL, PRIMARY KEY (symbol, date, kind));
CREATE TABLE IF NOT EXISTS share_count (
    symbol TEXT NOT NULL, shares REAL NOT NULL, as_of TEXT NOT NULL,
    source TEXT NOT NULL, PRIMARY KEY (symbol));
CREATE TABLE IF NOT EXISTS fund_nav (
    fund_code TEXT NOT NULL, date TEXT NOT NULL, nav REAL NOT NULL,
    currency TEXT, source TEXT NOT NULL, PRIMARY KEY (fund_code, date));
CREATE TABLE IF NOT EXISTS data_quality (
    symbol TEXT NOT NULL, kind TEXT NOT NULL, severity TEXT NOT NULL,
    occurred_on TEXT, detail TEXT, detected_at TEXT NOT NULL,
    PRIMARY KEY (symbol, kind, occurred_on));
"""


def canon(x):
    if isinstance(x, pd.DataFrame):
        idx = x.index
        return (
            "df",
            list(map(str, x.columns)),
            list(map(str, x.dtypes)),
            type(idx).__name__,
            str(idx.dtype),
            idx.name,
            str(getattr(idx, "freq", None)),
            [canon(v) for v in idx.tolist()],
            [canon(v) for v in x.astype(object).to_numpy().ravel().tolist()],
        )
    if isinstance(x, np.ndarray):
        return ("nd", str(x.dtype), [canon(v) for v in x.tolist()])
    if isinstance(x, float):
        return ("f", type(x).__name__, "nan" if math.isnan(x) else struct.pack("<d", x))
    if isinstance(x, dict):
        return ("dict", [(canon(k), canon(v)) for k, v in x.items()])
    if isinstance(x, (list, tuple)):
        return (type(x).__name__, [canon(v) for v in x])
    if isinstance(x, str):
        return ("s", STAMP.sub("<now>", x))
    if type(x) is dt.datetime:
        return ("datetime", "<now>")  # a parsed sync stamp, microseconds apart
    return (type(x).__name__, repr(x))


# Deprecations raised inside third-party libraries, not by either side's code.
# They fire only when a process-wide cache misses, so which side sees them
# depends on test order, not on the port: pandas 2.3's Timedelta constructor
# under NumPy 2.5, reached through pandas-market-calendars when the NYSE
# schedule cache is first built.
LIBRARY_WARNINGS = [r"The 'generic' unit for NumPy timedelta is deprecated"]


def call(fn, caplog):
    caplog.clear()
    with caplog.at_level(logging.DEBUG), warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        for text in LIBRARY_WARNINGS:
            warnings.filterwarnings("ignore", message=text, category=DeprecationWarning)
        try:
            out = ("ok", canon(fn()))
        except Exception as exc:
            out = ("raised", type(exc).__name__, STAMP.sub("<now>", str(exc)))
    logs = [
        (r.levelname, STAMP.sub("<now>", r.getMessage()))
        for r in caplog.records
        if r.name == "root"
    ]
    return out, logs, [(w.category.__name__, str(w.message)) for w in warned]


def tables(db):
    c = sqlite3.connect(db.db_path)
    try:
        names = [
            r[0]
            for r in c.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            )
        ]
        return [
            (n, canon(c.execute(f'SELECT * FROM "{n}" ORDER BY rowid').fetchall()))
            for n in names
        ]
    finally:
        c.close()


def make_pair(tmp_path, tag, legacy=False):
    dbs = []
    for side, mod in (("ref", ref), ("new", new)):
        path = str(tmp_path / f"{tag}_{side}.db")
        db = mod.MarketDatabase(path)
        c = sqlite3.connect(path)
        c.executescript(ARCHIVE)
        c.execute(
            "ALTER TABLE sync_metadata ADD COLUMN price_basis TEXT NOT NULL DEFAULT 'split_adj'"
        )
        if legacy:  # a database from before the provenance migrations
            c.execute("ALTER TABLE daily_ohlcv DROP COLUMN source")
            c.execute("ALTER TABLE daily_fx DROP COLUMN source")
        c.commit()
        c.close()
        dbs.append(db)
    return dbs


def frame(rng, extra=()):
    n = int(rng.integers(0, 40))
    r = rng.random()
    days = pd.bdate_range("2023-01-02", periods=n)
    if r < 0.6:
        idx = pd.DatetimeIndex(days.values)
    elif r < 0.75:
        idx = (
            pd.DatetimeIndex(days.values)
            .tz_localize("UTC")
            .tz_convert("America/New_York")
        )
    elif r < 0.85:
        idx = pd.DatetimeIndex(
            list(days.values[: max(n - 1, 0)]) + [pd.NaT] * min(n, 1)
        )
    elif r < 0.92:
        idx = pd.Index([d.strftime("%Y-%m-%d") for d in days])
    else:
        idx = (
            pd.DatetimeIndex(days.values)[rng.permutation(n)]
            if n
            else pd.DatetimeIndex([])
        )
    cols = {}
    for c in ("Open", "High", "Low", "Close", "Adj Close", "Volume", "price", *extra):
        if rng.random() < 0.75:
            vals = rng.lognormal(4, 0.3, n)
            if c == "Volume":
                vals = np.round(vals * 1e4)
                if n and rng.random() < 0.1:
                    vals[0] = float(rng.choice([np.inf, 1e20, -5.0]))
            if c == "Stock Splits":
                vals = rng.choice([0.0, 0.0, 0.0, 2.0, 0.5, 1e-7, 1.0, np.nan], n)
            if c == "Dividends":
                vals = rng.choice([0.0, 0.0, 0.25, -1.0, np.nan], n)
            col = list(vals)
            if n and rng.random() < 0.1:
                col[int(rng.integers(0, n))] = "x"
            if n and rng.random() < 0.15:
                col[int(rng.integers(0, n))] = np.nan
            cols[c] = col
    return pd.DataFrame(cols, index=idx)


def symbols(rng):
    out = [str(s) for s in rng.choice(SYMBOLS, int(rng.integers(0, 4)))]
    return tuple(out) if rng.random() < 0.1 else out


@pytest.mark.parametrize("seed", range(10))
def test_random_sequences_match(tmp_path, caplog, seed):
    rng = np.random.default_rng(8400 + seed)
    a, b = make_pair(tmp_path, f"s{seed}", legacy=seed % 5 == 4)
    for step in range(70):
        sym = str(rng.choice(SYMBOLS))
        lo = dt.date(2023, 1, 1) + dt.timedelta(days=int(rng.integers(0, 40)))
        hi = lo + dt.timedelta(days=int(rng.integers(-5, 80)))
        adjust = str(rng.choice(["split", "split", "none", "total_return", "bogus"]))
        r = rng.random()
        if r < 0.2:
            f = frame(rng, extra=("Dividends", "Stock Splits"))
            src, force = str(rng.choice(["yahoo", "tiingo"])), bool(rng.random() < 0.2)
            ops = [lambda d: d.upsert_ohlcv(sym, f.copy(), "1d", src, force)]
        elif r < 0.27:
            f = frame(rng, extra=("Dividends", "Stock Splits"))
            ops = [lambda d: d.upsert_actions(sym, f.copy(), "yahoo")]
        elif r < 0.31:
            basis = str(rng.choice(["raw", "split_adj", "weird"]))
            ops = [lambda d: d.set_price_basis(sym, basis)]
        elif r < 0.45:
            ops = [lambda d: d.get_ohlcv(sym, lo, hi, "1d", adjust)]
        elif r < 0.55:
            syms = symbols(rng)
            ops = [lambda d: d.get_ohlcv_batch(syms, lo, hi, "1d", adjust)]
        elif r < 0.6:
            syms, kind = (
                symbols(rng),
                [None, "split", "dividend"][int(rng.integers(0, 3))],
            )
            ops = [
                lambda d: d.get_actions(syms, kind),
                lambda d: d.get_price_basis(syms),
            ]
        elif r < 0.66:
            f = frame(rng)
            ops = [
                lambda d: d.upsert_fx(sym, f.copy(), "1d", "yahoo"),
                lambda d: d.get_fx(sym, lo, hi),
            ]
        elif r < 0.71:
            rows = [
                (f"2023-01-{int(k):02d}", v)
                for k, v in zip(
                    rng.integers(1, 28, 6),
                    [None, "abc", -1.0, np.nan, np.float64(1.5), 2],
                )
            ]
            if rng.random() < 0.1:
                rows.append(("2023-02-01",))
            fill = bool(rng.random() < 0.4)
            ops = [lambda d: d.upsert_fx_rows(sym, list(rows), "1d", "ecb", fill)]
        elif r < 0.75:
            f = frame(rng)
            f.index = pd.DatetimeIndex(
                pd.date_range("2023-01-03 14:30", periods=len(f), freq="5min", tz="UTC")
            )
            ops = [
                lambda d: d.upsert_intraday(sym, f.copy(), "5m"),
                lambda d: d.get_intraday(
                    sym,
                    dt.datetime(2023, 1, 3, tzinfo=dt.timezone.utc),
                    dt.datetime(2023, 1, 4, tzinfo=dt.timezone.utc),
                    "5m",
                ),
            ]
        elif r < 0.8:
            rows = [
                ("2023-01-05", 10.5),
                ("", 3.0),
                ("2023-01-06", None),
                ("2023-01-07", "11"),
            ]
            ops = [
                lambda d: d.upsert_fund_nav("scbrmS50", list(rows)),
                lambda d: d.get_fund_nav("scbrmS50", lo, hi),
                lambda d: d.get_latest_fund_navs(),
                lambda d: d.get_fund_nav_coverage(),
            ]
        elif r < 0.85:
            counts = {s: v for s, v in zip(SYMBOLS, [1e9, 0, -3, "5e8", None])}
            age = [None, 30][int(rng.integers(0, 2))]
            as_of = [None, lo][int(rng.integers(0, 2))]
            syms = symbols(rng)
            ops = [
                lambda d: d.upsert_share_counts(counts, dt.date(2026, 10, 1)),
                lambda d: d.get_share_counts(syms, age),
                lambda d: d.get_latest_closes(syms, as_of),
            ]
        elif r < 0.9:
            syms = list(symbols(rng)) or ["AAPL"]
            ops = [
                lambda d: d.get_last_date(sym),
                lambda d: d.get_last_dates(syms),
                lambda d: d.get_first_dates(syms, "daily_fx"),
                lambda d: d.get_sync_metadata_batch(syms),
            ]
        elif r < 0.94:
            rows = [(s, k, sev, "2023-01-0" + str(i % 9 + 1), f"d{i}", "2026-01-01")
                    for i, (s, k, sev) in enumerate(zip(rng.choice(SYMBOLS, 5), rng.choice(["mixed", "unapplied"], 5), rng.choice(["high", "medium"], 5)))]  # fmt: skip
            for d in (a, b):
                c = sqlite3.connect(d.db_path)
                c.executemany(
                    "INSERT OR REPLACE INTO data_quality VALUES (?, ?, ?, ?, ?, ?)",
                    rows,
                )
                c.commit()
                c.close()
            want = [None, [], list(symbols(rng))][int(rng.integers(0, 3))]
            ops = [lambda d: d.get_data_quality(want)]
        else:
            f = frame(rng)
            ops = [lambda d: d.check_integrity(sym, f.copy())]
        for op in ops:
            ga, gb = call(lambda: op(a), caplog), call(lambda: op(b), caplog)
            assert ga == gb, (seed, step, ga, gb)
        assert tables(a) == tables(b), (seed, step)


@pytest.mark.parametrize("seed", range(4))
def test_helpers_match(seed, caplog):
    rng = np.random.default_rng(8500 + seed)
    for trial in range(60):
        n = int(rng.integers(0, 30))
        unit = str(rng.choice(["ns", "us", "s"]))
        bases = ["1965-06-30", "2023-01-02"] + (["1600-03-01"] if unit != "ns" else [])
        base_day = np.datetime64(str(rng.choice(bases)), "D")
        offsets = np.sort(rng.integers(0 if unit == "ns" else -300, 3000, n))
        days64 = (base_day + offsets).astype(f"datetime64[{unit}]")
        if unit == "s" and rng.random() < 0.2 and n:
            days64[-1] = np.datetime64(
                "10000-01-01", "s"
            )  # a year NumPy writes as 5 digits
        if rng.random() < 0.2 and n:
            days64[-1] = np.datetime64("NaT")
        idx = pd.DatetimeIndex(days64)
        if rng.random() < 0.2:
            idx = idx.tz_localize("UTC").tz_convert("Asia/Bangkok")
        if rng.random() < 0.15 and n:
            idx = idx[rng.permutation(n)]
        assert call(lambda: ref._iso_days(idx), caplog) == call(
            lambda: new._iso_days(idx), caplog
        ), trial
        days = call(lambda: ref._iso_days(idx), caplog)
        if days[0][0] != "ok":
            continue
        date_strs = ref._iso_days(idx)
        k = int(rng.integers(0, 6))
        actions = pd.DataFrame(
            {
                "symbol": ["X"] * k,
                "date": [str(d) for d in (base_day + rng.integers(-300, 3000, k))],
                "kind": list(rng.choice(["split", "dividend"], k)),
                "value": list(
                    rng.choice([2.0, 0.5, 1.0, 0.0, -1.0, 0.3, np.nan, 1e-9], k)
                ),
            }
        )
        if rng.random() < 0.1:
            actions = actions.drop(columns=["kind"])
        closes = rng.lognormal(3, 0.5, len(date_strs))
        if len(closes) and rng.random() < 0.2:
            closes[0] = np.nan
        for fn, args in (
            ("_future_split_factors", (list(date_strs), actions.copy())),
            ("_dividend_factors", (list(date_strs), closes.copy(), actions.copy())),
        ):
            got_ref = call(lambda: getattr(ref.MarketDatabase, fn)(*args), caplog)
            got_new = call(lambda: getattr(new.MarketDatabase, fn)(*args), caplog)
            assert got_ref == got_new, (seed, trial, fn, got_ref, got_new)
