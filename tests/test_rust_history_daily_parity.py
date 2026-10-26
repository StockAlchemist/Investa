"""Phase 14d: portfolio_history's daily results and accumulation in Rust
(rust/investa_core/src/ph_daily.rs) against the originals frozen in
tests/reference/portfolio_history.py.

- Accumulation and resampling over random daily frames (UTC and naive
  indexes) at D, 1d, W, M, ME and an intraday interval, with benchmarks that
  are missing, NaN-ridden or carry duplicate dates, and requested windows
  inside, across and outside the data, so the t-1 / t0 / forward-search
  normalisation and its too-small-divisor guard all run.
- The daily results: the chronological pass with and without the L1
  holdings, daily and intraday (past dates only, so the "now" clip is not a
  race between the two runs), with an account filter, splits to re-adjust,
  and flows; the single-process worker pass; and the feather cache written by
  one run and read back by the next, each side in its own directory, with a
  stale key and a CSV mtime check. `logging.basicConfig(force=True)` in the
  worker would remove the test's log capture, so it is stubbed out.
"""

import logging
import os
import shutil
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import market_db  # noqa: E402
import portfolio_history as new  # noqa: E402
from reference import portfolio_history as ref  # noqa: E402
from test_rust_history_helpers_parity import FakeFunds  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

TIMING = ("Pool finished in", "1D Graph Clip")


def _masked(out, *dirs):
    result, logs, warned = out
    clean = []
    for lv, msg in logs:
        for d in dirs:
            msg = msg.replace(str(d), "<dir>")
        if any(t in msg for t in TIMING):
            msg = msg.split(":")[0]
        clean.append((lv, msg))
    return result, clean, warned


def _daily_frame(rng):
    n = int(rng.integers(0, 120))
    idx = pd.date_range("2023-01-02", periods=n, freq="D", tz="UTC" if rng.random() < 0.8 else None)
    value = np.cumsum(rng.normal(0, 50, n)) + 10000
    df = pd.DataFrame(
        {
            "value": value,
            "net_flow": rng.choice([0.0, 0.0, 500.0, -300.0], n),
            "daily_gain": rng.normal(0, 40, n),
            "daily_return": np.where(rng.random(n) < 0.05, np.nan, rng.normal(0, 0.01, n)),
            "SPY Price": np.where(rng.random(n) < 0.1, np.nan, 400 + np.cumsum(rng.normal(0, 2, n))),
        },
        index=idx,
    )
    if rng.random() < 0.2:
        df["QQQ Price"] = 0.0  # an all-zero accumulation skips normalisation
    if rng.random() < 0.1 and n > 3:
        df = pd.concat([df, df.iloc[:2]])  # duplicate dates
    if rng.random() < 0.05:
        df = df.drop(columns=["daily_return"])
    if rng.random() < 0.1:
        df["daily_return"] = 1e-9 - 1.0  # a near-zero accumulated gain
    return df


@pytest.mark.parametrize("seed", range(8))
def test_accumulated_gains_match(caplog, seed):
    rng = np.random.default_rng(15200 + seed)
    caplog.set_level(logging.DEBUG)
    for trial in range(40):
        df = _daily_frame(rng)
        interval = str(rng.choice(["D", "1d", "W", "M", "ME", "1h"]))
        # Callers always pass both dates; None (which fails the slice) is rare.
        start = [date(2023, 1, 2), date(2023, 2, 15), date(2023, 1, 20), date(2022, 6, 1), date(2025, 1, 1), None][
            int(rng.choice(6, p=[0.3, 0.3, 0.2, 0.08, 0.06, 0.06]))
        ]
        end = [date(2023, 4, 30), date(2023, 3, 1), None][int(rng.choice(3, p=[0.6, 0.34, 0.06]))]
        benchmarks = [["SPY"], ["SPY", "QQQ"], [], ["DIA"]][int(rng.integers(0, 4))]
        outs = []
        for mod in (ref, new):
            a = df.copy()
            out = call(lambda: mod._calculate_accumulated_gains_and_resample(a, benchmarks, interval, start, end), caplog)
            outs.append((out, canon(a)))
        assert outs[0] == outs[1], (seed, trial, interval)


ACCOUNTS = ["IBKR", "SCBS", "KIM ENG"]
SYMBOLS = ["AAPL", "MSFT", "PTT.BK", "$CASH"]
CCY = {"AAPL": "USD", "MSFT": "USD", "PTT.BK": "THB", "$CASH": "USD"}


def _inputs(rng):
    rows = []
    for i in range(int(rng.integers(3, 40))):
        sym = str(rng.choice(SYMBOLS))
        typ = str(rng.choice(["buy", "buy", "sell", "dividend", "transfer", "split"])) if sym != "$CASH" else str(rng.choice(["deposit", "withdrawal"]))
        rows.append(
            {
                "Date": pd.Timestamp("2024-01-02") + pd.Timedelta(days=int(rng.integers(0, 150))),
                "Symbol": sym,
                "Account": str(rng.choice(ACCOUNTS)),
                "To Account": str(rng.choice(ACCOUNTS)) if typ == "transfer" else None,
                "Type": typ,
                "Quantity": float(rng.choice([2.0, 10.0, 50.0])) if typ != "split" else float(rng.choice([2.0, 4.0])),
                "Price/Share": float(rng.choice([0.0, 50.0, 120.0])),
                "Commission": 1.0,
                "Total Amount": float(rng.choice([0.0, 500.0])),
                "Split Ratio": float(rng.choice([2.0, 0.0, np.nan])) if typ == "split" else 0.0,
                "Local Currency": CCY[sym],
                "Note": None,
                "original_index": i,
            }
        )  # fmt: skip
    txs = pd.DataFrame(rows).sort_values("Date", kind="stable").reset_index(drop=True)
    days = pd.date_range("2023-12-01", "2024-07-01", freq="B", tz="UTC")
    prices = {s: pd.DataFrame({"price": 100 + np.cumsum(rng.normal(0, 1, len(days)))}, index=days) for s in ("AAPL", "MSFT", "PTT.BK")}
    adjusted = dict(prices)
    adjusted["SPY"] = pd.DataFrame({"price": np.linspace(400, 450, len(days))}, index=days)
    fx = {"THB=X": pd.DataFrame({"price": 35 + rng.normal(0, 0.1, len(days))}, index=days)}
    return txs, prices, adjusted, fx


@pytest.mark.parametrize("seed", range(6))
def test_daily_results_match(caplog, monkeypatch, tmp_path, seed):
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(market_db, "MarketDatabase", FakeFunds)
    FakeFunds.mode = "ok"
    rng = np.random.default_rng(15300 + seed)
    for trial in range(8):
        txs, prices, adjusted, fx = _inputs(rng)
        maps = new.generate_mappings(txs)
        s2i, i2s, a2i, i2a, t2i, c2i, i2c = maps
        start, end = date(2024, 3, 1), date(2024, 6, 14)
        intraday = rng.random() < 0.25
        interval = "1h" if intraday else "1d"
        if intraday:
            start, end = date(2024, 6, 10), date(2024, 6, 14)
        included = [] if rng.random() < 0.6 else ["IBKR", "scbs"]
        method = "numba_chrono" if rng.random() < 0.75 else "numba"
        l1 = None
        if rng.random() < 0.5:
            l1 = new._get_or_calculate_all_daily_holdings(txs, txs["Date"].min().date(), end, s2i, a2i, t2i, None)
        use_cache = rng.random() < 0.4
        csv = tmp_path / f"{trial}.csv"
        csv.write_text("x")
        outs = []
        for side, mod in (("a", ref), ("b", new)):
            d = tmp_path / f"{trial}{side}"
            d.mkdir(exist_ok=True)
            cache_file = str(d / "daily.feather")
            runs = []
            for repeat in range(2 if use_cache else 1):  # write, then read back
                key = "KEY"

                def run():
                    return mod._load_or_calculate_daily_results(
                        use_cache, cache_file, key, interval, None, str(csv) if trial % 2 else None,
                        start, end, txs.copy(), prices, adjusted, fx, "USD",
                        {"AAPL": "AAPL", "MSFT": "MSFT", "PTT.BK": "PTT.BK"}, {"SCBS": "THB"}, "USD",
                        s2i, i2s, a2i, i2a, {"AAPL": {"price": 1.0}}, t2i, c2i, i2c,
                        None if l1 is None else l1[0], None if l1 is None else l1[1],
                        None if l1 is None else l1[2], None if l1 is None else txs["Date"].min().date(),
                        included, num_processes=1, filter_desc="Test", calc_method=method,
                    )  # fmt: skip

                out = _masked(call(run, caplog), d)
                runs.append(out)
            files = sorted(p.name for p in d.iterdir())
            outs.append((runs, files))
        assert outs[0] == outs[1], (seed, trial, method, interval)
        shutil.rmtree(tmp_path / f"{trial}a")
        shutil.rmtree(tmp_path / f"{trial}b")


def test_daily_cache_validation_paths_match(caplog, monkeypatch, tmp_path):
    """A stale key, a CSV newer than the cache, a missing CSV, corrupt metadata."""
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(market_db, "MarketDatabase", FakeFunds)
    rng = np.random.default_rng(15400)
    txs, prices, adjusted, fx = _inputs(rng)
    s2i, i2s, a2i, i2a, t2i, c2i, i2c = new.generate_mappings(txs)
    for case in ("stale_key", "csv_newer", "csv_missing", "corrupt_meta", "no_csv_path"):
        outs = []
        for side, mod in (("a", ref), ("b", new)):
            d = tmp_path / f"{case}{side}"
            d.mkdir()
            csv = d / "tx.csv"
            csv.write_text("x")
            cache_file = str(d / "daily.feather")
            args = lambda key, path: (  # noqa: E731
                True, cache_file, key, "1d", None, path, date(2024, 3, 1), date(2024, 6, 14), txs.copy(),
                prices, adjusted, fx, "USD", {"AAPL": "AAPL", "MSFT": "MSFT", "PTT.BK": "PTT.BK"}, {}, "USD",
                s2i, i2s, a2i, i2a, None, t2i, c2i, i2c, None, None, None, None, [],
            )  # fmt: skip
            mod._load_or_calculate_daily_results(*args("KEY", str(csv)), num_processes=1, calc_method="numba_chrono")
            meta = d / "daily.meta.json"
            if case == "csv_newer":
                os.utime(csv, (os.path.getmtime(meta) + 100,) * 2)
            if case == "csv_missing":
                csv.unlink()
            if case == "corrupt_meta":
                meta.write_text("{not json")
            key = "OTHER" if case == "stale_key" else "KEY"
            path = None if case == "no_csv_path" else str(csv)
            out = call(lambda: mod._load_or_calculate_daily_results(*args(key, path), num_processes=1, calc_method="numba_chrono"), caplog)
            outs.append(_masked(out, d))
        assert outs[0] == outs[1], case
