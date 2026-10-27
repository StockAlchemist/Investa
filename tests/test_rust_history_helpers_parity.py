"""Phase 14b: portfolio_history's helpers in Rust (rust/investa_core/src/
ph_small.rs) against the originals frozen in tests/reference/portfolio_history.py.

- Price unadjustment over frames with price, Close, Adj Close or no usable
  column, empty frames, naive and UTC date indexes, and splits dated by
  date, datetime, text or nothing, with good, zero, negative, text and
  missing ratios (the warnings set it fills is compared too).
- The per-date worker with the valuation and flow functions stubbed, in
  both comparison modes, failing and not; its `logging.basicConfig(force=True)`
  would remove the test's log capture, so it is recorded and compared.
- Manual prices, published NAV series (through a fake fund store that
  raises, answers None, or answers frames with and without prices), the
  price grid, the L1-cache pruning over a real directory, the intraday
  symbol window and the id mappings.
"""

import logging
import os
import sys
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import market_db  # noqa: E402
import portfolio_history as new  # noqa: E402
from reference import portfolio_history as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402


def both(fn, caplog, copy_args=None):
    """Run `fn(mod)` for the original and the port; the outcome and logs."""
    out = []
    for mod in (ref, new):
        out.append(call(lambda: fn(mod), caplog))
    return out


def _price_frame(rng):
    days = pd.date_range("2019-01-01", periods=int(rng.integers(0, 40)), freq="7D")
    if rng.random() < 0.3:
        days = days.tz_localize("UTC")
    col = str(rng.choice(["price", "Close", "Adj Close", "Volume"]))
    df = pd.DataFrame({col: rng.uniform(10, 200, len(days))}, index=days)
    if rng.random() < 0.2:
        df = df.iloc[::-1]
    return df


def _split(rng, kind):
    # One kind per symbol mostly: sorted() raises on mixed kinds (in both).
    k = kind if rng.random() < 0.9 else int(rng.integers(0, 5))
    day = int(rng.integers(1, 28))
    when = [date(2019, 3, day), datetime(2019, 5, day, 9), f"2019-04-{day:02d}", None, pd.Timestamp(2019, 6, day)][k]
    ratio = [2.0, 4, 0.0, -1.0, "x", "3"][int(rng.integers(0, 6))]
    split = {"Date": when, "Split Ratio": ratio}
    if rng.random() < 0.1:
        split.pop("Split Ratio")
    if rng.random() < 0.1:
        split.pop("Date")
    return split


@pytest.mark.parametrize("seed", range(6))
def test_unadjust_prices_match(caplog, seed):
    rng = np.random.default_rng(14300 + seed)
    caplog.set_level(logging.DEBUG)
    for trial in range(40):
        syms = ["AAPL", "MSFT", "PTT.BK", "XYZ"][: int(rng.integers(1, 5))]
        prices = {s: _price_frame(rng) for s in syms}
        yf_map = {s: s.lower() for s in syms if rng.random() < 0.8}
        splits = {}
        for s in syms:
            kind = int(rng.choice([0, 0, 1, 4, 2, 3]))
            splits[s.lower()] = [_split(rng, kind) for _ in range(int(rng.integers(0, 4)))]
        warned = [set(), set()]
        outs = []
        for i, mod in enumerate((ref, new)):
            p = {k: v.copy() for k, v in prices.items()}
            outs.append((call(lambda: mod._unadjust_prices(p, yf_map, splits, warned[i]), caplog), canon(p)))
        assert outs[0] == outs[1], (seed, trial)
        assert sorted(warned[0]) == sorted(warned[1])


@pytest.mark.parametrize("compare", [False, True])
def test_daily_metrics_worker_matches(caplog, monkeypatch, compare):
    rng = np.random.default_rng(14400 + compare)
    configured = []
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: configured.append(sorted(kw.items())))
    for trial in range(40):
        value = [12345.5, np.nan, np.float64(10.0), None][int(rng.integers(0, 4))]
        py_value = [12345.5, 12000.0, np.nan][int(rng.integers(0, 3))]
        fail_mode = int(rng.integers(0, 6))

        def valuer(*a, **k):
            if fail_mode == 0:
                raise RuntimeError("valuation exploded")
            return value, bool(fail_mode == 1)

        def py_valuer(*a, **k):
            if fail_mode == 2:
                raise ValueError("python valuer broke")
            return py_value, False

        def flows(*a, **k):
            if fail_mode == 3:
                raise KeyError("Date")
            return 25.0, False

        for mod in (new, ref):
            monkeypatch.setattr(mod, "_calculate_portfolio_value_at_date_unadjusted", valuer)
            monkeypatch.setattr(mod, "_calculate_portfolio_value_at_date_unadjusted_python", py_valuer)
            monkeypatch.setattr(mod, "_calculate_daily_net_cash_flow", flows)
            monkeypatch.setattr(mod, "HISTORICAL_COMPARE_METHODS", compare)
        day = [date(2024, 3, 1), date(2024, 3, 5)][trial % 2]
        method = ["numba", "python"][int(rng.integers(0, 2))]
        outs = []
        for mod in (ref, new):
            configured.clear()
            args = (day, pd.DataFrame(), {}, {}, {}, "USD", {}, {}, "USD", None, {}, {}, {}, {}, {}, {}, {})
            out = call(lambda: mod._calculate_daily_metrics_worker(*args, calc_method=method, included_accounts=["IBKR"]), caplog)
            masked = [(lv, msg if "COMPARE" not in msg else msg.split("(")[0]) for lv, msg in out[1]]
            outs.append((out[0], masked, list(configured)))
        assert outs[0] == outs[1], (compare, trial)


def test_manual_prices_by_symbol_id_match():
    rng = np.random.default_rng(14500)
    for _ in range(60):
        overrides = {
            k: v
            for k, v in {
                "aapl ": {"price": 10.5},
                "PTT.BK": 3,
                "Fund": {"price": "12.25"},
                "BAD": {"price": "n/a"},
                "ZERO": 0,
                "INF": float("inf"),
                "NONE": {"sector": "x"},
            }.items()
            if rng.random() < 0.6
        }
        ids = {i: s for i, s in enumerate(["AAPL", "ptt.bk", "FUND", "BAD", "ZERO", "INF", "NONE", "OTHER"]) if rng.random() < 0.8}
        overrides = overrides if rng.random() < 0.9 else None
        assert canon(new.manual_prices_by_symbol_id(overrides, ids)) == canon(ref.manual_prices_by_symbol_id(overrides, ids))


class FakeFunds:
    mode = "ok"

    def get_fund_nav_coverage(self):
        if self.mode == "down":
            raise RuntimeError("no fund store")
        return {} if self.mode == "empty" else {"SCBRMS&P500": 1, "scbrm1": 2, "BROKEN": 3, "NOPRICE": 4, "NAIVE": 5}

    def get_fund_nav(self, code, start, end):
        days = pd.date_range(start, end, freq="7D")
        if code == "BROKEN":
            raise OSError("read failed")
        if code == "NOPRICE":
            return pd.DataFrame({"nav": [1.0]})
        if code == "NAIVE":
            return pd.DataFrame({"price": np.linspace(10, 11, len(days))}, index=days.date)
        return pd.DataFrame({"price": np.where(np.arange(len(days)) % 3, 10.0, np.nan)}, index=days.tz_localize("UTC"))


@pytest.mark.parametrize("mode", ["ok", "down", "empty"])
def test_fund_nav_series_match(caplog, monkeypatch, mode):
    monkeypatch.setattr(market_db, "MarketDatabase", FakeFunds)
    FakeFunds.mode = mode
    rng_ids = {0: "SCBRMS&P500", 1: "SCBRM1", 2: "BROKEN", 3: "NOPRICE", 4: "naive ", 5: "AAPL"}
    grid = pd.date_range("2024-01-01", "2024-06-30", freq="D", tz="UTC")
    for ids in (rng_ids, {}, {5: "AAPL"}):
        a, b = both(lambda mod: mod._fund_nav_series(ids, grid), caplog)
        assert a == b


def test_align_prices_to_grid_match():
    rng = np.random.default_rng(14600)
    for _ in range(40):
        grid = pd.date_range("2024-01-01", periods=30, freq="D", tz="UTC")
        series = {
            i: pd.Series(
                rng.uniform(1, 9, k),
                index=pd.DatetimeIndex(sorted(rng.choice(pd.date_range("2023-12-20", periods=60, freq="6h", tz="UTC"), k, replace=False))),
            )
            for i, k in enumerate(rng.integers(1, 20, int(rng.integers(1, 5))))
        }
        assert canon(new.align_prices_to_grid(series, grid)) == canon(ref.align_prices_to_grid(series, grid))


def test_prune_holdings_cache_match(caplog, tmp_path):
    for case in range(4):
        dirs = []
        for side in ("a", "b"):
            d = tmp_path / f"{case}{side}"
            d.mkdir()
            for i in range(6):
                f = d / f"h{i}.pkl"
                f.write_text("x")
                age = [0, 1, 3, 9, 30, 400][i] * 86400
                os.utime(f, (time.time() - age, time.time() - age))
            (d / "sub").mkdir()
            dirs.append(d)
        missing = case == 3
        outs = []
        for mod, d in zip((ref, new), dirs):
            target = str(d / "nope") if missing else str(d)
            result, logs, warned = call(lambda: mod._prune_holdings_cache(target, [7, 2, 0, 7][case]), caplog)
            logs = [(lv, msg.replace(str(d), "<dir>")) for lv, msg in logs]
            outs.append(((result, logs, warned), sorted(p.name for p in d.iterdir())))
        assert outs[0] == outs[1], case


def test_intraday_symbols_for_window_match():
    rng = np.random.default_rng(14700)
    id_to_symbol = {0: "AAPL", 1: "MSFT", 2: "PTT.BK", 3: "$CASH"}
    yf_map = {"AAPL": "AAPL", "MSFT": "MSFT", "PTT.BK": "PTT.BK"}
    for _ in range(60):
        days = int(rng.integers(0, 30))
        qty = None if rng.random() < 0.1 else rng.choice([0.0, 0.0, 5.0, 1e-7], size=(days, 4, 2))
        tx = pd.DataFrame(
            {
                "Date": pd.date_range("2024-01-01", periods=8, freq="3D"),
                "Symbol": rng.choice(["AAPL", "MSFT", "PTT.BK", None, "OTHER"], 8),
            }
        )
        if rng.random() < 0.2:
            tx = tx.iloc[0:0]
        start = date(2024, 1, 1) + pd.Timedelta(days=int(rng.integers(-5, 20)))
        args = (["AAPL", "MSFT", "PTT.BK", "SPY", "QQQ"], ["SPY"], qty, date(2024, 1, 1), start.date() if hasattr(start, "date") else start, date(2024, 1, 20), id_to_symbol, yf_map, tx)
        assert new._intraday_symbols_for_window(*args) == ref._intraday_symbols_for_window(*args)


def test_generate_mappings_match():
    rng = np.random.default_rng(14800)
    for _ in range(60):
        n = int(rng.integers(1, 30))
        df = pd.DataFrame(
            {
                "Symbol": rng.choice(["aapl", " MSFT", "$CASH", "PTT.BK", None], n),
                "Account": rng.choice(["IBKR", "scbs ", "Kim Eng"], n),
                "To Account": rng.choice(["SCBS", None, "IBKR"], n),
                "Type": rng.choice(["Buy", "buy ", "SELL", "Transfer"], n),
                "Local Currency": rng.choice(["USD", "THB", None], n),
            }
        )
        if rng.random() < 0.2:
            df = df.drop(columns=["To Account"])
        assert canon(new.generate_mappings(df)) == canon(ref.generate_mappings(df))
