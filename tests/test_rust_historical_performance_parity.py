"""Phase 14e: portfolio_history.calculate_historical_performance in Rust
(rust/investa_core/src/ph_performance.rs) against the original frozen in
tests/reference/portfolio_history.py.

A deterministic fake market provider serves daily and intraday prices and
FX (with fetch-failure flags), and benchmark frames that are missing, empty,
or lack a usable column — the original reuses the previous benchmark's
series then, or raises when there is none, and so must the port. Requests
cover every interval, invalid intervals and date orders, benchmark lists of
every shape, account filters that keep, empty and void the ledger (an empty
effective ledger raises UnboundLocalError in the original), the daily cache
with and without a source path, and past dates only for intraday (the grid
clips to "now"). Each side gets its own L1 cache directory. The whole result
and the logs, with timings and directories masked, must agree.
"""

import logging
import os
import sys
from datetime import date

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import config  # noqa: E402
import market_db  # noqa: E402
import portfolio_history as new  # noqa: E402
from reference import portfolio_history as ref  # noqa: E402
from test_rust_history_daily_parity import _inputs  # noqa: E402
from test_rust_history_helpers_parity import FakeFunds  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

TODAY = date(2024, 6, 17)
TIMING = ("Calc Time", "Pool finished in", "1D Graph Clip")


class FakeProvider:
    def __init__(self, seed, flavour):
        self.seed, self.flavour = seed, flavour

    def _frame(self, rng, start, end, interval, col="price"):
        freq = {"1d": "B", "1h": "h", "5m": "5min"}.get(interval, "B")
        idx = pd.date_range(pd.Timestamp(start), pd.Timestamp(end), freq=freq, tz="UTC" if rng.random() < 0.7 else None)
        if interval != "1d":
            idx = idx[(idx.hour >= 14) & (idx.hour <= 20)]
        return pd.DataFrame({col: 100 + np.cumsum(rng.normal(0, 1, len(idx)))}, index=idx)

    def get_historical_data(self, symbols_yf, start_date, end_date, interval="1d", use_cache=True, cache_key=None, integrity_check_symbols=None):
        rng = np.random.default_rng(self.seed + len(symbols_yf) + (interval != "1d"))
        out = {}
        for s in sorted(symbols_yf):
            if s == "DIA":
                continue
            col = "Volume" if (s == "QQQ" and self.flavour == "bad-bench") else str(rng.choice(["price", "Close", "Adj Close"]))
            out[s] = self._frame(rng, start_date, end_date, interval, col)
        if "IWM" in out:
            out["IWM"] = out["IWM"].iloc[0:0]  # an empty benchmark frame
        return out, self.flavour == "fail"

    def get_historical_fx_rates(self, fx_pairs_yf, start_date, end_date, interval="1d", use_cache=True, cache_key=None):
        rng = np.random.default_rng(self.seed + 7 + (interval != "1d"))
        return {p: self._frame(rng, start_date, end_date, interval) for p in sorted(fx_pairs_yf)}, False


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


@pytest.fixture(autouse=True)
def _fixed(monkeypatch):
    monkeypatch.setattr(logging, "basicConfig", lambda **kw: None)
    monkeypatch.setattr(market_db, "MarketDatabase", FakeFunds)
    FakeFunds.mode = "ok"
    for mod in (new, ref):
        monkeypatch.setattr(mod, "get_latest_trading_date", lambda: date(2024, 6, 14))
        monkeypatch.setattr(mod, "get_est_today", lambda: TODAY)


@pytest.mark.parametrize("seed", range(8))
def test_historical_performance_matches(caplog, monkeypatch, tmp_path, seed):
    rng = np.random.default_rng(15500 + seed)
    for trial in range(6):
        txs, *_ = _inputs(rng)
        interval = str(rng.choice(["D", "1d", "W", "ME", "M", "1h", "5m", "bad"]))
        start, end = date(2024, 2, 1), date(2024, 6, 20)
        if interval in ("1h", "5m"):
            start, end = date(2024, 6, 10), date(2024, 6, 14)
        if rng.random() < 0.05:
            start, end = end, start
        benchmarks = [["SPY"], ["spy ", 5], "SPY", [], ["SPY", "QQQ", "DIA", "IWM"], ["QQQ", "SPY"]][int(rng.integers(0, 6))]
        include = [None, ["IBKR"], ["SCBS", "Kim Eng"], ["Nowhere"]][int(rng.integers(0, 4))]
        flavour = str(rng.choice(["plain", "plain", "fail", "bad-bench"]))
        csv = tmp_path / f"{seed}_{trial}.csv"
        csv.write_text("x")
        path = [None, str(csv)][int(rng.integers(0, 2))]
        outs = []
        for side, mod in (("a", ref), ("b", new)):
            d = tmp_path / f"{trial}{side}"
            d.mkdir()
            monkeypatch.setattr(config, "get_app_cache_dir", lambda d=d: str(d))
            provider = FakeProvider(seed * 10 + trial, flavour)
            monkeypatch.setattr(mod, "get_shared_mdp", lambda **kw: provider)
            out = call(
                lambda: mod.calculate_historical_performance(
                    txs.copy(), None, {3}, {3: "skipped"}, start, end, interval, benchmarks, "USD",
                    {"SCBS": "THB"}, "USD", use_daily_results_cache=bool(trial % 2), num_processes=1,
                    include_accounts=include, manual_overrides_dict={"AAPL": {"price": 3.0}},
                    original_csv_file_path=path,
                ),
                caplog,
            )  # fmt: skip
            outs.append(_masked(out, d))
        assert outs[0] == outs[1], (seed, trial, interval, benchmarks, include, flavour)


@pytest.mark.parametrize("case", ["empty_ledger", "stale_benchmark", "first_benchmark_unusable"])
def test_unbound_variable_quirks_match(caplog, monkeypatch, tmp_path, case):
    """The original's two unbound-variable paths, kept (see the port doc)."""
    txs, *_ = _inputs(np.random.default_rng(99))
    benchmarks = {"empty_ledger": ["SPY"], "stale_benchmark": ["SPY", "QQQ"], "first_benchmark_unusable": ["QQQ", "SPY"]}[case]
    exclude = ["IBKR", "SCBS", "KIM ENG"] if case == "empty_ledger" else None
    outs = []
    for side, mod in (("a", ref), ("b", new)):
        d = tmp_path / side
        d.mkdir()
        monkeypatch.setattr(config, "get_app_cache_dir", lambda d=d: str(d))
        provider = FakeProvider(5, "bad-bench")
        monkeypatch.setattr(mod, "get_shared_mdp", lambda **kw: provider)
        out = call(
            lambda: mod.calculate_historical_performance(
                txs.copy(), None, set(), {}, date(2024, 2, 1), date(2024, 6, 20), "1d", benchmarks, "USD",
                {}, "USD", use_daily_results_cache=False, num_processes=1, exclude_accounts=exclude,
            ),
            caplog,
        )  # fmt: skip
        outs.append(_masked(out, d))
    assert outs[0] == outs[1]
    if case != "stale_benchmark":
        assert outs[1][0][0] == "raised" and outs[1][0][1] == "UnboundLocalError", outs[1][0]
