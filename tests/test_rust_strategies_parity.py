"""Phase 15c: strategies in Rust (rust/investa_core/src/st.rs) against the
original frozen in tests/reference/strategies_p15.py.

The trend readings over random daily histories (naive and zoned, with gaps,
too short for the average, ending mid-month or on a month-end), and every
registered strategy's allocation through a fake market-data provider, a fake
ranking store and a fake SIC map: scores with NaNs, missing and text prices,
market caps for the large-cap filter, CIKs that are text or missing, live
quotes that fail or are missing, stale and fresh runs, short books, and
every AI-weight override. Results and logs must agree.
"""

import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_store  # noqa: E402
import edgar_sic  # noqa: E402
import server.route_utils as route_utils  # noqa: E402
import strategies as new  # noqa: E402
from reference import strategies_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402


def _prices(rng):
    n = int(rng.integers(0, 900))
    end = pd.Timestamp("2024-06-14") + pd.Timedelta(days=int(rng.integers(0, 40)))
    idx = pd.bdate_range(end=end, periods=n, tz="UTC" if rng.random() < 0.5 else None)
    s = pd.Series(400 + np.cumsum(rng.normal(0, 3, n)), index=idx)
    s[rng.random(n) < 0.03] = np.nan
    return s


@pytest.mark.parametrize("seed", range(6))
def test_trend_signal_matches(caplog, seed):
    rng = np.random.default_rng(15800 + seed)
    for _ in range(40):
        prices = _prices(rng)
        months = int(rng.choice([3, 10, 12]))
        today = [date(2024, 6, 14), date(2024, 7, 2), date(2024, 12, 31), None][int(rng.integers(0, 4))]
        a = call(lambda: ref.evaluate_trend_signal(prices, months, today), caplog)
        b = call(lambda: new.evaluate_trend_signal(prices, months, today), caplog)
        assert a == b


class FakeMdp:
    def __init__(self, rng, fail):
        self.rng, self.fail = rng, fail

    def get_historical_data(self, symbols, start, end):
        if self.fail:
            raise ConnectionError("yahoo down")
        out = {}
        for s in symbols:
            r = np.random.default_rng(abs(hash(s)) % 1000)
            if r.random() < 0.15:
                continue
            idx = pd.bdate_range(start, end)
            out[s] = pd.DataFrame({"price": 50 + np.cumsum(r.normal(0, 1, len(idx)))}, index=idx)
        return out, False


class FakeStore:
    def __init__(self, rng, mode):
        self.rng, self.mode = rng, mode
        n = int(rng.integers(5, 60)) if mode != "tiny" else 4
        self.frame = pd.DataFrame(
            {
                "symbol": [f"S{i:03d}" for i in range(n)],
                "name": [f"Company {i}" for i in range(n)],
                "cik": [str(1000 + i) if rng.random() < 0.9 else (None if rng.random() < 0.5 else "n/a") for i in range(n)],
                "quality_score": np.where(rng.random(n) < 0.05, np.nan, rng.uniform(0, 100, n)),
                "value_score": np.where(rng.random(n) < 0.1, np.nan, rng.uniform(0, 100, n)),
                "price": [float(rng.uniform(5, 500)) if rng.random() < 0.85 else None for _ in range(n)],
                "market_cap": rng.choice([5e9, 2e10, 3e11, np.nan], n),
            }
        )  # fmt: skip

    def get_run(self):
        if self.mode == "none":
            return None
        finished = (date(2024, 6, 14) - timedelta(days=int(self.rng.integers(0, 12)))).isoformat() + "T03:00:00"
        return {"run_id": 7, "finished_at": finished if self.mode != "bad_date" else "yesterday"}

    def get_scores_frame(self, run_id):
        return None if self.mode == "no_scores" else self.frame.copy()


def _attach(frame):
    frame = frame.copy()
    frame["ai_rating"] = [7.5 if i % 3 else np.nan for i in range(len(frame))]
    frame["ai_score"] = [60.0 if i % 3 else np.nan for i in range(len(frame))]
    return frame


@pytest.mark.parametrize("seed", range(8))
def test_allocation_matches(caplog, monkeypatch, seed):
    rng = np.random.default_rng(15900 + seed)
    for mod in (new, ref):
        monkeypatch.setattr(mod, "attach_ai_scores", _attach)
    monkeypatch.setattr(edgar_sic, "get_sic_map", lambda: {str(1000 + i).zfill(10): [6351, 7372, 6798, 2834][i % 4] for i in range(0, 60, 2)})
    for trial in range(12):
        mode = str(rng.choice(["ok", "ok", "ok", "none", "no_scores", "tiny", "bad_date"]))
        store_state = rng.bit_generator.state
        fail = bool(rng.random() < 0.15)
        strategy = new.list_strategies()[int(rng.integers(0, len(new.list_strategies())))]
        capital = float(rng.choice([10000.0, 250000.0, 0.0]))
        today = [date(2024, 6, 14), None][int(rng.integers(0, 2))]
        ai_weight = [None, 0.0, 0.35][int(rng.integers(0, 3))]
        outs = []
        for mod in (ref, new):
            rng.bit_generator.state = store_state
            store = FakeStore(rng, mode)
            monkeypatch.setattr(buffett_store, "get_store", lambda s=store: s)
            monkeypatch.setattr(route_utils, "get_mdp", lambda: FakeMdp(rng, fail))
            outs.append(call(lambda: mod.build_allocation(strategy, capital, today, ai_weight), caplog))
        assert outs[0] == outs[1], (seed, trial, mode, strategy.id)
        for s in new.list_strategies():
            assert new.strategy_payload(s) == ref.strategy_payload(s)


def test_market_signal_and_helpers_match(caplog, monkeypatch):
    rng = np.random.default_rng(16000)
    for fail in (False, True):
        monkeypatch.setattr(route_utils, "get_mdp", lambda: FakeMdp(rng, fail))
        for symbol in ("SPY", "qqq", "DIA", "NOPE"):
            a = call(lambda: ref.market_trend_signal(symbol, 10, date(2024, 6, 14)), caplog)
            b = call(lambda: new.market_trend_signal(symbol, 10, date(2024, 6, 14)), caplog)
            assert a == b
    for sic in ("", "6798", "6351", "7372", "9999", "60"):
        assert new._industry_label(sic) == ref._industry_label(sic)
    for v in (None, "3.5", "x", float("nan"), 2, np.float64(1.5)):
        assert new._optional_float(v) == ref._optional_float(v)
    for finished in (None, "", "2024-06-01T00:00:00", "garbage", "2024-07-01"):
        assert new.ranking_age_days(finished, date(2024, 6, 14)) == ref.ranking_age_days(finished, date(2024, 6, 14))
    for d in (date(2024, 1, 15), date(2024, 12, 3), date(2024, 2, 10)):
        assert new._next_month_end(d) == ref._next_month_end(d)
