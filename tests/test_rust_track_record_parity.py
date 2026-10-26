"""Phase 15b: track_record in Rust (rust/investa_core/src/tr.rs) against the
original frozen in tests/reference/track_record_p15.py.

The formatter over every unit with NaN, inf, text, zero and missing
denominators; the labels over every model's real pillar spec; the downturn
windows over random yearly series; money at every scale; revisions through a
fake fact store; and `build` with model inference, metrics, gates, the ranked
row, bands and stress each faked to succeed or fail. Results and logs agree.
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_metrics  # noqa: E402
import buffett_pipeline  # noqa: E402
import buffett_rank  # noqa: E402
import buffett_store  # noqa: E402
import edgar_provider  # noqa: E402
import track_record as new  # noqa: E402
import valuation_history  # noqa: E402
from reference import track_record_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402

UNITS = ["percent", "cagr", "points", "times", "share", "years", "ratio", "other"]


def both(caplog, f):
    return call(lambda: f(ref), caplog), call(lambda: f(new), caplog)


def test_format_matches(caplog):
    rng = np.random.default_rng(15700)
    values = [None, 0.0, 13.456, -0.034, float("nan"), float("inf"), "12", 7, np.float64(0.25), 2.5]
    denoms = [None, 0, 10.0, float("nan"), 3, np.float64(12.0), "x"]
    for _ in range(400):
        v = values[int(rng.integers(0, len(values)))]
        u = UNITS[int(rng.integers(0, len(UNITS)))]
        d = denoms[int(rng.integers(0, len(denoms)))]
        a, b = both(caplog, lambda m: m._format(v, u, d))
        assert a == b, (v, u, d)


def test_labelled_metrics_match(caplog):
    rng = np.random.default_rng(15701)
    keys = set()
    for spec in buffett_rank.PILLARS_BY_MODEL.values():
        for entries in spec.values():
            keys.update(k for k, _ in entries)
    keys |= {"roe_observation_years", "owner_earnings_years", "net_income_years", "ffo_years"}
    for _ in range(60):
        metrics = {k: [None, float(rng.normal(10, 8)), 0.0, 9.0][int(rng.integers(0, 4))] for k in keys if rng.random() < 0.8}
        model = str(rng.choice(list(buffett_rank.PILLARS_BY_MODEL) + ["unknown"]))
        a, b = both(caplog, lambda m: m.labelled_metrics(metrics, model))
        assert a == b, model


def _values(rng):
    def series(scale):
        return {f"{y}-{rng.choice(['12-31', '06-30', '01-31'])}": float(rng.normal(scale, scale * 0.6)) for y in range(2003, 2025) if rng.random() < 0.85}

    out = {c: series(5e9) for c in ("revenue", "net_income", "operating_cash_flow")}
    out["capex"] = {k: float(rng.uniform(0, 1e9)) if rng.random() < 0.8 else None for k in out["operating_cash_flow"]}
    out["operating_cash_flow"]["2015-12-31"] = None
    out["revenue"]["bad"] = 1.0
    for c in list(out):
        if rng.random() < 0.1:
            out.pop(c)
    return out


def test_stress_and_money_match(caplog):
    rng = np.random.default_rng(15702)
    for _ in range(80):
        values = _values(rng)
        a, b = both(caplog, lambda m: m.stress_response(values))
        assert a == b
    for v in [0.0, -5.0, 999.4, 1e3, -2.5e6, 7.25e9, 1.2e12, float("nan"), np.float64(4.5e8)]:
        assert new._money(v) == ref._money(v)


class FakeStore:
    def __init__(self, rows, fail):
        self.rows, self.fail = rows, fail

    def get_symbol_history(self, symbol, limit=1):
        if self.fail:
            raise RuntimeError("rank store locked")
        return self.rows


@pytest.mark.parametrize("seed", range(6))
def test_build_matches(caplog, monkeypatch, seed):
    rng = np.random.default_rng(15710 + seed)
    for trial in range(12):
        flags = rng.random(7) < 0.2
        company = SimpleNamespace(
            gate_failures=("negative equity",) if rng.random() < 0.3 else (),
            period_count=int(rng.integers(0, 20)),
            first_period="2006-12-31",
            latest_period="2024-12-31",
            coverage={"years": 12},
            metrics={"roe_median": 14.2, "revenue_cagr": 0.07, "loss_years": 2, "net_income_years": 10},
        )

        def infer(cik, f=flags[0]):
            if f:
                raise ValueError("no facts")
            return "bank" if cik.endswith("9") else "generic"

        def gates(c, f=flags[1]):
            if f:
                raise KeyError("pillar")
            return ["thin history"] if c.period_count < 5 else []

        row = {"run_id": 9, "rank": 3, "composite_score": 81.0, "name": "Apple Inc.", "returns_on_capital": 90.0}
        revs = [{"concept": "revenue", "period_end": "2017-06-30", "original": 9.6e10, "current": 9.0e10,
                 "change_pct": -6.25, "first_filed": "2017-08-01", "restated_filed": "2018-08-01"}] * int(rng.integers(0, 12))  # fmt: skip

        def revisions(cik, f=flags[2]):
            if f:
                raise OSError("fact store gone")
            return revs

        def values(cik, concepts, f=flags[3]):
            if f:
                raise RuntimeError("no values")
            return _values(rng)

        def bands(symbol, cik, f=flags[4]):
            if f:
                raise RuntimeError("no prices")
            return [{"metric": "earnings"}]

        monkeypatch.setattr(buffett_pipeline, "infer_model_from_facts", infer)
        monkeypatch.setattr(buffett_metrics, "compute_metrics", lambda cik, s, n, m: company)
        monkeypatch.setattr(buffett_rank, "evaluate_gates", gates)
        monkeypatch.setattr(buffett_store, "get_store", lambda: FakeStore([row] if not flags[5] else [], flags[6]))
        monkeypatch.setattr(edgar_provider, "revisions", revisions)
        monkeypatch.setattr(edgar_provider, "concept_labels", lambda: {"revenue": "Revenue"})
        monkeypatch.setattr(edgar_provider, "get_concept_values", values)
        monkeypatch.setattr(valuation_history, "bands", bands)
        cik = str(rng.choice(["0000320193", "0000019617"]))
        name = [None, "Given Name"][int(rng.integers(0, 2))]
        state = rng.bit_generator.state
        a = call(lambda: ref.build("aapl", cik, name), caplog)
        rng.bit_generator.state = state  # the same fake values for both runs
        b = call(lambda: new.build("aapl", cik, name), caplog)
        assert a == b, (seed, trial, flags)
