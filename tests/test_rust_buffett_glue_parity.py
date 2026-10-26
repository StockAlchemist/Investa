"""Phase 15d: the glue of buffett_metrics and buffett_value in Rust
(rust/investa_core/src/bm.rs) against the originals frozen in
tests/reference/buffett_metrics_p15.py and buffett_value_p15.py.

EDGAR (concepts, sector concepts, the split-consistent share series, the
statements and the company session) and the ratio engine are faked with
random series: ints, floats, NaN, infinity, NumPy scalars and text (which
must raise), missing concepts, empty fundamentals, failing loads, failing
share rebuilds and failing ratio runs, under every model. The market metrics
take every price/share shape and debt state; the value score and frame run
over random universes. Results (CompanyMetrics compared field by field) and
logs agree.
"""

import contextlib
import dataclasses
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_metrics as new_m  # noqa: E402
import buffett_value as new_v  # noqa: E402
import edgar_provider  # noqa: E402
import financial_ratios  # noqa: E402
from reference import buffett_metrics_p15 as ref_m  # noqa: E402
from reference import buffett_value_p15 as ref_v  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402

CONCEPTS = ["net_income", "revenue", "equity", "total_assets", "operating_cash_flow", "capex", "shares_diluted",
            "long_term_debt", "short_term_debt", "interest_expense", "pretax_income", "cash", "gross_profit",
            "depreciation", "dividends_paid", "current_assets", "current_liabilities"]  # fmt: skip


def _plain(x):
    """CompanyMetrics (either module's class) as a plain dict for canon."""
    if dataclasses.is_dataclass(x):
        return {f.name: getattr(x, f.name) for f in dataclasses.fields(x)}
    return x


class FakeEdgar:
    def __init__(self, rng):
        self.rng = rng
        self.mode = int(rng.integers(0, 10))

    def _series(self, r):
        out = {}
        for y in range(2008, 2025):
            x = r.random()
            if x < 0.1:
                continue
            v = float(r.normal(2e9, 3e9))
            out[f"{y}-12-31"] = [v, int(v), np.float64(v), float("nan"), float("inf")][int(r.choice(5, p=[0.7, 0.1, 0.1, 0.05, 0.05]))]
        return out

    def get_concept_values(self, cik, concepts=None):
        r = np.random.default_rng(abs(hash((cik, tuple(concepts or ())))) % 2**32)
        if self.mode == 0 and concepts is None:
            raise OSError("fact store unavailable")
        if self.mode == 1 and concepts is None:
            return {}
        names = concepts or CONCEPTS
        out = {c: self._series(r) for c in names if r.random() < 0.92}
        if self.mode == 2 and out:
            out[next(iter(out))]["2015-12-31"] = "1,000"  # text: _raw_number raises
        return out

    def split_consistent_series(self, cik, concept):
        if self.mode == 3:
            raise ValueError("no ratio filings")
        r = np.random.default_rng(7)
        return {f"{y}-12-31": float(r.uniform(1e8, 1e9)) for y in range(2010, 2025)}

    def get_statements(self, cik):
        return {"financials": "F", "balance_sheet": "B"}

    @contextlib.contextmanager
    def company_session(self, cik):
        yield


def _fake_ratios(mode):
    def calc(financials, balance):
        if mode == 4:
            raise ZeroDivisionError("bad statement")
        if mode == 5:
            return pd.DataFrame()
        idx = pd.to_datetime([f"{y}-12-31" for y in range(2012, 2025)])
        r = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "Period": [str(i.date()) for i in idx],
                "Return on Equity (ROE) (%)": r.normal(15, 6, len(idx)),
                "Return on Assets (ROA) (%)": r.normal(6, 3, len(idx)),
                "Gross Profit Margin (%)": r.normal(40, 10, len(idx)),
                "Net Profit Margin (%)": r.normal(12, 5, len(idx)),
                "Debt-to-Equity Ratio": np.where(r.random(len(idx)) < 0.1, np.nan, r.uniform(0, 2, len(idx))),
                "Current Ratio": r.uniform(0.5, 3, len(idx)),
            },
            index=idx if mode != 6 else [f"FY{y}" for y in range(2012, 2025)],
        )
        return frame

    return calc


@pytest.mark.parametrize("seed", range(10))
def test_compute_metrics_matches(caplog, monkeypatch, seed):
    rng = np.random.default_rng(16100 + seed)
    for trial in range(15):
        fake = FakeEdgar(rng)
        for name in ("get_concept_values", "split_consistent_series", "get_statements", "company_session"):
            monkeypatch.setattr(edgar_provider, name, getattr(fake, name))
        monkeypatch.setattr(financial_ratios, "calculate_key_ratios_timeseries", _fake_ratios(fake.mode))
        model = str(rng.choice(["generic", "bank", "insurer", "reit", "other"]))
        cik = f"{int(rng.integers(1, 99999)):010d}"
        a = call(lambda: _plain(ref_m.compute_metrics(cik, "SYM", "Name", model)), caplog)
        b = call(lambda: _plain(new_m.compute_metrics(cik, "SYM", "Name", model)), caplog)
        assert a == b, (seed, trial, fake.mode, model)


def test_helpers_match():
    def outcome(f, *args):
        try:
            return ("ok", canon(f(*args)))
        except Exception as e:  # 10**400 overflows float(), which neither catches
            return ("raised", type(e).__name__, str(e))

    values = [None, 0, 3, -2.5, float("nan"), float("inf"), "4.5", "x", True, np.float64(1.5), np.int64(7), [1], 10**400]
    for v in values:
        assert outcome(new_m._finite, v) == outcome(ref_m._finite, v), v
        for d in values:
            assert outcome(new_m._safe_divide, v, d) == outcome(ref_m._safe_divide, v, d), (v, d)
        assert outcome(new_m._raw_number, v) == outcome(ref_m._raw_number, v), v
    rows = [("a", 3.0, True), ("b", 2.5, False), ("c", None, False)]
    assert canon(new_m._from_rust(rows)) == canon(ref_m._from_rust(rows))


@pytest.mark.parametrize("seed", range(6))
def test_value_glue_matches(caplog, monkeypatch, seed):
    rng = np.random.default_rng(16200 + seed)
    fake = FakeEdgar(rng)
    fake.mode = 9
    monkeypatch.setattr(edgar_provider, "get_concept_values", fake.get_concept_values)
    for trial in range(10):
        companies = []
        market = {}
        for i in range(int(rng.integers(0, 30))):
            c = ref_m.CompanyMetrics(cik=f"{i:010d}", symbol=f"S{i}", name=f"Co {i}", model=str(rng.choice(["generic", "bank", "reit"])))
            c.latest_period = [None, "2024-12-31", "2023-12-31"][int(rng.integers(0, 3))]
            c.metrics = {
                "owner_earnings_latest": [None, 0.0, float(rng.normal(1e9, 1e9))][int(rng.integers(0, 3))],
                "ffo_latest": float(rng.normal(5e8, 1e8)),
                "total_debt": [None, 0, float(rng.uniform(0, 5e9)), np.float64(1e9)][int(rng.integers(0, 4))],
                "debt_free": bool(rng.random() < 0.3),
            }
            companies.append(c)
            if rng.random() < 0.85:
                market[c.symbol] = {
                    "price": [None, 0.0, -1.0, float(rng.uniform(5, 300)), "12", np.float64(40.0)][int(rng.integers(0, 6))],
                    "shares": [None, float(rng.uniform(1e7, 1e9)), 0][int(rng.integers(0, 3))],
                }
        a = call(lambda: ref_v.build_value_frame(companies, market), caplog)
        b = call(lambda: new_v.build_value_frame(companies, market), caplog)
        assert a == b, (seed, trial)
        frame = ref_v.build_value_frame(companies, market)
        a = call(lambda: ref_v.score_value(frame.copy()), caplog)
        b = call(lambda: new_v.score_value(frame.copy()), caplog)
        assert a == b, (seed, trial, "score")
