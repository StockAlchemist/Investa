"""Phase 15e: the glue of buffett_rank in Rust (rust/investa_core/src/br.rs)
against the original frozen in tests/reference/buffett_rank_p15.py.

Random universes of every model, with each pillar metric random, missing,
NaN or exactly at a gate threshold; gate failures carried in; coverage from
None to full. Run through the gates, the confidence factor, the quality
scoring, the composite with and without value scores, and the re-blend and
re-rank at several AI and quality weights. Results and logs must agree.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_rank as new  # noqa: E402
from reference import buffett_rank_p15 as ref  # noqa: E402
from reference.buffett_metrics_p15 import CompanyMetrics  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402

GATE_KEYS = ["equity_to_assets_latest", "loss_years", "roe_median", "debt_to_ffo", "negative_ffo_years",
             "interest_coverage", "net_debt_to_owner_earnings", "negative_owner_earnings_years", "debt_free"]  # fmt: skip
THRESHOLDS = {"equity_to_assets_latest": 4.0, "loss_years": 2, "roe_median": 0.0, "debt_to_ffo": 15.0,
              "negative_ffo_years": 2, "interest_coverage": 2.0, "net_debt_to_owner_earnings": 6.0,
              "negative_owner_earnings_years": 2}  # fmt: skip


def _universe(rng):
    keys = set(GATE_KEYS)
    for spec in new.PILLARS_BY_MODEL.values():
        for entries in spec.values():
            keys.update(k for k, _ in entries)
    companies = []
    for i in range(int(rng.integers(0, 60))):
        model = str(rng.choice(["generic", "generic", "bank", "insurer", "reit", "odd"]))
        c = CompanyMetrics(cik=f"{i:010d}", symbol=f"S{i:03d}", name=f"Co {i}", model=model)
        c.period_count = int(rng.integers(0, 18))
        c.latest_period = "2024-12-31"
        c.coverage = [None, 0.0, 0.4, 1.0, float(rng.uniform(0, 1))][int(rng.integers(0, 5))]
        if rng.random() < 0.1:
            c.gate_failures.append(str(rng.choice(["no_fundamentals", "fundamentals_unavailable", "other"])))
        for k in keys:
            r = rng.random()
            if r < 0.15:
                continue
            if r < 0.2:
                c.metrics[k] = float("nan")
            elif r < 0.25 and k in THRESHOLDS:
                c.metrics[k] = THRESHOLDS[k]
            elif k == "debt_free":
                c.metrics[k] = bool(rng.random() < 0.3)
            else:
                c.metrics[k] = float(rng.normal(8, 10))
        companies.append(c)
    return companies


@pytest.mark.parametrize("seed", range(8))
def test_rank_glue_matches(caplog, seed):
    rng = np.random.default_rng(16300 + seed)
    for trial in range(8):
        companies = _universe(rng)
        for c in companies:
            for mp in (8, 5):
                assert new.evaluate_gates(c, mp) == ref.evaluate_gates(c, mp)
            c.gate_failures.extend(ref.evaluate_gates(c))
        a = call(lambda: ref.score_quality(companies), caplog)
        b = call(lambda: new.score_quality(companies), caplog)
        assert a == b, (seed, trial)
        quality = ref.score_quality(companies)
        values = None
        if not quality.empty and rng.random() < 0.7:
            values = pd.Series(np.where(rng.random(len(quality)) < 0.2, np.nan, rng.uniform(0, 100, len(quality))), index=quality.index)
        qw, vw = [(0.6, 0.4), (0.8, 0.2), (1.0, 0.0)][int(rng.integers(0, 3))]
        a = call(lambda: ref.combine(quality, values, qw, vw), caplog)
        b = call(lambda: new.combine(quality, values, qw, vw), caplog)
        assert a == b, (seed, trial, "combine")
        try:
            combined = ref.combine(quality, values, qw, vw)
        except Exception:
            continue  # both raised alike (compared above), nothing to rerank
        if not combined.empty and rng.random() < 0.6:
            combined["ai_score"] = np.where(rng.random(len(combined)) < 0.3, np.nan, rng.uniform(0, 100, len(combined)))
        for w in (0.0, 0.2, 1.5, -1.0):
            a = call(lambda: ref.rerank(combined, qw, w), caplog)
            b = call(lambda: new.rerank(combined, qw, w), caplog)
            assert a == b, (seed, trial, "rerank", w)


def test_confidence_factor_matches():
    for cov in (None, 0.0, 0.5, 1.0, 2.0, -1.0, float("nan"), np.float64(0.3)):
        for floor in (0.5, 0.0, 1.0):
            assert repr(new.confidence_factor(cov, floor)) == repr(ref.confidence_factor(cov, floor)), (cov, floor)
