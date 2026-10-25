"""The Buffett ranking's scoring layer (rust/investa_core/src/rank.rs) against the
originals frozen in tests/reference/buffett_rank.py and buffett_value.py.

Percentiles, pillar means, the quality blend, score_quality and score_value
must match bitwise, with the same index, name and dtype, on random groups that
mix NaN, infinities, heavy ties, constant columns, tiny groups and object or
integer columns.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import buffett_rank as new_rank  # noqa: E402
import buffett_value as new_value  # noqa: E402
from buffett_metrics import CompanyMetrics  # noqa: E402
from reference import buffett_rank as ref_rank  # noqa: E402
from reference import buffett_value as ref_value  # noqa: E402

# The originals' NumPy arithmetic warns on the deliberate infinities; the
# comparison, not the warning, is what these tests check.
pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _same_series(a, b, what):
    assert type(a) is type(b), what
    assert a.index.equals(b.index), f"{what}: index"
    assert a.name == b.name, f"{what}: name {a.name!r} vs {b.name!r}"
    assert a.dtype == b.dtype, f"{what}: dtype {a.dtype} vs {b.dtype}"
    assert a.to_numpy().tobytes() == b.to_numpy().tobytes(), f"{what}: values"


def _same_frame(a, b, what):
    assert list(a.columns) == list(b.columns), f"{what}: columns"
    assert a.index.equals(b.index), f"{what}: index"
    for col in a.columns:
        x, y = a[col], b[col]
        if x.dtype.kind == "f" or y.dtype.kind == "f":
            _same_series(x, y, f"{what}[{col}]")
        else:
            assert x.equals(y), f"{what}[{col}]"


def _column(rng, n):
    r = rng.random()
    if r < 0.1:
        v = np.full(n, float(rng.normal()))  # constant: every value ties
    elif r < 0.2:
        v = rng.choice([1.0, 2.0, 3.0], n)  # heavy ties
    else:
        v = rng.normal(0, 10 ** float(rng.integers(-2, 4)), n)
    v = v.astype(float)
    v[rng.random(n) < float(rng.choice([0.0, 0.2, 0.6]))] = np.nan
    if rng.random() < 0.1 and n:
        v[rng.integers(0, n)] = float(rng.choice([np.inf, -np.inf]))
    return v


@pytest.mark.parametrize("seed", range(4))
def test_winsorised_percentile_matches(seed):
    rng = np.random.default_rng(4100 + seed)
    for _ in range(1500):
        n = int(rng.integers(0, 300))
        s = pd.Series(_column(rng, n), index=[f"S{i}" for i in range(n)], name="m")
        if rng.random() < 0.05 and not s.isna().any() and np.isfinite(s).all():
            s = s.round().astype("int64")
        if rng.random() < 0.05:
            s = s.astype(object)  # pd.to_numeric must coerce
        for hib in (True, False):
            _same_series(
                new_rank._winsorised_percentile(s, hib),
                ref_rank._winsorised_percentile(s, hib),
                "percentile",
            )


def _company(rng, i, model):
    keys = sorted({m for spec in ref_rank.PILLARS_BY_MODEL.values() for p in spec.values() for m, _ in p})
    metrics = {k: (None if rng.random() < 0.2 else float(rng.normal(10, 20))) for k in keys}
    if rng.random() < 0.1:
        metrics["roe_median"] = float("inf")
    return CompanyMetrics(
        cik=str(i), symbol=f"S{i}", name=f"Co {i}", model=model,
        period_count=int(rng.integers(0, 18)), latest_period="2024-12-31",
        metrics=metrics, coverage=float(rng.uniform(0, 1)),
        gate_failures=[] if rng.random() < 0.85 else ["unprofitable"],
    )  # fmt: skip


@pytest.mark.parametrize("seed", range(4))
def test_score_quality_matches(seed):
    rng = np.random.default_rng(4200 + seed)
    for _ in range(40):
        n = int(rng.integers(0, 150))
        models = rng.choice(["generic", "bank", "insurer", "reit"], n, p=[0.7, 0.1, 0.1, 0.1])
        companies = [_company(rng, i, str(m)) for i, m in enumerate(models)]
        _same_frame(new_rank.score_quality(companies), ref_rank.score_quality(companies), "score_quality")
        frame = ref_rank.metrics_to_frame(companies)
        for spec in ref_rank.PILLARS_BY_MODEL.values():
            if frame.empty:
                continue
            p_new, p_ref = new_rank._pillar_scores(frame, spec), ref_rank._pillar_scores(frame, spec)
            _same_frame(p_new, p_ref, "pillars")
            _same_series(new_rank._quality_score(p_new), ref_rank._quality_score(p_ref), "quality")


@pytest.mark.parametrize("seed", range(4))
def test_score_value_matches(seed):
    rng = np.random.default_rng(4300 + seed)
    for _ in range(80):
        n = int(rng.integers(0, 200))
        frame = pd.DataFrame(
            {
                "symbol": [f"S{i}" for i in range(n)],
                "model": rng.choice(["generic", "bank", "reit"], n),
                "earnings_yield": _column(rng, n),
                "fcf_yield": _column(rng, n),
                "price_to_book": _column(rng, n),
            }
        ).set_index("symbol", drop=False)
        if rng.random() < 0.1:
            frame = frame.drop(columns=["fcf_yield"])
        _same_series(new_value.score_value(frame), ref_value.score_value(frame), "score_value")
