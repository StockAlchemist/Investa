"""Phase 15i: the growth, normalization and Monte Carlo glue of
financial_ratios in Rust (rust/investa_core/src/fr_growth.rs) against the
original frozen in tests/reference/financial_ratios_p15.py.

Growth takes analyst rows that are missing, None, text, NaN or infinite, info
fallbacks, and random statement histories (losses, single periods, text
columns). Enrichment, the FCF margins and the normalized base FCF take sparse
profiles and random statements. The Monte Carlo wrappers run from the same
NumPy seed on both sides with ordinary, zero, negative, NaN and non-numeric
inputs, and the statistics take short, non-finite and malformed samples.
Results, logs, warnings and the inputs left behind agree.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import financial_ratios as new  # noqa: E402
from reference import financial_ratios_p15 as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402
from test_rust_ratio_glue_parity import _both  # noqa: E402
from test_rust_ratios_parity import _case  # noqa: E402

ROWS = [None, {}, {"growth": None}, {"growth": "x"}, {"growth": float("nan")}, {"growth": float("inf")},
        {"growth": 0.12}, {"growth": -0.3}, {"growth": "0.08"}, {"other": 1}]  # fmt: skip
VALUES = [None, 0, 0.0, 0.05, -0.2, 3e9, -1e8, float("nan"), "x", np.float64(0.07), 12]


def _info(rng):
    info = {}
    keys = ["earningsGrowth", "revenueGrowth", "sharesOutstanding", "totalRevenue", "totalCash", "totalDebt",
            "trailingEps", "dividendRate", "trailingAnnualDividendRate", "freeCashflow", "symbol"]  # fmt: skip
    for k in keys:
        if rng.random() < 0.6:
            info[k] = VALUES[int(rng.integers(0, len(VALUES)))]
    r = rng.random()
    if r < 0.5:
        info["_earnings_estimate"] = {p: ROWS[int(rng.integers(0, len(ROWS)))] for p in ("0y", "+1y") if rng.random() < 0.8}
    elif r < 0.6:
        info["_earnings_estimate"] = None
    return info


@pytest.mark.parametrize("seed", range(5))
def test_growth_and_normalization(caplog, seed):
    rng = np.random.default_rng(19000 + seed)
    for trial in range(60):
        fin, bs, cf, _ppy, _prices = _case(rng)
        info = _info(rng)
        if trial % 11 == 0:
            fin = None
        for item in ("Net Income", "Total Revenue", "Nope"):
            _both(caplog, "estimate_growth_rate", fin, info, item)
            _both(caplog, "estimate_growth_rate", fin, None, item, 3)
            _both(caplog, "_historical_growth_regression", fin, item)
            _both(caplog, "_historical_growth_regression", fin, item, 3)
            _both(caplog, "blended_growth_estimate", fin, info, item)
        _both(caplog, "blended_growth_estimate", fin, None)
        _both(caplog, "_enrich_ticker_info", info, fin, bs, cf)
        for years in (5, 2, 10):
            _both(caplog, "estimate_fcf_margin", fin, cf, years)
            _both(caplog, "through_cycle_fcf_margin", fin, cf, years)
            _both(caplog, "normalized_base_fcf", info, fin, cf, years)
    _both(caplog, "estimate_growth_rate", None, {"_earnings_estimate": {"0y": [1]}})
    _both(caplog, "through_cycle_fcf_margin", None, None)


def test_mc_stats(caplog):
    rng = np.random.default_rng(19100)
    samples = [rng.normal(100, 20, 500), rng.normal(100, 20, 30), np.array([np.nan] * 100), rng.normal(-5, 1, 200),
               list(rng.normal(50, 5, 80)), np.array([1, 2, 3] * 30, dtype=np.int64), "x", None,
               np.concatenate([rng.normal(10, 1, 100), [np.inf, -np.inf, np.nan]])]  # fmt: skip
    for s in samples:
        _both(caplog, "_generate_mc_stats", s)


MC_CASES = {
    "run_monte_carlo_dcf": lambda r, info: (info, r(1e9), r(0.08), r(0.09), 10, 0.02, 600),
    "run_monte_carlo_dcfo": lambda r, info: (info, r(2e9), r(0.05), r(0.1), 8, 0.02, 600),
    "run_monte_carlo_dni": lambda r, info: (r(5.0), r(0.06), r(0.1), 10, 0.02, 600),
    "run_monte_carlo_mean_pe": lambda r, info: (r(4.0), r(18.0), 600),
    "run_monte_carlo_peg": lambda r, info: (r(4.0), r(12.0), r(1.5), 1.0, 600),
    "run_monte_carlo_mean_pb": lambda r, info: (r(30.0), r(2.0), 600),
    "run_monte_carlo_mean_ps": lambda r, info: (r(40.0), r(3.0), 600),
    "run_monte_carlo_psg": lambda r, info: (r(40.0), r(15.0), r(0.4), 1.0, 600),
    "run_monte_carlo_graham": lambda r, info: (r(4.0), r(8.0), r(4.4), 600),
    "run_monte_carlo_ddm": lambda r, info: (r(2.0), r(0.04), r(0.09), 10, 0.02, 600),
    "run_monte_carlo_epv": lambda r, info: (info, r(5e8), 0.21, r(0.09), 600),
    "run_monte_carlo_lynch": lambda r, info: (r(4.0), r(12.0), r(1.0), 600),
}


@pytest.mark.parametrize("name", sorted(MC_CASES))
def test_monte_carlo_wrappers(caplog, name):
    rng = np.random.default_rng(abs(hash(name)) % 2**32)
    odd = [0.0, -1.0, float("nan"), None, "x", np.float64(1.5), 0]
    for trial in range(25):
        def r(v):
            return odd[int(rng.integers(0, len(odd)))] if rng.random() < 0.15 else v * float(rng.uniform(0.5, 1.5))
        info = {"sharesOutstanding": [1e8, 0, None, -5, "x", np.float64(2e8)][int(rng.integers(0, 6))] if rng.random() < 0.4 else 1e8,
                "totalCash": [None, 0, 5e8][int(rng.integers(0, 3))], "totalDebt": [None, 0, 2e9][int(rng.integers(0, 3))]}  # fmt: skip
        args = MC_CASES[name](r, info)
        outs = []
        for mod in (ref, new):
            np.random.seed(trial)
            outs.append(call(lambda: getattr(mod, name)(*args), caplog))
            outs.append(canon(np.random.random()))  # the generator is left at the same point
        assert outs[:2] == outs[2:], (name, trial, args)


def test_through_cycle_path(caplog):
    rng = np.random.default_rng(19200)
    for trial in range(40):
        n = int(rng.integers(4, 16))
        cols = list(pd.date_range("2008-12-31", periods=n, freq="YE")[::-1])
        rev = rng.uniform(1e9, 2e9, n)
        fin = pd.DataFrame([rev], index=["Total Revenue"], columns=cols)
        ocf = rev * rng.uniform(-0.1, 0.4, n)
        cf = pd.DataFrame([ocf, -rev * rng.uniform(0, 0.1, n)], index=["Operating Cash Flow", "Capital Expenditure"], columns=cols)
        if trial % 4 == 0:
            cf.loc["Free Cash Flow"] = rev * rng.uniform(-0.2, 0.9, n)  # some implausible years
        info = {"totalRevenue": float(rev[0]) * [1, 0, -1, 1.2][trial % 4], "freeCashflow": [None, 5e8, -1][trial % 3], "symbol": "T"}
        for years in (5, 10, 3):
            _both(caplog, "through_cycle_fcf_margin", fin, cf, years)
            _both(caplog, "normalized_base_fcf", info, fin, cf, years)
            _both(caplog, "estimate_fcf_margin", fin, cf, years)
