"""Phase 15j: the twelve valuation models of financial_ratios in Rust
(rust/investa_core/src/fr_models.rs) against the original frozen in
tests/reference/financial_ratios_p15.py.

Every model runs over random statements and sparse, odd profiles — prices,
share counts, dividends, payout ratios, margins and sectors that are missing,
zero, negative, NaN or text — with each optional argument either left to the
model or supplied (ordinary, zero, negative, NumPy and text values), and with
multiple histories that are missing, empty, zero or complete. Results
(errors and diagnostics included), logs, warnings and the inputs left behind
agree.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from test_rust_ratio_glue_parity import _both  # noqa: E402
from test_rust_ratios_parity import _case  # noqa: E402

INFO_KEYS = ["totalRevenue", "totalCash", "totalDebt", "sharesOutstanding", "trailingEps", "currentPrice",
             "regularMarketPrice", "dividendRate", "trailingAnnualDividendRate", "dividendYield", "payoutRatio",
             "operatingCashflow", "bookValue", "grossMargins", "freeCashflow", "beta", "marketCap", "symbol"]  # fmt: skip
ODD = [None, 0, 0.0, -1.0, float("nan"), "x", np.float64(2.5)]


def _num(rng, scale):
    if rng.random() < 0.15:
        return ODD[int(rng.integers(0, len(ODD)))]
    return float(rng.lognormal(0, 1)) * scale


SCALES = {"totalRevenue": 5e9, "totalCash": 1e9, "totalDebt": 2e9, "sharesOutstanding": 1e8, "trailingEps": 3.0,
          "currentPrice": 60.0, "regularMarketPrice": 60.0, "dividendRate": 1.5, "trailingAnnualDividendRate": 1.5,
          "dividendYield": 0.02, "payoutRatio": 0.6, "operatingCashflow": 1e9, "bookValue": 20.0, "grossMargins": 0.4,
          "freeCashflow": 6e8, "beta": 1.1, "marketCap": 6e9}  # fmt: skip


def _info(rng):
    info = {k: _num(rng, SCALES[k]) for k in INFO_KEYS if k != "symbol" and rng.random() < 0.7}
    info["symbol"] = "T"
    info["sector"] = str(rng.choice(["Technology", "Financial Services", "Real Estate", "", "Regional Banks"]))
    if rng.random() < 0.3:
        info["payoutRatio"] = float(rng.uniform(1.0, 2.5))
    if rng.random() < 0.3:
        info["_earnings_estimate"] = {"0y": {"growth": float(rng.normal(0.1, 0.1))}}
    return info


def _history(rng):
    return [None, {}, {"value": 0}, {"value": float(rng.uniform(5, 30)), "n": 9, "span": "2014-2024", "low": 10.0, "high": 20.0},
            {"value": "12", "n": 4, "span": "2020-2024"}][int(rng.integers(0, 5))]  # fmt: skip


def _opt(rng, value):
    return None if rng.random() < 0.5 else (ODD[int(rng.integers(0, len(ODD)))] if rng.random() < 0.2 else value)


@pytest.mark.parametrize("seed", range(6))
def test_models_match(caplog, seed):
    rng = np.random.default_rng(20000 + seed)
    for trial in range(50):
        fin, bs, cf, _ppy, _prices = _case(rng)
        info = _info(rng)
        dr, gr = _opt(rng, float(rng.uniform(0.05, 0.15))), _opt(rng, float(rng.normal(0.08, 0.1)))
        years, tg = [10, 5, 1, 0][int(rng.integers(0, 4))], [0.02, 0.0, np.float64(0.025)][int(rng.integers(0, 3))]
        _both(caplog, "calculate_intrinsic_value_dcf", info, fin, bs, cf, dr, gr, years, tg, _opt(rng, 0.1), _opt(rng, 5e8))
        _both(caplog, "calculate_intrinsic_value_dcf", info, fin, bs, cf)
        _both(caplog, "calculate_intrinsic_value_graham", info, fin, bs, _opt(rng, 8.0), _opt(rng, 3.0), _opt(rng, 4.5))
        _both(caplog, "calculate_intrinsic_value_ddm", info, fin, cf, bs, dr, gr, years, tg, _opt(rng, 1.2))
        _both(caplog, "calculate_intrinsic_value_lynch", info, fin, _opt(rng, 12.0), _opt(rng, 3.0))
        _both(caplog, "calculate_earnings_power_value", info, fin, bs, cf, dr)
        _both(caplog, "calculate_intrinsic_value_dcfo", info, fin, bs, cf, dr, gr, years, tg, _opt(rng, 1e9))
        _both(caplog, "calculate_intrinsic_value_dcfo", info, fin, bs, None)
        _both(caplog, "calculate_intrinsic_value_dni", info, fin, bs, dr, gr, years, tg, _opt(rng, 3.0))
        _both(caplog, "calculate_intrinsic_value_mean_pe", info, fin, _opt(rng, 15.0), _opt(rng, 3.0), _history(rng))
        _both(caplog, "calculate_intrinsic_value_peg", info, fin, _opt(rng, 12.0), _opt(rng, 3.0), [1.0, 0.2, 3.0][trial % 3], _opt(rng, 1.0))
        _both(caplog, "calculate_intrinsic_value_mean_pb", info, bs, _opt(rng, 20.0), _opt(rng, 2.0), _history(rng))
        _both(caplog, "calculate_intrinsic_value_mean_ps", info, fin, bs, _opt(rng, 30.0), _opt(rng, 3.0), _history(rng))
        _both(caplog, "calculate_intrinsic_value_psg", info, fin, bs, _opt(rng, 30.0), _opt(rng, 15.0), [1.0, 0.1, 5.0][trial % 3])
