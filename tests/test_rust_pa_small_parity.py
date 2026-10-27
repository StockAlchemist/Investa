"""Phase 13a: corporate_actions and portfolio_analyzer's standalone helpers
in Rust (rust/investa_core/src/ca.rs, pa_small.rs) against the originals
frozen in tests/reference/.

Every function takes random inputs — ints, floats, NumPy scalars, negatives
and NaN for the arithmetic; real and near-miss IBKR spin-off descriptions;
split frames with "All Accounts" and per-account duplicates across months and
ratios; summary frames and cash rows — and must return the same values, of
the same types, with the same logs and warnings.
"""

import datetime as dt
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import corporate_actions as new_ca  # noqa: E402
import portfolio_analyzer as new_pa  # noqa: E402
from reference import corporate_actions as ref_ca  # noqa: E402
from reference import portfolio_analyzer as ref_pa  # noqa: E402
from test_rust_market_db_parity import call  # noqa: E402

NUMS = [
    0,
    3,
    10,
    -2,
    0.0,
    1e-10,
    5.5,
    120.25,
    np.float64(7.5),
    np.int64(4),
    float("nan"),
]
DESCS = [
    "SPGI(US78409V1044) Spinoff 1 for 1 (MBGL, MOBILITY GLOBAL INC, US60744M1062)",
    "GE(US3696043013) Spin-off 1 for 4 (GEHC, GE HEALTHCARE, US36266G1076)",
    "ABC(US1) Spin off (ABC, SAME, X)",
    "xyz(US1) Spinoff 1 for 1 (NEW, N, I)",
    "AAPL(US0378331005) Cash Dividend",
    "",
    None,
    "T(US1) SPIN OFF 2 for 3 (WBD, WARNER, US2) (lower, x)",
]


def check(a, b, label):
    assert a == b, (label, a, b)


@pytest.mark.parametrize("seed", range(4))
def test_corporate_actions_match(caplog, seed):
    rng = np.random.default_rng(9100 + seed)
    for _ in range(80):
        x, y, z, w = (NUMS[int(i)] for i in rng.integers(0, len(NUMS), 4))
        for name, args in (
            ("apply_return_of_capital", (x, y, z)),
            ("apply_stock_dividend", (x, y, z)),
            ("apply_spin_off", (x, y, z, w)),
        ):
            check(
                call(lambda: getattr(ref_ca, name)(*args), caplog),
                call(lambda: getattr(new_ca, name)(*args), caplog),
                name,
            )
        desc = DESCS[int(rng.integers(0, len(DESCS)))]
        check(
            call(lambda: ref_ca.parse_spinoff_description(desc), caplog),
            call(lambda: new_ca.parse_spinoff_description(desc), caplog),
            desc,
        )
        args = (
            "SPGI",
            "MBGL",
            x,
            "2025-07-01",
            "IBKR",
            7,
            [None, 0, 1500.0, -3, "12"][int(rng.integers(0, 5))],
            "1 for 1" + "x" * int(rng.integers(0, 120)),
            "USD",
        )
        check(
            call(lambda: ref_ca.build_spinoff_legs(*args), caplog),
            call(lambda: new_ca.build_spinoff_legs(*args), caplog),
            "legs",
        )
    for _ in range(40):
        n = int(rng.integers(0, 14))
        df = pd.DataFrame(
            {
                "Date": pd.to_datetime("2024-01-01")
                + pd.to_timedelta(rng.integers(0, 90, n), unit="D"),
                "Type": rng.choice(["split", "Stock Split ", "buy", "SPLIT"], n),
                "Symbol": rng.choice(["AAPL", "NVDA"], n),
                "Account": rng.choice(["All Accounts", "IBKR", " all accounts "], n),
                "Split Ratio": rng.choice([2, 2.0, 2.0000001, 4.0, np.nan], n),
                "original_index": rng.permutation(n),
            }
        )
        if rng.random() < 0.2:
            df = df.drop(columns=["original_index"])
        if rng.random() < 0.1:
            df = df.drop(columns=["Account"])
        check(
            call(lambda: ref_ca.deduplicate_split_transactions(df.copy()), caplog),
            call(lambda: new_ca.deduplicate_split_transactions(df.copy()), caplog),
            "dedup",
        )


def summary(rng):
    n = int(rng.integers(0, 8))
    return pd.DataFrame(
        {
            "Symbol": list(rng.choice(["AAPL", "MSFT", "$CASH", "Total", "KO"], n)),
            "Account": list(rng.choice(["IBKR", "SCB"], n)),
            "Market Value (USD)": list(
                rng.choice([0.0, 1000.0, 2500.5, -10.0, np.nan], n)
            ),
            "Price (USD)": list(rng.choice([0.0, 50.0, 199.99], n)),
            "is_total": list(rng.choice([True, False, None], n)),
        }
    )


@pytest.mark.parametrize("seed", range(4))
def test_analyzer_helpers_match(caplog, seed):
    rng = np.random.default_rng(9200 + seed)
    for _ in range(40):
        t = str(
            rng.choice(
                [
                    "Buy",
                    " SELL ",
                    "fee",
                    "Stock Split",
                    "spin-off",
                    "Spinoff",
                    "merger",
                    "",
                    "Interest",
                    "withholding tax",
                    "transfer",
                ]
            )
        )
        check(
            call(lambda: ref_pa.get_type_id(t), caplog),
            call(lambda: new_pa.get_type_id(t), caplog),
            t,
        )
        ret = pd.DataFrame(
            rng.normal(0, 0.01, (int(rng.integers(0, 30)), 3)), columns=["a", "b", "c"]
        )
        if rng.random() < 0.2:
            ret["c"] = "x"
        check(
            call(lambda: ref_pa.calculate_correlation_matrix(ret), caplog),
            call(lambda: new_pa.calculate_correlation_matrix(ret), caplog),
            "corr",
        )
        betas, shocks = (
            {"Mkt-RF": float(rng.normal(1, 0.2)), "SMB": np.float64(0.3)},
            {"Mkt-RF": -0.1, "HML": 0.05, "SMB": 1},
        )
        check(
            call(lambda: ref_pa.run_scenario_analysis(betas, shocks, 1e6), caplog),
            call(lambda: new_pa.run_scenario_analysis(betas, shocks, 1e6), caplog),
            "scenario",
        )
        s = summary(rng)
        target = {"AAPL": 40, "MSFT": 30.5, "$CASH": 10, "NEW": 19.5}
        cash = float(rng.choice([0, 5000]))
        check(
            call(
                lambda: ref_pa.calculate_rebalancing_trades(s, target, cash, "USD"),
                caplog,
            ),
            call(
                lambda: new_pa.calculate_rebalancing_trades(s, target, cash, "USD"),
                caplog,
            ),
            "rebal",
        )
        w = list(
            rng.choice([0.0, 1.0, 2.5, np.float64(3.0), 7], int(rng.integers(0, 6)))
        )
        check(
            call(lambda: ref_pa.calculate_hhi(w), caplog),
            call(lambda: new_pa.calculate_hhi(w), caplog),
            "hhi",
        )
        risk = {
            "Sharpe Ratio": float(rng.choice([-0.5, 0.0, 0.7, 1.2, 2.0])),
            "Volatility (Ann.)": np.float64(rng.choice([0.0, 0.03, 0.15, 0.4])),
        }
        check(
            call(lambda: ref_pa.calculate_health_score(s, risk), caplog),
            call(lambda: new_pa.calculate_health_score(s, risk), caplog),
            "health",
        )
        rows = [{"Symbol": str(sym), "Account": acc, "Market Value (USD)": mv, "Name": nm}
                for sym, acc, mv, nm in zip(rng.choice(["$CASH", "Cash (THB)", "AAPL", "Cash X"], 5), rng.choice(["IBKR", "SCB", ""], 5), rng.choice([100000.0, 500.0, 5, True], 5), rng.choice(["Cash", "Apple"], 5))]  # fmt: skip
        rates, thresholds = {"IBKR": 4.5, "SCB": 0, "X": -1}, {"IBKR": 1000.0}
        end = dt.date.today() + dt.timedelta(days=int(rng.integers(0, 400)))
        check(
            call(
                lambda: ref_pa.generate_cash_interest_events(
                    rows, rates, thresholds, dt.date(2024, 1, 1), end
                ),
                caplog,
            ),
            call(
                lambda: new_pa.generate_cash_interest_events(
                    rows, rates, thresholds, dt.date(2024, 1, 1), end
                ),
                caplog,
            ),
            "interest",
        )
