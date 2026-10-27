"""Phase 13d: portfolio_analyzer._build_summary_rows in Rust
(rust/investa_core/src/pa_rows.rs) against the original frozen in
tests/reference/portfolio_analyzer.py.

Holdings are built by hand so every number type the rows meet is covered:
NumPy floats, Python ints and floats, NaN costs, negative and zero
quantities, short positions (whose "Avg Cost (Local)" reads the cost basis the
previous long holding left, or raises when none did), cash in two currencies.
Market data mixes valid, zero, text and NumPy-integer prices, missing changes
and dividend fields; FX tables miss a currency; published NAVs, manual prices
(valid, zero, text), interest rates with thresholds and account filters are
drawn at random. Rows, flags, logs down to DEBUG (every float formatted into a
message) and warnings must agree, and so must the ledger left behind.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import portfolio_analyzer as new  # noqa: E402
from reference import portfolio_analyzer as ref  # noqa: E402
from test_rust_market_db_parity import call, canon  # noqa: E402
from test_summary_rows_parity import ACCOUNTS, REPORT, _ledger  # noqa: E402

SYMBOLS = ["AAPL", "MSFT", "PTT.BK", "NOFEED", "FUND", "GONE", "$CASH", "Cash ($)"]
CCY = ["USD", "THB", "EUR", "GBP"]


def _num(rng, choices):
    v = choices[int(rng.integers(0, len(choices)))]
    return v() if callable(v) else v


def _holdings(rng):
    out = {}
    for _ in range(int(rng.integers(0, 14))):
        sym = str(rng.choice(SYMBOLS))
        acc = str(rng.choice(ACCOUNTS + ["ibkr "]))
        data = {
            "qty": _num(rng, [np.float64(10.0), 5, 0.0, -3.0, np.float64(-2.5), 1e-12, 40.0, np.nan]),
            "realized_gain_local": _num(rng, [0.0, np.float64(12.5), -4, np.nan]),
            "dividends_local": _num(rng, [0.0, np.float64(3.25), 7]),
            "commissions_local": _num(rng, [0.0, np.float64(1.0), 2]),
            "taxes_local": _num(rng, [0.0, np.float64(0.5)]),
            "local_currency": str(rng.choice(CCY)),
            "total_cost_local": _num(rng, [np.float64(1000.0), -5.0, 0, np.nan, 250.0]),
            "short_proceeds_local": _num(rng, [0.0, np.float64(300.0), 0]),
            "total_cost_invested_local": _num(rng, [np.float64(900.0), 0.0]),
            "cumulative_investment_local": _num(rng, [np.float64(1100.0), 0.0]),
            "total_buy_cost_local": _num(rng, [np.float64(1200.0), 0.0, 0]),
            "total_cost_display_historical_fx": _num(rng, [np.nan, np.float64(950.0), 0.0]),
            "realized_gain_display": _num(rng, [np.nan, np.float64(11.0)]),
            "tags": ["core"] if rng.random() < 0.3 else [],
        }
        for k in list(data):
            if rng.random() < 0.05:
                del data[k]  # the row's .get() default
        out[(sym, acc)] = data
    return out


def _market(rng):
    data = {}
    for sym in SYMBOLS:
        if sym in ("NOFEED", "FUND") or rng.random() < 0.15:
            continue
        data[sym] = {
            "price": _num(rng, [lambda: float(rng.uniform(10, 300)), 0.0, 120, np.float64(55.5), np.int64(80), "12", None]),
            "change": _num(rng, [lambda: float(rng.normal(0, 2)), None, np.nan, np.float64(-1.25)]),
            "changesPercentage": _num(rng, [lambda: float(rng.normal(0, 1)), None]),
            "dividendRate": _num(rng, [0.0, 0.96, 2.4, None]),
            "dividendYield": _num(rng, [0.0, 0.005, 0.03, 2.5, None]),
            "name": "Name " + sym,
            "exchange": "NMS" if rng.random() < 0.5 else None,
        }
        if rng.random() < 0.2:
            data[sym].pop("price")
    return data


@pytest.mark.parametrize("seed", range(10))
def test_summary_rows_match(caplog, seed):
    rng = np.random.default_rng(13400 + seed)
    for trial in range(30):
        df = _ledger(rng)
        r = rng.random()
        if r < 0.1:
            df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        elif r < 0.2:
            df = df.drop(columns=["To Account"])
        elif r < 0.3:
            df["Price/Share"] = df["Price/Share"].map(lambda v: "n/a" if v == 0 else v)
        holdings = _holdings(rng)
        fx = {"USD": 1.0, "THB": 35.0, "EUR": 0.9}
        fx_prev = {"USD": 1.0, "THB": 35.4} if rng.random() < 0.5 else {}
        navs = (
            {"FUND": ("2024-06-13", np.float64(10.5)), "NOFEED": ("2024-06-12", 3)}
            if rng.random() < 0.4
            else None
        )
        manual = _num(rng, [{}, {"NOFEED": 42.0}, {"NOFEED": 0}, {"GONE": "9"}, {"FUND": 7}])
        rates = (
            {"IBKR": 4.0, "SCBS": 0.0, "Kim Eng": 2.5} if rng.random() < 0.5 else None
        )
        thresholds = {"IBKR": 1000.0} if rng.random() < 0.5 else None
        include = (
            None
            if rng.random() < 0.7
            else list(rng.choice(ACCOUNTS + [" ibkr"], 2, replace=False))
        )
        excluded = {"GONE"} if rng.random() < 0.3 else set()
        display = str(rng.choice(["USD", "THB"]))
        a, b = df.copy(), df.copy()

        def run(mod, txs):
            return mod._build_summary_rows(
                holdings, _market(np.random.default_rng(trial)), fx, fx_prev, display,
                "USD", txs, REPORT, set(), excluded, {}, manual,
                published_navs=navs, account_interest_rates=rates,
                interest_free_thresholds=thresholds, include_accounts=include,
            )  # fmt: skip

        got = call(lambda: run(new, b), caplog)
        want = call(lambda: run(ref, a), caplog)
        assert got == want, (seed, trial)
        assert canon(a) == canon(b), (seed, trial, "ledger left behind")


def test_short_holding_reads_the_previous_cost_basis(caplog):
    """The original's leaked loop variable, kept: see the port doc."""
    long = {"qty": np.float64(10.0), "total_cost_local": np.float64(500.0), "local_currency": "USD"}
    short = {"qty": np.float64(-4.0), "short_proceeds_local": 200.0, "local_currency": "USD"}
    df = _ledger(np.random.default_rng(1))
    market = {"AAA": {"price": 50.0}, "BBB": {"price": 20.0}}
    for holdings in ({("AAA", "IBKR"): long, ("BBB", "IBKR"): short}, {("BBB", "IBKR"): short}):
        args = (holdings, market, {"USD": 1.0}, {"USD": 1.0}, "USD", "USD", df, REPORT,
                set(), set(), {}, {})  # fmt: skip
        got = call(lambda: new._build_summary_rows(*args), caplog)
        want = call(lambda: ref._build_summary_rows(*args), caplog)
        assert got == want
    assert got[0][0] == "raised" and got[0][1] == "UnboundLocalError"


@pytest.mark.parametrize("seed", range(8))
def test_aggregate_metrics_match(caplog, seed):
    """`_calculate_aggregate_metrics` over the rows above, with columns dropped
    or made text, every shape of account filter, and ledgers with and without
    a To Account column."""
    rng = np.random.default_rng(13500 + seed)
    for trial in range(25):
        df = _ledger(rng)
        holdings = _holdings(rng)
        display = str(rng.choice(["USD", "THB"]))
        try:
            rows, *_ = new._build_summary_rows(
                {k: v for k, v in holdings.items() if not v.get("qty", 0) < 0},  # no shorts
                _market(rng), {"USD": 1.0, "THB": 35.0, "EUR": 0.9}, {"USD": 1.0, "THB": 35.4},
                display, "USD", df, REPORT, set(), set(), {}, {},
            )  # fmt: skip
        except Exception:
            continue
        frame = pd.DataFrame(rows)
        r = rng.random()
        if r < 0.1 and len(frame.columns):
            frame = frame.drop(columns=[str(rng.choice(list(frame.columns)))])
        elif r < 0.2 and f"Market Value ({display})" in frame:
            frame[f"Market Value ({display})"] = frame[f"Market Value ({display})"].astype(object)
            frame.loc[frame.index[:1], f"Market Value ({display})"] = "n/a"
        elif r < 0.25:
            frame = frame.drop(columns=[c for c in ("Symbol", "Name") if c in frame])
        elif r < 0.3:
            frame = frame.iloc[0:0]
        accounts_seen = sorted({str(a) for a in frame.get("Account", [])})
        include = _num(rng, [None, [], ACCOUNTS[:2], list(ACCOUNTS), [" ibkr", "SCBS"], accounts_seen])
        all_accounts = _num(rng, [None, list(ACCOUNTS), accounts_seen])
        txs = _num(rng, [None, df, df.drop(columns=["To Account"]), df.iloc[0:0]])
        hist = None if rng.random() < 0.5 else {"THB=X": pd.DataFrame({"price": [35.0]}, index=[REPORT])}

        def run(mod, f, t):
            return mod._calculate_aggregate_metrics(
                f, display, REPORT, fx_rates={"USD": 1.0, "THB": 35.0, "EUR": 0.9},
                include_accounts=include, all_available_accounts=all_accounts,
                transactions_df=t, historical_fx_rates=hist,
            )  # fmt: skip

        fa, fb = frame.copy(), frame.copy()
        ta, tb = (None, None) if txs is None else (txs.copy(), txs.copy())
        got = call(lambda: run(new, fb, tb), caplog)
        want = call(lambda: run(ref, fa, ta), caplog)
        assert got == want, (seed, trial)
        assert canon(fa) == canon(fb) and (ta is None or canon(ta) == canon(tb)), (seed, trial)
