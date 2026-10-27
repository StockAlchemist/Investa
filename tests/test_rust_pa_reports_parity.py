"""Phase 13b: portfolio_analyzer's report extractors in Rust
(rust/investa_core/src/pa_reports.rs) against the originals frozen in
tests/reference/portfolio_analyzer.py.

- Periodic returns on random histories: DatetimeIndex and RangeIndex (which
  the function re-indexes in place), spans crossing weekends and US holidays,
  with and without the daily gain, net flow and benchmark columns.
- Dividend history on random ledgers: dividends, $CASH interest and withheld
  tax, account filters, amounts that need Quantity x Price, text amounts, a
  missing FX table.
- Realized gains on random ledgers, filtered by account and by dates, with a
  Date column that is not datetime64 (so the reassignment runs on a slice).
Results, logs and warnings must agree; the input frames must be left alike.
"""

import datetime as dt
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


def history(rng):
    n = int(rng.integers(1, 120))
    start = pd.Timestamp(str(rng.choice(["2023-12-20", "2024-06-28", "2024-11-25"])))
    idx = pd.date_range(start, periods=n, freq="D")
    df = pd.DataFrame(
        {"Portfolio Accumulated Gain": np.cumprod(1 + rng.normal(0, 0.01, n))},
        index=idx,
    )
    if rng.random() < 0.7:
        df["SPY Accumulated Gain"] = np.cumprod(1 + rng.normal(0, 0.01, n))
    if rng.random() < 0.6:
        df["Portfolio Daily Gain" if rng.random() < 0.5 else "daily_gain"] = rng.normal(
            0, 100, n
        )
    if rng.random() < 0.6:
        df["Cumulative Net Flow"] = np.cumsum(rng.choice([0, 0, 1000.0, -500.0], n))
    if rng.random() < 0.15:
        df = df.reset_index(drop=True)
    return df


def ledger(rng):
    n = int(rng.integers(0, 30))
    df = pd.DataFrame(
        {
            "Date": pd.to_datetime("2023-01-02")
            + pd.to_timedelta(np.sort(rng.integers(0, 500, n)), unit="D"),
            "Type": rng.choice(
                ["Buy", "Sell", "Dividend", "tax", "Interest", "deposit"], n
            ),
            "Symbol": rng.choice(["AAPL", "KO", "$CASH"], n),
            "Account": rng.choice(["IBKR", "SCBS"], n),
            "Quantity": rng.choice([1.0, 5.0, 10.0, np.nan], n),
            "Price/Share": rng.choice([10.0, 55.5, np.nan], n),
            "Total Amount": rng.choice([100.0, 12.5, np.nan, -3.0], n),
            "Commission": rng.choice([0.0, 1.0, np.nan], n),
            "Local Currency": rng.choice(["USD", "THB"], n),
            "original_index": np.arange(n),
        }
    )
    if n and rng.random() < 0.2:
        df["Total Amount"] = df["Total Amount"].astype(object)
        df.loc[df.index[0], "Total Amount"] = "abc"
    return df


FX = {
    "THB=X": pd.DataFrame(
        {"price": [35.0, 36.0]}, index=pd.to_datetime(["2023-01-01", "2024-01-01"])
    )
}


@pytest.mark.parametrize("seed", range(5))
def test_reports_match(caplog, seed):
    rng = np.random.default_rng(9300 + seed)
    for _ in range(25):
        h = history(rng)
        a, b = h.copy(), h.copy()
        assert call(
            lambda: ref.calculate_periodic_returns(a, ["SPY", "QQQ"]), caplog
        ) == call(lambda: new.calculate_periodic_returns(b, ["SPY", "QQQ"]), caplog)
        assert canon(a) == canon(b)
        txs = ledger(rng)
        fx = FX if rng.random() < 0.8 else {}
        accounts = [None, ["IBKR"], ["NONE"]][int(rng.integers(0, 3))]
        ccy = str(rng.choice(["USD", "THB"]))
        assert call(
            lambda: ref.extract_dividend_history(txs.copy(), ccy, fx, "USD", accounts),
            caplog,
        ) == call(
            lambda: new.extract_dividend_history(txs.copy(), ccy, fx, "USD", accounts),
            caplog,
        )
        if rng.random() < 0.3:
            txs["Date"] = txs["Date"].astype(object)
        lo, hi = (
            [None, dt.date(2023, 6, 1)][int(rng.integers(0, 2))],
            [None, dt.date(2024, 1, 31)][int(rng.integers(0, 2))],
        )
        g_accounts = [None, ["ibkr"], [" scbs "]][int(rng.integers(0, 3))]
        assert call(
            lambda: ref.extract_realized_capital_gains_history(
                txs.copy(),
                ccy,
                fx,
                "USD",
                set(),
                include_accounts=g_accounts,
                from_date=lo,
                to_date=hi,
            ),
            caplog,
        ) == call(
            lambda: new.extract_realized_capital_gains_history(
                txs.copy(),
                ccy,
                fx,
                "USD",
                set(),
                include_accounts=g_accounts,
                from_date=lo,
                to_date=hi,
            ),
            caplog,
        )
