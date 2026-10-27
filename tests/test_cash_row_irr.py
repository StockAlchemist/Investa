"""A cash row has no IRR.

A cash balance's only external flows are deposits and withdrawals; the sale
proceeds and purchases that make up most of what passes through it happen
inside the account. So an account funded by transferring stock in looks, from
the cash row alone, like a few hundred dollars deposited and $20,000 taken
out. IBKR Atcha did exactly that and logged an "IRR" of 59,551%. The account's
own money-weighted return is the meaningful figure; the cash row shows none.
"""

import os
import sys
from datetime import date

import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from portfolio_analyzer import _build_summary_rows  # noqa: E402

ACCOUNT = "IBKR Atcha"


def _cash_ledger():
    # $690 deposited, $20,050 withdrawn (proceeds of stock sold in the account,
    # which the cash row's flows never see), $57 left.
    rows = [
        ("2025-10-22", "Deposit", 29.18),
        ("2025-11-10", "Deposit", 129.59),
        ("2025-11-14", "Deposit", 118.23),
        ("2025-11-28", "Deposit", 358.53),
        ("2026-05-27", "Withdrawal", 50.00),
        ("2026-06-04", "Withdrawal", 20000.00),
    ]
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d, _t, _a in rows],
            "Symbol": ["$CASH"] * len(rows),
            "Quantity": [a for _d, _t, a in rows],
            "Price/Share": [1.0] * len(rows),
            "Type": [t for _d, t, _a in rows],
            "Account": [ACCOUNT] * len(rows),
            "Commission": [0.0] * len(rows),
            "Total Amount": [a if t == "Deposit" else -a for _d, t, a in rows],
            "Local Currency": ["USD"] * len(rows),
            "original_index": list(range(len(rows))),
        }
    )


def test_a_cash_row_reports_no_irr():
    holdings = {
        ("$CASH", ACCOUNT): {
            "qty": 56.95,
            "total_cost_local": 56.95,
            "local_currency": "USD",
            "total_cost_display_historical_fx": 56.95,
        }
    }
    rows, _, _, _ = _build_summary_rows(
        holdings=holdings,
        current_stock_data={},
        current_fx_rates_vs_usd={"USD": 1.0},
        current_fx_prev_close_vs_usd={"USD": 1.0},
        display_currency="USD",
        default_currency="USD",
        transactions_df=_cash_ledger(),
        report_date=date(2026, 10, 23),
        shortable_symbols=set(),
        user_excluded_symbols=set(),
        user_symbol_map={},
        manual_prices_dict={},
    )
    (cash,) = [r for r in rows if r["Account"] == ACCOUNT]
    assert cash["Symbol"] == "Cash ($)"
    assert pd.isna(cash["IRR (%)"])
