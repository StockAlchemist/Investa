"""The day change of a holding the price feed cannot move.

A fund priced from a published NAV, or a Thai position priced off its own last
trade, has no local price move to report — the feed publishes none. That is not
the same as "did not move": held in a foreign currency, the position is worth a
different amount today purely because FX moved, and the portfolio's headline
"today" figure is the sum of those amounts.

Dropping the row from the sum while its full market value stays in the total
(and in the denominator of the day-change percentage) is what made a $210k THB
book report +$1,713 on a day it had lost $1,915 of currency — a sign flip on
the headline.
"""

import os
import sys
from datetime import date

import pandas as pd
import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from portfolio_analyzer import _build_summary_rows  # noqa: E402

REPORT_DATE = date(2026, 8, 31)

# Units per USD. THB weakened 32.86 -> 33.16 overnight: a THB asset is worth
# 0.9047% less in USD today, having not moved a satang in its own currency.
FX_TODAY = {"USD": 1.0, "THB": 33.16}
FX_PREV = {"USD": 1.0, "THB": 32.86}
FX_ONLY_PCT = (32.86 / 33.16 - 1.0) * 100.0


def _transactions(symbol, account, rows):
    return pd.DataFrame(
        {
            "Date": [pd.Timestamp(d) for d, _q, _p in rows],
            "Symbol": [symbol] * len(rows),
            "Quantity": [q for _d, q, _p in rows],
            "Price/Share": [p for _d, _q, p in rows],
            "Type": ["Buy"] * len(rows),
            "Account": [account] * len(rows),
            "Commission": [0.0] * len(rows),
            "Total Amount": [q * p for _d, q, p in rows],
            "original_index": list(range(len(rows))),
        }
    )


def _row(*, display_currency, stock_data=None, published_navs=None, excluded=True):
    symbol = "SCBRMS&P500"
    account = "SCBAM"
    holdings = {
        (symbol, account): {
            "qty": 1000.0,
            "total_cost_local": 20000.0,
            "local_currency": "THB",
            "total_cost_display_historical_fx": 20000.0,
        }
    }
    rows, _, _, _ = _build_summary_rows(
        holdings=holdings,
        current_stock_data=stock_data or {},
        current_fx_rates_vs_usd=FX_TODAY,
        current_fx_prev_close_vs_usd=FX_PREV,
        display_currency=display_currency,
        default_currency="THB",
        transactions_df=_transactions(symbol, account, [("2026-07-27", 1000.0, 20.0)]),
        report_date=REPORT_DATE,
        shortable_symbols=set(),
        user_excluded_symbols={symbol} if excluded else set(),
        user_symbol_map={},
        manual_prices_dict={},
        published_navs=published_navs or {},
    )
    return rows[0]


def test_unpriced_foreign_holding_still_carries_its_fx_move():
    """No local price move known, but the currency moved — so the value did."""
    row = _row(
        display_currency="USD",
        published_navs={"SCBRMS&P500": ("2026-08-27", 23.0637)},
    )
    market_value = row["Market Value (USD)"]
    assert market_value == pytest.approx(1000.0 * 23.0637 / 33.16)

    day_change = row["Day Change (USD)"]
    assert day_change == pytest.approx(market_value * (1.0 - 33.16 / 32.86))
    assert day_change < 0
    assert row["Day Change %"] == pytest.approx(FX_ONLY_PCT)


def test_unpriced_holding_in_its_own_currency_reports_nothing():
    """Same holding, THB view: nothing is known, and no FX leg exists to know."""
    row = _row(
        display_currency="THB",
        published_navs={"SCBRMS&P500": ("2026-08-27", 23.0637)},
    )
    assert row["Market Value (THB)"] == pytest.approx(1000.0 * 23.0637)
    assert pd.isna(row["Day Change (THB)"])
    assert pd.isna(row["Day Change %"])


def test_priced_foreign_holding_compounds_its_price_move_with_fx():
    """The ordinary path is unchanged: both legs, measured in the display currency."""
    row = _row(
        display_currency="USD",
        excluded=False,
        stock_data={
            "SCBRMS&P500": {
                "price": 23.0637,
                "change": 0.2637,  # 22.80 -> 23.0637 in THB
                "changesPercentage": 1.1566,
            }
        },
    )
    val_today = 1000.0 * 23.0637 / 33.16
    val_yesterday = 1000.0 * 22.80 / 32.86
    assert row["Day Change (USD)"] == pytest.approx(val_today - val_yesterday)
    assert row["Day Change %"] == pytest.approx(
        (val_today / val_yesterday - 1.0) * 100.0
    )
