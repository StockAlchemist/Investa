"""A portfolio's value on one date, the way the performance graph computes it.

The chronological replay gives each day's holdings and cash, and the vectorized
valuation prices them. This is the history's own path (portfolio_history), so a
test that reads a value through it checks what the graph draws.
"""

from datetime import date
from typing import Dict, Optional

import pandas as pd

import config
import portfolio_history as ph


def value_on(
    transactions: pd.DataFrame,
    target: date,
    prices: Dict[str, pd.DataFrame],
    internal_to_yf: Dict[str, str],
    account_currency_map: Dict[str, str],
    cash_mode_map: Optional[Dict[str, str]] = None,
    currency: str = "USD",
) -> float:
    """The portfolio's value in `currency` at the end of `target`."""
    s2i, i2s, a2i, i2a, t2i, _, _ = ph.generate_mappings(transactions)
    start = transactions["Date"].min().date()
    qty, cash, last = all_daily_holdings(
        transactions, start, target, s2i, a2i, t2i, cash_mode_map
    )[:3]
    grid = pd.date_range(start, target, freq="D")
    values = ph._value_daily_holdings_vectorized(
        grid, qty, last, cash, prices, {}, currency, account_currency_map,
        i2s, i2a, internal_to_yf, set(a2i.values()), currency, "D",
    )[0]  # fmt: skip
    return float(pd.Series(values, index=grid).iloc[-1])


def all_daily_holdings(*args):
    """`_get_or_calculate_all_daily_holdings` with its file cache switched off,
    so a test ledger never lands in the app's real cache directory."""
    real = config.get_app_cache_dir
    config.get_app_cache_dir = lambda: None
    try:
        return ph._get_or_calculate_all_daily_holdings(*args)
    finally:
        config.get_app_cache_dir = real
