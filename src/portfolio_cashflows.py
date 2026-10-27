"""Daily net external cash-flow calculation (vectorized + reference implementation).

Split from portfolio_logic.py.
"""

# ruff: noqa: E402
from datetime import date
from typing import Dict, List, Optional, Tuple

import investa_core
import pandas as pd

from config import CASH_SYMBOL_CSV  # noqa: F401 -- read by the Rust port at call time
from finutils import get_historical_rate_via_usd_bridge, is_cash_symbol  # noqa: F401 -- read by the Rust port at call time

try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func


def _calculate_daily_net_cash_flow_vectorized(
    date_range: pd.DatetimeIndex,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,
    internal_to_yf_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.Series, bool]:
    """
    Vectorized calculation of daily net external cash flow (deposits/withdrawals + transfers).
    Returns a Series indexed by Date with the net flow in target_currency.
    """
    # Ported to Rust: rust/investa_core/src/pc_flows.rs
    return investa_core.pc_daily_net_cash_flow_vectorized(
        date_range,
        transactions_df,
        target_currency,
        historical_fx_yf,
        default_currency,
        included_accounts,
        historical_prices_yf_unadjusted,
        internal_to_yf_map,
    )



def _calculate_daily_net_cash_flow(
    target_date: date,
    transactions_df: pd.DataFrame,
    target_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    account_currency_map: Dict[str, str],
    default_currency: str,
    processed_warnings: set,
    included_accounts: Optional[List[str]] = None,
    historical_prices_yf_unadjusted: Optional[Dict[str, pd.DataFrame]] = None,  # ADDED
    internal_to_yf_map: Optional[Dict[str, str]] = None,  # ADDED
) -> Tuple[float, bool]:
    """
    Calculates the net external cash flow for a specific date in the target currency.
    Now includes ASSET TRANSFERS in/out of the included accounts as flows.
    """
    # Ported to Rust: rust/investa_core/src/pc_flows.rs
    return investa_core.pc_daily_net_cash_flow(
        target_date,
        transactions_df,
        target_currency,
        historical_fx_yf,
        account_currency_map,
        default_currency,
        processed_warnings,
        included_accounts,
        historical_prices_yf_unadjusted,
        internal_to_yf_map,
    )
