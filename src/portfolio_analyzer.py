# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          portfolio_analyzer.py
 Purpose:       Contains functions for calculating the current portfolio state,
                holdings, cash balances, and summary metrics from transactions
                and market data. (Refactored from portfolio_logic.py)

 Author:        Google Gemini (Derived from portfolio_logic.py)


 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
import logging
from datetime import (
    date,
    datetime,  # noqa: F401 -- read by the Rust port at call time
)

# --- ADDED: Import line_profiler if available, otherwise create dummy decorator ---
try:
    from line_profiler import profile
except ImportError:

    def profile(func):
        return func  # No-op decorator if line_profiler not installed


# --- END ADDED ---
from typing import List, Tuple, Dict, Optional, Any, Set
import investa_core

# Read by the Rust port at call time (portfolio_analyzer.USFederalHolidayCalendar).
from pandas.tseries.holiday import USFederalHolidayCalendar  # noqa: F401

# --- Import Configuration ---
try:
    from config import (
        CASH_SYMBOL_CSV,
        SHORTABLE_SYMBOLS,  # Used by _process_transactions...
        STOCK_QUANTITY_CLOSE_TOLERANCE,  # New tolerance
        _AGGREGATE_CASH_ACCOUNT_NAME_,
        # Add any other config constants used by these specific functions if needed
    )
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import from config.py in portfolio_analyzer.py."
    )
    # Define fallbacks if absolutely necessary
    CASH_SYMBOL_CSV = "$CASH"
    SHORTABLE_SYMBOLS = set()
    STOCK_QUANTITY_CLOSE_TOLERANCE = 1e-6  # Fallback if config fails
    _AGGREGATE_CASH_ACCOUNT_NAME_ = "Cash"

# --- Import Utilities ---
try:
    from finutils import (
        is_cash_symbol,  # noqa: F401 -- read by the Rust port at call time
        get_conversion_rate,  # Used by _build_summary_rows
        calculate_irr,  # Used by _build_summary_rows
        CashFlowLedger,  # noqa: F401 -- read by the Rust port at call time
        safe_sum,  # noqa: F401 -- read by the Rust port at call time
        get_currency_symbol_from_code,  # noqa: F401 -- read by the Rust port at call time
        get_historical_rate_via_usd_bridge,  # noqa: F401 -- read by the Rust port at call time
        get_cash_flows_for_mwr,  # noqa: F401 -- read by the Rust port at call time
        calculate_indicated_dividend,  # noqa: F401 -- read by the Rust port at call time
    )
except ImportError:
    logging.critical(
        "CRITICAL ERROR: Could not import from finutils.py in portfolio_analyzer.py."
    )

    # Define dummy functions if needed for structure, but fixing is better
    def get_conversion_rate(*args):
        return np.nan

    def calculate_irr(*args):
        return np.nan

    def get_cash_flows_for_symbol_account(*args):
        return [], []

    def safe_sum(*args):
        return 0.0


# --- Numba Constants ---
TYPE_BUY = 0
TYPE_SELL = 1
TYPE_DEPOSIT = 2
TYPE_WITHDRAWAL = 3
TYPE_DIVIDEND = 4
TYPE_FEES = 5
TYPE_SPLIT = 6
TYPE_TRANSFER = 7
TYPE_SHORT_SELL = 8
TYPE_BUY_TO_COVER = 9
TYPE_TAX = 10
TYPE_INTEREST = 11
TYPE_SPIN_OFF = 12
TYPE_UNKNOWN = -1

# The Rust replay (investa_core.process_core) hard-codes these numbers. A
# mismatch would silently book every row as the wrong type, so refuse to load.
_rust_mismatch = [
    _name
    for _name, _value in vars(investa_core.analyzer_types).items()
    if _name.startswith("TYPE_") and globals().get(_name) != _value
]
if _rust_mismatch:
    raise ImportError(
        f"portfolio_analyzer TYPE_* constants disagree with investa_core: {_rust_mismatch}"
    )


# --- Helper to map string types to ints ---
def get_type_id(t):
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_get_type_id(t)


def _process_numba_core(
    sym_ids,
    acc_ids,
    type_ids,
    qtys,
    prices,
    comms,
    totals,
    split_ratios,
    to_acc_ids,
    local_curr_ids,
    fx_rates_hist,
    shortable_sym_ids,
    stock_qty_tol,
    cash_sym_id,
    num_tx,
    num_syms,
    num_accs,
    acc_cash_modes,
):
    """Replay the ledger into per-position state.

    Runs in Rust: investa_core.process_core, a line-for-line port of the Numba
    kernel that used to live here (frozen in tests/reference/). Returns
    (state[num_syms, num_accs, 14], transfer_costs[num_tx]). The 14 state
    fields, in order: qty, total_cost_local, realized_gain_local,
    dividends_local, commissions_local, short_proceeds_local,
    short_original_qty, total_cost_invested_local, cumulative_investment_local,
    total_buy_cost_local, total_cost_display_historical_fx, local_curr_id (as
    float, -1 = untouched), realized_gain_display, taxes_local. transfer_costs
    holds the unit cost a transfer carried, NaN elsewhere.
    """
    i64 = lambda a: np.ascontiguousarray(a, dtype=np.int64)  # noqa: E731
    f64 = lambda a: np.ascontiguousarray(a, dtype=np.float64)  # noqa: E731
    return investa_core.process_core(
        i64(sym_ids),
        i64(acc_ids),
        i64(type_ids),
        f64(qtys),
        f64(prices),
        f64(comms),
        f64(totals),
        f64(split_ratios),
        i64(to_acc_ids),
        i64(local_curr_ids),
        f64(fx_rates_hist),
        i64(list(shortable_sym_ids)),
        float(stock_qty_tol),
        int(cash_sym_id),
        int(num_tx),
        int(num_syms),
        int(num_accs),
        i64(acc_cash_modes),
    )


@profile
def _process_transactions_to_holdings(
    transactions_df: pd.DataFrame,
    default_currency: str,
    shortable_symbols: Set[str],
    historical_fx_lookup: Dict[Tuple[date, str], float],
    display_currency_for_hist_fx: str,
    report_date: date,
    account_cash_mode_map: Optional[Dict[str, str]] = None,
) -> Tuple[
    Dict[Tuple[str, str], Dict],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    Set[int],
    Dict[int, str],
    Dict[int, float],
    bool,
]:
    # Ported to Rust: rust/investa_core/src/pa_holdings.rs
    return investa_core.pa_transactions_to_holdings(
        transactions_df,
        default_currency,
        shortable_symbols,
        historical_fx_lookup,
        display_currency_for_hist_fx,
        report_date,
        account_cash_mode_map,
    )


@profile
def _build_summary_rows(
    holdings: Dict[Tuple[str, str], Dict],
    current_stock_data: Dict[
        str, Dict[str, Optional[float]]
    ],  # Data from MarketDataProvider
    current_fx_rates_vs_usd: Dict[str, float],  # Data from MarketDataProvider
    current_fx_prev_close_vs_usd: Dict[str, float],  # NEW: For Asset Change calc
    display_currency: str,
    default_currency: str,
    transactions_df: pd.DataFrame,  # Filtered transactions for IRR/fallback
    report_date: date,
    shortable_symbols: Set[str],
    user_excluded_symbols: Set[str],
    user_symbol_map: Dict[str, str],  # New: Accept user symbol map
    manual_prices_dict: Dict[str, float],
    published_navs: Optional[Dict[str, Tuple[str, float]]] = None,
    account_interest_rates: Optional[Dict[str, float]] = None,  # NEW
    interest_free_thresholds: Optional[Dict[str, float]] = None,  # NEW
    include_accounts: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, str], bool, bool]:
    """
    Builds the detailed list of portfolio summary rows, converting values to the display currency.

    When ``include_accounts`` is supplied, rows for accounts not in that list
    are dropped at the end. This matters because ``calculate_portfolio_summary``
    intentionally pulls in a source account's prior history when one of its
    stock positions was transferred *into* a selected account (so cost basis
    survives the transfer). That backfill is needed for the engine but leaks
    a phantom cash balance for the source account in Auto cash mode — the
    Buy/Sell rows generate negative auto-cash deltas without the offsetting
    $CASH deposits, leaving a large negative cash row attributed to an
    account the user explicitly excluded. Filtering at the row-emit stage
    keeps the cost-basis machinery intact while hiding the side-effect.
    """
    # Ported to Rust: rust/investa_core/src/pa_rows.rs
    return investa_core.pa_build_summary_rows(
        holdings,
        current_stock_data,
        current_fx_rates_vs_usd,
        current_fx_prev_close_vs_usd,
        display_currency,
        default_currency,
        transactions_df,
        report_date,
        shortable_symbols,
        user_excluded_symbols,
        user_symbol_map,
        manual_prices_dict,
        published_navs,
        account_interest_rates,
        interest_free_thresholds,
        include_accounts,
    )


# --- REVISED: _calculate_aggregate_metrics (Removed transactions_df parameter) ---
# @profile
def _calculate_aggregate_metrics(
    full_summary_df: pd.DataFrame,
    display_currency: str,
    report_date: date,
    fx_rates: Optional[Dict[str, float]] = None,  # <-- New param
    include_accounts: Optional[List[str]] = None,  # <-- New param
    all_available_accounts: Optional[List[str]] = None,
    transactions_df: Optional[pd.DataFrame] = None,
    historical_fx_rates: Optional[
        Dict[str, pd.DataFrame]
    ] = None,  # ADDED: Historical FX Data
) -> Tuple[
    Dict[str, Any], Dict[str, Dict[str, float]], bool, bool
]:  # Fixed return type hint
    """
    Calculates account-level and overall portfolio summary metrics.
    (Implementation remains the same as provided previously - relies only on input df and helpers)
    """
    # Ported to Rust: rust/investa_core/src/pa_aggregate.rs
    return investa_core.pa_aggregate_metrics(
        full_summary_df,
        display_currency,
        report_date,
        fx_rates,
        include_accounts,
        all_available_accounts,
        transactions_df,
        historical_fx_rates,
    )


# --- NEW: Periodic Return Calculation ---
def calculate_periodic_returns(
    historical_df: pd.DataFrame, benchmark_symbols: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Calculates periodic (Weekly, Monthly, Annual) returns from cumulative gain factors.

    Args:
        historical_df (pd.DataFrame): DataFrame indexed by date, containing cumulative
            gain columns like 'Portfolio Accumulated Gain' and '{Benchmark} Accumulated Gain'.
        benchmark_symbols (List[str]): List of benchmark symbols (e.g., ['SPY', 'QQQ'])
            present in the historical_df columns.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are interval codes ('W', 'M', 'Y')
            and values are DataFrames containing the periodic returns for that interval.
            Returns empty dict if input is invalid.
    """
    # Ported to Rust: rust/investa_core/src/pa_reports.rs
    return investa_core.pa_periodic_returns(historical_df, benchmark_symbols)


def calculate_correlation_matrix(historical_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the correlation matrix for historical returns of assets.

    Args:
        historical_returns_df (pd.DataFrame): DataFrame where each column represents
                                             an asset's historical returns.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_correlation_matrix(historical_returns_df)


def run_scenario_analysis(
    factor_betas: Dict[str, float],
    scenario_shocks: Dict[str, float],
    portfolio_value: float,
) -> Dict[str, float]:
    """
    Calculates the estimated portfolio impact based on factor betas and scenario shocks.

    Args:
        factor_betas (Dict[str, float]): Dictionary of factor betas (e.g., {'Mkt-RF': 1.2, 'SMB': 0.5}).
        scenario_shocks (Dict[str, float]): Dictionary of factor shocks (e.g., {'Mkt-RF': -0.10}).
        portfolio_value (float): The current market value of the portfolio.

    Returns:
        Dict[str, float]: A dictionary containing the estimated portfolio return and impact.
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_run_scenario_analysis(factor_betas, scenario_shocks, portfolio_value)


def extract_dividend_history(
    all_transactions_df: pd.DataFrame,
    display_currency: str,  # <-- Keep existing parameters
    historical_fx_yf: Dict[str, pd.DataFrame],  # YF Ticker -> DataFrame
    default_currency: str,
    include_accounts: Optional[List[str]] = None,  # <-- ADD NEW PARAMETER
) -> pd.DataFrame:
    """
       Extracts dividend history from transactions and converts amounts to the display currency.

       Args:
           all_transactions_df (pd.DataFrame): The full, cleaned transactions DataFrame.
           display_currency (str): The target currency for displaying dividend amounts.
           historical_fx_yf (Dict[str, pd.DataFrame]): Dictionary mapping YF FX pair tickers
               (e.g., 'EUR=X') to DataFrames containing historical rates vs USD.
           default_currency (str): The default currency if a transaction's local currency is missing.

    include_accounts (Optional[List[str]], optional): A list of account names to include. If None, includes all accounts. Defaults to None.

       Returns:
           pd.DataFrame: A DataFrame with dividend history, including:
               'Date', 'Symbol', 'Account', 'LocalCurrency', 'DividendAmountLocal',
               'FXRateUsed', 'DividendAmountDisplayCurrency'.
               Returns an empty DataFrame if no dividend transactions or on critical error.
    """
    # Ported to Rust: rust/investa_core/src/pa_reports.rs
    return investa_core.pa_dividend_history(
        all_transactions_df, display_currency, historical_fx_yf, default_currency, include_accounts
    )


def calculate_fifo_lots_and_gains(
    transactions_df: pd.DataFrame,
    display_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    shortable_symbols: Set[str],
    stock_quantity_close_tolerance: float = STOCK_QUANTITY_CLOSE_TOLERANCE,
    current_fx_rates_vs_usd: Optional[Dict[str, float]] = None,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str], List[Dict[str, Any]]]]:
    """
    Core FIFO calculation logic.
    Returns:
        - realized_gains_df (pd.DataFrame): The history of realized gains.
        - open_lots (Dict): Dictionary of currently held open lots.
             Key: (Symbol, Account)
             Value: List of Lot Dicts

    Each row is read and priced, then turned into one lot event; the lot
    bookkeeping (transfers, global splits, spin-off legs, buys and FIFO sells)
    runs in the FIFO book.
    """
    # Ported to Rust: rust/investa_core/src/pa_fifo.rs
    return investa_core.pa_fifo_lots_and_gains(
        transactions_df,
        display_currency,
        historical_fx_yf,
        default_currency,
        shortable_symbols,
        stock_quantity_close_tolerance,
        current_fx_rates_vs_usd,
    )


@profile
def extract_realized_capital_gains_history(
    all_transactions_df: pd.DataFrame,
    display_currency: str,
    historical_fx_yf: Dict[str, pd.DataFrame],
    default_currency: str,
    shortable_symbols: Set[str],
    stock_quantity_close_tolerance: float = STOCK_QUANTITY_CLOSE_TOLERANCE,
    include_accounts: Optional[List[str]] = None,
    current_fx_rates_vs_usd: Optional[Dict[str, float]] = None,
    from_date: Optional[date] = None,
    to_date: Optional[date] = None,
) -> pd.DataFrame:
    """
    Calculates realized capital gains from transactions using FIFO accounting for long positions.
    Wrapper around calculate_fifo_lots_and_gains.
    """
    # Ported to Rust: rust/investa_core/src/pa_reports.rs
    return investa_core.pa_realized_gains_history(
        all_transactions_df,
        display_currency,
        historical_fx_yf,
        default_currency,
        shortable_symbols,
        stock_quantity_close_tolerance,
        include_accounts,
        current_fx_rates_vs_usd,
        from_date,
        to_date,
    )


def calculate_rebalancing_trades(
    holdings_df: pd.DataFrame,
    target_alloc_pct: dict,
    new_cash: float = 0.0,
    display_currency: str = "USD",
):
    """
    Calculates the trades required to rebalance the portfolio to the target allocation.

    Args:
        holdings_df (pd.DataFrame): DataFrame of current holdings.
        target_alloc_pct (dict): Dictionary mapping symbols to their target percentage.
        new_cash (float): New cash to be added or withdrawn from the portfolio.
        display_currency (str): The display currency.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: Suggested trades to execute.
            - dict: Summary metrics of the rebalancing plan.
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_rebalancing_trades(
        holdings_df, target_alloc_pct, new_cash, display_currency
    )


# --- Portfolio Health Analysis ---


def calculate_hhi(weights: List[float]) -> float:
    """
    Calculates the Herfindahl-Hirschman Index (HHI) for a list of weights.
    HHI = sum(w_i^2) where w_i are fractions (0 to 1).
    Returns value between 1/N (perfectly diversified) and 1.0 (concentrated).
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_calculate_hhi(weights)


def calculate_health_score(
    summary_df: pd.DataFrame,
    risk_metrics: Dict[str, float],
    risk_free_rate: float = 0.02,
) -> Dict[str, Any]:
    """
    Calculates a comprehensive 'Health Score' for the portfolio.

    Components:
    1. Diversification (HHI of asset weights).
    2. Efficiency (Sharpe Ratio).
    3. Volatility Check (Penalty if too high).
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_health_score(summary_df, risk_metrics, risk_free_rate)


# --- ADDED: Helper to calculate monthly cash interest events (Shared) ---
def generate_cash_interest_events(
    portfolio_summary_rows: List[Dict[str, Any]],
    interest_rates: Dict[str, float],
    thresholds: Dict[str, float],
    start_date: date,
    end_date: date,
    display_currency: str = "USD",
) -> List[Dict[str, Any]]:
    """
    Generates estimated monthly cash interest events using pre-calculated summary rows.
    """
    # Ported to Rust: rust/investa_core/src/pa_small.rs
    return investa_core.pa_cash_interest_events(
        portfolio_summary_rows, interest_rates, thresholds, start_date, end_date, display_currency
    )
