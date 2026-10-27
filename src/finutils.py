# -*- coding: utf-8 -*-
"""
-------------------------------------------------------------------------------
 Name:          finutils.py
 Purpose:       Financial and general utility functions for Investa Portfolio Dashboard.
                Includes IRR/NPV calculations, currency conversion helpers,
                cash flow generation, historical price lookups, and file hashing.

 Author:        Google Gemini (Derived from portfolio_logic.py)

 Copyright:     (c) Investa Contributors 2025
 Licence:       MIT
-------------------------------------------------------------------------------
SPDX-License-Identifier: MIT
"""

import pandas as pd
import numpy as np
from datetime import date, timedelta
import hashlib  # noqa: F401 -- read by the Rust port at call time
import config  # noqa: F401 -- read by the Rust port at call time
import logging
import investa_core
from typing import List, Tuple, Dict, Optional, Any, Set

# Import constants needed within these utility functions
# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    from config import (
        CASH_SYMBOL_CSV,
        DEFAULT_CURRENCY,
        SHORTABLE_SYMBOLS,
        CURRENCY_SYMBOLS,
    )  # YFINANCE_EXCLUDED_SYMBOLS, SYMBOL_MAP_TO_YFINANCE removed
except ImportError:
    # Fallback values if config import fails (should not happen in normal execution)
    logging.error("CRITICAL: Could not import constants from config.py in finutils.py")
    CASH_SYMBOL_CSV = "$CASH"
    DEFAULT_CURRENCY = "USD"
    SHORTABLE_SYMBOLS = {"AAPL", "RIMM"}
    CURRENCY_SYMBOLS = {"USD": "$"}


def infer_periods_per_year(index, default: float = 252.0) -> float:
    """Observations per year for a time series, inferred from its date index.

    The portfolio TWR series is calendar-daily (forward-filled over weekends/
    holidays, ~365/yr); raw market data is trading-daily (~252/yr). Annualizing
    returns/volatility with the wrong constant skews them by ~365/252 ≈ 1.45x, so
    both the projection model and the risk metrics infer the spacing from the
    actual dates. Returns ``default`` when the index isn't a usable DatetimeIndex.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_infer_periods_per_year(index, default)


# --- Constants for _get_file_hash ---
HASH_CHUNK_SIZE = 8192
HASH_ERROR_NOT_FOUND = "FILE_NOT_FOUND"
HASH_ERROR_PERMISSION = "HASHING_ERROR_PERMISSION"
HASH_ERROR_IO = "HASHING_ERROR_IO"
HASH_ERROR_UNEXPECTED = "HASHING_ERROR_UNEXPECTED"


# --- File Hashing Helper ---
def _get_file_hash(filepath: str) -> str:
    """Calculates the SHA256 hash of a file.

    If the file is a SQLite database (ends in .db, .sqlite, .sqlite3),
    it also checks for associated Write-Ahead Log (-wal) and Shared Memory (-shm) files.
    If found, their hashes are combined with the main file hash to ensure
    any pending changes in the WAL are captured in the cache key.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_get_file_hash(filepath)


# --- ADDED: Currency Symbol Helper ---
def get_currency_symbol_from_code(currency_code: str) -> str:
    """
    Returns the currency symbol for a given currency code.
    Falls back to the code itself if no symbol is mapped.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_currency_symbol_from_code(currency_code)


# --- END ADDED ---


# --- Indicated Dividend Helper ---
def get_dividend_details(info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculates dividend details: frequency_months and indicated_annual_rate.
    Unifies logic between summary metrics and projected income charts.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_dividend_details(info)


def calculate_indicated_dividend(info: Dict[str, Any]) -> float:
    """Convenience wrapper for backward compatibility."""
    return get_dividend_details(info)["indicated_annual_rate"]


#: Magnitude cutoff separating a fraction-encoded raw Yahoo ``dividendYield``
#: from a percent-encoded one, used only when no corroborating signal exists.
#: Mirrored by `web_app/lib/dividend.ts` and `macos_app/.../DividendYield.swift`.
_RAW_YIELD_FRACTION_CUTOFF = 0.10


def robust_dividend_yield(info: Dict[str, Any]) -> Optional[float]:
    """Dividend yield as a FRACTION (0.005 == 0.5%), derived consistently with
    the forward-looking indicated dividend rate.

    We deliberately avoid trusting Yahoo's raw ``dividendYield`` as the primary
    source: it is delivered inconsistently (sometimes a fraction like 0.005,
    sometimes a percentage number like 0.47), and the magnitude heuristics used
    to disambiguate it are unreliable for very small yields. Instead we prefer:

      1. the indicated annual rate from ``get_dividend_details`` (which prefers
         the forward ``dividendRate`` so a just-announced increase/cut is
         reflected immediately) divided by the current price — this keeps the
         yield in lock-step with the projected-income figures. e.g. after NVDA
         raised its quarterly dividend to $0.25 ($1.00/yr), this yields
         1.00 / price rather than the stale trailing $0.04 figure.
      2. ``trailingAnnualDividendYield`` (already a fraction),
      3. last resort: Yahoo's raw ``dividendYield`` with a magnitude guess.

    Returns ``None`` when there is no dividend or nothing usable, so callers
    can leave any existing value untouched.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_robust_dividend_yield(info)


# --- Cash Symbol Helpers ---
def is_cash_symbol(symbol: str) -> bool:
    """Checks if a symbol is a cash symbol (e.g., '$CASH', 'Cash ($)')."""
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_is_cash_symbol(symbol)


# --- External-Flow Classifier (Bookkeeping-Mode-Independent) ---
#
# Single source of truth for whether a transaction row represents money
# crossing the portfolio boundary. Consumed by both the TWR engine
# (`_calculate_daily_net_cash_flow_vectorized` in portfolio_logic.py) and the
# IRR/MWR engine (`get_cash_flows_for_mwr` in this file under
# `flow_basis="portfolio"`). Centralizing the rule guarantees both engines see
# the same external flows regardless of the bookkeeping mode of the source
# account (auto-cash with implicit trade-side cash legs, or manual with
# explicit $CASH buy/sell settlement rows).
#
# Convention — "always external" (trading-account view):
#   External iff
#     Symbol == $CASH
#     AND Type ∈ {Deposit, Withdrawal} (case-insensitive)
#
# Why not the GIPS heuristic (filtering "Auto-generated:" or "Commission" Notes)?
#   Because in this project's data, the import tool tags BOTH real ACH/wire
#   deposits AND synthetic per-trade settlement entries with the same
#   "Auto-generated: Cash deposit for X buy" prefix. The two cannot be
#   distinguished by Note content. Filtering them all out causes a critical
#   bug in the TWR engine: on early-portfolio buy days the NAV engine still
#   credits cash from the deposit row (because that's how cash balance works
#   regardless of mode), the auto-cash logic debits it on the paired buy, and
#   stock value rises by the cost — net NAV change is +cost. With the flow
#   filtered to 0, the engine attributes this NAV jump to RETURN, producing
#   massive phantom spikes when prior NAV is small. Concretely: dheematan's
#   2003 portfolio inception produced ~292% phantom TWR over 3 months under
#   the GIPS heuristic.
#
#   The trading-account convention sidesteps this by treating every
#   $CASH Deposit/Withdrawal at face value. Synthetic per-trade entries
#   slightly inflate "external capital deployed" but never produce phantom
#   returns — the deposit credit and the resulting NAV jump cancel in the
#   daily-return formula.
#
# Notes on what's NOT external:
#   - $CASH buy/sell rows (kitmatan-style settlement) — wrong Type.
#   - $CASH dividend/interest/fees/tax — wrong Type.
#   - Stock buy/sell/dividend/tax/fees/transfer — wrong Symbol.
#   - Inter-portfolio transfers crossing the included-accounts boundary are
#     handled by per-engine scope logic, not this row-level classifier.
#
# Data hygiene note: if your data contains $CASH Deposit/Withdrawal rows
# that semantically should NOT be external (e.g. broker commissions
# miscategorized as Withdrawal), fix them at the data layer by retyping to
# Type=Fees rather than adding string-prefix heuristics here.
def is_external_flow_row(symbol: Any, tx_type: Any, note: Any = None) -> bool:
    """Returns True if a single transaction row represents an external portfolio flow.

    Mode-independent — does not inspect account_cash_mode_map. The caller is
    responsible for applying scope-crossing transfer detection separately.

    `note` is accepted for backward compatibility but no longer consulted;
    the classifier looks only at Symbol and Type.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_is_external_flow_row(symbol, tx_type, note)


def compute_external_flow_mask(df: "pd.DataFrame") -> "pd.Series":
    """Vectorized version of `is_external_flow_row` for a DataFrame.

    Returns a boolean pd.Series aligned with df.index. Expects df to have
    'Symbol' and 'Type' columns. 'Note' is not consulted.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_external_flow_mask(df)


# --- IRR/MWR Calculation Functions ---
# investa_core.calculate_irr status codes (rust/investa_core/src/irr.rs IrrStatus).
_IRR_OK = 0
_IRR_SOLVER_FAILED = 5
_DAY_US = 86_400_000_000
_ONE_US = timedelta(microseconds=1)


def _irr_date_ticks(dates):
    """The dates as int64 microsecond ticks, so `(d - start).days` can be taken
    in Rust exactly as Python takes it (datetimes keep their time of day, and
    the day count floors). None when Python could not subtract them either."""
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_irr_date_ticks(dates)


def calculate_npv(rate: float, dates: List[date], cash_flows: List[float]) -> float:
    """
    Net Present Value of `cash_flows`, discounted to the first date.

    Time is measured in years of 365 days from `dates[0]`, so the first flow is
    undiscounted. Returns NaN for a non-finite rate, a rate of -100% or below,
    a flow dated before the first, or a non-finite result; non-finite flows are
    skipped. Computed in Rust (investa_core.npv); the arithmetic matches the
    Python original bit for bit (frozen in tests/reference/finance.py).

    Raises:
        ValueError: If `dates` and `cash_flows` differ in length.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_npv(rate, dates, cash_flows)


def calculate_irr(dates: List[date], cash_flows: List[float]) -> float:
    """
    Internal Rate of Return (money-weighted return) of a dated cash-flow series.

    The rate at which the flows' NPV is zero, as a decimal (0.1 for 10%). NaN
    when no IRR is defined: fewer than two flows, all on one date, a
    non-numeric or non-finite flow, a series that does not start with an
    investment or never returns anything, returns negligible against capital
    (effectively -100%), or NPV that never crosses zero.

    The solver runs in Rust (investa_core.calculate_irr): Newton's secant
    method from five starting rates, then a scan of the NPV curve bracketed by
    Brent's method — SciPy's algorithms, ported so the result matches the
    Python original bit for bit (frozen in tests/reference/finance.py).
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_calculate_irr(dates, cash_flows)


# --- Cash Flow Helpers ---
# Row types as both cash-flow builders compare them (lowercased, stripped),
# coded for investa_core.holding_flows / mwr_flows.
_FLOW_KINDS = {
    "buy": 1,
    "deposit": 2,
    "sell": 3,
    "withdrawal": 4,
    "transfer": 5,
    "short sell": 6,
    "buy to cover": 7,
    "dividend": 8,
    "fees": 9,
    "split": 10,
    "stock split": 10,
    "interest": 11,
}
_TRANSFER = _FLOW_KINDS["transfer"]


def _numeric_column(df: pd.DataFrame, name: str, with_types: bool = False):
    """`pd.to_numeric(row.get(name), errors="coerce")` for every row, as float64.

    A missing column reads as None, hence NaN. A NumPy numeric column already
    holds what to_numeric would return (iterating it yields Python numbers);
    anything else is coerced cell by cell, as the originals did. With
    `with_types`, also says which cells came back as NumPy scalars (to_numeric
    turns the text "12.5" into one), whose type the arithmetic carries on.
    """
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_numeric_column(df, name, with_types)


def _commission_column(df: pd.DataFrame) -> np.ndarray:
    """Commission with NaN read as 0.0, as both builders treat it."""
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_commission_column(df)


def _cell_text(values: Optional[list], i: int, default: str = "") -> str:
    """`str(row.get(column, default))`: a missing column reads as the default."""
    # Ported to Rust: rust/investa_core/src/fu.rs
    return investa_core.fu_cell_text(values, i, default)


# A transactions frame read once, for any number of per-holding IRR flows.
#
# `get_cash_flows_for_symbol_account` re-filtered the whole frame on every
# call: an equality mask, a case-insensitive regex over every Note, and an
# iterrows pass. That cost a summary about half its time (243 holdings,
# 780,000 regex searches). Here the rows are grouped by symbol once, each
# account's mask is built once with the original's own pandas expressions,
# and each row is parsed once. The flow rules run in investa_core.
#
# Nothing is read until the first request, so a frame the original could
# not read fails inside the caller's handler exactly as before.
# Ported to Rust: rust/investa_core/src/fu_flows.rs
CashFlowLedger = investa_core.CashFlowLedger


def get_cash_flows_for_symbol_account(
    symbol: str,
    account: Optional[str],
    transactions_df: pd.DataFrame,
    final_market_value: float,
    is_transfer_a_flow: bool,  # ADDED: New parameter to control transfer handling
    report_date: Optional[date] = None,
) -> Tuple[List[date], List[float]]:
    """
    Extracts LOCAL currency cash flows for a specific symbol/account pair for IRR calculation.

    Filters transactions for the given symbol and account (``account=None`` takes
    the symbol in every account). Each row's flow in the holding's local currency:
    buys and fees negative, sales and dividends positive, transfers by direction
    when ``is_transfer_a_flow``, shorting actions for shortable symbols. The final
    market value is added as a positive flow on ``report_date``.

    Returns:
        Tuple[List[date], List[float]]: the dates (sorted) and their flows, or
        ([], []) when there are none or they cannot have an IRR (fewer than two,
        or all of one sign).

    For many holdings of one frame, build a ``CashFlowLedger`` once and call
    ``flows_for``; this reads the frame afresh.
    """
    # Ported to Rust: rust/investa_core/src/fu_flows.rs
    return investa_core.fu_cash_flows_for_symbol_account(
        symbol,
        account,
        transactions_df,
        final_market_value,
        is_transfer_a_flow,
        report_date,
    )


def get_cash_flows_for_mwr(
    account_transactions: pd.DataFrame,
    final_account_market_value: float,  # Already in target_currency
    end_date: date,
    target_currency: str,
    fx_rates: Optional[Dict[str, float]],  # Expects standard 'FROM/TO' -> rate format
    display_currency: str,  # Used for warning msg only (REMOVED - fx_rates needed)
    historical_fx_rates: Optional[
        Dict[Tuple[date, str], float]
    ] = None,  # ADDED: Historical FX Data
    include_accounts: Optional[
        List[str]
    ] = None,  # ADDED: To determine transfer direction
    flow_basis: str = "per_trade",  # "per_trade" (legacy/symbol-level) or "portfolio"
) -> Tuple[List[date], List[float]]:
    """
    Calculates cash flows for Money-Weighted Return (MWR) for a specific account in the target currency.

    Each transaction's flow in its local currency, converted to `target_currency` at
    the `historical_fx_rates` rate for its date and currency ((date, currency) ->
    rate), falling back to the current `fx_rates`. Transactions after `end_date`
    are skipped.

    Under `flow_basis="portfolio"` only money crossing the scope counts: `$CASH`
    deposits and withdrawals (see is_external_flow_row) and transfers into or out
    of `include_accounts` (by default every account in the frame). Under
    "per_trade" every trade is a flow. The per-row rules run in investa_core.

    Returns:
        Tuple[List[date], List[float]]: Sorted dates and cash flows in target currency.
    """
    # Ported to Rust: rust/investa_core/src/fu_flows.rs
    return investa_core.fu_cash_flows_for_mwr(
        account_transactions, final_account_market_value, end_date, target_currency,
        fx_rates, display_currency, historical_fx_rates, include_accounts, flow_basis,
    )  # fmt: skip


# --- Currency Conversion Helpers ---
def get_conversion_rate(
    from_curr: str, to_curr: str, fx_rates: Optional[Dict[str, float]]
) -> float:
    """
    Gets CURRENT FX conversion rate (units of to_curr per 1 unit of from_curr).

    Calculates cross rates via USD using the provided `fx_rates` dictionary, which is
    assumed to contain rates relative to USD (units of OTHER_CURRENCY per 1 USD).
    Handles direct conversion, conversion via USD, and returns a fallback rate if
    necessary data is missing or invalid.

    Args:
        from_curr (str): The currency code to convert FROM.
        to_curr (str): The currency code to convert TO.
        fx_rates (Optional[Dict[str, float]]): A dictionary containing current FX rates,
            where keys are currency codes (str) and values are the rate per 1 USD (float).
            Example: {'JPY': 150.0, 'EUR': 0.9, 'USD': 1.0}.

    Returns:
        float: The conversion rate (units of `to_curr` for 1 unit of `from_curr`).
               Returns np.nan if the necessary rates are missing or invalid in the
               `fx_rates` dictionary, or if inputs are invalid.
               Returns 1.0 if `from_curr` equals `to_curr`.
    """
    # Ported to Rust: rust/investa_core/src/fu_fx.rs
    return investa_core.fu_conversion_rate(from_curr, to_curr, fx_rates)


# --- Historical Price/Rate Helpers ---
_EPOCH_ORDINAL = date(1970, 1, 1).toordinal()
_DAY_NS = 86_400_000_000_000
_TS_MIN_NS, _TS_MAX_NS = pd.Timestamp.min.value, pd.Timestamp.max.value
# id(frame) -> (weakref to it, its shape signature, PriceTable or None)
_PRICE_TABLES: Dict[int, Tuple[Any, Any, Any]] = {}


def _price_table(df: pd.DataFrame) -> Optional["investa_core.PriceTable"]:
    """The frame's lookup table, built once per frame, or None to use pandas.

    Covers what a request meets in practice (daily FX and price frames on a
    naive or UTC DatetimeIndex with a numeric price column), with the original's
    semantics exactly: `asof` takes the last row at or before the day with no
    NaN in any column, and a day before the first row takes that row's price.
    Price frames are treated as immutable once built, as the market-data cache
    already shares them between requests. A frame whose length, columns or
    first/last date change gets a new table.
    """
    # Ported to Rust: rust/investa_core/src/fu_fx.rs
    return investa_core.fu_price_table(df)


def _build_price_table(df: pd.DataFrame) -> Optional["investa_core.PriceTable"]:
    # Ported to Rust: rust/investa_core/src/fu_fx.rs
    return investa_core.fu_build_price_table(df)


def get_historical_price(
    symbol_key: str, target_date: date, prices_dict: Dict[str, pd.DataFrame]
) -> float:
    """
    Gets the historical price for a symbol on a specific date, using forward fill for missing dates.

    Looks up the symbol (which can be a YF ticker like 'AAPL' or an FX pair like 'EUR=X')
    in the provided dictionary of price DataFrames. Finds the price for the `target_date`.
    If the exact date is missing, it uses the price from the most recent previous date available
    (forward fill).

    Args:
        symbol_key (str): The key (YF ticker or FX pair string) to look up in `prices_dict`.
        target_date (date): The specific date for which the price is required.
        prices_dict (Dict[str, pd.DataFrame]): A dictionary where keys are symbol keys (str) and
            values are DataFrames indexed by date objects, containing a 'price' column.

    Returns:
        float: The price for the symbol on the target date (or the last available price
               before it). Returns np.nan if the symbol is not found, the date is before
               the first available price point, or an error occurs during lookup.
    """
    # Ported to Rust: rust/investa_core/src/fu_fx.rs
    return investa_core.fu_historical_price(symbol_key, target_date, prices_dict)


def get_historical_rate_via_usd_bridge(
    from_curr: str,
    to_curr: str,
    target_date: date,
    historical_fx_data: Dict[
        str, pd.DataFrame
    ],  # Expects {'EUR=X': df, 'THB=X': df, ...}
) -> float:
    """
    Gets the historical FX rate (units of to_curr per 1 unit of from_curr) for a specific date using a USD bridge.

    Calculates cross rates via USD using the provided historical FX data dictionary. It fetches
    the rate of `from_curr` per USD and `to_curr` per USD for the `target_date` (using forward fill
    via `get_historical_price`) and then computes the cross rate.

    Formula: Rate(TO/FROM) = Rate(TO/USD) / Rate(FROM/USD)

    Args:
        from_curr (str): The currency code to convert FROM.
        to_curr (str): The currency code to convert TO.
        target_date (date): The specific date for which the FX rate is required.
        historical_fx_data (Dict[str, pd.DataFrame]): A dictionary where keys are Yahoo Finance
            FX pair tickers against USD (e.g., 'EUR=X', 'JPY=X') and values are DataFrames
            indexed by date objects, containing a 'price' column (representing units of the
            other currency per 1 USD).

    Returns:
        float: The calculated historical conversion rate (units of `to_curr` for 1 unit of `from_curr`).
               Returns np.nan if `from_curr` equals `to_curr` (should return 1.0, fixed), if necessary
               intermediate rates (vs USD) cannot be found for the date, if inputs are invalid,
               or if a calculation error occurs (e.g., division by zero).
               Returns 1.0 if `from_curr` == `to_curr`.
    """
    # Ported to Rust: rust/investa_core/src/fu_fx.rs
    return investa_core.fu_rate_via_usd_bridge(
        from_curr, to_curr, target_date, historical_fx_data
    )


# --- Symbol Mapping Helper ---
def map_to_yf_symbol(
    internal_symbol: str,
    user_symbol_map: Dict[str, str],
    user_excluded_symbols: Set[str],  # Ensure this parameter is used
) -> Optional[str]:
    """
    Maps an internal symbol to a Yahoo Finance compatible ticker, handling specific cases.

    Checks the user-defined explicit map first, then handles excluded symbols,
    the cash symbol, and converts BKK (Thailand) stock exchange symbols
    (e.g., 'ADVANC:BKK' -> 'ADVANC.BK'). Returns None for symbols that should be excluded
    or cannot be reliably mapped.

    Args:
        internal_symbol (str): The internal symbol used in the transaction data.
        user_symbol_map (Dict[str, str]): User-defined mapping of internal symbols to YF tickers.
        user_excluded_symbols (Set[str]): User-defined set of symbols to exclude from YF fetching.

    Returns:
        Optional[str]: The corresponding Yahoo Finance ticker (e.g., 'AAPL', 'BRK-B', 'ADVANC.BK'),
                       or None if the symbol is excluded, is cash, or has an invalid format.
    """
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_map_to_yf_symbol(
        internal_symbol, user_symbol_map, user_excluded_symbols
    )


# --- DataFrame Aggregation Helper ---
def safe_sum(df, col):
    """
    Safely sums a DataFrame column, handling NaNs and non-numeric types.

    Attempts to convert the specified column to numeric, coercing errors to NaN.
    Fills any remaining NaNs with 0.0 and then calculates the sum. Returns 0.0
    if the column doesn't exist or if any error occurs during the process.

    Args:
        df (pd.DataFrame): The DataFrame containing the column.
        col (str): The name of the column to sum.

    Returns:
        float: The sum of the numeric values in the column, or 0.0 on failure or if empty.
    """
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_safe_sum(df, col)


# --- Value Formatting Utilities ---
def format_currency_value(
    value: Optional[float],
    currency_symbol: str = "$",
    decimals: int = 2,
    show_plus_sign: bool = False,
    is_abs: bool = False,  # If true, formats absolute value
) -> str:
    """Formats a numeric value as a currency string."""
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_format_currency_value(
        value, currency_symbol, decimals, show_plus_sign, is_abs
    )


def format_percentage_value(
    value: Optional[float], decimals: int = 2, show_plus_sign: bool = False
) -> str:
    """Formats a numeric value (decimal fraction) as a percentage string."""
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_format_percentage_value(value, decimals, show_plus_sign)


def format_large_number_display(
    value: Optional[float], currency_symbol: str = "$", decimals: int = 2
) -> str:
    """Formats large numbers with T, B, M suffixes (e.g., for Market Cap)."""
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_format_large_number_display(value, currency_symbol, decimals)


def format_integer_with_commas(value: Optional[float]) -> str:
    """Formats a number as an integer string with commas."""
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_format_integer_with_commas(value)


def format_float_with_commas(value: Optional[float], decimals: int = 2) -> str:
    """Formats a float string with commas and specified decimal places."""
    # Ported to Rust: rust/investa_core/src/fu_fmt.rs
    return investa_core.fu_format_float_with_commas(value, decimals)
